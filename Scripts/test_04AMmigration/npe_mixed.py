"""
npe_mixed.py
============
Neural posterior estimation with one discrete (count) parameter, e.g. ``n_gen``.

The joint posterior is factorised as

    p(n_gen, theta_c | x) = p(n_gen | x) * p(theta_c | x, n_gen)

* p(n_gen | x)            : a categorical classifier (MLP + softmax over the
                            possible counts), trained with cross-entropy and
                            temperature-scaled on a validation split, so it
                            returns a proper probability for every count.
* p(theta_c | x, n_gen)   : an sbi neural spline flow over the continuous
                            parameters, conditioned on x plus a one-hot n_gen.

Sampling draws n_gen from the classifier, then theta_c from the flow given
that n_gen. Without a count parameter this reduces to a plain NPE flow, and
with only the count parameter it reduces to the classifier.

``embedding="identity"`` hands the standardised features straight to the
flow (as in Scripts/nn_validate/ace_npe); ``embedding="mlp"`` inserts the
project's PgsPhenoEmbeddingNet (dropout 0.3).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_NN_DIR = Path(__file__).resolve().parent.parent / "nn_NPE"
if str(_NN_DIR) not in sys.path:
    sys.path.insert(0, str(_NN_DIR))


# ============================================================================
# COUNT CLASSIFIER
# ============================================================================

def train_classifier(X, g, n_classes, *, hidden=128, lr=1e-3, weight_decay=1e-4,
                     batch_size=64, max_epochs=500, patience=30, val_frac=0.1, seed=0):
    """Train p(count | x). ``g`` holds class indices 0..n_classes-1.

    Early stopping on validation cross-entropy, then a temperature is fitted on
    the same validation split. Returns (model, temperature).
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(np.asarray(g), dtype=torch.long)
    perm = rng.permutation(len(y))
    n_val = max(1, int(round(val_frac * len(y))))
    va, tr = perm[:n_val], perm[n_val:]

    model = nn.Sequential(nn.Linear(X.shape[1], hidden), nn.ReLU(),
                          nn.Linear(hidden, hidden), nn.ReLU(),
                          nn.Linear(hidden, n_classes))
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    best, best_state, bad = np.inf, None, 0
    for _ in range(max_epochs):
        model.train()
        for b in np.array_split(rng.permutation(tr), max(1, len(tr) // batch_size)):
            opt.zero_grad()
            loss_fn(model(X[b]), y[b]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = loss_fn(model(X[va]), y[va]).item()
        if v < best - 1e-4:
            best, bad = v, 0
            best_state = {k: t.clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        logits = model(X[va])
    temps = np.linspace(0.5, 3.0, 51)
    nll = [loss_fn(logits / t, y[va]).item() for t in temps]
    return model, float(temps[int(np.argmin(nll))])


# ============================================================================
# CONTINUOUS FLOW
# ============================================================================

def train_flow(X, theta, lo, hi, *, embedding="identity", dropout=0.3, seed=0,
               max_epochs=300, stop_after_epochs=30, batch_size=64, lr=5e-4,
               flow_hidden=64, flow_transforms=5):
    """sbi NPE (NSF) for p(theta | x) with a BoxUniform prior on [lo, hi]."""
    from sbi.neural_nets import posterior_nn
    from sbi.utils import BoxUniform
    import train_npe_pgs_and_pheno as tnpe
    from condition4_common import suppress_output

    torch.manual_seed(seed)
    prior = BoxUniform(low=torch.tensor(lo, dtype=torch.float32),
                       high=torch.tensor(hi, dtype=torch.float32))
    if embedding == "mlp":
        emb = tnpe.PgsPhenoEmbeddingNet(n_features=X.shape[1], hidden_sizes=[256, 256, 128, 128],
                                        dropout_rate=dropout)
    elif embedding == "identity":
        emb = nn.Identity()
    else:
        raise ValueError(f"unknown embedding '{embedding}'")
    de = posterior_nn(model="nsf", embedding_net=emb, hidden_features=flow_hidden,
                      num_transforms=flow_transforms,
                      z_score_theta="independent", z_score_x="independent")
    inference = tnpe.SNPEFeatureReg(prior=prior, density_estimator=de, n_pgs=0, n_y1=0,
                                    pgs_weight_decay=0.0, pheno_weight_decay=0.0,
                                    global_weight_decay=0.0)   # no L2, as benchmarked
    inference.append_simulations(theta=torch.tensor(theta, dtype=torch.float32),
                                 x=torch.tensor(X, dtype=torch.float32))
    with suppress_output():
        estimator = inference.train(training_batch_size=batch_size, learning_rate=lr,
                                    max_num_epochs=max_epochs, stop_after_epochs=stop_after_epochs,
                                    validation_fraction=0.1, show_train_summary=False)
    return inference.build_posterior(estimator)


# ============================================================================
# MIXED POSTERIOR
# ============================================================================

class MixedNPE:
    """Posterior over ``params`` (one of which may be the integer ``count_param``).

    Parameters
    ----------
    params : list of str        parameters to infer, in output order
    bounds : dict name -> (lo, hi)   prior ranges (inclusive integer range for the count)
    count_param : str or None   name of the discrete parameter, if any
    embedding : "identity" | "mlp"
    """

    def __init__(self, params, bounds, count_param=None, embedding="identity", seed=0,
                 max_epochs=300):
        self.params = list(params)
        self.bounds = bounds
        self.count = count_param if count_param in self.params else None
        self.cont = [p for p in self.params if p != self.count]
        self.embedding = embedding
        self.seed = seed
        self.max_epochs = max_epochs
        if self.count:
            self.gmin, self.gmax = (int(b) for b in bounds[self.count])
            self.n_classes = self.gmax - self.gmin + 1

    def _onehot(self, g_idx):
        return np.eye(self.n_classes)[np.asarray(g_idx, dtype=int)]

    def fit(self, X, Y):
        """X: standardised inputs (n, f). Y: true values (n, len(params))."""
        Y = np.asarray(Y, dtype=float)
        if self.count:
            g = np.rint(Y[:, self.params.index(self.count)]).astype(int) - self.gmin
            self.clf, self.temperature = train_classifier(X, g, self.n_classes, seed=self.seed)
        if self.cont:
            cols = [self.params.index(p) for p in self.cont]
            lo = np.array([self.bounds[p][0] for p in self.cont], dtype=float)
            hi = np.array([self.bounds[p][1] for p in self.cont], dtype=float)
            Xc = np.hstack([X, self._onehot(g)]) if self.count else X
            self.flow = train_flow(Xc, Y[:, cols], lo, hi, embedding=self.embedding,
                                   seed=self.seed, max_epochs=self.max_epochs)
        return self

    def count_pmf(self, X):
        """Posterior probability of each count (n, n_classes); columns = gmin..gmax."""
        with torch.no_grad():
            logits = self.clf(torch.tensor(X, dtype=torch.float32))
        return torch.softmax(logits / self.temperature, dim=1).numpy()

    def sample(self, x, n, rng=None):
        """Draw ``n`` joint samples for one observation ``x`` (1-D). Shape (n, len(params))."""
        from condition4_common import suppress_output

        rng = np.random.default_rng(self.seed) if rng is None else rng
        out = np.zeros((n, len(self.params)))
        x = np.asarray(x, dtype=float)
        if self.count:
            pmf = self.count_pmf(x[None, :])[0]
            g = rng.choice(self.n_classes, size=n, p=pmf / pmf.sum())
            out[:, self.params.index(self.count)] = g + self.gmin
        if not self.cont:
            return out
        cols = [self.params.index(p) for p in self.cont]
        groups = np.unique(g) if self.count else [None]
        for gv in groups:
            rows = np.where(g == gv)[0] if self.count else np.arange(n)
            xc = np.hstack([x, self._onehot(gv)]) if self.count else x
            with torch.no_grad(), suppress_output():
                s = self.flow.sample((len(rows),), x=torch.tensor(xc[None, :], dtype=torch.float32),
                                     show_progress_bars=False)
            out[np.ix_(rows, cols)] = s.numpy()
        return out
