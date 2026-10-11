"""
npe_count.py
============
Posterior for a parameter set that contains one integer-valued (count) parameter,
e.g. the number of generations ``n_gen``. Used by ``train_npe_pgs_and_pheno.py
--count_param``.

The joint posterior is factorised as

    p(count, theta_c | x) = p(count | x) * p(theta_c | x, count)

* p(count | x)          : a categorical classifier (MLP + softmax over the possible
                          counts), early-stopped and temperature-scaled on the
                          validation split, so every count gets a proper probability;
* p(theta_c | x, count) : an sbi NPE flow over the continuous parameters, trained on
                          x plus a one-hot encoding of the count.

``MixedPosterior.sample`` draws the count first and then the continuous parameters
given that count, and returns samples in the original parameter order, so the
training-script evaluation and ``predict_npe_pgs_and_pheno.py`` use it exactly like
an sbi posterior. This class lives in its own module (not in a script run as
``__main__``) so pickled posteriors load from any script.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def train_count_classifier(X_train, g_train, X_val, g_val, n_classes, *, hidden=128,
                           lr=1e-3, weight_decay=1e-4, batch_size=64, max_epochs=500,
                           patience=30, seed=0):
    """Train p(count | x); ``g_*`` are class indices 0..n_classes-1.

    Early stopping and the softmax temperature both use the validation split.
    Returns (model, temperature).
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    Xt = torch.tensor(X_train, dtype=torch.float32)
    yt = torch.tensor(np.asarray(g_train), dtype=torch.long)
    Xv = torch.tensor(X_val, dtype=torch.float32)
    yv = torch.tensor(np.asarray(g_val), dtype=torch.long)

    model = nn.Sequential(nn.Linear(Xt.shape[1], hidden), nn.ReLU(),
                          nn.Linear(hidden, hidden), nn.ReLU(),
                          nn.Linear(hidden, n_classes))
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    best, best_state, bad = np.inf, None, 0
    for _ in range(max_epochs):
        model.train()
        for b in np.array_split(rng.permutation(len(yt)), max(1, len(yt) // batch_size)):
            opt.zero_grad()
            loss_fn(model(Xt[b]), yt[b]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = loss_fn(model(Xv), yv).item()
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
        logits = model(Xv)
    temps = np.linspace(0.5, 3.0, 51)
    nll = [loss_fn(logits / t, yv).item() for t in temps]
    return model, float(temps[int(np.argmin(nll))])


class MixedPosterior:
    """Joint posterior over [continuous params..., count param] in ``param_names`` order.

    Parameters
    ----------
    flow_posterior : sbi posterior for the continuous parameters, conditioned on
                     [x, one-hot(count)]; may be None if the count is the only parameter
    classifier     : nn.Module returning logits over the counts
    temperature    : softmax temperature for the classifier
    count_values   : array of the possible counts (classifier class k -> count_values[k])
    param_names    : all parameter names, in output order
    count_param    : name of the count parameter
    """

    def __init__(self, flow_posterior, classifier, temperature, count_values,
                 param_names, count_param):
        self.flow_posterior = flow_posterior
        self.classifier = classifier
        self.temperature = float(temperature)
        self.count_values = np.asarray(count_values)
        self.param_names = list(param_names)
        self.count_param = count_param
        self.count_index = self.param_names.index(count_param)
        self.cont_index = [i for i, p in enumerate(self.param_names) if p != count_param]

    # so set_eval_mode() in the training/predict scripts finds the networks
    @property
    def posterior_estimator(self):
        return self.flow_posterior.posterior_estimator if self.flow_posterior is not None else None

    def eval(self):
        self.classifier.eval()
        if self.flow_posterior is not None:
            self.flow_posterior.posterior_estimator.eval()
        return self

    def count_pmf(self, x):
        """Posterior probability of each count, shape (n_obs, n_counts); columns = count_values."""
        x = torch.as_tensor(x, dtype=torch.float32)
        if x.dim() == 1:
            x = x[None, :]
        self.classifier.eval()
        with torch.no_grad():
            return torch.softmax(self.classifier(x) / self.temperature, dim=1).numpy()

    def sample(self, sample_shape=(1,), x=None, show_progress_bars=False,
               reject_outside_prior=True, **kwargs):
        """Draw joint samples for ONE observation x (shape (1, n_features) or (n_features,))."""
        n = int(np.prod(sample_shape))
        x = torch.as_tensor(x, dtype=torch.float32).reshape(-1)
        pmf = self.count_pmf(x)[0]
        rng = np.random.default_rng()
        k = rng.choice(len(pmf), size=n, p=pmf / pmf.sum())
        out = np.zeros((n, len(self.param_names)), dtype=np.float32)
        out[:, self.count_index] = self.count_values[k]
        if self.flow_posterior is not None:
            n_classes = len(self.count_values)
            for kk in np.unique(k):
                rows = np.where(k == kk)[0]
                onehot = torch.zeros(n_classes)
                onehot[kk] = 1.0
                xc = torch.cat([x, onehot])[None, :]
                with torch.no_grad():
                    s = self.flow_posterior.sample((len(rows),), x=xc,
                                                   show_progress_bars=False,
                                                   reject_outside_prior=reject_outside_prior)
                out[np.ix_(rows, self.cont_index)] = s.cpu().numpy()
        return torch.from_numpy(out)
