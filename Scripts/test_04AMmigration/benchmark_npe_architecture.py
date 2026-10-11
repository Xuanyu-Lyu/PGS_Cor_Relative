"""
benchmark_npe_architecture.py
=============================
Does dropping the embedding network (features straight into the flow, as in
Scripts/nn_validate/ace_npe) hurt NPE performance?

Arms (identical splits, seeds, flow and training settings; only the
conditioning network differs):
    mlp       : PgsPhenoEmbeddingNet [256,256,128,128], dropout 0.3 (current production)
    identity  : no embedding network (nn.Identity)
Exploratory arms (condition-04 data only):
    identity_big : no embedding, larger flow (128 hidden, 8 transforms)
    ensemble5    : equal mixture of 5 identity flows with different initialisations

Datasets:
    cond04 : output/design.csv (EA-migration test design, 2,000 sims, 30 features).
             Mixed estimator: one shared n_gen classifier per seed; the arms differ
             in the continuous flow p(move_p, vg1, prop_h2_latent1 | x, n_gen).
    cond02 : nn_NPE/nn_training_02AElatentAM.csv (production data, ~18k sims,
             the 20 features and 5 parameters of results_npe_unweighted_02AElatentAM),
             production settings: prior = training range +/- 10%, <= 500 epochs,
             early stopping patience 50.

Headline metric: held-out negative log posterior density of the true parameters
(NLL; lower is better), the proper scoring rule NPE is trained on. Also R^2,
calibration slope (true regressed on posterior mean; 1 = calibrated), 50/90%
interval coverage, posterior SD and edge bias.

Usage:
    python benchmark_npe_architecture.py --dataset cond04 --seeds 5
    python benchmark_npe_architecture.py --dataset cond02 --seeds 3
Outputs: output/benchmark/{dataset}_metrics.csv, {dataset}_nll.csv
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

import condition4_common as c4
import test_identifiability as tid
from npe_mixed import train_classifier, train_flow

OUT = c4.OUTPUT_DIR / "benchmark"
NN_DIR = Path(__file__).resolve().parent.parent / "nn_NPE"


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------

def load_cond04():
    design, feats = tid.load_design()
    params = c4.PARAMS
    X = design[feats].to_numpy(float)
    Y = design[[f"param_{p}" for p in params]].to_numpy(float)
    bounds = {p: c4.PARAM_BOUNDS[p] for p in params}
    return X, Y, params, bounds, "n_gen"


def load_cond02():
    run = NN_DIR / "results_npe_unweighted_02AElatentAM"
    feats = [l.strip() for l in open(run / "features.txt") if l.strip()]
    params = [l.strip() for l in open(run / "params.txt") if l.strip()]
    df = pd.read_csv(NN_DIR / "nn_training_02AElatentAM.csv")
    cols = feats + [f"param_{p}" for p in params]
    df = df.dropna(subset=cols)
    X = df[feats].to_numpy(float)
    Y = df[[f"param_{p}" for p in params]].to_numpy(float)
    lo, hi = Y.min(0), Y.max(0)
    buf = 0.1 * (hi - lo)
    bounds = {p: (lo[j] - buf[j], hi[j] + buf[j]) for j, p in enumerate(params)}
    return X, Y, params, bounds, None


# ----------------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------------

def flow_nll(posterior, theta, x):
    """-log q(theta | x) per row, straight from the trained density estimator."""
    est = posterior.posterior_estimator
    est.eval()
    with torch.no_grad():
        lp = est.log_prob(torch.tensor(theta, dtype=torch.float32)[None],
                          condition=torch.tensor(x, dtype=torch.float32))
    return -lp.squeeze(0).numpy()


rng_share = np.random.default_rng(12345)


def sample_flows(flows, x, n):
    """Equal-share samples from one flow or an ensemble of flows, for one x."""
    shares = np.full(len(flows), n // len(flows))
    shares[: n - shares.sum()] += 1
    shares = rng_share.permutation(shares)      # small n: spread the remainder over random members
    out = []
    for f, k in zip(flows, shares):
        if k == 0:
            continue
        with torch.no_grad(), c4.suppress_output():
            out.append(f.sample((int(k),), x=torch.tensor(x[None], dtype=torch.float32),
                                show_progress_bars=False).numpy())
    return np.concatenate(out)


def param_metrics(true, samp, lo, hi):
    pm, psd = samp.mean(1), samp.std(1)
    q = np.quantile(samp, [0.05, 0.25, 0.75, 0.95], axis=1)
    span = hi - lo
    low, high = true <= lo + 0.1 * span, true >= hi - 0.1 * span
    return {
        "r2": 1 - np.sum((pm - true) ** 2) / np.sum((true - true.mean()) ** 2),
        "rmse": np.sqrt(np.mean((pm - true) ** 2)),
        "calib_slope": np.polyfit(pm, true, 1)[0],
        "cover50": np.mean((true >= q[1]) & (true <= q[2])),
        "cover90": np.mean((true >= q[0]) & (true <= q[3])),
        "mean_sd": psd.mean(),
        "edge_bias_low": np.mean(pm[low] - true[low]) if low.any() else np.nan,
        "edge_bias_high": np.mean(pm[high] - true[high]) if high.any() else np.nan,
    }


# ----------------------------------------------------------------------------
# one seed
# ----------------------------------------------------------------------------

def run_seed(dataset, seed, arms, n_eval, n_post, max_epochs, patience):
    X, Y, params, bounds, count = load_cond04() if dataset == "cond04" else load_cond02()
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_test = int(round(0.15 * len(X)))
    te, tr = idx[:n_test][:n_eval], idx[n_test:]
    sc = StandardScaler().fit(X[tr])
    Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])

    cont = [p for p in params if p != count]
    ci = [params.index(p) for p in cont]
    lo = np.array([bounds[p][0] for p in cont], float)
    hi = np.array([bounds[p][1] for p in cont], float)

    if count:   # shared count classifier; flows condition on the one-hot count
        gmin, gmax = (int(b) for b in bounds[count])
        K = gmax - gmin + 1
        g_tr = np.rint(Y[tr, params.index(count)]).astype(int) - gmin
        g_te = np.rint(Y[te, params.index(count)]).astype(int) - gmin
        clf, T = train_classifier(Xtr, g_tr, K, seed=seed)
        with torch.no_grad():
            pmf = torch.softmax(clf(torch.tensor(Xte, dtype=torch.float32)) / T, 1).numpy()
        Ftr = np.hstack([Xtr, np.eye(K)[g_tr]])
        Fte_true = np.hstack([Xte, np.eye(K)[g_te]])     # for NLL: condition on the true count
    else:
        Ftr, Fte_true = Xtr, Xte

    metric_rows, nll_rows = [], []
    for arm in arms:
        t0 = time.time()
        kw = dict(seed=seed, max_epochs=max_epochs, stop_after_epochs=patience)
        if arm == "mlp":
            flows = [train_flow(Ftr, Y[tr][:, ci], lo, hi, embedding="mlp", **kw)]
        elif arm == "identity":
            flows = [train_flow(Ftr, Y[tr][:, ci], lo, hi, embedding="identity", **kw)]
        elif arm == "identity_big":
            flows = [train_flow(Ftr, Y[tr][:, ci], lo, hi, embedding="identity",
                                flow_hidden=128, flow_transforms=8, **kw)]
        elif arm == "ensemble5":
            flows = [train_flow(Ftr, Y[tr][:, ci], lo, hi, embedding="identity",
                                **{**kw, "seed": seed * 100 + m}) for m in range(5)]
        else:
            raise ValueError(arm)
        train_s = time.time() - t0

        # NLL of the true continuous parameters (ensemble: log of the mean density)
        lps = np.stack([-flow_nll(f, Y[te][:, ci], Fte_true) for f in flows])
        nll = -(np.logaddexp.reduce(lps, axis=0) - np.log(len(flows)))
        nll_rows.append({"dataset": dataset, "seed": seed, "arm": arm,
                         "nll": float(np.mean(nll)), "train_seconds": train_s})

        # joint posterior samples -> per-parameter metrics
        samp = np.zeros((len(te), n_post, len(cont)))
        for i in range(len(te)):
            if count:
                gs = rng.choice(K, size=n_post, p=pmf[i] / pmf[i].sum())
                pos = 0
                for g in np.unique(gs):
                    k = int((gs == g).sum())
                    samp[i, pos:pos + k] = sample_flows(flows, np.hstack([Xte[i], np.eye(K)[g]]), k)
                    pos += k
            else:
                samp[i] = sample_flows(flows, Xte[i], n_post)
        for j, p in enumerate(cont):
            m = param_metrics(Y[te, ci[j]], samp[:, :, j], lo[j], hi[j])
            metric_rows.append({"dataset": dataset, "seed": seed, "arm": arm, "parameter": p, **m})
        print(f"[{dataset}] seed {seed} {arm:13s} NLL={np.mean(nll):+.3f}  "
              + "  ".join(f"{r['parameter']}: R2={r['r2']:.3f}" for r in metric_rows[-len(cont):])
              + f"  (train {train_s:.0f}s, total {time.time() - t0:.0f}s)", flush=True)
    return metric_rows, nll_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cond04", "cond02"], required=True)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--threads", type=int, default=3)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    OUT.mkdir(parents=True, exist_ok=True)

    if args.dataset == "cond04":
        arms = ["mlp", "identity", "identity_big", "ensemble5"]
        cfg = dict(n_eval=300, n_post=500, max_epochs=300, patience=30)
    else:
        arms = ["mlp", "identity"]
        cfg = dict(n_eval=500, n_post=500, max_epochs=500, patience=50)

    metrics, nlls = [], []
    for seed in range(args.seeds):
        m, n = run_seed(args.dataset, seed, arms, **cfg)
        metrics += m
        nlls += n
        pd.DataFrame(metrics).to_csv(OUT / f"{args.dataset}_metrics.csv", index=False)
        pd.DataFrame(nlls).to_csv(OUT / f"{args.dataset}_nll.csv", index=False)
    print("done", flush=True)


if __name__ == "__main__":
    main()
