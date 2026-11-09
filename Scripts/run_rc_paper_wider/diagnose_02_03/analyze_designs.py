"""
Experiment 4 analysis: how well can each candidate design reproduce the observed correlations?

For each design: emulate the 52 correlations as smooth functions of the free parameters (a Gaussian
process on the leading principal components of the simulated correlations), find the parameters with
the smallest misfit to the observed values, and list the parameter values whose misfit is within 15% of that optimum.

Writes results/exp4_best_params.csv, results/exp4_good_region.csv, results/exp4_emulator_cv.csv.

    python analyze_designs.py --tag screen
"""
import argparse
import json
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from scipy.stats import qmc
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor as GP
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.model_selection import KFold

from common import NPE, RESULTS, TOL, load_observed
from run_design import DESIGNS

warnings.filterwarnings("ignore", category=ConvergenceWarning)
FEATURES = [l.strip() for l in open(NPE / "run_specs" / "02AElatentAM_wider" / "features.txt") if l.strip()]
OBS, SE = load_observed(FEATURES)
SIGMA = np.sqrt(SE ** 2 + TOL ** 2)
N_PC = 10


class Emulator:
    """GP regression from scaled parameters to the leading principal components of the correlations."""

    def __init__(self, U, Y):
        self.pca = PCA(n_components=min(N_PC, len(U) - 1)).fit(Y)
        Z = self.pca.transform(Y)
        self.gps = []
        d = U.shape[1]
        for j in range(Z.shape[1]):
            kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(np.ones(d), (0.15, 30)) + WhiteKernel(1e-2, (1e-6, 1))
            self.gps.append(GP(kernel, normalize_y=True, n_restarts_optimizer=1, random_state=j).fit(U, Z[:, j]))

    def predict(self, U):
        Z = np.column_stack([g.predict(U) for g in self.gps])
        return self.pca.inverse_transform(Z)


def load(tag):
    rows = [json.loads(l) for l in open(RESULTS / f"exp4_{tag}.jsonl")]
    df = pd.DataFrame(rows)
    if "error" in df:
        bad = df["error"].notna()
        if bad.any():
            print(f"{bad.sum()} failed simulations dropped:", df.loc[bad, "error"].unique()[:3])
        df = df[~bad]
    return df


def misfit(Y):
    return np.sqrt(np.mean(((Y - OBS) / SIGMA) ** 2, axis=-1))


def analyse(design, df):
    free = DESIGNS[design]["free"]
    names = list(free)
    lo = np.array([free[k][0] for k in names])
    hi = np.array([free[k][1] for k in names])
    sub = df[df["design"] == design].dropna(subset=FEATURES)
    U = (sub[[f"param_{k}" for k in names]].to_numpy(float) - lo) / (hi - lo)
    Y = sub[FEATURES].to_numpy(float)

    # 5-fold check of the emulator: correlation-scale error per family of features
    pred = np.zeros_like(Y)
    for train, test in KFold(5, shuffle=True, random_state=1).split(U):
        pred[test] = Emulator(U[train], Y[train]).predict(U[test])
    cv = dict(design=design, n=len(sub), cv_rmse_all=np.sqrt(np.mean((pred - Y) ** 2)),
              cv_rmse_S_Y1=np.sqrt(np.mean((pred[:, FEATURES.index("S_Y1")] - Y[:, FEATURES.index("S_Y1")]) ** 2)),
              cv_rmse_M_Y1=np.sqrt(np.mean((pred[:, FEATURES.index("M_Y1")] - Y[:, FEATURES.index("M_Y1")]) ** 2)))

    em = Emulator(U, Y)
    res = differential_evolution(lambda u: misfit(em.predict(u.T)), [(0, 1)] * len(names), vectorized=True,
                                 updating="deferred", popsize=20, maxiter=150, tol=1e-4, seed=3, polish=False)
    best_u = res.x
    best_y = em.predict(best_u[None])[0]
    best = dict(design=design, emulator_misfit=float(res.fun), best_sim_misfit=float(misfit(Y).min()),
                **{f"{k}": float(lo[i] + best_u[i] * (hi[i] - lo[i])) for i, k in enumerate(names)},
                **{f"pred_{f}": float(v) for f, v in zip(FEATURES, best_y)})

    # Which parameter values fit about as well? The designs do not fit to within the observed SEs (misfit > 1),
    # so a likelihood built on those SEs would be far too sharp; report the set within 15% of the best misfit.
    draws = qmc.LatinHypercube(d=len(names), seed=11).random(60000)
    fit = misfit(em.predict(draws))
    near = fit <= 1.15 * fit.min()
    theta = lo[None] + draws[near] * (hi - lo)[None]
    region = [dict(design=design, parameter=k, lower=lo[i], upper=hi[i], q05=np.quantile(theta[:, i], 0.05),
                   q50=np.quantile(theta[:, i], 0.5), q95=np.quantile(theta[:, i], 0.95), n_near=int(near.sum()))
              for i, k in enumerate(names)]
    return best, region, cv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="screen")
    ap.add_argument("--designs", nargs="+", default=None)
    args = ap.parse_args()
    df = load(args.tag)
    print(df.groupby("design").size().to_dict())
    best, regions, cvs = [], [], []
    for design in args.designs or sorted(df["design"].unique()):
        b, r, c = analyse(design, df)
        best.append(b)
        regions += r
        cvs.append(c)
        print(f"{design:10s} best sim {b['best_sim_misfit']:.2f}  emulator optimum {b['emulator_misfit']:.2f}  "
              f"cv rmse {c['cv_rmse_all']:.3f}  "
              f"S_Y1 {b['pred_S_Y1']:.3f}  M_Y1 {b['pred_M_Y1']:.3f}  S_PGS1 {b['pred_S_PGS1']:.3f}  M_PGS1 {b['pred_M_PGS1']:.3f}", flush=True)
    pd.DataFrame(best).to_csv(RESULTS / "exp4_best_params.csv", index=False)
    pd.DataFrame(regions).to_csv(RESULTS / "exp4_good_region.csv", index=False)
    pd.DataFrame(cvs).to_csv(RESULTS / "exp4_emulator_cv.csv", index=False)


if __name__ == "__main__":
    main()
