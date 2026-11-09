"""
Experiment 6: choose the fixed values of re and prop_h2_latent1 for condition 03, on the corrected
engine with s_mat applied (designs 03_s5 and 03_s3 of run_design.py: s = 0.5 and s = 0.3).

Over a grid of (re, prop_h2_latent1), finds on the emulator the best misfit the five parameters that stay
free in condition 03 (vg1, f11, f22, am22, rg) can reach. Also profiles vg1, to show which values s = 0.5
favours. The fixed values are chosen on a second grid with vg1 held at 0.35.
Writes results/exp6_grid.csv, results/exp6_grid_vg035.csv and results/exp6_profile_vg1.csv.
"""
import argparse
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import KFold

from analyze_designs import FEATURES, Emulator, load, misfit
from common import RESULTS
from run_design import DESIGNS

warnings.filterwarnings("ignore", category=ConvergenceWarning)
RE_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
PROP_GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", default="03_s5", help="design of run_design.py (03_s5 or 03_s3)")
    ap.add_argument("--tag", default="s5", help="tag the design was simulated under")
    ap.add_argument("--suffix", default="", help="added to the output file names")
    args = ap.parse_args()
    DESIGN = args.design
    free = DESIGNS[DESIGN]["free"]
    names = list(free)
    lo = np.array([free[k][0] for k in names])
    hi = np.array([free[k][1] for k in names])
    df = load(args.tag)
    sub = df[df["design"] == DESIGN].dropna(subset=FEATURES)
    U = (sub[[f"param_{k}" for k in names]].to_numpy(float) - lo) / (hi - lo)
    Y = sub[FEATURES].to_numpy(float)
    print("simulations", len(sub), " lowest simulated misfit", round(float(misfit(Y).min()), 2), flush=True)

    pred = np.zeros_like(Y)
    for train, test in KFold(5, shuffle=True, random_state=1).split(U):
        pred[test] = Emulator(U[train], Y[train]).predict(U[test])
    print("emulator 5-fold rmse (correlation units)", round(float(np.sqrt(np.mean((pred - Y) ** 2))), 3), flush=True)
    em = Emulator(U, Y)

    def fit_with(held, seed=5):
        """Best misfit with the parameters in `held` {name: value} fixed, the others free."""
        idx = {k: names.index(k) for k in held}
        others = [i for i in range(len(names)) if i not in idx.values()]
        scaled = {idx[k]: (v - lo[idx[k]]) / (hi[idx[k]] - lo[idx[k]]) for k, v in held.items()}

        def objective(u):
            full = np.empty((u.shape[1], len(names)))
            full[:, others] = u.T
            for j, value in scaled.items():
                full[:, j] = value
            return misfit(em.predict(full))

        res = differential_evolution(objective, [(0, 1)] * len(others), vectorized=True, updating="deferred",
                                     popsize=15, maxiter=80, tol=1e-3, seed=seed, polish=False)
        best = {names[i]: float(lo[i] + res.x[j] * (hi[i] - lo[i])) for j, i in enumerate(others)}
        return float(res.fun), best

    rows = []
    for re in RE_GRID:
        for prop in PROP_GRID:
            value, best = fit_with({"re": re, "prop_h2_latent1": prop})
            rows.append(dict(re=re, prop_h2_latent1=prop, misfit=value, **best))
    grid = pd.DataFrame(rows).sort_values("misfit")
    grid.to_csv(RESULTS / f"exp6_grid{args.suffix}.csv", index=False)
    print(grid.head(10).round(3).to_string(index=False), flush=True)
    print(grid.pivot(index="re", columns="prop_h2_latent1", values="misfit").round(2).to_string(), flush=True)

    # The grid above lets vg1 go where it fits best (its lower edge, with s = 0.5). The target is an EA
    # founder heritability of 0.3-0.4, so the fixed values are chosen at vg1 = 0.35.
    rows = []
    for re in RE_GRID:
        for prop in PROP_GRID:
            value, best = fit_with({"re": re, "prop_h2_latent1": prop, "vg1": 0.35})
            rows.append(dict(re=re, prop_h2_latent1=prop, vg1=0.35, misfit=value, **best))
    at_target = pd.DataFrame(rows).sort_values("misfit")
    at_target.to_csv(RESULTS / f"exp6_grid_vg035{args.suffix}.csv", index=False)
    print(at_target.head(8).round(3).to_string(index=False), flush=True)
    print(at_target.pivot(index="re", columns="prop_h2_latent1", values="misfit").round(2).to_string(), flush=True)

    rows = []
    for vg1 in np.arange(0.05, 0.96, 0.1):
        value, best = fit_with({"vg1": float(vg1)})
        rows.append(dict(vg1=float(vg1), misfit=value, **best))
    profile = pd.DataFrame(rows)
    profile.to_csv(RESULTS / f"exp6_profile_vg1{args.suffix}.csv", index=False)
    print(profile.round(3).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
