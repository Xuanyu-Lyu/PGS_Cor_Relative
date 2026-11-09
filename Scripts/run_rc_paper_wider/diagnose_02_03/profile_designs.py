"""
Profile curves on the emulators of analyze_designs.py: the best misfit a design can reach when one
parameter is held at each value of a grid and the others are free. A flat curve means the observed
correlations say little about that parameter. Writes results/exp4_profiles.csv.
"""
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from sklearn.exceptions import ConvergenceWarning

from analyze_designs import FEATURES, Emulator, load, misfit
from common import RESULTS
from run_design import DESIGNS

warnings.filterwarnings("ignore", category=ConvergenceWarning)
PROFILES = {"02_freevg": ["vg1", "re", "am22", "rg", "prop_h2_latent1"],
            "03_re_freevg": ["vg1", "re", "f11", "f22", "am22", "rg", "prop_h2_latent1"],
            "02_c": ["c1", "c2"], "03_re_c": ["c1", "c2", "f11", "f22"]}


def profile(design, df, names_to_profile, n_grid=9):
    free = DESIGNS[design]["free"]
    names = list(free)
    lo = np.array([free[k][0] for k in names])
    hi = np.array([free[k][1] for k in names])
    sub = df[df["design"] == design].dropna(subset=FEATURES)
    U = (sub[[f"param_{k}" for k in names]].to_numpy(float) - lo) / (hi - lo)
    em = Emulator(U, sub[FEATURES].to_numpy(float))
    rows = []
    for par in names_to_profile:
        j = names.index(par)
        others = [i for i in range(len(names)) if i != j]
        for g in np.linspace(0, 1, n_grid):
            def objective(u):
                full = np.empty((u.shape[1], len(names)))
                full[:, others] = u.T
                full[:, j] = g
                return misfit(em.predict(full))
            res = differential_evolution(objective, [(0, 1)] * len(others), vectorized=True, updating="deferred",
                                         popsize=12, maxiter=60, tol=1e-3, seed=5, polish=False)
            rows.append(dict(design=design, parameter=par, value=lo[j] + g * (hi[j] - lo[j]), misfit=float(res.fun)))
        print(design, par, "done", flush=True)
    return rows


def main():
    df = load("screen")
    rows = []
    for design, pars in PROFILES.items():
        if (df["design"] == design).sum() >= 100:
            rows += profile(design, df, pars)
    pd.DataFrame(rows).to_csv(RESULTS / "exp4_profiles.csv", index=False)


if __name__ == "__main__":
    main()
