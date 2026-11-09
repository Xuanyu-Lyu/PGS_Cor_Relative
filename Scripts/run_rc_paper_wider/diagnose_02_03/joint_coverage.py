"""
How much of a design's parameter range produces rows near the observed pattern?
Share of parameter draws whose emulated correlations reach the observed M_Y1, M_PGS1 and S_Y1 together,
for the full ranges of the designs and for the narrower ranges proposed in the notebook.
Writes results/exp4_joint_coverage.csv.
"""
import warnings

import numpy as np
import pandas as pd
from scipy.stats import qmc
from sklearn.exceptions import ConvergenceWarning

from analyze_designs import FEATURES, Emulator, load
from common import RESULTS
from run_design import DESIGNS

warnings.filterwarnings("ignore", category=ConvergenceWarning)
WINDOWS = dict(M_Y1=(0.41, 0.46), M_PGS1=(0.13, 0.19), S_Y1=(0.34, 0.40))
PROPOSED = {
    "02_freevg": dict(vg1=[0.1, 0.7], prop_h2_latent1=[0.4, 0.95], re=[0.1, 0.95], am22=[0.5, 0.95], rg=[0.3, 0.99]),
    "03_re_freevg": dict(vg1=[0.25, 0.45], f11=[0.0, 0.2], f22=[0.0, 0.25], re=[0.3, 0.95],
                         prop_h2_latent1=[0.4, 0.95], am22=[0.5, 0.95], rg=[0.3, 0.99]),
}


def share(em, lo, hi, box_lo, box_hi, n=200000, seed=7):
    u = box_lo + qmc.LatinHypercube(d=len(lo), seed=seed).random(n) * (box_hi - box_lo)
    y = em.predict((u - lo) / (hi - lo))
    ok = np.ones(n, bool)
    for name, (a, b) in WINDOWS.items():
        ok &= (y[:, FEATURES.index(name)] >= a) & (y[:, FEATURES.index(name)] <= b)
    return float(ok.mean())


def main():
    df = load("screen")
    rows = []
    for design, box in PROPOSED.items():
        free = DESIGNS[design]["free"]
        names = list(free)
        lo = np.array([free[k][0] for k in names])
        hi = np.array([free[k][1] for k in names])
        sub = df[df["design"] == design].dropna(subset=FEATURES)
        em = Emulator((sub[[f"param_{k}" for k in names]].to_numpy(float) - lo) / (hi - lo), sub[FEATURES].to_numpy(float))
        rows.append(dict(design=design, ranges="full", share=share(em, lo, hi, lo, hi)))
        rows.append(dict(design=design, ranges="proposed", share=share(em, lo, hi, np.array([box[k][0] for k in names]),
                                                                        np.array([box[k][1] for k in names]))))
    out = pd.DataFrame(rows)
    out["rows_of_20000"] = (out["share"] * 20000).round().astype(int)
    out.to_csv(RESULTS / "exp4_joint_coverage.csv", index=False)
    print(out.to_string())


if __name__ == "__main__":
    main()
