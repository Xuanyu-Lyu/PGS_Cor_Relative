"""Experiment 1: does vg1 set the base genetic variance? No AM, no VT, engine vs true scale."""
import os
os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

from sim_driver import simulate

VG1 = [0.05, 0.10, 0.20, 0.35, 0.50, 0.80, 0.95]


def one(job):
    scale, vg1, seed = job
    return simulate(dict(vg1=vg1, vg2=0.5, prop_h2_latent1=0.7), scale=scale, N=8000, n_cv=500,
                    n_gen=4, seed=seed, rels=["S", "M", "SC", "PSC"])


if __name__ == "__main__":
    jobs = [(s, v, 100 * i + r + 1) for s in ("engine", "true") for i, v in enumerate(VG1) for r in range(3)]
    with Pool(10) as pool:
        rows = pool.map(one, jobs)
    out = Path(__file__).parent / "results" / "exp1_engine_scale.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print("wrote", out, len(rows))
