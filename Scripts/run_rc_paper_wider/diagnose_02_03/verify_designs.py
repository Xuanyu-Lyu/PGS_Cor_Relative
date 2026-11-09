"""
Experiment 5: re-simulate each design's best parameters at a larger population, with replicates,
next to the current production estimates of conditions 02 and 03 (same N, same seeds).

Writes results/exp5_verify.jsonl.

    python verify_designs.py --workers 8 --reps 6
"""
import argparse
import json
import os

os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
from multiprocessing import Pool  # noqa: E402

import pandas as pd  # noqa: E402

from common import RESULTS  # noqa: E402
from run_design import DESIGNS  # noqa: E402
from sim_driver import simulate  # noqa: E402

# Posterior means now in run_rc_paper/run_predicted_condition_rc_{02AElatentAM,03EnvLatentAM}.py
CURRENT = {
    "02_current": dict(prop_h2_latent1=0.6736, vg1=0.0567, re=0.8062, am22=0.7259, rg=0.6727, vg2=0.5),
    "03_current": dict(vg1=0.0918, f11=0.0919, f22=0.1122, am22=0.6941, rg=0.9169, re=0.1,
                       prop_h2_latent1=0.7, vg2=0.5),
}


def one(job):
    name, params, scale, seed, N, n_cv = job
    try:
        out = simulate(params, scale=scale, N=N, n_cv=n_cv, n_gen=15, seed=seed)
    except Exception as exc:
        out = {"error": repr(exc)}
    out.update(design=name, rep=seed)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--N", type=int, default=12000)
    ap.add_argument("--n_cv", type=int, default=500)
    ap.add_argument("--best", default="exp4_best_params.csv")
    ap.add_argument("--out", default="exp5_verify.jsonl")
    args = ap.parse_args()

    best = pd.read_csv(RESULTS / args.best)
    sets = {name: (params, "engine") for name, params in CURRENT.items()}
    for _, row in best.iterrows():
        design = row["design"]
        spec = DESIGNS[design]
        sets[design] = ({**spec["fixed"], **{k: float(row[k]) for k in spec["free"]}}, "true")

    path = RESULTS / args.out
    done = set()
    if path.exists():
        done = {(r["design"], r["rep"]) for r in map(json.loads, open(path))}
    jobs = [(name, params, scale, 5000 + r, args.N, args.n_cv)
            for name, (params, scale) in sets.items() for r in range(args.reps) if (name, 5000 + r) not in done]
    print(f"{len(jobs)} simulations to run", flush=True)
    with Pool(args.workers) as pool, open(path, "a") as fh:
        for row in pool.imap_unordered(one, jobs, chunksize=1):
            fh.write(json.dumps(row) + "\n")
            fh.flush()


if __name__ == "__main__":
    main()
