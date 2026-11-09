"""
Experiment 6, verification: replicated simulations of candidate fixed (re, prop_h2_latent1) pairs for
condition 03 on the corrected engine (N = 12,000, 500 variants, 6 replicates).

Each pair uses the other parameters found for it by choose_03_fixed_values.py. Pairs in --pairs are run at
vg1 = 0.35; the pair in --free_vg_pair is run at the vg1 the search preferred.

    python verify_03_fixed_values.py                  # s = 0.5, writes results/exp6_verify.jsonl
    python verify_03_fixed_values.py --s_check        # the s = 0.5 choice at smaller s, not re-tuned
    python verify_03_fixed_values.py --s 0.3 --suffix _s3 --pairs 0.7,0.6 0.1,0.7 --free_vg_pair 0.8,0.8
"""
import argparse
import json
import os

os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
from multiprocessing import Pool  # noqa: E402

import pandas as pd  # noqa: E402

from common import RESULTS  # noqa: E402
from sim_driver import simulate  # noqa: E402

KEYS = ["vg1", "f11", "f22", "am22", "rg", "re", "prop_h2_latent1"]


def row_of(table, pair):
    re, prop = (float(x) for x in pair.split(","))
    t = pd.read_csv(RESULTS / table)
    row = t[(t["re"] == re) & (t["prop_h2_latent1"] == prop)].iloc[0]
    return {k: float(row[k]) for k in KEYS}, re, prop


def one(job):
    name, params, seed, s_value = job
    out = simulate({**params, "vg2": 0.5, "s": s_value}, scale="engine", N=12000, n_cv=500, n_gen=15, seed=seed)
    out.update(design=name, rep=seed)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", type=float, default=0.5)
    ap.add_argument("--suffix", default="")
    ap.add_argument("--pairs", nargs="+", default=["0.7,0.6", "0.7,0.7", "0.1,0.7"])
    ap.add_argument("--free_vg_pair", default="0.8,0.8")
    ap.add_argument("--s_check", action="store_true")
    args = ap.parse_args()

    target_table = f"exp6_grid_vg035{args.suffix}.csv"
    if args.s_check:
        chosen, _, _ = row_of(target_table, "0.7,0.6")
        jobs = [(f"s = {v}", chosen, 7000 + r, v) for v in (0.2, 0.3, 0.4) for r in range(6)]
        target = RESULTS / "exp6_s_check.jsonl"
    else:
        sets = {}
        for pair in args.pairs:
            params, re, prop = row_of(target_table, pair)
            sets[f"re {re}, latent {prop}, vg1 0.35"] = params
        params, re, prop = row_of(f"exp6_grid{args.suffix}.csv", args.free_vg_pair)
        sets[f"re {re}, latent {prop}, vg1 at its best ({params['vg1']:.2f})"] = params
        jobs = [(name, params, 7000 + r, args.s) for name, params in sets.items() for r in range(6)]
        target = RESULTS / f"exp6_verify{args.suffix}.jsonl"

    with Pool(10) as pool, open(target, "w") as fh:
        for row in pool.imap_unordered(one, jobs, chunksize=1):
            fh.write(json.dumps(row) + "\n")
            fh.flush()


if __name__ == "__main__":
    main()
