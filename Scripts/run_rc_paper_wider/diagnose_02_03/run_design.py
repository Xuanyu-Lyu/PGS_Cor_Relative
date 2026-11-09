"""
Experiment 4: small-population simulations of candidate designs at the true genetic scale.

Each design is a set of free parameters (sampled by Latin hypercube) and fixed parameters.
Results are appended to results/exp4_<tag>.jsonl one line per simulation, so a run can be stopped
and resumed (finished design/index pairs are skipped).

    python run_design.py --tag screen --designs 02_base 03_base --n 120 --workers 9
"""
import argparse
import json
import os

os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
from multiprocessing import Pool  # noqa: E402

import numpy as np  # noqa: E402
from scipy.stats import qmc  # noqa: E402

from common import RESULTS  # noqa: E402
from sim_driver import simulate  # noqa: E402

VG_TARGET = [0.30, 0.40]   # the EA base genetic variance asked for
FULL = dict(am22=[0.05, 0.95], rg=[0.0, 0.99])

DESIGNS = {
    # --- condition 02: AE, assortment on a latent trait, no family transmission ---
    "02_base":   dict(free=dict(vg1=VG_TARGET, prop_h2_latent1=[0.05, 0.95], re=[0.0, 0.95], **FULL),
                      fixed=dict(vg2=0.5)),
    "02_freevg": dict(free=dict(vg1=[0.05, 0.95], prop_h2_latent1=[0.05, 0.95], re=[0.0, 0.95], **FULL),
                      fixed=dict(vg2=0.5)),
    "02_vg2":    dict(free=dict(vg1=VG_TARGET, vg2=[0.1, 0.95], prop_h2_latent1=[0.05, 0.95], re=[0.0, 0.95], **FULL),
                      fixed=dict()),
    "02_c":      dict(free=dict(vg1=VG_TARGET, vg2=[0.1, 0.6], c1=[0.0, 0.35], c2=[0.0, 0.35],
                                prop_h2_latent1=[0.05, 0.95], re=[0.0, 0.95], **FULL),
                      fixed=dict()),
    # --- condition 03: AE + vertical transmission, assortment on a latent trait ---
    "03_base":   dict(free=dict(vg1=VG_TARGET, f11=[0.0, 0.4], f22=[0.0, 0.4], **FULL),
                      fixed=dict(vg2=0.5, re=0.1, prop_h2_latent1=0.7)),
    "03_freevg": dict(free=dict(vg1=[0.05, 0.95], f11=[0.0, 0.4], f22=[0.0, 0.4], **FULL),
                      fixed=dict(vg2=0.5, re=0.1, prop_h2_latent1=0.7)),
    "03_re":     dict(free=dict(vg1=VG_TARGET, f11=[0.0, 0.4], f22=[0.0, 0.4], re=[0.0, 0.95],
                                prop_h2_latent1=[0.05, 0.95], **FULL),
                      fixed=dict(vg2=0.5)),
    "03_re_freevg": dict(free=dict(vg1=[0.05, 0.95], f11=[0.0, 0.4], f22=[0.0, 0.4], re=[0.0, 0.95],
                                   prop_h2_latent1=[0.05, 0.95], **FULL),
                         fixed=dict(vg2=0.5)),
    "03_re_c":   dict(free=dict(vg1=VG_TARGET, f11=[0.0, 0.4], f22=[0.0, 0.4], c1=[0.0, 0.35], c2=[0.0, 0.35],
                                re=[0.0, 0.95], prop_h2_latent1=[0.05, 0.95], **FULL),
                      fixed=dict(vg2=0.5)),
    # corrected engine, s_mat applied: re and prop_h2_latent1 vary only to choose their fixed values
    "03_s5":     dict(free=dict(vg1=[0.05, 0.95], f11=[0.0, 0.4], f22=[0.0, 0.4], re=[0.0, 0.95],
                                prop_h2_latent1=[0.05, 0.95], **FULL),
                      fixed=dict(vg2=0.5, s=0.5)),
    "03_s3":     dict(free=dict(vg1=[0.05, 0.95], f11=[0.0, 0.4], f22=[0.0, 0.4], re=[0.0, 0.95],
                                prop_h2_latent1=[0.05, 0.95], **FULL),
                      fixed=dict(vg2=0.5, s=0.3)),
    "03_re_vg2": dict(free=dict(vg1=VG_TARGET, vg2=[0.1, 0.95], f11=[0.0, 0.4], f22=[0.0, 0.4], re=[0.0, 0.95],
                                prop_h2_latent1=[0.05, 0.95], **FULL),
                      fixed=dict()),
}


def draws(design, n, seed):
    free = DESIGNS[design]["free"]
    names = list(free)
    u = qmc.LatinHypercube(d=len(names), seed=seed).random(n)
    lo = np.array([free[k][0] for k in names])
    hi = np.array([free[k][1] for k in names])
    return [dict(zip(names, map(float, row))) for row in lo + u * (hi - lo)]


def one(job):
    design, index, params, seed, N, n_cv, scale = job
    try:
        out = simulate(params, scale=scale, N=N, n_cv=n_cv, n_gen=15, seed=seed)
    except Exception as exc:  # keep the batch alive; the failure is recorded
        out = {"error": repr(exc)}
    out.update(design=design, index=index)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--designs", nargs="+", default=list(DESIGNS), choices=list(DESIGNS))
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--N", type=int, default=6000)
    ap.add_argument("--n_cv", type=int, default=500)
    ap.add_argument("--workers", type=int, default=9)
    ap.add_argument("--scale", default="true", choices=["true", "engine"])
    args = ap.parse_args()

    path = RESULTS / f"exp4_{args.tag}.jsonl"
    done = set()
    if path.exists():
        done = {(r["design"], r["index"]) for r in map(json.loads, open(path))}

    jobs = []
    for d_i, design in enumerate(args.designs):
        seed0 = sum(ord(c) * (i + 1) for i, c in enumerate(design))  # same draws whatever else is requested
        fixed = DESIGNS[design]["fixed"]
        for i, free in enumerate(draws(design, args.n, seed0)):
            if (design, i) not in done:
                jobs.append((design, i, {**fixed, **free}, seed0 * 1000 + i + 1, args.N, args.n_cv, args.scale))
    print(f"{len(jobs)} simulations to run ({len(done)} already done)", flush=True)

    with Pool(args.workers) as pool, open(path, "a") as fh:
        for n, row in enumerate(pool.imap_unordered(one, jobs, chunksize=1), 1):
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            if n % 25 == 0:
                print(f"{n}/{len(jobs)}", flush=True)


if __name__ == "__main__":
    main()
