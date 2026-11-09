"""
Experiment 3: refit the NPE on subsets or noisier versions of the existing wider training data.

Variants (each for 02AElatentAM_wider and 03EnvLatentAM_wider):
  vgwin    only rows with vg1 in [0.25, 0.45]
  noise    every row twice, with observation noise of the size of the observed SE added to the features
  both     the two together

The trainer and predictor are the production scripts, unchanged. Output: results/npe/<fit>_<variant>/.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common import NPE, RESULTS, load_training

OBS = ["--correlations_pgs", "observed_correlations_PGS_ea4new.csv",
       "--correlations_pheno", "observed_correlations_pheno_021126.csv"]
CONDS = ["02AElatentAM_wider", "03EnvLatentAM_wider"]
VARIANTS = ["vgwin", "noise", "both"]


def make_data(cond, variant, path):
    df, features, _, _, se = load_training(cond)
    if variant in ("vgwin", "both"):
        df = df[df["param_vg1"].between(0.25, 0.45)].copy()
    if variant in ("noise", "both"):
        rng = np.random.default_rng(20261108)
        copies = []
        for _ in range(2):
            c = df.copy()
            c[features] = c[features].to_numpy(float) + rng.normal(0, se, size=(len(c), len(se)))
            copies.append(c)
        df = pd.concat(copies, ignore_index=True)
    df.to_csv(path, index=False)
    return len(df)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--variants", nargs="+", default=VARIANTS, choices=VARIANTS)
    ap.add_argument("--conds", nargs="+", default=CONDS, choices=CONDS)
    args = ap.parse_args()
    env = dict(os.environ, OMP_NUM_THREADS=str(args.threads), OPENBLAS_NUM_THREADS=str(args.threads),
               MKL_NUM_THREADS=str(args.threads))
    base = RESULTS / "npe"
    base.mkdir(exist_ok=True)
    for cond in args.conds:
        for variant in args.variants:
            name = f"{cond}_{variant}"
            out = base / name
            if (out / "predictions" / "posterior_summary.csv").exists():
                print("done already:", name, flush=True)
                continue
            data = base / f"{name}.csv"
            n = make_data(cond, variant, data)
            print(f"{name}: {n} rows", flush=True)
            spec = NPE / "run_specs" / cond
            log = open(base / f"{name}.log", "w")
            train = [sys.executable, "train_npe_pgs_and_pheno.py", "--data", str(data), "--output", str(out),
                     "--features_file", str(spec / "features.txt"), "--params_file", str(spec / "params.txt"),
                     "--group_col", "Condition"]
            predict = [sys.executable, "predict_npe_pgs_and_pheno.py", "--model_dir", str(out), *OBS,
                       "--n_samples", "4000", "--n_samples_mcmc", "20000", "--trait", "1",
                       "--output", str(out / "predictions"), "--no_pairplot"]
            for cmd in (train, predict):
                res = subprocess.run(cmd, cwd=NPE, stdout=log, stderr=subprocess.STDOUT, env=env)
                if res.returncode != 0:
                    print(f"{name}: FAILED ({cmd[1]}, exit {res.returncode}); see {name}.log", flush=True)
                    break
            else:
                print(f"{name}: finished", flush=True)
            data.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
