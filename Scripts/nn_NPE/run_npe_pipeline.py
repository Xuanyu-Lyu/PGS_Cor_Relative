"""
run_npe_pipeline.py
===================
Train the NPE model and predict the observed data for every condition, in one go.

For each run in ``RUNS``:
  1. archive the existing ``results_npe_unweighted_<run>/`` (moved, never deleted)
     to ``results_archive/<archive_tag>/``;
  2. train with ``train_npe_pgs_and_pheno.py`` on ``nn_training_<data>.csv``, using the
     feature and parameter lists in ``run_specs/<run>/`` (copied into the run folder);
  3. predict the observed correlations (``observed_correlations_PGS.csv`` /
     ``observed_correlations_pheno.csv``) with ``predict_npe_pgs_and_pheno.py``;
  4. carry over the predicted-condition ``summary_statistics_*.csv`` files from the
     archived ``predictions/`` folder, so Analysis_NPE.ipynb keeps working. Those
     files come from simulations at the *previous* posterior means and stay stale
     until the run_rc_paper predicted-condition jobs are rerun.

Finally, writes ``results_archive/<archive_tag>/posterior_mean_changes.csv`` with the
old and new posterior means side by side.

Usage
-----
    python run_npe_pipeline.py                         # all runs
    python run_npe_pipeline.py --runs 02AElatentAM 02AElatentAM_PGSonly
    python run_npe_pipeline.py --skip_train            # re-predict existing runs only
    python run_npe_pipeline.py --jobs 4                # train four runs at a time
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
from concurrent.futures import ThreadPoolExecutor
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent

# run name -> training data (nn_training_<data>.csv), the trait the features refer to,
# and optional extra training arguments
# nn_training_01DirAM.csv: 341 of its rows come from a later design (vg2 and s22 fixed,
# 20 generations) that was only partly simulated; keep the main design (vg2, s22 sampled).
DIRAM_ROWS = ["--row_filter", "param_vg2.notna()"]
# nn_training_05AMmigration_noVT.csv: 362 of its rows come from the earlier 5-parameter
# design (vg2, re, rg sampled); keep the am11 + move_p design.
NOVT2_ROWS = ["--row_filter", "param_vg2.isna()"]

RUNS = {
    "01DirAM_AE":                  dict(data="01DirAM", trait=1, train_args=DIRAM_ROWS),
    "01DirAM_AE_PGSonly":          dict(data="01DirAM", trait=1, train_args=DIRAM_ROWS),
    "01DirAM_AFE":                 dict(data="01DirAM", trait=2, train_args=DIRAM_ROWS),
    "01DirAM_AFE_PGSonly":         dict(data="01DirAM", trait=2, train_args=DIRAM_ROWS),
    "02AElatentAM":                dict(data="02AElatentAM", trait=1),
    "02AElatentAM_PGSonly":        dict(data="02AElatentAM", trait=1),
    # nn_training_03EnvLatentAM_all.csv: the first-design training data plus the 2026-10
    # cluster run, which reused the first design's conditions (labels restored from its
    # deterministic config); conditions with two replicates stay in one split.
    "03EnvLatentAM":               dict(data="03EnvLatentAM_all", trait=1, train_args=["--group_col", "Condition"]),
    "03EnvLatentAM_PGSonly":       dict(data="03EnvLatentAM_all", trait=1, train_args=["--group_col", "Condition"]),
    # (05AMmigration_noVT, the 5-parameter fit, is not rerun: its training data are gone;
    #  results_npe_unweighted_05AMmigration_noVT is the earlier fit, kept as is.)
    "05AMmigration_noVT2":         dict(data="05AMmigration_noVT", trait=1, train_args=NOVT2_ROWS),
    "05AMmigration_noVT2_PGSonly": dict(data="05AMmigration_noVT", trait=1, train_args=NOVT2_ROWS),
    # EA-driven migration: each simulation gives one row per n_gen (3-10), so splits
    # group on 'Condition'; n_gen gets a categorical posterior.
    # Data: copy nn_training_combined_04EAmigration.csv from the HPC to nn_training_04EAmigration.csv
    "04EAmigration":               dict(data="04EAmigration", trait=1,
                                        train_args=["--group_col", "Condition", "--count_param", "n_gen"]),
}


def run_dir(run):
    return HERE / f"results_npe_unweighted_{run}"


def call(cmd, log, threads=None):
    print("  $ " + " ".join(str(c) for c in cmd), flush=True)
    env = dict(os.environ, OMP_NUM_THREADS=str(threads)) if threads else None
    with open(log, "a") as fh:
        res = subprocess.run([str(c) for c in cmd], cwd=HERE, stdout=fh, stderr=subprocess.STDOUT,
                             env=env)
    if res.returncode != 0:
        raise RuntimeError(f"command failed (exit {res.returncode}); see {log}")


def posterior_means(path):
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    return dict(zip(df["parameter"], df["mean"]))


def process_run(run, args, archive, threads):
    spec = RUNS[run]
    rows = []
    out = run_dir(run)
    print(f"\n=== {run} ===", flush=True)
    data_file = HERE / f"nn_training_{spec['data']}.csv"
    if not args.skip_train and not data_file.exists():
        print(f"  ⚠ skipped: training data {data_file.name} not found", flush=True)
        return rows

    old_preds = out / "predictions"
    if not args.skip_train and out.exists():
        archive.mkdir(parents=True, exist_ok=True)
        dest = archive / out.name
        if dest.exists():
            raise FileExistsError(f"{dest} already exists; pick another --archive_tag")
        shutil.move(str(out), str(dest))
        old_preds = dest / "predictions"
        print(f"  archived previous run -> {dest.relative_to(HERE)}", flush=True)
    old_means = posterior_means(old_preds / "posterior_summary.csv")
    old_sumstats = sorted(old_preds.glob("summary_statistics_*.csv")) if old_preds.exists() else []

    out.mkdir(parents=True, exist_ok=True)
    log = out / "pipeline.log"
    if not args.skip_train:
        for name in ("features.txt", "params.txt"):
            shutil.copy(HERE / "run_specs" / run / name, out / name)
        call([sys.executable, "train_npe_pgs_and_pheno.py",
              "--data", f"nn_training_{spec['data']}.csv", "--output", out.name,
              "--features_file", out / "features.txt", "--params_file", out / "params.txt",
              *spec.get("train_args", []), *args.extra_train_args], log, threads)

    call([sys.executable, "predict_npe_pgs_and_pheno.py", "--model_dir", out.name,
          "--correlations_pgs", "observed_correlations_PGS.csv",
          "--correlations_pheno", "observed_correlations_pheno.csv",
          "--n_samples", args.n_samples, "--trait", spec["trait"], "--no_pairplot"], log, threads)

    new_preds = out / "predictions"
    for f in old_sumstats:
        if not (new_preds / f.name).exists():
            shutil.copy(f, new_preds / f.name)
            print(f"  carried over {f.name} (from simulations at the previous posterior means)")

    new_means = posterior_means(new_preds / "posterior_summary.csv")
    for p in new_means:
        rows.append({"run": run, "parameter": p, "old_mean": old_means.get(p),
                     "new_mean": new_means[p],
                     "change": None if p not in old_means else new_means[p] - old_means[p]})
    print("  " + "  ".join(f"{p}: {old_means.get(p, float('nan')):.4f} -> {v:.4f}"
                           for p, v in new_means.items()), flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser(description="Train + predict NPE for all conditions.")
    ap.add_argument("--runs", nargs="+", default=list(RUNS), choices=list(RUNS))
    ap.add_argument("--archive_tag", default=None,
                    help="archive folder name (default: before_<YYYYMMDD-HHMM>)")
    ap.add_argument("--skip_train", action="store_true", help="only re-run predictions")
    ap.add_argument("--n_samples", type=int, default=4000, help="posterior samples for prediction")
    ap.add_argument("--jobs", type=int, default=1, help="runs trained in parallel")
    ap.add_argument("--extra_train_args", nargs=argparse.REMAINDER, default=[],
                    help="passed through to train_npe_pgs_and_pheno.py (put last)")
    args = ap.parse_args()

    tag = args.archive_tag or "before_" + dt.datetime.now().strftime("%Y%m%d-%H%M")
    archive = HERE / "results_archive" / tag
    threads = max(1, (os.cpu_count() or 1) // args.jobs) if args.jobs > 1 else None
    rows, failed = [], []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {run: pool.submit(process_run, run, args, archive, threads) for run in args.runs}
        for run, fut in futures.items():
            try:
                rows += fut.result()
            except Exception as exc:
                failed.append(run)
                print(f"  ✗ {run} failed: {exc}", flush=True)

    if rows:
        archive.mkdir(parents=True, exist_ok=True)
        table = pd.DataFrame(rows)
        table.to_csv(archive / "posterior_mean_changes.csv", index=False)
        print(f"\nposterior means old vs new -> {(archive / 'posterior_mean_changes.csv').relative_to(HERE)}")
    if failed:
        print(f"\nFAILED: {failed}", flush=True)
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
