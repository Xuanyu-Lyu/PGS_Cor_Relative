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
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent

# run name -> training data (nn_training_<data>.csv), the trait the features refer to,
# and optional extra training arguments
RUNS = {
    "01DirAM_AE":                  dict(data="01DirAM", trait=1),
    "01DirAM_AE_PGSonly":          dict(data="01DirAM", trait=1),
    "01DirAM_AFE":                 dict(data="01DirAM", trait=2),
    "01DirAM_AFE_PGSonly":         dict(data="01DirAM", trait=2),
    "02AElatentAM":                dict(data="02AElatentAM", trait=1),
    "02AElatentAM_PGSonly":        dict(data="02AElatentAM", trait=1),
    "03EnvLatentAM":               dict(data="03EnvLatentAM", trait=1),
    "03EnvLatentAM_PGSonly":       dict(data="03EnvLatentAM", trait=1),
    "04AMmigration":               dict(data="04AMmigration", trait=1),
    "05AMmigration_noVT":          dict(data="05AMmigration_noVT", trait=1),
    "05AMmigration_noVT2":         dict(data="05AMmigration_noVT", trait=1),
    "05AMmigration_noVT2_PGSonly": dict(data="05AMmigration_noVT", trait=1),
    # EA-driven migration: each simulation gives one row per n_gen (3-10), so splits
    # group on 'Condition'; n_gen gets a categorical posterior.
    # Data: copy nn_training_combined_04EAmigration.csv from the HPC to nn_training_04EAmigration.csv
    "04EAmigration":               dict(data="04EAmigration", trait=1,
                                        train_args=["--group_col", "Condition", "--count_param", "n_gen"]),
}


def run_dir(run):
    return HERE / f"results_npe_unweighted_{run}"


def call(cmd, log):
    print("  $ " + " ".join(str(c) for c in cmd), flush=True)
    with open(log, "a") as fh:
        res = subprocess.run([str(c) for c in cmd], cwd=HERE, stdout=fh, stderr=subprocess.STDOUT)
    if res.returncode != 0:
        raise RuntimeError(f"command failed (exit {res.returncode}); see {log}")


def posterior_means(path):
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    return dict(zip(df["parameter"], df["mean"]))


def main():
    ap = argparse.ArgumentParser(description="Train + predict NPE for all conditions.")
    ap.add_argument("--runs", nargs="+", default=list(RUNS), choices=list(RUNS))
    ap.add_argument("--archive_tag", default=None,
                    help="archive folder name (default: before_<YYYYMMDD-HHMM>)")
    ap.add_argument("--skip_train", action="store_true", help="only re-run predictions")
    ap.add_argument("--n_samples", type=int, default=1000, help="posterior samples for prediction")
    ap.add_argument("--extra_train_args", nargs=argparse.REMAINDER, default=[],
                    help="passed through to train_npe_pgs_and_pheno.py (put last)")
    args = ap.parse_args()

    tag = args.archive_tag or "before_" + dt.datetime.now().strftime("%Y%m%d-%H%M")
    archive = HERE / "results_archive" / tag
    rows = []

    for run in args.runs:
        spec = RUNS[run]
        out = run_dir(run)
        print(f"\n=== {run} ===", flush=True)
        data_file = HERE / f"nn_training_{spec['data']}.csv"
        if not args.skip_train and not data_file.exists():
            print(f"  ⚠ skipped: training data {data_file.name} not found", flush=True)
            continue

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
                  *spec.get("train_args", []), *args.extra_train_args], log)

        call([sys.executable, "predict_npe_pgs_and_pheno.py", "--model_dir", out.name,
              "--correlations_pgs", "observed_correlations_PGS.csv",
              "--correlations_pheno", "observed_correlations_pheno.csv",
              "--n_samples", args.n_samples, "--trait", spec["trait"], "--no_pairplot"], log)

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

    if rows:
        archive.mkdir(parents=True, exist_ok=True)
        table = pd.DataFrame(rows)
        table.to_csv(archive / "posterior_mean_changes.csv", index=False)
        print(f"\nposterior means old vs new -> {(archive / 'posterior_mean_changes.csv').relative_to(HERE)}")
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
