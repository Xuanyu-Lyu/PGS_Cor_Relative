"""
Compile summary statistics from array-task outputs for predicted conditions.

Scans DATA_ROOT for every "predicted_condition_*" directory and, within each,
every condition subdirectory that has per-task CSV outputs:
  - task_XX_correlations.csv          -> all_correlations.csv + summary_statistics.csv
  - mate_pgs_correlations_task_XX.csv -> mate_pgs_correlations_all.csv

A condition directory is (re)compiled only if its combined outputs are
missing or older than the newest per-task file found in it, so this script
can be re-run at any time (e.g. periodically while array jobs are still
finishing) and it will only do work where there is something new to compile.

Usage:
    python compile_predicted_condition.py            # compile whatever needs it
    python compile_predicted_condition.py --force     # recompile everything found
    python compile_predicted_condition.py --dir PATH  # scan a different root
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# ── Directories ──────────────────────────────────────────────────────────────
DATA_ROOT = Path("/projects/xuly4739/Py_Projects/PGS_Cor_Relative/Data")


def find_condition_dirs(data_root: Path):
    """
    Find every directory under data_root/predicted_condition_* that contains
    per-task output files (task_*_correlations.csv or
    mate_pgs_correlations_task_*.csv).
    """
    condition_dirs = []
    for top in sorted(data_root.glob("predicted_condition_*")):
        if not top.is_dir():
            continue
        for cand in sorted(p for p in top.rglob("*") if p.is_dir()) + [top]:
            has_corr = any(cand.glob("task_*_correlations.csv"))
            has_mate = any(cand.glob("mate_pgs_correlations_task_*.csv"))
            if has_corr or has_mate:
                condition_dirs.append(cand)
    # De-duplicate while preserving order
    seen = set()
    unique_dirs = []
    for d in condition_dirs:
        if d not in seen:
            seen.add(d)
            unique_dirs.append(d)
    return unique_dirs


def newest_mtime(paths):
    return max((p.stat().st_mtime for p in paths), default=None)


def needs_compile(task_files, outputs, force: bool) -> bool:
    if force:
        return True
    if not all(o.exists() for o in outputs):
        return True
    newest_task = newest_mtime(task_files)
    oldest_output = min(o.stat().st_mtime for o in outputs)
    return newest_task is not None and newest_task > oldest_output


def compile_correlations(condition_dir: Path, force: bool) -> bool:
    """Concatenate per-task correlation files and compute summary statistics."""
    task_files = sorted(condition_dir.glob("task_*_correlations.csv"))
    if not task_files:
        return False

    out_all = condition_dir / "all_correlations.csv"
    out_summary = condition_dir / "summary_statistics.csv"

    if not needs_compile(task_files, [out_all, out_summary], force):
        print(f"  Relationship correlations up to date ({len(task_files)} task file(s)) – skipping.")
        return False

    print(f"  Found {len(task_files)} correlation file(s):")
    for f in task_files:
        print(f"    {f.name}")

    combined = pd.concat(
        [pd.read_csv(f) for f in task_files],
        ignore_index=True,
    )

    combined.to_csv(out_all, index=False)
    print(f"\n  Saved combined correlations ({len(combined):,} rows) -> {out_all.name}")

    # Summary statistics per relationship path × variable
    summary = (
        combined.groupby(["RelationshipPath", "Variable"])
        .agg(
            N_Iterations=("Iteration", "nunique"),
            N_Pairs_Total=("N_Pairs", "sum"),
            Correlation_Mean=("Correlation", "mean"),
            Correlation_SD=("Correlation", "std"),
            Correlation_Min=("Correlation", "min"),
            Correlation_Max=("Correlation", "max"),
        )
        .round(6)
        .reset_index()
    )

    summary.to_csv(out_summary, index=False)
    print(f"  Saved summary statistics ({len(summary):,} rows) -> {out_summary.name}")

    # Quick print for PGS1, PGS2, Y1, Y2
    for var in ["PGS1", "PGS2", "Y1", "Y2"]:
        sub = summary[summary["Variable"] == var][
            ["RelationshipPath", "Correlation_Mean", "Correlation_SD", "N_Iterations"]
        ]
        if not sub.empty:
            print(f"\n  {var} correlations by relationship:")
            print(sub.to_string(index=False))

    return True


def compile_mate_pgs(condition_dir: Path, force: bool) -> bool:
    """Concatenate per-task mate PGS correlation files."""
    task_files = sorted(condition_dir.glob("mate_pgs_correlations_task_*.csv"))
    if not task_files:
        return False

    out = condition_dir / "mate_pgs_correlations_all.csv"

    if not needs_compile(task_files, [out], force):
        print(f"  Mate PGS correlations up to date ({len(task_files)} task file(s)) – skipping.")
        return False

    print(f"  Found {len(task_files)} mate PGS correlation file(s).")

    combined = pd.concat(
        [pd.read_csv(f) for f in task_files],
        ignore_index=True,
    ).sort_values("iteration").reset_index(drop=True)

    combined.to_csv(out, index=False)
    print(f"  Saved combined mate PGS correlations ({len(combined):,} rows) -> {out.name}")

    for col in ["mate_pgs_correlation_trait1", "mate_pgs_correlation_trait2"]:
        if col in combined.columns:
            vals = combined[col].dropna()
            print(f"  {col}: mean={vals.mean():.4f}  sd={vals.std():.4f}  "
                  f"min={vals.min():.4f}  max={vals.max():.4f}  n={len(vals)}")

    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=DATA_ROOT,
                         help="Root directory to scan for predicted_condition_* folders "
                              f"(default: {DATA_ROOT})")
    parser.add_argument("--force", action="store_true",
                         help="Recompile every condition found, even if outputs look up to date.")
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("COMPILE PREDICTED CONDITION RESULTS")
    print("=" * 70)
    print(f"Scanning: {args.dir}")

    if not args.dir.exists():
        print(f"\nERROR: directory does not exist: {args.dir}")
        sys.exit(1)

    condition_dirs = find_condition_dirs(args.dir)
    if not condition_dirs:
        print("\nNo condition directories with task_*_correlations.csv or "
              "mate_pgs_correlations_task_*.csv files were found.")
        return

    print(f"\nFound {len(condition_dirs)} condition director{'y' if len(condition_dirs) == 1 else 'ies'} to check:")
    for d in condition_dirs:
        print(f"  {d}")

    n_updated = 0
    for condition_dir in condition_dirs:
        print("\n" + "-" * 70)
        print(f"Condition: {condition_dir}")
        print("-" * 70)

        print("\n── Relationship correlations ──")
        updated_corr = compile_correlations(condition_dir, args.force)

        print("\n── Mate PGS correlations ──")
        updated_mate = compile_mate_pgs(condition_dir, args.force)

        if updated_corr or updated_mate:
            n_updated += 1

    print("\n" + "=" * 70)
    print(f"COMPILATION COMPLETE — {n_updated}/{len(condition_dirs)} condition(s) (re)compiled")
    print("=" * 70)


if __name__ == "__main__":
    main()
