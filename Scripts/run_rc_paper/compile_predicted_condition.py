"""
Compile summary statistics from all array-task outputs for the predicted condition
(two independent univariate Direct AM models).

Looks in PROJECT_BASE/Predicted_Condition_uni_DirAM/ for per-task CSV files:
  - task_XX_correlations.csv        -> all_correlations.csv + summary_statistics.csv
  - mate_pgs_correlations_task_XX.csv -> mate_pgs_correlations_all.csv

Run after all array tasks have completed.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# ── Directories ──────────────────────────────────────────────────────────────
PROJECT_BASE = Path("/projects/xuly4739/Py_Projects/PGS_Cor_Relative/Data/predicted_condition_uni_DirAM")
CONDITION_DIR = PROJECT_BASE / "Predicted_Condition_uni_DirAM"


def compile_correlations(condition_dir: Path) -> None:
    """Concatenate per-task correlation files and compute summary statistics."""
    task_files = sorted(condition_dir.glob("task_*_correlations.csv"))
    if not task_files:
        print("  No task_*_correlations.csv files found – skipping.")
        return

    print(f"  Found {len(task_files)} correlation file(s):")
    for f in task_files:
        print(f"    {f.name}")

    combined = pd.concat(
        [pd.read_csv(f) for f in task_files],
        ignore_index=True,
    )

    out_all = condition_dir / "all_correlations.csv"
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

    out_summary = condition_dir / "summary_statistics.csv"
    summary.to_csv(out_summary, index=False)
    print(f"  Saved summary statistics ({len(summary):,} rows) -> {out_summary.name}")

    # Quick print for PGS1 and PGS2
    for var in ["PGS1", "PGS2", "Y1", "Y2"]:
        sub = summary[summary["Variable"] == var][
            ["RelationshipPath", "Correlation_Mean", "Correlation_SD", "N_Iterations"]
        ]
        if not sub.empty:
            print(f"\n  {var} correlations by relationship:")
            print(sub.to_string(index=False))


def compile_mate_pgs(condition_dir: Path) -> None:
    """Concatenate per-task mate PGS correlation files."""
    task_files = sorted(condition_dir.glob("mate_pgs_correlations_task_*.csv"))
    if not task_files:
        print("  No mate_pgs_correlations_task_*.csv files found – skipping.")
        return

    print(f"\n  Found {len(task_files)} mate PGS correlation file(s).")

    combined = pd.concat(
        [pd.read_csv(f) for f in task_files],
        ignore_index=True,
    ).sort_values("iteration").reset_index(drop=True)

    out = condition_dir / "mate_pgs_correlations_all.csv"
    combined.to_csv(out, index=False)
    print(f"  Saved combined mate PGS correlations ({len(combined):,} rows) -> {out.name}")

    for col in ["mate_pgs_correlation_trait1", "mate_pgs_correlation_trait2"]:
        if col in combined.columns:
            vals = combined[col].dropna()
            print(f"  {col}: mean={vals.mean():.4f}  sd={vals.std():.4f}  "
                  f"min={vals.min():.4f}  max={vals.max():.4f}  n={len(vals)}")


def main() -> None:
    print("\n" + "=" * 70)
    print("COMPILE PREDICTED CONDITION RESULTS")
    print("Two independent univariate Direct AM models")
    print("=" * 70)
    print(f"Condition directory: {CONDITION_DIR}")

    if not CONDITION_DIR.exists():
        print(f"\nERROR: directory does not exist: {CONDITION_DIR}")
        sys.exit(1)

    print("\n── Relationship correlations ──")
    compile_correlations(CONDITION_DIR)

    print("\n── Mate PGS correlations ──")
    compile_mate_pgs(CONDITION_DIR)

    print("\n" + "=" * 70)
    print("COMPILATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
