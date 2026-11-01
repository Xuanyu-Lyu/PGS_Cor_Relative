"""
Combine Neural Network Training Data from All Conditions - wider-prior designs

Combines the per-condition results of DataGeneratingNN_Combined_<condition>.py (one of
01DirAM_wider, 02AElatentAM_wider, 03EnvLatentAM_wider, 04EAmigration_wider) into a
single training dataset.

Run it after the whole array has finished (submit_combine_nn_wider.sh <condition>, with
--dependency=afterany on the array job; submit_all_wider.sh sets this up). It checks every
condition against conditions_config.csv: a condition whose stored parameters differ from the
config (output of another design) is left out, and if any condition is missing or left out,
nothing is written unless --allow_incomplete is given.

For 04EAmigration_wider, each condition (one simulation) contributes one row per n_gen
value (3-10). Those rows share a simulation, so the train/test split is made at the
condition level; keep 'Condition' as the group column when training.

Usage:
    python CombineNN_Data_Large_wider.py <condition> [--split] [--test_size 0.2] [--allow_incomplete]

Output: Data/to_nn_NPE/nn_training_<condition>.csv, the name Scripts/nn_NPE trains on.
Data/to_nn_NPE/ is laid out like Scripts/nn_NPE (compile_predicted_condition.py writes the
predicted-condition summaries there too), so one copy brings everything into place:
    rsync -av <curc>:/projects/xuly4739/Py_Projects/PGS_Cor_Relative/Data/to_nn_NPE/ Scripts/nn_NPE/
The collection summary and the --split files stay in Data/DataGeneratingNN_Paper_wider/.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================================
# CONFIGURATION
# ============================================================================

OUTPUT_BASE = Path("/projects/xuly4739/Py_Projects/PGS_Cor_Relative/Data/DataGeneratingNN_Paper_wider")
NN_NPE_DIR = Path("/projects/xuly4739/Py_Projects/PGS_Cor_Relative/Data/to_nn_NPE")   # mirrors Scripts/nn_NPE
CONDITIONS = ["01DirAM_wider", "02AElatentAM_wider", "03EnvLatentAM_wider", "04EAmigration_wider"]
ID_COLS = ["Iteration", "Condition", "Sample"]

# ============================================================================
# MAIN FUNCTIONS
# ============================================================================

def matches_config(df, config_row):
    """True if every sampled param_ column of a condition's output equals its config value.

    n_gen (04) is the read-out generation, not a config parameter.
    """
    for col in (c for c in df.columns if c.startswith("param_") and c != "param_n_gen"):
        name = col[len("param_"):]
        if name not in config_row.index or not np.allclose(df[col].to_numpy(float), float(config_row[name])):
            return False
    return True


def load_all_conditions(condition_base, tag, allow_incomplete=False):
    """Load all condition results and combine them into a single dataset."""
    print("\n" + "=" * 70)
    print(f"COMBINING NEURAL NETWORK TRAINING DATA: {tag}")
    print("=" * 70)
    print(f"Condition base: {condition_base}\n")
    config = pd.read_csv(condition_base / "conditions_config.csv").set_index("name")

    condition_dirs = sorted(d for d in condition_base.iterdir()
                            if d.is_dir() and d.name.startswith("Condition_"))
    print(f"Found {len(condition_dirs)} condition directories\n")

    all_data, condition_summary = [], []
    for i, condition_dir in enumerate(condition_dirs, 1):
        name = condition_dir.name
        nn_file = condition_dir / "nn_training_format.csv"
        if not nn_file.exists():
            print(f"{i:5d}. {name}: ✗ No training file")
            condition_summary.append({"Condition": name, "Status": "Missing", "N_Rows": 0})
            continue
        try:
            df = pd.read_csv(nn_file)
        except Exception as e:
            print(f"{i:5d}. {name}: ✗ Error - {e}")
            condition_summary.append({"Condition": name, "Status": "Error", "N_Rows": 0})
            continue
        if name not in config.index or not matches_config(df, config.loc[name]):
            print(f"{i:5d}. {name}: ✗ parameters differ from conditions_config.csv (other design)")
            condition_summary.append({"Condition": name, "Status": "Wrong design", "N_Rows": 0})
            continue
        all_data.append(df)
        condition_summary.append({"Condition": name, "Status": "Success", "N_Rows": len(df)})

    summary_file = OUTPUT_BASE / f"data_collection_summary_{tag}.csv"
    pd.DataFrame(condition_summary).to_csv(summary_file, index=False)
    print(f"\n✓ Saved collection summary to {summary_file}")

    n_success = sum(s["Status"] == "Success" for s in condition_summary)
    print(f"\nSummary:")
    print(f"  Conditions with data: {n_success}/{len(config)} configured")
    print(f"  Total rows: {sum(s['N_Rows'] for s in condition_summary)}")
    n_missing = len(config) - n_success
    if n_missing:
        print(f"\n⚠ {n_missing} of the {len(config)} configured conditions are missing or from another design.")
        if not allow_incomplete:
            print("  Nothing written. Wait for the array to finish (or rerun failed tasks), "
                  "or pass --allow_incomplete.")
            return None
    if not all_data:
        print("\n✗ No data to combine!")
        return None

    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"\n✓ Combined dataset shape: {combined_df.shape}")
    return combined_df


def create_train_test_split(df, test_size=0.2, random_state=42):
    """Create a train/test split at the condition level."""
    print(f"\nCreating train/test split (test_size={test_size})...")
    conditions = df["Condition"].unique()
    np.random.seed(random_state)
    shuffled = np.random.permutation(conditions)
    n_test = int(len(conditions) * test_size)
    test_conditions, train_conditions = shuffled[:n_test], shuffled[n_test:]
    train_df = df[df["Condition"].isin(train_conditions)].copy()
    test_df = df[df["Condition"].isin(test_conditions)].copy()
    print(f"  Train: {len(train_conditions)} conditions, {len(train_df)} rows")
    print(f"  Test:  {len(test_conditions)} conditions, {len(test_df)} rows")
    return train_df, test_df


def save_datasets(tag, combined_df, train_df=None, test_df=None):
    """Save the combined dataset to NN_NPE_DIR and the split datasets to OUTPUT_BASE."""
    print("\nSaving datasets...")
    NN_NPE_DIR.mkdir(parents=True, exist_ok=True)
    combined_file = NN_NPE_DIR / f"nn_training_{tag}.csv"
    combined_df.to_csv(combined_file, index=False)
    print(f"✓ Saved combined dataset to {combined_file} {combined_df.shape}")
    if train_df is not None and test_df is not None:
        for kind, d in (("train", train_df), ("test", test_df)):
            f = OUTPUT_BASE / f"nn_training_{kind}_{tag}.csv"
            d.to_csv(f, index=False)
            print(f"✓ Saved {kind} dataset to {f} {d.shape}")


def print_dataset_statistics(df):
    """Print parameter ranges and missing correlation columns."""
    print("\n" + "=" * 70)
    print("DATASET STATISTICS")
    print("=" * 70)
    if "param_n_gen" in df.columns:
        counts = df["param_n_gen"].value_counts().sort_index()
        print("\nRows per n_gen: " + ", ".join(f"{int(g)}: {n}" for g, n in counts.items()))
        print(f"Simulations (conditions): {df['Condition'].nunique()}")

    print("\nParameter ranges:")
    for col in sorted(c for c in df.columns if c.startswith("param_")):
        values = df[col].dropna()
        print(f"  {col:30s}: [{values.min():.3f}, {values.max():.3f}]")

    corr_cols = [c for c in df.columns
                 if not c.startswith("param_") and c not in ID_COLS and not c.endswith(("_N", "_cov"))]
    print(f"\nCorrelation features: {len(corr_cols)}")
    missing = df[corr_cols].isnull().sum()
    if missing.sum() > 0:
        print("\nColumns with missing data:")
        for col in missing[missing > 0].index:
            print(f"  {col}: {missing[col]} ({100 * missing[col] / len(df):.1f}%)")
    else:
        print("\n✓ No missing data in correlation columns")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Combine NN training data of a wider-prior condition")
    parser.add_argument("condition", choices=CONDITIONS)
    parser.add_argument("--split", action="store_true", help="Create train/test split")
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Proportion of conditions for testing (default: 0.2)")
    parser.add_argument("--allow_incomplete", action="store_true",
                        help="Write the dataset even if some configured conditions are missing")
    args = parser.parse_args()

    combined_df = load_all_conditions(OUTPUT_BASE / args.condition, args.condition,
                                      allow_incomplete=args.allow_incomplete)
    if combined_df is None:
        return
    print_dataset_statistics(combined_df)
    train_df = test_df = None
    if args.split:
        train_df, test_df = create_train_test_split(combined_df, test_size=args.test_size)
    save_datasets(args.condition, combined_df, train_df, test_df)

    print("\n" + "=" * 70)
    print("DATA COMBINATION COMPLETED SUCCESSFULLY")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
