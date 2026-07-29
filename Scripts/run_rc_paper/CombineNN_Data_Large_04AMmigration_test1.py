"""
Combine Neural Network Training Data from All Conditions - Large Dataset Version

This script combines the individual condition results from
DataGeneratingNN_Combined_04AMmigration_test1.py into a single training
dataset for neural network model training.

Only conditions where at least one iteration reached V_y equilibrium
(Status == 'Success' in iteration_status.csv) are included.

Usage:
    python CombineNN_Data_Large_04AMmigration_test1.py [--split] [--test_size 0.2]

Options:
    --split: Create train/test split
    --test_size: Proportion of data for testing (default: 0.2)
"""

import numpy as np
import pandas as pd
import argparse
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

CONDITION_BASE = Path("/projects/xuly4739/Py_Projects/PGS_Cor_Relative/Data/DataGeneratingNN_Paper/04AMmigration_test1")
OUTPUT_BASE    = Path("/projects/xuly4739/Py_Projects/PGS_Cor_Relative/Data/DataGeneratingNN_Paper")
CONDITION_TAG  = "04AMmigration_test1"

# ============================================================================
# MAIN FUNCTIONS
# ============================================================================

def load_all_conditions():
    """Load all condition results and combine into a single dataset."""
    print("\n" + "="*70)
    print("COMBINING NEURAL NETWORK TRAINING DATA")
    print(f"Condition: {CONDITION_TAG}")
    print("="*70)
    print(f"Condition base: {CONDITION_BASE}\n")

    # Find all condition directories
    condition_dirs = sorted([d for d in CONDITION_BASE.iterdir()
                             if d.is_dir() and d.name.startswith('Condition_')])

    print(f"Found {len(condition_dirs)} condition directories\n")

    all_data = []
    condition_summary = []

    n_no_equilibrium = 0

    for i, condition_dir in enumerate(condition_dirs, 1):
        condition_name = condition_dir.name

        # ---- Equilibrium status check ----
        status_file = condition_dir / "iteration_status.csv"
        if status_file.exists():
            try:
                status_df = pd.read_csv(status_file)
                n_success = (status_df['Status'] == 'Success').sum()
                n_no_eq   = (status_df['Status'] == 'No_Equilibrium').sum()
                eq_note   = f"  ({n_no_eq} iter(s) skipped: no equilibrium)" if n_no_eq > 0 else ""
            except Exception:
                n_success = None
                eq_note   = "  (could not read iteration_status.csv)"
        else:
            n_success = None
            eq_note   = "  (no iteration_status.csv)"

        # Load NN training format file
        nn_file = condition_dir / "nn_training_format.csv"

        if not nn_file.exists():
            print(f"{i:4d}. {condition_name}: ✗ No training file{eq_note}")
            condition_summary.append({
                'Condition': condition_name,
                'Status': 'Missing',
                'N_Iterations': 0,
                'N_Equilibrium_Skipped': n_no_eq if status_file.exists() else np.nan,
            })
            continue

        try:
            df = pd.read_csv(nn_file)
            n_iterations = len(df)
            all_data.append(df)

            print(f"{i:4d}. {condition_name}: ✓ {n_iterations} iterations{eq_note}")
            condition_summary.append({
                'Condition': condition_name,
                'Status': 'Success',
                'N_Iterations': n_iterations,
                'N_Equilibrium_Skipped': n_no_eq if status_file.exists() else np.nan,
            })

        except Exception as e:
            print(f"{i:4d}. {condition_name}: ✗ Error - {e}{eq_note}")
            condition_summary.append({
                'Condition': condition_name,
                'Status': 'Error',
                'N_Iterations': 0,
                'N_Equilibrium_Skipped': n_no_eq if status_file.exists() else np.nan,
            })

    # Save condition summary
    summary_df = pd.DataFrame(condition_summary)
    summary_file = OUTPUT_BASE / f"data_collection_summary_{CONDITION_TAG}.csv"
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_file, index=False)
    print(f"\n✓ Saved collection summary to {summary_file}")

    # Statistics
    n_success = sum(1 for s in condition_summary if s['Status'] == 'Success')
    n_total_iterations = sum(s['N_Iterations'] for s in condition_summary)
    n_eq_skipped_total = sum(
        s['N_Equilibrium_Skipped'] for s in condition_summary
        if not (isinstance(s['N_Equilibrium_Skipped'], float) and np.isnan(s['N_Equilibrium_Skipped']))
    )

    print(f"\nSummary:")
    print(f"  Conditions with data:          {n_success}/{len(condition_dirs)}")
    print(f"  Total iterations included:     {n_total_iterations}")
    print(f"  Total iterations skipped       ")
    print(f"  (no equilibrium):              {int(n_eq_skipped_total)}")

    if len(all_data) == 0:
        print("\n✗ No data to combine!")
        return None

    # Combine all data
    print(f"\nCombining data from {len(all_data)} conditions...")
    combined_df = pd.concat(all_data, ignore_index=True)

    print(f"✓ Combined dataset shape: {combined_df.shape}")
    print(f"  Rows: {len(combined_df)}")
    print(f"  Columns: {len(combined_df.columns)}")

    return combined_df


def create_train_test_split(df, test_size=0.2, random_state=42):
    """Create train/test split at the condition level."""
    print(f"\nCreating train/test split (test_size={test_size})...")

    # Get unique conditions
    conditions = df['Condition'].unique()
    print(f"  Total conditions: {len(conditions)}")

    # Split conditions using numpy
    np.random.seed(random_state)
    n_test = int(len(conditions) * test_size)

    # Shuffle and split
    shuffled_conditions = np.random.permutation(conditions)
    test_conditions  = shuffled_conditions[:n_test]
    train_conditions = shuffled_conditions[n_test:]

    print(f"  Train conditions: {len(train_conditions)}")
    print(f"  Test conditions:  {len(test_conditions)}")

    # Create train/test datasets
    train_df = df[df['Condition'].isin(train_conditions)].copy()
    test_df  = df[df['Condition'].isin(test_conditions)].copy()

    print(f"  Train samples: {len(train_df)}")
    print(f"  Test samples:  {len(test_df)}")

    return train_df, test_df


def save_datasets(combined_df, train_df=None, test_df=None):
    """Save combined and split datasets to OUTPUT_BASE with condition tag."""
    print("\nSaving datasets...")
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    # Save combined dataset
    combined_file = OUTPUT_BASE / f"nn_training_combined_{CONDITION_TAG}.csv"
    combined_df.to_csv(combined_file, index=False)
    print(f"✓ Saved combined dataset to {combined_file}")
    print(f"  Shape: {combined_df.shape}")

    # Save train/test splits if provided
    if train_df is not None and test_df is not None:
        train_file = OUTPUT_BASE / f"nn_training_train_{CONDITION_TAG}.csv"
        test_file  = OUTPUT_BASE / f"nn_training_test_{CONDITION_TAG}.csv"

        train_df.to_csv(train_file, index=False)
        test_df.to_csv(test_file, index=False)

        print(f"✓ Saved train dataset to {train_file}")
        print(f"  Shape: {train_df.shape}")
        print(f"✓ Saved test dataset to {test_file}")
        print(f"  Shape: {test_df.shape}")

    # Print column information
    param_cols = [col for col in combined_df.columns if col.startswith('param_')]
    other_cols = [col for col in combined_df.columns
                  if not col.startswith('param_') and col not in ['Iteration', 'Condition']]
    print(f"\nDataset columns:")
    print(f"  Parameter columns: {param_cols}")
    print(f"  Total correlation columns: {len(other_cols)}")


def print_dataset_statistics(df):
    """Print statistics about the dataset."""
    print("\n" + "="*70)
    print("DATASET STATISTICS")
    print("="*70)

    # Parameter ranges
    print("\nParameter ranges:")
    param_cols = [col for col in df.columns if col.startswith('param_')]
    for param_col in sorted(param_cols):
        values = df[param_col].dropna()
        print(f"  {param_col:30s}: [{values.min():.3f}, {values.max():.3f}]")

    # Correlation columns
    corr_cols = [col for col in df.columns
                 if not col.startswith('param_')
                 and col not in ['Iteration', 'Condition']
                 and not col.endswith('_N')]

    print(f"\nCorrelation features: {len(corr_cols)}")

    # Check for missing data
    missing_counts = df[corr_cols].isnull().sum()
    if missing_counts.sum() > 0:
        print("\nColumns with missing data:")
        for col in missing_counts[missing_counts > 0].index:
            pct = 100 * missing_counts[col] / len(df)
            print(f"  {col}: {missing_counts[col]} ({pct:.1f}%)")
    else:
        print("\n✓ No missing data in correlation columns")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description=f'Combine NN training data for condition {CONDITION_TAG}'
    )
    parser.add_argument('--split', action='store_true',
                        help='Create train/test split')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Proportion of data for testing (default: 0.2)')
    args = parser.parse_args()

    # Load and combine all condition data
    combined_df = load_all_conditions()

    if combined_df is None:
        return

    # Print statistics
    print_dataset_statistics(combined_df)

    # Create train/test split if requested
    train_df = None
    test_df  = None

    if args.split:
        train_df, test_df = create_train_test_split(combined_df, test_size=args.test_size)

    # Save datasets
    save_datasets(combined_df, train_df, test_df)

    print("\n" + "="*70)
    print("DATA COMBINATION COMPLETED SUCCESSFULLY")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
