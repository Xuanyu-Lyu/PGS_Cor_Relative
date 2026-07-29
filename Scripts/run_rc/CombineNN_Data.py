"""
Combine Neural Network Training Data

This script combines all individual condition NN training files into a single
master dataset for neural network training.

Usage:
    python CombineNN_Data.py [--output_dir OUTPUT_DIR] [--output_name OUTPUT_NAME]
    
Examples:
    # Use default output location
    python CombineNN_Data.py
    
    # Specify custom output directory
    python CombineNN_Data.py --output_dir /path/to/output
    
    # Specify custom output filename
    python CombineNN_Data.py --output_name my_nn_dataset.csv
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import glob

# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_BASE = Path("/projects/xuly4739/Py_Projects/PGS_Cor_Relative/Data/DataGeneratingNN")

# ============================================================================
# MAIN FUNCTIONS
# ============================================================================

def find_all_condition_directories(base_path):
    """Find all Condition_XXXX directories."""
    condition_dirs = sorted(base_path.glob("Condition_*"))
    return [d for d in condition_dirs if d.is_dir()]

def load_condition_data(condition_dir):
    """
    Load NN training format data for a single condition.
    Returns DataFrame or None if file doesn't exist.
    """
    nn_file = condition_dir / "nn_training_format.csv"
    
    if not nn_file.exists():
        print(f"  ⚠ Missing: {condition_dir.name}/nn_training_format.csv")
        return None
    
    try:
        df = pd.read_csv(nn_file)
        print(f"  ✓ Loaded: {condition_dir.name} ({len(df)} rows)")
        return df
    except Exception as e:
        print(f"  ✗ Error loading {condition_dir.name}: {e}")
        return None

def combine_all_data(base_path, verbose=True):
    """
    Combine all condition NN training data into a single DataFrame.
    
    Args:
        base_path: Path to DataGeneratingNN directory
        verbose: Print progress messages
        
    Returns:
        Combined DataFrame with all conditions
    """
    if verbose:
        print(f"\nSearching for condition directories in {base_path}...")
    
    condition_dirs = find_all_condition_directories(base_path)
    
    if not condition_dirs:
        print(f"✗ No condition directories found in {base_path}")
        return None
    
    if verbose:
        print(f"Found {len(condition_dirs)} condition directories\n")
        print("Loading data from each condition...")
    
    all_data = []
    conditions_loaded = 0
    conditions_missing = 0
    total_rows = 0
    
    for condition_dir in condition_dirs:
        df = load_condition_data(condition_dir)
        if df is not None:
            all_data.append(df)
            conditions_loaded += 1
            total_rows += len(df)
        else:
            conditions_missing += 1
    
    if not all_data:
        print("\n✗ No data loaded from any condition")
        return None
    
    # Combine all dataframes
    if verbose:
        print(f"\nCombining data...")
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"SUMMARY")
        print(f"{'='*70}")
        print(f"Conditions found: {len(condition_dirs)}")
        print(f"Conditions loaded: {conditions_loaded}")
        print(f"Conditions missing data: {conditions_missing}")
        print(f"Total iterations: {len(combined_df)}")
        print(f"Total features: {len(combined_df.columns)}")
        print(f"{'='*70}\n")
    
    return combined_df

def save_combined_data(combined_df, output_dir, output_name="nn_training_combined.csv"):
    """
    Save combined data to CSV file.
    
    Args:
        combined_df: Combined DataFrame
        output_dir: Output directory path
        output_name: Output filename
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / output_name
    combined_df.to_csv(output_file, index=False)
    
    print(f"✓ Saved combined dataset to: {output_file}")
    print(f"  File size: {output_file.stat().st_size / 1024**2:.2f} MB")
    
    return output_file

def print_dataset_info(combined_df):
    """Print detailed information about the combined dataset."""
    print(f"\n{'='*70}")
    print(f"DATASET INFORMATION")
    print(f"{'='*70}")
    
    # Basic info
    print(f"\nShape: {combined_df.shape[0]} rows × {combined_df.shape[1]} columns")
    
    # Parameter columns
    param_cols = [col for col in combined_df.columns if col.startswith('param_')]
    print(f"\nParameter columns ({len(param_cols)}):")
    for col in param_cols:
        unique_vals = combined_df[col].nunique()
        print(f"  {col}: {unique_vals} unique values")
    
    # Correlation columns
    cor_cols = [col for col in combined_df.columns if col.startswith('cor_')]
    print(f"\nCorrelation columns ({len(cor_cols)}):")
    print(f"  {len(cor_cols)} correlation features")
    
    # Sample size columns
    n_cols = [col for col in combined_df.columns if col.startswith('n_')]
    print(f"\nSample size columns ({len(n_cols)}):")
    print(f"  {len(n_cols)} sample size features")
    
    # Missing data
    missing_counts = combined_df.isnull().sum()
    cols_with_missing = missing_counts[missing_counts > 0]
    if len(cols_with_missing) > 0:
        print(f"\nColumns with missing values ({len(cols_with_missing)}):")
        for col, count in cols_with_missing.items():
            pct = 100 * count / len(combined_df)
            print(f"  {col}: {count} ({pct:.1f}%)")
    else:
        print(f"\n✓ No missing values")
    
    # Conditions and iterations
    if 'Condition' in combined_df.columns:
        n_conditions = combined_df['Condition'].nunique()
        print(f"\nUnique conditions: {n_conditions}")
    
    if 'Iteration' in combined_df.columns:
        iterations_per_condition = combined_df.groupby('Condition')['Iteration'].count()
        print(f"Iterations per condition: {iterations_per_condition.min()} - {iterations_per_condition.max()}")
        print(f"  Mean: {iterations_per_condition.mean():.1f}")
        print(f"  Median: {iterations_per_condition.median():.0f}")
    
    print(f"\n{'='*70}\n")

def create_train_test_split(combined_df, test_size=0.2, random_state=42, by_condition=True):
    """
    Create train/test split and save separate files.
    
    Args:
        combined_df: Combined DataFrame
        test_size: Proportion for test set (0.0 to 1.0)
        random_state: Random seed for reproducibility
        by_condition: If True, split by condition; if False, split by rows
        
    Returns:
        Tuple of (train_df, test_df)
    """
    from sklearn.model_selection import train_test_split
    
    if by_condition and 'Condition' in combined_df.columns:
        # Split by condition to avoid data leakage
        conditions = combined_df['Condition'].unique()
        train_conditions, test_conditions = train_test_split(
            conditions, test_size=test_size, random_state=random_state
        )
        
        train_df = combined_df[combined_df['Condition'].isin(train_conditions)].copy()
        test_df = combined_df[combined_df['Condition'].isin(test_conditions)].copy()
        
        print(f"\nTrain/test split by condition:")
        print(f"  Train: {len(train_conditions)} conditions, {len(train_df)} rows")
        print(f"  Test: {len(test_conditions)} conditions, {len(test_df)} rows")
    else:
        # Simple random split
        train_df, test_df = train_test_split(
            combined_df, test_size=test_size, random_state=random_state
        )
        
        print(f"\nTrain/test random split:")
        print(f"  Train: {len(train_df)} rows")
        print(f"  Test: {len(test_df)} rows")
    
    return train_df, test_df

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Combine NN training data from all conditions'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory (default: same as PROJECT_BASE)'
    )
    parser.add_argument(
        '--output_name',
        type=str,
        default='nn_training_combined.csv',
        help='Output filename (default: nn_training_combined.csv)'
    )
    parser.add_argument(
        '--split',
        action='store_true',
        help='Create train/test split files'
    )
    parser.add_argument(
        '--test_size',
        type=float,
        default=0.2,
        help='Test set proportion (default: 0.2)'
    )
    parser.add_argument(
        '--no_info',
        action='store_true',
        help='Skip printing detailed dataset information'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("COMBINE NEURAL NETWORK TRAINING DATA")
    print("="*70)
    print(f"Project base: {PROJECT_BASE}")
    
    # Combine all data
    combined_df = combine_all_data(PROJECT_BASE, verbose=True)
    
    if combined_df is None:
        print("\n✗ Failed to combine data")
        return
    
    # Print dataset information
    if not args.no_info:
        print_dataset_info(combined_df)
    
    # Determine output directory
    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_BASE
    
    # Save combined data
    output_file = save_combined_data(combined_df, output_dir, args.output_name)
    
    # Create train/test split if requested
    if args.split:
        try:
            train_df, test_df = create_train_test_split(
                combined_df, 
                test_size=args.test_size,
                by_condition=True
            )
            
            # Save train and test files
            train_file = output_dir / args.output_name.replace('.csv', '_train.csv')
            test_file = output_dir / args.output_name.replace('.csv', '_test.csv')
            
            train_df.to_csv(train_file, index=False)
            test_df.to_csv(test_file, index=False)
            
            print(f"✓ Saved train set to: {train_file}")
            print(f"✓ Saved test set to: {test_file}")
        except ImportError:
            print("\n⚠ sklearn not available. Install scikit-learn to create train/test split.")
        except Exception as e:
            print(f"\n✗ Error creating train/test split: {e}")
    
    print(f"\n{'='*70}")
    print(f"COMBINATION COMPLETE")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
