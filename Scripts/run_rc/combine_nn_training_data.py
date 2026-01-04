"""
Combine Neural Network Training Data

This script combines all the simulation results from different parameter conditions
into a single dataset suitable for neural network training.

The output will be a CSV with:
- Input features: PGS correlations for different relationship types
- Target outputs: The 7 parameters (f11, prop_h2_latent1, vg1, vg2, f22, am22, rg)

Usage:
    python combine_nn_training_data.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Directories
PROJECT_BASE = Path("/projects/xuly4739/Py_Projects/PGS_Cor_Relative/Data/DataGeneratingNN")
OUTPUT_DIR = PROJECT_BASE / "combined"

def combine_all_conditions():
    """
    Combine results from all conditions into a single training dataset.
    """
    print("="*70)
    print("COMBINING NEURAL NETWORK TRAINING DATA")
    print("="*70)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Find all condition directories
    condition_dirs = sorted([d for d in PROJECT_BASE.iterdir() 
                           if d.is_dir() and d.name.startswith('Condition_')])
    
    print(f"\nFound {len(condition_dirs)} condition directories")
    
    all_data = []
    failed_conditions = []
    
    for condition_dir in condition_dirs:
        condition_name = condition_dir.name
        corr_file = condition_dir / "all_iterations_correlations.csv"
        
        if not corr_file.exists():
            print(f"  ✗ {condition_name}: Missing correlations file")
            failed_conditions.append(condition_name)
            continue
        
        try:
            df = pd.read_csv(corr_file)
            all_data.append(df)
            print(f"  ✓ {condition_name}: {len(df)} rows")
        except Exception as e:
            print(f"  ✗ {condition_name}: Error reading file - {e}")
            failed_conditions.append(condition_name)
    
    if not all_data:
        print("\n✗ No data to combine!")
        return
    
    # Combine all data
    print(f"\nCombining data from {len(all_data)} conditions...")
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"  Total rows: {len(combined_df):,}")
    
    # Save combined dataset
    output_file = OUTPUT_DIR / "all_conditions_raw.csv"
    combined_df.to_csv(output_file, index=False)
    print(f"\n✓ Saved raw combined data to {output_file}")
    
    # Create wide-format dataset for NN training
    print("\nCreating wide-format dataset for NN training...")
    wide_data = create_wide_format(combined_df)
    
    if wide_data is not None:
        output_file_wide = OUTPUT_DIR / "nn_training_data.csv"
        wide_data.to_csv(output_file_wide, index=False)
        print(f"✓ Saved NN training data to {output_file_wide}")
        print(f"  Shape: {wide_data.shape}")
        print(f"  Features (correlations): {wide_data.shape[1] - 7}")
        print(f"  Targets (parameters): 7")
    
    # Print summary statistics
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Successful conditions: {len(all_data)}")
    print(f"Failed conditions: {len(failed_conditions)}")
    if failed_conditions:
        print(f"Failed: {', '.join(failed_conditions)}")
    print("="*70)

def create_wide_format(df):
    """
    Convert long-format correlation data to wide format suitable for NN training.
    
    Each row represents one iteration of one condition, with:
    - Columns for correlation values of each (RelationshipPath, Variable) combination
    - Columns for the 7 parameter values
    """
    # Parameters to use as targets
    param_cols = ['param_f11', 'param_prop_h2_latent1', 'param_vg1', 
                  'param_vg2', 'param_f22', 'param_am22', 'param_rg']
    
    # Check if parameter columns exist
    if not all(col in df.columns for col in param_cols):
        print("✗ Parameter columns not found in data!")
        return None
    
    # Create unique identifier for each simulation run
    # We'll use the parameter values to create a condition ID
    df['ConditionID'] = (df['param_f11'].astype(str) + '_' + 
                         df['param_vg1'].astype(str) + '_' + 
                         df['param_rg'].astype(str))
    
    # Create feature name from relationship and variable
    df['FeatureName'] = df['RelationshipPath'] + '_' + df['Variable'] + '_cor'
    
    # Pivot to wide format
    # Each row = one iteration, columns = correlations for different relationship/variable combos
    wide_df = df.pivot_table(
        index=['ConditionID', 'Iteration'] + param_cols,
        columns='FeatureName',
        values='Correlation',
        aggfunc='first'
    ).reset_index()
    
    # Remove condition ID (not needed for training)
    wide_df = wide_df.drop('ConditionID', axis=1)
    
    # Rename parameter columns (remove 'param_' prefix)
    rename_dict = {col: col.replace('param_', '') for col in param_cols}
    wide_df = wide_df.rename(columns=rename_dict)
    
    # Reorder columns: parameters first, then correlations
    param_cols_clean = [col.replace('param_', '') for col in param_cols]
    corr_cols = [col for col in wide_df.columns 
                 if col not in param_cols_clean + ['Iteration']]
    
    column_order = ['Iteration'] + param_cols_clean + corr_cols
    wide_df = wide_df[column_order]
    
    return wide_df

def main():
    """Main execution."""
    combine_all_conditions()

if __name__ == "__main__":
    main()
