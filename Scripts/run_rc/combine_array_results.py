"""
Helper script to combine results from all array tasks into single summary files.
Run this after all array jobs have completed.

Usage:
    python combine_array_results.py

This will:
1. Combine task-specific mate correlation files into a single summary
2. Combine task-specific correlation files into a single file
3. Generate overall summary statistics across all iterations
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Define directories
PROJECT_BASE = Path("/projects/xuly4739/Py_Projects/PGS_Cor_Relative/Data/predicted_condition_pgs_and_4pheno")
CONDITION_NAME = "Predicted_Condition_pgs_and_4pheno"

def combine_mate_correlations(project_dir):
    """
    Combine mate PGS correlation files from all tasks.
    """
    print("\n" + "="*70)
    print("Combining mate PGS correlations...")
    print("="*70)
    
    # Find all task-specific mate correlation files
    mate_files = sorted(project_dir.glob("mate_pgs_correlations_task_*.csv"))
    
    if not mate_files:
        print("  ✗ No mate correlation files found!")
        return None
    
    print(f"  Found {len(mate_files)} task files")
    
    # Read and combine
    all_mates = []
    for file in mate_files:
        df = pd.read_csv(file)
        all_mates.append(df)
        print(f"    {file.name}: {len(df)} iterations")
    
    combined_mates = pd.concat(all_mates, ignore_index=True)
    combined_mates = combined_mates.sort_values('iteration').reset_index(drop=True)
    
    # Save combined file
    output_file = project_dir / "mate_pgs_correlations_summary.csv"
    combined_mates.to_csv(output_file, index=False)
    print(f"\n  ✓ Saved combined file: {output_file.name}")
    print(f"    Total iterations: {len(combined_mates)}")
    print(f"    Trait 1 - Mean: {combined_mates['mate_pgs_correlation_trait1'].mean():.4f}, "
          f"SD: {combined_mates['mate_pgs_correlation_trait1'].std():.4f}")
    print(f"    Trait 2 - Mean: {combined_mates['mate_pgs_correlation_trait2'].mean():.4f}, "
          f"SD: {combined_mates['mate_pgs_correlation_trait2'].std():.4f}")
    
    return combined_mates

def combine_relationship_correlations(project_dir):
    """
    Combine relationship correlation files from all tasks.
    """
    print("\n" + "="*70)
    print("Combining relationship correlations...")
    print("="*70)
    
    # Find all task-specific correlation files
    corr_files = sorted(project_dir.glob("task_*_correlations.csv"))
    
    if not corr_files:
        print("  ✗ No correlation files found!")
        return None
    
    print(f"  Found {len(corr_files)} task files")
    
    # Read and combine
    all_corrs = []
    for file in corr_files:
        df = pd.read_csv(file)
        all_corrs.append(df)
        print(f"    {file.name}: {len(df)} rows")
    
    combined_corrs = pd.concat(all_corrs, ignore_index=True)
    
    # Save combined file
    output_file = project_dir / "all_iterations_correlations.csv"
    combined_corrs.to_csv(output_file, index=False)
    print(f"\n  ✓ Saved combined file: {output_file.name}")
    print(f"    Total rows: {len(combined_corrs)}")
    
    return combined_corrs

def create_summary_statistics(combined_corrs, project_dir):
    """
    Create overall summary statistics from combined correlations.
    """
    print("\n" + "="*70)
    print("Creating summary statistics...")
    print("="*70)
    
    # Group by relationship type and variable
    summary = combined_corrs.groupby(['RelationshipPath', 'Variable']).agg({
        'Correlation': ['mean', 'std', 'min', 'max'],
        'N_Pairs': 'sum',
        'Iteration': 'count'
    }).round(4)
    
    # Save summary
    summary_file = project_dir / "relationship_summary_statistics.csv"
    summary.to_csv(summary_file)
    print(f"  ✓ Saved summary statistics: {summary_file.name}")
    
    # Print PGS1 summary
    print("\n  Summary for PGS1 correlations:")
    pgs1_summary = combined_corrs[combined_corrs['Variable'] == 'PGS1'].groupby('RelationshipPath').agg({
        'Correlation': ['mean', 'std'],
        'N_Pairs': 'mean'
    }).round(4)
    print(pgs1_summary)
    
    # Print PGS2 summary
    print("\n  Summary for PGS2 correlations:")
    pgs2_summary = combined_corrs[combined_corrs['Variable'] == 'PGS2'].groupby('RelationshipPath').agg({
        'Correlation': ['mean', 'std'],
        'N_Pairs': 'mean'
    }).round(4)
    print(pgs2_summary)
    
    return summary

def main():
    """
    Main execution function.
    """
    print("\n" + "#"*70)
    print("# COMBINING ARRAY TASK RESULTS")
    print("#"*70)
    
    project_dir = PROJECT_BASE / CONDITION_NAME
    
    if not project_dir.exists():
        print(f"\n✗ Error: Directory not found: {project_dir}")
        sys.exit(1)
    
    print(f"\nProject directory: {project_dir}")
    
    # Combine mate correlations
    combined_mates = combine_mate_correlations(project_dir)
    
    # Combine relationship correlations
    combined_corrs = combine_relationship_correlations(project_dir)
    
    # Create summary statistics
    if combined_corrs is not None:
        summary = create_summary_statistics(combined_corrs, project_dir)
    
    print("\n" + "#"*70)
    print("# COMBINATION COMPLETED")
    print("#"*70)
    print(f"\nResults saved in: {project_dir}")
    print("\nKey files:")
    print("  - mate_pgs_correlations_summary.csv")
    print("  - all_iterations_correlations.csv")
    print("  - relationship_summary_statistics.csv")

if __name__ == "__main__":
    main()
