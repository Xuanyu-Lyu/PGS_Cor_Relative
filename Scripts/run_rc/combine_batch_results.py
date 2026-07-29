"""
Combine batch results from SLURM array jobs into final summary files.
Run this after all SLURM tasks complete to merge batch-level summaries.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import glob

def combine_uni_results():
    """Combine univariate approximation batch results."""
    print("\n" + "="*70)
    print("COMBINING UNIVARIATE APPROXIMATION RESULTS")
    print("="*70)
    
    project_dir = Path("/projects/xuly4739/Py_Projects/PGS_Cor_Relative/Data/approximation_uni")
    
    # Combine trait 1 mate correlations
    trait1_files = sorted(project_dir.glob("mate_pgs_correlation_trait1_batch_*.csv"))
    if trait1_files:
        dfs = [pd.read_csv(f) for f in trait1_files]
        combined = pd.concat(dfs, ignore_index=True).sort_values('iteration').reset_index(drop=True)
        output_file = project_dir / "mate_pgs_correlation_trait1_summary.csv"
        combined.to_csv(output_file, index=False)
        print(f"\n✓ Combined {len(trait1_files)} batches for Trait 1")
        print(f"  Total iterations: {len(combined)}")
        print(f"  Mean: {combined['mate_pgs_correlation_trait1'].mean():.4f}")
        print(f"  SD: {combined['mate_pgs_correlation_trait1'].std():.4f}")
        print(f"  Saved to: {output_file}")
    
    # Combine trait 2 mate correlations
    trait2_files = sorted(project_dir.glob("mate_pgs_correlation_trait2_batch_*.csv"))
    if trait2_files:
        dfs = [pd.read_csv(f) for f in trait2_files]
        combined = pd.concat(dfs, ignore_index=True).sort_values('iteration').reset_index(drop=True)
        output_file = project_dir / "mate_pgs_correlation_trait2_summary.csv"
        combined.to_csv(output_file, index=False)
        print(f"\n✓ Combined {len(trait2_files)} batches for Trait 2")
        print(f"  Total iterations: {len(combined)}")
        print(f"  Mean: {combined['mate_pgs_correlation_trait2'].mean():.4f}")
        print(f"  SD: {combined['mate_pgs_correlation_trait2'].std():.4f}")
        print(f"  Saved to: {output_file}")
    
    # Combine all correlations
    corr_files = sorted(project_dir.glob("correlations_batch_*.csv"))
    if corr_files:
        dfs = [pd.read_csv(f) for f in corr_files]
        combined = pd.concat(dfs, ignore_index=True)
        output_file = project_dir / "all_iterations_correlations.csv"
        combined.to_csv(output_file, index=False)
        print(f"\n✓ Combined {len(corr_files)} correlation batches")
        print(f"  Total rows: {len(combined):,}")
        print(f"  Saved to: {output_file}")
        
        # Create overall summary by relationship type
        summary = combined.groupby(['RelationshipPath', 'Variable']).agg({
            'Correlation': ['mean', 'std', 'min', 'max'],
            'N_Pairs': 'sum',
            'Iteration': 'count'
        }).round(4)
        summary_file = project_dir / "relationship_summary_statistics.csv"
        summary.to_csv(summary_file)
        print(f"  Saved summary to: {summary_file}")

def combine_bi_results():
    """Combine bivariate approximation batch results for all conditions."""
    print("\n" + "="*70)
    print("COMBINING BIVARIATE APPROXIMATION RESULTS")
    print("="*70)
    
    project_base = Path("/projects/xuly4739/Py_Projects/PGS_Cor_Relative/Data/approximation_bi")
    conditions = ['Condition_01', 'Condition_02', 'Condition_03', 'Condition_04']
    
    all_condition_summaries = []
    
    for condition_name in conditions:
        print(f"\n{condition_name}:")
        condition_dir = project_base / condition_name
        
        if not condition_dir.exists():
            print(f"  ✗ Directory not found: {condition_dir}")
            continue
        
        # Combine trait 1 mate correlations for this condition
        trait1_files = sorted(condition_dir.glob("mate_pgs_correlation_trait1_batch_*.csv"))
        if trait1_files:
            dfs = [pd.read_csv(f) for f in trait1_files]
            combined = pd.concat(dfs, ignore_index=True).sort_values('iteration').reset_index(drop=True)
            output_file = condition_dir / "mate_pgs_correlation_trait1_summary.csv"
            combined.to_csv(output_file, index=False)
            
            mean_cor = combined['mate_pgs_correlation_trait1'].mean()
            sd_cor = combined['mate_pgs_correlation_trait1'].std()
            
            print(f"  ✓ Combined {len(trait1_files)} batches for Trait 1")
            print(f"    Total iterations: {len(combined)}")
            print(f"    Mean: {mean_cor:.4f}, SD: {sd_cor:.4f}")
            
            # Store for overall summary
            all_condition_summaries.append({
                'Condition': condition_name,
                'Mean_Mate_Cor_PGS1': mean_cor,
                'SD_Mate_Cor_PGS1': sd_cor,
                'N_Iterations': len(combined)
            })
        
        # Combine all correlations for this condition
        corr_files = sorted(condition_dir.glob("correlations_batch_*.csv"))
        if corr_files:
            dfs = [pd.read_csv(f) for f in corr_files]
            combined = pd.concat(dfs, ignore_index=True)
            output_file = condition_dir / "all_iterations_correlations.csv"
            combined.to_csv(output_file, index=False)
            print(f"  ✓ Combined {len(corr_files)} correlation batches ({len(combined):,} rows)")
            
            # Create summary by relationship type
            summary = combined.groupby(['RelationshipPath', 'Variable']).agg({
                'Correlation': ['mean', 'std', 'min', 'max'],
                'N_Pairs': 'sum',
                'Iteration': 'count'
            }).round(4)
            summary_file = condition_dir / "relationship_summary_statistics.csv"
            summary.to_csv(summary_file)
    
    # Create overall summary across all conditions
    if all_condition_summaries:
        overall_summary = pd.DataFrame(all_condition_summaries)
        overall_file = project_base / "overall_conditions_summary.csv"
        overall_summary.to_csv(overall_file, index=False)
        print(f"\n✓ Saved overall summary to: {overall_file}")
        print("\nOverall Summary:")
        print(overall_summary.to_string(index=False))

def main():
    """Main function to combine all results."""
    print("\n" + "="*70)
    print("COMBINING SLURM BATCH RESULTS")
    print("="*70)
    
    combine_uni_results()
    combine_bi_results()
    
    print("\n" + "="*70)
    print("ALL RESULTS COMBINED")
    print("="*70)

if __name__ == "__main__":
    main()
