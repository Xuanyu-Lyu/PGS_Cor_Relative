#!/usr/bin/env python3
"""
LatentAM.py - Run multiple simulations to compute PGS and phenotypic correlations
for various relative types under direct assortative mating.

Specifications:
- Number of simulations: 10
- Generations per simulation: 15
- Population size: 1000
- Analysis: Only final 3 generations (12, 13, 14)
- Relationship types: S, M, MS, SMS, MSC, MSM, SMSC, SMSM, PSMS, MSMC, MSMSM, SMSMSM, MSMSMS, SMSMSMS
- Variables: PGS1 (TPO1 + TMO1), PGS2 (TPO2 + TMO2), Y1, Y2
- Output: Save final 3 generations data and correlation results for each iteration
"""

import numpy as np
import pandas as pd
import os
import sys
from datetime import datetime

# Add the parent directory to the path to import SimulationFunctions
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from SimulationFunctions.SimulationFunctions import AssortativeMatingSimulation
from SimulationFunctions.save_simulation_data import save_simulation_results
from SimulationFunctions.find_relative import extract_genealogy_info, find_relationship_pairs
from SimulationFunctions.extract_measures import extract_individual_measures, extract_measures_for_pairs, compute_correlations_for_multiple_variables, save_measures_to_file


def compute_pgs_from_components(measures_df):
    """
    Compute full PGS from transmissible components.
    PGS1 = TPO1 + TMO1
    PGS2 = TPO2 + TMO2
    
    Parameters:
    -----------
    measures_df : pd.DataFrame
        DataFrame with TPO1, TMO1, TPO2, TMO2 columns
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with added PGS1 and PGS2 columns
    """
    df = measures_df.copy()
    df['PGS1'] = df['TPO1'] + df['TMO1']
    df['PGS2'] = df['TPO2'] + df['TMO2']
    return df


def run_single_iteration(iteration_num, output_base_dir, n_jobs=6):
    """
    Run a single simulation iteration.
    
    Parameters:
    -----------
    iteration_num : int
        Iteration number (1-indexed)
    output_base_dir : str
        Base directory for outputs
    n_jobs : int
        Number of CPU cores to use for relationship finding
    
    Returns:
    --------
    dict
        Summary statistics for this iteration
    """
    print(f"\n{'='*70}")
    print(f"ITERATION {iteration_num}")
    print(f"{'='*70}")
    
    # Seed for reproducibility
    seed = 12345 + iteration_num
    
    # Simulation parameters (same as test_single_pedigree)
    base_params = {
        "num_generations": 15,
        "pop_size": 1000,
        "n_CV": 100,
        "rg_effects": 0.65,
        "maf_min": 0.25,
        "maf_max": 0.45,
        "avoid_inbreeding": True,
        "save_each_gen": True,
        "save_covs": True,
        "summary_file_scope": "all",
        "seed": seed
    }
    
    # Matrices
    k2_val = np.array([[1.0, base_params["rg_effects"]], 
                       [base_params["rg_effects"], 1.0]])
    d_mat_val = np.diag([np.sqrt(.3), np.sqrt(.4)])
    a_mat_val = np.diag([np.sqrt(.5), np.sqrt(.7)])
    f_mat_val = np.array([[.2, 0.0], [0.0, .2]])
    s_mat_val = np.array([[0, 0], [0, 0]]) 
    cove_val = np.array([[0.2, 0], [0, 0.2]])
    covy_val = np.array([[1.0, 0], [0, 1.0]])
    am_correlation = np.array([[0, 0], [0, 0.65]])
    am_list_val = [am_correlation] * base_params["num_generations"]
    
    # Create iteration-specific directory
    iter_dir = os.path.join(output_base_dir, f"Iteration_{iteration_num:02d}")
    os.makedirs(iter_dir, exist_ok=True)
    
    # Set up summary file for this iteration
    summary_filename = os.path.join(iter_dir, f"iteration_{iteration_num:02d}_summary.txt")
    
    base_params.update({
        "k2_matrix": k2_val,
        "d_mat": d_mat_val,
        "a_mat": a_mat_val,
        "f_mat": f_mat_val,
        "s_mat": s_mat_val,
        "cove_mat": cove_val,
        "covy_mat": covy_val,
        "am_list": am_list_val,
        "mating_type": "phenotypic",
        "output_summary_filename": summary_filename
    })
    
    # Run simulation
    print(f"Running simulation with seed {seed}...")
    sim = AssortativeMatingSimulation(**base_params)
    results = sim.run_simulation()
    
    if results is None:
        raise RuntimeError(f"Simulation failed for iteration {iteration_num}")
    
    print(f"✓ Simulation completed successfully")
    
    # Save final three generations (12, 13, 14) data
    print("Saving final three generations data...")
    final_three_gens = [12, 13, 14]
    
    # Check if we have HISTORY data
    if 'HISTORY' not in results or 'PHEN' not in results['HISTORY']:
        raise ValueError("Results do not contain HISTORY data. Make sure save_each_gen=True in simulation.")
    
    # Save the three generations in one call using the list scope
    save_simulation_results(
        results=results,
        output_folder=iter_dir,
        file_prefix=f"iteration_{iteration_num:02d}",
        scope=final_three_gens  # Pass list of generations to save
    )
    print(f"✓ Saved data for generations {final_three_gens}")
    
    # Extract measures for all individuals (all generations needed for genealogy)
    print("Extracting individual measures...")
    variables = ['Y1', 'Y2', 'TPO1', 'TPO2', 'TMO1', 'TMO2']
    individual_measures = extract_individual_measures(results, variables)
    individual_measures = compute_pgs_from_components(individual_measures)
    print(f"✓ Extracted measures for {len(individual_measures):,} individuals")
    
    # Define relationship types to analyze
    relationship_types = [
        'S',
        'PSC',
        'PPSCC',
        'M',
        'MS',
        'SMS',
        'MSC',
        'MSM',
        'SMSC',
        'SMSM',
        'SMSMS',
        'PSMSC',
        'MSMSC',
        'MSMSM',
        'SMSMSC',
        'MSMSMS'
    ]
    
    # Find all relationship pairs and compute correlations
    print(f"Finding relationship pairs for {len(relationship_types)} types...")
    all_correlations = []
    
    for rel_path in relationship_types:
        try:
            print(f"  Processing {rel_path}...", end=' ')
            
            # Find pairs for final three generations only
            pairs = find_relationship_pairs(
                results, rel_path, 
                output_format='long', 
                n_jobs=n_jobs,
                generations=final_three_gens
            )
            
            if len(pairs) == 0:
                print(f"No pairs found")
                continue
            
            print(f"{len(pairs):,} pairs found", end='')
            
            # Extract measures for pairs - use the individual_measures we already computed
            # Instead of calling extract_measures_for_pairs which re-extracts
            pairs_with_measures = pairs.copy()
            
            # Create lookup from individual_measures (which has PGS already)
            measures_lookup = individual_measures.set_index('ID').to_dict('index')
            
            # Add measures for first individual
            for var in variables + ['PGS1', 'PGS2']:
                pairs_with_measures[f'{var}_1'] = pairs_with_measures['Person_ID'].map(
                    lambda id_val: measures_lookup.get(id_val, {}).get(var, np.nan)
                )
            
            # Add measures for second individual
            for var in variables + ['PGS1', 'PGS2']:
                pairs_with_measures[f'{var}_2'] = pairs_with_measures['Relative_ID'].map(
                    lambda id_val: measures_lookup.get(id_val, {}).get(var, np.nan)
                )
            
            print(f" - extracted measures")
            
            # Compute correlations for PGS1, PGS2, Y1, Y2
            correlation_vars = ['PGS1', 'PGS2', 'Y1', 'Y2']
            correlations = compute_correlations_for_multiple_variables(
                pairs_with_measures, correlation_vars, relationship_col='Relationship'
            )
            
            # Add iteration and relationship type info
            correlations['Iteration'] = iteration_num
            correlations['RelationshipPath'] = rel_path
            
            all_correlations.append(correlations)
            
        except Exception as e:
            print(f"Error: {e}")
            continue
    
    # Combine all correlations
    if len(all_correlations) > 0:
        correlations_df = pd.concat(all_correlations, ignore_index=True)
        
        # Reorder columns
        cols = ['Iteration', 'RelationshipPath', 'Relationship', 
                'Variable', 'N_Pairs', 'Correlation', 'P_Value']
        correlations_df = correlations_df[cols]
        
        # Save to CSV
        corr_output_path = os.path.join(iter_dir, f"correlations_iteration_{iteration_num:02d}.csv")
        correlations_df.to_csv(corr_output_path, index=False)
        print(f"✓ Saved correlation results to: {corr_output_path}")
        
        # Create summary statistics
        summary_stats = {
            'iteration': iteration_num,
            'seed': seed,
            'total_individuals': len(individual_measures),
            'relationship_types_found': len(all_correlations),
            'total_pairs': correlations_df['N_Pairs'].sum(),
            'output_directory': iter_dir
        }
        
        return summary_stats, correlations_df
    else:
        print("⚠ No relationships found")
        return None, None


def main():
    """
    Main function to run all iterations.
    """
    print(f"\n{'='*70}")
    print("Latent ASSORTATIVE MATING - PGS CORRELATION STUDY")
    print(f"{'='*70}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nConfiguration:")
    print(f"  - Number of iterations: 10")
    print(f"  - Generations per simulation: 15")
    print(f"  - Population size: 1000")
    print(f"  - Analysis generations: 12, 13, 14 (final 3)")
    print(f"  - Variables: PGS1, PGS2, Y1, Y2")
    print(f"  - CPU cores: 6")
    
    # Set up output directory
    output_base_dir = "/Users/xuly4739/Library/CloudStorage/OneDrive-UCB-O365/Documents/coding/PyProject/PGS_Cor_Relative/Data/LatentAM"
    os.makedirs(output_base_dir, exist_ok=True)
    print(f"  - Output directory: {output_base_dir}")
    
    # Run iterations
    num_iterations = 10
    all_iteration_stats = []
    all_correlations = []
    
    for i in range(1, num_iterations + 1):
        try:
            stats, correlations = run_single_iteration(i, output_base_dir, n_jobs=6)
            
            if stats is not None:
                all_iteration_stats.append(stats)
            
            if correlations is not None:
                all_correlations.append(correlations)
                
        except Exception as e:
            print(f"\n✗ ERROR in iteration {i}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Combine all correlations across iterations
    if len(all_correlations) > 0:
        print(f"\n{'='*70}")
        print("COMBINING RESULTS ACROSS ITERATIONS")
        print(f"{'='*70}")
        
        combined_correlations = pd.concat(all_correlations, ignore_index=True)
        
        # Save combined results
        combined_output_path = os.path.join(output_base_dir, "all_iterations_correlations.csv")
        combined_correlations.to_csv(combined_output_path, index=False)
        print(f"✓ Saved combined correlations to: {combined_output_path}")
        
        # Create summary statistics by relationship type
        print(f"\n{'='*70}")
        print("SUMMARY STATISTICS ACROSS ITERATIONS")
        print(f"{'='*70}")
        
        summary = combined_correlations.groupby(['RelationshipPath', 'Variable']).agg({
            'Correlation': ['mean', 'std', 'min', 'max'],
            'N_Pairs': 'sum',
            'Iteration': 'count'
        }).round(4)
        
        print("\nMean correlations by relationship type and variable:")
        print(summary.to_string())
        
        # Save summary
        summary_output_path = os.path.join(output_base_dir, "summary_statistics.csv")
        summary.to_csv(summary_output_path)
        print(f"\n✓ Saved summary statistics to: {summary_output_path}")
    
    # Save iteration stats
    if len(all_iteration_stats) > 0:
        stats_df = pd.DataFrame(all_iteration_stats)
        stats_output_path = os.path.join(output_base_dir, "iteration_summary.csv")
        stats_df.to_csv(stats_output_path, index=False)
        print(f"✓ Saved iteration summary to: {stats_output_path}")
    
    print(f"\n{'='*70}")
    print("ALL ITERATIONS COMPLETED")
    print(f"{'='*70}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Successful iterations: {len(all_iteration_stats)}/{num_iterations}")
    print(f"Total relationship types analyzed: {len(all_correlations)}")
    print(f"\nAll results saved to: {output_base_dir}")


if __name__ == "__main__":
    main()