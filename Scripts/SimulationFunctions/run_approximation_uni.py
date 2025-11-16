"""
Simulation script for univariate approximation conditions.
Runs 50 iterations with N=20000 for two optimal parameter conditions.
Saves data for final 3 generations and PGS correlations summary statistics.
"""

import numpy as np
import pandas as pd
import os
import sys
from pathlib import Path

# Add the SimulationFunctions directory to path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

from SimulationFunctions import AssortativeMatingSimulation
from save_simulation_data import save_simulation_results
from find_relative_setbased import find_relationship_pairs
from extract_measures import extract_individual_measures, compute_correlations_for_multiple_variables

# Define output directory
OUTPUT_DIR = Path(__file__).parent.parent.parent / "Data" / "approximation_uni"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Combined bivariate simulation parameters
# Trait 1 uses Condition 1 (f11=0.20) and Trait 2 uses Condition 2 (f22=0.19)
SIMULATION_PARAMS = {
    'name': 'UnivariateCombined',
    # Trait 1 parameters (Condition 1: f11=0.20)
    'f11': 0.1500,
    'prop_h2_latent1': 0.72500,
    'vg1': 0.8100,
    'am11': 0.43,
    
    # Trait 2 parameters (Condition 2: f22=0.19)
    'f22': 0.1200,
    'prop_h2_latent2': 0.7500,
    'vg2': 0.8100,
    'am22': 0.43,
    
    # Cross-trait parameters (no cross-trait effects)
    'f12': 0.0,
    'f21': 0.0,
    'am12': 0,
    'am21': 0,
    'rg': 0,
    're': 0
}

# Simulation parameters
N_ITERATIONS = 50
POP_SIZE = 20000
N_GENERATIONS = 15  # Total generations (will save last 3)
FINAL_GENS = [12, 13, 14]  # Final 3 generations to analyze
N_CV = 1000
MAF_MIN = 0.01
MAF_MAX = 0.5

# Relationship types to analyze
RELATIONSHIP_TYPES = [
    'S',        # Siblings
    'PSC',      # Parent-sibling-child (avuncular)
    'PPSCC',    # First cousins
    'M',        # Mates
    'MS',       # Mate's sibling (sibling-in-law)
    'SMS',      # Sibling's mate's sibling
    'MSC',      # Mate's sibling's child (nibling-in-law)
    'MSM',      # Mate's sibling's mate
    'SMSC',     # Sibling's mate's sibling's child
    'SMSM',     # Sibling's mate's sibling's mate
    'SMSMS',    # Sibling's mate's sibling's mate's sibling
    'PSMSC',    # More distant relatives
    'MSMSC',
    'MSMSM',
    'SMSMSC',
    'MSMSMS'
]

def setup_matrices(params):
    """
    Setup covariance and other matrices based on simulation parameters.
    """
    vg1 = params['vg1']
    vg2 = params['vg2']
    rg = params['rg']
    re = params['re']
    prop_h2_latent1 = params['prop_h2_latent1']
    prop_h2_latent2 = params['prop_h2_latent2']
    
    # Implied variables (t0)
    k2_matrix = np.array([[1, rg], [rg, 1]])
    
    # Observable genetic variance components
    vg_obs1 = vg1 * (1 - prop_h2_latent1)
    vg_obs2 = vg2 * (1 - prop_h2_latent2)
    d11 = np.sqrt(vg_obs1)
    d21 = 0
    d22 = np.sqrt(vg_obs2 - d21**2)
    delta_mat = np.array([[d11, 0], [d21, d22]])
    
    # Latent genetic variance components
    vg_lat1 = vg1 * prop_h2_latent1
    vg_lat2 = vg2 * prop_h2_latent2
    a11 = np.sqrt(vg_lat1)
    a21 = 0
    a22 = np.sqrt(vg_lat2 - a21**2)
    a_mat = np.array([[a11, 0], [a21, a22]])
    
    # Total genetic covariance
    covg_mat = (delta_mat @ k2_matrix @ delta_mat.T) + (a_mat @ k2_matrix @ a_mat.T)
    
    # Environmental covariance
    ve1 = 1 - vg1
    ve2 = 1 - vg2
    cove = re * np.sqrt(ve1 * ve2)
    cove_mat = np.array([[ve1, cove], [cove, ve2]])
    
    # Total phenotypic covariance
    covy_mat = covg_mat + cove_mat
    
    # Assortative mating matrix
    am11 = params['am11']
    am12 = params['am12']
    am21 = params['am21']
    am22 = params['am22']
    mate_cor_mat = np.array([[am11, am12], [am21, am22]])
    
    # Vertical transmission matrix
    f11 = params['f11']
    f12 = params['f12']
    f21 = params['f21']
    f22 = params['f22']
    f_mat = np.array([[f11, f12], [f21, f22]])
    
    # Social homogamy matrix (set to zero - phenotypic AM only)
    s_mat = np.zeros((2, 2))
    
    # AM list: one mate correlation matrix per generation (for phenotypic AM)
    # The list should have num_generations entries, all with the same correlation
    am_list = [mate_cor_mat.copy() for _ in range(N_GENERATIONS)]
    
    return {
        'cove_mat': cove_mat,
        'f_mat': f_mat,
        's_mat': s_mat,
        'a_mat': a_mat,
        'd_mat': delta_mat,
        'am_list': am_list,
        'covy_mat': covy_mat,
        'k2_matrix': k2_matrix
    }

def run_single_iteration(iteration, params, matrices):
    """
    Run a single simulation iteration.
    """
    print(f"\n{'='*60}")
    print(f"Running {params['name']} - Iteration {iteration + 1}/{N_ITERATIONS}")
    print(f"{'='*60}")
    
    # Set seed for reproducibility
    seed = iteration + 1
    
    # Initialize simulation
    sim = AssortativeMatingSimulation(
        n_CV=N_CV,
        rg_effects=params['rg'],
        maf_min=MAF_MIN,
        maf_max=MAF_MAX,
        num_generations=N_GENERATIONS,
        pop_size=POP_SIZE,
        mating_type="phenotypic",
        avoid_inbreeding=True,
        save_each_gen=True,
        save_covs=True,
        seed=seed,
        **matrices
    )
    
    # Run simulation
    results = sim.run_simulation()
    
    return results

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

def extract_and_analyze_relationships(results, iteration, output_dir):
    """
    Extract measures and analyze correlations for different relationship types.
    
    Parameters:
    -----------
    results : dict
        Simulation results
    iteration : int
        Iteration number (0-indexed)
    output_dir : Path
        Output directory
    
    Returns:
    --------
    pd.DataFrame or None
        DataFrame with correlation results for all relationship types
    """
    print(f"\n  Extracting individual measures...")
    
    # Extract measures for all individuals (needed for genealogy)
    # Use TPO/TMO components to compute PGS, plus Y1, Y2
    variables = ['Y1', 'Y2', 'TPO1', 'TPO2', 'TMO1', 'TMO2']
    
    try:
        individual_measures = extract_individual_measures(results, variables)
        # Compute PGS from transmissible components
        individual_measures = compute_pgs_from_components(individual_measures)
        print(f"  ✓ Extracted measures for {len(individual_measures):,} individuals")
    except Exception as e:
        print(f"  ✗ Error extracting measures: {e}")
        return None
    
    # Create lookup for individual measures
    measures_lookup = individual_measures.set_index('ID').to_dict('index')
    
    # Find all relationship pairs and compute correlations
    print(f"  Finding relationship pairs for {len(RELATIONSHIP_TYPES)} types...")
    all_correlations = []
    
    for rel_path in RELATIONSHIP_TYPES:
        try:
            # Find pairs for final three generations only
            pairs = find_relationship_pairs(
                results, rel_path,
                output_format='long',
                generations=FINAL_GENS
            )
            
            if len(pairs) == 0:
                continue
            
            print(f"    {rel_path}: {len(pairs):,} pairs", end='')
            
            # Add measures for pairs using lookup
            pairs_with_measures = pairs.copy()
            
            # Add measures for first individual
            for var in variables:
                pairs_with_measures[f'{var}_1'] = pairs_with_measures['Person_ID'].map(
                    lambda id_val: measures_lookup.get(id_val, {}).get(var, np.nan)
                )
            
            # Add measures for second individual
            for var in variables:
                pairs_with_measures[f'{var}_2'] = pairs_with_measures['Relative_ID'].map(
                    lambda id_val: measures_lookup.get(id_val, {}).get(var, np.nan)
                )
            
            # Add PGS1 and PGS2 for both individuals
            for var in ['PGS1', 'PGS2']:
                pairs_with_measures[f'{var}_1'] = pairs_with_measures['Person_ID'].map(
                    lambda id_val: measures_lookup.get(id_val, {}).get(var, np.nan)
                )
                pairs_with_measures[f'{var}_2'] = pairs_with_measures['Relative_ID'].map(
                    lambda id_val: measures_lookup.get(id_val, {}).get(var, np.nan)
                )
            
            # Compute correlations for PGS1, PGS2, Y1, Y2
            correlation_vars = ['PGS1', 'PGS2', 'Y1', 'Y2']
            correlations = compute_correlations_for_multiple_variables(
                pairs_with_measures, correlation_vars, relationship_col='Relationship'
            )
            
            # Add iteration and relationship type info
            correlations['Iteration'] = iteration + 1
            correlations['RelationshipPath'] = rel_path
            
            all_correlations.append(correlations)
            print(f" ✓")
            
        except Exception as e:
            print(f"    {rel_path}: Error - {e}")
            continue
    
    # Combine all correlations
    if len(all_correlations) > 0:
        correlations_df = pd.concat(all_correlations, ignore_index=True)
        
        # Reorder columns
        cols = ['Iteration', 'RelationshipPath', 'Relationship',
                'Variable', 'N_Pairs', 'Correlation', 'P_Value']
        correlations_df = correlations_df[cols]
        
        return correlations_df
    
    return None

def extract_pgs_correlations(results):
    """
    Extract PGS correlations for both traits from simulation results.
    Returns correlations for final generation mates.
    PGS1 = TPO1 + TMO1, PGS2 = TPO2 + TMO2
    """
    final_gen = N_GENERATIONS - 1
    
    if 'HISTORY' not in results or final_gen >= len(results['HISTORY']['PHEN']):
        return None, None
    
    phen_df = results['HISTORY']['PHEN'][final_gen]
    
    if phen_df is None or phen_df.empty:
        return None, None
    
    # Get mates data
    if final_gen + 1 < len(results['HISTORY']['MATES']):
        mates_dict = results['HISTORY']['MATES'][final_gen + 1]
        males_df = mates_dict.get('males.PHENDATA')
        females_df = mates_dict.get('females.PHENDATA')
        
        if males_df is not None and females_df is not None:
            # Compute PGS from transmissible components
            males_df = males_df.copy()
            females_df = females_df.copy()
            males_df['PGS1'] = males_df['TPO1'] + males_df['TMO1']
            males_df['PGS2'] = males_df['TPO2'] + males_df['TMO2']
            females_df['PGS1'] = females_df['TPO1'] + females_df['TMO1']
            females_df['PGS2'] = females_df['TPO2'] + females_df['TMO2']
            
            # Merge to get spouse pairs
            merged = pd.merge(
                males_df[['ID', 'PGS1', 'PGS2', 'Spouse.ID']],
                females_df[['ID', 'PGS1', 'PGS2']],
                left_on='Spouse.ID',
                right_on='ID',
                suffixes=('_male', '_female')
            )
            
            if len(merged) > 1:
                # Calculate correlations for trait 1 and trait 2
                cor_pgs1 = np.corrcoef(merged['PGS1_male'], merged['PGS1_female'])[0, 1]
                cor_pgs2 = np.corrcoef(merged['PGS2_male'], merged['PGS2_female'])[0, 1]
                return cor_pgs1, cor_pgs2
    
    return None, None

def run_simulations(params, output_dir):
    """
    Run all iterations for the bivariate simulation.
    Trait 1 represents Condition 1 (f11=0.20), Trait 2 represents Condition 2 (f22=0.19).
    """
    print(f"\n{'#'*70}")
    print(f"# Starting bivariate simulations: {params['name']}")
    print(f"# Trait 1 (Condition 1): f11={params['f11']:.4f}, "
          f"prop_h2_latent1={params['prop_h2_latent1']:.4f}, vg1={params['vg1']:.4f}")
    print(f"# Trait 2 (Condition 2): f22={params['f22']:.4f}, "
          f"prop_h2_latent2={params['prop_h2_latent2']:.4f}, vg2={params['vg2']:.4f}")
    print(f"{'#'*70}\n")
    
    # Setup matrices
    matrices = setup_matrices(params)
    
    # Storage for PGS correlations and all correlations
    pgs_cor_trait1 = []
    pgs_cor_trait2 = []
    all_correlations = []
    
    # Run all iterations
    for iteration in range(N_ITERATIONS):
        try:
            # Run simulation
            results = run_single_iteration(iteration, params, matrices)
            
            # Save iteration data (final 3 generations) using save_simulation_results
            iter_dir = output_dir / f"Iteration_{iteration+1:02d}"
            save_simulation_results(
                results, 
                str(iter_dir), 
                file_prefix=f"iteration_{iteration+1:02d}",
                scope=FINAL_GENS
            )
            
            # Extract PGS correlations for mates (simple summary)
            cor1, cor2 = extract_pgs_correlations(results)
            if cor1 is not None:
                pgs_cor_trait1.append(cor1)
                pgs_cor_trait2.append(cor2)
                print(f"  → Mate PGS Correlations - Trait 1: {cor1:.4f}, Trait 2: {cor2:.4f}")
            
            # Extract and analyze all relationship types
            correlations_df = extract_and_analyze_relationships(results, iteration, output_dir)
            
            if correlations_df is not None:
                # Save iteration-specific correlations
                corr_file = iter_dir / f"correlations_iteration_{iteration+1:02d}.csv"
                correlations_df.to_csv(corr_file, index=False)
                print(f"  ✓ Saved correlations to {corr_file.name}")
                
                all_correlations.append(correlations_df)
            
        except Exception as e:
            print(f"  ✗ Error in Iteration {iteration+1}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save summary statistics
    # Save trait 1 PGS correlations (Condition 1: f11=0.20) - mate correlations only
    if pgs_cor_trait1:
        trait1_summary = pd.DataFrame({
            'iteration': range(1, len(pgs_cor_trait1) + 1),
            'mate_pgs_correlation_trait1': pgs_cor_trait1
        })
        trait1_file = output_dir / 'mate_pgs_correlation_trait1_summary.csv'
        trait1_summary.to_csv(trait1_file, index=False)
        print(f"\n  ✓ Saved Trait 1 (Condition 1, f11=0.20) mate correlation summary to {trait1_file.name}")
        print(f"    Mean: {np.mean(pgs_cor_trait1):.4f}, SD: {np.std(pgs_cor_trait1):.4f}")
    
    # Save trait 2 PGS correlations (Condition 2: f22=0.19) - mate correlations only
    if pgs_cor_trait2:
        trait2_summary = pd.DataFrame({
            'iteration': range(1, len(pgs_cor_trait2) + 1),
            'mate_pgs_correlation_trait2': pgs_cor_trait2
        })
        trait2_file = output_dir / 'mate_pgs_correlation_trait2_summary.csv'
        trait2_summary.to_csv(trait2_file, index=False)
        print(f"  ✓ Saved Trait 2 (Condition 2, f22=0.19) mate correlation summary to {trait2_file.name}")
        print(f"    Mean: {np.mean(pgs_cor_trait2):.4f}, SD: {np.std(pgs_cor_trait2):.4f}")
    
    # Combine and save all relationship correlations across iterations
    if all_correlations:
        print(f"\n  Combining relationship correlations across all iterations...")
        combined_correlations = pd.concat(all_correlations, ignore_index=True)
        
        # Save combined results
        combined_file = output_dir / "all_iterations_correlations.csv"
        combined_correlations.to_csv(combined_file, index=False)
        print(f"  ✓ Saved combined correlations to {combined_file.name}")
        
        # Create summary statistics by relationship type
        print(f"\n  Computing summary statistics by relationship type...")
        summary = combined_correlations.groupby(['RelationshipPath', 'Variable']).agg({
            'Correlation': ['mean', 'std', 'min', 'max'],
            'N_Pairs': 'sum',
            'Iteration': 'count'
        }).round(4)
        
        # Save summary
        summary_file = output_dir / "relationship_summary_statistics.csv"
        summary.to_csv(summary_file)
        print(f"  ✓ Saved relationship summary statistics to {summary_file.name}")
        
        # Print key summary for Trait 1 (PGS1) and Trait 2 (PGS2)
        print(f"\n  Summary - Mean correlations for PGS1 (Trait 1, Condition 1):")
        pgs1_summary = summary.xs('PGS1', level='Variable')['Correlation']['mean'].sort_values(ascending=False)
        for rel, val in pgs1_summary.head(10).items():
            print(f"    {rel}: {val:.4f}")
        
        print(f"\n  Summary - Mean correlations for PGS2 (Trait 2, Condition 2):")
        pgs2_summary = summary.xs('PGS2', level='Variable')['Correlation']['mean'].sort_values(ascending=False)
        for rel, val in pgs2_summary.head(10).items():
            print(f"    {rel}: {val:.4f}")

def main():
    """
    Main execution function.
    """
    print("\n" + "="*70)
    print("APPROXIMATION UNI SIMULATION SCRIPT")
    print("="*70)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Number of iterations: {N_ITERATIONS}")
    print(f"Population size: {POP_SIZE}")
    print(f"Number of generations: {N_GENERATIONS} (saving final 3)")
    print(f"Number of causal variants: {N_CV}")
    print("\nBivariate simulation with:")
    print(f"  - Trait 1 parameters from Condition 1 (f11=0.20)")
    print(f"  - Trait 2 parameters from Condition 2 (f22=0.19)")
    print("="*70 + "\n")
    
    # Run bivariate simulation
    run_simulations(SIMULATION_PARAMS, OUTPUT_DIR)
    
    print("\n" + "="*70)
    print("ALL SIMULATIONS COMPLETED")
    print("="*70)

if __name__ == "__main__":
    main()

# run run_approximation_bi.py to execute the simulations
