"""
Simulation script for bivariate approximation conditions (SLURM batch version).
Runs 200 iterations with N=40000 for four optimal parameter conditions.
Each condition is a full bivariate model with cross-trait effects.
Saves data for final 3 generations and PGS correlations summary statistics.
Each SLURM task runs 20 iterations.
"""

import numpy as np
import pandas as pd
import os
import sys
from pathlib import Path

# Add the SimulationFunctions directory to path
script_dir = Path(__file__).parent
simfunc_dir = script_dir.parent / "SimulationFunctions"
sys.path.insert(0, str(simfunc_dir))

from SimulationFunctions import AssortativeMatingSimulation
from save_simulation_data import save_simulation_results
from find_relative_setbased import find_relationship_pairs
from extract_measures import extract_individual_measures, compute_correlations_for_multiple_variables

# Define output directories
# SCRATCH_DIR for raw iteration data (large files)
# PROJECT_DIR for summary statistics (small files)
SCRATCH_BASE = Path("/scratch/alpine/xuly4739/PGS_Cor_Relative/Data/approximation_bi")
PROJECT_BASE = Path("/projects/xuly4739/Py_Projects/PGS_Cor_Relative/Data/approximation_bi")

# Get SLURM array task ID for batch processing
SLURM_TASK_ID = int(os.environ.get('SLURM_ARRAY_TASK_ID', '1'))

# Calculate which condition and iterations this task should run
# 4 conditions * 200 iterations / 20 per task = 40 tasks total
# Tasks 1-10: Condition_01, Tasks 11-20: Condition_02, etc.
ITERATIONS_PER_TASK = int(os.environ.get('ITERATIONS_PER_TASK', '20'))
TOTAL_ITERATIONS_PER_CONDITION = 200
TASKS_PER_CONDITION = TOTAL_ITERATIONS_PER_CONDITION // ITERATIONS_PER_TASK

CONDITION_INDEX = (SLURM_TASK_ID - 1) // TASKS_PER_CONDITION
TASK_WITHIN_CONDITION = (SLURM_TASK_ID - 1) % TASKS_PER_CONDITION
START_ITERATION = TASK_WITHIN_CONDITION * ITERATIONS_PER_TASK
END_ITERATION = START_ITERATION + ITERATIONS_PER_TASK

# Define the four bivariate conditions
CONDITIONS = [
    {
        'name': 'Condition_01',
        'f11': 0.1000,
        'prop_h2_latent1': 0.8000,
        'vg1': 0.6000,
        'vg2': 1.0000,
        'f22': 0.2000,
        'am22': 0.6500,
        'rg': 0.7500,
        # Fixed cross-trait parameters
        'am11': 0,
        'am12': 0,
        'am21': 0,
        'f12': 0,
        'f21': 0,
        're': 0,
        'prop_h2_latent2': 0.8/0.8
    },
    {
        'name': 'Condition_02',
        'f11': 0.2000,
        'prop_h2_latent1': 0.8000,
        'vg1': 0.8000,
        'vg2': 0.5000,
        'f22': 0.3000,
        'am22': 0.4500,
        'rg': 0.8500,
        # Fixed cross-trait parameters
        'am11': 0,
        'am12': 0,
        'am21': 0,
        'f12': 0,
        'f21': 0,
        're': 0,
        'prop_h2_latent2': 0.8/0.8
    },
    {
        'name': 'Condition_03',
        'f11': 0.1500,
        'prop_h2_latent1': 0.8000,
        'vg1': 0.6500,
        'vg2': 0.5000,
        'f22': 0.2500,
        'am22': 0.5500,
        'rg': 0.8500,
        # Fixed cross-trait parameters
        'am11': 0,
        'am12': 0,
        'am21': 0,
        'f12': 0,
        'f21': 0,
        're': 0,
        'prop_h2_latent2': 0.8/0.8
    },
    {
        'name': 'Condition_04',
        'f11': 0.1000,
        'prop_h2_latent1': 0.8000,
        'vg1': 0.6000,
        'vg2': 0.7500,
        'f22': 0.1500,
        'am22': 0.7500,
        'rg': 0.7500,
        # Fixed cross-trait parameters
        'am11': 0,
        'am12': 0,
        'am21': 0,
        'f12': 0,
        'f21': 0,
        're': 0,
        'prop_h2_latent2': 0.8/0.8
    }
]

# Simulation parameters
TOTAL_ITERATIONS = 200  # Total iterations per condition
POP_SIZE = 40000  # Increased from 20000
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

def run_single_iteration(iteration, condition_name, params, matrices):
    """
    Run a single simulation iteration.
    """
    print(f"\n{'='*60}")
    print(f"Running {condition_name} - Iteration {iteration + 1}/{TOTAL_ITERATIONS}")
    print(f"{'='*60}")
    
    # Set seed for reproducibility (unique per condition and iteration)
    # Use hash of condition name to ensure different seeds per condition
    condition_hash = hash(condition_name) % 10000
    seed = condition_hash * 1000 + iteration + 1
    
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

def run_condition(condition, scratch_base, project_base, start_iter, end_iter):
    """
    Run a batch of iterations for a single condition.
    """
    condition_name = condition['name']
    print(f"\n{'#'*70}")
    print(f"# Starting simulations: {condition_name}")
    print(f"# Bivariate model with cross-trait effects")
    print(f"# f11={condition['f11']:.4f}, vg1={condition['vg1']:.4f}, prop_h2_latent1={condition['prop_h2_latent1']:.4f}")
    print(f"# f22={condition['f22']:.4f}, vg2={condition['vg2']:.4f}, am22={condition['am22']:.4f}, rg={condition['rg']:.4f}")
    print(f"# Running iterations {start_iter + 1} to {end_iter}")
    print(f"# SLURM Task ID: {SLURM_TASK_ID}")
    print(f"{'#'*70}\n")
    
    # Create condition-specific directories
    scratch_dir = scratch_base / condition_name
    project_dir = project_base / condition_name
    scratch_dir.mkdir(parents=True, exist_ok=True)
    project_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup matrices
    matrices = setup_matrices(condition)
    
    # Storage for PGS correlations (Trait 1 only as requested)
    pgs_cor_trait1 = []
    all_correlations = []
    
    # Run iterations for this task
    for iteration in range(start_iter, end_iter):
        try:
            # Run simulation
            results = run_single_iteration(iteration, condition_name, condition, matrices)
            
            # Save iteration data to SCRATCH (final 3 generations)
            iter_dir = scratch_dir / f"Iteration_{iteration+1:03d}"
            save_simulation_results(
                results, 
                str(iter_dir), 
                file_prefix=f"iteration_{iteration+1:03d}",
                scope=FINAL_GENS
            )
            
            # Extract PGS correlations for mates (Trait 1 only)
            cor1, cor2 = extract_pgs_correlations(results)
            if cor1 is not None:
                pgs_cor_trait1.append(cor1)
                print(f"  → Mate PGS Correlation - Trait 1: {cor1:.4f}")
            
            # Extract and analyze all relationship types
            correlations_df = extract_and_analyze_relationships(results, iteration, scratch_dir)
            
            if correlations_df is not None:
                # Save iteration-specific correlations to SCRATCH
                corr_file = iter_dir / f"correlations_iteration_{iteration+1:03d}.csv"
                correlations_df.to_csv(corr_file, index=False)
                print(f"  ✓ Saved correlations to {corr_file.name}")
                
                all_correlations.append(correlations_df)
            
        except Exception as e:
            print(f"  ✗ Error in Iteration {iteration+1}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save summary statistics for this batch to PROJECT directory
    # Save trait 1 PGS correlations (Trait 1 only as requested) - mate correlations only
    if pgs_cor_trait1:
        trait1_summary = pd.DataFrame({
            'iteration': range(start_iter + 1, start_iter + len(pgs_cor_trait1) + 1),
            'mate_pgs_correlation_trait1': pgs_cor_trait1
        })
        trait1_file = project_dir / f'mate_pgs_correlation_trait1_batch_{SLURM_TASK_ID:03d}.csv'
        trait1_summary.to_csv(trait1_file, index=False)
        print(f"\n  ✓ Saved Trait 1 batch summary to {trait1_file.name}")
        print(f"    Mean: {np.mean(pgs_cor_trait1):.4f}, SD: {np.std(pgs_cor_trait1):.4f}")
    
    # Combine and save relationship correlations for this batch
    if all_correlations:
        print(f"\n  Combining relationship correlations for this batch...")
        combined_correlations = pd.concat(all_correlations, ignore_index=True)
        
        # Save combined results for this batch to PROJECT directory
        combined_file = project_dir / f"correlations_batch_{SLURM_TASK_ID:03d}.csv"
        combined_correlations.to_csv(combined_file, index=False)
        print(f"  ✓ Saved batch correlations to {combined_file.name}")
        
        # Create summary statistics by relationship type for this batch
        print(f"\n  Computing summary statistics for this batch...")
        summary = combined_correlations.groupby(['RelationshipPath', 'Variable']).agg({
            'Correlation': ['mean', 'std', 'min', 'max'],
            'N_Pairs': 'sum',
            'Iteration': 'count'
        }).round(4)
        
        # Save batch summary to PROJECT directory
        summary_file = project_dir / f"relationship_summary_batch_{SLURM_TASK_ID:03d}.csv"
        summary.to_csv(summary_file)
        print(f"  ✓ Saved batch relationship summary to {summary_file.name}")
    # Save summary statistics (Trait 1 only as requested)
    if pgs_cor_trait1:
        trait1_summary = pd.DataFrame({
            'iteration': range(1, len(pgs_cor_trait1) + 1),
            'mate_pgs_correlation_trait1': pgs_cor_trait1
        })
        trait1_file = condition_dir / 'mate_pgs_correlation_trait1_summary.csv'
        trait1_summary.to_csv(trait1_file, index=False)
        print(f"\n  ✓ Saved Trait 1 mate correlation summary to {trait1_file.name}")
        print(f"    Mean: {np.mean(pgs_cor_trait1):.4f}, SD: {np.std(pgs_cor_trait1):.4f}")
    
    # Combine and save all relationship correlations across iterations
    if all_correlations:
        print(f"\n  Combining relationship correlations across all iterations...")
        combined_correlations = pd.concat(all_correlations, ignore_index=True)
        
        # Save combined results
        combined_file = condition_dir / "all_iterations_correlations.csv"
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
        summary_file = condition_dir / "relationship_summary_statistics.csv"
        summary.to_csv(summary_file)
        print(f"  ✓ Saved relationship summary statistics to {summary_file.name}")


def main():
    """
    Main execution function for SLURM batch processing.
    """
    # Get the condition for this task
    current_condition = CONDITIONS[CONDITION_INDEX]
    
    print("\n" + "="*70)
    print("APPROXIMATION BI SIMULATION SCRIPT (SLURM BATCH)")
    print("="*70)
    print(f"Scratch base directory: {SCRATCH_BASE}")
    print(f"Project base directory: {PROJECT_BASE}")
    print(f"SLURM Task ID: {SLURM_TASK_ID}")
    print(f"Current condition: {current_condition['name']} (index {CONDITION_INDEX + 1}/{len(CONDITIONS)})")
    print(f"Total iterations per condition: {TOTAL_ITERATIONS_PER_CONDITION}")
    print(f"Iterations per task: {ITERATIONS_PER_TASK}")
    print(f"This task runs: {START_ITERATION + 1} to {END_ITERATION}")
    print(f"Population size: {POP_SIZE}")
    print(f"Number of generations: {N_GENERATIONS} (saving final 3)")
    print(f"Number of causal variants: {N_CV}")
    print(f"\nCondition parameters:")
    print(f"  f11={current_condition['f11']:.4f}, vg1={current_condition['vg1']:.4f}")
    print(f"  f22={current_condition['f22']:.4f}, vg2={current_condition['vg2']:.4f}")
    print(f"  rg={current_condition['rg']:.4f}, am22={current_condition['am22']:.4f}")
    print("="*70 + "\n")
    
    # Run the batch for this condition
    run_condition(current_condition, SCRATCH_BASE, PROJECT_BASE, START_ITERATION, END_ITERATION)
    
    print("\n" + "="*70)
    print(f"BATCH {SLURM_TASK_ID} COMPLETED - {current_condition['name']}")
    print("="*70)

if __name__ == "__main__":
    main()

