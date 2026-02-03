"""
Data Generation Script for Neural Network Training - COMBINED VERSION

This script runs forward-time simulations and immediately analyzes them without
saving raw simulation data. This increases efficiency and saves storage space.

Usage:
    python DataGeneratingNN_Combined.py
    (Run via SLURM array job - see submit_datagenerating_nn_combined.sh)
"""

import numpy as np
import pandas as pd
import sys
import os
import traceback
from pathlib import Path

# Add the SimulationFunctions directory to path
script_dir = Path(__file__).parent
simfunc_dir = script_dir.parent / "SimulationFunctions"
sys.path.insert(0, str(simfunc_dir))

from SimulationFunctions import AssortativeMatingSimulation
from find_relative_setbased import find_relationship_pairs
from extract_measures import extract_individual_measures, compute_correlations_for_multiple_variables

# ============================================================================
# CONFIGURATION
# ============================================================================

# Directory setup - using a new directory for this combined approach
PROJECT_BASE = Path("/projects/xuly4739/Py_Projects/PGS_Cor_Relative/Data/DataGeneratingNN_Large")

# Get SLURM array task ID
SLURM_TASK_ID = int(os.environ.get('SLURM_ARRAY_TASK_ID', '1'))

# Simulation parameters
ITERATIONS_PER_CONDITION = 40  # Number of iterations per condition
POP_SIZE = 40000
N_GENERATIONS = 15
FINAL_GENS = [12, 13, 14]  # Final 3 generations to analyze
N_CV = 1000
MAF_MIN = 0.01
MAF_MAX = 0.5

# Parameter ranges for data generation (5+ values each)
PARAM_RANGES = {
    'f11': [0.05, 0.10, 0.15, 0.20, 0.25, 0.35],
    'prop_h2_latent1': [0.5, 0.6, 0.7, 0.8, 0.9],
    'vg1': [0.4, 0.5, 0.6, 0.7, 0.8],
    'vg2': [0.375, 0.5, 0.625, 0.75, 0.875],
    'f22': [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
    'f12': [0.02, 0.05, 0.10, 0.15, 0.20, 0.25], 
    'f21': [0.02, 0.05, 0.10, 0.15, 0.20, 0.25], 
    're': [0.0, 0.1, 0.2, 0.3, 0.4],
    'am22': [0.35, 0.45, 0.525, 0.60, 0.675, 0.75],
    'rg': [0.40, 0.525, 0.60, 0.675, 0.75, 0.825, 0.90]
}

# Fixed parameters
FIXED_PARAMS = {
    'am11': 0,
    'am12': 0,
    'am21': 0,
    #'f12': 0,
    #'f21': 0,
    #'re': 0,
    'prop_h2_latent2': 1.0  # 0.8/0.8 = 1.0
}

# Relationship types to analyze
RELATIONSHIP_TYPES = [
    'S',        # Siblings
    'HSFS',     # Half-siblings
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
    'MSMSM',    # Mate's sibling's mate's sibling's mate
    'MSMSC',    # Mate's sibling's mate's sibling's child
    'PSMSC',    # Parent's sibling's mate's sibling's child
    'SMSMSC',   # Sibling's mate's sibling's mate's sibling's child
    'MSMSMS',   # Mate's sibling's mate's sibling's mate's sibling
]

# ============================================================================
# PARAMETER GENERATION
# ============================================================================

def generate_conditions(n_conditions=500, seed=42):
    """Generate parameter combinations."""
    np.random.seed(seed)
    
    conditions = []
    param_names = list(PARAM_RANGES.keys())
    
    for i in range(n_conditions):
        condition = {'name': f'Condition_{i+1:04d}'}
        
        for param_name in param_names:
            param_values = PARAM_RANGES[param_name]
            condition[param_name] = np.random.choice(param_values)
        
        condition.update(FIXED_PARAMS)
        conditions.append(condition)
    
    return conditions

CONDITIONS_FILE = PROJECT_BASE / "conditions_config.csv"

def save_conditions_config():
    """Save all conditions to a CSV file for reference and reproducibility."""
    PROJECT_BASE.mkdir(parents=True, exist_ok=True)
    
    n_conditions_needed = 500
    needs_regeneration = False
    
    if CONDITIONS_FILE.exists():
        # Check if existing file has the right number of conditions
        existing_df = pd.read_csv(CONDITIONS_FILE)
        if len(existing_df) != n_conditions_needed:
            print(f"⚠ Existing config has {len(existing_df)} conditions, need {n_conditions_needed}")
            needs_regeneration = True
        else:
            print(f"✓ Using existing conditions from {CONDITIONS_FILE} ({len(existing_df)} conditions)")
    else:
        needs_regeneration = True
    
    if needs_regeneration:
        print(f"Generating {n_conditions_needed} parameter conditions...")
        conditions = generate_conditions(n_conditions=n_conditions_needed, seed=42)
        df = pd.DataFrame(conditions)
        df.to_csv(CONDITIONS_FILE, index=False)
        print(f"✓ Saved {len(conditions)} conditions to {CONDITIONS_FILE}")

def load_condition(task_id):
    """Load the condition for a specific task ID."""
    df = pd.read_csv(CONDITIONS_FILE)
    if task_id > len(df):
        raise ValueError(f"Task ID {task_id} exceeds number of conditions ({len(df)})")
    return df.iloc[task_id - 1].to_dict()

# ============================================================================
# SIMULATION FUNCTIONS
# ============================================================================

def setup_matrices(params):
    """Setup covariance and other matrices based on simulation parameters."""
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
    
    # Get mate correlation for trait 2 (single-trait mating mode)
    am22 = params['am22']
    
    # Vertical transmission matrix
    f11 = params['f11']
    f12 = params['f12']
    f21 = params['f21']
    f22 = params['f22']
    f_mat = np.array([[f11, f12], [f21, f22]])
    
    # Social homogamy matrix (set to zero - phenotypic AM only)
    s_mat = np.zeros((2, 2))
    
    # AM list: list of scalar values for single-trait mating on trait 2
    am_list = [am22 for _ in range(N_GENERATIONS)]
    
    return {
        'cove_mat': cove_mat,
        'f_mat': f_mat,
        's_mat': s_mat,
        'a_mat': a_mat,
        'd_mat': delta_mat,
        'am_list': am_list,
        'mate_on_trait': 2,
        'covy_mat': covy_mat,
        'k2_matrix': k2_matrix
    }

# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def compute_pgs_from_components(measures_df):
    """Compute full PGS from transmissible components."""
    df = measures_df.copy()
    df['PGS1'] = df['TPO1'] + df['TMO1']
    df['PGS2'] = df['TPO2'] + df['TMO2']
    return df

def extract_and_analyze_relationships(results, iteration):
    """
    Extract measures and analyze correlations for different relationship types.
    Returns DataFrame with correlation results for all relationship types.
    """
    # Extract measures for all individuals
    # We'll extract component measures and compute PGS from them
    component_vars = ['Y1', 'Y2', 'TPO1', 'TPO2', 'TMO1', 'TMO2']
    
    try:
        individual_measures = extract_individual_measures(results, component_vars)
        individual_measures = compute_pgs_from_components(individual_measures)
    except Exception as e:
        print(f"    ✗ Error extracting measures: {e}")
        traceback.print_exc()
        return None
    
    # Create lookup for individual measures
    measures_lookup = individual_measures.set_index('ID').to_dict('index')
    
    # Find all relationship pairs and compute correlations
    all_correlations = []
    
    for rel_path in RELATIONSHIP_TYPES:
        try:
            # Note: trimmed_results only contains the final 3 generations
            # They are now indexed as 0, 1, 2 (corresponding to original gens 12, 13, 14)
            gen_indices = [0, 1, 2]
            
            pairs = find_relationship_pairs(
                results, rel_path,
                output_format='long',
                generations=gen_indices
            )
            
            if len(pairs) == 0:
                continue
            
            # Rename columns to expected format (Person_ID -> ID1, Relative_ID -> ID2)
            pairs = pairs.rename(columns={'Person_ID': 'ID1', 'Relative_ID': 'ID2'})
            
            # Add measures for pairs
            pairs_with_measures = pairs.copy()
            
            # Note: compute_correlations_for_multiple_variables expects columns named {var}_1 and {var}_2
            # We need to add all variables we want to compute correlations for
            all_vars_to_add = ['PGS1', 'PGS2', 'Y1', 'Y2']
            
            for var in all_vars_to_add:
                pairs_with_measures[f'{var}_1'] = pairs['ID1'].map(
                    lambda x: measures_lookup.get(x, {}).get(var, np.nan)
                )
                pairs_with_measures[f'{var}_2'] = pairs['ID2'].map(
                    lambda x: measures_lookup.get(x, {}).get(var, np.nan)
                )
            
            # Compute correlations
            correlation_vars = ['PGS1', 'PGS2', 'Y1', 'Y2']
            correlations = compute_correlations_for_multiple_variables(
                pairs_with_measures, correlation_vars, relationship_col='Relationship'
            )
            
            correlations['Iteration'] = iteration + 1
            correlations['RelationshipPath'] = rel_path
            
            all_correlations.append(correlations)
            
        except Exception as e:
            print(f"    {rel_path}: Error - {e}")
            continue
    
    # Combine all correlations
    if len(all_correlations) > 0:
        correlations_df = pd.concat(all_correlations, ignore_index=True)
        cols = ['Iteration', 'RelationshipPath', 'Relationship',
                'Variable', 'N_Pairs', 'Correlation', 'P_Value']
        correlations_df = correlations_df[cols]
        return correlations_df
    
    return None

def run_single_iteration(iteration, condition_name, params, matrices):
    """Run a single simulation iteration and analyze it immediately."""
    print(f"\n  Iteration {iteration + 1}/{ITERATIONS_PER_CONDITION}")
    
    # Set seed for reproducibility
    condition_hash = hash(condition_name) % 100000
    seed = condition_hash * 100 + iteration + 1
    
    # Extract mate_on_trait for single-trait mating mode
    mate_on_trait = matrices.pop('mate_on_trait', None)
    
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
        output_summary_filename=None,  # Don't save summary
        mate_on_trait=mate_on_trait,
        **matrices
    )
    
    # Run simulation
    print(f"    Running simulation...")
    results = sim.run_simulation()
    
    # Trim results to only keep final generations to save memory
    # Note: MATES has offset indexing - MATES[i+1] contains mates FROM generation i
    # For FINAL_GENS = [12, 13, 14], we need MATES[12], [13], [14], and [15] (for mating in gen 14)
    mates_indices = FINAL_GENS + [FINAL_GENS[-1] + 1] if FINAL_GENS[-1] + 1 < len(results['HISTORY']['MATES']) else FINAL_GENS
    
    trimmed_results = {
        'HISTORY': {
            'PHEN': [results['HISTORY']['PHEN'][i] for i in FINAL_GENS],
            'XO': [results['HISTORY']['XO'][i] for i in FINAL_GENS],
            'XL': [results['HISTORY']['XL'][i] for i in FINAL_GENS],
            'MATES': [results['HISTORY']['MATES'][i] if i < len(results['HISTORY']['MATES']) else None for i in mates_indices]
        }
    }
    
    # Analyze relationships immediately
    print(f"    Analyzing relationships...")
    correlations_df = extract_and_analyze_relationships(trimmed_results, iteration)
    
    if correlations_df is not None:
        print(f"    ✓ Computed {len(correlations_df)} correlations")
        return correlations_df
    else:
        print(f"    ✗ No correlations computed")
        return None

# ============================================================================
# MAIN CONDITION PROCESSING
# ============================================================================

def run_condition(condition, project_base):
    """Run all iterations for a single condition."""
    condition_name = condition['name']
    
    print(f"\n{'#'*70}")
    print(f"# {condition_name}")
    print(f"# Parameters:")
    print(f"#   f11={condition['f11']:.4f}, vg1={condition['vg1']:.4f}, prop_h2_latent1={condition['prop_h2_latent1']:.4f}")
    print(f"#   f22={condition['f22']:.4f}, vg2={condition['vg2']:.4f}, am22={condition['am22']:.4f}")
    print(f"#   rg={condition['rg']:.4f}")
    print(f"# Running {ITERATIONS_PER_CONDITION} iterations")
    print(f"# SLURM Task ID: {SLURM_TASK_ID}")
    print(f"{'#'*70}\n")
    
    # Create condition-specific directory
    project_dir = project_base / condition_name
    project_dir.mkdir(parents=True, exist_ok=True)
    
    # Save condition parameters immediately
    params_file = project_dir / "condition_parameters.csv"
    params_df = pd.DataFrame([condition])
    params_df.to_csv(params_file, index=False)
    print(f"  ✓ Saved parameters to {params_file}")
    
    # Setup matrices
    matrices = setup_matrices(condition)
    
    # Track iteration status and results
    iteration_status = []
    all_correlations = []
    
    # Run iterations
    for iteration in range(ITERATIONS_PER_CONDITION):
        iter_info = {
            'Iteration': iteration + 1,
            'Status': 'Failed',
            'Error': None,
            'N_Correlations': 0
        }
        
        try:
            correlations_df = run_single_iteration(iteration, condition_name, condition, matrices.copy())
            
            if correlations_df is not None:
                iter_info['Status'] = 'Success'
                iter_info['N_Correlations'] = len(correlations_df)
                all_correlations.append(correlations_df)
        except Exception as e:
            iter_info['Error'] = str(e)
            print(f"    ✗ Error in Iteration {iteration+1}: {e}")
            traceback.print_exc()
        
        iteration_status.append(iter_info)
    
    # Save iteration status summary
    status_file = project_dir / "iteration_status.csv"
    status_df = pd.DataFrame(iteration_status)
    status_df.to_csv(status_file, index=False)
    print(f"\n  ✓ Saved iteration status to {status_file}")
    
    # Save results for this condition
    if all_correlations:
        print(f"\n  Saving correlation results for {condition_name}...")
        combined_correlations = pd.concat(all_correlations, ignore_index=True)
        
        # Add parameter values to each row for NN training
        for param in ['f11', 'prop_h2_latent1', 'vg1', 'vg2', 'f22', 'am22', 'rg']:
            combined_correlations[f'param_{param}'] = condition[param]
        
        # Save detailed correlations with all iterations
        output_file = project_dir / "all_iterations_correlations.csv"
        combined_correlations.to_csv(output_file, index=False)
        print(f"  ✓ Saved {len(combined_correlations)} correlation records to {output_file}")
        
        # Create summary statistics by relationship and variable
        summary = combined_correlations.groupby(['RelationshipPath', 'Variable']).agg({
            'Correlation': ['mean', 'std', 'min', 'max'],
            'N_Pairs': 'sum',
            'Iteration': 'count'
        }).round(4)
        
        summary_file = project_dir / "summary_statistics.csv"
        summary.to_csv(summary_file)
        print(f"  ✓ Saved summary statistics to {summary_file}")
        
        # Create a wide-format summary for NN training (one row per iteration)
        print(f"  Creating NN training format...")
        nn_training_data = []
        
        for iteration in sorted(combined_correlations['Iteration'].unique()):
            iter_data = combined_correlations[combined_correlations['Iteration'] == iteration]
            if len(iter_data) == 0:
                continue
            
            row = {
                'Iteration': int(iteration),
                'Condition': condition_name,
            }
            
            # Add parameters
            for param in ['f11', 'prop_h2_latent1', 'vg1', 'vg2', 'f22', 'am22', 'rg']:
                row[f'param_{param}'] = condition[param]
            
            # Add correlations for each relationship-variable combination
            for _, corr_row in iter_data.iterrows():
                col_name = f"{corr_row['RelationshipPath']}_{corr_row['Variable']}"
                row[col_name] = corr_row['Correlation']
                row[f"{col_name}_N"] = corr_row['N_Pairs']
            
            nn_training_data.append(row)
        
        if nn_training_data:
            nn_training_file = project_dir / "nn_training_format.csv"
            nn_training_df = pd.DataFrame(nn_training_data)
            nn_training_df.to_csv(nn_training_file, index=False)
            print(f"  ✓ Saved NN training format to {nn_training_file}")
        
        n_success = sum(1 for s in iteration_status if s['Status'] == 'Success')
        print(f"\n  Summary: {n_success}/{ITERATIONS_PER_CONDITION} iterations successful")
        return True
    else:
        print(f"\n  ⚠ No correlations computed for {condition_name}")
        return False

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print("\n" + "="*70)
    print("DATA GENERATION FOR NEURAL NETWORK TRAINING - COMBINED VERSION")
    print("="*70)
    print(f"Project base: {PROJECT_BASE}")
    print(f"SLURM Task ID: {SLURM_TASK_ID}")
    print(f"Population size: {POP_SIZE}")
    print(f"Generations: {N_GENERATIONS} (analyzing final 3)")
    print(f"Iterations per condition: {ITERATIONS_PER_CONDITION}")
    print(f"Causal variants: {N_CV}")
    print("="*70 + "\n")
    
    # Save/load conditions configuration
    save_conditions_config()
    
    # Load condition for this task
    try:
        condition = load_condition(SLURM_TASK_ID)
        print(f"Loaded condition for Task {SLURM_TASK_ID}")
    except Exception as e:
        print(f"Error loading condition: {e}")
        return
    
    # Run the condition
    success = run_condition(condition, PROJECT_BASE)
    
    if success:
        print(f"\n{'='*70}")
        print(f"TASK {SLURM_TASK_ID} COMPLETED SUCCESSFULLY")
        print(f"{'='*70}\n")
    else:
        print(f"\n{'='*70}")
        print(f"TASK {SLURM_TASK_ID} COMPLETED WITH ERRORS")
        print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
