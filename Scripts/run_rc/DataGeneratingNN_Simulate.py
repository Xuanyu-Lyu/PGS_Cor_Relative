"""
Data Generation Script for Neural Network Training - PART 1: SIMULATION

This script runs forward-time simulations across a wide range of parameter combinations
and saves the raw simulation data. The analysis of correlations is done separately.

Usage:
    python DataGeneratingNN_Simulate.py
    (Run via SLURM array job - see submit_datagenerating_nn_simulate.sh)
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
from save_simulation_data import save_simulation_results

# ============================================================================
# CONFIGURATION
# ============================================================================

# Directory setup
SCRATCH_BASE = Path("/scratch/alpine/xuly4739/PGS_Cor_Relative/Data/DataGeneratingNN")
PROJECT_BASE = Path("/projects/xuly4739/Py_Projects/PGS_Cor_Relative/Data/DataGeneratingNN")

# Get SLURM array task ID
SLURM_TASK_ID = int(os.environ.get('SLURM_ARRAY_TASK_ID', '1'))

# Simulation parameters
ITERATIONS_PER_CONDITION = 10  # Number of iterations per condition
POP_SIZE = 40000
N_GENERATIONS = 15
FINAL_GENS = [12, 13, 14]  # Final 3 generations to analyze
N_CV = 1000
MAF_MIN = 0.01
MAF_MAX = 0.5

# Parameter ranges for data generation (5+ values each)
PARAM_RANGES = {
    'f11': [0.05, 0.10, 0.15, 0.20, 0.25],
    'prop_h2_latent1': [0.6, 0.7, 0.8, 0.9, 1.0],
    'vg1': [0.4, 0.5, 0.6, 0.7, 0.8],
    'vg2': [0.5, 0.625, 0.75, 0.875, 1.0],
    'f22': [0.10, 0.15, 0.20, 0.25, 0.30],
    'am22': [0.45, 0.525, 0.60, 0.675, 0.75],
    'rg': [0.60, 0.675, 0.75, 0.825, 0.90]
}

# Fixed parameters
FIXED_PARAMS = {
    'am11': 0,
    'am12': 0,
    'am21': 0,
    'f12': 0,
    'f21': 0,
    're': 0,
    'prop_h2_latent2': 1.0  # 0.8/0.8 = 1.0
}

# ============================================================================
# PARAMETER GENERATION
# ============================================================================

def generate_conditions(n_conditions=200, seed=42):
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
    
    if not CONDITIONS_FILE.exists():
        print("Generating parameter conditions...")
        conditions = generate_conditions(n_conditions=200, seed=42)
        df = pd.DataFrame(conditions)
        df.to_csv(CONDITIONS_FILE, index=False)
        print(f"✓ Saved {len(conditions)} conditions to {CONDITIONS_FILE}")
    else:
        print(f"✓ Using existing conditions from {CONDITIONS_FILE}")

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

def run_single_iteration(iteration, condition_name, params, matrices, scratch_dir):
    """Run a single simulation iteration."""
    print(f"\n  Running {condition_name} - Iteration {iteration + 1}/{ITERATIONS_PER_CONDITION}")
    
    # Set seed for reproducibility
    condition_hash = hash(condition_name) % 100000
    seed = condition_hash * 100 + iteration + 1
    
    # Create iteration directory
    iter_dir = scratch_dir / f"Iteration_{iteration+1:02d}"
    iter_dir.mkdir(parents=True, exist_ok=True)
    summary_filename = str(iter_dir / f"iteration_{iteration+1:02d}_summary.txt")
    
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
        output_summary_filename=summary_filename,
        mate_on_trait=mate_on_trait,
        **matrices
    )
    
    # Run simulation
    results = sim.run_simulation()
    
    # Save raw simulation data for final generations
    save_simulation_results(
        results, 
        str(iter_dir), 
        file_prefix=f"iteration_{iteration+1:02d}",
        scope=FINAL_GENS
    )
    
    print(f"    ✓ Iteration {iteration+1} simulated and saved")
    return True

def run_condition(condition, scratch_base, project_base):
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
    
    # Create condition-specific directories
    scratch_dir = scratch_base / condition_name
    project_dir = project_base / condition_name
    scratch_dir.mkdir(parents=True, exist_ok=True)
    project_dir.mkdir(parents=True, exist_ok=True)
    
    # Save condition parameters immediately
    params_file = project_dir / "condition_parameters.csv"
    params_df = pd.DataFrame([condition])
    params_df.to_csv(params_file, index=False)
    print(f"  ✓ Saved parameters to {params_file}")
    
    # Setup matrices
    matrices = setup_matrices(condition)
    
    # Track iteration status
    iteration_status = []
    
    # Run iterations
    for iteration in range(ITERATIONS_PER_CONDITION):
        iter_info = {
            'Iteration': iteration + 1,
            'Status': 'Failed',
            'Error': None
        }
        
        try:
            success = run_single_iteration(iteration, condition_name, condition, matrices, scratch_dir)
            if success:
                iter_info['Status'] = 'Success'
        except Exception as e:
            iter_info['Error'] = str(e)
            print(f"    ✗ Error in Iteration {iteration+1}: {e}")
            traceback.print_exc()
        
        iteration_status.append(iter_info)
    
    # Save iteration status summary
    status_file = project_dir / "simulation_status.csv"
    status_df = pd.DataFrame(iteration_status)
    status_df.to_csv(status_file, index=False)
    print(f"\n  ✓ Saved simulation status to {status_file}")
    
    n_success = sum(1 for s in iteration_status if s['Status'] == 'Success')
    return n_success > 0

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print("\n" + "="*70)
    print("DATA GENERATION FOR NEURAL NETWORK TRAINING - PART 1: SIMULATION")
    print("="*70)
    print(f"Scratch base: {SCRATCH_BASE}")
    print(f"Project base: {PROJECT_BASE}")
    print(f"SLURM Task ID: {SLURM_TASK_ID}")
    print(f"Population size: {POP_SIZE}")
    print(f"Generations: {N_GENERATIONS} (saving final 3)")
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
    success = run_condition(condition, SCRATCH_BASE, PROJECT_BASE)
    
    if success:
        print(f"\n{'='*70}")
        print(f"SIMULATION FOR TASK {SLURM_TASK_ID} COMPLETED SUCCESSFULLY")
        print(f"Run DataGeneratingNN_Analyze.py to process the results")
        print(f"{'='*70}\n")
    else:
        print(f"\n{'='*70}")
        print(f"SIMULATION FOR TASK {SLURM_TASK_ID} COMPLETED WITH ERRORS")
        print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
