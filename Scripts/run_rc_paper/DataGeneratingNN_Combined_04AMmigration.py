"""
Data Generation Script for Neural Network Training - COMBINED VERSION

This script runs forward-time simulations and immediately analyzes them without
saving raw simulation data. This increases efficiency and saves storage space.

This simulates one condition:
4) AM-Migration model:
   - Trait 1: EA (Educational Attainment) — the assortative mating trait.
   - Trait 2: Migration — correlated with EA; all genetic effects are latent
     (prop_h2_latent2 = 1, so no observable PGS signal for migration).
   - Genetic (rg) and environmental (re) correlations between traits.
   - Within-trait vertical transmission only (f11, f22 non-zero; f12 = f21 = 0).
   - No shared environmental effects (s_mat = 0 for all traits).
   - Assortative mating on EA (trait 1) only within each island (am11 varies).
   - 5 islands with migration sorted by trait 2 (migration trait).
   - Migration proportion (move_p) varies as a sampled parameter.
   - 40 generations to allow equilibration.

Usage:
    python DataGeneratingNN_Combined_04AMmigration.py
    (Run via SLURM array job - see submit_datagenerating_nn_combined_04AMmigration.sh)
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

from IslandSimulation import IslandMigrationSimulation
from find_relative_setbased import find_relationship_pairs
from extract_measures import extract_individual_measures, compute_correlations_for_multiple_variables

# ============================================================================
# CONFIGURATION
# ============================================================================

# Directory setup
PROJECT_BASE = Path("/projects/xuly4739/Py_Projects/PGS_Cor_Relative/Data/DataGeneratingNN_Paper/04AMmigration")

# Get SLURM array task ID
SLURM_TASK_ID = int(os.environ.get('SLURM_ARRAY_TASK_ID', '1'))

# Simulation parameters
ITERATIONS_PER_CONDITION = 1   # Each condition is unique; one simulation per condition
CONDITIONS_PER_JOB = 40        # Conditions (simulations) processed per SLURM job
N_CONDITIONS_TOTAL = 20000     # Total unique conditions (500 jobs × 40 conditions)
POP_SIZE = 40000               # Must be divisible by N_ISLANDS * 2
N_ISLANDS = 5                  # Number of islands (40000 / (5*2) = 4000 per sex per island)
N_GENERATIONS = 40
FINAL_GENS = [37, 38, 39]      # Final 3 generations to analyze (0-indexed)
N_CV = 1000
MAF_MIN = 0.01
MAF_MAX = 0.5

# Parameter bounds for uniform sampling: [min, max]
# Trait 1 = EA (mating trait), Trait 2 = Migration (latent genetic, no PGS)
PARAM_BOUNDS = {
    'prop_h2_latent1': [0.5,  0.9],   # EA: proportion of h2 that is latent (no PGS)
    'vg1':             [0.4,  0.8],   # EA total genetic variance
    'vg2':             [0.4,  0.8],   # Migration total genetic variance
    'f11':             [0.05, 0.30],  # Within-trait vertical transmission for EA
    'f22':             [0.05, 0.30],  # Within-trait vertical transmission for migration
    're':              [0.0,  0.4],   # Environmental correlation between traits
    'am11':            [0.25, 0.75],  # Within-island spousal correlation on EA (trait 1)
    'rg':              [0.25, 0.90],  # Genetic correlation between EA and migration
    'move_p':          [0.05, 0.30],  # Proportion of each island's population that migrates per generation
}

# Fixed parameters (not sampled)
FIXED_PARAMS = {
    'prop_h2_latent2': 1.0,   # Migration: all genetic effects are latent (no observable PGS)
    'f12': 0.0,               # No cross-trait vertical transmission
    'f21': 0.0,               # No cross-trait vertical transmission
    'am12': 0.0,              # No cross-mate AM
    'am21': 0.0,
    'am22': 0.0,              # No AM on migration trait
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

def generate_conditions(n_conditions=N_CONDITIONS_TOTAL, seed=42):
    """Generate parameter combinations by sampling uniformly from PARAM_BOUNDS."""
    np.random.seed(seed)

    conditions = []
    param_names = list(PARAM_BOUNDS.keys())

    for i in range(n_conditions):
        condition = {'name': f'Condition_{i+1:05d}'}

        for param_name in param_names:
            lo, hi = PARAM_BOUNDS[param_name]
            condition[param_name] = float(np.random.uniform(lo, hi))

        condition.update(FIXED_PARAMS)
        conditions.append(condition)

    return conditions


CONDITIONS_FILE = PROJECT_BASE / "conditions_config.csv"


def save_conditions_config():
    """Write conditions CSV atomically so parallel tasks never read a partial file."""
    import time
    PROJECT_BASE.mkdir(parents=True, exist_ok=True)

    # --- Reader path: wait for the file to appear and be complete ---
    if CONDITIONS_FILE.exists():
        for attempt in range(60):          # wait up to 5 minutes
            try:
                existing_df = pd.read_csv(CONDITIONS_FILE)
                if len(existing_df) == N_CONDITIONS_TOTAL:
                    print(f"✓ Using existing conditions from {CONDITIONS_FILE} ({len(existing_df)} conditions)")
                    return
                else:
                    print(f"  Waiting for conditions file (has {len(existing_df)}/{N_CONDITIONS_TOTAL} rows)…")
            except Exception:
                print(f"  Waiting for conditions file to be readable (attempt {attempt+1})…")
            time.sleep(5)
        # File exists but never reached the right size — regenerate below
        print(f"⚠ Conditions file incomplete after waiting; regenerating.")

    # --- Writer path: generate and write atomically via rename ---
    print(f"Generating {N_CONDITIONS_TOTAL} parameter conditions...")
    conditions = generate_conditions(n_conditions=N_CONDITIONS_TOTAL, seed=42)
    df = pd.DataFrame(conditions)
    tmp_file = CONDITIONS_FILE.with_suffix(".tmp")
    df.to_csv(tmp_file, index=False)
    tmp_file.replace(CONDITIONS_FILE)          # atomic on POSIX/GPFS
    print(f"✓ Saved {len(conditions)} conditions to {CONDITIONS_FILE}")


def load_conditions_for_task(task_id):
    """Load the batch of CONDITIONS_PER_JOB conditions assigned to this SLURM task."""
    df = pd.read_csv(CONDITIONS_FILE)
    start = (task_id - 1) * CONDITIONS_PER_JOB   # 0-based start index
    end   = start + CONDITIONS_PER_JOB            # exclusive
    if start >= len(df):
        raise ValueError(f"Task ID {task_id} exceeds available conditions ({len(df)})")
    return [row.to_dict() for _, row in df.iloc[start:end].iterrows()]


# ============================================================================
# SIMULATION FUNCTIONS
# ============================================================================

def setup_matrices(params):
    """
    Setup covariance and other matrices for the 04AMmigration condition.

    Trait 1 = EA:        has both observable and latent genetic components.
    Trait 2 = Migration: prop_h2_latent2 = 1 (all genetic effects are latent;
                         d_mat second diagonal = 0, so no observable PGS signal).

    Vertical transmission is within-trait only (f12 = f21 = 0).
    No shared environmental effects (s_mat = 0).
    Within-island AM on EA (trait 1); migration sorted by trait 2 across 5 islands.
    """
    vg1 = params['vg1']
    vg2 = params['vg2']
    rg  = params['rg']
    re  = params['re']
    prop_h2_latent1 = params['prop_h2_latent1']
    prop_h2_latent2 = params['prop_h2_latent2']   # fixed at 1.0

    # Genetic correlation matrix
    k2_matrix = np.array([[1, rg], [rg, 1]])

    # Observable genetic variance components
    # Trait 2: vg_obs2 = 0 because prop_h2_latent2 = 1 (entirely latent)
    vg_obs1 = vg1 * (1.0 - prop_h2_latent1)
    vg_obs2 = vg2 * (1.0 - prop_h2_latent2)   # = 0 when prop_h2_latent2 = 1
    d11 = np.sqrt(max(vg_obs1, 0.0))
    d21 = 0.0
    d22 = np.sqrt(max(vg_obs2 - d21**2, 0.0))
    delta_mat = np.array([[d11, 0.0], [d21, d22]])

    # Latent genetic variance components
    vg_lat1 = vg1 * prop_h2_latent1
    vg_lat2 = vg2 * prop_h2_latent2            # = vg2 when prop_h2_latent2 = 1
    a11 = np.sqrt(max(vg_lat1, 0.0))
    a21 = 0.0
    a22 = np.sqrt(max(vg_lat2 - a21**2, 0.0))
    a_mat = np.array([[a11, 0.0], [a21, a22]])

    # Total genetic covariance matrix
    # covg[0,0] = vg1, covg[1,1] = vg2
    # covg[0,1] = rg * sqrt(vg_lat1) * sqrt(vg2)  (only latent part of trait 1 co-varies
    #             with all of trait 2 since trait 2 is entirely latent)
    covg_mat = (delta_mat @ k2_matrix @ delta_mat.T) + (a_mat @ k2_matrix @ a_mat.T)

    # Environmental covariance
    ve1  = 1.0 - vg1
    ve2  = 1.0 - vg2
    cove = re * np.sqrt(ve1 * ve2)
    cove_mat = np.array([[ve1, cove], [cove, ve2]])

    # Total phenotypic covariance
    covy_mat = covg_mat + cove_mat

    # Vertical transmission: within-trait only (f12 = f21 = 0)
    f11 = params['f11']
    f22 = params['f22']
    f_mat = np.array([[f11, 0.0], [0.0, f22]])

    # No shared environmental effects
    s_mat = np.zeros((2, 2))

    # Within-island AM on EA (trait 1); am_list also needed by base class constructor
    am11             = params['am11']
    am_list          = [am11 for _ in range(N_GENERATIONS)]
    within_island_am = am11   # constant across generations

    return {
        'cove_mat':         cove_mat,
        'f_mat':            f_mat,
        's_mat':            s_mat,
        'a_mat':            a_mat,
        'd_mat':            delta_mat,
        'am_list':          am_list,
        'within_island_am': within_island_am,
        'covy_mat':         covy_mat,
        'k2_matrix':        k2_matrix,
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
            # Trimmed results are indexed 0, 1, 2 (corresponding to FINAL_GENS)
            gen_indices = [0, 1, 2]

            pairs = find_relationship_pairs(
                results, rel_path,
                output_format='long',
                generations=gen_indices
            )

            if len(pairs) == 0:
                continue

            # Rename columns to expected format
            pairs = pairs.rename(columns={'Person_ID': 'ID1', 'Relative_ID': 'ID2'})

            pairs_with_measures = pairs.copy()
            all_vars_to_add = ['PGS1', 'PGS2', 'Y1', 'Y2']

            for var in all_vars_to_add:
                pairs_with_measures[f'{var}_1'] = pairs['ID1'].map(
                    lambda x: measures_lookup.get(x, {}).get(var, np.nan)
                )
                pairs_with_measures[f'{var}_2'] = pairs['ID2'].map(
                    lambda x: measures_lookup.get(x, {}).get(var, np.nan)
                )

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

    # Extract within_island_am (not a base-class kwarg)
    within_island_am = matrices.pop('within_island_am')

    # Initialize island simulation
    # mate_on_trait is set internally by IslandMigrationSimulation via mating_trait=1
    sim = IslandMigrationSimulation(
        n_islands=N_ISLANDS,
        move_p=params['move_p'],
        within_island_am=within_island_am,
        migration_trait=2,   # islands stratified by Y2 (migration trait)
        mating_trait=1,      # within-island AM on Y1 (EA)
        n_jobs=1,
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
        **matrices
    )

    # Run simulation
    print(f"    Running simulation ({N_GENERATIONS} generations)...")
    results = sim.run_simulation()

    # Trim results to only keep final generations to save memory
    mates_indices = (
        FINAL_GENS + [FINAL_GENS[-1] + 1]
        if FINAL_GENS[-1] + 1 < len(results['HISTORY']['MATES'])
        else FINAL_GENS
    )

    trimmed_results = {
        'HISTORY': {
            'PHEN': [results['HISTORY']['PHEN'][i] for i in FINAL_GENS],
            'XO':   [results['HISTORY']['XO'][i]   for i in FINAL_GENS],
            'XL':   [results['HISTORY']['XL'][i]   for i in FINAL_GENS],
            'MATES': [
                results['HISTORY']['MATES'][i]
                if i < len(results['HISTORY']['MATES']) else None
                for i in mates_indices
            ]
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
    print(f"# Parameters (04AMmigration):")
    print(f"#   EA (trait 1):        vg1={condition['vg1']:.4f}, "
          f"prop_h2_latent1={condition['prop_h2_latent1']:.4f}, am11={condition['am11']:.4f}")
    print(f"#   Migration (trait 2): vg2={condition['vg2']:.4f}, "
          f"prop_h2_latent2={condition['prop_h2_latent2']:.4f} (fixed=1)")
    print(f"#   Islands: {N_ISLANDS}, move_p={condition['move_p']:.4f}")
    print(f"#   rg={condition['rg']:.4f}, re={condition['re']:.4f}")
    print(f"#   f11={condition['f11']:.4f}, f22={condition['f22']:.4f} "
          f"(within-trait VT only; f12=f21=0, s=0)")
    print(f"# Running {ITERATIONS_PER_CONDITION} iterations, {N_GENERATIONS} generations")
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
            correlations_df = run_single_iteration(
                iteration, condition_name, condition, matrices.copy()
            )

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
        for param in PARAM_BOUNDS.keys():
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

            # Add sampled parameters
            for param in PARAM_BOUNDS.keys():
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
    print("Condition: 04AMmigration (EA mating + migration trait)")
    print("="*70)
    print(f"Project base: {PROJECT_BASE}")
    print(f"SLURM Task ID: {SLURM_TASK_ID}")
    print(f"Population size: {POP_SIZE}")
    print(f"Generations: {N_GENERATIONS} (analyzing final 3: {FINAL_GENS})")
    print(f"Iterations per condition: {ITERATIONS_PER_CONDITION}")
    print(f"Causal variants: {N_CV}")
    print(f"Fixed: prop_h2_latent2=1, f12=f21=0, s=0, am12=am21=am22=0")
    print(f"Islands: {N_ISLANDS} (migration sorted by trait 2, mating on trait 1 within islands)")
    print(f"Sampled: move_p in {PARAM_BOUNDS['move_p']}")
    print("="*70 + "\n")

    # Save/load conditions configuration
    save_conditions_config()

    # Load all conditions assigned to this task
    try:
        conditions = load_conditions_for_task(SLURM_TASK_ID)
        print(f"Loaded {len(conditions)} conditions for Task {SLURM_TASK_ID}")
    except Exception as e:
        print(f"Error loading conditions: {e}")
        return

    # Run each condition
    n_success = 0
    for condition in conditions:
        success = run_condition(condition, PROJECT_BASE)
        if success:
            n_success += 1

    print(f"\n{'='*70}")
    print(f"TASK {SLURM_TASK_ID}: {n_success}/{len(conditions)} conditions completed successfully")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
