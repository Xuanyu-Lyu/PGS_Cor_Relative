"""
Data Generation Script for Neural Network Training - COMBINED VERSION

This script runs forward-time simulations and immediately analyzes them without
saving raw simulation data. This increases efficiency and saves storage space.

This simulates one condition:
4) EA-driven island migration (overhauled condition 04; tested in
   Scripts/test_04AMmigration):
   - Trait 1: EA = additive genetics (PGS part + latent part) + unique environment.
     No assortative mating (random mating within islands), no vertical
     transmission, no shared environment.
   - 5 islands. Each generation a fraction move_p of each sex on each island
     migrates; movers are sorted directly on their EA phenotype (Y1) and
     redistributed so high-EA movers end up on the high-numbered islands.
   - Trait 2 is an inert dummy required by the bivariate engine
     (vg2 = 0.5, fully latent, rg = re = 0); it drives nothing.
   - Estimable parameters: move_p, vg1, prop_h2_latent1 (sampled per condition)
     and n_gen, the number of generations since the random founders (3-10).

Several n_gen values from ONE simulation
----------------------------------------
Each condition runs a single simulation for N_GEN_MAX = 10 generations and is
read out at every n_gen = G in GEN_VALUES (3..10). For G, the observed sample is
the last three generations before G, exactly as a simulation run for G
generations would be analysed:

    PHEN/XO/XL at generations [G-3, G-2, G-1] and MATES at [G-3, G-2, G-1, G]
    (MATES[i+1] holds the mating pairs formed in generation i)

Generation G's state depends only on generations 0..G, so this window is
identical to the final window of a G-generation run with the same seed (checked
in Scripts/test_04AMmigration). One simulation therefore yields len(GEN_VALUES)
training rows, one per n_gen, which share the other parameters and overlap in
the individuals they contain. They are NOT independent: every row carries the
condition name in 'Condition', and train/val/test splits must group on it
(CombineNN_Data_Large_04EAmigration.py --split and
train_npe_pgs_and_pheno.py --group_col Condition both do).

Usage:
    python DataGeneratingNN_Combined_04EAmigration.py
    (Run via SLURM array job - see submit_datagenerating_nn_combined_04EAmigration.sh)

    Local test (small and fast):
    python DataGeneratingNN_Combined_04EAmigration.py --task_id 1 --n_conditions 4 \
        --conditions_per_job 4 --pop_size 4000 --n_cv 300 --project_base /tmp/nn04
"""

import argparse
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

from island_migration import IslandMigrationSimulation
from relationship_finder import find_relationship_pairs
from postprocessing import extract_individual_measures, compute_correlations_for_multiple_variables

# ============================================================================
# CONFIGURATION  (production defaults; the CLI can override them for testing)
# ============================================================================

# Directory setup
PROJECT_BASE = Path("/projects/xuly4739/Py_Projects/PGS_Cor_Relative/Data/DataGeneratingNN_Paper/04EAmigration")

# Get SLURM array task ID
SLURM_TASK_ID = int(os.environ.get('SLURM_ARRAY_TASK_ID', '1'))

# Simulation parameters
ITERATIONS_PER_CONDITION = 1   # one simulation per condition (each gives len(GEN_VALUES) rows)
CONDITIONS_PER_JOB = 10        # Conditions (simulations) processed per SLURM job
N_CONDITIONS_TOTAL = 2500      # 250 jobs x 10 conditions; x 8 n_gen values = 20,000 rows
POP_SIZE = 40000               # Must be divisible by N_ISLANDS * 2
N_ISLANDS = 5                  # 40000 / (5*2) = 4000 per sex per island
N_GEN_MAX = 10                 # generations simulated per condition
GEN_VALUES = list(range(3, N_GEN_MAX + 1))   # n_gen values read out of each simulation
N_FINAL = 3                    # generations in each observed window
N_CV = 1000
MAF_MIN = 0.01
MAF_MAX = 0.5

# Parameter bounds for uniform sampling: [min, max]   (n_gen is not sampled: see GEN_VALUES)
PARAM_BOUNDS = {
    'move_p':          [0.01, 0.70],  # fraction of each sex per island migrating per generation
    'vg1':             [0.20, 0.80],  # EA total genetic variance
    'prop_h2_latent1': [0.30, 0.90],  # share of EA genetic variance NOT captured by the PGS
}

# Fixed parameters
FIXED_PARAMS = {
    'vg2': 0.5,                # dummy trait 2 (drives nothing)
    'prop_h2_latent2': 1.0,
    'rg': 0.0,
    're': 0.0,
    'f11': 0.0, 'f12': 0.0, 'f21': 0.0, 'f22': 0.0,   # no vertical transmission
    'am11': 0.0, 'am12': 0.0, 'am21': 0.0, 'am22': 0.0,   # no assortative mating
}

# Relationship types to analyze (HSFS is always empty without remating; PPSCC is
# empty when n_gen = 3 because the founders have no siblings)
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

def generate_conditions(n_conditions, seed=42):
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


def conditions_file():
    return PROJECT_BASE / "conditions_config.csv"


def save_conditions_config():
    """Write conditions CSV atomically so parallel tasks never read a partial file."""
    import time
    PROJECT_BASE.mkdir(parents=True, exist_ok=True)
    cfile = conditions_file()

    # --- Reader path: wait for the file to appear and be complete ---
    if cfile.exists():
        for attempt in range(60):          # wait up to 5 minutes
            try:
                existing_df = pd.read_csv(cfile)
                if len(existing_df) == N_CONDITIONS_TOTAL:
                    print(f"✓ Using existing conditions from {cfile} ({len(existing_df)} conditions)")
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
    tmp_file = cfile.with_suffix(f".{os.getpid()}.tmp")   # per-process temp name: no rename races
    df.to_csv(tmp_file, index=False)
    tmp_file.replace(cfile)                                # atomic on POSIX/GPFS
    print(f"✓ Saved {len(conditions)} conditions to {cfile}")


def load_conditions_for_task(task_id):
    """Load the batch of CONDITIONS_PER_JOB conditions assigned to this SLURM task."""
    df = pd.read_csv(conditions_file())
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
    Model matrices for the EA-migration condition.

    Trait 1 = EA: observable (PGS) + latent genetic components + unique environment.
    Trait 2 = dummy (fully latent, uncorrelated with EA).
    No vertical transmission, no shared environment, no assortative mating.
    """
    vg1 = params['vg1']
    vg2 = params['vg2']
    rg  = params['rg']
    re  = params['re']
    prop_h2_latent1 = params['prop_h2_latent1']
    prop_h2_latent2 = params['prop_h2_latent2']

    k2_matrix = np.array([[1.0, rg], [rg, 1.0]])

    # Observable (PGS) genetic path matrix
    d11 = np.sqrt(max(vg1 * (1.0 - prop_h2_latent1), 0.0))
    d22 = np.sqrt(max(vg2 * (1.0 - prop_h2_latent2), 0.0))
    delta_mat = np.array([[d11, 0.0], [0.0, d22]])

    # Latent genetic path matrix
    a11 = np.sqrt(max(vg1 * prop_h2_latent1, 0.0))
    a22 = np.sqrt(max(vg2 * prop_h2_latent2, 0.0))
    a_mat = np.array([[a11, 0.0], [0.0, a22]])

    covg_mat = (delta_mat @ k2_matrix @ delta_mat.T) + (a_mat @ k2_matrix @ a_mat.T)

    # Unique environment (total phenotypic variance = 1 in the founders)
    ve1 = 1.0 - vg1
    ve2 = 1.0 - vg2
    cove = re * np.sqrt(ve1 * ve2)
    cove_mat = np.array([[ve1, cove], [cove, ve2]])

    return {
        'cove_mat':  cove_mat,
        'f_mat':     np.zeros((2, 2)),          # no vertical transmission
        's_mat':     np.zeros((2, 2)),          # no shared environment
        'a_mat':     a_mat,
        'd_mat':     delta_mat,
        'am_list':   [0.0] * N_GEN_MAX,         # no assortative mating
        'covy_mat':  covg_mat + cove_mat,
        'k2_matrix': k2_matrix,
    }


def condition_seed(condition_name, iteration):
    """Deterministic seed from the condition number (Python's hash() is salted per process)."""
    return int(condition_name.split('_')[-1]) * 100 + iteration + 1


def window_for_generation(history, n_gen, n_final=N_FINAL):
    """HISTORY slice that a simulation run for ``n_gen`` generations would be analysed on.

    History index i is generation i (0 = founders). The observed window is
    generations [n_gen-3, n_gen-1]; MATES[i+1] holds the pairs formed in generation i,
    so the mates of the window are MATES[n_gen-3 .. n_gen].
    """
    gens = list(range(n_gen - n_final, n_gen))
    n_mates = len(history['MATES'])
    if gens[0] < 0 or gens[-1] >= len(history['PHEN']):
        raise ValueError(f"n_gen={n_gen} needs generations {gens}, but history has "
                         f"{len(history['PHEN'])} generations")
    mates_idx = gens + [gens[-1] + 1] if gens[-1] + 1 < n_mates else gens
    return {
        'HISTORY': {
            'PHEN':  [history['PHEN'][i] for i in gens],
            'XO':    [history['XO'][i] for i in gens],
            'XL':    [history['XL'][i] for i in gens],
            'MATES': [history['MATES'][i] if i < n_mates else None for i in mates_idx],
        }
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

    measures_lookup = individual_measures.set_index('ID').to_dict('index')

    all_correlations = []
    for rel_path in RELATIONSHIP_TYPES:
        try:
            # windowed results are indexed 0, 1, 2
            pairs = find_relationship_pairs(results, rel_path, output_format='long',
                                            generations=[0, 1, 2])
            if len(pairs) == 0:
                continue

            pairs = pairs.rename(columns={'Person_ID': 'ID1', 'Relative_ID': 'ID2'})
            pairs_with_measures = pairs.copy()
            for var in ['PGS1', 'PGS2', 'Y1', 'Y2']:
                pairs_with_measures[f'{var}_1'] = pairs['ID1'].map(
                    lambda x: measures_lookup.get(x, {}).get(var, np.nan))
                pairs_with_measures[f'{var}_2'] = pairs['ID2'].map(
                    lambda x: measures_lookup.get(x, {}).get(var, np.nan))

            correlations = compute_correlations_for_multiple_variables(
                pairs_with_measures, ['PGS1', 'PGS2', 'Y1', 'Y2'], relationship_col='Relationship')
            correlations['Iteration'] = iteration + 1
            correlations['RelationshipPath'] = rel_path
            all_correlations.append(correlations)

        except Exception as e:
            print(f"    {rel_path}: Error - {e}")
            continue

    if all_correlations:
        correlations_df = pd.concat(all_correlations, ignore_index=True)
        cols = ['Iteration', 'RelationshipPath', 'Relationship',
                'Variable', 'N_Pairs', 'Correlation', 'P_Value']
        return correlations_df[cols]
    return None


def run_single_iteration(iteration, condition_name, params, matrices):
    """Run ONE simulation of N_GEN_MAX generations and analyse it at every n_gen in GEN_VALUES.

    Returns a long DataFrame of correlations with an 'n_gen' column, or None.
    """
    print(f"\n  Iteration {iteration + 1}/{ITERATIONS_PER_CONDITION}")
    seed = condition_seed(condition_name, iteration)

    # The engine draws its causal variants in __init__ BEFORE it applies `seed`, so
    # seed the global generator here too; otherwise identical seeds give different runs.
    np.random.seed(seed)
    sim = IslandMigrationSimulation(
        n_islands=N_ISLANDS,
        move_p=params['move_p'],
        within_island_am=0.0,  # random mating within islands
        migration_trait=1,     # islands sorted directly on EA (Y1)
        mating_trait=1,
        n_jobs=1,
        n_CV=N_CV,
        rg_effects=params['rg'],
        maf_min=MAF_MIN,
        maf_max=MAF_MAX,
        num_generations=N_GEN_MAX,
        pop_size=POP_SIZE,
        mating_type="phenotypic",
        avoid_inbreeding=True,
        save_each_gen=True,
        save_covs=False,
        seed=seed,
        output_summary_filename=None,
        **matrices
    )

    print(f"    Running simulation ({N_GEN_MAX} generations, seed {seed})...")
    results = sim.run_simulation()
    history = results['HISTORY']
    if len(history['PHEN']) < N_GEN_MAX + 1:
        print(f"    ✗ Simulation stopped early ({len(history['PHEN']) - 1} generations)")

    frames = []
    for n_gen in GEN_VALUES:
        try:
            window = window_for_generation(history, n_gen)
        except ValueError as e:
            print(f"    n_gen={n_gen}: skipped ({e})")
            continue
        print(f"    Analyzing relationships for n_gen={n_gen} "
              f"(generations {n_gen - N_FINAL}-{n_gen - 1})...")
        corr = extract_and_analyze_relationships(window, iteration)
        if corr is not None:
            corr['n_gen'] = n_gen
            frames.append(corr)

    del results, history, sim
    if frames:
        out = pd.concat(frames, ignore_index=True)
        print(f"    ✓ Computed {len(out)} correlations over {len(frames)} n_gen values")
        return out
    print(f"    ✗ No correlations computed")
    return None


# ============================================================================
# MAIN CONDITION PROCESSING
# ============================================================================

def run_condition(condition, project_base):
    """Run all iterations for a single condition (one row per iteration x n_gen)."""
    condition_name = condition['name']

    print(f"\n{'#'*70}")
    print(f"# {condition_name}")
    print(f"# Parameters (04EAmigration):")
    print(f"#   EA: vg1={condition['vg1']:.4f}, prop_h2_latent1={condition['prop_h2_latent1']:.4f}")
    print(f"#   Islands: {N_ISLANDS}, move_p={condition['move_p']:.4f} (migration sorted on EA)")
    print(f"#   No AM, no VT, no shared environment")
    print(f"# {ITERATIONS_PER_CONDITION} simulation(s) of {N_GEN_MAX} generations, "
          f"read out at n_gen = {GEN_VALUES}")
    print(f"# SLURM Task ID: {SLURM_TASK_ID}")
    print(f"{'#'*70}\n")

    project_dir = project_base / condition_name
    project_dir.mkdir(parents=True, exist_ok=True)

    params_file = project_dir / "condition_parameters.csv"
    pd.DataFrame([condition]).to_csv(params_file, index=False)
    print(f"  ✓ Saved parameters to {params_file}")

    matrices = setup_matrices(condition)

    iteration_status = []
    all_correlations = []

    for iteration in range(ITERATIONS_PER_CONDITION):
        iter_info = {'Iteration': iteration + 1, 'Status': 'Failed', 'Error': None,
                     'N_Correlations': 0, 'N_gen_values': 0}
        try:
            correlations_df = run_single_iteration(iteration, condition_name, condition,
                                                   dict(matrices))
            if correlations_df is not None:
                iter_info['Status'] = 'Success'
                iter_info['N_Correlations'] = len(correlations_df)
                iter_info['N_gen_values'] = correlations_df['n_gen'].nunique()
                all_correlations.append(correlations_df)
        except Exception as e:
            iter_info['Error'] = str(e)
            print(f"    ✗ Error in Iteration {iteration+1}: {e}")
            traceback.print_exc()
        iteration_status.append(iter_info)

    status_file = project_dir / "iteration_status.csv"
    pd.DataFrame(iteration_status).to_csv(status_file, index=False)
    print(f"\n  ✓ Saved iteration status to {status_file}")

    if not all_correlations:
        print(f"\n  ⚠ No correlations computed for {condition_name}")
        return False

    print(f"\n  Saving correlation results for {condition_name}...")
    combined = pd.concat(all_correlations, ignore_index=True)
    for param in PARAM_BOUNDS:
        combined[f'param_{param}'] = condition[param]
    combined['param_n_gen'] = combined['n_gen']

    output_file = project_dir / "all_iterations_correlations.csv"
    combined.to_csv(output_file, index=False)
    print(f"  ✓ Saved {len(combined)} correlation records to {output_file}")

    summary = combined.groupby(['n_gen', 'RelationshipPath', 'Variable']).agg({
        'Correlation': ['mean', 'std', 'min', 'max'],
        'N_Pairs': 'sum',
        'Iteration': 'count'
    }).round(4)
    summary_file = project_dir / "summary_statistics.csv"
    summary.to_csv(summary_file)
    print(f"  ✓ Saved summary statistics to {summary_file}")

    # Wide format for NN training: one row per (iteration, n_gen)
    print(f"  Creating NN training format...")
    nn_rows = []
    for (iteration, n_gen), d in combined.groupby(['Iteration', 'n_gen'], sort=True):
        row = {
            'Iteration': int(iteration),
            'Condition': condition_name,          # group id: all n_gen rows of one simulation
            'Sample': f"{condition_name}_it{int(iteration)}_G{int(n_gen)}",
        }
        for param in PARAM_BOUNDS:
            row[f'param_{param}'] = condition[param]
        row['param_n_gen'] = int(n_gen)
        for _, corr_row in d.iterrows():
            col_name = f"{corr_row['RelationshipPath']}_{corr_row['Variable']}"
            row[col_name] = corr_row['Correlation']
            row[f"{col_name}_N"] = corr_row['N_Pairs']
        nn_rows.append(row)

    nn_training_file = project_dir / "nn_training_format.csv"
    pd.DataFrame(nn_rows).to_csv(nn_training_file, index=False)
    print(f"  ✓ Saved NN training format ({len(nn_rows)} rows) to {nn_training_file}")

    n_success = sum(1 for s in iteration_status if s['Status'] == 'Success')
    print(f"\n  Summary: {n_success}/{ITERATIONS_PER_CONDITION} iterations successful")
    return True


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Generate NN training data for condition 04EAmigration")
    p.add_argument("--task_id", type=int, default=SLURM_TASK_ID)
    p.add_argument("--project_base", type=str, default=str(PROJECT_BASE))
    p.add_argument("--n_conditions", type=int, default=N_CONDITIONS_TOTAL)
    p.add_argument("--conditions_per_job", type=int, default=CONDITIONS_PER_JOB)
    p.add_argument("--pop_size", type=int, default=POP_SIZE)
    p.add_argument("--n_cv", type=int, default=N_CV)
    return p.parse_args()


def main():
    """Main execution function."""
    global PROJECT_BASE, SLURM_TASK_ID, N_CONDITIONS_TOTAL, CONDITIONS_PER_JOB, POP_SIZE, N_CV
    args = parse_args()
    PROJECT_BASE = Path(args.project_base)
    SLURM_TASK_ID = args.task_id
    N_CONDITIONS_TOTAL = args.n_conditions
    CONDITIONS_PER_JOB = args.conditions_per_job
    POP_SIZE = args.pop_size
    N_CV = args.n_cv
    if POP_SIZE % (N_ISLANDS * 2) != 0:
        raise ValueError(f"pop_size must be divisible by {N_ISLANDS * 2}")

    print("\n" + "="*70)
    print("DATA GENERATION FOR NEURAL NETWORK TRAINING - COMBINED VERSION")
    print("Condition: 04EAmigration (EA-driven island migration; no AM, no VT)")
    print("="*70)
    print(f"Project base: {PROJECT_BASE}")
    print(f"SLURM Task ID: {SLURM_TASK_ID}")
    print(f"Population size: {POP_SIZE}, islands: {N_ISLANDS}, causal variants: {N_CV}")
    print(f"Generations simulated: {N_GEN_MAX}; read out at n_gen = {GEN_VALUES} "
          f"(last {N_FINAL} generations each)")
    print(f"Sampled: {PARAM_BOUNDS}")
    print("="*70 + "\n")

    save_conditions_config()

    try:
        conditions = load_conditions_for_task(SLURM_TASK_ID)
        print(f"Loaded {len(conditions)} conditions for Task {SLURM_TASK_ID}")
    except Exception as e:
        print(f"Error loading conditions: {e}")
        return

    n_success = 0
    for condition in conditions:
        if run_condition(condition, PROJECT_BASE):
            n_success += 1

    print(f"\n{'='*70}")
    print(f"TASK {SLURM_TASK_ID}: {n_success}/{len(conditions)} conditions completed successfully")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
