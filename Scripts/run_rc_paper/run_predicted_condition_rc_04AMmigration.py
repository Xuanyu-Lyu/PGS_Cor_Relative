"""
Runs TOTAL_ITERATIONS with N=40000 across 5 islands with phenotype-driven
migration.  Saves final 3 generations and summary statistics.

Condition 04_AMmigration:
  - Trait 1 (EA): AE model, latent AM on trait 1 (am11), no VT
  - Trait 2 (migration): fully latent (prop_h2_latent2=1.0), no AM, no VT
  - Island migration driven by trait 2 (migration_trait=2)
  - Mating driven by trait 1 (mating_trait=1)
  - Parameter values are the posterior means from the NPE fit to the observed data.
"""

import numpy as np
import pandas as pd
import os
import sys
import traceback
from pathlib import Path

# Add the SimulationFunctions directory to path
script_dir = Path(__file__).parent
simfunc_dir = script_dir.parent / "SimulationFunctions"
sys.path.insert(0, str(simfunc_dir))

from IslandSimulation import IslandMigrationSimulation
from save_simulation_data import save_simulation_results
from find_relative_setbased import find_relationship_pairs
from extract_measures import extract_individual_measures, compute_correlations_for_multiple_variables

# Define output directories
# SCRATCH_DIR for raw iteration data (large files)
# PROJECT_DIR for summary statistics (small files)
SCRATCH_BASE = Path("/scratch/alpine/xuly4739/PGS_Cor_Relative/Data/predicted_condition_04AMmigration")
PROJECT_BASE = Path("/projects/xuly4739/Py_Projects/PGS_Cor_Relative/Data/predicted_condition_04AMmigration")

# Condition 04_AMmigration: AE model with latent AM on trait 1 and island migration
# driven by trait 2.  prop_h2_latent2 is fixed at 1.0 (fully latent migration trait).
# Parameter values are the posterior means from the NPE fit to the observed data.
CONDITION = {
    'name': 'Predicted_Condition_04AMmigration',
    # Trait 1 (EA): AE model, latent AM (posterior mean)
    'prop_h2_latent1': 0.7359,
    'vg1': 0.4467,
    'am11': 0.3224,   # latent AM on trait 1
    'f11':  0.0569,   # vertical transmission (trait 1; posterior mean)
    # Trait 2 (migration): fully latent, no AM, no VT
    'prop_h2_latent2': 1.0,   # fixed
    'vg2': 0.5928,
    'am22': 0.0,
    'f22':  0.1867,   # vertical transmission (trait 2; posterior mean)
    # Cross-trait parameters (posterior means)
    'f12': 0.0,
    'f21': 0.0,
    'am12': 0.0,
    'am21': 0.0,
    'rg': 0.4511,     # genetic correlation
    're': 0.1341,     # environmental correlation
    # Island migration
    'move_p': 0.1511, # fraction migrating per generation (posterior mean)
}

# Simulation parameters
TOTAL_ITERATIONS = 100   # Total iterations across all array tasks
POP_SIZE    = 40000
N_ISLANDS   = 5
N_GENERATIONS = 40       # 40 generations to reach equilibrium
FINAL_GENS  = [37, 38, 39]  # Final 3 generations to analyse
N_CV        = 1000
MAF_MIN     = 0.01
MAF_MAX     = 0.5

# Relationship types to analyze
RELATIONSHIP_TYPES = [
    'S',        # Siblings
    'HSFS',     # Half-siblings / first cousins once removed
    'PSC',      # Avuncular
    'PPSCC',    # First cousins
    'M',        # Mates
    'MS',       # Mate's sibling
    'SMS',      # Sibling's mate's sibling
    'MSC',      # Mate's sibling's child
    'MSM',      # Mate's sibling's mate
    'SMSC',     # Sibling's mate's sibling's child
    'SMSM',     # Sibling's mate's sibling's mate
    'SMSMS',    # Sibling's mate's sibling's mate's sibling
    'PSMSC',
    'MSMSC',
    'MSMSM',
    'SMSMSC',
    'MSMSMS',
]


def setup_matrices(params):
    """
    Build covariance matrices for condition 04_AMmigration.

    Trait 1: AE model, latent AM (am11), optional VT (f11).
    Trait 2: fully latent (prop_h2_latent2=1.0), no direct AM.
    Cross-trait: rg, re.

    Returns a dict ready to be unpacked into IslandMigrationSimulation
    (except 'move_p' and 'within_island_am' which are passed separately).
    """
    vg1 = params['vg1']
    vg2 = params['vg2']
    rg  = params['rg']
    re  = params['re']
    prop_h2_latent1 = params['prop_h2_latent1']
    prop_h2_latent2 = params['prop_h2_latent2']   # = 1.0 (fixed)

    k2_matrix = np.array([[1, rg], [rg, 1]])

    # Observable genetic variance components
    vg_obs1 = vg1 * (1 - prop_h2_latent1)
    vg_obs2 = vg2 * (1 - prop_h2_latent2)   # = 0 since prop_h2_latent2=1
    d11 = np.sqrt(vg_obs1)
    d22 = np.sqrt(vg_obs2)
    delta_mat = np.array([[d11, 0], [0, d22]])

    # Latent genetic variance components
    vg_lat1 = vg1 * prop_h2_latent1
    vg_lat2 = vg2 * prop_h2_latent2
    a11 = np.sqrt(vg_lat1)
    a22 = np.sqrt(vg_lat2)
    a_mat = np.array([[a11, 0], [0, a22]])

    # Total genetic covariance
    covg_mat = (delta_mat @ k2_matrix @ delta_mat.T) + (a_mat @ k2_matrix @ a_mat.T)

    # Environmental covariance
    ve1 = 1 - vg1
    ve2 = 1 - vg2
    cove = re * np.sqrt(ve1 * ve2)
    cove_mat = np.array([[ve1, cove], [cove, ve2]])

    # Total phenotypic covariance
    covy_mat = covg_mat + cove_mat

    # am_list governs initial genetic structure (trait 1 AM only)
    am_list = [params['am11'] for _ in range(N_GENERATIONS)]

    # Vertical transmission matrix (f12=f21=0 fixed)
    f_mat = np.array([[params['f11'], params['f12']],
                       [params['f21'], params['f22']]])

    # No shared family environment
    s_mat = np.zeros((2, 2))

    # within_island_am: AM on trait 1 within each island
    within_island_am = params['am11']

    return {
        'cove_mat':       cove_mat,
        'f_mat':          f_mat,
        's_mat':          s_mat,
        'a_mat':          a_mat,
        'd_mat':          delta_mat,
        'am_list':        am_list,
        'covy_mat':       covy_mat,
        'k2_matrix':      k2_matrix,
        'within_island_am': within_island_am,
    }


def run_single_iteration(iteration, condition_name, params, matrices, scratch_dir):
    """
    Run one simulation iteration using IslandMigrationSimulation.
    """
    print(f"\n{'='*60}")
    print(f"Running {condition_name} - Iteration {iteration + 1}")
    print(f"{'='*60}")

    seed = 12345 + iteration

    iter_dir = scratch_dir / f"Iteration_{iteration+1:03d}"
    iter_dir.mkdir(parents=True, exist_ok=True)
    summary_filename = str(iter_dir / f"iteration_{iteration+1:03d}_summary.txt")

    # Pop island-specific keys before unpacking into base-class kwargs
    within_island_am = matrices.pop('within_island_am')

    sim = IslandMigrationSimulation(
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
        n_islands=N_ISLANDS,
        move_p=params['move_p'],
        within_island_am=within_island_am,
        migration_trait=2,
        mating_trait=1,
        **matrices,
    )

    results = sim.run_simulation()
    return results


def compute_pgs_from_components(measures_df):
    """PGS1 = TPO1 + TMO1; PGS2 = TPO2 + TMO2."""
    df = measures_df.copy()
    df['PGS1'] = df['TPO1'] + df['TMO1']
    df['PGS2'] = df['TPO2'] + df['TMO2']
    return df


def extract_and_analyze_relationships(results, iteration, output_dir):
    """
    Extract individual measures and compute correlations for all relationship types.
    Returns a DataFrame or None.
    """
    print(f"\n  Extracting individual measures...")

    variables = ['Y1', 'Y2', 'TPO1', 'TPO2', 'TMO1', 'TMO2']

    try:
        individual_measures = extract_individual_measures(results, variables)
        individual_measures = compute_pgs_from_components(individual_measures)
        print(f"  ✓ Extracted measures for {len(individual_measures):,} individuals")
    except Exception as e:
        print(f"  ✗ Error extracting measures: {e}")
        return None

    measures_lookup = individual_measures.set_index('ID').to_dict('index')

    print(f"  Finding relationship pairs for {len(RELATIONSHIP_TYPES)} types...")
    all_correlations = []

    for rel_path in RELATIONSHIP_TYPES:
        try:
            pairs = find_relationship_pairs(
                results, rel_path,
                output_format='long',
                generations=FINAL_GENS,
            )

            if len(pairs) == 0:
                continue

            print(f"    {rel_path}: {len(pairs):,} pairs", end='')

            pairs_with_measures = pairs.copy()

            for var in variables:
                pairs_with_measures[f'{var}_1'] = pairs_with_measures['Person_ID'].map(
                    lambda id_val: measures_lookup.get(id_val, {}).get(var, np.nan)
                )
                pairs_with_measures[f'{var}_2'] = pairs_with_measures['Relative_ID'].map(
                    lambda id_val: measures_lookup.get(id_val, {}).get(var, np.nan)
                )

            for var in ['PGS1', 'PGS2']:
                pairs_with_measures[f'{var}_1'] = pairs_with_measures['Person_ID'].map(
                    lambda id_val: measures_lookup.get(id_val, {}).get(var, np.nan)
                )
                pairs_with_measures[f'{var}_2'] = pairs_with_measures['Relative_ID'].map(
                    lambda id_val: measures_lookup.get(id_val, {}).get(var, np.nan)
                )

            correlation_vars = ['PGS1', 'PGS2', 'Y1', 'Y2']
            correlations = compute_correlations_for_multiple_variables(
                pairs_with_measures, correlation_vars, relationship_col='Relationship'
            )

            correlations['Iteration'] = iteration + 1
            correlations['RelationshipPath'] = rel_path

            all_correlations.append(correlations)
            print(f" ✓")

        except Exception as e:
            print(f"    {rel_path}: Error - {e}")
            continue

    if all_correlations:
        correlations_df = pd.concat(all_correlations, ignore_index=True)
        cols = ['Iteration', 'RelationshipPath', 'Relationship',
                'Variable', 'N_Pairs', 'Correlation', 'P_Value']
        correlations_df = correlations_df[cols]
        return correlations_df

    return None


def extract_pgs_correlations(results):
    """
    Compute mate PGS correlations (PGS1 and PGS2) from the final generation.
    """
    final_gen = N_GENERATIONS - 1

    if 'HISTORY' not in results or final_gen >= len(results['HISTORY']['PHEN']):
        return None, None

    phen_df = results['HISTORY']['PHEN'][final_gen]
    if phen_df is None or phen_df.empty:
        return None, None

    if final_gen + 1 < len(results['HISTORY']['MATES']):
        mates_dict  = results['HISTORY']['MATES'][final_gen + 1]
        males_df    = mates_dict.get('males.PHENDATA')
        females_df  = mates_dict.get('females.PHENDATA')

        if males_df is not None and females_df is not None:
            males_df   = males_df.copy()
            females_df = females_df.copy()
            males_df['PGS1']   = males_df['TPO1']   + males_df['TMO1']
            males_df['PGS2']   = males_df['TPO2']   + males_df['TMO2']
            females_df['PGS1'] = females_df['TPO1'] + females_df['TMO1']
            females_df['PGS2'] = females_df['TPO2'] + females_df['TMO2']

            merged = pd.merge(
                males_df[['ID', 'PGS1', 'PGS2', 'Spouse.ID']],
                females_df[['ID', 'PGS1', 'PGS2']],
                left_on='Spouse.ID',
                right_on='ID',
                suffixes=('_male', '_female'),
            )

            if len(merged) > 1:
                cor_pgs1 = np.corrcoef(merged['PGS1_male'], merged['PGS1_female'])[0, 1]
                cor_pgs2 = np.corrcoef(merged['PGS2_male'], merged['PGS2_female'])[0, 1]
                return cor_pgs1, cor_pgs2

    return None, None


def run_predicted_condition(condition, scratch_base, project_base):
    """
    Run the iterations assigned to this SLURM array task.
    """
    condition_name = condition['name']

    task_id            = int(os.environ.get('SLURM_ARRAY_TASK_ID', '1'))
    iterations_per_task = int(os.environ.get('ITERATIONS_PER_TASK', '5'))

    start_iter = (task_id - 1) * iterations_per_task
    end_iter   = min(task_id * iterations_per_task, TOTAL_ITERATIONS)
    n_iterations = end_iter - start_iter

    print(f"\n{'#'*70}")
    print(f"# Starting simulations: {condition_name}")
    print(f"# Condition 04_AMmigration: AE model, latent AM on trait 1, island migration on trait 2")
    print(f"# Trait 1 (AE+latentAM): prop_h2_latent1={condition['prop_h2_latent1']:.4f}, "
          f"vg1={condition['vg1']:.4f}, am11={condition['am11']:.4f}, f11={condition['f11']:.4f}")
    print(f"# Trait 2 (migration, fully latent): prop_h2_latent2={condition['prop_h2_latent2']:.4f}, "
          f"vg2={condition['vg2']:.4f}, f22={condition['f22']:.4f}")
    print(f"# Islands: {N_ISLANDS},  move_p={condition['move_p']:.4f}")
    print(f"# Cross-trait: rg={condition['rg']:.4f}, re={condition['re']:.4f}")
    print(f"# Array Task {task_id}: Running iterations {start_iter+1} to {end_iter}")
    print(f"# ({n_iterations} iterations in this task)")
    print(f"{'#'*70}\n")

    scratch_dir = scratch_base / condition_name
    project_dir = project_base / condition_name
    scratch_dir.mkdir(parents=True, exist_ok=True)
    project_dir.mkdir(parents=True, exist_ok=True)

    matrices = setup_matrices(condition)

    pgs_cor_trait1  = []
    pgs_cor_trait2  = []
    all_correlations = []

    for iteration in range(start_iter, end_iter):
        try:
            results = run_single_iteration(
                iteration, condition_name, condition, matrices.copy(), scratch_dir
            )

            iter_dir = scratch_dir / f"Iteration_{iteration+1:03d}"
            save_simulation_results(
                results,
                str(iter_dir),
                file_prefix=f"iteration_{iteration+1:03d}",
                scope=FINAL_GENS,
            )

            cor1, cor2 = extract_pgs_correlations(results)
            if cor1 is not None:
                pgs_cor_trait1.append(cor1)
                pgs_cor_trait2.append(cor2)
                print(f"  → Mate PGS Correlation - Trait 1: {cor1:.4f},  Trait 2: {cor2:.4f}")

            correlations_df = extract_and_analyze_relationships(results, iteration, scratch_dir)

            if correlations_df is not None:
                corr_file = iter_dir / f"correlations_iteration_{iteration+1:03d}.csv"
                correlations_df.to_csv(corr_file, index=False)
                print(f"  ✓ Saved correlations to {corr_file.name}")
                all_correlations.append(correlations_df)

        except Exception as e:
            print(f"  ✗ Error in Iteration {iteration+1}: {e}")
            traceback.print_exc()
            continue

    # ── Save per-task summaries to PROJECT ──────────────────────────────────
    if pgs_cor_trait1:
        mate_summary = pd.DataFrame({
            'iteration':                   range(start_iter + 1, end_iter + 1),
            'mate_pgs_correlation_trait1': pgs_cor_trait1,
            'mate_pgs_correlation_trait2': pgs_cor_trait2,
        })
        mate_file = project_dir / f'mate_pgs_correlations_task_{task_id:02d}.csv'
        mate_summary.to_csv(mate_file, index=False)
        print(f"\n  ✓ Saved mate correlation summary → {mate_file.name}")
        print(f"    Trait 1 – Mean: {np.mean(pgs_cor_trait1):.4f},  SD: {np.std(pgs_cor_trait1):.4f}")
        print(f"    Trait 2 – Mean: {np.mean(pgs_cor_trait2):.4f},  SD: {np.std(pgs_cor_trait2):.4f}")

    if all_correlations:
        print(f"\n  Combining relationship correlations for this task...")
        combined_correlations = pd.concat(all_correlations, ignore_index=True)

        combined_file = project_dir / f"task_{task_id:02d}_correlations.csv"
        combined_correlations.to_csv(combined_file, index=False)
        print(f"  ✓ Saved task correlations → {combined_file.name}")

        summary = combined_correlations.groupby(['RelationshipPath', 'Variable']).agg({
            'Correlation': ['mean', 'std', 'min', 'max'],
            'N_Pairs':     'sum',
            'Iteration':   'count',
        }).round(4)

        summary_file = project_dir / f"task_{task_id:02d}_summary_statistics.csv"
        summary.to_csv(summary_file)
        print(f"  ✓ Saved task summary statistics → {summary_file.name}")

        pgs1_summary = (
            combined_correlations[combined_correlations['Variable'] == 'PGS1']
            .groupby('RelationshipPath')
            .agg(Correlation_Mean=('Correlation', 'mean'),
                 Correlation_SD=('Correlation', 'std'),
                 N_Pairs_Mean=('N_Pairs', 'mean'))
            .round(4)
        )
        print(f"\n  Summary for PGS1 correlations (this task):")
        print(pgs1_summary)


def main():
    print("\n" + "="*70)
    print("PREDICTED CONDITION SIMULATION SCRIPT - 04_AMmigration")
    print("="*70)
    print(f"Scratch base directory: {SCRATCH_BASE}")
    print(f"Project base directory: {PROJECT_BASE}")
    print(f"Condition: {CONDITION['name']}")
    print(f"Total iterations: {TOTAL_ITERATIONS}")
    print(f"Population size: {POP_SIZE}")
    print(f"Number of islands: {N_ISLANDS}")
    print(f"Number of generations: {N_GENERATIONS} (saving final 3)")
    print(f"Number of causal variants: {N_CV}")
    print(f"\nTrait 1 parameters (AE + latent AM, 04AMmigration posterior means):")
    print(f"  prop_h2_latent1={CONDITION['prop_h2_latent1']:.4f},  vg1={CONDITION['vg1']:.4f},  "
          f"am11={CONDITION['am11']:.4f},  f11={CONDITION['f11']:.4f}")
    print(f"\nTrait 2 parameters (fully latent migration trait, posterior means):")
    print(f"  prop_h2_latent2={CONDITION['prop_h2_latent2']:.4f},  vg2={CONDITION['vg2']:.4f},  "
          f"f22={CONDITION['f22']:.4f}")
    print(f"\nIsland migration: N_ISLANDS={N_ISLANDS},  move_p={CONDITION['move_p']:.4f}")
    print(f"Cross-trait (posterior means): rg={CONDITION['rg']:.4f},  re={CONDITION['re']:.4f}")
    print("="*70 + "\n")

    run_predicted_condition(CONDITION, SCRATCH_BASE, PROJECT_BASE)

    print("\n" + "="*70)
    print(f"SIMULATION COMPLETED - {CONDITION['name']}")
    print("="*70)


if __name__ == "__main__":
    main()
