"""
Data Generation Script for Neural Network Training - PART 2: ANALYSIS

This script loads previously saved simulation data and computes correlations
between relatives for neural network training.

Usage:
    python DataGeneratingNN_Analyze.py [--task_id TASK_ID] [--condition CONDITION_NAME]
    
Examples:
    # Analyze condition from SLURM array job
    python DataGeneratingNN_Analyze.py
    
    # Analyze specific condition by name
    python DataGeneratingNN_Analyze.py --condition Condition_0001
    
    # Analyze specific task ID
    python DataGeneratingNN_Analyze.py --task_id 1
"""

import numpy as np
import pandas as pd
import sys
import os
import traceback
import argparse
from pathlib import Path

# Add the SimulationFunctions directory to path
script_dir = Path(__file__).parent
simfunc_dir = script_dir.parent / "SimulationFunctions"
sys.path.insert(0, str(simfunc_dir))

from find_relative_setbased import find_relationship_pairs
from extract_measures import extract_individual_measures, compute_correlations_for_multiple_variables

# ============================================================================
# CONFIGURATION
# ============================================================================

# Directory setup
SCRATCH_BASE = Path("/scratch/alpine/xuly4739/PGS_Cor_Relative/Data/DataGeneratingNN")
PROJECT_BASE = Path("/projects/xuly4739/Py_Projects/PGS_Cor_Relative/Data/DataGeneratingNN")

# Simulation parameters
FINAL_GENS = [12, 13, 14]  # Final 3 generations to analyze

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
]

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_simulation_results(output_folder, file_prefix):
    """
    Load simulation results from saved TSV files.
    
    Args:
        output_folder (str): Path to folder containing saved files
        file_prefix (str): Prefix used when saving files
        
    Returns:
        dict: Results dictionary compatible with find_relationship_pairs
    """
    import glob
    
    results = {
        'HISTORY': {
            'PHEN': [],
            'XO': [],
            'XL': [],
            'MATES': []
        }
    }
    
    print(f"    Loading files from {output_folder} with prefix {file_prefix}")
    
    # Find all generation files
    phen_files = sorted(glob.glob(os.path.join(output_folder, f"{file_prefix}_phen_gen*.tsv")))
    xo_files = sorted(glob.glob(os.path.join(output_folder, f"{file_prefix}_xo_gen*.tsv")))
    xl_files = sorted(glob.glob(os.path.join(output_folder, f"{file_prefix}_xl_gen*.tsv")))
    
    if not phen_files:
        print(f"    Warning: No phenotype files found")
        return None
    
    # Load PHEN data
    for phen_file in phen_files:
        try:
            phen_df = pd.read_csv(phen_file, sep='\t')
            results['HISTORY']['PHEN'].append(phen_df)
        except Exception as e:
            print(f"    Error loading {phen_file}: {e}")
            return None
    
    # Load XO data
    for xo_file in xo_files:
        try:
            xo_arr = np.loadtxt(xo_file, delimiter='\t')
            results['HISTORY']['XO'].append(xo_arr)
        except Exception as e:
            print(f"    Error loading {xo_file}: {e}")
            return None
    
    # Load XL data
    for xl_file in xl_files:
        try:
            xl_arr = np.loadtxt(xl_file, delimiter='\t')
            results['HISTORY']['XL'].append(xl_arr)
        except Exception as e:
            print(f"    Error loading {xl_file}: {e}")
            return None
    
    # Load MATES data (males and females for each generation)
    # Extract generation numbers from phen files to know what to look for
    for phen_file in phen_files:
        gen_num = int(phen_file.split('_gen')[1].split('.')[0])
        males_file = os.path.join(output_folder, f"{file_prefix}_mates_gen{gen_num}_males.tsv")
        females_file = os.path.join(output_folder, f"{file_prefix}_mates_gen{gen_num}_females.tsv")
        
        mates_dict = {}
        if os.path.exists(males_file):
            try:
                mates_dict['males.PHENDATA'] = pd.read_csv(males_file, sep='\t')
            except Exception as e:
                print(f"    Warning: Could not load {males_file}: {e}")
        
        if os.path.exists(females_file):
            try:
                mates_dict['females.PHENDATA'] = pd.read_csv(females_file, sep='\t')
            except Exception as e:
                print(f"    Warning: Could not load {females_file}: {e}")
        
        results['HISTORY']['MATES'].append(mates_dict if mates_dict else None)
    
    print(f"    Loaded {len(results['HISTORY']['PHEN'])} generations")
    return results

def load_condition(task_id):
    """Load the condition for a specific task ID."""
    conditions_file = PROJECT_BASE / "conditions_config.csv"
    df = pd.read_csv(conditions_file)
    if task_id > len(df):
        raise ValueError(f"Task ID {task_id} exceeds number of conditions ({len(df)})")
    return df.iloc[task_id - 1].to_dict()

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
    variables = ['Y1', 'Y2', 'TPO1', 'TPO2', 'TMO1', 'TMO2']
    
    try:
        individual_measures = extract_individual_measures(results, variables)
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
            pairs = find_relationship_pairs(
                results, rel_path,
                output_format='long',
                generations=FINAL_GENS
            )
            
            if len(pairs) == 0:
                print(f"    {rel_path}: No pairs found")
                continue
            
            # Add measures for pairs
            pairs_with_measures = pairs.copy()
            
            for var in variables:
                pairs_with_measures[f'{var}_1'] = pairs_with_measures['Person_ID'].map(
                    lambda id_val: measures_lookup.get(id_val, {}).get(var, np.nan)
                )
                pairs_with_measures[f'{var}_2'] = pairs_with_measures['Relative_ID'].map(
                    lambda id_val: measures_lookup.get(id_val, {}).get(var, np.nan)
                )
            
            # Add PGS1 and PGS2
            for var in ['PGS1', 'PGS2']:
                pairs_with_measures[f'{var}_1'] = pairs_with_measures['Person_ID'].map(
                    lambda id_val: measures_lookup.get(id_val, {}).get(var, np.nan)
                )
                pairs_with_measures[f'{var}_2'] = pairs_with_measures['Relative_ID'].map(
                    lambda id_val: measures_lookup.get(id_val, {}).get(var, np.nan)
                )
            
            # Compute correlations
            correlation_vars = ['PGS1', 'PGS2', 'Y1', 'Y2']
            correlations = compute_correlations_for_multiple_variables(
                pairs_with_measures, correlation_vars, relationship_col='Relationship'
            )
            
            correlations['Iteration'] = iteration + 1
            correlations['RelationshipPath'] = rel_path
            
            all_correlations.append(correlations)
            print(f"    {rel_path}: {len(pairs)} pairs, {len(correlations)} correlations")
            
        except Exception as e:
            print(f"    {rel_path}: Error - {e}")
            traceback.print_exc()
            continue
    
    # Combine all correlations
    if len(all_correlations) > 0:
        correlations_df = pd.concat(all_correlations, ignore_index=True)
        cols = ['Iteration', 'RelationshipPath', 'Relationship',
                'Variable', 'N_Pairs', 'Correlation', 'P_Value']
        correlations_df = correlations_df[cols]
        return correlations_df
    
    return None

def analyze_condition(condition_name, scratch_base, project_base):
    """Analyze all iterations for a single condition."""
    
    print(f"\n{'#'*70}")
    print(f"# Analyzing {condition_name}")
    print(f"{'#'*70}\n")
    
    # Setup directories
    scratch_dir = scratch_base / condition_name
    project_dir = project_base / condition_name
    
    # Check if directories exist
    if not scratch_dir.exists():
        print(f"✗ Scratch directory not found: {scratch_dir}")
        return False
    
    if not project_dir.exists():
        print(f"✗ Project directory not found: {project_dir}")
        return False
    
    # Load condition parameters from central conditions_config.csv
    conditions_file = project_base / "conditions_config.csv"
    if not conditions_file.exists():
        print(f"✗ Conditions config file not found: {conditions_file}")
        return False
    
    conditions_df = pd.read_csv(conditions_file)
    condition_row = conditions_df[conditions_df['name'] == condition_name]
    
    if len(condition_row) == 0:
        print(f"✗ Condition {condition_name} not found in {conditions_file}")
        return False
    
    condition = condition_row.iloc[0].to_dict()
    print(f"Loaded parameters:")
    print(f"  f11={condition['f11']:.4f}, vg1={condition['vg1']:.4f}, prop_h2_latent1={condition['prop_h2_latent1']:.4f}")
    print(f"  f22={condition['f22']:.4f}, vg2={condition['vg2']:.4f}, am22={condition['am22']:.4f}")
    print(f"  rg={condition['rg']:.4f}\n")
    
    # Find all iteration directories
    iteration_dirs = sorted(scratch_dir.glob("Iteration_*"))
    print(f"Found {len(iteration_dirs)} iteration directories\n")
    
    # Storage for results
    all_correlations = []
    iteration_status = []
    
    # Process each iteration
    for iter_dir in iteration_dirs:
        # Extract iteration number from directory name
        iter_num = int(iter_dir.name.split('_')[1])
        
        iter_info = {
            'Iteration': iter_num,
            'Status': 'Failed',
            'Error': None,
            'N_Correlations': 0
        }
        
        print(f"  Processing Iteration {iter_num:02d}...")
        
        try:
            # Load saved simulation results
            print(f"    Loading simulation data from {iter_dir}")
            results = load_simulation_results(str(iter_dir), f"iteration_{iter_num:02d}")
            
            if results is None:
                iter_info['Error'] = 'Failed to load simulation data'
                print(f"    ✗ Failed to load simulation data")
                iteration_status.append(iter_info)
                continue
            
            # Extract and analyze relationships
            correlations_df = extract_and_analyze_relationships(results, iter_num - 1)
            
            if correlations_df is not None:
                all_correlations.append(correlations_df)
                iter_info['Status'] = 'Success'
                iter_info['N_Correlations'] = len(correlations_df)
                print(f"    ✓ Computed {len(correlations_df)} correlation records")
            else:
                iter_info['Error'] = 'No correlations computed'
                print(f"    ⚠ No correlations computed")
            
        except Exception as e:
            iter_info['Error'] = str(e)
            print(f"    ✗ Error: {e}")
            traceback.print_exc()
        
        iteration_status.append(iter_info)
    
    # Save iteration status summary
    status_file = project_dir / "analysis_status.csv"
    status_df = pd.DataFrame(iteration_status)
    status_df.to_csv(status_file, index=False)
    print(f"\n  ✓ Saved analysis status to {status_file}")
    
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
                col_name = f"cor_{corr_row['RelationshipPath']}_{corr_row['Variable']}"
                row[col_name] = corr_row['Correlation']
                # Also add N_Pairs for reference
                col_name_n = f"n_{corr_row['RelationshipPath']}_{corr_row['Variable']}"
                row[col_name_n] = corr_row['N_Pairs']
            
            nn_training_data.append(row)
        
        if nn_training_data:
            nn_training_file = project_dir / "nn_training_format.csv"
            nn_training_df = pd.DataFrame(nn_training_data)
            nn_training_df.to_csv(nn_training_file, index=False)
            print(f"  ✓ Saved NN training format to {nn_training_file}")
        
        return True
    else:
        print(f"\n  ⚠ No correlations computed for {condition_name}")
        print(f"  Check analysis_status.csv for details")
        return False

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Analyze simulation data for neural network training')
    parser.add_argument('--task_id', type=int, help='SLURM task ID (1-based)')
    parser.add_argument('--condition', type=str, help='Condition name (e.g., Condition_0001)')
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("DATA GENERATION FOR NEURAL NETWORK TRAINING - PART 2: ANALYSIS")
    print("="*70)
    print(f"Scratch base: {SCRATCH_BASE}")
    print(f"Project base: {PROJECT_BASE}")
    print("="*70 + "\n")
    
    # Determine which condition to analyze
    condition_name = None
    
    if args.condition:
        condition_name = args.condition
        print(f"Analyzing specified condition: {condition_name}")
    elif args.task_id:
        try:
            condition = load_condition(args.task_id)
            condition_name = condition['name']
            print(f"Analyzing condition for task ID {args.task_id}: {condition_name}")
        except Exception as e:
            print(f"Error loading condition for task {args.task_id}: {e}")
            return
    else:
        # Try to get from environment (SLURM)
        task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', '0'))
        if task_id > 0:
            try:
                condition = load_condition(task_id)
                condition_name = condition['name']
                print(f"Analyzing condition for SLURM task ID {task_id}: {condition_name}")
            except Exception as e:
                print(f"Error loading condition for SLURM task {task_id}: {e}")
                return
        else:
            print("Error: Must specify --task_id, --condition, or run via SLURM array job")
            parser.print_help()
            return
    
    # Analyze the condition
    success = analyze_condition(condition_name, SCRATCH_BASE, PROJECT_BASE)
    
    if success:
        print(f"\n{'='*70}")
        print(f"ANALYSIS FOR {condition_name} COMPLETED SUCCESSFULLY")
        print(f"{'='*70}\n")
    else:
        print(f"\n{'='*70}")
        print(f"ANALYSIS FOR {condition_name} COMPLETED WITH ERRORS")
        print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
