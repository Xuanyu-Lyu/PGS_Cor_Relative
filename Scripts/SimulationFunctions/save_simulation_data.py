# File: save_simulation_data.py
# Or append to SimulationFunctions.py

import os
import pandas as pd
import numpy as np

def _ensure_folder_exists(folder_path):
    """Helper function to create a folder if it doesn't exist."""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Created output folder: {folder_path}")

def _save_dataframe_to_tsv(df, folder_path, filename):
    """Saves a pandas DataFrame to a tab-separated values (TSV) file."""
    if df is not None and not df.empty:
        filepath = os.path.join(folder_path, filename)
        df.to_csv(filepath, sep='\t', index=False, na_rep='NA')
        print(f"Saved DataFrame to: {filepath}")
    else:
        print(f"Skipping empty or None DataFrame for: {filename}")

def _save_numpy_array_to_tsv(arr, folder_path, filename, header=None):
    """Saves a NumPy array to a tab-separated values (TSV) file."""
    if arr is not None and arr.size > 0:
        filepath = os.path.join(folder_path, filename)
        if header and len(header) == arr.shape[1]:
            pd.DataFrame(arr, columns=header).to_csv(filepath, sep='\t', index=False, na_rep='NA')
        else:
            np.savetxt(filepath, arr, delimiter='\t', fmt='%s', header=header if isinstance(header, str) else "", comments='')
        print(f"Saved NumPy array to: {filepath}")
    else:
        print(f"Skipping empty or None NumPy array for: {filename}")


def _save_mates_data_to_tsv(mates_data_list, folder_path, base_filename_prefix, scope):
    """
    Saves mates data. Can save a specific generation, the final generation, or all.
    """
    if not mates_data_list:
        print(f"No MATES data in history to save for prefix: {base_filename_prefix}")
        return

    if isinstance(scope, int):
        gen_index = scope
        if not (0 <= gen_index < len(mates_data_list)):
            print(f"Warning: Mates data for generation {gen_index} not found in history.")
            return

        mates_dict = mates_data_list[gen_index]
        if mates_dict: 
            _save_dataframe_to_tsv(mates_dict.get('males.PHENDATA'), folder_path, f"{base_filename_prefix}_gen{gen_index}_males.tsv")
            _save_dataframe_to_tsv(mates_dict.get('females.PHENDATA'), folder_path, f"{base_filename_prefix}_gen{gen_index}_females.tsv")
        else:
            print(f"No mates were formed in generation {gen_index}, nothing to save.")
    
    elif scope == "final":
        last_mates_dict = mates_data_list[-1] if mates_data_list else None
        if last_mates_dict is not None:
            gen_index = len(mates_data_list) - 1 
            _save_dataframe_to_tsv(last_mates_dict.get('males.PHENDATA'), folder_path, f"{base_filename_prefix}_gen{gen_index}_males.tsv")
            _save_dataframe_to_tsv(last_mates_dict.get('females.PHENDATA'), folder_path, f"{base_filename_prefix}_gen{gen_index}_females.tsv")
        else:
            print(f"No MATES data for the final generation to save for prefix: {base_filename_prefix}")
    
    elif scope == "all":
        for gen_index, mates_dict in enumerate(mates_data_list):
            if mates_dict:
                _save_dataframe_to_tsv(mates_dict.get('males.PHENDATA'), folder_path, f"{base_filename_prefix}_gen{gen_index}_males.tsv")
                _save_dataframe_to_tsv(mates_dict.get('females.PHENDATA'), folder_path, f"{base_filename_prefix}_gen{gen_index}_females.tsv")
    else:
        # This part of the helper should not be reached if main function validates scope, but good for safety.
        print(f"Invalid scope '{scope}' for saving MATES data.")


def save_simulation_results(results, output_folder, file_prefix="sim_output", scope="final"):
    """
    Saves the main components of the simulation results to TSV files.

    Args:
        results (dict): The dictionary returned by AssortativeMatingSimulation.run_simulation().
        output_folder (str): Path to the folder where files will be saved.
        file_prefix (str): Prefix for all output filenames.
        scope (str or int or list): "final" to save only the last generation's data.
                                    "all" to save data for all generations from history.
                                    An integer (e.g., 2) to save data for that specific generation.
                                    A list of integers (e.g., [0, 2, 4]) to save for those specific generations.
    """
    if not results:
        print("No results to save.")
        return

    _ensure_folder_exists(output_folder)
    print(f"\n--- Saving Simulation Output to Folder: {output_folder} ---")
    print(f"File Prefix: {file_prefix}, Scope: {scope}")

    # 1. Save summary and covariance logs (these always contain all generations)
    if results.get('SUMMARY.RES'):
        summary_df = pd.DataFrame(results['SUMMARY.RES'])
        _save_dataframe_to_tsv(summary_df, output_folder, f"{file_prefix}_summary_all_generations.tsv")
    if results.get('COVARIANCES'):
        try:
            covariances_df = pd.DataFrame([c if isinstance(c, dict) else {'cov_data': c} for c in results['COVARIANCES']])
            _save_dataframe_to_tsv(covariances_df, output_folder, f"{file_prefix}_covariances_log.tsv")
        except Exception as e:
            print(f"Could not save COVARIANCES as a single TSV due to structure: {e}.")

    # 2. Save PHEN, XO, XL, MATES based on the specified scope
    history_available = results.get('HISTORY') is not None and results['HISTORY'].get('PHEN')

    # --- Logic for saving one or more specific generations ---
    if isinstance(scope, int) or (isinstance(scope, list) and all(isinstance(i, int) for i in scope)):
        # Standardize to a list of generations to save
        generations_to_save = [scope] if isinstance(scope, int) else sorted(list(set(scope)))
        
        print(f"Saving data for specific generation(s): {generations_to_save}...")

        if not history_available:
            print(f"Error: Cannot save specific generations. History was not saved during simulation "
                  f"(parameter 'save_each_gen' must be True).")
            return

        num_generations_in_history = len(results['HISTORY']['PHEN'])
        for gen_to_save in generations_to_save:
            if not (0 <= gen_to_save < num_generations_in_history):
                print(f"Warning: Invalid generation number {gen_to_save}. Skipping. "
                      f"Available generations in history: 0 to {num_generations_in_history - 1}.")
                continue
            
            print(f"  -> Saving generation {gen_to_save}...")
            _save_dataframe_to_tsv(results['HISTORY']['PHEN'][gen_to_save], output_folder, f"{file_prefix}_phen_gen{gen_to_save}.tsv")
            _save_numpy_array_to_tsv(results['HISTORY']['XO'][gen_to_save], output_folder, f"{file_prefix}_xo_gen{gen_to_save}.tsv")
            _save_numpy_array_to_tsv(results['HISTORY']['XL'][gen_to_save], output_folder, f"{file_prefix}_xl_gen{gen_to_save}.tsv")
            
            mates_history = results['HISTORY'].get('MATES', [])
            _save_mates_data_to_tsv(mates_history, output_folder, f"{file_prefix}_mates", scope=gen_to_save)

    # --- Logic for saving only the final generation's data ---
    elif scope == "final":
        print("Saving final generation data...")
        _save_dataframe_to_tsv(results.get('PHEN'), output_folder, f"{file_prefix}_phen_final_gen.tsv")
        _save_numpy_array_to_tsv(results.get('XO'), output_folder, f"{file_prefix}_xo_final_gen.tsv")
        _save_numpy_array_to_tsv(results.get('XL'), output_folder, f"{file_prefix}_xl_final_gen.tsv")
        if history_available and results['HISTORY'].get('MATES'):
            _save_mates_data_to_tsv(results['HISTORY']['MATES'], output_folder, f"{file_prefix}_mates", scope="final")

    # --- Logic for saving data for all generations ---
    elif scope == "all":
        if history_available:
            print("Saving data for all generations from history...")
            for i, phen_df in enumerate(results['HISTORY'].get('PHEN', [])):
                _save_dataframe_to_tsv(phen_df, output_folder, f"{file_prefix}_phen_gen{i}.tsv")
            for i, xo_arr in enumerate(results['HISTORY'].get('XO', [])):
                _save_numpy_array_to_tsv(xo_arr, output_folder, f"{file_prefix}_xo_gen{i}.tsv")
            for i, xl_arr in enumerate(results['HISTORY'].get('XL', [])):
                _save_numpy_array_to_tsv(xl_arr, output_folder, f"{file_prefix}_xl_gen{i}.tsv")
            _save_mates_data_to_tsv(results['HISTORY'].get('MATES', []), output_folder, f"{file_prefix}_mates", scope="all")
        else:
            print("Scope is 'all', but no history was saved during simulation. Saving final generation data only.")
            _save_dataframe_to_tsv(results.get('PHEN'), output_folder, f"{file_prefix}_phen_final_gen.tsv")
            _save_numpy_array_to_tsv(results.get('XO'), output_folder, f"{file_prefix}_xo_final_gen.tsv")
            _save_numpy_array_to_tsv(results.get('XL'), output_folder, f"{file_prefix}_xl_final_gen.tsv")
    
    else:
        print(f"Error: Invalid scope '{scope}'. Choose 'final', 'all', or an integer/list of integers.")

    print(f"--- Finished Saving Simulation Output ---")