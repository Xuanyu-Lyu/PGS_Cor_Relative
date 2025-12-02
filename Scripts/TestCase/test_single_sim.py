#!/usr/bin/env python3

import numpy as np
import os
import sys

# Add the parent directory to the path to import SimulationFunctions
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from SimulationFunctions.SimulationFunctions import AssortativeMatingSimulation
from SimulationFunctions.save_simulation_data import save_simulation_results

def main():
    """
    Run a single simulation with specified parameters:
    - s matrix with non-zero values but all less than 0.4
    - f matrix set to zero
    - Save all generations' data using save_simulation_data
    """
    
    # Base simulation parameters
    base_params = {
        "num_generations": 20,     # Number of generations to simulate
        "pop_size": 3000,          # Population size (smaller for testing)
        "n_CV": 300,               # Number of causal variants
        "rg_effects": 0,         # Genetic correlation between effects
        "maf_min": 0.25,           # Minimum minor allele frequency
        "maf_max": 0.45,           # Maximum minor allele frequency
        "avoid_inbreeding": True,
        "save_each_gen": True,     # Save all generations' data for output
        "save_covs": True,         # Save covariances
        "summary_file_scope": "all",  # Save all generation summaries
        "seed": 12345
    }
    
    # Define the matrices as per your requirements
    
    # k2_matrix: genetic correlation matrix
    k2_val = np.array([[1.0, base_params["rg_effects"]], 
                       [base_params["rg_effects"], 1.0]])
    
    # d_mat: environmental variance matrix (diagonal) - using sqrt of variance for scaling
    d_mat_val = np.diag([np.sqrt(.3), np.sqrt(.2)])
    
    # a_mat: additive genetic variance matrix (diagonal) - using sqrt of variance for scaling
    a_mat_val = np.diag([np.sqrt(.5), np.sqrt(.6)])
    
    # f_mat: ZERO matrix as requested
    f_mat_val = np.array([[.4, 0.0], 
                          [0.0, .5]])
    
    # s_mat: NON-ZERO values but all less than 0.4 as requested
    s_mat_val = np.array([[0, 0], 
                          [0, 0]]) 
    
    # cove_mat: environmental covariance matrix
    cove_val = np.array([[0.2, 0], 
                         [0, 0.2]])
    
    # covy_mat: phenotypic covariance matrix
    covy_val = np.array([[1.0, 0], 
                         [0, 1.0]])
    
    # am_list: assortative mating correlation matrices for each generation
    am_correlation = np.array([[0.43, 0], 
                               [0, 0.43]])
    am_list_val = [am_correlation] * base_params["num_generations"]
    
    # Update base parameters with matrices
    base_params.update({
        "k2_matrix": k2_val,
        "d_mat": d_mat_val,
        "a_mat": a_mat_val,
        "f_mat": f_mat_val,
        "s_mat": s_mat_val,
        "cove_mat": cove_val,
        "covy_mat": covy_val,
        "am_list": am_list_val,
        "mating_type": "phenotypic"  # Type of assortative mating
    })
    
    # Set up output file
    output_dir = "/Users/xuly4739/Library/CloudStorage/OneDrive-UCB-O365/Documents/coding/PyProject/PGS_Cor_Relative/Data/TestCase"
    summary_filename = os.path.join(output_dir, "test_simulation_summary.txt")
    base_params["output_summary_filename"] = summary_filename
    
    print("Starting simulation with parameters:")
    print(f"- Generations: {base_params['num_generations']}")
    print(f"- Population size: {base_params['pop_size']}")
    print(f"- Number of causal variants: {base_params['n_CV']}")
    print(f"- Genetic correlation (rg): {base_params['rg_effects']}")
    print(f"- Mating type: {base_params['mating_type']}")
    print(f"- Save each generation: {base_params['save_each_gen']}")
    print(f"- Summary scope: {base_params['summary_file_scope']}")
    print(f"- Output file: {summary_filename}")
    # print(f"- F matrix (should be zeros):\n{f_mat_val}")
    # print(f"- S matrix (non-zero, all < 0.4):\n{s_mat_val}")
    # print(f"- D matrix (environmental scaling):\n{d_mat_val}")
    # print(f"- A matrix (genetic scaling):\n{a_mat_val}")
    print()
    
    try:
        # Create and run the simulation
        sim = AssortativeMatingSimulation(**base_params)
        
        print("Simulation initialized successfully!")
        print("Running simulation...")
        
        # Run the simulation
        results = sim.run_simulation()
        
        if results is not None:
            print("\nSimulation completed successfully!")
            print(f"Summary saved to: {summary_filename}")
            print(f"Final generation population size: {len(results['PHEN'])}")
            print(f"Number of summary results: {len(results['SUMMARY.RES'])}")
            
            # Print some basic statistics from the final generation
            final_summary = results['SUMMARY.RES'][-1]
            print(f"\nFinal generation statistics:")
            print(f"- Generation: {final_summary['GEN']}")
            print(f"- Population size: {final_summary['POPSIZE']}")
            print(f"- Heritability (h2): {final_summary.get('h2', 'N/A')}")
            
            # Save all simulation data for all generations
            print("\n--- Saving all generation data to TSV files ---")
            out_put_dir_data = os.path.join(output_dir, "All_Generations_Data")
            # check and create directory if not exists
            if not os.path.exists(out_put_dir_data):
                os.makedirs(out_put_dir_data)
            save_simulation_results(
                results=results,
                output_folder=out_put_dir_data,
                file_prefix="test_sim",
                scope="all"  # Save all generations
            )
            print("All data files saved successfully!")
            
        else:
            print("Simulation failed to complete.")
            
    except Exception as e:
        print(f"Error running simulation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

