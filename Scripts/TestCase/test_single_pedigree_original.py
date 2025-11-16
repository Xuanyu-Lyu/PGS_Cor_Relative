#!/usr/bin/env python3

import numpy as np
import os
import sys

# Add the parent directory to the path to import SimulationFunctions
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from SimulationFunctions.SimulationFunctions import AssortativeMatingSimulation
from SimulationFunctions.save_simulation_data import save_simulation_results
from SimulationFunctions.find_relative import extract_genealogy_info, find_relationship_pairs, save_sparse_relationship_matrix, load_sparse_relationship_matrix, sparse_matrix_to_long_format, get_matrix_memory_info
from SimulationFunctions.extract_measures import extract_individual_measures, extract_measures_for_pairs, compute_correlations_for_multiple_variables, save_measures_to_file

def main():
    """
    Run a single simulation with specified parameters:
    - s matrix with non-zero values but all less than 0.4
    - f matrix set to zero
    - Save all generations' data using save_simulation_data
    """
    
    # Base simulation parameters
    base_params = {
        "num_generations": 15,     # Number of generations to simulate
        "pop_size": 500,          # Population size (smaller for testing)
        "n_CV": 100,               # Number of causal variants
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
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
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
            # print("\n" + "="*70)
            # print("SIMULATION COMPLETED SUCCESSFULLY!")
            # print("="*70)
            # print(f"Summary saved to: {summary_filename}")
            # print(f"Final generation population size: {len(results['PHEN'])}")
            # print(f"Number of summary results: {len(results['SUMMARY.RES'])}")
            
            # # Print some basic statistics from the final generation
            # final_summary = results['SUMMARY.RES'][-1]
            # print(f"\nFinal generation statistics:")
            # print(f"- Generation: {final_summary['GEN']}")
            # print(f"- Population size: {final_summary['POPSIZE']}")
            # print(f"- Heritability (h2): {final_summary.get('h2', 'N/A')}")
            
            # # Save all simulation data for all generations
            # print("\n" + "="*70)
            # print("SAVING SIMULATION DATA")
            # print("="*70)
            # out_put_dir_data = os.path.join(output_dir, "All_Generations_Data")
            # # check and create directory if not exists
            # if not os.path.exists(out_put_dir_data):
            #     os.makedirs(out_put_dir_data)
            # save_simulation_results(
            #     results=results,
            #     output_folder=out_put_dir_data,
            #     file_prefix="test_sim",
            #     scope="all"  # Save all generations
            # )
            # print("All data files saved successfully!")
            
            # ===================================================================
            # TEST GENEALOGY EXTRACTION AND RELATIONSHIP FINDING
            # ===================================================================
            print("\n" + "="*70)
            print("TESTING GENEALOGY AND RELATIONSHIP FUNCTIONS")
            print("="*70)
            
            # Test 1: Extract genealogy info
            print("\n--- Test 1: Extract Genealogy Info ---")
            print("Extracting genealogy for all generations...")
            genealogy_all = extract_genealogy_info(results)
            print(f"✓ Total individuals extracted: {len(genealogy_all):,}")
            print(f"✓ Generations covered: {genealogy_all['Generation'].min()} to {genealogy_all['Generation'].max()}")
            print(f"✓ Sample data:\n{genealogy_all.head()}")
            
            # Test 2: Get genealogy summary
            print("\n--- Test 2: Genealogy Summary Statistics ---")
            from SimulationFunctions.find_relative import get_genealogy_summary
            summary = get_genealogy_summary(genealogy_all)
            print(f"✓ Total individuals: {summary['total_individuals']:,}")
            print(f"✓ Mated individuals: {summary['mated_individuals']:,}")
            print(f"✓ Individuals per generation (first 5):")
            for gen, count in list(summary['individuals_per_generation'].items())[:5]:
                print(f"  Generation {gen}: {count:,}")
            
            # Test 3: Check memory requirements
            print("\n--- Test 3: Memory Requirements Check ---")
            mem_info = get_matrix_memory_info(results)
            
            # # Test 4: Find siblings
            # print("\n--- Test 4: Find Siblings (All Generations) ---")
            # print("Finding all sibling pairs using path 'S' with 6 CPU cores...")
            # siblings = find_relationship_pairs(results, "S", output_format='long', n_jobs=6, generations=[11,12,13,14,15])
            # print(f"✓ Found {len(siblings):,} sibling pairs")
            # if len(siblings) > 0:
            #     print(f"✓ Sample pairs:\n{siblings.head()}")
            
            # Test 5: Find first cousins (PSC)
            print("\n--- Test 5: Find First Cousins (PSC) - All Generations ---")
            print("Finding all first cousin pairs using path 'PSC' with 4 CPU cores...")
            cousins = find_relationship_pairs(results, "PSC", output_format='long', n_jobs=6, generations=[11,12,13,14,15])
            print(f"✓ Found {len(cousins):,} first cousin pairs")
            if len(cousins) > 0:
                print(f"✓ Sample pairs:\n{cousins.head()}")
            
            # # Test 6: Find aunts/uncles (PS)
            # print("\n--- Test 6: Find Aunts/Uncles (PS) - All Generations ---")
            # print("Finding all aunt/uncle relationships using path 'PS' with 4 CPU cores...")
            # aunts_uncles = find_relationship_pairs(results, "PS", output_format='long', n_jobs=6, generations=[11,12,13,14,15])
            # print(f"✓ Found {len(aunts_uncles):,} aunt/uncle pairs")
            # if len(aunts_uncles) > 0:
            #     print(f"✓ Sample pairs:\n{aunts_uncles.head()}")
            
            # # Test 7: Test sparse matrix functionality
            # print("\n--- Test 7: Sparse Matrix Functionality ---")
            # print("Generating sparse matrix for cousins (PSC) - all generations with 4 CPU cores...")
            # sparse_mat, ids = find_relationship_pairs(
            #     results, "PSC", output_format='sparse', n_jobs=6, generations=[11,12,13,14,15]
            # )
            # print(f"✓ Sparse matrix shape: {sparse_mat.shape}")
            # print(f"✓ Non-zero elements: {sparse_mat.nnz:,}")
            # print(f"✓ Number of IDs: {len(ids):,}")
            
            # # Test 8: Save and load sparse matrix
            # print("\n--- Test 8: Save and Load Sparse Matrix ---")
            # sparse_output_dir = os.path.join(output_dir, "Sparse_Matrices")
            # if not os.path.exists(sparse_output_dir):
            #     os.makedirs(sparse_output_dir)
            
            # sparse_filepath = os.path.join(sparse_output_dir, "cousins_all_gens")
            # print(f"Saving sparse matrix to: {sparse_filepath}")
            # save_sparse_relationship_matrix(sparse_mat, ids, sparse_filepath, "PSC")
            
            # print("\nLoading sparse matrix back...")
            # loaded_sparse, loaded_ids, loaded_rel = load_sparse_relationship_matrix(sparse_filepath)
            # print(f"✓ Matrix loaded successfully")
            # print(f"✓ Shapes match: {loaded_sparse.shape == sparse_mat.shape}")
            # print(f"✓ Non-zero elements match: {loaded_sparse.nnz == sparse_mat.nnz}")
            
            # # Test 9: Convert sparse to long format
            # print("\n--- Test 9: Convert Sparse Matrix to Long Format ---")
            # pairs_from_sparse = sparse_matrix_to_long_format(loaded_sparse, loaded_ids, loaded_rel)
            # print(f"✓ Converted to {len(pairs_from_sparse):,} pairs")
            # print(f"✓ Matches original: {len(pairs_from_sparse) == len(cousins)}")
            # if len(pairs_from_sparse) > 0:
            #     print(f"✓ Sample:\n{pairs_from_sparse.head()}")
            
            # Test 10: Test multiple relationship types
            print("\n--- Test 10: Multiple Relationship Types (All Generations) ---")
            relationship_types = {
                "P": "Parents",
                "C": "Children",
                "M": "Spouses",
                "PSCC": "First Cousin Once Removed",
                "PPMSC": "test"
            }
            
            for rel_path, rel_name in relationship_types.items():
                try:
                    rel_pairs = find_relationship_pairs(
                        results, rel_path, output_format='long', n_jobs=6, generations=[11,12,13,14,15]
                    )
                    print(f"✓ {rel_name} ({rel_path}): {len(rel_pairs):,} pairs")
                except Exception as e:
                    print(f"✗ {rel_name} ({rel_path}): Error - {e}")
            
            # ===================================================================
            # TEST MEASURE EXTRACTION AND CORRELATION COMPUTATION
            # ===================================================================
            print("\n" + "="*70)
            print("TESTING MEASURE EXTRACTION AND POLYGENIC SCORE CORRELATIONS")
            print("="*70)
            
            # Define variables to extract
            variables_to_extract = ['Y1', 'Y2', 'TPO1', 'TPO2', 'TMO1', 'TMO2']
            
            # Test 11: Extract individual measures
            print("\n--- Test 11: Extract Individual Measures ---")
            print(f"Extracting variables: {variables_to_extract}")
            print("Note: Extracting from generations 1-15 (generation 0 founders don't have parent-based polygenic scores)")
            individual_measures = extract_individual_measures(results, variables_to_extract, generations=list(range(1, 16)))
            print(f"✓ Extracted measures for {len(individual_measures):,} individuals")
            print(f"✓ Columns: {individual_measures.columns.tolist()}")
            print(f"✓ Sample data:\n{individual_measures.head()}")
            
            # Save individual measures to file
            measures_output_path = os.path.join(output_dir, "individual_measures.tsv")
            save_measures_to_file(individual_measures, measures_output_path)
            
            # # Test 12: Extract measures for sibling pairs
            # print("\n--- Test 12: Extract Measures for Sibling Pairs ---")
            # print("Extracting measures for sibling pairs...")
            # siblings_with_measures = extract_measures_for_pairs(results, siblings, variables_to_extract)
            # print(f"✓ Extracted measures for {len(siblings_with_measures):,} sibling pairs")
            # print(f"✓ Columns: {siblings_with_measures.columns.tolist()}")
            # print(f"✓ Sample data:\n{siblings_with_measures.head()}")
            
            # # Save sibling pairs with measures
            # siblings_output_path = os.path.join(output_dir, "siblings_with_measures.tsv")
            # save_measures_to_file(siblings_with_measures, siblings_output_path)
            
            # # Test 13: Compute correlations for siblings
            # print("\n--- Test 13: Compute Polygenic Score Correlations for Siblings ---")
            # sibling_correlations = compute_correlations_for_multiple_variables(
            #     siblings_with_measures, variables_to_extract, relationship_col='Relationship'
            # )
            # print(f"✓ Computed correlations for {len(variables_to_extract)} variables")
            # print("\nSibling Correlations:")
            # print(sibling_correlations.to_string(index=False))
            
            # # Save correlations
            # corr_output_path = os.path.join(output_dir, "sibling_correlations.tsv")
            # save_measures_to_file(sibling_correlations, corr_output_path)
            
            # Test 14: Extract measures for cousin pairs
            print("\n--- Test 14: Extract Measures for Cousin Pairs ---")
            print("Extracting measures for first cousin pairs...")
            cousins_with_measures = extract_measures_for_pairs(results, cousins, variables_to_extract)
            print(f"✓ Extracted measures for {len(cousins_with_measures):,} cousin pairs")
            
            # Save cousin pairs with measures
            cousins_output_path = os.path.join(output_dir, "cousins_with_measures.tsv")
            save_measures_to_file(cousins_with_measures, cousins_output_path)
            
            # Compute correlations for cousins
            cousin_correlations = compute_correlations_for_multiple_variables(
                cousins_with_measures, variables_to_extract, relationship_col='Relationship'
            )
            print("\nCousin Correlations:")
            print(cousin_correlations.to_string(index=False))
            
            # Save cousin correlations
            cousin_corr_output_path = os.path.join(output_dir, "cousin_correlations.tsv")
            save_measures_to_file(cousin_correlations, cousin_corr_output_path)
            
            # Summary of all tests
            print("\n" + "="*70)
            print("ALL TESTS COMPLETED SUCCESSFULLY!")
            print("="*70)
            print(f"\nFiles saved in: {output_dir}")
            print("\nOutput files:")
            print(f"  - individual_measures.tsv: All individual measures")
            print(f"  - siblings_with_measures.tsv: Sibling pairs with measures")
            print(f"  - sibling_correlations.tsv: Correlations for siblings")
            print(f"  - cousins_with_measures.tsv: Cousin pairs with measures")
            print(f"  - cousin_correlations.tsv: Correlations for cousins")
            
        else:
            print("Simulation failed to complete.")
            
    except Exception as e:
        print(f"Error running simulation or tests: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
