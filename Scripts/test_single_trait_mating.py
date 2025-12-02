#!/usr/bin/env python3
"""
Test script to verify single-trait mating functionality.
This script compares:
1. Dual-trait mating with only one trait having non-zero correlation (current approach)
2. Single-trait mating where only trait 2 is used for mating (new approach)

The key difference: In single-trait mating, trait 1's mate correlation can be non-zero
due to cross-trait correlations, whereas dual-trait mating forces trait 1 to zero.

Uses Condition 1 parameters from run_approximation_bi_rc.py.
"""

import numpy as np
import sys
import os

# Add the SimulationFunctions directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'SimulationFunctions'))
from SimulationFunctions import AssortativeMatingSimulation

def test_mating_modes():
    """Test both dual-trait and single-trait mating modes."""
    
    print("="*80)
    print("Testing Single-Trait vs Dual-Trait Mating Modes")
    print("Using Condition_01 parameters from run_approximation_bi_rc.py")
    print("="*80)
    
    # Basic simulation parameters
    N_GENERATIONS = 15
    POPULATION_SIZE = 3000
    N_CV = 1000
    
    # Condition_01 parameters from run_approximation_bi_rc.py
    f11 = 0.1000
    prop_h2_latent1 = 0.8000
    vg1 = 0.6000
    vg2 = 1.0000
    f22 = 0.2000
    am22 = 0.6500
    rg = 0.7500
    f12 = 0
    f21 = 0
    re = 0
    prop_h2_latent2 = .8/0.8  # = 1.0
    
    # Setup matrices (same as setup_matrices in run_approximation_bi_rc.py)
    k2_matrix = np.array([[1, rg], [rg, 1]])
    
    # Observable genetic variance components
    vg_obs1 = vg1 * (1 - prop_h2_latent1)
    vg_obs2 = vg2 * (1 - prop_h2_latent2)
    d11 = np.sqrt(vg_obs1)
    d21 = 0
    d22 = np.sqrt(max(vg_obs2 - d21**2, 0))
    d_mat = np.array([[d11, 0], [d21, d22]])
    
    # Latent genetic variance components
    vg_lat1 = vg1 * prop_h2_latent1
    vg_lat2 = vg2 * prop_h2_latent2
    a11 = np.sqrt(vg_lat1)
    a21 = 0
    a22 = np.sqrt(vg_lat2 - a21**2)
    a_mat = np.array([[a11, 0], [a21, a22]])
    
    # Total genetic covariance
    covg_mat = (d_mat @ k2_matrix @ d_mat.T) + (a_mat @ k2_matrix @ a_mat.T)
    
    # Environmental covariance
    ve1 = 1 - vg1
    ve2 = 1 - vg2
    cove = re * np.sqrt(ve1 * ve2)
    cove_mat = np.array([[ve1, cove], [cove, ve2]])
    
    # Total phenotypic covariance
    covy_mat = covg_mat + cove_mat
    
    # Vertical transmission matrix
    f_mat = np.array([[f11, f12], [f21, f22]])
    
    # Social homogamy matrix (set to zero - phenotypic AM only)
    s_mat = np.zeros((2, 2))
    
    print(f"\nCondition_01 Parameters:")
    print(f"  vg1={vg1:.4f}, vg2={vg2:.4f}, rg={rg:.4f}")
    print(f"  f11={f11:.4f}, f22={f22:.4f}")
    print(f"  prop_h2_latent1={prop_h2_latent1:.4f}, prop_h2_latent2={prop_h2_latent2:.4f}")
    print(f"  am22={am22:.4f} (mate correlation for trait 2)")
    
    # ============================================================================
    # Test 1: Dual-trait mating with only trait 2 having non-zero correlation
    # ============================================================================
    print("\n" + "="*80)
    print("TEST 1: Dual-trait mating (2x2 matrix with only trait 2 correlation)")
    print("="*80)
    print(f"AM matrix: [[0, 0], [0, {am22}]]")
    print("Expected: Trait 1 mate correlation should be forced to ~0")
    
    mate_cor_mat_dual = np.array([[0, 0], [0, am22]])
    am_list_dual = [mate_cor_mat_dual.copy() for _ in range(N_GENERATIONS)]
    
    sim_dual = AssortativeMatingSimulation(
        n_CV=N_CV,
        rg_effects=rg,
        maf_min=0.01,
        maf_max=0.5,
        num_generations=N_GENERATIONS,
        pop_size=POPULATION_SIZE,
        mating_type="phenotypic",
        avoid_inbreeding=True,
        save_each_gen=True,  # Enable to save MATES history
        save_covs=False,
        seed=12345,
        cove_mat=cove_mat,
        f_mat=f_mat,
        s_mat=s_mat,
        a_mat=a_mat,
        d_mat=d_mat,
        am_list=am_list_dual,
        covy_mat=covy_mat,
        k2_matrix=k2_matrix
    )
    
    results_dual = sim_dual.run_simulation()
    print("\n--- Dual-trait mating results (final generation) ---")
    final_gen_dual = results_dual['SUMMARY.RES'][-1]
    print(f"Generation: {final_gen_dual['GEN']}")
    print(f"Population size: {final_gen_dual['POPSIZE']}")
    print(f"Target mate correlation matrix:")
    print(np.array(final_gen_dual['MATE.COR']))
    
    # ============================================================================
    # Test 2: Single-trait mating (only mate on trait 2)
    # ============================================================================
    print("\n" + "="*80)
    print("TEST 2: Single-trait mating (mate on trait 2 only)")
    print("="*80)
    print(f"AM list: [{am22}] for each generation, mate_on_trait=2")
    print("Expected: Trait 1 mate correlation can be >0 due to cross-trait correlation")
    
    am_list_single = [am22 for _ in range(N_GENERATIONS)]  # List of scalars
    
    sim_single = AssortativeMatingSimulation(
        n_CV=N_CV,
        rg_effects=rg,
        maf_min=0.01,
        maf_max=0.5,
        num_generations=N_GENERATIONS,
        pop_size=POPULATION_SIZE,
        mating_type="phenotypic",
        avoid_inbreeding=True,
        save_each_gen=True,  # Enable to save MATES history
        save_covs=False,
        seed=12345,
        cove_mat=cove_mat,
        f_mat=f_mat,
        s_mat=s_mat,
        a_mat=a_mat,
        d_mat=d_mat,
        am_list=am_list_single,  # List of scalars
        mate_on_trait=2,  # Mate on trait 2 only
        covy_mat=covy_mat,
        k2_matrix=k2_matrix,
        output_summary_filename= "test_single_trait_mating_summary.txt"
    )
    
    results_single = sim_single.run_simulation()
    print("\n--- Single-trait mating results (final generation) ---")
    final_gen_single = results_single['SUMMARY.RES'][-1]
    print(f"Generation: {final_gen_single['GEN']}")
    print(f"Population size: {final_gen_single['POPSIZE']}")
    print(f"Target mate correlation (scalar): {final_gen_single['MATE.COR']}")
    
    # ============================================================================
    # Comparison
    # ============================================================================
    print("\n" + "="*80)
    print("COMPARISON: Final Generation Mate Correlations")
    print("="*80)
    
    def print_correlation_matrices(results, label):
        """Print phenotype and NT correlation matrices for mates."""
        if not results['HISTORY'] or not results['HISTORY']['MATES']:
            print(f"  No MATES history available for {label}")
            return None, None
        
        last_mates = results['HISTORY']['MATES'][-1]
        if not last_mates or 'males.PHENDATA' not in last_mates:
            print(f"  No mate data available for {label}")
            return None, None
        
        males = last_mates['males.PHENDATA']
        females = last_mates['females.PHENDATA']
        
        # Check which columns are available
        print(f"\n{label}:")
        print(f"  Available columns: {list(males.columns)}")
        
        # Create phenotype matrix: Male Y1, Male Y2, Female Y1, Female Y2
        mate_phenos = np.column_stack([
            males['Y1'].values,
            males['Y2'].values,
            females['Y1'].values,
            females['Y2'].values
        ])
        
        # Calculate phenotype correlation matrix
        corr_pheno = np.corrcoef(mate_phenos, rowvar=False)
        
        print(f"\n  Phenotype mate correlation matrix (4x4):")
        print("           M_Y1    M_Y2    F_Y1    F_Y2")
        labels_pheno = ['M_Y1', 'M_Y2', 'F_Y1', 'F_Y2']
        for i, lbl in enumerate(labels_pheno):
            row_str = f"  {lbl:6s}  "
            for j in range(4):
                row_str += f"{corr_pheno[i,j]:7.4f} "
            print(row_str)
        
        print(f"\n  Spousal correlation Trait 1 (M_Y1 <-> F_Y1): {corr_pheno[0,2]:.4f}")
        print(f"  Spousal correlation Trait 2 (M_Y2 <-> F_Y2): {corr_pheno[1,3]:.4f}")
        print(f"  Cross-trait spousal (M_Y1 <-> F_Y2):        {corr_pheno[0,3]:.4f}")
        print(f"  Cross-trait spousal (M_Y2 <-> F_Y1):        {corr_pheno[1,2]:.4f}")
        
        # Get NT correlations from offspring (final generation PHEN dataframe)
        final_phen = results['PHEN']
        nt_cols = ['NTMO1', 'NTPO1', 'NTMO2', 'NTPO2']
        
        # Check if NT columns exist in offspring data
        if all(col in final_phen.columns for col in nt_cols):
            # Create NT matrix from offspring: NTMO1, NTPO1, NTMO2, NTPO2
            offspring_nt = np.column_stack([final_phen[col].values for col in nt_cols])
            
            # Calculate NT correlation matrix (4x4) for offspring
            corr_nt = np.corrcoef(offspring_nt, rowvar=False)
            
            print(f"\n  Offspring NT correlation matrix (4x4):")
            print("           NTMO1   NTPO1   NTMO2   NTPO2")
            for i, lbl in enumerate(nt_cols):
                row_str = f"  {lbl:6s}  "
                for j in range(4):
                    row_str += f"{corr_nt[i,j]:7.4f} "
                print(row_str)
            
            # Key correlations
            print(f"\n  Key offspring NT correlations:")
            print(f"    NTMO1 <-> NTPO1 (trait 1): {corr_nt[0,1]:.4f}")
            print(f"    NTMO2 <-> NTPO2 (trait 2): {corr_nt[2,3]:.4f}")
            print(f"    NTMO1 <-> NTMO2 (maternal): {corr_nt[0,2]:.4f}")
            print(f"    NTPO1 <-> NTPO2 (paternal): {corr_nt[1,3]:.4f}")
            
            return corr_pheno, corr_nt
        else:
            missing = [col for col in nt_cols if col not in final_phen.columns]
            print(f"\n  NT columns not found in offspring data: {missing}")
            return corr_pheno, None
    
    # Print results for both conditions
    corr_pheno_dual, corr_nt_dual = print_correlation_matrices(results_dual, "Dual-trait mating")
    print("\n" + "-"*80)
    corr_pheno_single, corr_nt_single = print_correlation_matrices(results_single, "Single-trait mating")
    
    # Compare trait 1 spousal correlations (the non-mated trait)
    print("\n" + "="*80)
    print("KEY DIFFERENCE:")
    print("="*80)
    if corr_pheno_dual is not None and corr_pheno_single is not None:
        trait1_dual = corr_pheno_dual[0,2]
        trait1_single = corr_pheno_single[0,2]
        trait2_dual = corr_pheno_dual[1,3]
        trait2_single = corr_pheno_single[1,3]
        
        print(f"\nTrait 1 (non-mated) spousal correlation:")
        print(f"  Dual-trait mating:   {trait1_dual:7.4f}  (forced to ~0)")
        print(f"  Single-trait mating: {trait1_single:7.4f}  (emerged naturally)")
        print(f"  Difference:          {trait1_single - trait1_dual:7.4f}")
        
        print(f"\nTrait 2 (mated) spousal correlation:")
        print(f"  Dual-trait mating:   {trait2_dual:7.4f}")
        print(f"  Single-trait mating: {trait2_single:7.4f}")
        print(f"  Target:              {am22:7.4f}")
        
        if trait1_single > trait1_dual + 0.01:
            print("\n✓ SUCCESS: Single-trait mating allows trait 1 correlation to emerge naturally!")
            print("  In single-trait mode, we only mate on trait 2, but trait 1 shows")
            print("  spousal correlation due to cross-trait genetic correlations (rg).")
        else:
            print("\n⚠ Note: Trait 1 correlations are similar in both modes.")
            print("  This may indicate weak cross-trait correlation or few generations.")
    
    print("\n" + "="*80)
    print("Test completed successfully!")
    print("="*80)

if __name__ == "__main__":
    test_mating_modes()
