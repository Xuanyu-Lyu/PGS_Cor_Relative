import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.linalg import svd as scipy_svd # Or numpy.linalg.svd
import datetime

def is_even(x: int) -> bool:
    """Checks if an integer is even."""
    return x % 2 == 0

# R: is.odd <- function(x) {x%%2 != 0}
def is_odd(x: int) -> bool:
    """Checks if an integer is odd."""
    return x % 2 != 0

def next_smallest(x_list: list, value_thresh: float, return_index: bool = True):
    """
    Finds the largest element in x_list that is smaller than value_thresh.
    This is equivalent to finding the element in the sorted version of x_list
    at the highest possible index i such that x_sorted[i] < value_thresh.

    Args:
        x_list: A list of numbers.
        value_thresh: The threshold value.
        return_index: If True, returns the 0-based index of this element
                      in the sorted version of x_list.
                      If False, returns the value of this element.

    Returns:
        The 0-based index or the value, or None if no element in x_list is smaller than value_thresh.
    """
    # Sort a copy of the input list.
    x2_sorted = sorted(list(x_list))

    # Find 0-based indices of elements in x2_sorted that are less than value_thresh
    potential_indices = [i for i, val in enumerate(x2_sorted) if val < value_thresh]

    if not potential_indices:
        # R's `max(which(condition_is_all_false))` results in -Inf.
        # Accessing an array with -Inf index would be an error in R.
        # Pythonic behavior is to return None if no such element is found.
        return None

    # The R equivalent `max(which(x2<value))` gives the largest 1-based index.
    # We use the largest 0-based index here.
    actual_index_in_x2_sorted = max(potential_indices)

    if return_index:
        return actual_index_in_x2_sorted
    else:
        return x2_sorted[actual_index_in_x2_sorted]

def cor2cov(X, var1: float, var2: float):
    """
    Converts a 2x2 correlation matrix X to a covariance matrix using specified variances.

    Args:
        X: A 2x2 matrix-like object (e.g., list of lists or numpy array)
           representing the correlation matrix.
        var1: Variance of the first variable. Must be non-negative.
        var2: Variance of the second variable. Must be non-negative.

    Returns:
        A 2x2 numpy array representing the covariance matrix.

    Raises:
        ValueError: If X is not a 2x2 matrix or if variances are negative.
    """
    X_np = np.asarray(X)
    if X_np.shape != (2, 2):
        raise ValueError("Input matrix X must be a 2x2 matrix.")
    if var1 < 0 or var2 < 0:
        raise ValueError("Variances cannot be negative.")

    # sd_mat is the diagonal matrix of standard deviations
    sd_mat = np.array([
        [np.sqrt(var1), 0],
        [0,             np.sqrt(var2)]
    ])

    # Covariance matrix = sd_mat * Correlation_matrix * sd_mat
    cov_mat = sd_mat @ X_np @ sd_mat
    return cov_mat

class AssortativeMatingSimulation: 
    
    @staticmethod
    def prepare_CV(n_CV, rg_effects, maf_min, maf_max, maf_dist="uniform"):
        """
        Prepares a DataFrame of causal variants (CV) for the simulation.
        
        Args:
            n_CV: Number of causal variants.
            rg_effects: Genetic correlation between the effects (alphas) on the two traits. 
            maf_min: Minimum minor allele frequency.
            maf_max: Maximum minor allele frequency.
            maf_dist: Distribution for MAF generation ("uniform" or "normal").
        Returns:
            A DataFrame containing the minor allele frequencies (MAF) and effect sizes (alpha1, alpha2) for two traits.
        """
        
        # Generate minor allele frequencies (MAF) based on the specified distribution
        if maf_dist == "uniform":
            maf = np.random.uniform(maf_min, maf_max, n_CV)
        elif maf_dist == "normal":
            # Mean at the center of the interval, std dev to keep most values within bounds (approx 3 sigma)
            mean_maf = (maf_min + maf_max) / 2
            std_maf = (maf_max - maf_min) / 6 
            maf = np.random.normal(mean_maf, std_maf, n_CV)
            maf = np.clip(maf, maf_min, maf_max)  # Ensure MAF is within bounds
        else:
            raise ValueError("Invalid MAF distribution specified. Choose 'uniform' or 'normal'.")

        # Define the covariance matrix for the effect sizes (alphas)
        # This reflects the genetic correlation (rg_effects) between traits due to pleiotropy
        alpha_cov_mat = np.array([[1, rg_effects], [rg_effects, 1]])
        
        # Generate initial effect sizes from a multivariate normal distribution
        # These alphas are for traits before considering allele frequency
        alpha_pre = np.random.multivariate_normal(mean=np.zeros(2), cov=alpha_cov_mat, size=n_CV)
        
        # Calculate the genotypic variance attributable to each SNP if its alpha was 1
        # Genotypic variance for a biallelic SNP: 2 * p * q = 2 * maf * (1 - maf)
        var_gene_per_snp_unscaled_alpha = 2 * maf * (1 - maf)
        
        # Scale the effect sizes (alphas)
        # The scaling aims to make the sum of (alpha_i^2 * var_gene_i) for each trait equal to 1 (or some target variance)
        # This specific scaling: scaler = np.sqrt(1 / (var_gene_per_snp_unscaled_alpha * n_CV))
        # means that if rg_effects=0, for each trait j, Sum_i (alpha_ij^2 * 2*pi*(1-pi)) * (1/(2*pi*(1-pi)*n_CV)) = Sum_i (alpha_ij^2 / n_CV)
        # If alpha_pre were drawn from N(0,1), then E[alpha_pre^2] = 1. So Sum_i (alpha_ij^2 / n_CV) would be approx. 1.
        # This implies that the total genetic variance contributed by these n_CV SNPs for each trait is standardized.
        scaler = np.sqrt(1 / (var_gene_per_snp_unscaled_alpha * n_CV))
        alpha_final = alpha_pre * scaler[:, np.newaxis]  # Apply scaling per SNP
        
        return pd.DataFrame({
            "maf": maf,
            "alpha1": alpha_final[:, 0],
            "alpha2": alpha_final[:, 1]
        })
        
    def __init__(self, 
                 cv_info=None, 
                 n_CV=None, rg_effects=None, maf_min=None, maf_max=None, maf_dist="uniform",
                 num_generations=None, pop_size=None, mating_type="phenotypic", avoid_inbreeding=True,
                 save_each_gen=True, save_covs=True, seed=0,
                 output_summary_filename=None, 
                 summary_file_scope="final", 
                 cove_mat=None, f_mat=None, 
                 s_mat=None, 
                 a_mat=None, d_mat=None, 
                 am_list=None, covy_mat=None, k2_matrix=None):

        # ... (CV info loading and essential param check from previous version - remains unchanged) ...
        if cv_info is not None:
            self.cv_info = pd.DataFrame(cv_info)
        elif n_CV is not None and rg_effects is not None and maf_min is not None and maf_max is not None:
            self.cv_info = AssortativeMatingSimulation.prepare_CV(n_CV, rg_effects, maf_min, maf_max, maf_dist)
        else:
            raise ValueError("Either cv_info or parameters to generate it must be provided.")

        essential_params = {
            "num_generations": num_generations, "pop_size": pop_size,
            "cove_mat": cove_mat, "f_mat": f_mat, "s_mat": s_mat,
            "a_mat": a_mat, "d_mat": d_mat,
            "am_list": am_list, "covy_mat": covy_mat, "k2_matrix": k2_matrix
        }
        for param_name, param_val in essential_params.items():
            if param_val is None: raise ValueError(f"Parameter '{param_name}' must be provided.")
        
        self.mating_type = mating_type 
        if self.mating_type not in ["phenotypic", "social", "genotypic"]:
            raise ValueError("mating_type must be 'phenotypic', 'social', or 'genotypic'.")

        self.num_generations = int(num_generations)
        self.initial_pop_size = int(pop_size) if isinstance(pop_size, (int, float)) else int(pop_size[0])
        
        if isinstance(pop_size, (list, np.ndarray)):
            self.pop_vector = np.array(pop_size, dtype=int)
        else: 
            self.pop_vector = np.full(self.num_generations, int(pop_size), dtype=int)
            
        self.avoid_inbreeding = avoid_inbreeding
        self.save_each_gen = save_each_gen
        self.save_covs = save_covs
        self.seed = int(seed) 
        if self.seed != 0: np.random.seed(self.seed)

        self.output_summary_filename = output_summary_filename 
        self.summary_file_scope = summary_file_scope # *** STORE NEW PARAMETER ***
        if self.summary_file_scope not in ["final", "all"]:
            raise ValueError("summary_file_scope must be 'final' or 'all'.")


        self.cove_mat = np.array(cove_mat); self.f_mat = np.array(f_mat)
        self.s_mat = np.array(s_mat) if s_mat is not None else None
        self.a_mat = np.array(a_mat); self.d_mat = np.array(d_mat)
        self.am_list = [np.array(m) for m in am_list]
        self.covy_mat = np.array(covy_mat); self.k2_matrix = np.array(k2_matrix)
        self.num_cvs = len(self.cv_info)

        self.phen_df, self.xo, self.xl = None, None, None
        self.summary_results = []
        self.history = {'MATES': [], 'PHEN': [], 'XO': [], 'XL': []} if self.save_each_gen else None
        self.covariances_log = [] if self.save_covs else None
        
        self.phen_column_names = [
            "ID", "Father.ID", "Mother.ID", "Fathers.Father.ID", "Fathers.Mother.ID",
            "Mothers.Father.ID", "Mothers.Mother.ID", "Sex",
            "AO_std1", "AO_std2", "AL_std1", "AL_std2", "AO1", "AO2", "AL1", "AL2",
            "F1", "F2", "E1", "E2", "Y1", "Y2", "Y1P", "Y2P", "Y1M", "Y2M",
            "F1P", "F2P", "F1M", "F2M", "TPO1", "TPO2", "TMO1", "TMO2", 
            "NTPO1", "NTPO2", "NTMO1", "NTMO2", "TPL1", "TPL2", "TML1", "TML2", 
            "NTPL1", "NTPL2", "NTML1", "NTML2",
        ]
        
        self.n_CV_param = n_CV; self.rg_effects_param = rg_effects
        self.maf_min_param = maf_min; self.maf_max_param = maf_max
        self.maf_dist_param = maf_dist
        
        self._initialize_generation_zero()
        
    def _is_positive_definite(self, matrix):
        if not isinstance(matrix, np.ndarray) or matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]: return False
        if not np.allclose(matrix, matrix.T): return False
        try:
            np.linalg.cholesky(matrix)
            return True
        except np.linalg.LinAlgError:
            return False
        
    def _find_unique_closest_indices(self, dist_matrix_input, max_ord=100):
        dist_matrix = dist_matrix_input.copy() 
        n_to_match_from = dist_matrix.shape[0] 
        n_targets = dist_matrix.shape[1]     

        if n_targets == 0 or n_to_match_from == 0:
            return np.array([], dtype=int)

        rand_index_cols = np.random.permutation(n_targets)
        dist_matrix_shuffled_cols = dist_matrix[:, rand_index_cols]
        
        dist_ord_mat = np.argsort(dist_matrix_shuffled_cols, axis=0)
        new_closest_indices_for_shuffled_targets = np.full(n_targets, -1, dtype=int) 
        assigned_individuals = set()

        for i in range(n_targets): 
            my_min_idx_val = -1 
            k = 0 
            potential_match_found = False
            while k < max_ord and k < n_to_match_from:
                current_preferred_individual_idx = dist_ord_mat[k, i]
                if current_preferred_individual_idx not in assigned_individuals:
                    my_min_idx_val = current_preferred_individual_idx
                    potential_match_found = True
                    break 
                k += 1
            
            if potential_match_found:
                new_closest_indices_for_shuffled_targets[i] = my_min_idx_val
                assigned_individuals.add(my_min_idx_val) 
            else:
                new_closest_indices_for_shuffled_targets[i] = -1 

        final_ordered_indices = np.full(n_targets, -1, dtype=int)
        original_indices_of_shuffled_cols = np.argsort(rand_index_cols) 
        final_ordered_indices = new_closest_indices_for_shuffled_targets[original_indices_of_shuffled_cols]
        
        return final_ordered_indices
    
    def _assort_mate(self, phendata_df_current_gen, mating_type, pheno_mate_corr_target_xsim_mu, pop_size_target_offspring):
        """
        Performs assortative mating based on the specified mating type.
        It matches individuals to a template dataset (Xsim) that has the target
        spousal correlation structure.
        Includes a step to make Xsim's sample covariance match the target covariance.
        """
        # 1. Define columns needed for the selected mating type to save memory
        ancestor_id_cols = ['Father.ID', 'Mother.ID', 'Fathers.Father.ID', 'Fathers.Mother.ID', 'Mothers.Father.ID', 'Mothers.Mother.ID']
        
        if mating_type == "phenotypic":
            cols_for_mating_slim = ["ID", "Sex"] + ["Y1", "Y2"] + ancestor_id_cols
        elif mating_type == "social":
            cols_for_mating_slim = ["ID", "Sex"] + ["F1", "F2", "E1", "E2"] + ancestor_id_cols
        elif mating_type == "genotypic":
            cols_for_mating_slim = ["ID", "Sex"] + ["AO1", "AO2", "AL1", "AL2"] + ancestor_id_cols
        else: 
            raise ValueError(f"Invalid mating_type: {mating_type}")

        # 2. Create slimmed-down DataFrames for mating
        males_condition = phendata_df_current_gen["Sex"] == 1
        females_condition = phendata_df_current_gen["Sex"] == 0
        males_phen_slim = phendata_df_current_gen.loc[males_condition, cols_for_mating_slim].copy()
        females_phen_slim = phendata_df_current_gen.loc[females_condition, cols_for_mating_slim].copy()

        # 3. Create standardized 'mating1' and 'mating2' columns
        if mating_type == "phenotypic":
            males_phen_slim.rename(columns={"Y1": "mating1", "Y2": "mating2"}, inplace=True)
            females_phen_slim.rename(columns={"Y1": "mating1", "Y2": "mating2"}, inplace=True)
        elif mating_type == "social":
            males_phen_slim["mating1"] = males_phen_slim["F1"] + males_phen_slim["E1"]
            males_phen_slim["mating2"] = males_phen_slim["F2"] + males_phen_slim["E2"]
            females_phen_slim["mating1"] = females_phen_slim["F1"] + females_phen_slim["E1"]
            females_phen_slim["mating2"] = females_phen_slim["F2"] + females_phen_slim["E2"]
        elif mating_type == "genotypic":
            males_phen_slim["mating1"] = males_phen_slim["AO1"] + males_phen_slim["AL1"]
            males_phen_slim["mating2"] = males_phen_slim["AO2"] + males_phen_slim["AL2"]
            females_phen_slim["mating1"] = females_phen_slim["AO1"] + females_phen_slim["AL1"]
            females_phen_slim["mating2"] = females_phen_slim["AO2"] + females_phen_slim["AL2"]

        # 4. Equalize numbers of males and females for pairing
        num_males, num_females = len(males_phen_slim), len(females_phen_slim)
        if num_males > num_females:
            drop_indices = np.random.choice(males_phen_slim.index, num_males - num_females, replace=False)
            males_phen_slim.drop(drop_indices, inplace=True)
        elif num_females > num_males:
            drop_indices = np.random.choice(females_phen_slim.index, num_females - num_males, replace=False)
            females_phen_slim.drop(drop_indices, inplace=True)
        
        n_potential_pairs = len(males_phen_slim)
        if n_potential_pairs == 0:
             return {'males.PHENDATA': pd.DataFrame(columns=phendata_df_current_gen.columns), 
                     'females.PHENDATA': pd.DataFrame(columns=phendata_df_current_gen.columns),
                     'achieved_spousal_corr': np.full((2,2), np.nan)}

        # 5. Construct the 4x4 target correlation matrix for Xsim
        corr_mating_vars_males = 0.0
        if len(males_phen_slim) > 1 and not males_phen_slim[['mating1','mating2']].isnull().values.any():
            corr_df_m = males_phen_slim[['mating1','mating2']].corr()
            if not corr_df_m.empty and not pd.isna(corr_df_m.iloc[0,1]): corr_mating_vars_males = corr_df_m.iloc[0,1]
        
        corr_mating_vars_females = 0.0
        if len(females_phen_slim) > 1 and not females_phen_slim[['mating1','mating2']].isnull().values.any():
            corr_df_f = females_phen_slim[['mating1','mating2']].corr()
            if not corr_df_f.empty and not pd.isna(corr_df_f.iloc[0,1]): corr_mating_vars_females = corr_df_f.iloc[0,1]
            
        matcor_xsim_target = np.eye(4) # This will be the target covariance/correlation matrix for Xsim
        matcor_xsim_target[0,1] = matcor_xsim_target[1,0] = corr_mating_vars_males
        matcor_xsim_target[2,3] = matcor_xsim_target[3,2] = corr_mating_vars_females
        matcor_xsim_target[0:2, 2:4] = pheno_mate_corr_target_xsim_mu
        matcor_xsim_target[2:4, 0:2] = pheno_mate_corr_target_xsim_mu.T
        
        if not self._is_positive_definite(matcor_xsim_target):
            print(f"Warning: Target MATCOR for Xsim (4x4) is not positive definite. Attempting correction. Original:\n{matcor_xsim_target}")
            eigenvalues, eigenvectors = np.linalg.eigh(matcor_xsim_target)
            matcor_xsim_target = eigenvectors @ np.diag(np.maximum(eigenvalues, 1e-12)) @ eigenvectors.T
            if not self._is_positive_definite(matcor_xsim_target):
                 print("Error: Target MATCOR for Xsim could not be made positive definite. Using identity matrix as fallback for Xsim generation.")
                 matcor_xsim_target = np.eye(4)

        # Generate initial Xsim template data
        x_sim_initial = np.random.multivariate_normal(np.zeros(4), matcor_xsim_target, size=n_potential_pairs, check_valid='warn')

        # *** NEW: Force x_sim_initial to have the empirical covariance of matcor_xsim_target ***
        if n_potential_pairs > x_sim_initial.shape[1]: # Need more samples than variables
            try:
                # Center the generated data
                x_sim_centered = x_sim_initial - np.mean(x_sim_initial, axis=0)
                
                # Cholesky of the empirical covariance of the centered data
                # Need to handle potential non-positive definiteness of sample cov if n_potential_pairs is small
                emp_cov = np.cov(x_sim_centered, rowvar=False)
                if not self._is_positive_definite(emp_cov): # Check if sample cov is PD
                    # If not PD, try to make it PD (e.g. by adding small value to diagonal or eigenvalue method)
                    # For simplicity, if not PD, we might skip the empirical adjustment or use original x_sim_initial
                    # A more robust approach would be to regularize emp_cov.
                    print("Warning: Empirical covariance of x_sim_initial is not PD. Skipping empirical adjustment for Xsim.")
                    x_sim = x_sim_initial # Use original if adjustment fails
                else:
                    chol_emp = np.linalg.cholesky(emp_cov)
                    
                    # Cholesky of the target covariance matrix
                    chol_target = np.linalg.cholesky(matcor_xsim_target)
                    
                    # Transform the data
                    # x_transformed = x_centered @ inv(chol_emp.T) @ chol_target.T
                    # inv(A.T) = (inv(A)).T. Also, inv(L.T) where L is Cholesky.
                    x_sim = x_sim_centered @ np.linalg.solve(chol_emp.T, chol_target.T) # More stable than inv()
                    # x_sim now has sample covariance matcor_xsim_target
                    # print("Applied empirical covariance adjustment to Xsim.") # For debugging
            except np.linalg.LinAlgError as e:
                print(f"Warning: LinAlgError during empirical covariance adjustment for Xsim: {e}. Using original Xsim.")
                x_sim = x_sim_initial # Fallback to original if Cholesky or solve fails
        else:
            # Not enough samples to reliably force empirical covariance or cov matrix might be singular
            print("Warning: Not enough samples (n_potential_pairs) to force empirical covariance for Xsim. Using original Xsim.")
            x_sim = x_sim_initial
        
        x_sim_m, x_sim_f = x_sim[:, 0:2], x_sim[:, 2:4]
        # *** END OF NEW EMPIRICAL COVARIANCE ADJUSTMENT ***

        # 6. Match actual individuals to Xsim profiles
        male_indices_for_xsim = np.full(n_potential_pairs, -1, dtype=int)
        if not males_phen_slim.empty:
            m_mating_values = males_phen_slim[['mating1', 'mating2']].values.astype(float)
            m_mating_std = np.std(m_mating_values, axis=0, ddof=1); m_mating_std[m_mating_std < 1e-9] = 1.0
            m_mating_scaled = (m_mating_values - np.mean(m_mating_values, axis=0)) / m_mating_std
            dm_dist = cdist(m_mating_scaled, x_sim_m)
            male_indices_for_xsim = self._find_unique_closest_indices(dm_dist)

        female_indices_for_xsim = np.full(n_potential_pairs, -1, dtype=int)
        if not females_phen_slim.empty:
            f_mating_values = females_phen_slim[['mating1', 'mating2']].values.astype(float)
            f_mating_std = np.std(f_mating_values, axis=0, ddof=1); f_mating_std[f_mating_std < 1e-9] = 1.0
            f_mating_scaled = (f_mating_values - np.mean(f_mating_values, axis=0)) / f_mating_std
            df_dist = cdist(f_mating_scaled, x_sim_f)
            female_indices_for_xsim = self._find_unique_closest_indices(df_dist)
            
        # 7. Form valid pairs based on successful matches to the *same* Xsim profile
        valid_pair_mask = (male_indices_for_xsim != -1) & (female_indices_for_xsim != -1)
        paired_male_slim_indices = male_indices_for_xsim[valid_pair_mask]
        paired_female_slim_indices = female_indices_for_xsim[valid_pair_mask]

        if len(paired_male_slim_indices) == 0: 
             return {'males.PHENDATA': pd.DataFrame(columns=phendata_df_current_gen.columns), 
                     'females.PHENDATA': pd.DataFrame(columns=phendata_df_current_gen.columns),
                     'achieved_spousal_corr': np.full((2,2), np.nan)}

        males_slim_paired_for_corr = males_phen_slim.iloc[paired_male_slim_indices].reset_index(drop=True)
        females_slim_paired_for_corr = females_phen_slim.iloc[paired_female_slim_indices].reset_index(drop=True)
        
        paired_male_ids = males_slim_paired_for_corr['ID'].values
        paired_female_ids = females_slim_paired_for_corr['ID'].values
        
        males_paired_full = phendata_df_current_gen[phendata_df_current_gen['ID'].isin(paired_male_ids)]
        males_paired_full = males_paired_full.set_index('ID').loc[paired_male_ids].reset_index()
        females_paired_full = phendata_df_current_gen[phendata_df_current_gen['ID'].isin(paired_female_ids)]
        females_paired_full = females_paired_full.set_index('ID').loc[paired_female_ids].reset_index()
        
        if self.avoid_inbreeding and not males_paired_full.empty:
            sentinel_na = -999999 
            m_ids_df = males_paired_full[ancestor_id_cols].fillna(sentinel_na).astype(float).astype(int)
            f_ids_df = females_paired_full[ancestor_id_cols].fillna(sentinel_na).astype(float).astype(int)
            no_sib_inbreeding = (m_ids_df["Father.ID"] != f_ids_df["Father.ID"]).values
            m_FF,m_FM = m_ids_df["Fathers.Father.ID"].values,m_ids_df["Fathers.Mother.ID"].values
            m_MF,m_MM = m_ids_df["Mothers.Father.ID"].values,m_ids_df["Mothers.Mother.ID"].values
            f_FF,f_FM = f_ids_df["Fathers.Father.ID"].values,f_ids_df["Fathers.Mother.ID"].values
            f_MF,f_MM = f_ids_df["Mothers.Father.ID"].values,f_ids_df["Mothers.Mother.ID"].values
            no_cousin_mask = (m_FF != f_FF)&(m_FF != f_MF)&(m_MF != f_FF)&(m_MF != f_MF)& \
                             (m_FM != f_FM)&(m_FM != f_MM)&(m_MM != f_FM)&(m_MM != f_MM)
            no_inbreeding_mask = no_sib_inbreeding & no_cousin_mask
            males_paired_full = males_paired_full[no_inbreeding_mask].reset_index(drop=True)
            females_paired_full = females_paired_full[no_inbreeding_mask].reset_index(drop=True)
            males_slim_paired_for_corr = males_slim_paired_for_corr[no_inbreeding_mask].reset_index(drop=True)
            females_slim_paired_for_corr = females_slim_paired_for_corr[no_inbreeding_mask].reset_index(drop=True)

        achieved_spousal_corr_matrix = np.full((2,2), np.nan)
        if not males_slim_paired_for_corr.empty and not females_slim_paired_for_corr.empty and \
           len(males_slim_paired_for_corr) >= 2 : 
            m_m1 = males_slim_paired_for_corr['mating1'].values; m_m2 = males_slim_paired_for_corr['mating2'].values
            f_m1 = females_slim_paired_for_corr['mating1'].values; f_m2 = females_slim_paired_for_corr['mating2'].values
            paired_vars_for_corr_df = pd.DataFrame({'MM1':m_m1,'MM2':m_m2,'FM1':f_m1,'FM2':f_m2})
            actual_corr_of_paired_vars = paired_vars_for_corr_df.corr()
            if not actual_corr_of_paired_vars.empty and \
               all(c in actual_corr_of_paired_vars.columns for c in ['MM1','MM2','FM1','FM2']):
                achieved_spousal_corr_matrix = actual_corr_of_paired_vars.loc[['MM1','MM2'],['FM1','FM2']].values
            else: print("Warning: Could not compute achieved spousal correlation matrix from paired data.")
        else: print("Warning: Not enough valid pairs after filtering to compute achieved spousal correlation.")

        if males_paired_full.empty:
             return {'males.PHENDATA': pd.DataFrame(columns=phendata_df_current_gen.columns), 
                     'females.PHENDATA': pd.DataFrame(columns=phendata_df_current_gen.columns),
                     'achieved_spousal_corr': achieved_spousal_corr_matrix}

        males_paired_full['Spouse.ID'] = females_paired_full['ID'].values
        females_paired_full['Spouse.ID'] = males_paired_full['ID'].values
        num_actual_mating_pairs = len(males_paired_full)

        if num_actual_mating_pairs == 0: num_offspring_per_pair = np.array([])
        else:
            lambda_offspring = pop_size_target_offspring / num_actual_mating_pairs
            num_offspring_per_pair = np.random.poisson(lambda_offspring, num_actual_mating_pairs)
            current_total_offspring = np.sum(num_offspring_per_pair)
            diff_offspring = pop_size_target_offspring - current_total_offspring
            if diff_offspring > 0: 
                add_indices = np.random.choice(num_actual_mating_pairs, diff_offspring, replace=True)
                for idx in add_indices: num_offspring_per_pair[idx] += 1
            elif diff_offspring < 0: 
                eligible_indices_to_subtract = np.where(num_offspring_per_pair > 0)[0]
                if len(eligible_indices_to_subtract) > 0:
                    for _ in range(abs(diff_offspring)):
                        if not np.any(num_offspring_per_pair[eligible_indices_to_subtract] > 0): break
                        chosen_idx_to_subtract = np.random.choice(eligible_indices_to_subtract)
                        if num_offspring_per_pair[chosen_idx_to_subtract] > 0:
                             num_offspring_per_pair[chosen_idx_to_subtract] -= 1
        
        males_paired_full['num.offspring'] = num_offspring_per_pair
        females_paired_full['num.offspring'] = num_offspring_per_pair
        
        return {'males.PHENDATA': males_paired_full, 
                'females.PHENDATA': females_paired_full,
                'achieved_spousal_corr': achieved_spousal_corr_matrix}
    
    
    def _reproduce(self, mates_dict, xo_parent_gen_full, xl_parent_gen_full, phendata_parent_gen_df_full):
        males_phen_producing = mates_dict['males.PHENDATA'][mates_dict['males.PHENDATA']['num.offspring'] > 0]
        females_phen_producing = mates_dict['females.PHENDATA'][mates_dict['females.PHENDATA']['num.offspring'] > 0]

        if males_phen_producing.empty or len(males_phen_producing) != len(females_phen_producing):
            return {'PHEN': pd.DataFrame(columns=self.phen_column_names), 'XO': np.array([]), 'XL': np.array([])}

        id_to_original_idx_map = {id_val: i for i, id_val in enumerate(phendata_parent_gen_df_full['ID'])}
        male_original_indices = [id_to_original_idx_map[id_val] for id_val in males_phen_producing['ID'] if id_val in id_to_original_idx_map]
        female_original_indices = [id_to_original_idx_map[id_val] for id_val in females_phen_producing['ID'] if id_val in id_to_original_idx_map]

        males_gentp_obs_producers = xo_parent_gen_full[male_original_indices, :]
        males_gentp_lat_producers = xl_parent_gen_full[male_original_indices, :]
        females_gentp_obs_producers = xo_parent_gen_full[female_original_indices, :]
        females_gentp_lat_producers = xl_parent_gen_full[female_original_indices, :]

        num_offspring_counts = males_phen_producing['num.offspring'].astype(int).values
        total_offspring_generated = np.sum(num_offspring_counts)

        if total_offspring_generated == 0:
            return {'PHEN': pd.DataFrame(columns=self.phen_column_names), 'XO': np.array([]), 'XL': np.array([])}

        male_rep_indices_in_producers = np.repeat(np.arange(len(males_phen_producing)), num_offspring_counts)
        female_rep_indices_in_producers = np.repeat(np.arange(len(females_phen_producing)), num_offspring_counts) 

        fathers_phen_for_offspring = males_phen_producing.iloc[male_rep_indices_in_producers].reset_index(drop=True)
        mothers_phen_for_offspring = females_phen_producing.iloc[female_rep_indices_in_producers].reset_index(drop=True)
        
        males_gentp_obs_rep = males_gentp_obs_producers[male_rep_indices_in_producers, :]
        males_gentp_lat_rep = males_gentp_lat_producers[male_rep_indices_in_producers, :]
        females_gentp_obs_rep = females_gentp_obs_producers[female_rep_indices_in_producers, :]
        females_gentp_lat_rep = females_gentp_lat_producers[female_rep_indices_in_producers, :]

        # --- Genetic Transmission ---
        male_adder_obs = np.random.randint(0, 2, size=males_gentp_obs_rep.shape)
        xm1_obs = (males_gentp_obs_rep == 1).astype(int); xm2_obs = (males_gentp_obs_rep == 2).astype(int)
        males_haps_obs = (male_adder_obs * xm1_obs) + xm2_obs
        female_adder_obs = np.random.randint(0, 2, size=females_gentp_obs_rep.shape)
        xf1_obs = (females_gentp_obs_rep == 1).astype(int); xf2_obs = (females_gentp_obs_rep == 2).astype(int)
        females_haps_obs = (female_adder_obs * xf1_obs) + xf2_obs
        xo_new = males_haps_obs + females_haps_obs

        male_adder_lat = np.random.randint(0, 2, size=males_gentp_lat_rep.shape)
        xl1_lat = (males_gentp_lat_rep == 1).astype(int); xl2_lat = (males_gentp_lat_rep == 2).astype(int)
        males_haps_lat = (male_adder_lat * xl1_lat) + xl2_lat
        female_adder_lat = np.random.randint(0, 2, size=females_gentp_lat_rep.shape)
        fl1_lat = (females_gentp_lat_rep == 1).astype(int); fl2_lat = (females_gentp_lat_rep == 2).astype(int)
        females_haps_lat = (female_adder_lat * fl1_lat) + fl2_lat
        xl_new = males_haps_lat + females_haps_lat

        male_nt_adder_obs = 1 - male_adder_obs; males_nt_haps_obs = (male_nt_adder_obs * xm1_obs) + xm2_obs
        female_nt_adder_obs = 1 - female_adder_obs; females_nt_haps_obs = (female_nt_adder_obs * xf1_obs) + xf2_obs
        male_nt_adder_lat = 1 - male_adder_lat; males_nt_haps_lat = (male_nt_adder_lat * xl1_lat) + xl2_lat
        female_nt_adder_lat = 1 - female_adder_lat; females_nt_haps_lat = (female_nt_adder_lat * fl1_lat) + fl2_lat

        alphas = self.cv_info[['alpha1', 'alpha2']].values
        ao_new_raw = xo_new @ alphas; al_new_raw = xl_new @ alphas
        tpo_new = males_haps_obs @ alphas; tmo_new = females_haps_obs @ alphas
        ntpo_new = males_nt_haps_obs @ alphas; ntmo_new = females_nt_haps_obs @ alphas
        tpl_new = males_haps_lat @ alphas; tml_new = females_haps_lat @ alphas
        ntpl_new = males_nt_haps_lat @ alphas; ntml_new = females_nt_haps_lat @ alphas
        
        ao_std_new = ao_new_raw @ self.d_mat.T; al_std_new = al_new_raw @ self.a_mat.T
        
        # --- Offspring F Component Calculation (Modified) ---
        # 1. Contribution from parental phenotypes (Y) via f_mat
        f_from_parental_y = (fathers_phen_for_offspring[['Y1', 'Y2']].values @ self.f_mat.T) + \
                              (mothers_phen_for_offspring[['Y1', 'Y2']].values @ self.f_mat.T)

        # 2. Contribution from parental social environment (F+E) via s_mat
        f_from_parental_social_env = np.zeros_like(f_from_parental_y) # Initialize to zero
        if self.s_mat is not None:
            parental_social_env_father = np.column_stack((
                fathers_phen_for_offspring['F1'].values + fathers_phen_for_offspring['E1'].values,
                fathers_phen_for_offspring['F2'].values + fathers_phen_for_offspring['E2'].values
            ))
            parental_social_env_mother = np.column_stack((
                mothers_phen_for_offspring['F1'].values + mothers_phen_for_offspring['E1'].values,
                mothers_phen_for_offspring['F2'].values + mothers_phen_for_offspring['E2'].values
            ))
            
            f_from_parental_social_env = (parental_social_env_father @ self.s_mat.T) + \
                                         (parental_social_env_mother @ self.s_mat.T)
        
        # Total Y-scaled F component for offspring
        f_new_y_scaled = f_from_parental_y + f_from_parental_social_env

        # --- Offspring E Component (Unique/Random) ---
        e_new_y_scaled = np.random.multivariate_normal(np.zeros(2), self.cove_mat, size=total_offspring_generated)
        
        # --- Final Phenotype Y ---
        y_new = ao_std_new + al_std_new + f_new_y_scaled + e_new_y_scaled

        # --- Assemble DataFrame (same as before, using the new f_new_y_scaled and e_new_y_scaled) ---
        new_phen_df = pd.DataFrame(index=np.arange(total_offspring_generated), columns=self.phen_column_names)
        current_max_id = phendata_parent_gen_df_full['ID'].max() if not phendata_parent_gen_df_full.empty else 0
        new_phen_df['ID'] = np.arange(1, total_offspring_generated + 1) + current_max_id 
        new_phen_df['Father.ID'] = fathers_phen_for_offspring['ID'].values; new_phen_df['Mother.ID'] = mothers_phen_for_offspring['ID'].values
        new_phen_df['Fathers.Father.ID'] = fathers_phen_for_offspring['Father.ID'].values; new_phen_df['Fathers.Mother.ID'] = fathers_phen_for_offspring['Mother.ID'].values
        new_phen_df['Mothers.Father.ID'] = mothers_phen_for_offspring['Father.ID'].values; new_phen_df['Mothers.Mother.ID'] = mothers_phen_for_offspring['Mother.ID'].values
        sex_vec_offspring = np.zeros(total_offspring_generated, dtype=int); half_pop = total_offspring_generated // 2; sex_vec_offspring[half_pop:] = 1 
        if total_offspring_generated % 2 != 0: sex_vec_offspring[-1] = np.random.randint(0,2)
        new_phen_df['Sex'] = np.random.permutation(sex_vec_offspring)
        new_phen_df['AO_std1'], new_phen_df['AO_std2'] = ao_std_new[:,0], ao_std_new[:,1]; new_phen_df['AL_std1'], new_phen_df['AL_std2'] = al_std_new[:,0], al_std_new[:,1]
        new_phen_df['AO1'], new_phen_df['AO2'] = ao_new_raw[:,0], ao_new_raw[:,1]; new_phen_df['AL1'], new_phen_df['AL2'] = al_new_raw[:,0], al_new_raw[:,1]
        new_phen_df['F1'], new_phen_df['F2'] = f_new_y_scaled[:,0], f_new_y_scaled[:,1]; new_phen_df['E1'], new_phen_df['E2'] = e_new_y_scaled[:,0], e_new_y_scaled[:,1]
        new_phen_df['Y1'], new_phen_df['Y2'] = y_new[:,0], y_new[:,1]
        new_phen_df['TPO1'], new_phen_df['TPO2'] = tpo_new[:,0], tpo_new[:,1]; new_phen_df['TMO1'], new_phen_df['TMO2'] = tmo_new[:,0], tmo_new[:,1]
        new_phen_df['NTPO1'], new_phen_df['NTPO2'] = ntpo_new[:,0], ntpo_new[:,1]; new_phen_df['NTMO1'], new_phen_df['NTMO2'] = ntmo_new[:,0], ntmo_new[:,1]
        new_phen_df['TPL1'], new_phen_df['TPL2'] = tpl_new[:,0], tpl_new[:,1]; new_phen_df['TML1'], new_phen_df['TML2'] = tml_new[:,0], tml_new[:,1]
        new_phen_df['NTPL1'], new_phen_df['NTPL2'] = ntpl_new[:,0], ntpl_new[:,1]; new_phen_df['NTML1'], new_phen_df['NTML2'] = ntml_new[:,0], ntml_new[:,1]
        new_phen_df['Y1P'] = fathers_phen_for_offspring['Y1'].values; new_phen_df['Y2P'] = fathers_phen_for_offspring['Y2'].values
        new_phen_df['Y1M'] = mothers_phen_for_offspring['Y1'].values; new_phen_df['Y2M'] = mothers_phen_for_offspring['Y2'].values
        new_phen_df['F1P'] = fathers_phen_for_offspring['F1'].values; new_phen_df['F2P'] = fathers_phen_for_offspring['F2'].values 
        new_phen_df['F1M'] = mothers_phen_for_offspring['F1'].values; new_phen_df['F2M'] = mothers_phen_for_offspring['F2'].values 
            
        return {'PHEN': new_phen_df, 'XO': xo_new, 'XL': xl_new}
    
    def _initialize_generation_zero(self):
        print("Initializing Generation 0 (Founders)...")
        pop_size_gen0 = self.pop_vector[0]
        n_pop = pop_size_gen0 

        maf_probs = self.cv_info['maf'].values.flatten()
        self.xo = np.random.binomial(2, maf_probs, size=(n_pop, self.num_cvs))
        self.xl = np.random.binomial(2, maf_probs, size=(n_pop, self.num_cvs))

        alphas = self.cv_info[['alpha1', 'alpha2']].values
        ao_raw_gen0 = self.xo @ alphas 
        al_raw_gen0 = self.xl @ alphas 

        ao_std_gen0 = ao_raw_gen0 @ self.d_mat.T 
        al_std_gen0 = al_raw_gen0 @ self.a_mat.T

        # For Gen0, F component is zero.
        f_y_scaled_gen0 = np.zeros((n_pop, 2)) 
        
        e_y_scaled_gen0 = np.random.multivariate_normal(np.zeros(2), self.cove_mat, size=n_pop)
        
        y_gen0 = ao_std_gen0 + al_std_gen0 + f_y_scaled_gen0 + e_y_scaled_gen0 
        
        self.phen_df = pd.DataFrame(index=np.arange(n_pop), columns=self.phen_column_names)

        unique_ids_start = 1_000_000 
        id_vec_gen0 = np.full((n_pop, 7), 0, dtype=int) 
        id_vec_gen0[:,0] = np.arange(unique_ids_start, unique_ids_start + n_pop)
        placeholder_ancestor_start = unique_ids_start + n_pop + 1000 
        for i in range(1, 7): 
             id_vec_gen0[:,i] = np.arange(placeholder_ancestor_start + (i-1)*n_pop, 
                                           placeholder_ancestor_start + i*n_pop)
        self.phen_df['ID'] = id_vec_gen0[:,0]; self.phen_df['Father.ID'] = id_vec_gen0[:,1]
        self.phen_df['Mother.ID'] = id_vec_gen0[:,2]; self.phen_df['Fathers.Father.ID'] = id_vec_gen0[:,3]
        self.phen_df['Fathers.Mother.ID'] = id_vec_gen0[:,4]; self.phen_df['Mothers.Father.ID'] = id_vec_gen0[:,5]
        self.phen_df['Mothers.Mother.ID'] = id_vec_gen0[:,6]

        sex_vec_gen0 = np.zeros(n_pop, dtype=int); half_pop_g0 = n_pop // 2
        sex_vec_gen0[half_pop_g0:] = 1
        if n_pop % 2 != 0: sex_vec_gen0[-1] = np.random.randint(0,2)
        self.phen_df['Sex'] = np.random.permutation(sex_vec_gen0)

        self.phen_df['AO_std1'], self.phen_df['AO_std2'] = ao_std_gen0[:,0], ao_std_gen0[:,1]
        self.phen_df['AL_std1'], self.phen_df['AL_std2'] = al_std_gen0[:,0], al_std_gen0[:,1]
        self.phen_df['AO1'], self.phen_df['AO2'] = ao_raw_gen0[:,0], ao_raw_gen0[:,1]
        self.phen_df['AL1'], self.phen_df['AL2'] = al_raw_gen0[:,0], al_raw_gen0[:,1]
        self.phen_df['F1'], self.phen_df['F2'] = f_y_scaled_gen0[:,0], f_y_scaled_gen0[:,1] 
        self.phen_df['E1'], self.phen_df['E2'] = e_y_scaled_gen0[:,0], e_y_scaled_gen0[:,1]
        self.phen_df['Y1'], self.phen_df['Y2'] = y_gen0[:,0], y_gen0[:,1]
        
        nan_cols_gen0 = ["Y1P", "Y2P", "Y1M", "Y2M", "F1P", "F2P", "F1M", "F2M",
                         "TPO1", "TPO2", "TMO1", "TMO2", "NTPO1", "NTPO2", "NTMO1", "NTMO2",
                         "TPL1", "TPL2", "TML1", "TML2", "NTPL1", "NTPL2", "NTML1", "NTML2"]
        for col in nan_cols_gen0: self.phen_df[col] = np.nan

        if self.save_each_gen: self.history['MATES'].append(None); self.history['PHEN'].append(self.phen_df.copy()); self.history['XO'].append(self.xo.copy()); self.history['XL'].append(self.xl.copy())
        if self.save_covs: self.covariances_log.append(None)
        summary_gen0 = {'GEN': 0, 'NUM.CVs': self.num_cvs, 'MATE.COR': self.am_list[0].tolist() if len(self.am_list) > 0 else None, 'POPSIZE': n_pop}
        for comp_name, cols in {"VAO": ["AO_std1", "AO_std2"], "VAL": ["AL_std1", "AL_std2"], "VF":  ["F1", "F2"], "VE":  ["E1", "E2"], "VP":  ["Y1", "Y2"]}.items():
             if all(c in self.phen_df.columns for c in cols):
                 df_subset = self.phen_df[cols].dropna()
                 summary_gen0[comp_name] = np.cov(df_subset, rowvar=False).tolist() if len(df_subset) >=2 else np.full((2,2),np.nan).tolist()
             else: summary_gen0[comp_name] = np.full((2,2),np.nan).tolist()
        vp_diag_g0 = np.diag(np.array(summary_gen0.get('VP', [[np.nan,np.nan]]))); vao_diag_g0 = np.diag(np.array(summary_gen0.get('VAO', [[np.nan,np.nan]]))); val_diag_g0 = np.diag(np.array(summary_gen0.get('VAL', [[np.nan,np.nan]])))
        with np.errstate(divide='ignore', invalid='ignore'): summary_gen0['h2'] = ((vao_diag_g0 + val_diag_g0) / vp_diag_g0).tolist(); summary_gen0['h2.obs'] = (vao_diag_g0 / vp_diag_g0).tolist(); summary_gen0['h2.lat'] = (val_diag_g0 / vp_diag_g0).tolist()
        for key in ['covY', 'covG', 'covH', 'covI', 'w', 'v', 'covF', 'covE', 'hapsO.covs', 'hapsL.covs', 'omega', 'gamma', 'thetaNT', 'thetaT']: # Changed 'q' to 'v'
            summary_gen0[key] = np.nan 
        self.summary_results.append(summary_gen0)
        print("Generation 0 initialized successfully (F components are zero).")
            
    def _format_single_generation_summary(self, gen_summary_dict):
        s = []
        s.append(f"Generation: {gen_summary_dict.get('GEN', 'N/A')}")
        s.append(f"Population Size: {gen_summary_dict.get('POPSIZE', 'N/A')}")
        s.append(f"Mating Correlation (mu for these parents): {self._format_matrix_for_file(gen_summary_dict.get('MATE.COR', 'N/A'))}")
        s.append("\nVariance Components (for Y-scaled values):")
        for key in ['VAO', 'VAL', 'VF', 'VE', 'VP']:
            s.append(f"  {key}: {self._format_matrix_for_file(gen_summary_dict.get(key, 'N/A'))}")
        s.append("\nHeritabilities:")
        for key in ['h2', 'h2.obs', 'h2.lat']:
            val = gen_summary_dict.get(key, ['N/A', 'N/A'])
            val1_str = f"{val[0]:.4f}" if isinstance(val[0], (float, np.floating)) else str(val[0])
            val2_str = f"{val[1]:.4f}" if isinstance(val[1], (float, np.floating)) else str(val[1])
            s.append(f"  {key} (Trait1, Trait2): ({val1_str}, {val2_str})")
        s.append("\nKey Covariance Matrices from This Generation:")
        cov_keys = ['covG', 'covH', 'covI', 'omega', 'gamma', 'w', 'v', 'covF', 'covE', 'thetaNT', 'thetaT'] # Changed 'q' to 'v'
        for key in cov_keys:
            if key in gen_summary_dict and not (isinstance(gen_summary_dict[key], float) and np.isnan(gen_summary_dict[key])):
                 s.append(f"  {key}: {self._format_matrix_for_file(gen_summary_dict.get(key))}")
            elif key in gen_summary_dict: 
                 s.append(f"  {key}: Not Calculated or N/A")
        return "\n".join(s)

    def _format_matrix_for_file(self, m):
        if isinstance(m, list): 
            try:
                m_arr = np.array(m); 
                try: m_arr = m_arr.astype(float) 
                except ValueError: pass 
                if m_arr.ndim == 2: return "\n" + np.array2string(m_arr, precision=4, separator=', ', floatmode='fixed', suppress_small=True)
                elif m_arr.ndim == 1: return np.array2string(m_arr, precision=4, separator=', ', floatmode='fixed', suppress_small=True)
            except Exception: pass 
        if isinstance(m, np.ndarray):
            try: m = m.astype(float)
            except ValueError: pass
            return "\n" + np.array2string(m, precision=4, separator=', ', floatmode='fixed', suppress_small=True)
        if isinstance(m, pd.DataFrame): return "\n" + m.to_string(float_format="%.4f")
        if isinstance(m, pd.Series): return "\n" + m.to_string(float_format="%.4f")
        if isinstance(m, (float, np.floating)): return f"{m:.4f}"
        if m is None or (isinstance(m, float) and np.isnan(m)): return "N/A"
        return str(m)

    def _write_simulation_summary_to_file(self):
        if not self.output_summary_filename: return 
        if not self.summary_results: print("Warning: No summary results to write."); return
        try:
            with open(self.output_summary_filename, 'w') as f:
                f.write(f"--- Simulation Summary Output ---\nTimestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\nOutput File: {self.output_summary_filename}\n\n")
                f.write("--- Simulation Setup Parameters ---\n")
                f.write(f"Total Generations Simulated (target): {self.num_generations}\nInitial Population Size: {self.initial_pop_size}\n")
                if isinstance(self.pop_vector, np.ndarray) and not np.all(self.pop_vector == self.initial_pop_size): f.write(f"Population Size Vector: {self.pop_vector.tolist()}\n")
                f.write(f"Mating Type: {self.mating_type}\nAvoid Inbreeding: {self.avoid_inbreeding}\nSeed: {self.seed}\n")
                f.write(f"Number of CVs: {self.n_CV_param if self.n_CV_param is not None else len(self.cv_info)}\n")
                if self.rg_effects_param is not None: f.write(f"rg (effects for CVs): {self.rg_effects_param}\n")
                if self.maf_min_param is not None: f.write(f"MAF Range: {self.maf_min_param} - {self.maf_max_param} ({self.maf_dist_param})\n")
                f.write("\n--- Key Model Matrices (Initial Values) ---\n")
                f.write(f"k2_matrix: {self._format_matrix_for_file(self.k2_matrix)}\nd_mat: {self._format_matrix_for_file(self.d_mat)}\na_mat: {self._format_matrix_for_file(self.a_mat)}\n")
                f.write(f"f_mat: {self._format_matrix_for_file(self.f_mat)}\n")
                f.write(f"s_mat: {self._format_matrix_for_file(self.s_mat) if self.s_mat is not None else 'Not used'}\n")
                f.write(f"cove_mat: {self._format_matrix_for_file(self.cove_mat)}\ncovy_mat: {self._format_matrix_for_file(self.covy_mat)}\n")
                if self.am_list: f.write(f"am_list (mu for Gen0 parents): {self._format_matrix_for_file(self.am_list[0])}\n")
                
                final_gen_summary_dict = self.summary_results[-1]
                f.write(f"\n\n--- Summary for Final Generation (Generation {final_gen_summary_dict.get('GEN', 'N/A')}) ---\n")
                f.write(self._format_single_generation_summary(final_gen_summary_dict))
                f.write("\n")

                if self.summary_file_scope == "all" and len(self.summary_results) > 1:
                    f.write("\n\n--- Summaries for Preceding Generations ---\n")
                    for i in range(len(self.summary_results) - 1): # Iterate up to second to last
                        gen_summary_dict = self.summary_results[i]
                        f.write(f"\n--- Generation {gen_summary_dict.get('GEN', 'N/A')} Summary ---\n")
                        f.write(self._format_single_generation_summary(gen_summary_dict))
                        f.write("\n")
                f.write("\n--- End of Summary File ---\n")
            print(f"Summary written to {self.output_summary_filename}")
        except IOError as e: print(f"Error writing summary to file {self.output_summary_filename}: {e}")
        except Exception as e: print(f"An unexpected error occurred while writing summary: {e}; {type(e)}")


    def run_simulation(self):
            """
            Runs the entire simulation loop from Generation 1 up to num_generations.
            It orchestrates assortative mating, reproduction, and summary statistics calculation
            for each generation.
            """
            print(f"Starting simulation for {self.num_generations} generations using '{self.mating_type}' assortment.")
            
            if self.phen_df is None:
                print("Error: Generation 0 not initialized. Cannot run simulation.")
                return None

            # Loop through generations, starting from Gen 1 (Python index 0 for first loop iter)
            for cur_gen_py_idx in range(self.num_generations): 
                r_cur_gen_num = cur_gen_py_idx + 1 # R-style 1-indexed generation number for prints/logs
                
                print(f"\n--- Beginning Generation {r_cur_gen_num} ---")

                # Determine population size for the offspring of the current generation
                pop_size_target_for_offspring = self.pop_vector[cur_gen_py_idx]
                
                # Get assortative mating correlation matrix (mu) for the current set of parents
                # self.phen_df currently holds the parental generation data for this iteration.
                # self.am_list[cur_gen_py_idx] provides the target AM correlations (mu_m1f1, mu_m1f2 etc.)
                # that these parents will use.
                mate_cor_mu_for_current_parents = self.am_list[cur_gen_py_idx]
                
                # --- 1. Assortative Mating ---
                print(f"Generation {r_cur_gen_num}: Performing '{self.mating_type}' assortative mating "
                    f"(target offspring: {pop_size_target_for_offspring})...")
                
                # The `_assort_mate` method now internally handles the construction of the 
                # 4x4 Xsim correlation matrix based on the mating_type and current parental phenotypes.
                mates = self._assort_mate(
                    phendata_df_current_gen=self.phen_df, # Current generation acts as parents
                    mating_type=self.mating_type,         # Type of assortment (phenotypic, social, genotypic)
                    pheno_mate_corr_target_xsim_mu=mate_cor_mu_for_current_parents, # Target mu_ij matrix
                    pop_size_target_offspring=pop_size_target_for_offspring
                )
                
                num_m_mated = len(mates['males.PHENDATA'])
                sum_offspring = mates['males.PHENDATA']['num.offspring'].sum() \
                    if num_m_mated > 0 and 'num.offspring' in mates['males.PHENDATA'].columns else 0

                if num_m_mated == 0 or sum_offspring == 0:
                    print(f"Generation {r_cur_gen_num}: No reproducing pairs formed or no offspring targeted. "
                        "Simulation halting.")
                    break 
                print(f"Generation {r_cur_gen_num}: Assortative mating done. "
                    f"{num_m_mated} initial pairs, targeting {sum_offspring} total offspring.")
                
                # --- 2. Reproduction ---
                print(f"Generation {r_cur_gen_num}: Simulating reproduction...")
                # Offspring are generated from the `mates` dictionary.
                # xo, xl, and phen_df for the current (parental) generation are passed.
                offspring_data = self._reproduce(
                    mates_dict=mates, 
                    xo_parent_gen_full=self.xo, 
                    xl_parent_gen_full=self.xl, 
                    phendata_parent_gen_df_full=self.phen_df
                )
                
                if offspring_data['PHEN'].empty:
                    print(f"Generation {r_cur_gen_num}: No offspring resulted from reproduction. Simulation halting.")
                    break
                print(f"Generation {r_cur_gen_num}: Reproduction done. {len(offspring_data['PHEN'])} offspring created.")

                # Update master state variables to the new offspring generation
                self.xo = offspring_data['XO']
                self.xl = offspring_data['XL']
                self.phen_df = offspring_data['PHEN'] # self.phen_df now represents the new generation

                # --- 3. Calculate and Log Summary Statistics for the New Generation ---
                print(f"Generation {r_cur_gen_num}: Calculating summary statistics for the new generation...")
                
                # Create a temporary DataFrame for covariance calculation that might include derived columns
                # like BV.NT.O1/2, as these were part of the `covs` matrix in the R script.
                temp_phen_df_for_cov = self.phen_df.copy()
                if "NTPO1" in temp_phen_df_for_cov and "NTMO1" in temp_phen_df_for_cov and \
                "NTPO2" in temp_phen_df_for_cov and "NTMO2" in temp_phen_df_for_cov:
                    temp_phen_df_for_cov["BV.NT.O1"] = temp_phen_df_for_cov["NTPO1"] + temp_phen_df_for_cov["NTMO1"]
                    temp_phen_df_for_cov["BV.NT.O2"] = temp_phen_df_for_cov["NTPO2"] + temp_phen_df_for_cov["NTMO2"]
                # Similarly for BV.NT.L if it was used in R's `covs` and subsequent calcs.
                # if "NTPL1" in temp_phen_df_for_cov and ...:
                #    temp_phen_df_for_cov["BV.NT.L1"] = ... 
                
                # Define the list of columns for the main covariance matrix, matching R's `covs`
                cols_for_full_cov_calc = [
                    'TPO1','TMO1','NTPO1','NTMO1','TPL1','TML1','NTPL1','NTML1', # Haps for Trait 1
                    'TPO2','TMO2','NTPO2','NTMO2','TPL2','TML2','NTPL2','NTML2', # Haps for Trait 2
                    'AO1','AO2','AL1','AL2',       # Raw Genetic Values
                    'F1','F2','E1','E2',           # Y-scaled Environmental Components for offspring
                    "BV.NT.O1","BV.NT.O2",         # Non-transmitted Observed BV (sum of NTPO+NTMO)
                    'Y1','Y2',                     # Final Phenotypes of offspring
                    'Y1P','Y2P','Y1M','Y2M',       # Parental Phenotypes
                    'F1P','F2P','F1M','F2M'        # Parental Y-scaled F components
                ]
                
                valid_cols_for_cov_df = [col for col in cols_for_full_cov_calc 
                                        if col in temp_phen_df_for_cov.columns and \
                                            pd.api.types.is_numeric_dtype(temp_phen_df_for_cov[col])]
                
                full_cov_df = pd.DataFrame() # Initialize as empty
                if len(valid_cols_for_cov_df) > 1 and len(temp_phen_df_for_cov.dropna(subset=valid_cols_for_cov_df)) >= 2 :
                    full_cov_df = temp_phen_df_for_cov[valid_cols_for_cov_df].cov()
                else:
                    print(f"Warning: Not enough data or valid columns to compute full covariance matrix for Gen {r_cur_gen_num}.")


                # Initialize covariance components to NaN matrices or scalars
                nan_2x2_matrix = np.full((2,2), np.nan)
                covG_val, covH_val, covI_val = nan_2x2_matrix.copy(), nan_2x2_matrix.copy(), nan_2x2_matrix.copy()
                w_val, v_val, covF_calc_val, covE_calc_val = nan_2x2_matrix.copy(), nan_2x2_matrix.copy(), nan_2x2_matrix.copy(), nan_2x2_matrix.copy()
                omega_val, gamma_val, thetaNT_val, thetaT_val = nan_2x2_matrix.copy(), nan_2x2_matrix.copy(), nan_2x2_matrix.copy(), nan_2x2_matrix.copy()
                hapsO_covs_dict, hapsL_covs_dict = np.nan, np.nan # Store as dicts or DataFrames

                # Proceed with detailed covariance calculations only if full_cov_df is not empty
                if not full_cov_df.empty:
                    # Define haplotype column groups for easier slicing
                    o_h1_cols = ['TPO1','TMO1','NTPO1','NTMO1']; o_h2_cols = ['TPO2','TMO2','NTPO2','NTMO2']
                    l_h1_cols = ['TPL1','TML1','NTPL1','NTML1']; l_h2_cols = ['TPL2','TML2','NTPL2','NTML2']

                    # Calculate hapsO.covs & covG
                    if all(c in temp_phen_df_for_cov.columns for c in o_h1_cols+o_h2_cols) and \
                    temp_phen_df_for_cov[o_h1_cols+o_h2_cols].notna().all().all() and len(temp_phen_df_for_cov) >=2 :
                        hapsO_covs_df_loc = temp_phen_df_for_cov[o_h1_cols+o_h2_cols].cov()
                        hapsO_covs_dict = hapsO_covs_df_loc.to_dict() # Store the full hapsO_covs matrix

                        hapsO1_vals = hapsO_covs_df_loc.loc[o_h1_cols, o_h1_cols].values
                        g11pre = hapsO1_vals - np.eye(4) * (self.k2_matrix[0,0] / 2.0)
                        g11 = g11pre[np.tril_indices_from(g11pre)]
                        
                        hapsO2_vals = hapsO_covs_df_loc.loc[o_h2_cols, o_h2_cols].values
                        g22pre = hapsO2_vals - np.eye(4) * (self.k2_matrix[1,1] / 2.0)
                        g22 = g22pre[np.tril_indices_from(g22pre)]
                        
                        hapsO12_vals = hapsO_covs_df_loc.loc[o_h2_cols, o_h1_cols].values 
                        g12pre = hapsO12_vals - np.diag(np.full(4, self.k2_matrix[0,1] / 2.0)) # Corrected: diag for non-square block if k2 is scalar
                        g12 = g12pre.flatten() 
                        covG_val = np.array([[np.mean(g11), np.mean(g12)], [np.mean(g12), np.mean(g22)]])

                    # Calculate hapsL.covs & covH
                    if all(c in temp_phen_df_for_cov.columns for c in l_h1_cols+l_h2_cols) and \
                    temp_phen_df_for_cov[l_h1_cols+l_h2_cols].notna().all().all() and len(temp_phen_df_for_cov) >=2 :
                        hapsL_covs_df_loc = temp_phen_df_for_cov[l_h1_cols+l_h2_cols].cov()
                        hapsL_covs_dict = hapsL_covs_df_loc.to_dict()

                        hapsL1_vals = hapsL_covs_df_loc.loc[l_h1_cols, l_h1_cols].values
                        h11pre = hapsL1_vals - np.eye(4) * (self.k2_matrix[0,0] / 2.0) 
                        h11 = h11pre[np.tril_indices_from(h11pre)]
                        
                        hapsL2_vals = hapsL_covs_df_loc.loc[l_h2_cols, l_h2_cols].values
                        h22pre = hapsL2_vals - np.eye(4) * (self.k2_matrix[1,1] / 2.0)
                        h22 = h22pre[np.tril_indices_from(h22pre)]
                        
                        hapsL12_vals = hapsL_covs_df_loc.loc[l_h2_cols, l_h1_cols].values
                        h12pre = hapsL12_vals - np.diag(np.full(4, self.k2_matrix[0,1] / 2.0))
                        h12 = h12pre.flatten()
                        covH_val = np.array([[np.mean(h11), np.mean(h12)], [np.mean(h12), np.mean(h22)]])

                    # Calculate covI (Cov(Latent Haps, Observed Haps))
                    # Uses full_cov_df as it involves cross-haplotype-type covariances
                    if all(c in full_cov_df.columns for c in o_h1_cols+l_h1_cols+o_h2_cols+l_h2_cols):
                        i11_b = full_cov_df.loc[l_h1_cols, o_h1_cols].values.flatten() # Cov(L1, O1)
                        i22_b = full_cov_df.loc[l_h2_cols, o_h2_cols].values.flatten() # Cov(L2, O2)
                        i12_b = full_cov_df.loc[l_h2_cols, o_h1_cols].values.flatten() # Cov(L2, O1) as per R (i12 = cov(O1,L2))
                        i21_b = full_cov_df.loc[l_h1_cols, o_h2_cols].values.flatten() # Cov(L1, O2) as per R (i21 = cov(O2,L1))
                        covI_val = np.array([[np.mean(i11_b), np.mean(i12_b)], [np.mean(i21_b), np.mean(i22_b)]])
                    
                    # Calculate w, v, covF_calc
                    f_env_cols = ['F1','F2']; ao_raw_cols = ['AO1','AO2']; al_raw_cols = ['AL1','AL2']
                    if all(c in full_cov_df.columns for c in f_env_cols+ao_raw_cols+al_raw_cols):
                        w_val = full_cov_df.loc[f_env_cols, ao_raw_cols].values 
                        v_val = full_cov_df.loc[f_env_cols, al_raw_cols].values 
                        covF_calc_val = full_cov_df.loc[f_env_cols, f_env_cols].values
                    
                    # Calculate covE_calc
                    e_env_cols = ['E1','E2'] 
                    if all(c in full_cov_df.columns for c in e_env_cols):
                        covE_calc_val = full_cov_df.loc[e_env_cols, e_env_cols].values

                    # Calculate Omega, Gamma, Theta
                    # Define specific haplotype component columns for clarity
                    tpo_cols = ['TPO1','TPO2']; tmo_cols = ['TMO1','TMO2']
                    ntpo_cols = ['NTPO1','NTPO2']; ntmo_cols = ['NTMO1','NTMO2']
                    tpl_cols = ['TPL1','TPL2']; tml_cols = ['TML1','TML2']
                    ntpl_cols = ['NTPL1','NTPL2']; ntml_cols = ['NTML1','NTML2']
                    yp_cols = ['Y1P','Y2P']; ym_cols = ['Y1M','Y2M']; y_offspring_cols = ['Y1','Y2']

                    if all(c in full_cov_df.columns for c in yp_cols+ym_cols+tpo_cols+tmo_cols+ntpo_cols+ntmo_cols):
                        omega_p_T = full_cov_df.loc[yp_cols, tpo_cols].values 
                        omega_m_T = full_cov_df.loc[ym_cols, tmo_cols].values
                        omega_T_val = (omega_p_T + omega_m_T) * 0.5
                        omega_p_NT = full_cov_df.loc[yp_cols, ntpo_cols].values
                        omega_m_NT = full_cov_df.loc[ym_cols, ntmo_cols].values
                        omega_NT_val = (omega_p_NT + omega_m_NT) * 0.5
                        omega_val = (omega_T_val + omega_NT_val) * 0.5

                    if all(c in full_cov_df.columns for c in yp_cols+ym_cols+tpl_cols+tml_cols+ntpl_cols+ntml_cols):
                        gamma_p_T = full_cov_df.loc[yp_cols, tpl_cols].values
                        gamma_m_T = full_cov_df.loc[ym_cols, tml_cols].values
                        gamma_T_val = (gamma_p_T + gamma_m_T) * 0.5
                        gamma_p_NT = full_cov_df.loc[yp_cols, ntpl_cols].values
                        gamma_m_NT = full_cov_df.loc[ym_cols, ntml_cols].values
                        gamma_NT_val = (gamma_p_NT + gamma_m_NT) * 0.5
                        gamma_val = (gamma_T_val + gamma_NT_val) * 0.5

                    if all(c in full_cov_df.columns for c in y_offspring_cols+ntpo_cols+ntmo_cols):
                        theta_NTp = full_cov_df.loc[y_offspring_cols, ntpo_cols].values 
                        theta_NTm = full_cov_df.loc[y_offspring_cols, ntmo_cols].values 
                        thetaNT_val = theta_NTp + theta_NTm
                    
                    if all(c in full_cov_df.columns for c in y_offspring_cols+tpo_cols+tmo_cols):
                        theta_Tp = full_cov_df.loc[y_offspring_cols, tpo_cols].values 
                        theta_Tm = full_cov_df.loc[y_offspring_cols, tmo_cols].values 
                        thetaT_val = theta_Tp + theta_Tm
                
                # --- Store Summary for Current Generation ---
                summary_this_gen = {
                    'GEN': r_cur_gen_num, 
                    'NUM.CVs': self.num_cvs,
                    'MATE.COR': self.am_list[r_cur_gen_num].tolist() if r_cur_gen_num < len(self.am_list) else (self.am_list[-1].tolist() if self.am_list else None), # AM parameters for *next* mating
                    'POPSIZE': len(self.phen_df)
                }
                
                # VAO, VAL are from AO_std, AL_std. VF, VE from F1, E1 (Y-scaled env). VP from Y1,Y2.
                for comp_name, cols in {"VAO": ["AO_std1", "AO_std2"], "VAL": ["AL_std1", "AL_std2"],
                                        "VF":  ["F1", "F2"], "VE":  ["E1", "E2"], "VP":  ["Y1", "Y2"]}.items():
                    if all(c in self.phen_df.columns for c in cols):
                        df_subset_s = self.phen_df[cols].dropna()
                        summary_this_gen[comp_name] = np.cov(df_subset_s, rowvar=False).tolist() if len(df_subset_s) >=2 else np.full((2,2),np.nan).tolist()
                    else:
                        summary_this_gen[comp_name] = np.full((2,2),np.nan).tolist()
                
                # Heritabilities
                vp_diag_s = np.diag(np.array(summary_this_gen.get('VP', [[np.nan,np.nan]])))
                vao_diag_s = np.diag(np.array(summary_this_gen.get('VAO', [[np.nan,np.nan]])))
                val_diag_s = np.diag(np.array(summary_this_gen.get('VAL', [[np.nan,np.nan]])))
                with np.errstate(divide='ignore', invalid='ignore'):
                    summary_this_gen['h2'] = ((vao_diag_s + val_diag_s) / vp_diag_s).tolist()
                    summary_this_gen['h2.obs'] = (vao_diag_s / vp_diag_s).tolist()
                    summary_this_gen['h2.lat'] = (val_diag_s / vp_diag_s).tolist()

                # Store all other calculated covariance components
                summary_this_gen.update({
                    'covY': full_cov_df.loc[['Y1P','Y2P','Y1M','Y2M','Y1','Y2'], ['Y1P','Y2P','Y1M','Y2M','Y1','Y2']].values.tolist() if all(c in full_cov_df.columns for c in ['Y1P','Y1']) and not full_cov_df.empty else np.nan,
                    'covG': covG_val.tolist(), 'covH': covH_val.tolist(), 'covI': covI_val.tolist(),
                    'w': w_val.tolist(), 'v': v_val.tolist(),
                    'covF': covF_calc_val.tolist(), # This is Cov(F1_off, F2_off)
                    'covE': covE_calc_val.tolist(), # This is Cov(E1_off, E2_off)
                    'hapsO.covs': hapsO_covs_dict, 
                    'hapsL.covs': hapsL_covs_dict,
                    'omega': omega_val.tolist(), 'gamma': gamma_val.tolist(),
                    'thetaNT': thetaNT_val.tolist(), 'thetaT': thetaT_val.tolist()
                })
                self.summary_results.append(summary_this_gen)

                # --- Save History if enabled ---
                if self.save_each_gen:
                    self.history['MATES'].append(mates) # Store the actual mates dictionary
                    self.history['PHEN'].append(self.phen_df.copy())
                    self.history['XO'].append(self.xo.copy())
                    self.history['XL'].append(self.xl.copy())
                
                if self.save_covs:
                    self.covariances_log.append(full_cov_df.round(3).to_dict() if not full_cov_df.empty else None)

                print(f"--- Generation {r_cur_gen_num} Processing Done ---")
                # --- Write final summary to file if filename is provided ---
                if self.output_summary_filename:
                    self._write_simulation_summary_to_file()
                
            return {
                'SUMMARY.RES': self.summary_results,
                'XO': self.xo, 'XL': self.xl, 'PHEN': self.phen_df,
                'HISTORY': self.history,
                'COVARIANCES': self.covariances_log
            }