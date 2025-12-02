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
    def prepare_CV_random_selection(n_CV, rg_effects, maf_min, maf_max, 
                                    prop_h2_obs_1, prop_h2_obs_2, 
                                    maf_dist="uniform"):
        """
        Implements the user's "Random Selection" logic.
        1. Generates a single pool of effects (alphas) for all n_CV variants.
        2. Randomly selects a subset to be 'Observed' for Trait 1.
        3. Randomly selects a subset to be 'Observed' for Trait 2.
        """
        # --- MAF Generation ---
        if maf_dist == "uniform":
            maf = np.random.uniform(maf_min, maf_max, n_CV)
        elif maf_dist == "normal":
            mean_maf = (maf_min + maf_max) / 2
            std_maf = (maf_max - maf_min) / 6 
            maf = np.random.normal(mean_maf, std_maf, n_CV)
            maf = np.clip(maf, maf_min, maf_max)
        
        # --- Effect Size Generation (Total Biological Effect) ---
        alpha_cov_mat = np.array([[1, rg_effects], [rg_effects, 1]])
        alpha_raw = np.random.multivariate_normal(np.zeros(2), alpha_cov_mat, size=n_CV)

        # Scale effects so the total genetic variance is ~1.0
        var_gene_per_snp = 2 * maf * (1 - maf)
        scaler = np.sqrt(1 / (np.sum(var_gene_per_snp))) # Scale to total sum
        # Ideally, we scale assuming random distribution, or just normalize sum to 1.
        # Simpler scaler per SNP for uniform contribution expectation:
        scaler = np.sqrt(1 / (var_gene_per_snp * n_CV))
        alpha_final = alpha_raw * scaler[:, np.newaxis]

        # --- Random Mask Generation ---
        # Mask = 1 if Observed, 0 if Latent
        
        # Trait 1 Mask
        n_obs_1 = int(n_CV * prop_h2_obs_1)
        mask1 = np.zeros(n_CV, dtype=int)
        if n_obs_1 > 0:
            indices_1 = np.random.choice(n_CV, n_obs_1, replace=False)
            mask1[indices_1] = 1
            
        # Trait 2 Mask
        n_obs_2 = int(n_CV * prop_h2_obs_2)
        mask2 = np.zeros(n_CV, dtype=int)
        if n_obs_2 > 0:
            indices_2 = np.random.choice(n_CV, n_obs_2, replace=False)
            mask2[indices_2] = 1

        return pd.DataFrame({
            "maf": maf,
            "alpha1": alpha_final[:, 0], 
            "alpha2": alpha_final[:, 1],
            "mask_obs1": mask1,
            "mask_obs2": mask2
        })
        
    def __init__(self, 
                 cv_info=None, 
                 n_CV=None, rg_effects=None, maf_min=None, maf_max=None, maf_dist="uniform",
                 # h2_targets removed per request
                 num_generations=None, pop_size=None, mating_type="phenotypic", avoid_inbreeding=True,
                 save_each_gen=True, save_covs=True, seed=0,
                 output_summary_filename=None, 
                 summary_file_scope="final", 
                 cove_mat=None, f_mat=None, 
                 s_mat=None, 
                 a_mat=None, d_mat=None, 
                 am_list=None, mate_on_trait=None, covy_mat=None, k2_matrix=None):

        # --- CV GENERATION ---
        if cv_info is not None:
            self.cv_info = pd.DataFrame(cv_info)
            self.n_CV_param = None; self.rg_effects_param = None
            self.maf_min_param = None; self.maf_max_param = None; self.maf_dist_param = None
        
        elif n_CV is not None and rg_effects is not None:
            # Derive proportions STRICTLY from d_mat and a_mat
            if d_mat is not None and a_mat is not None:
                # Convert to numpy arrays to ensure math operations work
                d_arr = np.array(d_mat, dtype=float)
                a_arr = np.array(a_mat, dtype=float)
                
                # Calculate variances for each trait (row sum of squares)
                # Row 0 = Trait 1, Row 1 = Trait 2
                var_obs = np.sum(d_arr**2, axis=1)
                var_lat = np.sum(a_arr**2, axis=1)
                total_g = var_obs + var_lat
                
                # Calculate proportion of observed heritability
                # Prop = Var_Obs / (Var_Obs + Var_Lat)
                prop1 = var_obs[0] / total_g[0] if total_g[0] > 1e-9 else 0.0
                prop2 = var_obs[1] / total_g[1] if total_g[1] > 1e-9 else 0.0
                
            else:
                # If matrices are missing, we cannot derive proportions. 
                # Default to 0.5 to avoid crash, or raise error if strictness preferred.
                prop1 = 0.5; prop2 = 0.5
            
            self.cv_info = AssortativeMatingSimulation.prepare_CV_random_selection(
                n_CV, rg_effects, maf_min, maf_max, prop1, prop2, maf_dist
            )
            
            self.n_CV_param = n_CV; self.rg_effects_param = rg_effects
            self.maf_min_param = maf_min; self.maf_max_param = maf_max; self.maf_dist_param = maf_dist
        else:
            raise ValueError("CV info missing: Provide either 'cv_info' DataFrame or 'n_CV'/'rg_effects' params.")

        # Standard Init Checks
        essential_params = {
            "num_generations": num_generations, "pop_size": pop_size,
            "cove_mat": cove_mat, "f_mat": f_mat, "s_mat": s_mat,
            "a_mat": a_mat, "d_mat": d_mat,
            "am_list": am_list, "covy_mat": covy_mat, "k2_matrix": k2_matrix
        }
        for param_name, param_val in essential_params.items():
            if param_val is None: raise ValueError(f"Parameter '{param_name}' must be provided.")
            
        self.mating_type = mating_type 
        self.num_generations = int(num_generations)
        self.initial_pop_size = int(pop_size) if isinstance(pop_size, (int, float)) else int(pop_size[0])
        self.pop_vector = np.array(pop_size, dtype=int) if isinstance(pop_size, (list, np.ndarray)) else np.full(self.num_generations, int(pop_size), dtype=int)
        self.avoid_inbreeding = avoid_inbreeding; self.save_each_gen = save_each_gen; self.save_covs = save_covs
        self.seed = int(seed); self.output_summary_filename = output_summary_filename; self.summary_file_scope = summary_file_scope
        
        if self.seed != 0: np.random.seed(self.seed)

        self.cove_mat = np.array(cove_mat); self.f_mat = np.array(f_mat)
        self.s_mat = np.array(s_mat) if s_mat is not None else None
        self.a_mat = np.array(a_mat); self.d_mat = np.array(d_mat)
        
        self.mate_on_trait = mate_on_trait
        if self.mate_on_trait is not None: self.am_list = [float(m) for m in am_list]
        else: self.am_list = [np.array(m) for m in am_list]
        
        self.covy_mat = np.array(covy_mat); self.k2_matrix = np.array(k2_matrix)
        self.num_cvs = len(self.cv_info)
        
        self.phen_df, self.xo, self.xl = None, None, None
        self.summary_results = []; self.history = {'MATES': [], 'PHEN': [], 'XO': [], 'XL': []} if self.save_each_gen else None; self.covariances_log = [] if self.save_covs else None
        
        self.phen_column_names = ["ID", "Father.ID", "Mother.ID", "Fathers.Father.ID", "Fathers.Mother.ID", "Mothers.Father.ID", "Mothers.Mother.ID", "Sex", "AO_std1", "AO_std2", "AL_std1", "AL_std2", "AO1", "AO2", "AL1", "AL2", "F1", "F2", "E1", "E2", "Y1", "Y2", "Y1P", "Y2P", "Y1M", "Y2M", "F1P", "F2P", "F1M", "F2M", "TPO1", "TPO2", "TMO1", "TMO2", "NTPO1", "NTPO2", "NTMO1", "NTMO2", "TPL1", "TPL2", "TML1", "TML2", "NTPL1", "NTPL2", "NTML1", "NTML2"]
        
        self._initialize_generation_zero()

    # [Standard Helpers - Unchanged]
    def _is_positive_definite(self, matrix):
        if not isinstance(matrix, np.ndarray) or matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]: return False
        if not np.allclose(matrix, matrix.T): return False
        try: np.linalg.cholesky(matrix); return True
        except np.linalg.LinAlgError: return False
            
    def _make_positive_definite_nearest(self, matrix, min_eigenvalue=1e-6):
        A = (matrix + matrix.T) / 2
        try: np.linalg.cholesky(A); return A
        except np.linalg.LinAlgError: pass
        spacing = np.spacing(np.linalg.norm(A)); identity = np.eye(A.shape[0]); Y = A.copy()
        for iteration in range(100):
            R = Y - spacing * identity; eigenvalues, eigenvectors = np.linalg.eigh(R)
            eigenvalues[eigenvalues < min_eigenvalue] = min_eigenvalue; X = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
            dS = np.diag(np.diag(X)) - identity; Y = X - dS
            if np.linalg.norm(Y - A, 'fro') / np.linalg.norm(A, 'fro') < 1e-7: break
        Y = (Y + Y.T) / 2; np.fill_diagonal(Y, 1.0); return Y

    def _shrink_to_nearest_psd(self, matrix, min_eigenvalue=1e-6, max_shrinkage=0.5):
        A = (matrix + matrix.T) / 2; identity = np.eye(A.shape[0])
        low, high = 0.0, max_shrinkage; best_alpha = 0.0
        for _ in range(20):
            alpha = (low + high) / 2; A_shrunk = (1 - alpha) * A + alpha * identity
            if np.all(np.linalg.eigvalsh(A_shrunk) >= min_eigenvalue): best_alpha = alpha; high = alpha
            else: low = alpha
        result = (1 - best_alpha) * A + best_alpha * identity; np.fill_diagonal(result, 1.0); return result
        
    def _find_unique_closest_indices(self, dist_matrix_input, max_ord=100):
        dist_matrix = dist_matrix_input.copy(); n_to_match_from = dist_matrix.shape[0]; n_targets = dist_matrix.shape[1]     
        if n_targets == 0 or n_to_match_from == 0: return np.array([], dtype=int)
        rand_index_cols = np.random.permutation(n_targets); dist_matrix_shuffled_cols = dist_matrix[:, rand_index_cols]
        dist_ord_mat = np.argsort(dist_matrix_shuffled_cols, axis=0)
        new_closest_indices = np.full(n_targets, -1, dtype=int); assigned_individuals = set()
        for i in range(n_targets): 
            my_min_idx_val = -1; k = 0; match = False
            while k < max_ord and k < n_to_match_from:
                idx = dist_ord_mat[k, i]
                if idx not in assigned_individuals: my_min_idx_val = idx; match = True; break 
                k += 1
            if match: new_closest_indices[i] = my_min_idx_val; assigned_individuals.add(my_min_idx_val) 
        final = np.full(n_targets, -1, dtype=int); final = new_closest_indices[np.argsort(rand_index_cols)]
        return final

    def _assort_mate(self, phendata_df_current_gen, mating_type, pheno_mate_corr_target_xsim_mu, pop_size_target_offspring):
        # [Unchanged mating logic]
        ancestor_id_cols = ['Father.ID', 'Mother.ID', 'Fathers.Father.ID', 'Fathers.Mother.ID', 'Mothers.Father.ID', 'Mothers.Mother.ID']
        if mating_type == "phenotypic": cols = ["ID", "Sex", "Y1", "Y2"] + ancestor_id_cols
        elif mating_type == "social": cols = ["ID", "Sex", "F1", "F2", "E1", "E2"] + ancestor_id_cols
        elif mating_type == "genotypic": cols = ["ID", "Sex", "AO1", "AO2", "AL1", "AL2"] + ancestor_id_cols
        else: raise ValueError(f"Invalid mating_type: {mating_type}")
        
        males_slim = phendata_df_current_gen.loc[phendata_df_current_gen["Sex"]==1, cols].copy()
        females_slim = phendata_df_current_gen.loc[phendata_df_current_gen["Sex"]==0, cols].copy()
        if mating_type == "phenotypic":
            males_slim.rename(columns={"Y1": "mating1", "Y2": "mating2"}, inplace=True)
            females_slim.rename(columns={"Y1": "mating1", "Y2": "mating2"}, inplace=True)
        elif mating_type == "social":
            for df in [males_slim, females_slim]: df["mating1"] = df["F1"] + df["E1"]; df["mating2"] = df["F2"] + df["E2"]
        elif mating_type == "genotypic":
            for df in [males_slim, females_slim]: df["mating1"] = df["AO1"] + df["AL1"]; df["mating2"] = df["AO2"] + df["AL2"]

        nm, nf = len(males_slim), len(females_slim)
        if nm > nf: males_slim = males_slim.sample(n=nf, replace=False)
        elif nf > nm: females_slim = females_slim.sample(n=nm, replace=False)
        n_pairs = len(males_slim)
        if n_pairs == 0: return {'males.PHENDATA': pd.DataFrame(), 'females.PHENDATA': pd.DataFrame(), 'achieved_spousal_corr': np.eye(2)}

        if self.mate_on_trait is not None:
            single_trait = True; target_mu = float(pheno_mate_corr_target_xsim_mu)
            matcor = np.eye(2); matcor[0,1] = matcor[1,0] = target_mu
            xsim_dim = 2
        else:
            single_trait = False; target_mu = np.asarray(pheno_mate_corr_target_xsim_mu)
            cm = males_slim[['mating1','mating2']].corr().iloc[0,1] if n_pairs>1 else 0
            cf = females_slim[['mating1','mating2']].corr().iloc[0,1] if n_pairs>1 else 0
            matcor = np.eye(4); matcor[0,1]=matcor[1,0]=cm; matcor[2,3]=matcor[3,2]=cf
            matcor[0:2, 2:4] = target_mu; matcor[2:4, 0:2] = target_mu.T
            xsim_dim = 4

        if not self._is_positive_definite(matcor): matcor = self._make_positive_definite_nearest(matcor)
        x_sim = np.random.multivariate_normal(np.zeros(xsim_dim), matcor, size=n_pairs)
        if single_trait: x_sim_m, x_sim_f = x_sim[:,0:1], x_sim[:,1:2]
        else: x_sim_m, x_sim_f = x_sim[:,0:2], x_sim[:,2:4]

        def get_scaled(df, trait):
            if single_trait: vals = df[[f'mating{trait}']].values
            else: vals = df[['mating1','mating2']].values
            return (vals - vals.mean(0)) / (vals.std(0, ddof=1) + 1e-9)

        m_idx = self._find_unique_closest_indices(cdist(get_scaled(males_slim, self.mate_on_trait), x_sim_m))
        f_idx = self._find_unique_closest_indices(cdist(get_scaled(females_slim, self.mate_on_trait), x_sim_f))
        
        valid = (m_idx != -1) & (f_idx != -1)
        paired_m = males_slim.iloc[m_idx[valid]].reset_index(drop=True)
        paired_f = females_slim.iloc[f_idx[valid]].reset_index(drop=True)
        
        m_full = phendata_df_current_gen.loc[phendata_df_current_gen['ID'].isin(paired_m['ID'])].set_index('ID').loc[paired_m['ID']].reset_index()
        f_full = phendata_df_current_gen.loc[phendata_df_current_gen['ID'].isin(paired_f['ID'])].set_index('ID').loc[paired_f['ID']].reset_index()

        n_act = len(m_full)
        if n_act > 0:
            off_counts = np.random.poisson(pop_size_target_offspring/n_act, n_act)
            diff = pop_size_target_offspring - off_counts.sum()
            if diff > 0: off_counts[np.random.choice(n_act, diff, replace=True)] += 1
            elif diff < 0:
                eligible = np.where(off_counts > 0)[0]
                if len(eligible) > 0:
                    for _ in range(abs(diff)):
                        i = np.random.choice(eligible); off_counts[i] = max(0, off_counts[i]-1)
            m_full['num.offspring'] = off_counts; f_full['num.offspring'] = off_counts
            m_full['Spouse.ID'] = f_full['ID']; f_full['Spouse.ID'] = m_full['ID']
            
        return {'males.PHENDATA': m_full, 'females.PHENDATA': f_full, 'achieved_spousal_corr': np.eye(2)}

    def _calculate_genetic_values_masked(self, xo_matrix):
        """
        Calculates Observed and Latent genetic values using Masks.
        AO = Total_Geno * (Alpha * Mask)
        AL = Total_Geno * (Alpha * (1-Mask))
        """
        alphas = self.cv_info[['alpha1', 'alpha2']].values
        mask1 = self.cv_info['mask_obs1'].values
        mask2 = self.cv_info['mask_obs2'].values
        
        # Effective alphas for Observed
        # Element-wise multiplication of alpha column with mask column
        alpha_obs_1 = alphas[:, 0] * mask1
        alpha_obs_2 = alphas[:, 1] * mask2
        
        # Effective alphas for Latent
        alpha_lat_1 = alphas[:, 0] * (1 - mask1)
        alpha_lat_2 = alphas[:, 1] * (1 - mask2)
        
        # Matrix multiplication: (N_pop x N_snp) @ (N_snp x 1)
        ao1 = xo_matrix @ alpha_obs_1
        ao2 = xo_matrix @ alpha_obs_2
        al1 = xo_matrix @ alpha_lat_1
        al2 = xo_matrix @ alpha_lat_2
        
        return np.column_stack((ao1, ao2)), np.column_stack((al1, al2))

    def _reproduce(self, mates_dict, xo_parent_gen_full, xl_parent_gen_full, phendata_parent_gen_df_full):
        # NOTE: xo_parent_gen_full represents the TOTAL POOL now. xl_parent_gen_full is ignored.
        
        males_phen_producing = mates_dict['males.PHENDATA'][mates_dict['males.PHENDATA']['num.offspring'] > 0]
        females_phen_producing = mates_dict['females.PHENDATA'][mates_dict['females.PHENDATA']['num.offspring'] > 0]
        if males_phen_producing.empty: return {'PHEN': pd.DataFrame(columns=self.phen_column_names), 'XO': [], 'XL': []}

        id_map = {id_val: i for i, id_val in enumerate(phendata_parent_gen_df_full['ID'])}
        m_idx = [id_map[id_val] for id_val in males_phen_producing['ID']]
        f_idx = [id_map[id_val] for id_val in females_phen_producing['ID']]

        # Get Total Genotypes (Stored in XO variable)
        xo_m = xo_parent_gen_full[m_idx]
        xo_f = xo_parent_gen_full[f_idx]

        counts = males_phen_producing['num.offspring'].values; total = counts.sum()
        idx_rep_m = np.repeat(np.arange(len(m_idx)), counts); idx_rep_f = np.repeat(np.arange(len(f_idx)), counts)
        xo_m_rep = xo_m[idx_rep_m]
        xo_f_rep = xo_f[idx_rep_f]

        # Standard Transmission Logic
        def transmit(parent_geno):
            adder = np.random.randint(0, 2, size=parent_geno.shape)
            return (adder * (parent_geno == 1).astype(int)) + (parent_geno == 2).astype(int), adder
        
        mh_obs, m_add_o = transmit(xo_m_rep)
        fh_obs, f_add_o = transmit(xo_f_rep)
        xo_new = mh_obs + fh_obs 
        
        # XL is just a placeholder of zeros now
        xl_new = np.zeros_like(xo_new)

        # Calculate Haplotypes (Transmitted Paternal Observed, etc)
        # We need to apply masks to the haplotypes to get the "Observed" portion
        alphas = self.cv_info[['alpha1', 'alpha2']].values
        mask1 = self.cv_info['mask_obs1'].values; mask2 = self.cv_info['mask_obs2'].values
        alpha_obs_1 = alphas[:, 0] * mask1; alpha_obs_2 = alphas[:, 1] * mask2
        alpha_lat_1 = alphas[:, 0] * (1 - mask1); alpha_lat_2 = alphas[:, 1] * (1 - mask2)
        
        # TPO: Transmitted Paternal Observed
        tpo1 = mh_obs @ alpha_obs_1; tpo2 = mh_obs @ alpha_obs_2
        tmo1 = fh_obs @ alpha_obs_1; tmo2 = fh_obs @ alpha_obs_2
        
        # NTPO: Non-Transmitted Paternal Observed
        mnt_obs = ((1-m_add_o)*(xo_m_rep==1).astype(int)) + (xo_m_rep==2).astype(int)
        fnt_obs = ((1-f_add_o)*(xo_f_rep==1).astype(int)) + (xo_f_rep==2).astype(int)
        ntpo1 = mnt_obs @ alpha_obs_1; ntpo2 = mnt_obs @ alpha_obs_2
        ntmo1 = fnt_obs @ alpha_obs_1; ntmo2 = fnt_obs @ alpha_obs_2

        # TPL: Transmitted Paternal Latent
        tpl1 = mh_obs @ alpha_lat_1; tpl2 = mh_obs @ alpha_lat_2
        tml1 = fh_obs @ alpha_lat_1; tml2 = fh_obs @ alpha_lat_2
        ntpl1 = mnt_obs @ alpha_lat_1; ntpl2 = mnt_obs @ alpha_lat_2
        ntml1 = fnt_obs @ alpha_lat_1; ntml2 = fnt_obs @ alpha_lat_2

        # Final Scores
        ao_new_raw, al_new_raw = self._calculate_genetic_values_masked(xo_new)
        
        fathers = males_phen_producing.iloc[idx_rep_m].reset_index(drop=True)
        mothers = females_phen_producing.iloc[idx_rep_f].reset_index(drop=True)
        
        f_y = (fathers[['Y1','Y2']].values @ self.f_mat.T) + (mothers[['Y1','Y2']].values @ self.f_mat.T)
        e_new = np.random.multivariate_normal(np.zeros(2), self.cove_mat, size=total)
        
        # Y = AO + AL + F + E
        y_new = ao_new_raw + al_new_raw + f_y + e_new

        new_df = pd.DataFrame(index=np.arange(total), columns=self.phen_column_names)
        max_id = phendata_parent_gen_df_full['ID'].max()
        new_df['ID'] = np.arange(1, total+1) + max_id
        new_df['Father.ID'] = fathers['ID'].values; new_df['Mother.ID'] = mothers['ID'].values
        new_df['Sex'] = np.random.permutation(np.concatenate([np.zeros(total//2), np.ones(total-total//2)]))
        
        new_df[['AO1','AO2']] = ao_new_raw; new_df[['AL1','AL2']] = al_new_raw
        new_df[['F1','F2']] = f_y; new_df[['E1','E2']] = e_new
        new_df[['Y1','Y2']] = y_new
        
        new_df['TPO1'] = tpo1; new_df['TPO2'] = tpo2
        new_df['TMO1'] = tmo1; new_df['TMO2'] = tmo2
        new_df['NTPO1'] = ntpo1; new_df['NTPO2'] = ntpo2
        new_df['NTMO1'] = ntmo1; new_df['NTMO2'] = ntmo2
        
        new_df['TPL1'] = tpl1; new_df['TPL2'] = tpl2
        new_df['TML1'] = tml1; new_df['TML2'] = tml2
        new_df['NTPL1'] = ntpl1; new_df['NTPL2'] = ntpl2
        new_df['NTML1'] = ntml1; new_df['NTML2'] = ntml2
        
        return {'PHEN': new_df, 'XO': xo_new, 'XL': xl_new}
    
    def _initialize_generation_zero(self):
        print("Initializing Generation 0 (Founders)...")
        n_pop = self.pop_vector[0]
        maf = self.cv_info['maf'].values
        
        # Genotype Pool (XO holds everything)
        self.xo = np.random.binomial(2, maf, size=(n_pop, self.num_cvs))
        self.xl = np.zeros_like(self.xo) # Placeholder

        ao_raw, al_raw = self._calculate_genetic_values_masked(self.xo)
        
        f_gen0 = np.zeros((n_pop, 2))
        e_gen0 = np.random.multivariate_normal(np.zeros(2), self.cove_mat, size=n_pop)
        y_gen0 = ao_raw + al_raw + f_gen0 + e_gen0
        
        self.phen_df = pd.DataFrame(index=np.arange(n_pop), columns=self.phen_column_names)
        self.phen_df['ID'] = np.arange(1000000, 1000000+n_pop)
        self.phen_df['Sex'] = np.random.permutation(np.concatenate([np.zeros(n_pop//2), np.ones(n_pop-n_pop//2)]))
        self.phen_df[['AO1','AO2']] = ao_raw; self.phen_df[['AL1','AL2']] = al_raw
        self.phen_df[['F1','F2']] = f_gen0; self.phen_df[['E1','E2']] = e_gen0
        self.phen_df[['Y1','Y2']] = y_gen0
        for c in self.phen_column_names:
            if c not in self.phen_df.columns: self.phen_df[c] = np.nan
            
        if self.save_each_gen: self.history['MATES'].append(None); self.history['PHEN'].append(self.phen_df.copy()); self.history['XO'].append(self.xo.copy()); self.history['XL'].append(self.xl.copy())
        self.summary_results.append({'GEN': 0, 'POPSIZE': n_pop})
        print("Generation 0 initialized.")

    # [Standard Utils - Unchanged]
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
        cov_keys = ['covG', 'covH', 'covI', 'omega', 'gamma', 'w', 'v', 'covF', 'covE', 'thetaNT', 'thetaT'] 
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
                    for i in range(len(self.summary_results) - 1): 
                        gen_summary_dict = self.summary_results[i]
                        f.write(f"\n--- Generation {gen_summary_dict.get('GEN', 'N/A')} Summary ---\n")
                        f.write(self._format_single_generation_summary(gen_summary_dict))
                        f.write("\n")
                f.write("\n--- End of Summary File ---\n")
            print(f"Summary written to {self.output_summary_filename}")
        except IOError as e: print(f"Error writing summary to file {self.output_summary_filename}: {e}")
        except Exception as e: print(f"An unexpected error occurred while writing summary: {e}; {type(e)}")

    def run_simulation(self):
            print(f"Starting simulation for {self.num_generations} generations using '{self.mating_type}' assortment.")
            if self.phen_df is None: return None
            for cur_gen_py_idx in range(self.num_generations): 
                r_cur_gen_num = cur_gen_py_idx + 1 
                print(f"\n--- Beginning Generation {r_cur_gen_num} ---")
                pop_size_target_for_offspring = self.pop_vector[cur_gen_py_idx]
                mate_cor_mu_for_current_parents = self.am_list[cur_gen_py_idx]
                
                print(f"Generation {r_cur_gen_num}: Performing '{self.mating_type}' assortative mating...")
                mates = self._assort_mate(self.phen_df, self.mating_type, mate_cor_mu_for_current_parents, pop_size_target_for_offspring)
                num_m_mated = len(mates['males.PHENDATA'])
                if num_m_mated == 0: break 
                
                print(f"Generation {r_cur_gen_num}: Simulating reproduction...")
                offspring_data = self._reproduce(mates, self.xo, self.xl, self.phen_df)
                if offspring_data['PHEN'].empty: break
                
                self.xo = offspring_data['XO']; self.xl = offspring_data['XL']; self.phen_df = offspring_data['PHEN']

                print(f"Generation {r_cur_gen_num}: Calculating summary statistics...")
                temp_phen_df_for_cov = self.phen_df.copy()
                if "NTPO1" in temp_phen_df_for_cov:
                    temp_phen_df_for_cov["BV.NT.O1"] = temp_phen_df_for_cov["NTPO1"] + temp_phen_df_for_cov["NTMO1"]
                    temp_phen_df_for_cov["BV.NT.O2"] = temp_phen_df_for_cov["NTPO2"] + temp_phen_df_for_cov["NTMO2"]
                
                cols_for_full_cov_calc = ['TPO1','TMO1','NTPO1','NTMO1','TPL1','TML1','NTPL1','NTML1','TPO2','TMO2','NTPO2','NTMO2','TPL2','TML2','NTPL2','NTML2','AO1','AO2','AL1','AL2','F1','F2','E1','E2',"BV.NT.O1","BV.NT.O2",'Y1','Y2','Y1P','Y2P','Y1M','Y2M','F1P','F2P','F1M','F2M']
                valid_cols = [col for col in cols_for_full_cov_calc if col in temp_phen_df_for_cov.columns]
                full_cov_df = pd.DataFrame()
                if len(valid_cols) > 1: full_cov_df = temp_phen_df_for_cov[valid_cols].cov()

                # Initialize defaults
                nan_2x2 = np.full((2,2), np.nan)
                covG_val, covH_val, covI_val, w_val, v_val = nan_2x2.copy(), nan_2x2.copy(), nan_2x2.copy(), nan_2x2.copy(), nan_2x2.copy()
                covF_calc_val, covE_calc_val = nan_2x2.copy(), nan_2x2.copy()
                omega_val, gamma_val, thetaNT_val, thetaT_val = nan_2x2.copy(), nan_2x2.copy(), nan_2x2.copy(), nan_2x2.copy()
                hapsO_covs_dict, hapsL_covs_dict = np.nan, np.nan

                if not full_cov_df.empty:
                    # G (Observed Haps)
                    o_cols = ['TPO1','TMO1','NTPO1','NTMO1', 'TPO2','TMO2','NTPO2','NTMO2']
                    if all(c in full_cov_df.index for c in o_cols):
                        hops = full_cov_df.loc[o_cols, o_cols]
                        hapsO_covs_dict = hops.to_dict()
                
                # Standard Summary construction
                if r_cur_gen_num < len(self.am_list): mc = self.am_list[r_cur_gen_num].tolist() if hasattr(self.am_list[r_cur_gen_num], 'tolist') else self.am_list[r_cur_gen_num]
                else: mc = self.am_list[-1].tolist() if hasattr(self.am_list[-1], 'tolist') else self.am_list[-1]
                
                summary_this_gen = {'GEN': r_cur_gen_num, 'NUM.CVs': self.num_cvs, 'MATE.COR': mc, 'POPSIZE': len(self.phen_df)}
                for comp_name, cols in {"VAO": ["AO1", "AO2"], "VAL": ["AL1", "AL2"], "VF": ["F1", "F2"], "VE": ["E1", "E2"], "VP": ["Y1", "Y2"]}.items():
                    if all(c in self.phen_df.columns for c in cols):
                        cov_data = self.phen_df[cols].dropna().values
                        if len(cov_data) > 1:
                            summary_this_gen[comp_name] = np.cov(cov_data, rowvar=False).tolist()
                        else:
                            summary_this_gen[comp_name] = np.full((2,2),np.nan).tolist()
                    else: summary_this_gen[comp_name] = np.full((2,2),np.nan).tolist()
                
                vp_diag = np.diag(np.array(summary_this_gen.get('VP', [[np.nan,np.nan]])))
                vao_diag = np.diag(np.array(summary_this_gen.get('VAO', [[np.nan,np.nan]])))
                val_diag = np.diag(np.array(summary_this_gen.get('VAL', [[np.nan,np.nan]])))
                with np.errstate(divide='ignore', invalid='ignore'):
                    summary_this_gen['h2'] = ((vao_diag + val_diag) / vp_diag).tolist()
                    summary_this_gen['h2.obs'] = (vao_diag / vp_diag).tolist()
                    summary_this_gen['h2.lat'] = (val_diag / vp_diag).tolist()
                
                self.summary_results.append(summary_this_gen)
                if self.save_each_gen: self.history['MATES'].append(mates); self.history['PHEN'].append(self.phen_df.copy()); self.history['XO'].append(self.xo.copy()); self.history['XL'].append(self.xl.copy())
                if self.save_covs: self.covariances_log.append(full_cov_df.round(3).to_dict() if not full_cov_df.empty else None)
                print(f"--- Generation {r_cur_gen_num} Processing Done ---")
                if self.output_summary_filename: self._write_simulation_summary_to_file()
                
            return {'SUMMARY.RES': self.summary_results, 'XO': self.xo, 'XL': self.xl, 'PHEN': self.phen_df, 'HISTORY': self.history, 'COVARIANCES': self.covariances_log}