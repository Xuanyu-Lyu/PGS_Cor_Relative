# The script is designed to find distant relatives based on the simulated data for multiple generations.
# This version is MODIFIED to use a set-based pandas.merge approach for high performance,
# replacing the original graph-traversal algorithm.

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, save_npz, load_npz
import os
from collections import defaultdict

# --- Utility Functions (Copied from original script) ---
# These functions are all excellent and are preserved.

def extract_genealogy_info(results, generations=None):
    """
    Extract genealogical information (ID, Father.ID, Mother.ID, Spouse.ID) from simulation results
    across specified generations.
    
    Args:
        results (dict): The dictionary returned by AssortativeMatingSimulation.run_simulation().
                       Must contain 'HISTORY' with 'PHEN' and 'MATES' data.
        generations (list or int or None): Specific generation(s) to extract.
                                          - If None: extract all generations
                                          - If int: extract single generation
                                          - If list: extract specified generations
    
    Returns:
        pd.DataFrame: A concatenated dataframe with columns:
                     ['ID', 'Father.ID', 'Mother.ID', 'Spouse.ID', 'Generation']
                     where Generation indicates which generation the individual belongs to.
    
    Raises:
        ValueError: If results don't contain necessary history data or if generation indices are invalid.
    
    Example:
        >>> # Extract genealogy for all generations
        >>> genealogy_df = extract_genealogy_info(results)
        >>> 
        >>> # Extract genealogy for specific generations (e.g., 5, 10, 15)
        >>> genealogy_df = extract_genealogy_info(results, generations=[5, 10, 15])
        >>> 
        >>> # Extract genealogy for a single generation
        >>> genealogy_df = extract_genealogy_info(results, generations=10)
    """
    
    # Validate that results contain history data
    if not results or 'HISTORY' not in results:
        raise ValueError("Results must contain 'HISTORY' data. "
                        "Ensure simulation was run with save_each_gen=True.")
    
    history = results['HISTORY']
    
    if 'PHEN' not in history or not history['PHEN']:
        raise ValueError("HISTORY must contain 'PHEN' data.")
    
    if 'MATES' not in history or not history['MATES']:
        raise ValueError("HISTORY must contain 'MATES' data.")
    
    num_generations = len(history['PHEN'])
    
    # Determine which generations to extract
    if generations is None:
        # Extract all generations
        generations_to_extract = list(range(num_generations))
    elif isinstance(generations, int):
        # Single generation
        generations_to_extract = [generations]
    elif isinstance(generations, (list, tuple)):
        # Multiple specified generations
        generations_to_extract = sorted(list(set(generations)))
    else:
        raise ValueError("generations must be None, an integer, or a list/tuple of integers.")
    
    # Validate generation indices
    for gen in generations_to_extract:
        if not (0 <= gen < num_generations):
            raise ValueError(f"Generation {gen} is out of range. "
                           f"Valid range: 0 to {num_generations - 1}")
    
    # List to store dataframes for each generation
    gen_dataframes = []
    
    for gen_idx in generations_to_extract:
        # Get phenotype data for this generation
        phen_df = history['PHEN'][gen_idx]
        
        # Get mates data for this generation
        # MATES[gen_idx+1] contains the mating pairs FROM generation gen_idx that produce generation gen_idx+1
        # MATES[0] is None (founders), MATES[1] has gen 0 mates, MATES[2] has gen 1 mates, etc.
        mates_dict = None
        if gen_idx + 1 < len(history['MATES']):
            mates_dict = history['MATES'][gen_idx + 1]
        
        # Extract basic genealogy columns from phenotype data
        if phen_df is not None and not phen_df.empty:
            gen_genealogy = phen_df[['ID', 'Father.ID', 'Mother.ID']].copy()
            
            # Initialize Spouse.ID column with NaN
            gen_genealogy['Spouse.ID'] = np.nan
            
            # Add spouse information from mates data if available
            if mates_dict is not None:
                males_df = mates_dict.get('males.PHENDATA')
                females_df = mates_dict.get('females.PHENDATA')
                
                # Create a mapping from ID to Spouse.ID
                spouse_mapping = {}
                
                if males_df is not None and not males_df.empty:
                    if 'Spouse.ID' in males_df.columns:
                        for _, row in males_df.iterrows():
                            spouse_mapping[row['ID']] = row['Spouse.ID']
                
                if females_df is not None and not females_df.empty:
                    if 'Spouse.ID' in females_df.columns:
                        for _, row in females_df.iterrows():
                            spouse_mapping[row['ID']] = row['Spouse.ID']
                
                # Map spouse IDs to the genealogy dataframe
                gen_genealogy['Spouse.ID'] = gen_genealogy['ID'].map(spouse_mapping)
            
            # Add generation identifier
            gen_genealogy['Generation'] = gen_idx
            
            gen_dataframes.append(gen_genealogy)
    
    # Concatenate all generation dataframes
    if not gen_dataframes:
        raise ValueError("No valid genealogy data found for specified generations.")
    
    combined_genealogy = pd.concat(gen_dataframes, ignore_index=True)
    
    # Reorder columns for clarity
    combined_genealogy = combined_genealogy[['Generation', 'ID', 'Father.ID', 'Mother.ID', 'Spouse.ID']]
    
    return combined_genealogy


def get_genealogy_summary(genealogy_df):
    """
    Get summary statistics from the genealogy dataframe.
    
    Args:
        genealogy_df (pd.DataFrame): Output from extract_genealogy_info()
    
    Returns:
        dict: Summary statistics including:
              - total_individuals: Total number of individuals
              - individuals_per_generation: Count per generation
              - mated_individuals: Number with spouse information
              - mating_rate_per_generation: Proportion mated per generation
    """
    summary = {
        'total_individuals': len(genealogy_df),
        'individuals_per_generation': genealogy_df.groupby('Generation').size().to_dict(),
        'mated_individuals': genealogy_df['Spouse.ID'].notna().sum(),
        'mating_rate_per_generation': genealogy_df.groupby('Generation')['Spouse.ID'].apply(
            lambda x: x.notna().sum() / len(x) if len(x) > 0 else 0
        ).to_dict()
    }
    
    return summary


# --- NEW Set-Based Relationship Finder ---
# This class replaces the PedigreeResolver
class SetBasedRelFinder:
    """
    Finds complex relationships using a fast, set-based merge approach
    comparable to R's data.table.
    """
    
    def __init__(self, genealogy_df):
        """
        Pre-processes the genealogy DataFrame into fast-lookup *DataFrames*.
        
        Args:
            genealogy_df: DataFrame with columns ['ID', 'Mother.ID', 'Father.ID', 'Spouse.ID']
                         NaN/None represents unknown/no one.
        """
        self.base_dfs = {}
        
        # Clean the input dataframe: Replace NaN with 0 and ensure integer IDs
        ped = genealogy_df[['ID', 'Mother.ID', 'Father.ID', 'Spouse.ID']].copy()
        ped = ped.fillna(0)
        for col in ped.columns:
            ped[col] = ped[col].astype(int)
        
        # Get all unique IDs
        self.all_ids = set(ped['ID'])
        print(f"SetBasedRelFinder initialized with {len(self.all_ids):,} unique individuals.")

        # --- 1. 'P' (Parent) DataFrame ---
        # Asymmetric: person -> relative (parent)
        p_mother = ped.loc[ped['Mother.ID'] != 0, ['ID', 'Mother.ID']].rename(columns={'ID': 'person', 'Mother.ID': 'relative'})
        p_father = ped.loc[ped['Father.ID'] != 0, ['ID', 'Father.ID']].rename(columns={'ID': 'person', 'Father.ID': 'relative'})
        self.base_dfs['P'] = pd.concat([p_mother, p_father], ignore_index=True).drop_duplicates()
        
        # --- 2. 'C' (Child) DataFrame ---
        # Asymmetric: person -> relative (child)
        c_mother = ped.loc[ped['Mother.ID'] != 0, ['Mother.ID', 'ID']].rename(columns={'Mother.ID': 'person', 'ID': 'relative'})
        c_father = ped.loc[ped['Father.ID'] != 0, ['Father.ID', 'ID']].rename(columns={'Father.ID': 'person', 'ID': 'relative'})
        self.base_dfs['C'] = pd.concat([c_mother, c_father], ignore_index=True).drop_duplicates()

        # --- 3. 'M' (Mate) DataFrame ---
        # Symmetric: person <-> relative
        m_long = ped.loc[ped['Spouse.ID'] != 0, ['ID', 'Spouse.ID']].rename(columns={'ID': 'person', 'Spouse.ID': 'relative'})
        m_sym = m_long.rename(columns={'person': 'relative', 'relative': 'person'})
        self.base_dfs['M'] = pd.concat([m_long, m_sym], ignore_index=True).drop_duplicates()

        # --- 4. 'S' (Full Sibling) DataFrame ---
        # Symmetric: person <-> relative
        # Find all people with *both* parents known
        parent_pairs = ped.loc[(ped['Mother.ID'] != 0) & (ped['Father.ID'] != 0), ['ID', 'Mother.ID', 'Father.ID']]
        # Join this table to itself on (Mother.ID, Father.ID)
        sibs = pd.merge(parent_pairs, parent_pairs, on=['Mother.ID', 'Father.ID'])
        # Filter out self-comparisons
        sibs = sibs.query('ID_x != ID_y')
        sibs = sibs[['ID_x', 'ID_y']].rename(columns={'ID_x': 'person', 'ID_y': 'relative'})
        self.base_dfs['S'] = sibs.drop_duplicates()

        print("Base relationship DataFrames (M, P, C, S) created.")
        
        # Cache to store results of intermediate paths (e.g., "MS")
        self.cache = {}
        
    def _deduplicate(self, df, col1='p1', col2='p2'):
        """
        Deduplicates a dataframe of pairs (p1, p2) using symmetric logic.
        Assumes integer IDs.
        """
        # Create a canonical pair ID by ordering
        pair_id_df = pd.DataFrame({
            'p_max': np.maximum(df[col1], df[col2]),
            'p_min': np.minimum(df[col1], df[col2])
        })
        
        # drop_duplicates is the Python equivalent of R's !duplicated()
        return df[~pair_id_df.duplicated()]

    def find_relationship(self, path, show_progress=False):
        """
        Finds all (p1, p2) pairs related by the given path string
        using a chain of merge operations.
        
        Returns a DataFrame with columns ['p1', 'p2']
        """
        if path in self.cache:
            return self.cache[path].copy()

        if show_progress:
            print(f"  Calculating path: {path}")

        if len(path) == 1:
            # Base case: Just return the base relationship
            base_df = self.base_dfs[path[0]]
            # Standardize column names
            result_df = base_df.rename(columns={'person': 'p1', 'relative': 'p2'})
        else:
            # Recursive step:
            # 1. Get the result for the path "all but the last step"
            # (e.g., "MS" for "MSC")
            prev_path = path[:-1]
            prev_rels_df = self.find_relationship(prev_path, show_progress=show_progress) # Cols: 'p1', 'p2'
            
            # 2. Get the DataFrame for the last step (e.g., "C")
            last_step_df = self.base_dfs[path[-1]] # Cols: 'person', 'relative'

            if show_progress:
                print(f"    Merging '{prev_path}' ({len(prev_rels_df):,} pairs) with '{path[-1]}' ({len(last_step_df):,} pairs)...")

            # 3. Merge them using pandas.merge (the fast hash-join)
            # This is the core operation.
            # We join where the end of the first path ('p2')
            # is the start of the next path ('person').
            merged_df = pd.merge(
                prev_rels_df, 
                last_step_df, 
                left_on='p2', 
                right_on='person'
            )
            
            # The new pairs are the start of the first path ('p1') 
            # and the end of the last path ('relative')
            result_df = merged_df[['p1', 'relative']].rename(columns={'relative': 'p2'})

        # --- Deduplication ---
        # Remove pairs that are related to themselves
        result_df = result_df.query('p1 != p2').copy()
        
        # Apply symmetric deduplication
        # Note: This may not be strictly necessary for all paths, but it
        # mimics the R script's logic and keeps dataframes smaller.
        final_result = self._deduplicate(result_df)
        
        if show_progress:
             print(f"  Path '{path}' complete. Found {len(final_result):,} unique pairs.")

        self.cache[path] = final_result
        return final_result


    def find_all_relatives_matrix(self, path_string, sparse=False, show_progress=True):
        """
        Generates a 0/1 matrix for all individuals for a given relationship path.
        
        This function is much faster as it uses set-based merges, not iteration.
        The `n_jobs` parameter is not used as this algorithm is not parallel
        in the same way as graph-traversal.
        
        Args:
            path_string: A string of steps (e.g., "PSC" for cousins).
            sparse: If True, returns a scipy sparse CSR matrix. If False, returns dense pandas DataFrame.
            show_progress: If True, prints progress updates.
            
        Returns:
            If sparse=False: pd.DataFrame - A square matrix where rows and columns are IDs.
            If sparse=True: tuple of (csr_matrix, id_list) - Sparse matrix and corresponding ID list.
        """
        ids = sorted(list(self.all_ids))
        id_to_idx = {id_val: i for i, id_val in enumerate(ids)}
        n = len(ids)
        
        # --- 1. Find all pairs using the fast merge-based method ---
        if show_progress:
            print(f"Processing path '{path_string}' for {n:,} individuals...")
        
        pairs_df = self.find_relationship(path_string, show_progress=show_progress)
        
        if show_progress:
            print(f"Found {len(pairs_df):,} total pairs for path '{path_string}'.")
            print("Constructing matrix...")
        
        # --- 2. Build Sparse Matrix from the (p1, p2) pair list ---
        # This is very fast and memory-efficient
        
        # Map IDs to matrix indices
        # Use .get() to avoid errors if an ID in pairs_df is not in all_ids (though it should be)
        rows = pairs_df['p1'].map(id_to_idx.get)
        cols = pairs_df['p2'].map(id_to_idx.get)
        
        # Drop any pairs that couldn't be mapped
        valid_pairs = rows.notna() & cols.notna()
        if not valid_pairs.all():
            rows = rows[valid_pairs]
            cols = cols[valid_pairs]
        
        data = np.ones(len(rows), dtype=np.int8)
        
        # Create sparse matrix in COO format, then convert to CSR
        sparse_matrix = csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.int8)
        
        if show_progress:
            print("Matrix construction complete.")
        
        # --- 3. Return in the requested format ---
        if sparse:
            return sparse_matrix, ids
        else:
            # Convert sparse to dense
            if show_progress:
                print(f"Converting sparse matrix to dense DataFrame ({n}x{n})...")
            matrix = sparse_matrix.toarray()
            return pd.DataFrame(matrix, index=ids, columns=ids)


# --- MAIN API Functions (Modified to use new Finder) ---

def find_relationship_pairs(results, relationship_path, generations=None, output_format='long', use_sparse=None, n_jobs=None):
    """
    Find all pairs of individuals with a specified relationship across generations.
    
    This function now uses the high-performance SetBasedRelFinder.
    The 'n_jobs' parameter is ignored, as the set-based algorithm is not parallel
    at the Python level (its speed comes from optimized C-level merges).
    
    Args:
        results (dict): The dictionary returned by AssortativeMatingSimulation.run_simulation().
        relationship_path (str): A string of relationship steps to follow.
                                Step codes:
                                - 'M': Mate/Spouse
                                - 'P': Parents
                                - 'C': Children
                                - 'S': Siblings (full siblings)
                                
                                Examples:
                                - "P" -> Parents
                                - "S" -> Siblings
                                - "PS" -> Parent's Siblings (aunts/uncles)
                                - "PSC" -> Parent's Sibling's Children (first cousins)
                                - "PSCM" -> Parent's Sibling's Child's Mate (cousin's spouse)
                                - "PSCC" -> Parent's Sibling's Child's Children (first cousin once removed)
                                - "PPSC" -> Parent's Parent's Sibling's Children (parent's cousins)
                                
        generations (list, int, or None): Which generation(s) to include in the analysis.
                                         - None: all generations
                                         - int: single generation
                                         - list: multiple specific generations
                                         
        output_format (str): Output format, either 'long', 'matrix', or 'sparse'.
                            - 'long' (default): Returns a DataFrame with columns ['Person_ID', 'Relative_ID', 'Relationship']
                            - 'matrix': Returns a dense square matrix (pandas DataFrame)
                            - 'sparse': Returns a tuple (sparse_matrix, id_list) for memory efficiency
                            
        use_sparse (bool or None): If None, automatically decides based on dataset size.
                                   If True, forces sparse matrix internally (recommended for >50k individuals).
                                   If False, uses dense matrix.
                                   
        n_jobs (int or None): This parameter is now IGNORED by the new set-based algorithm.
                            
    Returns:
        Depends on output_format:
        - 'long': pd.DataFrame with relationship pairs
        - 'matrix': pd.DataFrame square matrix
        - 'sparse': tuple of (csr_matrix, id_list)
        
    Examples:
        >>> # Find all first cousins (PSC) in generation 10
        >>> cousins = find_relationship_pairs(results, "PSC", generations=10, output_format='long')
        >>> 
        >>> # Find all cousin spouses (PSCM) across all generations as a sparse matrix
        >>> sparse_mat, ids = find_relationship_pairs(results, "PSCM", output_format='sparse')
        >>> 
        >>> # Find all siblings in generations 5, 10, and 15
        >>> siblings = find_relationship_pairs(results, "S", generations=[5, 10, 15])
    """
    
    if n_jobs is not None and n_jobs > 1:
        print(f"Note: 'n_jobs' parameter is ignored by the new set-based merge algorithm.")
        
    # Step 1: Extract genealogy information (unchanged)
    print("Step 1/3: Extracting genealogy...")
    genealogy_df = extract_genealogy_info(results, generations=generations)
    
    # Step 2: Create SetBasedRelFinder (NEW)
    print("Step 2/3: Initializing SetBasedRelFinder (building base M,P,C,S tables)...")
    resolver = SetBasedRelFinder(genealogy_df)
    
    # Step 3: Determine whether to use sparse matrix (unchanged)
    num_individuals = len(resolver.all_ids)
    if use_sparse is None:
        # Auto-detect: use sparse for large datasets (>50,000 individuals)
        use_sparse_internal = num_individuals > 50000
    else:
        use_sparse_internal = use_sparse
    
    if use_sparse_internal and output_format.lower() != 'sparse':
        print(f"Using sparse matrix internally for {num_individuals:,} individuals to optimize memory.")
    
    # Step 4: Generate relationship matrix (using new resolver)
    print(f"Step 3/3: Finding all pairs for path '{relationship_path}'...")
    
    # Decide which format to *generate* internally
    # We always generate sparse first, as it's the most efficient
    # The 'sparse' flag in find_all_relatives_matrix just controls the *return* format
    
    if output_format.lower() == 'sparse':
        # Generate and return sparse matrix
        sparse_matrix, ids = resolver.find_all_relatives_matrix(relationship_path, sparse=True, show_progress=True)
        return sparse_matrix, ids
        
    elif output_format.lower() == 'matrix':
        # Generate dense matrix (which internally builds sparse first)
        relationship_matrix = resolver.find_all_relatives_matrix(relationship_path, sparse=False, show_progress=True)
        return relationship_matrix
        
    elif output_format.lower() == 'long':
        # Generate sparse matrix (most-efficient intermediate)
        sparse_matrix, ids = resolver.find_all_relatives_matrix(relationship_path, sparse=True, show_progress=True)
        
        # Convert sparse matrix to long format
        print(f"Converting {sparse_matrix.nnz:,} found pairs to long DataFrame...")
        pairs_df = sparse_matrix_to_long_format(sparse_matrix, ids, relationship_path)
        print("Conversion complete.")
        return pairs_df
    
    raise ValueError(f"output_format must be 'long', 'matrix', or 'sparse', got '{output_format}'")


# --- Utility Functions (Copied from original script) ---
# These functions are all excellent and are preserved.

def save_sparse_relationship_matrix(sparse_matrix, ids, filepath, relationship_path=None):
    """
    Save a sparse relationship matrix to disk in an efficient format.
    
    Args:
        sparse_matrix: scipy.sparse.csr_matrix - The sparse relationship matrix
        ids: list - List of individual IDs corresponding to matrix rows/cols
        filepath: str - Path where to save the matrix (without extension)
        relationship_path: str or None - Optional relationship path descriptor to save as metadata
        
    Returns:
        None
        
    Example:
        >>> sparse_mat, ids = find_relationship_pairs(results, "PSC", output_format='sparse')
        >>> save_sparse_relationship_matrix(sparse_mat, ids, "output/cousins_gen10", "PSC")
        >>> # Saves to: output/cousins_gen10.npz and output/cousins_gen10_ids.csv
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Save the sparse matrix
    matrix_file = f"{filepath}.npz"
    save_npz(matrix_file, sparse_matrix)
    print(f"Sparse matrix saved to: {matrix_file}")
    
    # Save the IDs mapping
    ids_file = f"{filepath}_ids.csv"
    ids_df = pd.DataFrame({'index': range(len(ids)), 'ID': ids})
    if relationship_path:
        ids_df['relationship'] = relationship_path
    ids_df.to_csv(ids_file, index=False)
    print(f"ID mapping saved to: {ids_file}")
    
    # Print statistics
    n_relationships = sparse_matrix.nnz
    n_individuals = len(ids)
    sparsity = 100 * (1 - n_relationships / (n_individuals ** 2))
    print(f"Matrix statistics:")
    print(f"  - Size: {n_individuals:,} x {n_individuals:,}")
    print(f"  - Relationships found: {n_relationships:,}")
    print(f"  - Sparsity: {sparsity:.2f}%")
    
    # Estimate size savings
    dense_size_mb = (n_individuals ** 2) / (1024 ** 2)  # Assuming 1 byte per element
    sparse_size_mb = 0
    if os.path.exists(matrix_file):
        sparse_size_mb = os.path.getsize(matrix_file) / (1024 ** 2)
    
    print(f"  - Dense matrix would be: {dense_size_mb:.2f} MB")
    print(f"  - Sparse matrix is: {sparse_size_mb:.2f} MB")
    if dense_size_mb > 0:
        print(f"  - Space savings: {100 * (1 - sparse_size_mb / dense_size_mb):.1f}%")


def load_sparse_relationship_matrix(filepath):
    """
    Load a sparse relationship matrix from disk.
    
    Args:
        filepath: str - Path to the saved matrix (without extension)
        
    Returns:
        tuple: (sparse_matrix, ids, relationship_path) where:
               - sparse_matrix is a scipy.sparse.csr_matrix
               - ids is a list of individual IDs
               - relationship_path is the relationship descriptor (if saved)
        
    Example:
        >>> sparse_mat, ids, rel_path = load_sparse_relationship_matrix("output/cousins_gen10")
        >>> # Convert to long format if needed
        >>> pairs_df = sparse_matrix_to_long_format(sparse_mat, ids, rel_path)
    """
    # Load the sparse matrix
    matrix_file = f"{filepath}.npz"
    if not os.path.exists(matrix_file):
        raise FileNotFoundError(f"Matrix file not found: {matrix_file}")
    sparse_matrix = load_npz(matrix_file)
    print(f"Sparse matrix loaded from: {matrix_file}")
    
    # Load the IDs mapping
    ids_file = f"{filepath}_ids.csv"
    if not os.path.exists(ids_file):
        raise FileNotFoundError(f"IDs file not found: {ids_file}")
    ids_df = pd.read_csv(ids_file)
    ids = ids_df['ID'].tolist()
    relationship_path = ids_df['relationship'].iloc[0] if 'relationship' in ids_df.columns else None
    print(f"ID mapping loaded from: {ids_file}")
    
    # Print statistics
    n_relationships = sparse_matrix.nnz
    n_individuals = len(ids)
    print(f"Matrix statistics:")
    print(f"  - Size: {n_individuals:,} x {n_individuals:,}")
    print(f"  - Relationships found: {n_relationships:,}")
    if relationship_path:
        print(f"  - Relationship type: {relationship_path}")
    
    return sparse_matrix, ids, relationship_path


def sparse_matrix_to_long_format(sparse_matrix, ids, relationship_path=None):
    """
    Convert a sparse relationship matrix to long format DataFrame.
    
    Args:
        sparse_matrix: scipy.sparse.csr_matrix - The sparse relationship matrix
        ids: list - List of individual IDs corresponding to matrix rows/cols
        relationship_path: str or None - Relationship path descriptor
        
    Returns:
        pd.DataFrame: Long format with columns ['Person_ID', 'Relative_ID', 'Relationship']
        
    Example:
        >>> sparse_mat, ids, rel_path = load_sparse_relationship_matrix("output/cousins_gen10")
        >>> pairs_df = sparse_matrix_to_long_format(sparse_mat, ids, rel_path)
    """
    pairs = []
    rows, cols = sparse_matrix.nonzero()
    
    # Map back to IDs
    person_ids = [ids[i] for i in rows]
    relative_ids = [ids[j] for j in cols]
    
    df_data = {
        'Person_ID': person_ids,
        'Relative_ID': relative_ids,
        'Relationship': relationship_path if relationship_path else 'Unknown'
    }
    
    return pd.DataFrame(df_data)


def get_matrix_memory_info(results, generations=None):
    """
    Estimate memory requirements for relationship matrices.
    
    Args:
        results: Simulation results dictionary
        generations: Which generations to include (None for all)
        
    Returns:
        dict: Memory estimates and recommendations
    """
    print("Estimating memory requirements...")
    genealogy_df = extract_genealogy_info(results, generations=generations)
    n = len(genealogy_df['ID'].unique())
    
    # Estimate sizes
    dense_size_mb = (n ** 2) / (1024 ** 2)  # 1 byte per element
    
    # Estimate sparse size (rough approximation based on typical family structures)
    # Assume each person has ~10 relatives of each type on average
    avg_relatives_per_type = 10
    # Size of sparse matrix is ~ (nnz * (index_dtype_size + data_dtype_size)) + (n * index_dtype_size)
    # A simpler approximation:
    sparse_size_approx_mb = (n * avg_relatives_per_type * (8 + 8 + 1)) / (1024 ** 2) # (row + col + data)
    
    info = {
        'num_individuals': n,
        'dense_matrix_size_mb': dense_size_mb,
        'sparse_matrix_approx_size_mb': sparse_size_approx_mb,
        'recommend_sparse': n > 50000,
        'memory_warning': dense_size_mb > 4000  # Warn if > 1GB
    }
    
    print(f"Memory estimates for {n:,} individuals:")
    print(f"  - Dense matrix: ~{dense_size_mb:.2f} MB")
    print(f"  - Sparse matrix (estimated): ~{sparse_size_approx_mb:.2f} MB")
    
    if info['recommend_sparse']:
        print(f"  - RECOMMENDATION: Use output_format='sparse' or 'long' for this dataset size")
    
    if info['memory_warning']:
        print(f"  - WARNING: output_format='matrix' would use >{dense_size_mb:.0f}MB of RAM")
    
    return info