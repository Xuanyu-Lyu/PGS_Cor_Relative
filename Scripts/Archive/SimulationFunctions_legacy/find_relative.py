# The script is designed to find distant relatives based on the simulated data for multiple generations. 

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, save_npz, load_npz
import os
from multiprocessing import Pool, cpu_count
from functools import partial


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


from collections import defaultdict


def _process_batch_relatives(args):
    """
    Helper function for multiprocessing to find relatives for a batch of individuals.
    This function needs to be at module level to be picklable.
    
    Args:
        args: tuple of (batch_ids, path_string, resolver_data, id_to_idx)
        
    Returns:
        tuple of (rows, cols) for sparse matrix construction
    """
    batch_ids, path_string, resolver_data, id_to_idx = args
    rows = []
    cols = []
    
    # Reconstruct a minimal resolver for this batch
    all_ids, mate_map, parent_map, child_map, parental_pair_map = resolver_data
    
    for start_id in batch_ids:
        # Find relatives using the same logic as PedigreeResolver.find_relatives
        current_people = {start_id}
        
        for step in path_string:
            next_people = set()
            for person in current_people:
                # Handle 0 or invalid IDs
                if not person or person == 0 or person not in all_ids:
                    continue
                
                if step == 'M':  # Mate
                    mate = mate_map.get(person, 0)
                    if mate != 0:
                        next_people.add(mate)
                elif step == 'P':  # Parents
                    ma, pa = parent_map.get(person, (0, 0))
                    if ma != 0: next_people.add(ma)
                    if pa != 0: next_people.add(pa)
                elif step == 'C':  # Children
                    next_people.update(child_map.get(person, set()))
                elif step == 'S':  # Siblings
                    parents = parent_map.get(person, (0, 0))
                    if parents != (0, 0):
                        all_sibs = parental_pair_map.get(parents, set())
                        next_people.update(all_sibs - {person})
            
            current_people = next_people
            if not current_people:
                break
        
        # Add results
        if current_people:
            start_idx = id_to_idx[start_id]
            for rel_id in current_people:
                if rel_id in id_to_idx:
                    rel_idx = id_to_idx[rel_id]
                    rows.append(start_idx)
                    cols.append(rel_idx)
    
    return rows, cols


class PedigreeResolver:
    """
    Resolves complex relationships by traversing a pedigree graph.
    Compatible with simulation output from extract_genealogy_info().
    """
    
    def __init__(self, genealogy_df):
        """
        Pre-processes the genealogy DataFrame into fast-lookup maps.
        
        Args:
            genealogy_df: DataFrame with columns ['ID', 'Mother.ID', 'Father.ID', 'Spouse.ID']
                         NaN/None represents unknown/no one.
        """
        # Convert genealogy_df to the expected format
        pedigree_df = genealogy_df[['ID', 'Mother.ID', 'Father.ID', 'Spouse.ID']].copy()
        pedigree_df.columns = ['ID', 'maID', 'paID', 'mateID']
        
        # Replace NaN with 0 for consistency with the algorithm
        pedigree_df = pedigree_df.fillna(0)
        
        # Convert float IDs to int if needed
        for col in ['ID', 'maID', 'paID', 'mateID']:
            pedigree_df[col] = pedigree_df[col].astype(int)
        
        # Set ID as index
        pedigree_df = pedigree_df.set_index('ID')
        
        self.all_ids = set(pedigree_df.index)
        
        # 1. Mate map: {person -> mate}
        self.mate_map = pedigree_df['mateID'].to_dict()
        # Ensure mates are two-way for this logic
        for p, m in list(self.mate_map.items()):
            if m != 0 and m in self.mate_map and self.mate_map[m] == 0:
                self.mate_map[m] = p
                
        # 2. Parent map: {person -> (ma, pa)}
        self.parent_map = {idx: (row['maID'], row['paID']) 
                           for idx, row in pedigree_df.iterrows()}
        
        # 3. Child map: {parent -> set_of_children}
        self.child_map = defaultdict(set)
        for child, (ma, pa) in self.parent_map.items():
            if ma != 0:
                self.child_map[ma].add(child)
            if pa != 0:
                self.child_map[pa].add(child)

        # 4. Sibling map: pre-build the parent-to-children map
        self.parental_pair_map = defaultdict(set)
        for child, (ma, pa) in self.parent_map.items():
            if ma != 0 or pa != 0:
                self.parental_pair_map[(ma, pa)].add(child)

    def get_relatives_step(self, person_id, step):
        """
        Gets the set of relatives for a *single* step (M, P, S, or C).
        
        Step codes:
            M: Mate/Spouse
            P: Parents (both mother and father)
            C: Children
            S: Siblings (full siblings only)
        """
        relatives = set()
        
        # Handle 0 or invalid IDs
        if not person_id or person_id == 0 or person_id not in self.all_ids:
            return relatives

        if step == 'M': # Mate
            mate = self.mate_map.get(person_id, 0)
            if mate != 0:
                relatives.add(mate)
                
        elif step == 'P': # Parents
            ma, pa = self.parent_map.get(person_id, (0, 0))
            if ma != 0: relatives.add(ma)
            if pa != 0: relatives.add(pa)
            
        elif step == 'C': # Children
            relatives.update(self.child_map.get(person_id, set()))
            
        elif step == 'S': # Siblings (full)
            parents = self.parent_map.get(person_id, (0, 0))
            if parents != (0, 0):
                all_sibs = self.parental_pair_map.get(parents, set())
                # Siblings are children of same parents, but not yourself
                relatives.update(all_sibs - {person_id})
            
        return relatives

    def find_relatives(self, start_person_id, path_string):
        """
        Finds all relatives for one person by following the path string.
        
        Args:
            start_person_id: The ID of the person to start from.
            path_string: A string of steps (e.g., "MS", "PSCP", "PSCM").
            
        Returns:
            A set of IDs of the relatives found at the end of the path.
            
        Examples:
            "P" -> Parents
            "S" -> Siblings
            "PS" -> Parent's Siblings (aunts/uncles)
            "PSC" -> Parent's Sibling's Children (cousins)
            "PSCM" -> Parent's Sibling's Child's Mate (cousin's spouse)
        """
        # Start with a set containing only the starting person
        current_people = {start_person_id}
        
        for step in path_string:
            next_people = set()
            for person in current_people:
                # For each person in the current set, find all relatives
                # for the next step and add them to the next_people set
                next_people.update(self.get_relatives_step(person, step))
            
            # The set of people found becomes the starting point for the next step
            current_people = next_people
            
            # If at any point the set becomes empty, no one can be found
            if not current_people:
                return set()
                
        return current_people

    def find_all_relatives_matrix(self, path_string, sparse=False, show_progress=True, n_jobs=None):
        """
        Generates a 0/1 matrix for all individuals for a given relationship path.
        
        Args:
            path_string: A string of steps (e.g., "PSC" for cousins).
            sparse: If True, returns a scipy sparse CSR matrix. If False, returns dense pandas DataFrame.
                   Use sparse=True for large datasets to save memory.
            show_progress: If True, prints progress updates for large datasets.
            n_jobs: Number of parallel jobs to use. If None, uses all available CPU cores.
                   Set to 1 to disable multiprocessing.
            
        Returns:
            If sparse=False: pd.DataFrame - A square matrix where rows and columns are IDs.
            If sparse=True: tuple of (csr_matrix, id_list) - Sparse matrix and corresponding ID list.
                           matrix[i, j] = 1 if person i has relative j via the path.
        """
        ids = sorted(list(self.all_ids))
        id_to_idx = {id_val: i for i, id_val in enumerate(ids)}
        n = len(ids)
        
        # Determine number of jobs
        if n_jobs is None:
            n_jobs = cpu_count()
        n_jobs = max(1, min(n_jobs, cpu_count()))
        
        # Use multiprocessing for large datasets and when multiple cores requested
        use_multiprocessing = n > 1000 and n_jobs > 1
        
        # Show progress for large datasets
        show_progress = show_progress and n > 5000
        
        if show_progress:
            if use_multiprocessing:
                print(f"  Processing {n:,} individuals using {n_jobs} CPU cores...", flush=True)
            else:
                print(f"  Processing {n:,} individuals...", end='', flush=True)
        
        if sparse:
            # Build sparse matrix using row, col, data format (COO format first)
            rows = []
            cols = []
            
            if use_multiprocessing:
                # Split IDs into batches for parallel processing
                batch_size = max(1, n // (n_jobs * 4))  # Create 4x batches per job for better load balancing
                batches = [ids[i:i + batch_size] for i in range(0, n, batch_size)]
                
                # Prepare resolver data for pickling
                resolver_data = (
                    self.all_ids,
                    self.mate_map,
                    self.parent_map,
                    self.child_map,
                    self.parental_pair_map
                )
                
                # Create argument tuples for each batch
                batch_args = [(batch, path_string, resolver_data, id_to_idx) for batch in batches]
                
                # Process batches in parallel
                with Pool(processes=n_jobs) as pool:
                    results = pool.map(_process_batch_relatives, batch_args)
                
                # Combine results from all batches
                for batch_rows, batch_cols in results:
                    rows.extend(batch_rows)
                    cols.extend(batch_cols)
                
                if show_progress:
                    print(f"  ✓ Completed processing {n:,} individuals")
            else:
                # Single-threaded processing
                progress_interval = max(1, n // 20) if show_progress else n + 1
                
                for idx, start_id in enumerate(ids):
                    # Find all relatives for start_id
                    relatives = self.find_relatives(start_id, path_string)
                    
                    if relatives:
                        start_idx = id_to_idx[start_id]
                        for rel_id in relatives:
                            if rel_id in id_to_idx:
                                rel_idx = id_to_idx[rel_id]
                                rows.append(start_idx)
                                cols.append(rel_idx)
                    
                    # Progress update
                    if show_progress and (idx + 1) % progress_interval == 0:
                        percent = (idx + 1) * 100 // n
                        print(f"\r  Processing {n:,} individuals... {percent}%", end='', flush=True)
                
                if show_progress:
                    print(f"\r  Processing {n:,} individuals... 100% ✓")
            
            # Create sparse matrix (CSR format for efficient operations)
            data = np.ones(len(rows), dtype=np.int8)
            sparse_matrix = csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.int8)
            
            return sparse_matrix, ids
        else:
            # Dense matrix implementation (less common for large datasets)
            matrix = np.zeros((n, n), dtype=int)
            
            if use_multiprocessing:
                # For dense matrices, we still parallelize but need to merge differently
                batch_size = max(1, n // (n_jobs * 4))
                batches = [ids[i:i + batch_size] for i in range(0, n, batch_size)]
                
                resolver_data = (
                    self.all_ids,
                    self.mate_map,
                    self.parent_map,
                    self.child_map,
                    self.parental_pair_map
                )
                
                batch_args = [(batch, path_string, resolver_data, id_to_idx) for batch in batches]
                
                with Pool(processes=n_jobs) as pool:
                    results = pool.map(_process_batch_relatives, batch_args)
                
                # Fill dense matrix from results
                for batch_rows, batch_cols in results:
                    for r, c in zip(batch_rows, batch_cols):
                        matrix[r, c] = 1
                
                if show_progress:
                    print(f"  ✓ Completed processing {n:,} individuals")
            else:
                # Single-threaded processing
                progress_interval = max(1, n // 20) if show_progress else n + 1
                
                for idx, start_id in enumerate(ids):
                    # Find all relatives for start_id
                    relatives = self.find_relatives(start_id, path_string)
                    
                    if relatives:
                        start_idx = id_to_idx[start_id]
                        for rel_id in relatives:
                            if rel_id in id_to_idx:
                                rel_idx = id_to_idx[rel_id]
                                matrix[start_idx, rel_idx] = 1
                    
                    # Progress update
                    if show_progress and (idx + 1) % progress_interval == 0:
                        percent = (idx + 1) * 100 // n
                        print(f"\r  Processing {n:,} individuals... {percent}%", end='', flush=True)
                
                if show_progress:
                    print(f"\r  Processing {n:,} individuals... 100% ✓")
                            
            return pd.DataFrame(matrix, index=ids, columns=ids)


def find_relationship_pairs(results, relationship_path, generations=None, output_format='long', use_sparse=None, n_jobs=None):
    """
    Find all pairs of individuals with a specified relationship across generations.
    
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
                                   
        n_jobs (int or None): Number of parallel jobs to use for processing.
                             If None, uses all available CPU cores.
                             Set to 1 to disable multiprocessing.
                            
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
        >>> # Find all siblings in generations 5, 10, and 15 (auto-detect sparse)
        >>> siblings = find_relationship_pairs(results, "S", generations=[5, 10, 15])
    """
    # Step 1: Extract genealogy information
    genealogy_df = extract_genealogy_info(results, generations=generations)
    
    # Step 2: Create PedigreeResolver
    resolver = PedigreeResolver(genealogy_df)
    
    # Step 3: Determine whether to use sparse matrix
    num_individuals = len(genealogy_df['ID'].unique())
    if use_sparse is None:
        # Auto-detect: use sparse for large datasets (>50,000 individuals)
        use_sparse_internal = num_individuals > 50000
    else:
        use_sparse_internal = use_sparse
    
    # Print info about sparse usage for large datasets
    if use_sparse_internal and output_format.lower() != 'sparse':
        print(f"Using sparse matrix internally for {num_individuals:,} individuals to optimize memory.")
    
    # Step 4: Generate relationship matrix (sparse or dense based on decision)
    if output_format.lower() == 'sparse' or (use_sparse_internal and output_format.lower() != 'matrix'):
        # Generate sparse matrix
        sparse_matrix, ids = resolver.find_all_relatives_matrix(relationship_path, sparse=True, n_jobs=n_jobs)
        
        if output_format.lower() == 'sparse':
            return sparse_matrix, ids
        elif output_format.lower() == 'long':
            # Convert sparse matrix to long format efficiently
            pairs = []
            # Get non-zero elements from sparse matrix
            rows, cols = sparse_matrix.nonzero()
            for i, j in zip(rows, cols):
                pairs.append({
                    'Person_ID': ids[i],
                    'Relative_ID': ids[j],
                    'Relationship': relationship_path
                })
            
            if len(pairs) == 0:
                return pd.DataFrame(columns=['Person_ID', 'Relative_ID', 'Relationship'])
            
            return pd.DataFrame(pairs)
    else:
        # Generate dense matrix (original behavior)
        relationship_matrix = resolver.find_all_relatives_matrix(relationship_path, sparse=False, n_jobs=n_jobs)
        
        if output_format.lower() == 'matrix':
            return relationship_matrix
        
        elif output_format.lower() == 'long':
            # Convert dense matrix to long format
            pairs = []
            for person_id in relationship_matrix.index:
                for relative_id in relationship_matrix.columns:
                    if relationship_matrix.loc[person_id, relative_id] == 1:
                        pairs.append({
                            'Person_ID': person_id,
                            'Relative_ID': relative_id,
                            'Relationship': relationship_path
                        })
            
            if len(pairs) == 0:
                return pd.DataFrame(columns=['Person_ID', 'Relative_ID', 'Relationship'])
            
            return pd.DataFrame(pairs)
    
    raise ValueError(f"output_format must be 'long', 'matrix', or 'sparse', got '{output_format}'")




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
    import os
    sparse_size_mb = os.path.getsize(matrix_file) / (1024 ** 2)
    print(f"  - Dense matrix would be: {dense_size_mb:.2f} MB")
    print(f"  - Sparse matrix is: {sparse_size_mb:.2f} MB")
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
        >>> pairs = sparse_matrix_to_long_format(sparse_mat, ids, rel_path)
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
    
    for i, j in zip(rows, cols):
        pairs.append({
            'Person_ID': ids[i],
            'Relative_ID': ids[j],
            'Relationship': relationship_path if relationship_path else 'Unknown'
        })
    
    if len(pairs) == 0:
        return pd.DataFrame(columns=['Person_ID', 'Relative_ID', 'Relationship'])
    
    return pd.DataFrame(pairs)


def get_matrix_memory_info(results, generations=None):
    """
    Estimate memory requirements for relationship matrices.
    
    Args:
        results: Simulation results dictionary
        generations: Which generations to include (None for all)
        
    Returns:
        dict: Memory estimates and recommendations
    """
    genealogy_df = extract_genealogy_info(results, generations=generations)
    n = len(genealogy_df['ID'].unique())
    
    # Estimate sizes
    dense_size_mb = (n ** 2) / (1024 ** 2)  # 1 byte per element
    
    # Estimate sparse size (rough approximation based on typical family structures)
    # Assume each person has ~10 relatives of each type on average
    avg_relatives_per_type = 10
    sparse_size_approx_mb = (n * avg_relatives_per_type) / (1024 ** 2)
    
    info = {
        'num_individuals': n,
        'dense_matrix_size_mb': dense_size_mb,
        'sparse_matrix_approx_size_mb': sparse_size_approx_mb,
        'recommend_sparse': n > 50000,
        'memory_warning': dense_size_mb > 1000  # Warn if > 1GB
    }
    
    print(f"Memory estimates for {n:,} individuals:")
    print(f"  - Dense matrix: ~{dense_size_mb:.2f} MB")
    print(f"  - Sparse matrix (estimated): ~{sparse_size_approx_mb:.2f} MB")
    
    if info['recommend_sparse']:
        print(f"  - RECOMMENDATION: Use sparse=True for this dataset size")
    
    if info['memory_warning']:
        print(f"  - WARNING: Dense matrix would use >{dense_size_mb:.0f}MB of RAM")
    
    return info

