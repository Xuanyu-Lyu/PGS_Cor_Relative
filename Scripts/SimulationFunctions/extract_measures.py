# Script to extract individual-level measures from simulated data for computing polygenic score correlations

import pandas as pd
import numpy as np


def extract_individual_measures(results, variable_names, generations=None):
    """
    Extract individual-level measures (e.g., polygenic scores, phenotypes) from simulation results.
    
    Args:
        results (dict): The dictionary returned by AssortativeMatingSimulation.run_simulation().
                       Must contain 'HISTORY' with 'PHEN' data.
        variable_names (list): List of variable names to extract (e.g., ['Y1', 'Y2', 'TPO1', 'TPO2']).
        generations (list, int, or None): Which generation(s) to extract.
                                         - If None: extract all generations
                                         - If int: extract single generation
                                         - If list: extract specified generations
    
    Returns:
        pd.DataFrame: A dataframe with columns ['Generation', 'ID'] plus all requested variables.
                     Each row represents one individual.
    
    Raises:
        ValueError: If results don't contain necessary data or if variables don't exist.
    
    Examples:
        >>> # Extract Y1, Y2, and polygenic scores for all generations
        >>> measures = extract_individual_measures(results, ['Y1', 'Y2', 'TPO1', 'TPO2'])
        >>> 
        >>> # Extract specific variables for generation 10
        >>> measures = extract_individual_measures(results, ['Y1', 'Y2'], generations=10)
        >>> 
        >>> # Extract for multiple generations
        >>> measures = extract_individual_measures(results, ['TPO1', 'TPO2'], generations=[5, 10, 15])
    """
    
    # Validate that results contain history data
    if not results or 'HISTORY' not in results:
        raise ValueError("Results must contain 'HISTORY' data. "
                        "Ensure simulation was run with save_each_gen=True.")
    
    history = results['HISTORY']
    
    if 'PHEN' not in history or not history['PHEN']:
        raise ValueError("HISTORY must contain 'PHEN' data.")
    
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
    
    # Validate variable names
    if not variable_names or not isinstance(variable_names, (list, tuple)):
        raise ValueError("variable_names must be a non-empty list or tuple of strings.")
    
    # List to store dataframes for each generation
    gen_dataframes = []
    
    for gen_idx in generations_to_extract:
        # Get phenotype data for this generation
        phen_df = history['PHEN'][gen_idx]
        
        if phen_df is not None and not phen_df.empty:
            # Check that all requested variables exist
            missing_vars = [var for var in variable_names if var not in phen_df.columns]
            if missing_vars:
                raise ValueError(f"Variables {missing_vars} not found in generation {gen_idx} data. "
                               f"Available variables: {phen_df.columns.tolist()}")
            
            # Extract ID and requested variables
            columns_to_extract = ['ID'] + list(variable_names)
            gen_measures = phen_df[columns_to_extract].copy()
            
            # Add generation identifier
            gen_measures.insert(0, 'Generation', gen_idx)
            
            gen_dataframes.append(gen_measures)
    
    # Concatenate all generation dataframes
    if not gen_dataframes:
        raise ValueError("No valid data found for specified generations.")
    
    combined_measures = pd.concat(gen_dataframes, ignore_index=True)
    
    return combined_measures


def extract_measures_for_pairs(results, pairs_df, variable_names, id_col1='Person_ID', id_col2='Relative_ID'):
    """
    Extract measures for pairs of individuals (e.g., from relationship finding).
    
    This function takes a DataFrame of pairs (e.g., siblings, cousins) and extracts
    the specified variables for both individuals in each pair, allowing for computation
    of correlations between relatives.
    
    Args:
        results (dict): The dictionary returned by AssortativeMatingSimulation.run_simulation().
        pairs_df (pd.DataFrame): DataFrame with at least two columns identifying pairs of individuals.
        variable_names (list): List of variable names to extract for each individual.
        id_col1 (str): Name of the column containing the first individual's ID (default: 'Person_ID').
        id_col2 (str): Name of the column containing the second individual's ID (default: 'Relative_ID').
    
    Returns:
        pd.DataFrame: A dataframe where each row is a pair with columns:
                     - Original columns from pairs_df
                     - For each variable in variable_names:
                       - {var}_1: value for first individual
                       - {var}_2: value for second individual
    
    Examples:
        >>> # Find siblings and extract their polygenic scores
        >>> siblings = find_relationship_pairs(results, "S", output_format='long')
        >>> siblings_scores = extract_measures_for_pairs(results, siblings, ['Y1', 'Y2', 'TPO1', 'TPO2'])
        >>> 
        >>> # Calculate correlation between siblings' Y1 values
        >>> correlation = siblings_scores['Y1_1'].corr(siblings_scores['Y1_2'])
    """
    
    # Validate inputs
    if id_col1 not in pairs_df.columns:
        raise ValueError(f"Column '{id_col1}' not found in pairs_df")
    if id_col2 not in pairs_df.columns:
        raise ValueError(f"Column '{id_col2}' not found in pairs_df")
    
    # Extract all individual measures
    all_measures = extract_individual_measures(results, variable_names, generations=None)
    
    # Create a lookup dictionary: ID -> measures
    # Keep only ID and the requested variables (drop Generation column for lookup)
    measures_lookup = all_measures.set_index('ID')[variable_names].to_dict('index')
    
    # Start with the original pairs dataframe
    result_df = pairs_df.copy()
    
    # Add measures for first individual
    for var in variable_names:
        result_df[f'{var}_1'] = result_df[id_col1].map(
            lambda id_val: measures_lookup.get(id_val, {}).get(var, np.nan)
        )
    
    # Add measures for second individual
    for var in variable_names:
        result_df[f'{var}_2'] = result_df[id_col2].map(
            lambda id_val: measures_lookup.get(id_val, {}).get(var, np.nan)
        )
    
    return result_df


def compute_correlation_by_relationship(pairs_with_measures, variable_name, relationship_col='Relationship'):
    """
    Compute correlations for a variable between pairs, grouped by relationship type.
    
    Args:
        pairs_with_measures (pd.DataFrame): Output from extract_measures_for_pairs().
        variable_name (str): Name of the variable to compute correlation for.
        relationship_col (str): Name of the column containing relationship type (default: 'Relationship').
    
    Returns:
        pd.DataFrame: A dataframe with columns:
                     - Relationship: The relationship type
                     - Variable: The variable name
                     - Correlation: Pearson correlation coefficient
                     - N_Pairs: Number of pairs used in computation
                     - P_Value: P-value for the correlation (if available)
    
    Examples:
        >>> # Compute correlations for Y1 across different relationship types
        >>> correlations = compute_correlation_by_relationship(siblings_scores, 'Y1')
    """
    
    var_1 = f'{variable_name}_1'
    var_2 = f'{variable_name}_2'
    
    if var_1 not in pairs_with_measures.columns or var_2 not in pairs_with_measures.columns:
        raise ValueError(f"Columns '{var_1}' and '{var_2}' not found in dataframe. "
                        f"Make sure to run extract_measures_for_pairs() first.")
    
    results = []
    
    if relationship_col in pairs_with_measures.columns:
        # Group by relationship type
        for rel_type, group in pairs_with_measures.groupby(relationship_col):
            # Remove rows with missing values
            clean_data = group[[var_1, var_2]].dropna()
            
            if len(clean_data) >= 2:  # Need at least 2 pairs for correlation
                correlation = clean_data[var_1].corr(clean_data[var_2])
                
                # Compute p-value using scipy if available
                try:
                    from scipy.stats import pearsonr
                    _, p_value = pearsonr(clean_data[var_1], clean_data[var_2])
                except ImportError:
                    p_value = np.nan
                
                results.append({
                    'Relationship': rel_type,
                    'Variable': variable_name,
                    'Correlation': correlation,
                    'N_Pairs': len(clean_data),
                    'P_Value': p_value
                })
            else:
                results.append({
                    'Relationship': rel_type,
                    'Variable': variable_name,
                    'Correlation': np.nan,
                    'N_Pairs': len(clean_data),
                    'P_Value': np.nan
                })
    else:
        # No relationship column, compute overall correlation
        clean_data = pairs_with_measures[[var_1, var_2]].dropna()
        
        if len(clean_data) >= 2:
            correlation = clean_data[var_1].corr(clean_data[var_2])
            
            try:
                from scipy.stats import pearsonr
                _, p_value = pearsonr(clean_data[var_1], clean_data[var_2])
            except ImportError:
                p_value = np.nan
            
            results.append({
                'Relationship': 'All',
                'Variable': variable_name,
                'Correlation': correlation,
                'N_Pairs': len(clean_data),
                'P_Value': p_value
            })
    
    return pd.DataFrame(results)


def compute_correlations_for_multiple_variables(pairs_with_measures, variable_names, relationship_col='Relationship'):
    """
    Compute correlations for multiple variables between pairs, grouped by relationship type.
    
    Args:
        pairs_with_measures (pd.DataFrame): Output from extract_measures_for_pairs().
        variable_names (list): List of variable names to compute correlations for.
        relationship_col (str): Name of the column containing relationship type (default: 'Relationship').
    
    Returns:
        pd.DataFrame: A dataframe with correlations for all variables and relationship types.
    
    Examples:
        >>> # Compute correlations for multiple variables
        >>> variables = ['Y1', 'Y2', 'TPO1', 'TPO2']
        >>> correlations = compute_correlations_for_multiple_variables(siblings_scores, variables)
    """
    
    all_results = []
    
    for var in variable_names:
        var_results = compute_correlation_by_relationship(pairs_with_measures, var, relationship_col)
        all_results.append(var_results)
    
    combined_results = pd.concat(all_results, ignore_index=True)
    
    return combined_results


def save_measures_to_file(measures_df, output_path, include_index=False):
    """
    Save extracted measures to a file (CSV or TSV format based on extension).
    
    Args:
        measures_df (pd.DataFrame): DataFrame to save.
        output_path (str): Path where to save the file. Extension determines format (.csv or .tsv).
        include_index (bool): Whether to include the index in the output file (default: False).
    
    Examples:
        >>> # Save measures to CSV
        >>> save_measures_to_file(measures, "output/individual_measures.csv")
        >>> 
        >>> # Save to TSV
        >>> save_measures_to_file(pairs_scores, "output/sibling_scores.tsv")
    """
    
    if output_path.endswith('.tsv'):
        measures_df.to_csv(output_path, sep='\t', index=include_index)
        print(f"Measures saved to: {output_path} (TSV format)")
    else:
        measures_df.to_csv(output_path, index=include_index)
        print(f"Measures saved to: {output_path} (CSV format)")
