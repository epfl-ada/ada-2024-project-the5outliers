import pandas as pd
import plotly
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from collections import Counter
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np
import seaborn as sn
import networkx as nx

def create_treemap_data(df):
    """
    Processes the DataFrame Categories to generate labels, parents, values, and ids for the treemap.
    """
    labels = []
    ids = []
    parents = []
    values = []

    # Create a dictionary to keep track of node IDs
    node_ids = {}

    # Get counts at each level
    counts_level_1 = df.groupby('level_1').size().reset_index(name='count')
    counts_level_2 = df.groupby(['level_1', 'level_2']).size().reset_index(name='count')
    counts_level_3 = df.groupby(['level_1', 'level_2', 'level_3']).size().reset_index(name='count')

    # Process level 1 nodes
    for _, row in counts_level_1.iterrows():
        level_1 = row['level_1']
        label = level_1
        id = level_1
        parent_id = ''
        labels.append(label)
        ids.append(id)
        parents.append(parent_id)
        values.append(row['count'])
        node_ids[(level_1,)] = id

    # Process level 2 nodes
    for _, row in counts_level_2.iterrows():
        level_1 = row['level_1']
        level_2 = row['level_2']
        label = level_2
        id = f"{level_1}/{level_2}"
        parent_id = level_1
        labels.append(label)
        ids.append(id)
        parents.append(parent_id)
        values.append(row['count'])
        node_ids[(level_1, level_2)] = id

    # Process level 3 nodes
    for _, row in counts_level_3.iterrows():
        level_1 = row['level_1']
        level_2 = row['level_2']
        level_3 = row['level_3']
        label = level_3
        id = f"{level_1}/{level_2}/{level_3}"
        parent_id = f"{level_1}/{level_2}"
        labels.append(label)
        ids.append(id)
        parents.append(parent_id)
        values.append(row['count'])
        node_ids[(level_1, level_2, level_3)] = id

    return labels, parents, values, ids

def assign_world_region_categories(df_categories, world_region_categories):
    """
    Processes a DataFrame to standardize and categorize subject categories, 
    specifically handling those related to 'World Region'.

    Steps:
    1. Strips the prefix 'subject.' from values in the 'category' column if it exists.
    2. Replaces categories containing any string from `world_region_categories` with 'World Region'.
    3. Updates rows where 'category' is 'World Region':
       - Sets 'level_1' to 'World Region'.
       - Sets 'level_2' and 'level_3' to None.

    Parameters:
    ----------
    df_categories : pandas.DataFrame
        A DataFrame containing a 'category' column and hierarchical columns 
        ('level_1', 'level_2', 'level_3') to represent category levels.

    Returns:
    -------
    pandas.DataFrame
        The updated DataFrame with processed categories and hierarchy levels.
    """
    df_categories_filtered = df_categories.copy()

    df_categories_filtered['category'] = df_categories_filtered['category'].apply(
        lambda category: category.split('subject.', 1)[-1] if 'subject.' in category else category
    )
    df_categories_filtered['category'] = [
        'World Region' if any(region in category for region in world_region_categories) else category
        for category in df_categories_filtered['category']
    ]
    # Updating level_1, level_2, and level_3 based on 'World Region' in 'category'
    df_categories_filtered.loc[df_categories_filtered['category'] == 'World Region', ['level_1', 'level_2', 'level_3']] = ['World Region', None, None]
    return df_categories_filtered

def voyages_categories(df_categories_filtered, voyage_categories):
    """
    Processes a DataFrame to standardize and categorize subject categories, 
    specifically handling those related to 'Voyages'.

    Steps:
    1. Strips the prefix 'subject.' from values in the 'category' column if it exists.
    2. Replaces categories containing any string from `voyage_categories` with 'Voyages'.
    3. Updates rows where 'category' is 'Voyages':
       - Sets 'level_1' to 'Voyages'.
       - Sets 'level_2' and 'level_3' to None.

    Parameters:
    ----------
    df_categories_filtered : pandas.DataFrame
        A DataFrame containing a 'category' column and hierarchical columns 
        ('level_1', 'level_2', 'level_3') to represent category levels.

    Returns:
    -------
    pandas.DataFrame
        The updated DataFrame with processed categories and hierarchy levels.
    """
    df_categories_filtered['category'] = df_categories_filtered['category'].apply(
        lambda category: category.split('subject.', 1)[-1] if 'subject.' in category else category
    )
    df_categories_filtered['category'] = [
        'Voyages' if any(voyage in category for voyage in voyage_categories) else category
        for category in df_categories_filtered['category']
    ]
    # Updating level_1, level_2, and level_3 based on 'Voyages' in 'category'
    df_categories_filtered.loc[df_categories_filtered['category'] == 'Voyages', ['level_1', 'level_2', 'level_3']] = ['Voyages', None, None]
    return df_categories_filtered

def get_main_categories_paths(df_paths, df_categories, omit_loops=False, one_level=True, finished=True):
    """
    Give category paths from article paths and start-end categories.

    Parameters:
        df_paths (pd.DataFrame): DataFrame containing article paths with a 'path' column. 
        df_categories (pd.DataFrame): DataFrame mapping articles to main categories, with 'article' and 'level_1' columns. 
        omit_loops (bool): Optional; if True, removes consecutive repetitions of the same category within a path. 
        one_level (bool): True when want only the first level of categories.
        finished (bool): True if working on a finished path.

    Returns:
        pd.DataFrame: A DataFrame of the most common category paths, with columns:
            - 'Category Path': The sequence of main categories (with optional loop removal).
            - 'source_maincategory': The main category of the start.
            - 'target_maincategory': The main category of the end.
            
    Notes:
        - Paths are created by mapping each article in 'df_paths' to its primary category from 'df_categories'.
        - If an article lacks a category, it remains unchanged in the path.
        - Each path is represented as a string with categories separated by ' -> '.
    """
    # Map articles to main categories
    if one_level:
        article_to_category = dict(zip(df_categories['article'], df_categories['level_1']))
    else: article_to_category = dict(zip(df_categories['article'], df_categories['category']))
    
    category_paths = []
    start_categories = []
    end_categories = []
    
    for index, path in df_paths['path'].items():
        articles = path.split(';')
        categories = [article_to_category.get(article, article) for article in articles]

        # Remove consecutive duplicate categories if omit_loops is True
        if omit_loops:
            categories = [category for i, category in enumerate(categories) 
                          if i == 0 or category != categories[i - 1]]

        category_paths.append(categories)
        
        # Extract start and end categories
        start_categories.append(categories[0] if categories else None)

        if finished:
            end_categories.append(categories[-1] if categories else None)
        else:
            end_categories.append(map_path_to_categories(df_paths[index]['target'], article_to_category))
    
    # Create DataFrame from the collected category paths
    df_common_paths = pd.DataFrame({
        'Category Path': category_paths,
        'source_maincategory': start_categories,
        'target_maincategory': end_categories
    })
    
    return df_common_paths
    
def analyze_categories_paths(df_paths, df_categories, users=True, omit_loops=False):
    """
    Analyze and summarize common category paths from article paths.

    Parameters:
        df_paths (pd.DataFrame): DataFrame containing article paths with a 'path' column. 
        df_categories (pd.DataFrame): DataFrame mapping articles to main categories, with 'article' and 'level_1' columns.
        users (bool): True if the dataset contain users paths and false if it contains the optimal paths
        omit_loops (bool): Optional; if True, removes consecutive repetitions of the same category within a path. 

    Returns:
        pd.DataFrame: A DataFrame of the most common category paths, with columns:
            - 'Category Path': The sequence of main categories (with optional loop removal).
            - 'Count': Number of occurrences of each unique category path.

    Notes:
        - Paths are created by mapping each article in 'df_paths' to its primary category from 'df_categories'.
        - If an article lacks a category, it remains unchanged in the path.
        - Each path is represented as a string with categories separated by ' -> '.
    """
    # Map articles to main categories
    article_to_category = dict(zip(df_categories['article'], df_categories['level_1']))
    
    category_paths = []
    path_counts = {}
    
    for path in df_paths['path']:
        if users:
            articles = path.split(';')
        else: articles=path
        categories = [article_to_category.get(article, article) for article in articles]

        # Remove consecutive duplicate categories if omit_loops is True
        if omit_loops:
            categories = [category for i, category in enumerate(categories) 
                          if i == 0 or category != categories[i - 1]]

        # Create a string representation of the category path
        category_path = ' -> '.join(categories)
        category_paths.append(category_path)
        
        # Count path occurrences
        if category_path in path_counts:
            path_counts[category_path] += 1
        else:
            path_counts[category_path] = 1
    
    # Most common paths
    sorted_paths = sorted(path_counts.items(), key=lambda x: x[1], reverse=True)
    df_common_paths = pd.DataFrame(sorted_paths, columns=['Category Path', 'Count'])
    
    return df_common_paths


def find_all_source_target_pairs(df_finished, df_unfinished, df_links):
    """
    Extracts and deduplicates all source-target pairs from finished and unfinished paths,
    ensuring both nodes exist in the links dataset.

    Args:
        df_finished (pd.DataFrame): DataFrame containing finished paths, where 'path' is a 
                                    semicolon-separated string of nodes.
        df_unfinished (pd.DataFrame): DataFrame containing unfinished paths, where 'path' is a 
                                       semicolon-separated string of nodes.
        df_links (pd.DataFrame): DataFrame containing graph links with 'linkSource' and 
                                 'linkTarget' columns.

    Returns:
        pd.DataFrame: DataFrame with unique source-target pairs where both nodes exist in 
                      the set of nodes defined by df_links.
    """
    optimal_paths = pd.DataFrame()
    optimal_paths['source'] = df_finished['path'].apply(lambda x: x.split(';')[0])
    optimal_paths['target'] = df_finished['path'].apply(lambda x: x.split(';')[-1])
    df_unfinished['source'] = df_unfinished['path'].apply(lambda x: x.split(';')[0])
    optimal_paths = pd.concat([optimal_paths, df_unfinished[['source', 'target']]], ignore_index=True)
    optimal_paths = optimal_paths.drop_duplicates(subset=['source', 'target'])
    
    # Ensure both source and target are in the set of nodes in df_links
    unique_nodes = set(df_links['linkSource']).union(set(df_links['linkTarget']))
    optimal_paths = optimal_paths[optimal_paths['source'].isin(unique_nodes) & optimal_paths['target'].isin(unique_nodes)]
    
    return optimal_paths

def count_start_and_target_per_articles(df_finished, df_unfinished, df_article):
    '''
    adds 2 columns to the df_articles df :
        - start_count : number of time this article was assigned as start 
        - target_count : number of time this article was assigned as target 
    '''
    #extrtct start and target of finished and unfinished paths 
    start_finished = df_finished['path'].str.split(';').str[0]
    target_finished = df_finished['path'].str.split(';').str[-1]
    start_unfinished = df_unfinished['path'].str.split(',').str[0]
    target_unfinished = df_unfinished['target']

    #count start and targets
    start_count = Counter(start_finished)+Counter(start_unfinished)
    target_count = Counter(target_finished)+Counter(target_unfinished)

    df_article['start_count'] = df_article['article'].map(start_count)
    df_article['target_count'] = df_article['article'].map(target_count)
    
    return None


def find_shortest_path(row, G):
    """
    Finds the shortest paths between the source and target nodes in a graph.

    Args:
        row (pd.Series): A row containing 'source' and 'target' nodes.
        G (networkx.DiGraph): A directed graph built using NetworkX.

    Returns:
        list or None: The shortest paths as a list of nodes if it exists; otherwise None.
    """
    source, target = row['source'], row['target']
    try:
        return list(nx.all_shortest_paths(G, source=source, target=target))
    except nx.NetworkXNoPath:
        return None  # If no path exists

def compare_with_matrix(row, df_shortest_path):
    """
    Compares the computed shortest path length with the expected length from a matrix.

    Args:
        row (pd.Series): A row containing 'source', 'target', and 'path' information.
        df_shortest_path (pd.DataFrame): A DataFrame containing the shortest path lengths 
                                         between all source-target pairs.

    Returns:
        tuple: A tuple containing:
               - computed_length (int): The length of the computed shortest path.
               - matrix_length (int): The length of the shortest path from the matrix.
               - matches_matrix (bool): Whether the computed and matrix lengths match.
    """
    source, target = row['source'], row['target']
    matrix_length = df_shortest_path.loc[source, target]
    computed_length = len(row['path']) - 1 if row['path'] is not None else -1
    matches_matrix = computed_length == matrix_length

    return computed_length, matrix_length, matches_matrix

def calculate_optimal_path(df_links, optimal_paths, df_shortest_path):
    """
    Computes the shortest paths for all source-target pairs and compares them to 
    the expected lengths from a matrix.

    Args:
        df_links (pd.DataFrame): DataFrame containing graph links with 'linkSource' and 
                                 'linkTarget' columns.
        optimal_paths (pd.DataFrame): DataFrame with source-target pairs to evaluate.
        df_shortest_path (pd.DataFrame): DataFrame containing the shortest path lengths 
                                         between all source-target pairs.

    Returns:
        pd.DataFrame: DataFrame with computed paths, path lengths, and comparison results. 
                      Rows with mismatched path lengths are included.
    """
    # Build the directed graph from the links
    G = nx.DiGraph()
    G.add_edges_from(df_links[['linkSource', 'linkTarget']].itertuples(index=False, name=None))

    # Compute shortest paths and compare with matrix
    optimal_paths['path'] = optimal_paths.apply(find_shortest_path, G=G, axis=1)
    optimal_paths = optimal_paths.explode('path', ignore_index=True)
    optimal_paths[['computed_length', 'matrix_length', 'matches_matrix']] = \
        optimal_paths.apply(compare_with_matrix, df_shortest_path=df_shortest_path, axis=1, result_type='expand')

    # Check if any values in the matches_matrix column are False
    any_false = not optimal_paths['matches_matrix'].all()
    if any_false:
        print("There are pairs where computed path length does not match the expected path length.")
    else:
        print("All computed path lengths match the expected lengths.")
    
    optimal_paths = optimal_paths.dropna()

    return optimal_paths

def map_path_to_categories(path, article_to_category):
    """
    Maps a path of articles to their respective categories.

    Parameters:
    - path: List of article names in a path, or None.
    - article_to_category: Dictionary mapping articles to categories.

    Returns:
    - List of categories corresponding to the articles in the path, or None if path is None.
    """
    if path is not None:
        return [article_to_category.get(article, 'Unknown') for article in path]
    return None


def clean_path_list(path):
    """
    Cleans a path list by removing occurrences of '<' and the preceding element.

    Parameters:
    - path: List of elements representing a path.

    Returns:
    - A cleaned list with '<' and its preceding elements removed.
    """
    while '<' in path:
        idx = path.index('<')
        del path[idx - 1:idx + 1]  # Remove the element before '<' and '<' itself
    return path


def users_paths(df_finished, df_unfinished, article_to_category):
    """
    Processes paths for finished and unfinished users by extracting sources, targets, and path categories.

    Parameters:
    - df_finished: DataFrame containing finished paths with a 'path_list' column.
    - df_unfinished: DataFrame containing unfinished paths with a 'path' column (semicolon-separated strings).
    - article_to_category: Dictionary mapping articles to categories.

    Returns:
    - Tuple of DataFrames (users_finished, users_unfinished), each containing processed paths and path categories.
    """
    # Process finished paths
    df_finished['source'] = df_finished['path_list'].apply(lambda x: x[0])  # First element in path
    df_finished['target'] = df_finished['path_list'].apply(lambda x: x[-1])  # Last element in path
    users_finished = df_finished[['source', 'target', 'path_list']].copy()
    users_finished['path_list'] = users_finished['path_list'].apply(clean_path_list)
    users_finished['path_categories'] = users_finished['path_list'].apply(
        lambda path: map_path_to_categories(path, article_to_category)
    )

    # Process unfinished paths
    users_unfinished = df_unfinished[['source', 'target', 'path']].copy()
    users_unfinished['path'] = users_unfinished['path'].str.split(';')
    users_unfinished['path'] = users_unfinished['path'].apply(clean_path_list)
    users_unfinished = users_unfinished[users_unfinished['path'].apply(lambda x: len(x) > 1)]
    users_unfinished['path_categories'] = users_unfinished['path'].apply(
        lambda path: map_path_to_categories(path, article_to_category)
    )

    return users_finished, users_unfinished


def filter_pairs(optimal_paths, users_finished, users_unfinished):
    """
    Filters and aligns source-target pairs across optimal, finished, and unfinished datasets.

    Parameters:
    - optimal_paths: DataFrame containing optimal paths with 'source', 'target', and 'computed_length' columns.
    - users_finished: DataFrame containing finished user paths with 'source' and 'target' columns.
    - users_unfinished: DataFrame containing unfinished user paths with 'source' and 'target' columns.

    Returns:
    - Tuple of DataFrames (optimal_fin, users_finished, optimal_unf, users_unfinished) with filtered and aligned pairs.
    """
    # Drop pairs where target is in the source article and filter by computed length
    filtered_pairs = optimal_paths[optimal_paths['computed_length'] > 1][['source', 'target']].apply(tuple, axis=1)
    optimal_paths = optimal_paths[optimal_paths[['source', 'target']].apply(tuple, axis=1).isin(filtered_pairs)]
    users_finished = users_finished[users_finished[['source', 'target']].apply(tuple, axis=1).isin(filtered_pairs)]
    users_unfinished = users_unfinished[users_unfinished[['source', 'target']].apply(tuple, axis=1).isin(filtered_pairs)]

    # Align finished pairs
    common_pairs = set(optimal_paths[['source', 'target']].apply(tuple, axis=1)) & \
                   set(users_finished[['source', 'target']].apply(tuple, axis=1))
    optimal_fin = optimal_paths[optimal_paths[['source', 'target']].apply(tuple, axis=1).isin(common_pairs)]
    users_finished = users_finished[users_finished[['source', 'target']].apply(tuple, axis=1).isin(common_pairs)]

    # Align unfinished pairs
    common_pairs = set(optimal_paths[['source', 'target']].apply(tuple, axis=1)) & \
                   set(users_unfinished[['source', 'target']].apply(tuple, axis=1))
    optimal_unf = optimal_paths[optimal_paths[['source', 'target']].apply(tuple, axis=1).isin(common_pairs)]
    users_unfinished = users_unfinished[users_unfinished[['source', 'target']].apply(tuple, axis=1).isin(common_pairs)]

    return optimal_fin, users_finished, optimal_unf, users_unfinished


def calculate_group_step_percentages(group, unfinished):
    """
    Calculates the percentage of categories at each step within a group.

    Parameters:
    - group: DataFrame group containing a 'path_categories' column.
    - unfinished: Boolean indicating if the paths are unfinished (affects slicing logic).

    Returns:
    - DataFrame with percentage distributions at each step.
    """
    paths = pd.DataFrame(group['path_categories'].tolist())
    if unfinished:
        paths = paths.iloc[:, 1:]  # Exclude the source
    else:
        paths = paths.iloc[:, 1:-1]  # Exclude source and target

    step_percentages = [
        paths[step].value_counts(normalize=True).rename(f'step_{i+1}') * 100
        for i, step in enumerate(paths.columns)
    ]
    return pd.concat(step_percentages, axis=1).fillna(0)


def calculate_step_percentages(optimal_fin, users_finished, optimal_unf, users_unfinished):
    """
    Calculates step-wise percentages for optimal, finished, and unfinished paths.

    Parameters:
    - optimal_fin: DataFrame with optimal finished paths grouped by source and target.
    - users_finished: DataFrame with finished user paths grouped by source and target.
    - optimal_unf: DataFrame with optimal unfinished paths grouped by source and target.
    - users_unfinished: DataFrame with unfinished user paths grouped by source and target.

    Returns:
    - Tuple of DataFrames (S_T_opt_fin_percentages, S_T_fin_percentages, S_T_opt_unf_percentages, S_T_unf_percentages)
      containing step-wise percentage distributions for each dataset.
    """
    S_T_opt_fin_percentages = optimal_fin.groupby(['source', 'target']).apply(
        lambda g: calculate_group_step_percentages(g, unfinished=False)
    )
    S_T_fin_percentages = users_finished.groupby(['source', 'target']).apply(
        lambda g: calculate_group_step_percentages(g, unfinished=False)
    )
    S_T_opt_unf_percentages = optimal_unf.groupby(['source', 'target']).apply(
        lambda g: calculate_group_step_percentages(g, unfinished=False)
    )
    S_T_unf_percentages = users_unfinished.groupby(['source', 'target']).apply(
        lambda g: calculate_group_step_percentages(g, unfinished=True)
    )
    return S_T_opt_fin_percentages, S_T_fin_percentages, S_T_opt_unf_percentages, S_T_unf_percentages


def calculate_average_percentages(dataframes, column_names):
    """
    Calculates the average percentages for the provided dataframes and renames columns.

    Parameters:
    - dataframes: List of tuples, where each tuple contains a DataFrame and its new column suffix.
    - column_names: List of column names to rename in the format ['source', 'target', 'categories', 'suffix'].

    Returns:
    - List of processed DataFrames with renamed columns.
    """
    results = []
    for df, suffix in dataframes:
        avg_df = df.mean(axis=1, skipna=True).reset_index()
        avg_df.columns = column_names[:3] + [f'percentage_{suffix}']
        results.append(avg_df)
    return results


def merge_and_calculate_difference(df1, df2, key_columns, diff_columns):
    """
    Merges two DataFrames and calculates the difference between specified columns.

    Parameters:
    - df1, df2: DataFrames to merge.
    - key_columns: List of column names to merge on.
    - diff_columns: Tuple with column names to calculate the difference (e.g., ('percentage_fin', 'percentage_opt')).

    Returns:
    - A merged DataFrame with an additional column for the calculated difference.
    """
    merged_df = pd.merge(df1, df2, on=key_columns, how='outer').fillna(0)
    merged_df['percentage_diff'] = merged_df[diff_columns[0]] - merged_df[diff_columns[1]]
    return merged_df

def process_category_means(diff_df, article_to_category):
    """
    Processes category means.

    Parameters:
    - diff_df: DataFrame containing source, target, categories, and percentage_diff.
    - article_to_category: Mapping of articles to categories.

    Returns:
    - Processed DataFrame with mean percentage differences by category.
    """
    # Ensure all categories are present
    category_means = diff_df.groupby(['source', 'target', 'categories'])['percentage_diff'].mean().reset_index()
    category_means = category_means.set_index(['source', 'target', 'categories']).unstack(fill_value=0).stack().reset_index()

    # Map source and target to their respective categories
    category_means['source_category'] = category_means['source'].map(article_to_category)
    category_means['target_category'] = category_means['target'].map(article_to_category)
    category_means['source_target'] = category_means['source_category'] + ' -> ' + category_means['target_category']

    # Group by source-target-category combination and calculate mean percentage_diff
    category_means_norm = category_means.groupby(
        ['source_category', 'target_category', 'source_target', 'categories']
    )['percentage_diff'].mean().reset_index()

    # Generate aggregated data for plotting
    category_means = pd.DataFrame(category_means.groupby('categories')['percentage_diff'].mean().reset_index())
    category_means_norm = pd.DataFrame(category_means_norm.groupby('categories')['percentage_diff'].mean().reset_index())
    return category_means, category_means_norm

def filter_most_specific_category(df_categories):
    """
    Filter the DataFrame to retain only the most specific category (lowest count of articles) for each article.
    
    Parameters:
        df_categories (DataFrame): DataFrame with 'article' and 'level_1' columns.
    
    Returns:
        DataFrame: Filtered DataFrame with only the most specific category per article.
    """
    # Step 1: Count the number of articles in each category
    category_counts = df_categories.groupby('level_1')['article'].nunique().reset_index()
    category_counts.columns = ['level_1', 'count']

    # Step 2: Merge category counts back to df_categories
    df_categories_with_counts = df_categories.merge(category_counts, on='level_1')

    # Step 3: Sort by 'count' to ensure the most specific (lowest count) category is first, then drop duplicates
    df_categories_filtered = df_categories_with_counts.sort_values('count').drop_duplicates('article', keep='first')

    # Step 4: Drop the 'count' column if no longer needed
    df_categories_filtered = df_categories_filtered.drop(columns=['count']).reset_index(drop=True)

    return df_categories_filtered

def get_position_frequencies(df, max_position=5):
    """
    Calculate frequencies for each category at each position in the path up to `max_position`.
    
    Parameters:
        df (DataFrame): The DataFrame containing paths.
        max_position (int): The maximum position in the path to analyze.
    
    Returns:
        DataFrame: Frequencies of each category across the specified range of positions.
    """
    position_data = []
    
    for pos in range(max_position):
        # Extract the category at the current position if it exists
        position_column = f"Step {pos+1}"
        df[position_column] = df['Category Path'].apply(
            lambda x: x.split(' -> ')[pos] if len(x.split(' -> ')) > pos else None
        )
        
        # Aggregate counts for the current position
        position_counts = df.dropna(subset=[position_column]).groupby(position_column)['Count'].sum()
            
        position_counts = position_counts.reset_index(name='Frequency')
        position_counts['Position'] = pos + 1  # Record the position number
        position_counts.rename(columns={position_column: 'Category'}, inplace=True)
        
        # Append to overall position data
        position_data.append(position_counts)
    
    # Concatenate all position data into a single DataFrame
    position_data_df = pd.concat(position_data, ignore_index=True)

    return position_data_df

def plot_position_line(df_position, df_article, title="Category transitions frequencies across Path Positions"):
    """
    Plot an interactive line plot of category frequencies across positions with both normalized and non-normalized views.
    
    Parameters:
        df_position (DataFrame): DataFrame with position frequencies for each category.
        df_article (DataFrame): DataFrame containing article categories for dynamic palette generation.
        title (str): Title of the plot.
    """
    # Extract and sort categories
    categories = sorted(df_article["category"].unique())
    palette_category = sn.color_palette("tab20", len(categories))
    
    # Add black for the `<` category
    categories.append("<")  # Add `<` to the category list
    palette_category = [f"rgb({r*255},{g*255},{b*255})" for r, g, b in palette_category]
    color_mapping = dict(zip(categories, palette_category))
    color_mapping["<"] = "rgb(0,0,0)"  # Explicitly assign black to `<`
    
    # Prepare data for normalized frequencies
    df_position_norm = df_position.copy()
    df_position_norm['Normalized Frequency'] = df_position_norm.groupby('Position')['Frequency'].transform(lambda x: (x / x.sum()) * 100)
    
    # Create subplots with separate y-axes
    fig = make_subplots(
        rows=1, cols=2, subplot_titles=("Non-Normalised Frequencies", "Frequencies Normalised by Total Number of Articles per Position"),
        horizontal_spacing=0.05
    )
    # Add non-normalized line plot traces
    unique_categories = sorted(df_position['Category'].unique())
    for category in unique_categories:
        category_data = df_position[df_position['Category'] == category]
        fig.add_trace(
            go.Scatter(
                x=category_data['Position'], 
                y=category_data['Frequency'],
                mode="lines+markers",
                name=category,
                line=dict(color=color_mapping.get(category, "rgb(0,0,0)"))  # Use black as default if not mapped
            ), row=1, col=1
        )
    
    # Add normalized line plot traces
    for category in unique_categories:
        category_data_norm = df_position_norm[df_position_norm['Category'] == category]
        fig.add_trace(
            go.Scatter(
                x=category_data_norm['Position'], 
                y=category_data_norm['Normalized Frequency'],
                mode="lines+markers",
                name=category,
                line=dict(color=color_mapping.get(category, "rgb(0,0,0)")),
                showlegend=False  # Show legend only on the first subplot
            ), row=1, col=2
        )

    # Update layout for better readability with separate y-axes
    fig.update_layout(
        title=title,
        xaxis_title="Position in Path",
        yaxis=dict(title="Frequency"),
        xaxis2_title="Position in Path",
        yaxis2=dict(title="Percentage (%)"),
        legend_title_text="Category",
        template="plotly_white",
        width=1200,
        height=600
    )
    
    # Show the interactive plot
    fig.show()

def plot_normalized_position_bar(df_position, title="Normalized Category Frequencies Across Path Positions"):
    """
    Plot a normalized stacked bar chart of category frequencies across positions.
    
    Parameters:
        df_position (DataFrame): DataFrame with position frequencies for each category.
    """
    # Prepare data for normalized frequencies
    df_position_norm = df_position.copy()
    df_position_norm['Normalized Frequency'] = df_position_norm.groupby('Position')['Frequency'].transform(lambda x: (x / x.sum()) * 100)
    
    # Define a color map for categories
    unique_categories = df_position['Category'].unique()
    colors = px.colors.qualitative.Plotly  # Select a color scheme
    color_map = {category: colors[i % len(colors)] for i, category in enumerate(unique_categories)}
    
    # Create a subplot structure (only one subplot here for normalized bar chart)
    fig = make_subplots(
        rows=1, cols=1, subplot_titles=("Normalized Category Frequencies Across Path Positions",),
        horizontal_spacing=0.15
    )
    
    # Add traces for each category with consistent colors
    for category in unique_categories:
        category_data_norm = df_position_norm[df_position_norm['Category'] == category]
        fig.add_trace(
            go.Bar(
                x=category_data_norm['Position'], 
                y=category_data_norm['Normalized Frequency'],
                name=category,
                marker_color=color_map[category]
            ), row=1, col=1
        )
    
    # Set bar mode to stack
    fig.update_layout(
        barmode="stack",
        title=title,
        xaxis=dict(title="Position in Path", tickmode="linear", dtick=1),  # Ensure integer x-axis ticks
        yaxis=dict(title="Percentage (%)"),
        legend_title_text="Category",
        template="plotly_white",
        width=900,
        height=600
    )
    
    # Show the interactive plot
    fig.show()
    
# def check_voyage_status(paths, finished, n, voyage_categories = ['Geography', 'Countries']):
#     """
#     Check if the category path is voyage or not voyage, that is whether the first n categories after the first are 'Geography' or 'Countries'. 

#     Parameters:
#         paths (str): A category path in the form 'Geography -> Countries -> Geography'.
#         finished (bool): Whether the path is finished or not finished.
#         n (int): Number of different categories following the first to consider.

#     Returns:
#         bool: True if the path is a 'voyage', False otherwise.
#     """    
#     # Ensure that paths is a string
#     if not isinstance(paths, str):
#         return False  # Return False for invalid paths

#     # Split the category path
#     categories = paths.split(' -> ')
#     path_len = len(categories)
    
#     # Exclude paths that start with categories on voyage_categories
#     if categories[0] in voyage_categories or categories[-1] in voyage_categories:
#         return False
    
#     if finished: 
#         # Case 1: Path with 1 category -> always False
#         if path_len <= 2:
#             return False
#         # Case 2: Path length between 3 and n+2 -> check middle categories
#         elif 2 < path_len <= n + 2:
#             return any(category in categories[1:-1] for category in voyage_categories)
#         # Case 3: Path longer than n+2 -> check the first n categories after the first
#         else:
#             return any(category in categories[1:n+1] for category in voyage_categories)
        
#     else: 
#         # Case 1: Path with 1 or 2 categories -> always False
#         if path_len <= 1:
#             return False
#         # Case 2: Path length between 3 and n+2 -> check middle categories
#         elif 1 < path_len <= n + 1:
#             return any(category in categories[1:] for category in voyage_categories)
#         # Case 3: Path longer than n+2 -> check the first n categories after the first
#         else:
#             return any(category in categories[1:n+1] for category in voyage_categories)
    

        
def check_voyage_status(row):
    """
    Check if the path is a Wikispeedia_Voyage, that is whether categories (not source and target) are 'Voyages'. 

    Parameters:
        row: A row of a dataframe containing information about paths and categories.

    Returns:
        bool: True if the path is a Wikispeedia_Voyage, False otherwise.
    """    
    
    if row['target_maincategory']=='World Region' or row['source_maincategory']=='World Region':
        return False
    else: return any('World Region' in category for category in row['Category Path'])

def game_voyage_sorting(df_article_paths, df_categories):
    """
    Adds a boolean 'Wikispeedia_Voyage' column to each game.
    First maps articles to categories (requirement for using check_voyage_status()), then checks if the category path qualifies as a Wikispeedia_Voyage.

    Parameters:
        df_article_path (DataFrame): DataFrame with 'path' column containing article paths
        df_categories (DataFrame): DataFrame with 'article' and categories columns for mapping

    Returns:
        DataFrame: Original DataFrame with a additional columns.
    """
    # Map articles to their main categories
    category_path_df = get_main_categories_paths(df_article_paths, df_categories, omit_loops=True, one_level=True)
    
    # Apply the transformation and check voyage status
    df_article_paths['Category Path'] = category_path_df['Category Path']
    df_article_paths['source_maincategory'] = category_path_df['source_maincategory']
    df_article_paths['target_maincategory'] = category_path_df['target_maincategory']
    df_article_paths['Wikispeedia_Voyage'] = df_article_paths.apply(lambda row: check_voyage_status(row), axis=1)
    
    return df_article_paths


def plot_sankey_voyage(df_all_voyage):
    """
    Plots a Sankey diagram to visualize the distribution of paths classified as 'Voyage' or 'Non-Voyage'.

    Parameters:
        df_all_voyage (DataFrame): DataFrame containing boolean 'Wiki_Voyage' and 'source_maincategory', 'target_maincategory'  columns

    Returns:
        None: it displays the Sankey diagram 
    """

    #voyage_categories = ['Geography of Great Britain', 'Geography of Asia', 'Geography of Oceania Australasia', 'North American Geography', 'European Geography', 'African Geography', 'Central and South American Geography', 'Antarctica', 'Geography of the Middle East','Countries']
    
    # Mapping for start, voyage, and end nodes
    df_all_voyage['source_category_label'] = df_all_voyage['source_maincategory'].apply(lambda x: 'Source is a World Region' if x=='Voyages' else 'Source is not a World Region')
    df_all_voyage['target_category_label'] = df_all_voyage['target_maincategory'].apply(lambda x: 'Target is a World Region' if x=='Voyages' else 'Target is not a World Region')
    df_all_voyage['voyage_label'] = df_all_voyage['Wikispeedia_Voyage'].apply(lambda x: 'Voyages' if x else 'Non Voyages')

    # Start→Voyage flows
    start_voyage_flows = df_all_voyage.groupby(['source_category_label', 'voyage_label']).size().reset_index(name='count')

    # Voyage→End flows
    voyage_end_flows = df_all_voyage.groupby(['voyage_label', 'target_category_label']).size().reset_index(name='count')

    # Define node labels
    labels = ['Source is a World Region', 'Source is not a World Region',
              'Voyages', 'Non Voyages',
              'Target is a World Region', 'Target is not a World Region']

    # Create mappings for source and target node indices
    label_map = {label: i for i, label in enumerate(labels)}

    # Initialize lists for diagram data
    sources = []
    targets = []
    values = []

    # Add Start→Voyage flows
    for _, row in start_voyage_flows.iterrows():
        sources.append(label_map[row['source_category_label']])
        targets.append(label_map[row['voyage_label']])
        values.append(row['count'])

    # Add Voyage→End flows
    for _, row in voyage_end_flows.iterrows():
        sources.append(label_map[row['voyage_label']])
        targets.append(label_map[row['target_category_label']])
        values.append(row['count'])

    # Define node colors
    node_colors = [
        '#2CB5AE',  # First in Countries/Geography
        '#4b4b4b',  # First not in Countries/Geography
        '#2CB5AE',  # Voyage
        '#4b4b4b',  # Non-Voyage
        '#2CB5AE',  # Target in Countries/Geography
        '#4b4b4b'  # Target not in Countries/Geography
    ]

    link_colors=['rgba(75, 75, 75, 0.3)',  # not voyage
        'rgba(75, 75, 75, 0.3)',  # not voyage
        'rgba(44, 181, 174, 0.3)',  # Voyage
        'rgba(75, 75, 75, 0.3)',  # not voyage
        'rgba(75, 75, 75, 0.3)',  # not voyage
        'rgba(44, 181, 174, 0.3)'  # Voyage
    ]

    # Create the Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        node=dict(pad=15, thickness=20, line=dict(color="black", width=0), label=labels, color=node_colors),
        link=dict(source=sources, target=targets, value=values, hovercolor=link_colors)
    )])
    
    fig.update_layout(
        title_text="Voyage and Non-Voyage Paths",
        font_size=10,
        title_font_size=14,
        title_x=0.5,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    fig.show()
    return plotly.offline.plot(fig, include_plotlyjs=False, output_type='div')

def backtrack(paths) :
    """
    Compute the number of backtracks in each path.
    """

    paths["path_list"] = paths["path"].apply(lambda x: x.split(";"))
    paths["back_nb"]=paths["path_list"].apply(lambda x: x.count("<"))
    paths["size"]=paths["path_list"].apply(lambda x: len(x))
    paths["have_back"] = paths["back_nb"] > 0

    return paths

def find_category_path(path_list, categories) :
    """
    Find the list of category for a given list of articles.
    """
    categories = [categories[categories["article"]==article]["level_1"].values[0] for article in path_list if article in categories["article"].values]
    return categories

def extract_category_path(paths, categories) :
    """
    Extract the category path for each path.
    """
    paths["path_list"] = paths["path"].apply(lambda x: x.split(";"))
    paths["category"] = paths["path_list"].apply(lambda x: find_category_path(x, categories))
    paths["category"] = paths["category"].apply(lambda x : list(set(list(x))))
    return paths

def find_categories_start_end(paths, categories) :
    
    """
    Find the start and end category of each path.
    """

    paths["start"] = paths["path"].apply(lambda x: x.split(";")[0])
    if "target" in paths.columns:
        paths["end"] = paths["target"]
    else :
        paths["end"] = paths["path"].apply(lambda x: x.split(";")[-1])
    paths["start_category"] = paths["start"].apply(lambda x: categories[categories["article"] == x]["category"].values[0].split(".")[1] if x in categories["article"].values else None)
    paths["end_category"] = paths["end"].apply(lambda x: categories[categories["article"] == x]["category"].values[0].split(".")[1] if x in categories["article"].values else None)

    return paths

def plot_cooccurrence_cat_matrix(df_categories, abbreviations=None):
    """
    Plots a co-occurrence matrix using abbreviations for category labels in the heatmap.
    
    Parameters:
    - df_categories (DataFrame): The DataFrame with 'article' and 'level_1' columns.
    - abbreviations (dict, optional): A dictionary mapping full category names to abbreviations.
    """
    # Get the unique categories from the DataFrame
    categories_full = df_categories['level_1'].unique()
    
    # If abbreviations are provided, map the full category names to abbreviations
    if abbreviations:
        categories_abbr = [abbreviations.get(cat, cat) for cat in categories_full]
    else:
        categories_abbr = categories_full

    # Group by article and collect unique level_1 categories for each article
    article_combinations = (
        df_categories.groupby("article")["level_1"]
        .apply(lambda x: tuple(sorted(x.unique())))  # Sort and get unique level_1 values as a tuple
    )
    combination_counts = article_combinations.value_counts()

    # Create a co-occurrence matrix using abbreviations for the plot
    matrix = pd.DataFrame(0, index=categories_abbr, columns=categories_abbr)

    # Fill the matrix with co-occurrence counts
    for comb, count in zip(combination_counts.index, combination_counts):
        for i in comb:
            for j in comb:
                matrix.loc[abbreviations.get(i, i), abbreviations.get(j, j)] += count

    # Calculate the total articles for each category using the diagonal
    total_articles = pd.Series(np.diag(matrix), index=categories_abbr)

    # Mask for the upper triangle excluding the diagonal
    mask = np.triu(np.ones_like(matrix, dtype=bool), k=1) | (matrix == 0)

    # Plot the co-occurrence matrix

    # Set up the color map for masked cells
    cmap = sn.color_palette("YlGnBu", as_cmap=True)
    cmap.set_bad(color='white')  # Set masked cells to appear white

    plt.figure(figsize=(10, 8))
    sn.heatmap(
        matrix,
        annot=True,
        fmt="g",
        cmap=cmap,
        cbar_kws={'label': 'Number of Co-occurrences'},
        mask=mask,
        linewidths=0.5
    )

    # Annotate each off-diagonal cell in the upper triangle to suggest a main category
    for i, cat1 in enumerate(categories_abbr):
        for j, cat2 in enumerate(categories_abbr):
            if i < j and matrix.loc[cat1, cat2] > 0:  # Only upper triangle and non-zero cells
                # Determine which category has fewer total articles
                if total_articles[cat1] < total_articles[cat2]:
                    main_category = cat1
                else:
                    main_category = cat2
                
                # Annotate the cell with the suggested main category
                plt.text(
                    j + 0.5, i + 0.5,
                    f"{main_category}",
                    ha='center', va='center', color="red", fontsize=8, fontweight='bold'
                )

    # Customize labels and layout
    plt.title("Co-occurrence of Level 1 Categories in Articles with Main Category Suggestion")
    plt.xlabel("Level 1 Category")
    plt.ylabel("Level 1 Category")
    plt.xticks(rotation=0, fontsize=10)
    plt.yticks(fontsize=10, rotation=0)

    # Add an enhanced legend for abbreviations
    if abbreviations:
        legend_labels = [f"{abbr} = {name}" for name, abbr in abbreviations.items()]
        plt.figtext(0.98, 0.5, "\n".join(legend_labels), ha="left", fontsize=10, bbox=dict(
            facecolor="lightgrey", edgecolor="black", boxstyle="round,pad=0.5", linewidth=1))

    plt.tight_layout()
    plt.show()

def matrix_common_paths(data):  
    # Extract transitions and create transition counts
    transitions = {}
    for _, row in data.iterrows():
        path = row['Category Path'].split(" -> ")
        count = row['Count']
        for i in range(len(path) - 1):
            from_cat = path[i]
            to_cat = path[i + 1]
            if from_cat not in transitions:
                transitions[from_cat] = {}
            if to_cat not in transitions[from_cat]:
                transitions[from_cat][to_cat] = 0
            transitions[from_cat][to_cat] += count

    # Create transition matrix DataFrame
    categories = sorted(set([key for key in transitions] + [k for subdict in transitions.values() for k in subdict]))
    transition_matrix = pd.DataFrame(0, index=categories, columns=categories)

    # Populate the transition matrix with counts
    for from_cat, to_cats in transitions.items():
        for to_cat, count in to_cats.items():
            transition_matrix.at[from_cat, to_cat] = count
    
    return transition_matrix

def transition_cat_matrix(df):
    # Get max value for setting the color scale
    max_value = df.values.max()

    # Define thresholds and corresponding colors
    thresholds = [0, 100, 500, 1000, 5000, 10000, max_value]  # Adjust based on data range
    colors = ["#f0f0f0", "#a6bddb", "#3690c0", "#034e7b", "#feb24c", "#f03b20"]
    cmap = ListedColormap(colors)
    cmap.set_bad(color='white')  # Set masked cells to appear white
    norm = BoundaryNorm(thresholds, len(colors))


    plt.figure(figsize=(9, 9))

    # Mask zeros for better visibility in log scale
    mask = df == 0
    sn.heatmap(df, annot=True, annot_kws={"size": 9}, fmt="d", 
                mask=mask, cmap=cmap, norm=norm, cbar_kws={'label': 'Count', 'shrink': 0.6} , square=True)
    plt.title('Transition within categories')
    plt.xlabel('To Category')
    plt.ylabel('From Category')
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=8) 

    plt.tight_layout()
    plt.show()

def plot_articles_pie_chart(df, palette, abbreviations=None):
    """
    Plots a simplified pie chart of the total number of articles per Level 1 category.

    Parameters:
    - df (DataFrame): The DataFrame containing 'article' and 'level_1' columns.
    - abbreviations (dict, optional): A dictionary mapping full category names to abbreviations.
    """
    # Group by Level 1 category and count the number of articles
    category_counts = df['level_1'].value_counts()

    # Sort the categories by the article count in ascending order
    category_counts = category_counts.sort_values(ascending=True)

    # Handle small categories (less than 3%) by grouping them as 'Others'
    threshold = 3  # percentage threshold
    small_categories = category_counts[category_counts / category_counts.sum() * 100 < threshold]
    small_categories_total = small_categories.sum()
    large_categories = category_counts[category_counts / category_counts.sum() * 100 >= threshold]

    # Add "Others" for small categories
    if not small_categories.empty:
        others = pd.Series({f'Others': small_categories_total})
        large_categories = pd.concat([large_categories, others])

    # Prepare the labels: Use abbreviations if provided
    if abbreviations:
        labels = [abbreviations.get(cat, cat) for cat in large_categories.index]
        legend_labels = [f"{cat} ({abbreviations.get(cat, 'N/A')})" for cat in large_categories.index]
    else:
        labels = large_categories.index
        legend_labels = labels

    # Plot the pie chart
    fig, ax = plt.subplots(figsize=(7, 7))
    wedges, texts, autotexts = ax.pie(
        large_categories, 
        labels=labels, 
        autopct='%1.1f%%', 
        startangle=90,
        pctdistance=0.8,
        colors=[palette[label] for label in large_categories.index],
    )

    # Customize the font and color of the numbers
    for autotext in autotexts:
        autotext.set_fontsize(9)  # Change font size

    # Set the title of the plot
    ax.set_title('Articles Distribution per Level 1 Category')

    # Place the legend outside the pie chart to avoid overlap
    ax.legend(
        legend_labels, 
        title="Categories", 
        loc='center left', 
        bbox_to_anchor=(1, 0.5), 
        fontsize=10
    )

    # Display the pie chart
    plt.tight_layout()  # Adjust layout to ensure everything fits
    plt.show()
    
def plot_proportion_links_in_cat_pie_chart(df, in_or_out , palette, abbreviations=None):
    """
    use plot_proportions_of_in_and_out_degree_in_categories() for both on same plot 
    Plots pie chart of the total number (sum) of all links for category.
    ex for out degree: 25% of all links are in country articles 
    ex for in degree: 25% of all links target country articles 
    Parameters:
    - df (DataFrame): The DataFrame containing 'article' and 'category', 'in_degree', and 'out_degree' columns.
    - in_else_out : in degree if true, out degree if false
    - pallette (dict): palette to use for categories containing others country and geo
    - abbreviations (dict, optional): A dictionary mapping full category names to abbreviations.

    """
    if in_or_out:
        in_degree_tot = df.groupby('category')['in_degree'].sum().sort_values(ascending=False)
    else :
        in_degree_tot = df.groupby('category')['out_degree'].sum().sort_values(ascending=False)
    
    labels_cat = in_degree_tot.keys()

    # Handle small categories (less than 3%) by grouping them as 'Others'
    threshold = 3  # percentage threshold
    small_categories = in_degree_tot[in_degree_tot / in_degree_tot.sum() * 100 < threshold]
    small_categories_total = small_categories.sum()
    large_categories = in_degree_tot[in_degree_tot / in_degree_tot.sum() * 100 >= threshold]

    # Add "Others" for small categories
    if not small_categories.empty:
        others = pd.Series({f'Others': small_categories_total})
        large_categories = pd.concat([large_categories, others])

    # Prepare the labels: Use abbreviations if provided
    if abbreviations:
        labels = [abbreviations.get(cat, cat) for cat in large_categories.index]
        legend_labels = [f"{cat} ({abbreviations.get(cat, 'N/A')})" for cat in large_categories.index]
    else:
        labels = large_categories.index
        legend_labels = labels 

    # Plot the pie chart
    fig, ax = plt.subplots(figsize=(7, 7))
    wedges, texts, autotexts = ax.pie(
        large_categories, 
        labels=labels, 
        autopct='%1.1f%%', 
        startangle=90,
        pctdistance=0.8,
        colors=[palette[label] for label in labels_cat]
    )

    # Customize the font and color of the numbers
    for autotext in autotexts:
        autotext.set_fontsize(9)  # Change font size

    # Set the title of the plot
    if in_or_out:
        ax.set_title('Category-wise share of all out-degree links')
    else :
        ax.set_title('Proportions of links targetting each categories')
        
    # Place the legend outside the pie chart to avoid overlap
    ax.legend(
        legend_labels, 
        title="Categories", 
        loc='center left', 
        bbox_to_anchor=(1, 0.5), 
        fontsize=10
    )

    # Display the pie chart
    plt.tight_layout()  # Adjust layout to ensure everything fits
    plt.show()
    
def plot_proportions_of_in_and_out_degree_in_categories(df, palette, abbreviations=None, threshold=3):
    """
    Plots pie charts for proportions of in-degree and out-degree links across categories.

    Parameters:
    - df (DataFrame): The DataFrame containing 'category', 'in_degree', and 'out_degree' columns.
    - palette (dict): Palette to use for categories, including 'Others'.
    - abbreviations (dict, optional): A dictionary mapping full category names to abbreviations.
    - threshold (int, optional): Minimum percentage to display a category individually; others are grouped under 'Others'.
    """
    # Sum in-degree and out-degree by category
    in_degree_tot = df.groupby('category')['in_degree'].sum()
    out_degree_tot = df.groupby('category')['out_degree'].sum()

    # Handle small categories by grouping them as "Others"
    def handle_small_categories(category_dict):
        small_categories = category_dict[category_dict / category_dict.sum() * 100 < threshold]
        small_categories_total = small_categories.sum()
        large_categories = category_dict[category_dict / category_dict.sum() * 100 >= threshold]
        if not small_categories.empty:
            others = pd.Series({'Others': small_categories_total})
            large_categories = pd.concat([large_categories, others])
        return large_categories

    in_degree_tot = handle_small_categories(in_degree_tot)
    out_degree_tot = handle_small_categories(out_degree_tot)

    # Prepare labels
    def prepare_labels(category_dict):
        if abbreviations:
            labels = [abbreviations.get(cat, cat) for cat in category_dict.index]
        else:
            labels = category_dict.index
        return labels

    in_labels = prepare_labels(in_degree_tot)
    out_labels = prepare_labels(out_degree_tot)

    # Create subplots
    fig, ax = plt.subplots(1, 2, figsize=(15, 7))

    # Plot in-degree pie chart
    ax[0].pie(
        in_degree_tot.values,
        labels=in_labels,
        autopct='%1.1f%%',
        startangle=90,
        pctdistance=0.8,
        colors=[palette.get(cat, "#cccccc") for cat in in_degree_tot.index],
    )
    ax[0].set_title("Proportion of Links Targeting Each Category", fontsize=14)

    # Plot out-degree pie chart
    ax[1].pie(
        out_degree_tot.values,
        labels=out_labels,
        autopct='%1.1f%%',
        startangle=90,
        pctdistance=0.8,
        colors=[palette.get(cat, "#cccccc") for cat in out_degree_tot.index],
    )
    ax[1].set_title("Proportion of Links Leaving Each Category", fontsize=14)

    # Add a single legend
    add_legend_category(
        fig=fig,
        palette_category=palette,
        categories=palette.keys(),
        bbox_to_anchor=(1, 0.75)
    )

    # Adjust layout
    plt.suptitle("Proportions of in and out degree links by Category", fontsize=16, y=1.05)
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust for the legend
    plt.show()


  
    
def add_legend_category(fig, palette_category, categories, bbox_to_anchor=(1.15, 0.85)):

    handles = [
        plt.Line2D([0], [0], marker='o', color=color, linestyle='', markersize=10) 
        for color in palette_category.values()  # Use values from the dictionary
    ]
    labels = list(palette_category.keys())
    fig.legend(
        handles, 
        labels, 
        bbox_to_anchor=bbox_to_anchor, 
        title="Categories", 
    )
    
def plot_proportion_category_start_stop_pies(df_article, palette, abbreviations=None, threshold=2.3):
    """
    Makes pie charts showing the proportion of categories in start/target articles.
    
    Parameters:
    - df_article (DataFrame): DataFrame containing 'start_count' and 'target_count'
    - palette (dict): Color palette for the categories
    - abbreviations (dict, optional): Dictionary mapping full category names to abbreviations 
    - threshold (int, optional): Minimum percentage to display a category individually: Others are grouped under 'Others'
    """
    # count number of articles given as start and target for each category
    start_dict = df_article.groupby('category')['start_count'].sum()
    target_dict = df_article.groupby('category')['target_count'].sum()

    # Handle small categories by grouping them as "Others"
    def handle_small_categories(category_dict):
        small_categories = category_dict[category_dict / category_dict.sum() * 100 < threshold]
        small_categories_total = small_categories.sum()
        large_categories = category_dict[category_dict / category_dict.sum() * 100 >= threshold]
        if not small_categories.empty:
            others = pd.Series({'Others': small_categories_total})
            large_categories = pd.concat([large_categories, others])
        return large_categories

    start_dict = handle_small_categories(start_dict)
    target_dict = handle_small_categories(target_dict)

    # Use abbreviations if provided
    def prepare_labels(category_dict):
        if abbreviations:
            labels = [abbreviations.get(cat, cat) for cat in category_dict.index]
            legend_labels = [f"{cat} ({abbreviations.get(cat, 'N/A')})" for cat in category_dict.index]
        else:
            labels = category_dict.index
            legend_labels = labels
        return labels, legend_labels

    start_labels, start_legend_labels = prepare_labels(start_dict)
    target_labels, target_legend_labels = prepare_labels(target_dict)


    # subplots
    fig, ax = plt.subplots(1, 2, figsize=(15, 7))

    # Plot start atricles pie in 0 slot
    ax[0].pie(
        start_dict.values,
        labels=start_labels,
        colors=[palette.get(cat, "#cccccc") for cat in start_dict.index],
        autopct='%1.1f%%',
        startangle=90,
        pctdistance=0.8
    )
    ax[0].set_title("Proportion of categories in start articles", fontsize=14)

    # Plot target articles pie in 1 slot
    ax[1].pie(
        target_dict.values,
        labels=target_labels,
        colors=[palette.get(cat, "#cccccc") for cat in target_dict.index],
        autopct='%1.1f%%',
        startangle=90,
        pctdistance=0.8
    )
    ax[1].set_title("Proportion of categories in target articles", fontsize=14)

    add_legend_category(
        fig=fig,
        palette_category=palette,
        categories=palette.keys(),
        bbox_to_anchor=(1.15, 0.85)
    )
    plt.suptitle("Proportions of Categories in Start and Target Articles", fontsize=16, y=1.05)
    plt.tight_layout()

    plt.show()

def plot_metrics_by_category(df_article, metrics, palette_category_dict, category_abbreviations):
    """
    Plots bar charts for multiple metrics by category using Plotly.

    Parameters:
    - df_article (DataFrame): DataFrame containing article data.
    - metrics (list): List of metric column names to plot.
    - palette_category_dict (dict): Color palette for the categories.
    - category_abbreviations (dict): Abbreviations for categories.
    """
    # Loop through metrics and plot
    fig, ax = plt.subplots(2, 3, figsize=(15, 5))
    
    for i, metric in enumerate(metrics):
        row, col = divmod(i, 3)
        order = df_article.groupby("category")[metric].mean().sort_values(ascending=False).reset_index()["category"]
        sn.barplot(
            x="category", 
            y=metric, 
            hue="category", 
            palette=palette_category_dict, 
            data=df_article, 
            ax=ax[row, col], 
            order=order
        )
        ax[row, col].set_title(f'{metric.replace("_", " ").capitalize()} by Category')
        ax[row, col].set_xticklabels([])
        if row == 0 :
            ax[row, col].set_xlabel('')

    add_legend_category(fig,palette_category_dict, category_abbreviations)

    plt.suptitle("Articles Complexity by Categories", y=1, fontsize=16)
    plt.tight_layout()
    plt.show()

def plot_article_popularity_link_density(df_article, df_finished_voyage, palette_category_dict, category_abbreviations, df_categories_filtered):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    #Plot the most visited articles in finished paths
    all_articles = []
    df_finished_voyage['path'].apply(lambda x: all_articles.extend(x.split(';')))
    df_path_articles = pd.Series(all_articles).value_counts().rename_axis('article_name').reset_index(name='value_counts')
    df_path_articles["category"]=df_path_articles["article_name"].apply(lambda x: df_categories_filtered[df_categories_filtered["article"]==x]["level_1"].values[0] if len(df_categories_filtered[df_categories_filtered["article"]==x]["category"].values)>0 else "None")
    df_path_articles = df_path_articles[df_path_articles['article_name'] != '<']

    sn.barplot(x='value_counts', y='article_name', hue="category", palette=palette_category_dict, data=df_path_articles.head(15), ax=ax[0])
    ax[0].set_title('Most visited articles in paths')
    ax[0].legend_.remove() 

    for i, metric in enumerate(["in_degree", "out_degree"]):
        sn.barplot(x=metric, y='article', hue="category", palette=palette_category_dict, data=df_article.sort_values(metric, ascending=False).head(15), ax=ax[i+1])
        ax[i+1].set_title(f'Articles with the most links ({metric.replace("_", " ").capitalize()}) (without duplicates)')
        ax[i+1].legend_.remove()
        ax[i+1].set_ylabel('')

    add_legend_category(fig,palette_category_dict, category_abbreviations)
    plt.suptitle("Correlation between article popularity and link density", y=1, fontsize=16)
    plt.tight_layout()
    plt.show()


def remove_outliers(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    filtered_df = df[(df[col] >= (Q1 - 1.5 * IQR)) & (df[col] <= (Q3 + 1.5 * IQR))]
    return filtered_df


def plot_difficulties_voyage (df_finished_voyage, df_unfinished_voyage, palette_category_dict):
    color_voyage = palette_category_dict['World Region']
    
    df_finished_voyage["finished"] = True
    df_finished_voyage["cte"] = 1
    df_unfinished_voyage["finished"] = False
    df_unfinished_voyage["cte"] = 1
    df_voyage = pd.concat([df_finished_voyage, df_unfinished_voyage])

    fig = make_subplots(
        rows=2, cols=2, 
        subplot_titles=(
            "Duration Distribution", 
            "Completion Ratios", 
            "Rating Distribution for Voyage Game", 
            "Rating Distribution for Non-Voyage Game"
        )
    )

    # ==== PLOT 1 (Violin Plot: Duration Distribution) ====
    df_voyage_duration = df_finished_voyage[df_finished_voyage["Wikispeedia_Voyage"] == True]
    df_voyage_duration = remove_outliers(df_voyage_duration, "durationInSec")

    df_non_voyage_duration = df_finished_voyage[df_finished_voyage["Wikispeedia_Voyage"] == False]
    df_non_voyage_duration = remove_outliers(df_non_voyage_duration, "durationInSec")

    fig.add_trace(
        go.Violin(
            x=df_voyage_duration["cte"], 
            y=df_voyage_duration["durationInSec"],
            legendgroup="Yes", 
            scalegroup="Yes", 
            name="Voyage",
            side="negative", 
            line_color=color_voyage, 
            box_visible=True,
            meanline_visible=True,
            showlegend=False),
        row=1, col=1
    )
    fig.add_trace(
        go.Violin(
            x=df_non_voyage_duration["cte"],
            y=df_non_voyage_duration["durationInSec"],
            legendgroup="No", 
            scalegroup="No", 
            name="Non-Voyage",
            side="positive", 
            line_color="gray",
            box_visible=True,
            meanline_visible=True,
            showlegend=False
        ),
        row=1, col=1
    )

    # Axis labels
    fig.update_yaxes(title_text="Duration (seconds)", row=1, col=1)

    # ==== PLOT 2 (Bar Plot: Completion Ratios) ====
    df_voyage_comparison = df_voyage.groupby(["finished", "Wikispeedia_Voyage"])[["Wikispeedia_Voyage"]].count() 
    df_voyage_comparison.columns = ["count"]
    df_voyage_comparison = df_voyage_comparison.reset_index()
    df_voyage_comparison = df_voyage_comparison.sort_values(by="finished", ascending=False)
    df_voyage_comparison["percentage"] = df_voyage_comparison.groupby("Wikispeedia_Voyage")["count"].transform(lambda x: (x / x.sum()) * 100).round(1)
    df_voyage_comparison["voyage_label"] = df_voyage_comparison["Wikispeedia_Voyage"].map({False: "Non-Voyage", True: "Voyage"})
    df_voyage_comparison["finished_label"] = df_voyage_comparison["finished"].map({False: "Unfinished", True: "Finished"})

    for voyage_label, color in [("Voyage", color_voyage), ("Non-Voyage", 'gray')]:
        filtered_data = df_voyage_comparison[df_voyage_comparison["voyage_label"] == voyage_label]
        fig.add_trace(
            go.Bar(
                x=filtered_data["finished_label"],
                y=filtered_data["count"],
                text=filtered_data["percentage"],
                name=voyage_label,
                marker_color=color,
                texttemplate="%{text}%",
            ),
            row=1, col=2
        )
        
    # Axis labels
    fig.update_yaxes(title_text="Count", row=1, col=2)
    fig.update_xaxes(title_text="Path Type", row=1, col=2)    
        
    # ==== PLOT 3 (Bar Plot: Rating Distribution for Voyage Games) ====
    df_voyage_rating = df_finished_voyage[df_finished_voyage["Wikispeedia_Voyage"] == True].copy()
    df_voyage_rating["rating"] = df_voyage_rating["rating"].fillna('NaN').astype(str)
    df_voyage_rating = df_voyage_rating.groupby("rating")["rating"].count().reset_index(name="count")
    df_voyage_rating["count"] = df_voyage_rating["count"] / df_voyage_rating["count"].sum() * 100
    
    fig.add_trace(
        go.Bar(
            x=df_voyage_rating["rating"], 
            y=df_voyage_rating["count"], 
            marker_color=color_voyage, 
            name="Voyage",
            showlegend=False
        ),
        row=2, col=1
    )

    # axis labels
    fig.update_yaxes(title_text="Pourcentage", row=2, col=1)
    fig.update_xaxes(title_text="Rating", row=2, col=1)

    # ==== PLOT 4 (Bar Plot: Rating Distribution for Non-Voyage Games) ====
    df_non_voyage_rating = df_finished_voyage[df_finished_voyage["Wikispeedia_Voyage"] == False].copy()
    df_non_voyage_rating["rating"] = df_non_voyage_rating["rating"].fillna('NaN').astype(str)
    df_non_voyage_rating = df_non_voyage_rating.groupby("rating")["rating"].count().reset_index(name="count")
    df_non_voyage_rating["count"] = df_non_voyage_rating["count"] / df_non_voyage_rating["count"].sum() * 100

    fig.add_trace(
        go.Bar(
            x=df_non_voyage_rating["rating"], 
            y=df_non_voyage_rating["count"], 
            marker_color="gray", 
            name="Non-Voyage",
            showlegend=False
        ),
        row=2, col=2
    )

    # axis labels
    fig.update_yaxes(title_text="Pourcentage", row=2, col=2)
    fig.update_xaxes(title_text="Rating", row=2, col=2)

    # ==== Final Layout Update ====
    fig.update_layout(
        height=1000, width=1000,  # Adjust size of the overall figure
        title="Summary of Voyage and Non-Voyage Game Metrics",
        showlegend=True,
        legend_title="Legend",
        xaxis_title="Game Type",
        yaxis_title="Count/Percentage",
        violingap=0.4, 
        violinmode="overlay"
    )

    fig.show()

    
def plot_shortest_paths_matrix(df_shortest_path):

    # Total number of article pairs
    total_pairs = df_shortest_path.size

    # Number of reachable pairs (distance from 1 to 9)
    # Exclude self-pairs where distance is 0
    # Unreachable pairs are represented by -1

    # Create a mask for self-pairs (distance == 0)
    self_pairs_mask = (df_shortest_path == 0)

    # Create a mask for reachable pairs (distance between 1 and 9)
    reachable_mask = (df_shortest_path >= 1) & (df_shortest_path <= 9)


    reachable_pairs = np.count_nonzero(reachable_mask)
    unreachable_pairs = np.count_nonzero(df_shortest_path == -1)

    # Sparsity percentage: proportion of unreachable pairs
    sparsity_percentage = ((unreachable_pairs + len(self_pairs_mask) ) / total_pairs) * 100

    print(f"Total pairs: {total_pairs}")
    print(f"Reachable pairs: {reachable_pairs}")
    print(f"Unreachable pairs: {unreachable_pairs}")
    print(f"Sparsity percentage: {sparsity_percentage:.2f}%")

    # Create a binary matrix where 1 represents a reachable path and 0 represents an unreachable path
    sparsity_matrix = np.where(df_shortest_path == -1, 0, 1)

    plt.figure(figsize=(10, 8))
    plt.imshow(sparsity_matrix, cmap='Greys', interpolation='nearest')
    plt.title('Sparsity Pattern in Shortest Path Matrix')
    plt.xlabel('Target Article')
    plt.ylabel('Source Article')
    plt.colorbar(label='Reachability (1=Reachable, 0=Unreachable)')
    plt.show()

   
def generate_random_path(start, articles_links, nb_articles=30):
    
    if start not in articles_links or len(articles_links[start]) == 0:
        return start
    
    random_path = [start]
    
    while len(random_path) < nb_articles:
        # print(random_path)
        if random_path[-1] == "<":
            valid_art = [random_path[i] for i in range(len(random_path)) if (i == 0 or random_path[i-1] != '<') and (i == len(random_path)-1 or random_path[i+1] != '<') and random_path[i] != '<']
            next_article = np.random.choice(articles_links[valid_art[-1] if len(valid_art)>0 else valid_art])
            
        elif random_path[-1] not in articles_links :
            # print(random_path)
            random_path.pop(-1) 
            continue
        
        elif len(articles_links[random_path[-1]]) == 0:
            next_article = "<"
            
        else :
            next_article = np.random.choice(articles_links[random_path[-1]]) 
        
        if next_article in articles_links or next_article == "<":
            random_path.append(next_article)  
                     
    return ";".join(random_path)

def compute_mean_of_lists(df):
    for column in df.columns:
        # Apply mean to each list in the column and replace the list with its mean
        df[column] = df[column].apply(lambda x: sum(x) / len(x) if isinstance(x, list) else x)
    return df

def compute_proba_links(df_categories, parser) : 
    article_to_category = dict(zip(df_categories['article'], df_categories['level_1']))
    articles_links = {article: data["total_links"] for article, data in parser.parsed_articles.items()}
    articles_categories_list = {}
    articles_categories_proba = {}
    cat_to_cat_proba = {}
    for article in articles_links :
        articles_categories_list[article] = []
        for link in articles_links[article] :
            articles_categories_list[article].append(article_to_category.get(link, "None"))
        articles_categories_proba[article] = {category: articles_categories_list[article].count(category) / len(articles_categories_list[article]) for category in set(articles_categories_list[article])}
        
        if article_to_category.get(article, "None") not in cat_to_cat_proba.keys() :
            cat_to_cat_proba[article_to_category.get(article, "None")] = {}
            
        for category in set(articles_categories_list[article]) :
            if category not in cat_to_cat_proba[article_to_category.get(article, "None")].keys() :
                cat_to_cat_proba[article_to_category.get(article, "None")][category] = [articles_categories_proba[article][category]]
            else :  
                cat_to_cat_proba[article_to_category.get(article, "None")][category].append(articles_categories_proba[article][category])
                
    df = pd.DataFrame.from_dict(cat_to_cat_proba)
    df = compute_mean_of_lists(df)
    df["<"] = 1
    df.loc["<"] = 1
    df
    return df, articles_categories_proba, articles_categories_list

def compute_proba_path(path, cat_to_cat_proba_df):
    path = path.split(" -> ")
    proba_path = 1
    for i in range(min(len(path)-1, 1)):
        proba_path *= cat_to_cat_proba_df[path[i]][path[i+1]]
    return proba_path #** (1 / (len(path)))

def location_click_on_page(df_finished, parser):
    df_finished['position'] = np.NaN

    for i in range(len(df_finished)):
        articles = df_finished['path'][i].split(';')
        
        position = []
        for a in range(len(articles)-1):
            if articles[a+1] == '<' or articles[a] == '<':
                continue
            else:
                info = parser.find_link_positions(articles[a], articles[a+1])
                position.append(info['article_link_position'][0]/info['total_links'] if len(info['article_link_position']) != 0 else np.NaN)
        df_finished.loc[i, 'position'] = np.mean(position)
    return df_finished

from tqdm import tqdm

def find_category_position_articles(parser, df_categories, categories_others) :

    def mean_link_position_per_category(parser, df_categories, category= "Country") : 
        
        #parser.parse_all()
        articles_links = {article: data["total_links"] for article, data in parser.parsed_articles.items()}
        article_to_category = dict(zip(df_categories['article'], df_categories['level_1']))
        articles_links_voyage = {k: [v_select for v_select in v if v_select in article_to_category.keys() and article_to_category[v_select] == category] for k, v in articles_links.items()}
        position_voyage = []
        for article, voyage_list in tqdm(articles_links_voyage.items()):
            position = []
            # if category not in list(df_categories[df_categories['article'] == article]["level_1"]) :
            for a in voyage_list:
                info = parser.find_link_positions(article, a)
                position.append(info['article_link_position'][0]/info['total_links'] if len(info['article_link_position']) != 0 else np.NaN)
            position_voyage.append(np.mean(position))
            # else :
            #     position_voyage.append(np.NaN)
        return position_voyage
    link_per_cat = {}

    for category in categories_others:
        link_per_cat[category] = mean_link_position_per_category(parser, df_categories, category=category)
    return link_per_cat
