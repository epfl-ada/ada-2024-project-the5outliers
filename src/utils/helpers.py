import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
import seaborn as sn
import networkx as nx
from tqdm import tqdm

def create_treemap_data(df, show_articles=True):
    """
    Processes the DataFrame to generate labels, parents, values, and ids for a treemap.
    Hierarchy:
    level_1 -> level_2 (if present) -> level_3 (if present) -> article (if show_articles=True)

    If level_2 is missing for an article, it is placed directly under level_1.
    If level_3 is missing for an article but level_2 is present, it is placed directly under (level_1, level_2).
    Otherwise, if level_3 is present, articles go under (level_1, level_2, level_3).

    Parameters:
    - df (DataFrame): Must contain columns 'level_1', 'level_2', 'level_3', 'article'.
    - show_articles (bool): Whether to include articles as leaves in the treemap. Default is True.
    """

    # Replace None with empty strings to simplify checks
    df = df.fillna('')
    labels, ids, parents, values, colors = [], [], [], [], []

    # Dictionary to quickly find parent IDs at different levels
    node_ids = {}

    # Count how many rows (articles) at each level
    counts_level_1 = df.groupby(['level_1']).size().reset_index(name='count')
    # Only create level_2 nodes for those actually having a non-empty level_2
    counts_level_2 = df[df['level_2'] != ''].groupby(['level_1', 'level_2']).size().reset_index(name='count')
    # Only create level_3 nodes for those actually having a non-empty level_3
    counts_level_3 = df[(df['level_2'] != '') & (df['level_3'] != '')].groupby(['level_1', 'level_2', 'level_3']).size().reset_index(name='count')

    # For articles, split into three categories based on their lowest level
    articles_level_1 = df[(df['level_2'] == '')].groupby(['level_1', 'article']).size().reset_index(name='count')
    articles_level_2 = df[(df['level_2'] != '') & (df['level_3'] == '')].groupby(['level_1', 'level_2', 'article']).size().reset_index(name='count')
    articles_level_3 = df[(df['level_2'] != '') & (df['level_3'] != '')].groupby(['level_1', 'level_2', 'level_3', 'article']).size().reset_index(name='count')

    # -------------------- Create Level 1 Nodes --------------------
    for _, row in counts_level_1.iterrows():
        level_1 = row['level_1']
        label = level_1
        _id = level_1
        parent_id = ''  # top-level node
        labels.append(label)
        ids.append(_id)
        parents.append(parent_id)
        values.append(row['count'])
        node_ids[(level_1,)] = _id

    # -------------------- Create Level 2 Nodes --------------------
    for _, row in counts_level_2.iterrows():
        level_1 = row['level_1']
        level_2 = row['level_2']
        if level_2 != '':
            label = level_2
            _id = f"{level_1}/{level_2}"
            parent_id = node_ids.get((level_1,), level_1)
            labels.append(label)
            ids.append(_id)
            parents.append(parent_id)
            values.append(row['count'])
            node_ids[(level_1, level_2)] = _id

    # -------------------- Create Level 3 Nodes --------------------
    for _, row in counts_level_3.iterrows():
        level_1 = row['level_1']
        level_2 = row['level_2']
        level_3 = row['level_3']
        if level_3 != '':
            label = level_3
            _id = f"{level_1}/{level_2}/{level_3}"
            parent_id = node_ids.get((level_1, level_2), f"{level_1}/{level_2}")
            labels.append(label)
            ids.append(_id)
            parents.append(parent_id)
            values.append(row['count'])
            node_ids[(level_1, level_2, level_3)] = _id

    # -------------------- Create Article Nodes (if show_articles) --------------------
    if show_articles:
        # Articles directly under level_1 (no level_2)
        for _, row in articles_level_1.iterrows():
            level_1 = row['level_1']
            article = row['article']
            label = article
            _id = f"{level_1}/{article}"
            parent_id = node_ids.get((level_1,), level_1)
            labels.append(label)
            ids.append(_id)
            parents.append(parent_id)
            values.append(row['count'])

        # Articles under (level_1, level_2) but no level_3
        for _, row in articles_level_2.iterrows():
            level_1 = row['level_1']
            level_2 = row['level_2']
            article = row['article']
            label = article
            _id = f"{level_1}/{level_2}/{article}"
            parent_id = node_ids.get((level_1, level_2), f"{level_1}/{level_2}")
            labels.append(label)
            ids.append(_id)
            parents.append(parent_id)
            values.append(row['count'])

        # Articles under (level_1, level_2, level_3)
        for _, row in articles_level_3.iterrows():
            level_1 = row['level_1']
            level_2 = row['level_2']
            level_3 = row['level_3']
            article = row['article']
            label = article
            _id = f"{level_1}/{level_2}/{level_3}/{article}"
            parent_id = node_ids.get((level_1, level_2, level_3), f"{level_1}/{level_2}/{level_3}")
            labels.append(label)
            ids.append(_id)
            parents.append(parent_id)
            values.append(row['count'])

    return labels, parents, values, ids

def create_colored_treemap(labels, parents, values, ids, color_palette=None, title="Treemap", background_color='transparent'):
    """
    Creates a Plotly Treemap with colors propagated from level_1 to all children.

    Parameters:
    - labels (list): List of node labels.
    - parents (list): List of parent nodes.
    - values (list): List of values (used for proportional sizing).
    - ids (list): List of unique node IDs.
    - color_palette (dict): Dictionary mapping level_1 labels to colors. If None, a default palette is used.
    - title (str): Title of the treemap.
    - background_color (str): Background color of the plot ('white' or 'transparent').

    Returns:
    - fig (plotly.graph_objects.Figure): A Plotly Treemap figure.
    """

    # Function to propagate level_1 color to all children
    def get_colors_for_hierarchy(ids, color_palette):
        colors = []
        for tag in ids:
            # Extract the level_1 part of the label (before any slash '/')
            level_1 = tag.split('/')[0]
            # Get the color for level_1; default to light gray if not found
            color = color_palette.get(level_1, '#d3d3d3')
            colors.append(color)
        return colors

    # Generate colors for the hierarchy if palette is given
    colors = get_colors_for_hierarchy(ids, color_palette) if color_palette else None

    # Determine the background color settings
    if background_color == 'transparent':
        paper_bgcolor = 'rgba(0,0,0,0)'  # Transparent
        plot_bgcolor = 'rgba(0,0,0,0)'   # Transparent
    else:
        paper_bgcolor = background_color  # Solid color
        plot_bgcolor = background_color   # Solid color

    # Create the Treemap
    fig = go.Figure(go.Treemap(
        labels=labels,
        parents=parents,
        values=values,
        ids=ids,
        marker=dict(colors=colors), # Apply colors if available
        textfont=dict(size=18),
        branchvalues='total'  # Ensures proportional sizing by summation of children
    ))

    # Update the layout with background color and title
    fig.update_layout(
        margin=dict(t=50, l=10, r=10, b=5),
        title=title,
        paper_bgcolor=paper_bgcolor,
        plot_bgcolor=plot_bgcolor
    )

    return fig

def assign_world_region_categories(df_categories, world_region_categories):
    """
    Processes a DataFrame to standardize and categorize subject categories, 
    specifically handling those related to 'World Regions'.

    Steps:
    1. Strips the prefix 'subject.' from values in the 'category' column if it exists.
    2. Replaces categories containing any string from `world_region_categories` with 'World Regions'.
    3. Updates rows where 'category' is 'World Regions':
       - Sets 'level_1' to 'World Regions'.
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
        'World Regions' if any(region in category for region in world_region_categories) else category
        for category in df_categories_filtered['category']
    ]
    # Updating level_1, level_2, and level_3 based on 'World Region' in 'category'
    df_categories_filtered.loc[df_categories_filtered['category'] == 'World Regions', ['level_2', 'level_1']] = df_categories_filtered.loc[
        df_categories_filtered['category'] == 'World Regions'
    ].apply(
        lambda row: (
            'Countries' if row['level_1'] == 'Countries' else row['level_2'],
            'World Regions'  # level_1 is always 'World Regions'
        ),
        axis=1
    ).to_list()

    return df_categories_filtered

def compute_optimal_paths(df_links, df_shortest_path, df_finished, df_unfinished):
    optimal_paths = find_all_source_target_pairs(df_finished, df_unfinished, df_links)
    optimal_paths = calculate_optimal_path(df_links, optimal_paths, df_shortest_path)
    optimal_paths = optimal_paths.drop(columns=['matrix_length','matches_matrix'])

    # Save the optimal paths to a pickle file
    optimal_paths.to_pickle('data/paths-and-graph/optimal_paths.pkl')

    return optimal_paths

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
    df_finished['source'] = df_finished['path'].str.split(';').apply(lambda x: x[0])  # First element in path
    df_finished['target'] = df_finished['path'].str.split(';').apply(lambda x: x[-1])  # Last element in path
    users_finished = df_finished[['source', 'target', 'path']].copy()
    users_finished['path'] = users_finished['path'].str.split(';')
    users_finished['path'] = users_finished['path'].apply(clean_path_list)
    users_finished['Category Path'] = users_finished['path'].apply(
        lambda path: map_path_to_categories(path, article_to_category)
    )

    # Process unfinished paths
    users_unfinished = df_unfinished[['target', 'path']].copy()
    users_unfinished['path'] = users_unfinished['path'].str.split(';')
    users_unfinished['path'] = users_unfinished['path'].apply(clean_path_list)
    users_unfinished['source'] = users_unfinished['path'].str[0]
    users_unfinished = users_unfinished[users_unfinished['path'].apply(lambda x: len(x) > 1)]
    users_unfinished['Category Path'] = users_unfinished['path'].apply(
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
    - group: DataFrame group containing a 'Category Path' column.
    - unfinished: Boolean indicating if the paths are unfinished (affects slicing logic).

    Returns:
    - DataFrame with percentage distributions at each step.
    """
    paths = pd.DataFrame(group['Category Path'].tolist())
    if unfinished:
        paths = paths.iloc[:, :]  # Exclude the source
    else:
        paths = paths.iloc[:, :-1]  # Exclude source and target

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
        df = df.drop(columns='step_1')
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
    category_means = category_means.set_index(['source', 'target', 'categories']).unstack(fill_value=0).stack(future_stack=True).reset_index()

    # Map source and target to their respective categories
    category_means['source_category'] = category_means['source'].map(article_to_category)
    category_means['target_category'] = category_means['target'].map(article_to_category)
    category_means['source_target'] = category_means['source_category'] + ' -> ' + category_means['target_category']

    # Group by source-target-category combination and calculate mean percentage_diff
    category_means = category_means.groupby(
        ['source_category', 'target_category', 'source_target', 'categories']
    )['percentage_diff'].mean().reset_index()

    # Generate aggregated data for plotting
    category_means = pd.DataFrame(category_means.groupby('categories')['percentage_diff'].mean().reset_index())
    return category_means

def process_and_calculate_differences(dataframes, article_to_category, column_names=['source', 'target', 'categories']):
    """
    Combines the calculation of averages, merging of differences, and processing of category means 
    into a single function.

    Parameters:
    - dataframes: List of tuples, each containing a DataFrame and its suffix for renaming.
    - key_columns: List of column names to merge on.
    - article_to_category: Mapping of articles to categories.
    - column_names: List of column names for renaming.

    Returns:
    - category_fin_means, category_fin_means_norm, category_unf_means, category_unf_means_norm
    """
    key_columns = ['source', 'target', 'categories']

    # Calculate averages
    S_T_opt_fin_avg, S_T_fin_avg, S_T_opt_unf_avg, S_T_unf_avg = calculate_average_percentages(
        dataframes, column_names
    )

    # Merge and calculate differences
    S_T_diff_fin = merge_and_calculate_difference(S_T_opt_fin_avg, S_T_fin_avg, key_columns, ('percentage_fin', 'percentage_opt'))
    S_T_diff_unf = merge_and_calculate_difference(S_T_opt_unf_avg, S_T_unf_avg, key_columns, ('percentage_unf', 'percentage_opt'))

    # Process category means
    category_fin_means = process_category_means(S_T_diff_fin, article_to_category)
    category_unf_means = process_category_means(S_T_diff_unf, article_to_category)

    return category_fin_means, category_unf_means

def plot_position_line(S_T_fin_percentages_norm_steps, S_T_opt_fin_percentages_norm_steps, category_fin_means_norm, 
                       category_unf_means_norm, palette, title="Category Percentage Used at Each Step of Finished Paths",background_color='white'):
    """
    Plot an interactive line plot of category frequencies across positions with both normalized and non-normalized views.
    Add bar plot as a subplot with inverted axes and hashed bars for 'unfinished'.
    
    Parameters:
        S_T_fin_percentages_norm_steps (DataFrame): DataFrame with user data.
        S_T_opt_fin_percentages_norm_steps (DataFrame): DataFrame with optimal data.
        category_fin_means_norm (DataFrame): Finished category percentages.
        category_unf_means_norm (DataFrame): Unfinished category percentages.
        palette (dict): Color palette for categories.
        title (str): Title of the plot.
        background_color (str): Background color of the plot ('white' or 'transparent').
    """
    # Extract and sort categories
    categories = S_T_fin_percentages_norm_steps["categories"].unique()
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=3, 
        subplot_titles=("Users' paths", "Optimal paths", "Percentage difference across all steps"),
        horizontal_spacing=0.05
    )
    
    # Add non-normalized line plot traces (Users)
    for category in categories:
        category_data = S_T_fin_percentages_norm_steps[S_T_fin_percentages_norm_steps["categories"] == category]
        fig.add_trace(
            go.Scatter(
                x=category_data['step'], 
                y=category_data['percentage'],
                mode="lines+markers",
                name=category,
                line=dict(color=palette.get(category, 'grey'))
            ), row=1, col=1
        )
    
    # Add normalized line plot traces (Optimal)
    for category in categories:
        category_data = S_T_opt_fin_percentages_norm_steps[S_T_opt_fin_percentages_norm_steps["categories"] == category]
        fig.add_trace(
            go.Scatter(
                x=category_data['step'], 
                y=category_data['percentage'],
                mode="lines+markers",
                name=category,
                line=dict(color=palette.get(category, 'grey')),
                showlegend=False  # Show legend only on the first subplot
            ), row=1, col=2
        )
    
    # Combine datasets for bar plot
    category_fin_means_norm['path'] = 'finished'
    category_unf_means_norm['path'] = 'unfinished'
    concat_means_norm = pd.concat([category_fin_means_norm, category_unf_means_norm])

    # Prepare bar plot data
    for category in categories:
        bar_data_finished = concat_means_norm[(concat_means_norm['categories'] == category) & (concat_means_norm['path'] == 'finished')]
        bar_data_unfinished = concat_means_norm[(concat_means_norm['categories'] == category) & (concat_means_norm['path'] == 'unfinished')]
        
   # Prepare bar plot data
    bar_width = 0.4  # Set width for each bar
        
    for i, category in enumerate(categories):
        bar_data_finished = concat_means_norm[(concat_means_norm['categories'] == category) & (concat_means_norm['path'] == 'finished')]
        bar_data_unfinished = concat_means_norm[(concat_means_norm['categories'] == category) & (concat_means_norm['path'] == 'unfinished')]

        # Add finished bar
        fig.add_trace(
            go.Bar(
                x=[bar_data_finished['percentage_diff'].values[0]],
                y=[i],
                orientation='h',
                width=bar_width,
                marker=dict(color=palette.get(category, 'grey'), line=dict(width=0)),
                name='Finished', 
                showlegend=(i == 14)
            ), row=1, col=3
        )

        # Add unfinished bar with hashing
        fig.add_trace(
            go.Bar(
                x=[bar_data_unfinished['percentage_diff'].values[0]],
                y=[i + bar_width],
                orientation='h',
                width=bar_width,
                marker=dict(color=palette.get(category, 'grey'), pattern_shape="/"),
                name='Unfinished',  
                showlegend=(i == 14)
            ), row=1, col=3
        )

    # Determine the background color
    if background_color == 'transparent':
        paper_bgcolor = 'rgba(0,0,0,0)'  # Transparent
        plot_bgcolor = 'rgba(0,0,0,0)'   # Transparent
    else:
        paper_bgcolor = background_color  # Solid color
        plot_bgcolor = background_color   # Solid color

    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Step",
        xaxis2_title="Step",
        yaxis=dict(title="Percentage", automargin=True, range=[0, 50]),
        yaxis2=dict(automargin=True, range=[0, 50]),
        yaxis3=dict(autorange="reversed"),  # Reverse the y-axis order
        xaxis3_title="Percentage difference",
        legend_title_text="Path Type",
        template="plotly_white",
        width=1300,
        height=550,
        paper_bgcolor=paper_bgcolor,
        plot_bgcolor=plot_bgcolor
    )

    # Remove y-axis for the third subplot
    fig.update_yaxes(showticklabels=False, title=None, row=1, col=3)

    # Show the plot
    fig.show()

    return fig


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

def process_percentages(S_T_opt_fin_percentages, S_T_fin_percentages, category_map, max_step=10):
    """
    Processes and normalizes percentages for both user and optimal dataframes,
    including reshaping, grouping, and normalization.

    Parameters:
    - S_T_opt_fin_percentages: DataFrame for optimal finished percentages.
    - S_T_fin_percentages: DataFrame for user finished percentages.
    - category_map: Mapping of articles to categories.
    - max_step: Maximum step to keep in the output (default is 10).

    Returns:
    - S_T_fin_percentages_norm_steps: Processed DataFrame for user percentages.
    - S_T_opt_fin_percentages_norm_steps: Processed DataFrame for optimal percentages.
    """
    def prepare_percentages(df, category_map, max_step):
        """Unstacks, fills, maps categories, groups, and reshapes a DataFrame."""
        # Unstack and reset
        df = df.unstack(fill_value=0).stack(future_stack=True).reset_index()
        
        # Fill step columns with 0
        step_cols = [col for col in df.columns if col.startswith('step_')]
        df[step_cols] = df[step_cols].fillna(0)
        
        # Map categories
        df['source_category'] = df['source'].map(category_map)
        df['target_category'] = df['target'].map(category_map)
        df['source_target'] = df['source_category'] + ' -> ' + df['target_category']
        
        # Group by source_target and level_2
        df_norm_steps = df.groupby(['source_target', 'level_2'])[step_cols].mean().reset_index()
        
        # Further group by level_2
        df_norm_steps = df_norm_steps.groupby(['level_2'])[step_cols].mean().reset_index()
        
        # Rename and filter columns
        df_norm_steps = df_norm_steps.rename(columns={'level_2': 'categories'})
        step_columns = [col for col in step_cols if int(col.split('_')[1]) <= max_step]
        columns_to_keep = ['categories'] + step_columns
        df_norm_steps = df_norm_steps[columns_to_keep]
        
        # Melt to long format
        df_norm_steps = df_norm_steps.melt(id_vars=['categories'], 
                                           var_name='step', 
                                           value_name='percentage')
        
        # Extract step numbers and normalize percentages
        df_norm_steps['step'] = df_norm_steps['step'].str.extract('(\d+)').astype(int)
        df_norm_steps['percentage'] = df_norm_steps.groupby('step')['percentage'].transform(lambda x: x / x.sum() * 100)
        return df_norm_steps

    # Process both user and optimal dataframes
    S_T_fin_percentages_norm_steps = prepare_percentages(S_T_fin_percentages, category_map, max_step)
    S_T_opt_fin_percentages_norm_steps = prepare_percentages(S_T_opt_fin_percentages, category_map, max_step)
    
    return S_T_fin_percentages_norm_steps, S_T_opt_fin_percentages_norm_steps


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

    
def check_voyage_status(row):
    """
    Check if the path is a Wikispeedia_Voyage, that is whether categories (not source and target) are 'Voyages'. 

    Parameters:
        row: A row of a dataframe containing information about paths and categories.

    Returns:
        bool: True if the path is a Wikispeedia_Voyage, False otherwise.
    """    
    
    if row['target_maincategory']=='World Regions' or row['source_maincategory']=='World Regions':
        return False
    else: return any('World Regions' in category for category in row['Category Path'])

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
    # omit_loops=True thus showing only the transition within categories
    category_path_df = get_main_categories_paths(df_article_paths, df_categories, omit_loops=True, one_level=True)

    # Get the full category path for each article path
    # omit_loops=False thus showing the full category path
    category_path_df_loops = get_main_categories_paths(df_article_paths, df_categories, omit_loops=False, one_level=True)
    
    # Apply the transformation and check voyage status
    df_article_paths['Transition Category Path'] = category_path_df['Category Path']
    df_article_paths['Category Path'] = category_path_df_loops['Category Path']
    df_article_paths['source_maincategory'] = category_path_df['source_maincategory']
    df_article_paths['target_maincategory'] = category_path_df['target_maincategory']
    df_article_paths['Wikispeedia_Voyage'] = df_article_paths.apply(lambda row: check_voyage_status(row), axis=1)
    
    return df_article_paths

def plot_sankey_voyage(df, background_color='transparent'):
    """
    Plots a Sankey diagram to visualize the distribution of paths classified as 'Voyage' or 'Non-Voyage'.

    Parameters:
        df (DataFrame): DataFrame containing boolean 'Wiki_Voyage' and 'source_maincategory', 'target_maincategory' columns
        background_color (str): Background color of the plot ('white' or 'transparent').

    Returns:
        Figure: Sankey diagram as a Plotly Figure object.
    """

    # Mapping for start, voyage, and end nodes
    df_all_voyage = df.copy()
    df_all_voyage['source_category_label'] = df_all_voyage['source_maincategory'].apply(lambda x: 'Source is a World Regions' if x == 'World Regions' else 'Source is not a World Regions')
    df_all_voyage['target_category_label'] = df_all_voyage['target_maincategory'].apply(lambda x: 'Target is a World Regions' if x == 'World Regions' else 'Target is not a World Regions')
    df_all_voyage['voyage_label'] = df_all_voyage['Wikispeedia_Voyage'].apply(lambda x: 'Voyages' if x else 'Non-Voyages')

    # Start→Voyage flows
    start_voyage_flows = df_all_voyage.groupby(['source_category_label', 'voyage_label']).size().reset_index(name='count')

    # Voyage→End flows
    voyage_end_flows = df_all_voyage.groupby(['voyage_label', 'target_category_label']).size().reset_index(name='count')

    # Define node labels
    labels = ['Source is a World Regions', 'Source is not a World Regions',
              'Voyages', 'Non-Voyages',
              'Target is a World Regions', 'Target is not a World Regions']

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
        '#4b4b4b'   # Target not in Countries/Geography
    ]

    link_colors = [
        'rgba(75, 75, 75, 0.3)',  # not voyage
        'rgba(75, 75, 75, 0.3)',  # not voyage
        'rgba(44, 181, 174, 0.3)',  # Voyage
        'rgba(75, 75, 75, 0.3)',  # not voyage
        'rgba(75, 75, 75, 0.3)',  # not voyage
        'rgba(44, 181, 174, 0.3)'  # Voyage
    ]

    # Determine the background color
    if background_color == 'transparent':
        paper_bgcolor = 'rgba(0,0,0,0)'  # Transparent
        plot_bgcolor = 'rgba(0,0,0,0)'   # Transparent
    else:
        paper_bgcolor = background_color  # Solid color
        plot_bgcolor = background_color   # Solid color

    # Create the Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        node=dict(pad=20, thickness=20, line=dict(color="white", width=0), label=labels, color=node_colors),
        link=dict(source=sources, target=targets, value=values, color=link_colors) #color='rgba(60,60,60,0.3)
    )])
    
    fig.update_layout(
        title_text="Voyage and Non-Voyage Paths",
        font_size=10,
        title_font_size=14,
        title_x=0.5,
        paper_bgcolor=paper_bgcolor,  #remove to set bg white 
        plot_bgcolor=plot_bgcolor   #remove to set bg white 
    )
    
    return fig

def plot_articles_pie_chart(df, palette, abbreviations=None):
    """
    Plots a simplified pie chart of the total number of articles per Level 1 category.

    Parameters:
    - df (DataFrame): The DataFrame containing 'article' and 'level_1' columns.
    - abbreviations (dict, optional): A dictionary mapping full category names to abbreviations.
    """
    # Group by Level 1 category and count the number of articles
    category_counts = df['level_1'].value_counts()
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
        colors = [palette.get(label, '#cccccc') for label in large_categories.index]
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
        #textprops={'color':"w"}
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
        bbox_to_anchor=(1.15, 0.7)
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


def plot_difficulties_voyage (df_finished, df_unfinished, palette_category_dict):
    color_voyage = palette_category_dict['World Regions']
    
    df_finished_voyage = df_finished.copy()
    df_unfinished_voyage = df_unfinished.copy()

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
                y=filtered_data["percentage"],
                text=filtered_data["count"],
                name=voyage_label,
                marker_color=color,
                texttemplate="Count: %{text}",
            ),
            row=1, col=2
        )
        
    # Axis labels
    fig.update_yaxes(title_text="Pourcentage (%)", row=1, col=2)
    fig.update_xaxes(title_text="Path Type", row=1, col=2)    
        
    # ==== PLOT 3 (Bar Plot: Rating Distribution for Voyage Games) ====
    df_voyage_rating = df_finished_voyage[df_finished_voyage["Wikispeedia_Voyage"] == True].copy()
    df_voyage_rating["rating"] = df_voyage_rating["rating"].fillna('NaN').astype(str)
    df_voyage_rating["have_back_click"] = df_voyage_rating["back_clicks"] > 0
    back_click_per_rating = pd.DataFrame(df_voyage_rating.groupby("rating")["have_back_click"].mean()).reset_index() # count the number of path with a rating for each rating
    df_voyage_rating = df_voyage_rating.groupby("rating")["rating"].count().reset_index(name="count")
    df_voyage_rating["pourcent"] = df_voyage_rating["count"] / df_voyage_rating["count"].sum() * 100
    df_voyage_rating["back_clicks"] = back_click_per_rating["have_back_click"].values
    df_voyage_rating["Back-Click"] = df_voyage_rating["back_clicks"]*df_voyage_rating["pourcent"]
    df_voyage_rating["Without Back-Click"] = (1 - df_voyage_rating["back_clicks"])*df_voyage_rating["pourcent"]
    df_voyage_rating = pd.melt(df_voyage_rating, id_vars=['rating', 'count', 'pourcent'], value_vars=['Back-Click', 'Without Back-Click'], var_name='back_click')

    df = df_voyage_rating[df_voyage_rating["back_click"]== 'Back-Click']
    fig.add_trace(go.Bar(x=df["rating"],
                            y=df["value"],
                            marker_color=color_voyage,
                            marker_pattern_shape="/",
                            offsetgroup=0,
                            legendgroup='Back-Click',
                            name = 'Back-Click'),
                    row=2, col=1)
    df = df_voyage_rating[df_voyage_rating["back_click"]== 'Without Back-Click']
    fig.add_trace(go.Bar(x=df["rating"],
                            y=df["value"],
                            marker_color=color_voyage,
                            offsetgroup=0,
                            base = df_voyage_rating[df_voyage_rating["back_click"]== 'Back-Click']["value"],
                            legendgroup='Without Back-Click',
                            name = 'Without Back-Click'),
                    row=2, col=1)

    # axis labels
    fig.update_yaxes(title_text="Pourcentage", row=2, col=1)
    fig.update_xaxes(title_text="Rating", row=2, col=1)

    # # ==== PLOT 4 (Bar Plot: Rating Distribution for Non-Voyage Games) ====
    df_non_voyage_rating = df_finished_voyage[df_finished_voyage["Wikispeedia_Voyage"] == False].copy()
    df_non_voyage_rating["rating"] = df_non_voyage_rating["rating"].fillna('NaN').astype(str)
    df_non_voyage_rating["have_back_click"] = df_non_voyage_rating["back_clicks"] > 0
    back_click_per_rating = pd.DataFrame(df_non_voyage_rating.groupby("rating")["have_back_click"].mean()).reset_index() # count the number of path with a rating for each rating
    df_non_voyage_rating = df_non_voyage_rating.groupby("rating")["rating"].count().reset_index(name="count")
    df_non_voyage_rating["pourcent"] = df_non_voyage_rating["count"] / df_non_voyage_rating["count"].sum() * 100
    df_non_voyage_rating["back_clicks"] = back_click_per_rating["have_back_click"].values
    df_non_voyage_rating["Back-Click"] = df_non_voyage_rating["back_clicks"]*df_non_voyage_rating["pourcent"]
    df_non_voyage_rating["Without Back-Click"] = (1 - df_non_voyage_rating["back_clicks"])*df_non_voyage_rating["pourcent"]
    df_non_voyage_rating = pd.melt(df_non_voyage_rating, id_vars=['rating', 'count', 'pourcent'], value_vars=['Back-Click', 'Without Back-Click'], var_name='back_click')

    df = df_non_voyage_rating[df_non_voyage_rating["back_click"]== 'Back-Click']
    fig.add_trace(go.Bar(x=df["rating"],
                            y=df["value"],
                            marker_color="gray",
                            marker_pattern_shape="/",
                            offsetgroup=0,
                            showlegend=False,
                            name = 'Back-Click'),
                    row=2, col=2)
    df = df_non_voyage_rating[df_non_voyage_rating["back_click"]== 'Without Back-Click']
    fig.add_trace(go.Bar(x=df["rating"],
                            y=df["value"],
                            marker_color="gray",
                            offsetgroup=0,
                            base = df_non_voyage_rating[df_non_voyage_rating["back_click"]== 'Back-Click']["value"],
                            showlegend=False,
                            name = 'Without Back-Click'),
                    row=2, col=2)

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

def location_click_on_page(df, parser):

    for i in range(len(df)):
        articles = df['path'][i].split(';')
        
        position = []
        core = np.zeros(len(articles)-1)
        total = 0
        abstract = np.zeros(len(articles)-1)
        total_abstract = 0
        infobox = np.zeros(len(articles)-1)
        total_infobox = 0
        for j, a in enumerate(range(len(articles)-1)):
            if articles[a+1] == '<' or articles[a] == '<':
                continue
            else:
                info = parser.find_link_positions(articles[a], articles[a+1])
                position.append(np.mean(info['article_link_position'])/info['total_links'] if len(info['article_link_position']) != 0 else np.nan)
                
                # case no table in abstract :
                table_link = info.get("table_link_position", [])
                table_not_in_abstract = [table for table in table_link if table not in info.get("article_link_position", [])]
                core[j] = 1 if len(info["article_link_position"]) > (len(info.get("abstract_link_position", [])) + len(table_not_in_abstract)) else 0
                
                total += info['total_links']
                abstract[j] = 1 if 'abstract_link_position' in info.keys() else 0
                total_abstract += info["abstract_links"] if 'abstract_links' in info.keys() else 0
                infobox[j]= 1 if 'table_link_position' in info.keys() else 0
                total_infobox += info["tables"] if 'tables' in info.keys() else 0
                
        df.loc[i, 'position'] = np.mean(position) if len(position) != 0 else np.nan
        df.loc[i, 'link_in_core'] = np.sum(core)
        df.loc[i, 'total_links'] = total
        df.loc[i, 'link_in_abstract'] = np.sum(abstract)
        
        df.loc[i, 'total_link_in_abstract'] = total_abstract
        df.loc[i, 'link_in_infobox'] = np.sum(infobox)
        df.loc[i, 'total_link_in_infobox'] = total_infobox
    return df

def find_category_position_articles(parser, df_categories, categories_others) :

    def mean_link_position_per_category(parser, df_categories, category= "Country") : 
        
        #parser.parse_all()
        articles_links = {article: data["total_links"] for article, data in parser.parsed_articles.items()}
        article_to_category = dict(zip(df_categories['article'], df_categories['level_1']))
        articles_links_voyage = {k: [v_select for v_select in v if v_select in article_to_category.keys() and article_to_category[v_select] == category] for k, v in articles_links.items()}
        position_voyage = []
        for article, voyage_list in articles_links_voyage.items():
            position = []
            # if category not in list(df_categories[df_categories['article'] == article]["level_1"]) :
            for a in voyage_list:
                info = parser.find_link_positions(article, a)
                position.append(np.mean(info['article_link_position'])/info['total_links'] if len(info['article_link_position']) != 0 else np.nan)
            position_voyage.append(np.mean(position) if len(position) != 0 else np.nan)
            # else :
            #     position_voyage.append(np.nan)
        return position_voyage
    link_per_cat = {}

    for category in categories_others:
        link_per_cat[category] = mean_link_position_per_category(parser, df_categories, category=category)
    return link_per_cat

def plot_comparison_category_click_position(df_merged, df_category_position, colors = {'Clicked Link Position in Paths': '#AFD2E9', 'Article Link Position in Articles': '#9A7197'}):
    df_merged["category"] = "All"
    df_merged["Legend :"] = "Clicked Link Position in Paths"
    df_melted = pd.melt(df_category_position, var_name='category', value_name='position').dropna()
    df_melted["Legend :"] = "Article Link Position in Articles"
    df_comparison_path_category = pd.concat([df_merged[["category", "position", "Legend :"]], df_melted])

    fig = px.box(df_comparison_path_category, x="category", y="position", color="Legend :", title="Position of the clicked link in articles compared to position of each category in articles", color_discrete_map=colors)
    fig.update_xaxes(tickangle=45)

    fig.update_layout(
        autosize=False,
        width=1500,
        height=500,
        boxgroupgap=0.2, # update
        boxgap=0)
    fig.show()

def plot_donut_link_position(df_merged, colors):

    df_info = df_merged[["total_links", "total_link_in_abstract", "total_link_in_infobox", "link_in_core", "link_in_abstract", "link_in_infobox"]]
    df_info["total_link_core"] = df_info["total_links"] - df_info["total_link_in_abstract"] - df_info["total_link_in_infobox"]
    df_info["link_all"] = df_info["link_in_core"] + df_info["link_in_abstract"] + df_info["link_in_infobox"]
    df_info

    df_ratio_path = df_info[["total_link_in_abstract", "total_link_in_infobox", "total_link_core"]].div(df_info["total_links"], axis=0)
    df_ratio_path = df_ratio_path.mean()
    df_ratio_click = df_info[["link_in_abstract", "link_in_infobox", "link_in_core"]].div(df_info["link_all"], axis=0)
    df_ratio_click = df_ratio_click.mean()

    data = [

        go.Pie(values=[df_ratio_path["total_link_core"], df_ratio_path["total_link_in_abstract"], df_ratio_path["total_link_in_infobox"]],
        labels=['Body','Abstract', "Info-box"],
        domain={'x':[0.3,0.7], 'y':[0.16,0.8]}, 
        hole=0.5,
        direction='clockwise',
        sort=False,
        title=dict(
                text="Link position<br>in articles",
                font=dict(size=16)
            ),
        texttemplate="%{percent:.0%}",
        marker={'colors':colors},), 

        go.Pie(values=[df_ratio_click["link_in_core"] ,df_ratio_click["link_in_abstract"] ,df_ratio_click["link_in_infobox"]],
            labels=['Body','Abstract', "Info-box"],
            domain={'x':[0.1,0.9], 'y':[0,1]},
            hole=0.75,
            direction='clockwise',
            sort=False,
            marker={'colors':colors},
            title=dict(
                text="Link position in clicks",
                font=dict(size=16)
            ),
            titleposition='top center',
            texttemplate="%{percent:.0%}",
            showlegend=True)
    ]
        
    figure=go.Figure(data=data, layout={'title':'Mean distribution of link positions across all articles in all paths <br>compared to the mean distribution of clicked link positions in all paths.', 'width': 800, 'height': 600})

    figure.show()