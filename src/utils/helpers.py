import pandas as pd
import plotly.graph_objects as go
from collections import Counter
import numpy as np
import networkx as nx

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

    fig.show()

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
    """
    Computes the optimal paths for all source-target pairs.
    
    Parameters:
        df_links (pd.DataFrame): DataFrame containing graph links with 'linkSource' and 'linkTarget' columns.
        df_shortest_path (pd.DataFrame): DataFrame containing the shortest path lengths between all source-target pairs.
        df_finished (pd.DataFrame): DataFrame containing finished paths.
        df_unfinished (pd.DataFrame): DataFrame containing unfinished paths.
    
    Returns:
        pd.DataFrame: DataFrame with computed paths, path lengths, and comparison results.
    """
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

def find_category_position_articles(parser, df_categories, categories_others) :
    """
    Find the link position going to a specific category for each article.
    
    Parameters:
        parser (wikispeedia.Parser): The parser object containing the parsed articles.
        df_categories (DataFrame): DataFrame containing the categories of articles.
        categories_others (list): List of categories to analyze.
    
    Returns:
        dict: Dictionary containing the mean link position for each category.
    """

    def mean_link_position_per_category(parser, df_categories, category= "Country") : 
        
        articles_links = {article: data["total_links"] for article, data in parser.parsed_articles.items()}
        article_to_category = dict(zip(df_categories['article'], df_categories['level_1']))
        articles_links_voyage = {k: [v_select for v_select in v if v_select in article_to_category.keys() and article_to_category[v_select] == category] for k, v in articles_links.items()}
        position_voyage = []
        for article, voyage_list in articles_links_voyage.items():
            position = []

            for a in voyage_list:
                info = parser.find_link_positions(article, a)
                position.append(np.mean(info['article_link_position'])/info['total_links'] if len(info['article_link_position']) != 0 else np.nan)
            position_voyage.append(np.mean(position) if len(position) != 0 else np.nan)
            
        return position_voyage
    link_per_cat = {}

    for category in categories_others:
        link_per_cat[category] = mean_link_position_per_category(parser, df_categories, category=category)
    return link_per_cat


def location_click_on_page(df, parser):
    """
    Extracts the location of the click on the page for each article pair.
    
    Parameters:
    - df (DataFrame): DataFrame containing 'path' column.
    - parser (HTMLParser): HTMLParser object to extract the location of the click.
    
    Returns:
    - DataFrame: DataFrame with the location of the click for each article pair.
    """

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