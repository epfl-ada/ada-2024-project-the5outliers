import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
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

def analyze_categories_paths(df_paths, df_categories, users=True, omit_loops=False):
    """
    Analyze and summarize common category paths from article paths.

    Parameters:
        df_paths (pd.DataFrame): DataFrame containing article paths with a 'path' column. 
        df_categories (pd.DataFrame): DataFrame mapping articles to main categories, with 'article' and 'level_1' columns. 
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


def find_shortest_path(row, G):
    """
    Finds the shortest path between the source and target nodes in a graph.

    Args:
        row (pd.Series): A row containing 'source' and 'target' nodes.
        G (networkx.DiGraph): A directed graph built using NetworkX.

    Returns:
        list or None: The shortest path as a list of nodes if it exists; otherwise None.
    """
    source, target = row['source'], row['target']
    try:
        path = nx.shortest_path(G, source=source, target=target)
    except nx.NetworkXNoPath:
        path = None  # If no path exists
    return path


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

def plot_position_line(df_position, title="Category transitions frequencies across Path Positions"):
    """
    Plot an interactive line plot of category frequencies across positions with both normalized and non-normalized views.
    
    Parameters:
        df_position (DataFrame): DataFrame with position frequencies for each category.
    """
    # Prepare data for normalized frequencies
    df_position_norm = df_position.copy()
    df_position_norm['Normalized Frequency'] = df_position_norm.groupby('Position')['Frequency'].transform(lambda x: (x / x.sum()) * 100)
    
    # Define a color map for categories
    unique_categories = df_position['Category'].unique()
    colors = px.colors.qualitative.Plotly  # Choose a color scheme
    color_map = {category: colors[i % len(colors)] for i, category in enumerate(unique_categories)}
    
    # Create subplots with separate y-axes
    fig = make_subplots(
        rows=1, cols=2, subplot_titles=("Non-Normalized Frequencies", "Normalized Frequencies"),
        horizontal_spacing=0.05
    )
    
    # Add non-normalized line plot traces
    for category in unique_categories:
        category_data = df_position[df_position['Category'] == category]
        fig.add_trace(
            go.Scatter(
                x=category_data['Position'], 
                y=category_data['Frequency'],
                mode="lines+markers",
                name=category,
                line=dict(color=color_map[category])
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
                line=dict(color=color_map[category]),
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

def check_voyage_status(category_path, finished, n):
    """
    Check is the category path is voyage or not voyage, that is wether the first n categories visited are 'Geography' or 'Countries' 

    Parameters:
        category_paths (DataFrame): DataFrame with 'Category Path' column
        finished (Boolean): whether it is a finished or not-finished path 
        n (int): number of different categories following the first to consider 

    Returns:
        Boolean : is the path of category a voyage 
    """    
    # Split the category path
    categories = category_path.split(' -> ')
    path_len = len(categories)
    
    if finished : 
        # Case 1: Path with 1 category -> always False
        if path_len <= 2:
            return False
        # Case 2: Path length between 3 and n+2 -> check middle categories
        elif 2 < path_len <= n + 2:
            return any(category in categories[1:-1] for category in ['Geography', 'Countries'])
        # Case 3: Path longer than n+2 -> check the first n categories after the first
        else:
            return any(category in categories[1:n+1] for category in ['Geography', 'Countries'])
        
    else : 
        # Case 1: Path with 1 or 2 categories -> always False
        if path_len <= 1:
            return False
        # Case 2: Path length between 3 and n+2 -> check middle categories
        elif 1 < path_len <= n + 1:
            return any(category in categories[1:] for category in ['Geography', 'Countries'])
        # Case 3: Path longer than n+2 -> check the first n categories after the first
        else:
            return any(category in categories[1:n+1] for category in ['Geography', 'Countries'])    

def category_voyage_sorting(category_paths, finished, n=3):
    """
    Adds a boolean column filtering paths into voyage or not voyage, that is whether the first n categories visited are 'Geography' or 'Countries' 

    Parameters:
        category_paths (DataFrame): DataFrame with 'Category Path' column
        finished (Boolean): whetehr the paths are finished 
        n (int): number of different categories following the first to consider 

    Returns:
        DataFrame: category paths with an additional 'voyage' column marked as True or False
    """    
    category_paths['voyage'] = category_paths['Category Path'].apply(lambda p: check_voyage_status(p, finished, n)) 
    return category_paths 

def game_voyage_sorting(df_article_paths, df_categories, finished, n=3):
    """
    Adds a boolean 'voyage' column to each game.
    First maps articles to categories, then checks if the category path qualifies as a 'voyage'.

    Parameters:
        df_article_path (DataFrame): DataFrame with 'path' column containing article paths
        df_categories (DataFrame): DataFrame with 'article' and 'level_1' columns for mapping
        n (int): Number of different categories following the first to consider for voyage condition
        finished (bool): Indicates whether the paths are finished 

    Returns:
        DataFrame: Original DataFrame with an additional 'voyage' column (True/False).
    """
    # Map articles to their main categories
    article_to_category = dict(zip(df_categories['article'], df_categories['level_1']))
    
    # Convert article path to category path, omitting consecutive duplicates (loops)
    def get_category_path(path):
        articles = path.split(';')
        categories = [article_to_category.get(article, article) for article in articles]
        categories_no_loops = [cat for i, cat in enumerate(categories) if i == 0 or cat != categories[i - 1]]
        return ' -> '.join(categories_no_loops)
    
    # Apply the transformation and check voyage status
    df_article_paths['Category Path'] = df_article_paths['path'].apply(get_category_path)
    df_article_paths['voyage'] = df_article_paths['Category Path'].apply(lambda p: check_voyage_status(p, finished, n))
    
    return df_article_paths

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
    plt.xticks(rotation=0, ha="right", fontsize=10)
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

def plot_articles_pie_chart(df, abbreviations=None):
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