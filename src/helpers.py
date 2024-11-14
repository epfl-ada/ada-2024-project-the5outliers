import pandas as pd
import plotly.express as px


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


def analyze_categories_paths(df_paths, df_categories, omit_loops=False):
    """
    Analyze and summarize common category paths from article paths.

    Parameters:
        df_paths (pd.DataFrame): DataFrame containing article paths with a 'path' column. Each entry in 'path' represents a 
            sequence of articles separated by semicolons.
        df_categories (pd.DataFrame): DataFrame mapping articles to main categories, with 'article' and 'level_1' columns. 
            'article' contains article names, and 'level_1' holds the primary category each article belongs to.
        omit_loops (bool): Optional; if True, removes consecutive repetitions of the same category within a path. Defaults to False.

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
        articles = path.split(';')
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


def get_position_frequencies(df, max_position=5, normalize=False):
    """
    Calculate frequencies for each category at each position in the path up to `max_position`.
    Optionally normalize frequencies.
    
    Parameters:
        df (DataFrame): The DataFrame containing paths.
        max_position (int): The maximum position in the path to analyze.
        normalize (bool): Whether to normalize frequencies to percentages.
    
    Returns:
        DataFrame: Frequencies (normalized if specified) of each category across the specified range of positions.
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
        
        # Normalize if specified
        if normalize:
            position_counts = (position_counts / position_counts.sum()) * 100  # Normalize to percentage
            
        position_counts = position_counts.reset_index(name='Frequency' if not normalize else 'Normalized Frequency')
        position_counts['Position'] = pos + 1  # Record the position number
        position_counts.rename(columns={position_column: 'Category'}, inplace=True)
        
        # Append to overall position data
        position_data.append(position_counts)
    
    # Concatenate all position data into a single DataFrame
    position_data_df = pd.concat(position_data, ignore_index=True)

    return position_data_df

def plot_position_interactive(df_position, plot_type="line", normalized=False):
    """
    Plot an interactive plot of category frequencies across positions with hue as categories.
    Supports both line and bar plots, with optional normalization and stacking.
    
    Parameters:
        df_position (DataFrame): DataFrame with position frequencies for each category.
        plot_type (str): Type of plot to generate ("line" or "bar").
        normalized (bool): If True, indicates that frequencies are normalized.
    """
    # Select y-axis label based on normalization
    y_col = "Normalized Frequency" if normalized else "Frequency"
    title = f"{'Normalized ' if normalized else ''}Category Frequencies Across Path Positions"
    
    # Choose plot type
    if plot_type == "line":
        fig = px.line(
            df_position, 
            x="Position", 
            y=y_col, 
            color="Category", 
            markers=True,
            title=title,
            labels={"Position": "Position in Path", y_col: "Percentage (%)" if normalized else "Frequency"}
        )
    elif plot_type == "bar":
        fig = px.bar(
            df_position, 
            x="Position", 
            y=y_col, 
            color="Category", 
            title=title,
            labels={"Position": "Position in Path", y_col: "Percentage (%)" if normalized else "Frequency"},
            barmode="stack"
        )
    else:
        raise ValueError("Invalid plot_type. Choose 'line' or 'bar'.")
    
    # Update layout for better readability
    fig.update_layout(
        legend_title_text="Category",
        xaxis=dict(tickmode="linear", tick0=1, dtick=1),  # Ensure integer x-axis ticks
        template="plotly_white",
        width=900,
        height=600
    )
    
    # Show the interactive plot
    fig.show()
    
def find_categories_start_end(paths, categories) :
    """
    Adds 2 columns for the start and end articles
    and  2 columns for the start and end categories 

    Parameters:
        paths (DataFrame): containing columns 'paths' and 'target'
        categories (Dataframe): containing columns 'article' and 'category'

    Returns:
        DataFrame: with first and last article and category
    """
    paths["start"] = paths["path"].apply(lambda x: x.split(";")[0])
    if "target" in paths.columns:
        paths["end"] = paths["target"]
    else :
        paths["end"] = paths["path"].apply(lambda x: x.split(";")[-1])
        
    paths["start_category"] = paths["start"].apply(lambda x: categories[categories["article"] == x]["category"].values[0].split(".")[1] if x in categories["article"].values else None)
    paths["end_category"] = paths["end"].apply(lambda x: categories[categories["article"] == x]["category"].values[0].split(".")[1] if x in categories["article"].values else None)

    return paths

def voyage_sorting(paths, categories):
    """
    Filters paths into voyage or not voyage 
    Parameters:
        paths (DataFrame): DataFrame with a 'path' column
        categories (DataFrame): DataFrame with 'article' and 'category' columns
    Returns:
        DataFrame: paths with an additional 'voyage' column marked as True or False
    """    
    voyage_articles = categories[categories['category'].str.contains('Geography|Countries', regex=True)]['article'].unique()
    paths['voyage'] = paths['path'].apply(lambda p: any(article in p.split(';') for article in voyage_articles))
    # Display the filtered paths
    return paths

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

def voyage_sorting_category_path(category_paths, finished, n=3):
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
