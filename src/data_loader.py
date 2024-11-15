import pandas as pd
import numpy as np
from urllib.parse import unquote


def get_bad_articles():
    '''
    Returns the name and index of the articles in articles.tsv that are not wikipedia articles or 
    that have missing propreties (e.g. missing categories, wrong shortest path) and should be removed.
    '''

    filepath='./data/paths-and-graph/articles.tsv'
    articles = pd.read_csv(filepath, comment='#', names=["article"])
    articles = articles["article"].apply(unquote).replace('_', ' ', regex=True)

    # The bad articles are not real wikipedia pages, cannot be parsed by the html parser or do not appear in user paths
    bad_articles = ['Directdebit', 'Donation', 'Friend Directdebit', 'Sponsorship Directdebit', 'Wowpurchase', 'Pikachu','Wikipedia Text of the GNU Free Documentation License']
    bad_articles_idx = articles[articles.isin(bad_articles)].index

    return bad_articles, bad_articles_idx.to_list()

def read_articles():
    '''
    Return a Series with the article names, with '_' removed and the percent encoding unquoted.
    Removes the articles in the list that are not wikipedia articles (all articles in get_bad_articles).
    '''

    filepath='./data/paths-and-graph/articles.tsv'
    articles = pd.read_csv(filepath, comment='#', names=["article"])
    articles = articles["article"].apply(unquote).replace('_', ' ', regex=True)
    
    # Remove invalid articles
    bad_articles, _ = get_bad_articles()
    articles = articles[~articles.isin(bad_articles)].reset_index(drop=True)

    return articles

def read_categories():
    '''
    Loads the dataframe with the category information of articles. Removes all invalid articles.
    Split categories into main and sub-categories into different columns
    '''

    # Step 1: Load the data
    filepath='./data/paths-and-graph/categories.tsv'
    categories = pd.read_csv(filepath, sep='\t', comment='#', names=["article", "category"])
    categories["article"] = categories["article"].apply(unquote).replace('_', ' ', regex=True)
    categories["category"] = categories["category"].apply(unquote).replace('_', ' ', regex=True)

    # Remove invalid articles
    categories = categories.loc[categories['article'].isin(read_articles()), :].reset_index(drop=True)

    # Step 2: Separate categories by hierarchical levels
    # Find the maximum depth by checking the highest number of splits in any category
    max_depth = categories['category'].str.split('.').map(len).max()

    # Dynamically generate column names based on the max depth
    category_levels = categories['category'].str.split('.', expand=True)
    category_levels.columns = [f'level_{i+1}' for i in range(max_depth)]

    # Concatenate the levels with the original DataFrame
    df_expanded = pd.concat([categories, category_levels], axis=1)

    # Check if level_1 has only one unique value and adjust accordingly, 
    # by removing the column and renaming the rest
    level_1_values = df_expanded['level_1'].unique()
    if len(level_1_values) == 1:
        df_expanded.drop(columns='level_1', inplace=True)
        df_expanded.columns = ['article', 'category'] + [f'level_{i}' for i in range(1, max_depth)]

    return df_expanded

def read_links():
    '''Finds all the existing links between articles, removes invalid articles'''

    filepath='./data/paths-and-graph/links.tsv'
    links = pd.read_csv(filepath, sep='\t', comment='#', names=["linkSource", "linkTarget"])
    links["linkSource"] = links["linkSource"].apply(unquote).replace('_', ' ', regex=True)
    links["linkTarget"] = links["linkTarget"].apply(unquote).replace('_', ' ', regex=True)
    
    # Remove invalid articles
    links = links.loc[links['linkSource'].isin(read_articles()), :].reset_index(drop=True)
    links = links.loc[links['linkTarget'].isin(read_articles()), :].reset_index(drop=True)

    return links

def read_shortest_path_matrix():
    '''
    Read the shortest path lenght matrix (in the range 1-9, or -1 if there is no path)
    The rows are the source articles and the columns are the destination articles
    '''
    
    filepath='./data/paths-and-graph/shortest-path-distance-matrix.txt'
    with open(filepath, 'r') as file:
        lines = file.readlines()

    # Process each line to convert it into a list of distances
    data = []
    for line in lines:
        stripped_line = line.strip()
        if not stripped_line or stripped_line.startswith('#'):
            continue  # Skip comment lines and empty lines
        distances = [int(char) if char != '_' else -1 for char in line.strip()]
        data.append(distances)

    matrix = pd.DataFrame(data, dtype=int)

    # Drop bad articles
    _, bad_articles_idx = get_bad_articles()
    matrix.drop(index=bad_articles_idx, inplace=True)
    matrix.drop(columns=bad_articles_idx, inplace=True)

    # Read the articles.tsv file to use as column headers & index
    names_articles = read_articles()
    matrix.columns = names_articles
    matrix.index = names_articles  

    return matrix

def read_unfinished_paths():
    '''
    Load the dataset of unfinished game paths. Removes all paths with invalid articles in the path or as the target.
    Removes 'Timeout' paths that are shorter than 30 minutes (as per Timeout definition)
    '''

    filepath='./data/paths-and-graph/paths_unfinished.tsv'
    column_names = ['hashedIpAddress', 'timestamp', 'durationInSec', 'path', 'target', 'type']
    df = pd.read_csv(filepath, sep='\t', comment='#', names=column_names)

    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df['path'] = df['path'].apply(unquote).replace('_', ' ', regex=True)
    df['target'] = df['target'].apply(unquote).replace('_', ' ', regex=True)

    # Drop invalid articles
    print("Unfinished Paths\n---------------- \nNumber of rows before filtering:", len(df))
    valid_articles = set(read_articles())
    valid_articles.add("<")

    # Drop rows with invalid target articles
    invalid_target_articles = set()
    for target in df['target']:
        if target not in valid_articles:
            invalid_target_articles.add(target)
    print("Invalid target articles found:", invalid_target_articles)
    df = df.loc[df['target'].isin(valid_articles)].reset_index(drop=True)

    # Drop invalid articles in the path
    invalid_articles_set = set()
    def check_path(path):
        articles = path.split(';')
        for article in articles:
            article = article.strip()
            if article not in valid_articles:
                invalid_articles_set.add(article)  
                return False  # Exclude this row if any article is invalid
        return True  # Include this row if all articles are valid
    df = df[df["path"].apply(check_path)].reset_index(drop=True)
    print("Invalid articles found in path:", invalid_articles_set)

    len_df = len(df)
    # Drop rows where 'type' is 'timeout' and 'durationInSec' is less than 1800
    df = df[~((df['type'] == 'timeout') & (df['durationInSec'] < 1800))].reset_index(drop=True)
    print("Number of 'timeout' games with a duration of less than 30 minutes:", len_df - len(df))

    print("Number of rows after filtering:", len(df),"\n")

    return df

def read_finished_paths():
    '''
    Load the dataset of finished game paths. Removes all paths with invalid articles in the path.
    Removes paths with duration 0 (same start and target).
    '''

    filepath = './data/paths-and-graph/paths_finished.tsv'
    column_names = ['hashedIpAddress', 'timestamp', 'durationInSec', 'path', 'rating']
    df = pd.read_csv(filepath, sep='\t', comment='#', names=column_names)

    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df['path'] = df['path'].apply(unquote).replace('_', ' ', regex=True)

    # Drop invalid articles
    print(f"Finished Paths\n-------------- \nNumber of rows before filtering: {len(df)}")
    valid_articles = set(read_articles())
    valid_articles.add("<")

    invalid_articles_set = set()

    def check_path(path):
        articles = path.split(';')
        for article in articles:
            article = article.strip()
            if article not in valid_articles:
                invalid_articles_set.add(article)  
                return False  # Exclude this row if any article is invalid
        return True  # Include this row if all articles are valid
    
    df = df[df["path"].apply(check_path)].reset_index(drop=True)
    df = df[~((df['durationInSec'] == 0) & (df['rating'].isnull()))]
    df.reset_index(drop=True, inplace=True)
    print(f"Invalid articles found in path: {invalid_articles_set}")
    print(f"Number of rows after filtering: {len(df)}")
    
    return df

def read_similartiy_matrix():
    '''
    Read the semantic similarity matrix, assuming it was computed only for valid articles.
    Adds the article names as indices for the matrix
    '''

    filepath='./data/paths-and-graph/similarity_matrix.npy'
    article_names = read_articles()
    sm = np.load(filepath)
    df_sm = pd.DataFrame(sm)
    df_sm.columns = article_names
    df_sm.index = article_names

    return df_sm

def read_categories_matrix():
    '''
    Read the semantic category similarity matrix (Jaccard similarity), assuming it was computed only for valid articles.
    Adds the article names as indices for the matrix

    '''

    filepath='./data/paths-and-graph/category_jaccard_similarity.npy'
    articles_names = read_articles()
    df_cm = pd.DataFrame(np.load(filepath, allow_pickle=True))
    df_cm.columns = articles_names
    df_cm.index = articles_names
    return df_cm

def find_shortest_distance(row, distance_matrix):
    '''
    Finds the start and the target of a path and returns the shortest distance between the two.
    Distance can be anything: shortest path, semantic cosine similarity, ...
    '''
    
    articles = row['path'].split(';')
    if 'target' in row.index:
        return distance_matrix.loc[articles[0]][row['target']]
    return distance_matrix.loc[articles[0]][articles[-1]]

def replace_back_clicks(path):
    '''
    Replaces back clicks < with the article that was landed on.
    '''
    articles = path.split(';')
    resolved_path = []
    consecutive_backclicks = 0
    for i, art in enumerate(articles):
        if art == '<':
            resolved_path.append(resolved_path[i-2-consecutive_backclicks])
            consecutive_backclicks += 2
        else:
            consecutive_backclicks=0
            resolved_path.append(art)
    
    # Join the resolved path with ';' as a separator
    return ';'.join(resolved_path)