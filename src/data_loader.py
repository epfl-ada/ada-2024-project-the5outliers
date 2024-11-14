import pandas as pd
import numpy as np
from urllib.parse import unquote


def get_bad_articles():
    '''Returns the name and index of the articles in articles.tsv that are not wikipedia articles or that have missing propreties (e.g. missing categories) and should be removed.'''

    filepath='./data/paths-and-graph/articles.tsv'
    articles = pd.read_csv(filepath, comment='#', names=["article"])
    articles = articles["article"].apply(unquote).replace('_', ' ', regex=True)

    # The bad articles are not real wikipedia pages, cannot be parsed by the html parser and are do not appear in user paths
    bad_articles = ['Directdebit', 'Donation', 'Friend Directdebit', 'Sponsorship Directdebit', 'Wowpurchase', 'Pikachu','Wikipedia Text of the GNU Free Documentation License']
    bad_articles_idx = articles[articles.isin(bad_articles)].index

    return bad_articles, bad_articles_idx.to_list()

def read_articles():
    '''
    Return a Series with the article names, with '_' removed and the percent encoding unquoted.
    Removes the articles in the list that are not wikipedia articles.
    '''

    filepath='./data/paths-and-graph/articles.tsv'
    articles = pd.read_csv(filepath, comment='#', names=["article"])
    articles = articles["article"].apply(unquote).replace('_', ' ', regex=True)
    
    # Remove invalid articles
    bad_articles, _ = get_bad_articles()
    articles = articles[~articles.isin(bad_articles)].reset_index(drop=True)

    return articles

def read_categories():

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
    '''The rows are the source articles and the columns are the destination articles'''
    
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

    filepath='./data/paths-and-graph/paths_unfinished.tsv'
    column_names = ['hashedIpAddress', 'timestamp', 'durationInSec', 'path', 'target', 'type']
    df = pd.read_csv(filepath, sep='\t', comment='#', names=column_names)

    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df['path'] = df['path'].apply(unquote).replace('_', ' ', regex=True)
    df['target'] = df['target'].apply(unquote).replace('_', ' ', regex=True)

    print("Unfinished Paths \nNumber of rows before filtering:", len(df))
    # Drop invalid articles
    valid_articles = set(read_articles())
    valid_articles.add("<")

    invalid_target_articles = set()
    for target in df['target']:
        if target not in valid_articles:
            invalid_target_articles.add(target)
    print("Invalid target articles found:", invalid_target_articles)
    # Drop rows with invalid target articles
    df = df.loc[df['target'].isin(valid_articles)].reset_index(drop=True)
    
    invalid_articles_set = set()
    # Filter and find invalid articles
    def check_path(path):
        articles = path.split(';')
        for article in articles:
            article = article.strip()
            if article not in valid_articles:
                invalid_articles_set.add(article)  # Add only the invalid article
                return False  # Exclude this row if any invalid article is found
        return True  # Include this row if all articles are valid

    # Apply the filter with the custom function
    df = df[df["path"].apply(check_path)].reset_index(drop=True)

    # Print unique invalid articles
    print("Invalid articles found in path:", invalid_articles_set)
    print("Number of rows after filtering:", len(df),"\n")

    return df

def read_finished_paths():

    filepath='./data/paths-and-graph/paths_finished.tsv'
    column_names = ['hashedIpAddress', 'timestamp', 'durationInSec', 'path', 'rating']
    df = pd.read_csv(filepath, sep='\t', comment='#', names=column_names)

    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df['path'] = df['path'].apply(unquote).replace('_', ' ', regex=True)

    print("Finished Paths \nNumber of rows before filtering:", len(df))
    # Drop invalid articles
    valid_articles = set(read_articles())
    valid_articles.add("<")
    
    # Set to store unique invalid articles
    invalid_articles_set = set()

    # Filter and find invalid articles
    def check_path(path):
        articles = path.split(';')
        for article in articles:
            article = article.strip()
            if article not in valid_articles:
                invalid_articles_set.add(article)  # Add only the invalid article
                return False  # Exclude this row if any invalid article is found
        return True  # Include this row if all articles are valid

    # Apply the filter with the custom function
    df = df[df["path"].apply(check_path)].reset_index(drop=True)

    # Print unique invalid articles
    print("Invalid articles found in path:", invalid_articles_set)
    print("Number of rows after filtering:", len(df),"\n")
    
    return df

def read_similartiy_matrix():
    filepath='./data/paths-and-graph/similarity_matrix.npy'
    article_names = read_articles()
    sm = np.load(filepath)
    df_sm = pd.DataFrame(sm)
    df_sm.columns = article_names
    df_sm.index = article_names

    return df_sm

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
    '''Replaces back clicks < with the article that was landed on'''
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
    
    return resolved_path