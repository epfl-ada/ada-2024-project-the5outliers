import pandas as pd
import numpy as np
from urllib.parse import unquote

def read_articles(file_path='./data/paths-and-graph/articles.tsv'):

    articles = pd.read_csv(file_path, comment='#', names=["article"])
    articles = articles["article"].apply(unquote).replace('_', ' ', regex=True)

    return articles

def read_categories(file_path='./data/paths-and-graph/categories.tsv'):

    
    # Step 1: Load the data
    categories = pd.read_csv(file_path, sep='\t', comment='#', names=["article", "category"])
    categories["article"] = categories["article"].apply(unquote).replace('_', ' ', regex=True)

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

def read_links(file_path='./data/paths-and-graph/links.tsv'):

    links = pd.read_csv(file_path, sep='\t', comment='#', names=["linkSource", "linkTarget"])
    links["linkSource"] = links["linkSource"].apply(unquote).replace('_', ' ', regex=True)
    links["linkTarget"] = links["linkTarget"].apply(unquote).replace('_', ' ', regex=True)

    return links

def read_matrix(file_path='./data/paths-and-graph/shortest-path-distance-matrix.txt'):

    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Process each line to convert it into a list of distances
    data = []
    for line in lines:
        stripped_line = line.strip()
        if not stripped_line or stripped_line.startswith('#'):
            continue  # Skip comment lines and empty lines
        distances = [int(char) if char != '_' else np.nan for char in line.strip()]
        data.append(distances)

    matrix = pd.DataFrame(data)

    # Read the articles.tsv file to use as column headers & index
    names_articles = read_articles()
    matrix.columns = names_articles
    matrix.index = names_articles    
    print("The rows are the source articles and the columns are the destination articles")

    return matrix

def read_unfinished_paths(file_path='./data/paths-and-graph/paths_unfinished.tsv'):

    column_names = ['hashedIpAddress', 'timestamp', 'durationInSec', 'path', 'target', "type"]
    df = pd.read_csv(file_path, sep='\t', comment='#', names=column_names)

    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

    return df

def read_finished_paths(file_path='./data/paths-and-graph/paths_finished.tsv'):

    column_names = ['hashedIpAddress', 'timestamp', 'durationInSec', 'path', 'rating']
    df = pd.read_csv(file_path, sep='\t', comment='#', names=column_names)

    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

    return df