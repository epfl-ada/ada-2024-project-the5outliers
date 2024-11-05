import pandas as pd
import numpy as np
from urllib.parse import unquote

def read_articles(file_path='./data/paths-and-graph/articles.tsv'):

    articles = pd.read_csv(file_path, comment='#', names=["article"])
    articles = articles["article"].apply(unquote).replace('_', ' ', regex=True)

    return articles

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