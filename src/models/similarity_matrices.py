from FlagEmbedding import BGEM3FlagModel
import numpy as np
import networkx as nx
import os
import pandas as pd
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
import torch

def compute_save_all():
    '''Call this function from the base of the repo. Computes article embeddings, the two similarity matrices and the articles parsed and saves them to data.'''
    from src.data.data_loader import read_articles , read_categories
    from src.utils.HTMLParser import HTMLParser

    df_article_names = read_articles()

    flag_embeddings = BGEM3_embedding(df_article_names)
    flag_similarity_matrix = compute_embedding_similarity_matrix(flag_embeddings)
    bert_embeddings = bert_embedding(df_article_names)
    bert_similarity_matrix = compute_embedding_similarity_matrix(bert_embeddings)

    df_categories = read_categories()
    df_scat = category_jaccard_similarity(df_categories,{'level_1': 1, 'level_2': 2, 'level_3': 3}) 
    df_scat = df_scat.loc[df_article_names, df_article_names]
    df_scat = df_scat.astype(np.float32)

    parser = HTMLParser()
    parser.parse_save_valid(df_article_names) # saves to pickle file

    np.save('./data/paths-and-graph/embedded_articles_BGEM3.npy', flag_embeddings)
    np.save('./data/paths-and-graph/similarity_matrix_BGEM3.npy', flag_similarity_matrix)
    np.save('./data/paths-and-graph/embedded_articles_bert.npy', bert_embeddings)
    np.save('./data/paths-and-graph/similarity_matrix_bert.npy', bert_similarity_matrix)
    np.save('./data/paths-and-graph/category_jaccard_similarity.npy', df_scat)

def bert_embedding(df_article_names):
    '''Generate embeddings for article names using Bert'''

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    embeddings = []
    for art in tqdm(df_article_names):
        # Tokenize the word (add special tokens for BERT)
        input_ids = tokenizer(art, return_tensors='pt')['input_ids']

        # Pass through the BERT model
        with torch.no_grad():
            outputs = model(input_ids)

        # Extract the embeddings for the [CLS] token (index 0)
        word_embedding = outputs.last_hidden_state[0][0].numpy()
        embeddings.append(word_embedding)

    embeddings = np.array(embeddings)
    embeddings /= np.linalg.norm(embeddings, axis=1)[:, np.newaxis]

    return embeddings

def BGEM3_embedding(df_article_names):
    '''BGEM3 embedding from article names: run in colab for speed'''

    emb_model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=False) # using fp32 for better precision
    embeddings = emb_model.encode(df_article_names.tolist())['dense_vecs']

    return embeddings

def compute_embedding_similarity_matrix(embeddings):
    '''Computes the cosine similarity of embeddings. Embeddings are already normalised to 1, so the similarity is the dot product'''
    
    return np.matmul(embeddings, embeddings.T)

def get_indices(df_articles, article_names):
    '''
    Returns the indices of the provided article names in the dataframe of articles. 
    If an article does not exist, the value returned is -1 for that article.

    Example of use: indices = sm.get_indices(df_articles, ['cat', 'Dog', 'Cat'])
    '''

    article_indices = np.zeros_like(article_names, dtype=np.int32) - 1
    for idx, article in enumerate(article_names):
        
        article_index = df_articles[df_articles == article].index
        if len(article_index)==1: # checking if the article exists and is unique
            article_indices[idx] = article_index.item()

    return article_indices

def similarity_mean_std(similarity_matrix):
    '''Compute the mean similarity and standard deviation for all the articles'''

    no_diag = similarity_matrix[~np.eye(similarity_matrix.shape[0],dtype=bool)].reshape(similarity_matrix.shape[0],-1)
    print(f'Removed the diagonal to get a matrix of shape {no_diag.shape}')
    mean_similarity = np.mean(no_diag, axis=1)
    similarity_std = np.std(no_diag, axis=1)
    print(f'Mean similarity and std per article have shape {mean_similarity.shape}')

    return mean_similarity, similarity_std

def central_words(similarity_matrix):
    '''
    Compute how important a node (an article) in the graph is based on the similarity with other nodes (other articles).
    Uses both eigenvector centrality and PageRank and returns the sorted values of nodes and centrality measure.
    '''

    # Convert similarity matrix to a graph
    G = nx.from_numpy_array(similarity_matrix)
    print(f'The graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.')

    # Eigenvector Centrality
    eigenvector_centrality = nx.eigenvector_centrality(G)
    eigenvector_centrality_sorted = sorted(eigenvector_centrality.items(), key=lambda x: x[1], reverse=True)

    # PageRank
    pagerank_scores = nx.pagerank(G)
    pagerank_sorted = sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)

    return eigenvector_centrality_sorted, pagerank_sorted

def category_jaccard_similarity(categories, level_weights):
    """
    Calculates the Weighted Jaccard Similarity between articles based on category levels and weights.

    Parameters:
    - categories (DataFrame): Contains columns 'article', 'category', 'level_1', 'level_2', 'level_3' for each article's category and level information.
    - level_weights (dict): A dictionary with weights for each level, e.g., {'level_1': 3, 'level_2': 2, 'level_3': 1}.

    Returns:
    - DataFrame: Weighted Jaccard Similarity matrix with articles as both rows and columns.
    """

    # Assign weights to each article's category based on its levels
    categories['weight'] = (
        categories['level_1'].apply(lambda x: level_weights['level_1'] if pd.notnull(x) else 0) +
        categories['level_2'].apply(lambda x: level_weights['level_2'] if pd.notnull(x) else 0) +
        categories['level_3'].apply(lambda x: level_weights['level_3'] if pd.notnull(x) else 0)
    )

    # Create pivot table with articles and categories
    category_pivot = categories.pivot_table(
        index='article',
        columns='category',
        values='weight',
        aggfunc='max',
        fill_value=0
    )

    # Convert pivot table to a NumPy array
    A = category_pivot.values  # Shape: (n_articles, n_categories)
    n = A.shape[0]  # Number of articles

    # Initialize similarity matrix
    similarity_weighted_jaccard = np.zeros((n, n))

    # Compute Weighted Jaccard Similarity for each article pair
    for i in tqdm(range(n), desc="Computing Weighted Jaccard Similarity"):
        # Compute min and max with all other articles
        min_vals = np.minimum(A[i], A)
        max_vals = np.maximum(A[i], A)

        # Sum over categories to get intersection and union
        intersection = min_vals.sum(axis=1)
        union = max_vals.sum(axis=1)

        # Compute similarity, handling division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            similarity = np.divide(intersection, union, out=np.zeros_like(intersection, dtype=float), where=union != 0)

        similarity_weighted_jaccard[i] = similarity

    # Convert the similarity matrix to a DataFrame with article indices
    similarity_weighted_jaccard_df = pd.DataFrame(
        similarity_weighted_jaccard,
        index=category_pivot.index,
        columns=category_pivot.index
    )

    return similarity_weighted_jaccard_df  