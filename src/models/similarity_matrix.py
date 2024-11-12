from FlagEmbedding import BGEM3FlagModel
from gensim.models import KeyedVectors
import numpy as np
import networkx as nx
import os


def gensim_embedding(df):
    '''Gensim word2vec from article names: does not work for all articles because takes precise keys as input'''

    # Load pretrained model
    word2vec = KeyedVectors.load_word2vec_format(r'./data/GoogleNews-vectors-negative300.bin', binary=True)

    article_embeddings = np.zeros((len(df), 3000))

    for n in range(len(df)):
        article_embeddings[n] = word2vec(df[n])

    return article_embeddings

def BGEM3_embedding(df):
    '''BGEM3 embedding from article names: run in colab for speed'''

    emb_model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=False) # using fp32 for better precision
    embeddings = emb_model.encode(df.tolist())['dense_vecs']

    return embeddings

def compute_similarity_matrix(embeddings):
    '''Computes the cosine similarity of embeddings. Embeddings are already normalised to 1, so the similarity is the dot product'''
    
    return np.matmul(embeddings, embeddings.T)

def load_embedding_similarity_matrix():
    '''Assumes the embedding and similarity matrix exist and are in ./data/paths-and-graph starting from the base of the repo'''

    data_path = './data/paths-and-graph'
    embeddings = np.load(os.path.join(data_path, 'embedded_articles.npy'))
    similarity_matrix = np.load(os.path.join(data_path, 'similarity_matrix.npy'))

    return embeddings, similarity_matrix

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