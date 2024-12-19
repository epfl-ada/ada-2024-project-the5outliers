import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import math

def build_network_from_paths(df_paths, df_categories, include_self_loops=True):
    """
    Build a directed network graph from paths with main categories and edge weights.
    """
    # Map each article to its main category
    article_to_main_category = dict(zip(df_categories['article'], df_categories['level_1']))

    # Initialize directed graph
    G = nx.DiGraph()

    # Build the network with main categories and aggregate edge weights
    for path in df_paths:
        nodes = path.split(';')
        # Map articles to their main categories
        main_category_nodes = [article_to_main_category.get(node, node) for node in nodes]

        for i in range(len(main_category_nodes) - 1):
            u = main_category_nodes[i]
            v = main_category_nodes[i + 1]
            if u != v or include_self_loops:
                # Add or update edge with weight
                if G.has_edge(u, v):
                    G[u][v]['weight'] += 1
                else:
                    G.add_edge(u, v, weight=1)

    return G

def build_network_from_matrix(matrix_df, article_to_category, include_self_loops=False):
    """
    Build a directed network graph from a transition matrix, with nodes mapped to main categories.

    Parameters:
    - matrix_df (DataFrame): A square DataFrame where rows = source, columns = target, values = weights.
    - article_to_category (dict): Mapping of articles to their main categories.
    - include_self_loops (bool): Whether to include self-loops (source == target).

    Returns:
    - G (networkx.DiGraph): A directed graph with edges and weights.
    """
    # Initialize directed graph
    G = nx.DiGraph()

    # Iterate over the transition matrix
    for source_article in matrix_df.index:
        for target_article in matrix_df.columns:
            weight = matrix_df.at[source_article, target_article]
            if weight > 0:  # Only consider non-zero transitions
                # Map source and target articles to their main categories
                source_category = article_to_category.get(source_article, source_article)
                target_category = article_to_category.get(target_article, target_article)

                # Handle self-loops based on the flag
                if source_category != target_category or include_self_loops:
                    # Add or update edge with weight
                    if G.has_edge(source_category, target_category):
                        G[source_category][target_category]['weight'] += weight
                    else:
                        G.add_edge(source_category, target_category, weight=weight)

    return G

def filter_network(G, weight_threshold=5, top_n=None):
    """
    Filter the network based on weight threshold and top connections.
    """
    # Apply weight threshold
    filtered_edges = [(u, v) for u, v, weight in G.edges(data='weight') if weight >= weight_threshold]
    H = G.edge_subgraph(filtered_edges).copy()

    # Limit each node to its top_n strongest connections, if specified
    if top_n is not None:
        for node in list(H.nodes):
            edges = sorted(H.edges(node, data=True), key=lambda x: x[2]['weight'], reverse=True)
            if len(edges) > top_n:
                edges_to_remove = edges[top_n:]
                H.remove_edges_from([(u, v) for u, v, _ in edges_to_remove])

    return H

def normalize_edge_weights(G, df_categories):
    """
    Normalize edge weights by the number of articles in the level_1 category.
    """
    # Count articles per category
    category_counts = df_categories['level_1'].value_counts().to_dict()

    # Create a copy of the graph for normalized weights
    G_normalized = G.copy()

    # Normalize edge weights
    for u, v, data in G.edges(data=True):
        source_count = category_counts.get(u, 1)  # Default to 1 if category count not available
        normalized_weight = data['weight'] / source_count
        G_normalized[u][v]['weight'] = normalized_weight

    return G_normalized

def analyze_edge_weights(G):
    """
    Analyze and plot the distribution of edge weights, and return stati.
    """
    edge_weights = [weight for u, v, weight in G.edges(data='weight')]

    # Plot the distribution of edge weights
    plt.figure(figsize=(8, 6))
    plt.hist(edge_weights, bins=50, color="skyblue", edgecolor="black")
    plt.xlabel("Edge Weight")
    plt.ylabel("Frequency")
    plt.title("Distribution of Edge Weights")
    plt.show()

    stats = {
        'Count': len(edge_weights),
        'Mean': np.mean(edge_weights),
        'Median': np.median(edge_weights),
        'Standard Deviation': np.std(edge_weights),
        'Min': np.min(edge_weights),
        'Max': np.max(edge_weights),
        '25th Percentile': np.percentile(edge_weights, 25),
        '75th Percentile': np.percentile(edge_weights, 75),
    }

    df_stats = pd.DataFrame.from_dict(stats, orient='index', columns=['Edge Weight Statistics'])
    return df_stats


def plot_network(H, df_categories, palette, title="Network Graph", show_edge_labels=True, node_abbreviations=None, save_path=None, background_color='white'):
    """
    Plot the directed network graph.

    Parameters:
        H (Graph): The directed graph to plot.
        df_categories (DataFrame): DataFrame with category information for nodes.
        palette (dict): Dictionary mapping nodes to colors.
        title (str): The title of the plot.
        show_edge_labels (bool): Whether to show edge weight labels.
        node_abbreviations (dict): Optional dictionary mapping nodes to their abbreviated labels in the plot.
        save_path (str, optional): Path to save the plot as an image file. If None, the plot is not saved.
        background_color (str): Background color of the plot ('white' or 'transparent').
    """
    # Determine the background color
    if background_color == 'transparent':
        facecolor = 'none'  # Transparent background
    else:
        facecolor = background_color

    # FIGURE -------------------------------------------------------------------------
    fig = plt.figure(figsize=(15, 10), dpi=400, facecolor=facecolor)
    
    # NODES positions: spring layout----------------------------------------------------
    pos = nx.spring_layout(H, seed=42, k=0.5, scale=3)

    # Node sizes: number of articles in the category
    size_range = [800, 2500]
    nb_articles_per_category = df_categories.groupby('level_1').agg('count')['article']
    max_nb_articles = max(nb_articles_per_category.get(node, 0) for node in H.nodes())
    
    for node in H.nodes():
        nb_articles_in_category = nb_articles_per_category.get(node, 0)
        H.nodes[node]['cat_size'] = nb_articles_in_category
        # Interpolate size of current node from range of category-sizes and range of node-sizes
        H.nodes[node]['size'] = np.interp(nb_articles_in_category, [0, max_nb_articles], size_range) 
    
    node_sizes = [H.nodes[node]['size'] for node in H.nodes()]
    node_sizes_dict = {node: H.nodes[node]['size'] for node in H.nodes()}

    # Node colors
    node_colors = [palette.get(node, '#808080') for node in H.nodes()]

    # Node labels: abbreviations or default full labels
    if node_abbreviations:
        abbrev_labels = {node: node_abbreviations.get(node, node) for node in H.nodes()}
    else:
        abbrev_labels = {node: node for node in H.nodes()}
    
    # Node label font sizes
    font_sizes = [np.interp(H.nodes[node]['size'], size_range, [8, 20]) for node in H.nodes()]

    # Plot nodes and node labels
    nx.draw_networkx_nodes(H, pos, node_size=node_sizes, node_color=node_colors, edgecolors="white", linewidths=1.5)
    for node, font_size in zip(H.nodes, font_sizes):
        nx.draw_networkx_labels(H, pos, labels={node: abbrev_labels[node]}, font_size=font_size, font_color='white')
    
    # EDGES attributes--------------------------------------------------------------------
    all_weights = [H[u][v]['weight'] for u, v in H.edges()]
    max_weight = max(all_weights) if all_weights else 1

    # Edges adjusted width and margins: where arrows land 
    edge_labels = {}
    for u, v in H.edges():
        # Edges width: count of this transition
        weight = H[u][v]['weight']
        edge_width = min(weight/max_weight*3, 5)
        nx.draw_networkx_edges(
            H, pos,
            edgelist=[(u, v)],
            width=edge_width,
            arrowstyle='-|>',
            arrowsize=15,
            edge_color='#3b3b3b',
            connectionstyle=f'arc3,rad=0.1' if H.has_edge(v, u) else 'arc3,rad=0.0',
            # Edges length: stops at border of target node v
            min_source_margin=math.sqrt(node_sizes_dict[u] / math.pi),
            min_target_margin=math.sqrt(node_sizes_dict[v] / math.pi)
        )
        edge_labels[(u, v)] = f"{weight:.2f}"

    # Draw edge labels if enabled
    if show_edge_labels:
        nx.draw_networkx_edge_labels(
            H, pos,
            edge_labels=edge_labels,
            font_size=8,
            label_pos=0.5,
            bbox=dict(alpha=0),
            verticalalignment='center'
        )
    
    # LEGEND BOX----------------------------------------------------------------------------
    palette_without_others = {k: v for k, v in palette.items() if k != 'Others'}

    if node_abbreviations:
        legend_entries = {node: f"{node} ({node_abbreviations.get(node, node)})" for node in H.nodes}
    else:
        legend_entries = {node: node for node in H.nodes()}
        
    # Handles and labels for the legend
    handles = [plt.Line2D([0], [0], marker='o', color=palette_without_others[node], linestyle='', markersize=10) for node in legend_entries.keys()]
    labels = list(legend_entries.values())

    # Add the legend to the plot
    plt.legend(
        handles,
        labels,
        loc='best',
        title="Categories",
        fontsize=10,        
        title_fontsize=12  
    )

    # PLOT figure----------------------------------------------------------------------------
    plt.title(title, fontsize=23, color='#3b3b3b')
    plt.axis("off")
    plt.tight_layout()

    # Save the plot if save_path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=1000, facecolor=facecolor)

    plt.show()