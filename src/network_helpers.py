import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def build_network(df_paths, df_categories, include_self_loops=True):
    """
    Build a directed network graph from paths with main categories and edge weights.
    """
    # Map each article to its main category
    article_to_main_category = dict(zip(df_categories['article'], df_categories['level_1']))

    # Initialize directed graph
    G = nx.DiGraph()

    # Build the network with main categories and aggregate edge weights
    for path in df_paths['path']:
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
    plt.hist(edge_weights, bins=30, color="skyblue", edgecolor="black")
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

def plot_network(H, title="Network Graph", node_size=700, show_edge_labels=True, node_abbreviations=None):
    """
    Plot the directed network graph with a separate subplot for the legend.

    Parameters:
        H (Graph): The directed graph to plot.
        title (str): The title of the plot.
        node_size (int): Size of the nodes.
        show_edge_labels (bool): Whether to show edge weight labels.
        node_abbreviations (dict): Optional dictionary mapping nodes to their abbreviated labels in the plot.
    """
    # Create a figure with two subplots: one for the network, one for the legend
    fig, (ax_network, ax_legend) = plt.subplots(1, 2, figsize=(14, 8), gridspec_kw={'width_ratios': [4, 1]})

    pos = nx.spring_layout(H, seed=42, k=0.5)  # Layout for consistent spacing

    # Define a color mapping for each node
    nodes = list(H.nodes)
    color_palette = plt.cm.Paired.colors  # A list of colors from a color map
    color_map = {node: color_palette[i % len(color_palette)] for i, node in enumerate(nodes)}
    node_colors = [color_map[node] for node in H.nodes]

    # Use provided abbreviations or default to full node labels
    if node_abbreviations:
        abbrev_labels = {node: node_abbreviations.get(node, node) for node in H.nodes}
    else:
        abbrev_labels = {node: node for node in H.nodes}

    # Draw nodes and labels in the network subplot
    nx.draw_networkx_nodes(H, pos, ax=ax_network, node_size=node_size, node_color=node_colors, edgecolors="black")
    nx.draw_networkx_labels(H, pos, ax=ax_network, labels=abbrev_labels, font_size=10, font_weight="bold")

    # Prepare edge attributes
    all_weights = [H[u][v]['weight'] for u, v in H.edges()]
    max_weight = max(all_weights) if all_weights else 1
    margin = node_size / 60  # Adjust target margin as needed

    # Draw edges with adjusted arrow arrival points in the network subplot
    edge_labels = {}
    for u, v in H.edges():
        weight = H[u][v]['weight']
        edge_width = min(weight / max_weight * 5, 5)  # Scale edge width based on weight
        nx.draw_networkx_edges(
            H, pos, ax=ax_network,
            edgelist=[(u, v)],
            width=edge_width,
            arrowstyle='-|>',
            arrowsize=15,
            edge_color='black',
            connectionstyle=f'arc3,rad=0.1' if H.has_edge(v, u) else 'arc3,rad=0.0',
            min_source_margin=margin,
            min_target_margin=margin
        )
        edge_labels[(u, v)] = f"{weight:.2f}"

    # Draw edge labels if show_edge_labels is True
    if show_edge_labels:
        nx.draw_networkx_edge_labels(
            H, pos, ax=ax_network,
            edge_labels=edge_labels,
            font_size=8,
            label_pos=0.5,
            bbox=dict(alpha=0),
            verticalalignment='center'
        )

    # Prepare legend labels with abbreviations
    if node_abbreviations:
        legend_entries = {
            node: f"{node} ({node_abbreviations.get(node, node)})" for node in H.nodes
        }
    else:
        legend_entries = {node: node for node in H.nodes}

    # Create patches for the legend and add them to the legend subplot
    legend_handles = [
        mpatches.Patch(color=color_map[node], label=legend_entries[node]) for node in H.nodes
    ]
    ax_legend.legend(handles=legend_handles, title="Nodes", loc='upper left')
    ax_legend.axis("off")  

    # Set title and other configurations for the network subplot
    ax_network.set_title(title, fontsize=16)
    ax_network.axis("off")  
    plt.tight_layout()
    plt.show()
