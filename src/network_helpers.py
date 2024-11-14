import networkx as nx
import matplotlib.pyplot as plt

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

def plot_network(H, title="Network Graph", node_size=700, show_edge_labels=True):
    """
    Plot the directed network graph with clear edge directions and optionally show edge weights.
    
    Parameters:
        H (Graph): The directed graph to plot.
        title (str): The title of the plot.
        node_size (int): Size of the nodes.
        show_edge_labels (bool): Whether to show edge weight labels.
    """
    plt.figure(figsize=(8, 8))
    pos = nx.spring_layout(H, seed=42, k=0.5)  # Layout for consistent spacing
    #plt.gca().set_facecolor("white")  # Set background to white

    # Draw nodes
    nx.draw_networkx_nodes(H, pos, node_size=node_size, node_color="skyblue", edgecolors="black")
    nx.draw_networkx_labels(H, pos, font_size=10, font_weight="bold")

    # Prepare edge attributes
    all_weights = [H[u][v]['weight'] for u, v in H.edges()]
    max_weight = max(all_weights) if all_weights else 1

    # Draw edges with arrowheads
    edge_labels = {}
    for u, v in H.edges():
        weight = H[u][v]['weight']
        edge_width = min(weight / max_weight * 5, 5)  # Scale edge width based on weight

        # Draw edge with arrow
        nx.draw_networkx_edges(
            H, pos,
            edgelist=[(u, v)],
            width=edge_width,
            arrowstyle='-|>',
            arrowsize=15,
            edge_color='black',
            connectionstyle='arc3,rad=0.1' if H.has_edge(v, u) else 'arc3,rad=0.0'
        )

        # Prepare edge label with weight
        edge_labels[(u, v)] = f"{weight:.2f}"

    # Draw edge labels (weights) if show_edge_labels is True
    if show_edge_labels:
        nx.draw_networkx_edge_labels(
            H, pos,
            edge_labels=edge_labels,
            font_size=8,
            label_pos=0.5,
            bbox=dict(alpha=0),
            verticalalignment='center'
        )

    plt.title(title, fontsize=16)
    plt.axis("off")
    plt.show()