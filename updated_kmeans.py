import networkx as nx
import matplotlib.pyplot as plt


def balanced_kmeans(graph, k):
    import networkx as nx
    import numpy as np
    
    nodes = list(graph.nodes)
    np.random.seed(43)
    centroids = np.random.choice(nodes, size=k, replace=False)
    max_iterations = 100
    iteration = 0

    while iteration < max_iterations:
        clusters = {i: [] for i in range(k)}
        node_assignments = [-1] * len(nodes)
        
        # Step 1: Assign nodes to clusters with balance
        for idx, node in enumerate(nodes):
            distances = [
                nx.shortest_path_length(graph, source=node, target=centroid, weight="weight")
                for centroid in centroids
            ]
            sorted_indices = np.argsort(distances)
            for cluster_idx in sorted_indices:
                if len(clusters[cluster_idx]) < (len(nodes) + k - 1) // k:
                    clusters[cluster_idx].append(node)
                    node_assignments[idx] = cluster_idx
                    break

        # Step 2: Update centroids
        new_centroids = []
        for cluster_id in clusters:
            subgraph = graph.subgraph(clusters[cluster_id])
            if len(subgraph) == 0:
                new_centroids.append(np.random.choice(nodes))
            else:
                new_centroids.append(
                    min(subgraph.nodes, key=lambda n: sum(nx.shortest_path_length(subgraph, source=n).values()))
                )

        # Convergence check
        if set(new_centroids) == set(centroids):
            break
        
        centroids = new_centroids
        iteration += 1

    if iteration == max_iterations:
        print("Warning: Maximum iterations reached. Clustering may not have converged.")

    return clusters





def kmeans_visualization(graph, communities):
    """
    Visualize a graph with community coloring.

    Args:
        graph (nx.Graph): The graph to be visualized.
        communities (list): A list of lists, where each sublist contains nodes
            belonging to the same community.

    Notes:
        The graph is visualized with a spring layout and each community is
        assigned a color from the "tab20" colormap.
    """
    # Map nodes to their community ID
    node_community_map = {}
    for community_id, community in enumerate(communities):
        for node in community:
            node_community_map[node] = community_id

    # Identify unassigned nodes and assign them to a default community (-1)
    unassigned_nodes = set(graph.nodes()) - set(node_community_map.keys())
    if unassigned_nodes:
        print(f"Warning: Unassigned nodes detected: {unassigned_nodes}")
    for node in unassigned_nodes:
        node_community_map[node] = -1  # Default community ID

    # Assign colors based on community
    colors = [node_community_map[node] for node in graph.nodes()]

    # Graph layout creation
    pos = nx.spring_layout(graph, seed=3)

    # Graph visualization
    plt.figure(figsize=(12, 8))
    nx.draw_networkx(
        graph,
        pos,
        node_color=colors,
        cmap=plt.colormaps["tab20"],
        with_labels=True,
        node_size=500,
        font_size=10,
        edge_color="gray",
        width=0.5,
    )

    plt.title("K-means Visualization")
    plt.show()