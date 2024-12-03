import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def balanced_kmeans(graph, k):
    """
    Balanced K-Means clustering algorithm for NetworkX graphs.

    This algorithm extends the traditional K-Means algorithm to ensure that the
    number of nodes in each cluster is roughly equal. It does this by initially
    assigning nodes to the nearest centroid and then reassigning nodes to other
    clusters if the initial assignment results in uneven cluster sizes.

    Parameters
    ----------
    graph : NetworkX graph
        The graph to be clustered.
    k : int
        The number of clusters to form.

    Returns
    -------
    clusters : dict
        A dictionary where the keys are the cluster IDs and the values are lists
        of nodes in that cluster.

    Notes
    -----
    The algorithm iterates until either the maximum number of iterations is
    reached or the centroids do not change between iterations. If the maximum
    number of iterations is reached, a warning message is printed to indicate
    that the clustering may not have converged.
    """
    nodes = list(graph.nodes)
    np.random.seed(42)
    centroids = np.random.choice(nodes, size=k, replace=False)
    max_iterations = 100
    iteration = 0

    while iteration < max_iterations:
        clusters = {i: [] for i in range(k)}
        for node in nodes:
            distances = [
                nx.shortest_path_length(graph, source=node, target=centroid, weight="weight")
                for centroid in centroids
            ]
            assigned_cluster = np.argmin(distances)

            if len(clusters[assigned_cluster]) < len(nodes) // k:
                clusters[assigned_cluster].append(node)
            else:
                sorted_distances = sorted(enumerate(distances), key=lambda x: x[1])
                for idx, _ in sorted_distances:
                    if len(clusters[idx]) < len(nodes) // k:
                        clusters[idx].append(node)
                        break

        for cluster_id in clusters:
            if not clusters[cluster_id]:  # Handle empty clusters
                clusters[cluster_id].append(nodes.pop())

        new_centroids = []
        for i in range(k):
            subgraph = graph.subgraph(clusters[i])
            if not subgraph.nodes:  # Handle empty subgraph case
                new_centroid = np.random.choice(nodes)
            else:
                new_centroid = min(
                    subgraph.nodes,
                    key=lambda n: sum(nx.shortest_path_length(subgraph, source=n).values()),
                )
            new_centroids.append(new_centroid)

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
    pos = nx.spring_layout(graph, seed=42)

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