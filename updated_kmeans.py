import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize


# Custom k-means with even distribution
def balanced_kmeans(graph, k, max_iter=100):
    # Step 1: Initialize centroids randomly
    nodes = list(graph.nodes)
    np.random.seed(42)
    centroids = np.random.choice(nodes, size=k, replace=False)

    # Step 2: Iterative clustering
    for _ in range(max_iter):
        # Assign nodes to nearest centroid with even distribution
        clusters = {i: [] for i in range(k)}
        for node in nodes:
            distances = [
                nx.shortest_path_length(
                    graph, source=node, target=centroid, weight="weight"
                )
                for centroid in centroids
            ]
            assigned_cluster = np.argmin(distances)

            # Enforce even distribution by limiting cluster size
            if len(clusters[assigned_cluster]) < len(nodes) // k:
                clusters[assigned_cluster].append(node)
            else:
                # If full, assign to the next closest cluster
                sorted_distances = sorted(enumerate(distances), key=lambda x: x[1])
                for idx, _ in sorted_distances:
                    if len(clusters[idx]) < len(nodes) // k:
                        clusters[idx].append(node)
                        break

        # Step 3: Update centroids
        new_centroids = []
        for i in range(k):
            subgraph = graph.subgraph(clusters[i])
            new_centroid = min(
                subgraph.nodes,
                key=lambda n: sum(nx.shortest_path_length(subgraph, source=n).values()),
            )
            new_centroids.append(new_centroid)

        if set(new_centroids) == set(centroids):  # Check for convergence
            break
        centroids = new_centroids

    return clusters


# Visualize the results
def visualize_clusters(graph, clusters, title="Balanced K-Means Clustering"):
    pos = nx.spring_layout(graph, seed=42)
    plt.figure(figsize=(12, 8))
    colors = plt.cm.get_cmap("tab10", len(clusters))

    for cluster_id, nodes in clusters.items():
        nx.draw_networkx_nodes(
            graph,
            pos,
            nodelist=nodes,
            node_color=[colors(cluster_id)],
            label=f"Cluster {cluster_id}",
        )
    nx.draw_networkx_edges(graph, pos, alpha=0.5)
    nx.draw_networkx_labels(graph, pos, font_size=10)

    plt.title(title)
    plt.legend()
    plt.show()


# Load your graph
# Replace this with your graph loading code
data = pd.read_csv("data.csv")
data.rename(
    columns={
        "Назва міста1": "Point1",
        "Назва міста2": "Point2",
        "Відстань (км)": "distance",
    },
    inplace=True,
)

G = nx.Graph()
for _, row in data.iterrows():
    G.add_edge(row["Point1"], row["Point2"], weight=row["distance"])

# Run balanced k-means
k = 3  # Number of clusters
clusters = balanced_kmeans(G, k)

# Visualize the balanced clusters
visualize_clusters(G, clusters)
