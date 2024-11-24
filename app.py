import argparse
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


######## ------------------------ ########
#        Working with files
######## ------------------------ ########
def readfile(filename):
    df = pd.read_csv(filename, names=["Point1", "Point2", "distance"])

    # Nodes
    nodes = list(set(df["Point1"]).union(set(df["Point2"])))
    node_indices = {node: index for index, node in enumerate(nodes)}

    n = len(nodes)

    adj_matrix = np.full((n, n), np.inf)

    for _, row in df.iterrows():
        i, j = node_indices[row["Point1"]], node_indices[row["Point2"]]
        adj_matrix[i, j] = row["distance"]
        adj_matrix[j, i] = row["distance"]

    adj_matrix_pretty = pd.DataFrame(adj_matrix, index=nodes, columns=nodes)
    
    return adj_matrix_pretty




def visualize_communities(G, communities):
    """
    Visualize graph with community coloring
    """
    plt.figure(figsize=(12, 8))
    
    # Create color map
    color_palette = plt.cm.get_cmap('tab20')
    community_colors = {}
    for i, comm in enumerate(communities):
        for node in comm:
            community_colors[node] = color_palette(i % 20)
    
    # Draw graph
    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=[community_colors[node] for node in G.nodes()],
        node_size=300
    )
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    nx.draw_networkx_labels(G, pos)
    
    plt.title("Community Structure Visualization")
    plt.axis('off')
    plt.tight_layout()
    plt.show()





if __name__ == "__main__":
    data = readfile("data.csv")

    print(data)

