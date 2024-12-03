import argparse
import sys
import warnings
import networkx as nx
import matplotlib.pyplot as plt
from rich.console import Console

import random

from kmeans import (
    kmeans_clustering,
    clusters_to_nx_graph,
    read_distance_data,
    create_distance_matrix,
)

from louvian_algo import (
    louvain_community_detection,
    communities_to_dict,
    louvian_visualize_communities,
    create_graph_from_csv
)

def visualize_communities(graph, communities):
    """
    Visualize graph with community coloring
    """
    # Create a dictionary that maps each node to its community
    node_community_map = {}
    for community_id, community in enumerate(communities):
        for node in community:
            node_community_map[node] = community_id

    # Generate a list of colors corresponding to each community
    colors = [node_community_map[node] for node in graph.nodes()]

    # Graph Layout creation
    pos = nx.spring_layout(graph, seed=42)

    # Graph Visualization
    plt.figure(figsize=(12, 8))
    nx.draw_networkx(
        graph,
        pos,
        node_color=colors,
        cmap=plt.colormaps["tab20"],  # Different Colors from tab20 table
        with_labels=True,
        node_size=500,
        font_size=10,
        edge_color="gray",
        width=0.5,
    )

    plt.title("K-means Visualization")
    plt.show()

def print_clusters_visual(clusters):
    """
    Prints a visual representation of clusters and their corresponding cities.

    Args:
        clusters (dict): A dictionary where keys are cluster names (or IDs) and
                         values are lists of city names.
    """
    print("\nClusters and their corresponding cities:\n")
    for cluster_id, cities in clusters.items():
        print(f"Cluster {cluster_id+1}:")
        for city in cities:
            print(f"  - {city}")
        print("\n")

def create_graph(data):
    """
    Create a NetworkX graph from distance data.
    """
    G = nx.Graph()
    for _, row in data.iterrows():
        G.add_edge(row["Назва міста1"], row["Назва міста2"], weight=row["Відстань (км)"])
    return G


def main():
    """
    Main Function with argparse active
    """
    warnings.filterwarnings("ignore")
    console = Console()

    parser = argparse.ArgumentParser(
        description="Graph Clustering and Visualization"
    )

    # Required argument
    parser.add_argument(
        "file_path",
        type=str,
        help="Path to the CSV data file containing city distances",
    )
    parser.add_argument(
        "--visual",
        type=bool,
        default=False,
        help="Visualize algorithm. False by default",
    )

    # Optional arguments
    parser.add_argument(
        "--kmeans",
        action="store_true",  # This makes --kmeans a boolean flag
        help="Use k-means clustering algorithm (default is Louvain)",
    )
    parser.add_argument(
        "--num_clusters",
        type=int,
        default=3,
        help="Number of clusters for k-means (default is 3)",
    )

    # Parse arguments
    args = parser.parse_args()

    console.print(
        "Starting graph clustering...", style="bold green"
    )
    try:
        console.print(f"Reading data from: {args.file_path}", style="bold blue")
        data = read_distance_data(args.file_path)
        graph = create_graph(data)
        
        if args.kmeans:  # Check if --kmeans flag is provided
            from updated_kmeans import balanced_kmeans, visualize_clusters
            console.print("Performing balanced k-means clustering...", style="bold yellow")
            
            clusters = balanced_kmeans(graph, args.num_clusters)

            # Only visualize if --visual flag is explicitly True
            if args.visual:
                console.print("Visualizing clusters...", style="bold magenta")
                visualize_clusters(graph, clusters)
            else:
                console.print("Showing cluster distribution (non-visual)", style="bold magenta")
                print_clusters_visual(clusters)

        else:
            console.print("Performing Louvain community detection...", style="bold yellow")
            communities = louvain_community_detection(graph)
            
            if args.visual:
                console.print("Visualizing communities...", style="bold magenta")
                G = create_graph_from_csv("data.csv")
                louvian_visualize_communities(G, louvain_community_detection(G))
            else:
                console.print("Showing community distribution", style="bold magenta")
                communities = communities_to_dict(communities)
                print_clusters_visual(communities)
        
        console.print("Community detection and visualization complete!", style="bold green")
    except Exception as e:
        console.print(f"Error: {e}", style="bold red")
        sys.exit(1)



if __name__ == "__main__":
    main()
