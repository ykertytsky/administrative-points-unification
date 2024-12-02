import argparse
import sys
import warnings
import networkx as nx
import matplotlib.pyplot as plt
from rich.console import Console

from updated_kmeans import (
    kmeans_clustering,
    clusters_to_nx_graph,
    read_distance_data,
    create_distance_matrix,
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
        Console().print(f"Cluster {cluster_id+1}:", style="bold red")
        for city in cities:
            print(f"  - {city}")
        print("\n")

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
        "-visual",
        type=bool,
        default=False,
        help="Visualize algorithm. False by default",
    )

    # Optional arguments
    parser.add_argument(
        "--kmeans",
        action="store_true",
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
        df = read_distance_data(args.file_path)
        distances, cities = create_distance_matrix(df)

        if args.kmeans:
            console.print("Performing k-means clustering...", style="bold yellow")
            city_clusters = kmeans_clustering(distances, cities, args.num_clusters)

        # Group cities by clusters
        communities = {}
        for city, cluster in city_clusters.items():
            if cluster not in communities:
                communities[cluster] = []
            communities[cluster].append(city)

        communities_list = list(communities.values())

        # Visualize the results
        if args.visual:
            console.print("Visualizing clusters...", style="bold magenta")
            G = clusters_to_nx_graph(city_clusters, cities, distances)
            visualize_communities(G, communities_list)
        else:
            console.print("Showing cluster distribution")
            print_clusters_visual(communities)

        console.print("Clustering and visualization complete!", style="bold green")

    except Exception as e:
        console.print(f"Error: {e}")
        sys.exit()

if __name__ == "__main__":
    main()
