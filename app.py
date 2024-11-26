"""
ADministative Points Unification
"""

import argparse
import networkx as nx
import matplotlib.pyplot as plt
from rich import print
from rich.console import Console
from rich.progress import track


from kmeans import *


######## ------------------------ ########
#        Working with files
######## ------------------------ ########
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

    # Cтворення лейауту для графу
    pos = nx.spring_layout(graph, seed=42)

    # Візуалізація графу з обраними параметрами
    plt.figure(figsize=(12, 8))
    nx.draw_networkx(
        graph,
        pos,
        node_color=colors,  
        cmap = plt.colormaps['tab20'](len(communities)),  # Інший колір для кожного комʼюніті
        with_labels=True,
        node_size=500,
        font_size=10,
        edge_color='gray',
        width=0.5
    )

    plt.title("Louvain Community Detection Visualization")
    plt.show()



def main():
    console = Console()

    parser = argparse.ArgumentParser(
        description="🎨 Graph Clustering and Visualization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required argument
    parser.add_argument(
        "file_path", 
        type=str, 
        help="Path to the CSV data file containing city distances"
    )

    # Optional arguments
    parser.add_argument(
        "--kmeans", 
        action="store_true", 
        help="Use k-means clustering algorithm (default is Louvain)"
    )
    parser.add_argument(
        "--num_clusters", 
        type=int, 
        default=3, 
        help="Number of clusters for k-means (default is 3)"
    )

    # Parse arguments
    args = parser.parse_args()

    # Log starting message
    console.print(f"🚀 Starting graph clustering and visualization...", style="bold green")

    try:
        # Read file and process data
        console.print(f"📖 Reading data from: {args.file_path}", style="bold blue")
        df = read_distance_data(args.file_path)
        distances, cities = create_distance_matrix(df)

        if args.kmeans:
            console.print(f"🔍 Performing k-means clustering...", style="bold yellow")
            city_clusters = kmeans_clustering(distances, cities, args.num_clusters)

        # Group cities by clusters
        communities = {}
        for city, cluster in city_clusters.items():
            if cluster not in communities:
                communities[cluster] = []
            communities[cluster].append(city)

        communities_list = list(communities.values())

        # Visualize the results
        console.print(f"📊 Visualizing clusters...", style="bold magenta")
        G = clusters_to_nx_graph(city_clusters, cities, distances)
        visualize_communities(G, communities_list)

        console.print(f"✔️ Clustering and visualization complete!", style="bold green")

    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        exit(1)

if __name__ == "__main__":
    main()

