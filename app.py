"""
Administrative points unification
"""

import argparse
import networkx as nx
import pandas as pd


from rich.console import Console


from updated_kmeans import balanced_kmeans, kmeans_visualization

from louvian_algo import (
    louvain_community_detection,
    louvian_visualize_communities,
    communities_to_dict,
)


def read_distance_data(file_path):
    """
    Read CSV file with city distance data

    Args:
        file_path (str): Path to the CSV file

    Returns:
        pd.DataFrame: Processed distance data
    """
    try:
        # Read CSV file
        data = pd.read_csv(file_path, encoding="utf-8")

        # Rename columns to standard format
        data.rename(
            columns={
                "Назва міста1": "Point1",
                "Назва міста2": "Point2",
                "Відстань (км)": "distance",
            },
            inplace=True,
        )

        required_columns = ["Point1", "Point2", "distance"]
        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"Missing required columns. Need: {required_columns}")

        return data

    except Exception as e:
        print(f"Error reading CSV file: {e}")
        raise


def create_graph_from_csv(file_path):
    """
    Create NetworkX graph from CSV distance data

    Args:
        file_path (str): Path to the CSV file

    Returns:
        nx.Graph: Graph with cities as nodes and distances as edge weights
    """
    # Read distance data
    data = read_distance_data(file_path)

    # Create graph
    g = nx.Graph()
    for _, row in data.iterrows():
        g.add_edge(row["Point1"], row["Point2"], weight=row["distance"])

    return g


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
    g = nx.Graph()
    for _, row in data.iterrows():
        g.add_edge(
            row["Назва міста1"], row["Назва міста2"], weight=row["Відстань (км)"]
        )
    return g


def main():
    """
    Main Function
    """
    parser = argparse.ArgumentParser(
        description="Process a file with optional features for visualization and clustering."
    )

    # Required flag for the path to the file
    parser.add_argument(
        "--path", required=True, type=str, help="Path to the file to be processed"
    )

    parser.add_argument("--visual", action="store_true", help="Enable visualization")

    parser.add_argument(
        "--kmeans",
        type=int,
        metavar="NUM_CLUSTERS",
        help="Use the k-means algorithm with the specified number of clusters",
    )

    args = parser.parse_args()
    # Initialize the rich console
    console = Console()

    # Define custom styles
    file_path_style = "bold blue"
    info_style = "bold cyan"
    success_style = "bold green"
    warning_style = "bold yellow"

    # Display the file path
    console.print(f"🎯 File path: {args.path}", style=file_path_style)

    if args.kmeans is not None:
        # K-means enabled message
        console.print(
            f"📊 K-means enabled with {args.kmeans} clusters.", style=info_style
        )
        g = create_graph_from_csv("data.csv")

        if args.visual:
            # Visualization task message
            console.print(
                "🎨 Generating balanced K-means visualization...", style=warning_style
            )
            clusters = balanced_kmeans(g, args.kmeans)
            communities = list(clusters.values())
            kmeans_visualization(g, communities)
            console.print("✅ Visualization complete!", style=success_style)
        else:
            # Print clusters message
            console.print("📋 Printing cluster information...", style=info_style)
            print_clusters_visual(communities_to_dict(balanced_kmeans(g, args.kmeans)))
    else:
        # Louvain algorithm message
        console.print(
            "🧠 Using Louvain algorithm for community detection.", style=info_style
        )
        g = create_graph_from_csv("data.csv")

        if args.visual:
            # Visualization task message
            console.print("🎨 Visualizing Louvain communities...", style=warning_style)
            louvian_visualize_communities(g, louvain_community_detection(g))
            console.print("✅ Visualization complete!", style=success_style)
        else:
            # Print clusters message
            console.print("📋 Printing cluster information...", style=info_style)
            print_clusters_visual(communities_to_dict(louvain_community_detection(g)))


if __name__ == "__main__":
    main()
