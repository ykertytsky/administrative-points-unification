import argparse
import sys
import networkx as nx
import matplotlib.pyplot as plt

import pandas as pd


from updated_kmeans import (
    balanced_kmeans,
    kmeans_visualization
)

from louvian_algo import (
    louvain_community_detection,
    louvian_visualize_communities,
    communities_to_dict
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
        data = pd.read_csv(file_path, encoding='utf-8')
        
        # Rename columns to standard format
        data.rename(
            columns={
                "Назва міста1": "Point1", 
                "Назва міста2": "Point2", 
                "Відстань (км)": "distance"
            },
            inplace=True
        )
        
        required_columns = ['Point1', 'Point2', 'distance']
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
    G = nx.Graph()
    for _, row in data.iterrows():
        G.add_edge(row["Point1"], row["Point2"], weight=row["distance"])
    
    return G




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
    parser = argparse.ArgumentParser(description="Process a file with optional features for visualization and clustering.")
    
    # Required flag for the path to the file
    parser.add_argument(
        "--path",
        required=True,
        type=str,
        help="Path to the file to be processed"
    )


    parser.add_argument(
        "--visual",
        action="store_true",
        help="Enable visualization"
    )

    parser.add_argument(
        "--kmeans",
        type=int,
        metavar="NUM_CLUSTERS",
        help="Use the k-means algorithm with the specified number of clusters"
    )

    args = parser.parse_args()

    print(f"File path: {args.path}")

    if args.kmeans is not None:
        print(f"K-means enabled with {args.kmeans} clusters.")
        G = create_graph_from_csv("data.csv")
        if args.visual:
            clusters = balanced_kmeans(G, args.kmeans)
            communities = list(clusters.values())
            kmeans_visualization(G, communities)
        else:
            print_clusters_visual(communities_to_dict(balanced_kmeans(G, args.kmeans)))
        
    else:
        print("Using Louvian algorithm for community detection.")
        G = create_graph_from_csv("data.csv")
        if args.visual:
            louvian_visualize_communities(G, louvain_community_detection(G))
        else:
            print_clusters_visual(communities_to_dict(louvain_community_detection(G)))

    if args.visual:
        print("Visualization enabled.")





if __name__ == "__main__":
    main()
