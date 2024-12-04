import networkx as nx
import random

import matplotlib.pyplot as plt

import pandas as pd

def read_distance_data(file_path: str):
    """
    Read CSV file with city distance data
    
    Args:
        file_path (str): Path to the CSV file
    
    Returns:
        pd.DataFrame: Processed distance data

    >>> import tempfile
        >>> import pandas as pd
        >>> data = '''Назва міста1,Назва міста2,Відстань (км)
        ... Львів,Київ,540
        ... Харків,Одеса,700
        ... Київ,Одеса,480'''
        >>> with tempfile.NamedTemporaryFile(mode='w+', suffix='.csv', delete=False) as tmp:
        ...     _ = tmp.write(data)
        ...     file_path = tmp.name
        >>> df = read_distance_data(file_path)
        >>> print(df)
           Point1 Point2  distance
        0   Львів   Київ       540
        1  Харків  Одеса       700
        2    Київ  Одеса       480
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
        
        # Validate data
        required_columns = ['Point1', 'Point2', 'distance']
        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"Missing required columns. Need: {required_columns}")
        
        return data
    
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        raise

def create_graph_from_csv(file_path: str):
    """
    Create NetworkX graph from CSV distance data
    
    Args:
        file_path (str): Path to the CSV file
    
    Returns:
        nx.Graph: Graph with cities as nodes and distances as edge weights
    >>> import tempfile
    >>> import networkx as nx
    >>> data = '''Назва міста1,Назва міста2,Відстань (км)
    ... Львів,Київ,540
    ... Харків,Одеса,700
    ... Київ,Одеса,480'''
    >>> with tempfile.NamedTemporaryFile(mode='w+', suffix='.csv', delete=False) as tmp:
    ...     _ = tmp.write(data)
    ...     file_path = tmp.name
    >>> G = create_graph_from_csv(file_path)
    >>> sorted(G.nodes())
    ['Київ', 'Львів', 'Одеса', 'Харків']
    >>> G['Львів']['Київ']['weight']
    540
    >>> G['Київ']['Одеса']['weight']
    480
    """
    # Read distance data
    data = read_distance_data(file_path)
    
    # Create graph
    G = nx.Graph()
    for _, row in data.iterrows():
        G.add_edge(row["Point1"], row["Point2"], weight=row["distance"])
    
    return G


def create_graph_dict(graph):
    """
    Convert NetworkX graph to adjacency dictionary with edge weights
    >>> G = nx.Graph()
    >>> G.add_edge("A", "B", weight=3)
    >>> G.add_edge("A", "C", weight=2)
    >>> G.add_edge("B", "C", weight=1)
    >>> graph_dict = create_graph_dict(G)
    >>> print(graph_dict)  # doctest: +NORMALIZE_WHITESPACE
    {'A': {'B': 3, 'C': 2},
        'B': {'A': 3, 'C': 1},
        'C': {'A': 2, 'B': 1}}
    """
    graph_dict = {}
    for node in graph.nodes():
        graph_dict[node] = {
            neighbor: graph[node][neighbor].get("weight", 1.0)
            for neighbor in graph.neighbors(node)
        }
    return graph_dict

def calculate_total_weight(graph_dict: dict) -> float:
    """
    Calculate total weight of the graph

    >>> graph_dict = {'A': {'B': 3, 'C': 2},
    ...               'B': {'A': 3, 'C': 1},
    ...               'C': {'A': 2, 'B': 1}}
    >>> calculate_total_weight(graph_dict)
    6.0
    """
    return sum(sum(neighbors.values()) for neighbors in graph_dict.values()) / 2


def calculate_node_degree(graph_dict: dict) -> float:
    """
    Calculate degree (weighted) for each node
    >>> graph_dict = {'A': {'B': 3, 'C': 2},
    ...               'B': {'A': 3, 'C': 1},
    ...               'C': {'A': 2, 'B': 1}}
    >>> calculate_node_degree(graph_dict)
    {'A': 5, 'B': 4, 'C': 3}
    """
    return {node: sum(neighbors.values()) for node, neighbors in graph_dict.items()}


def modularity_gain(graph_dict: dict, total_weight: float, node: float, community: 'str', node_communities: dict) -> float:
    """
    Calculate modularity gain when moving a node to a community
    >>> graph_dict = {
    ...     'A': {'B': 3, 'C': 2},
    ...     'B': {'A': 3, 'C': 4, 'D': 1},
    ...     'C': {'A': 2, 'B': 4},
    ...     'D': {'B': 1}
    ... }
    >>> total_weight = 10.0
    >>> node = 'A'
    >>> test_community = 'community2'
    >>> node_communities = {
    ...     'community1': {'C', 'B'},
    ...     'community2': {'D'}
    ... }
    >>> modularity_gain(graph_dict, total_weight, node, test_community, node_communities)
    -0.025
    """
    node_degree = sum(graph_dict[node].values())
    community_internal_weight = sum(
        graph_dict[node].get(other_node, 0.0)
        for other_node in node_communities[community]
    )

    community_total_weight = sum(
        sum(graph_dict[comm_node].values()) for comm_node in node_communities[community]
    )

    delta_modularity = community_internal_weight / total_weight - (
        community_total_weight * node_degree
    ) / (2 * total_weight**2)

    return delta_modularity


def louvain_community_detection(graph):
    """
    Implement Louvain community detection algorithm
    >>> G = nx.Graph()
    >>> G.add_edge("A", "B", weight=1)
    >>> G.add_edge("B", "C", weight=1)
    >>> G.add_edge("C", "D", weight=1)
    >>> G.add_edge("D", "A", weight=1)
    >>> G.add_edge("A", "C", weight=1)
    >>> communities = louvain_community_detection(G)
    >>> sorted([sorted(list(c)) for c in communities])
    [['A'], ['B'], ['C'], ['D']]
    """
    import random
    # Convert graph to dictionary representation
    graph_dict = create_graph_dict(graph)
    total_weight = calculate_total_weight(graph_dict)

    # Initialize each node in its own community
    node_communities = {node: {node} for node in graph_dict}
    community_of_node = {node: node for node in graph_dict}

    improved = True
    while improved:
        improved = False

        # Randomize node order for each iteration
        nodes = list(graph_dict.keys())
        random.shuffle(nodes)

        for node in nodes:
            current_community = community_of_node[node]
            best_community = current_community
            best_modularity_gain = 0

            # Remove node from current community
            node_communities[current_community].remove(node)

            # Check modularity gain for neighboring communities
            neighbor_communities = set()
            for neighbor in graph_dict[node]:
                neighbor_community = community_of_node[neighbor]
                neighbor_communities.add(neighbor_community)

            for test_community in neighbor_communities:
                # Add node to test community
                node_communities[test_community].add(node)

                # Calculate modularity gain
                gain = modularity_gain(
                    graph_dict, total_weight, node, test_community, node_communities
                )

                if gain > best_modularity_gain:
                    best_modularity_gain = gain
                    best_community = test_community

                # Remove node from test community if not chosen
                if best_community != test_community:
                    node_communities[test_community].remove(node)

            # Add node to best community
            node_communities[best_community].add(node)
            community_of_node[node] = best_community

            # Check if improvement occurred
            if best_community != current_community:
                improved = True

    # Convert communities to list of sets
    return node_communities


def louvian_visualize_communities(graph, communities):
    """
    Visualize graph with community coloring
    Robust version to handle missing nodes
    """
    # Create a dictionary that maps each node to its community
    node_community_map = {}
    for community_id, community in enumerate(communities.values()):
        for node in community:
            node_community_map[node] = community_id

    # Generate a list of colors with fallback
    colors = []
    for node in graph.nodes():
        # Use a default color (e.g., gray) if node not in any community
        colors.append(node_community_map.get(node, len(communities)))

    # Create a layout for the graph
    pos = nx.spring_layout(graph, seed=43)

    # Draw the graph with colored nodes
    plt.figure(figsize=(12, 8))
    nx.draw_networkx(
        graph,
        pos,
        node_color=colors,
        cmap=plt.cm.get_cmap("tab20", len(communities) + 1),
        with_labels=True,
        node_size=500,
        font_size=10,
        edge_color="gray",
        width=0.5,
    )

    # Title for the plot
    plt.title("Louvain Community Detection Visualization")
    plt.show()


def communities_to_dict(communities):
    """
    Convert communities from Louvain detection to a dictionary where
    keys are re-indexed community IDs and values are lists of non-empty communities.

    Args:
        communities (dict): Communities dictionary from louvain_community_detection

    Returns:
        dict: Mapping of re-indexed community IDs to lists of nodes, excluding empty communities
    >>> communities = {
    ...     'community1': {'A', 'B'},
    ...     'community2': {'C', 'D'},
    ...     'community3': set()
    ... }
    >>> communities_to_dict(communities) == {0: {'A', 'B'}, 1: {'D', 'C'}}
    True
    """
    # Filter out empty communities and create a new list
    non_empty_communities = [community for community in communities.values() if community]

    # Re-index the communities and return as a dictionary
    return {
        idx: list(community)
        for idx, community in enumerate(non_empty_communities)
    }



# if __name__ == "__main__":
#     G = create_graph_from_csv("data.csv")
#     louvian_visualize_communities(G, louvain_community_detection(G))


if __name__ == "__main__":
    import doctest
    print(doctest.testmod())