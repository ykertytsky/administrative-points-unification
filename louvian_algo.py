import networkx as nx
import random

import matplotlib.pyplot as plt

def create_graph_dict(graph):
    """
    Convert NetworkX graph to adjacency dictionary with edge weights
    """
    graph_dict = {}
    for node in graph.nodes():
        graph_dict[node] = {
            neighbor: graph[node][neighbor].get('weight', 1.0) 
            for neighbor in graph.neighbors(node)
        }
    return graph_dict

def calculate_total_weight(graph_dict):
    """
    Calculate total weight of the graph
    """
    return sum(sum(neighbors.values()) for neighbors in graph_dict.values()) / 2

def calculate_node_degree(graph_dict):
    """
    Calculate degree (weighted) for each node
    """
    return {node: sum(neighbors.values()) for node, neighbors in graph_dict.items()}

def modularity_gain(graph_dict, total_weight, node, community, node_communities):
    """
    Calculate modularity gain when moving a node to a community
    """
    node_degree = sum(graph_dict[node].values())
    community_internal_weight = sum(
        graph_dict[node].get(other_node, 0.0) 
        for other_node in node_communities[community]
    )
    
    community_total_weight = sum(
        sum(graph_dict[comm_node].values()) 
        for comm_node in node_communities[community]
    )
    
    delta_modularity = (
        community_internal_weight / total_weight - 
        (community_total_weight * node_degree) / (2 * total_weight ** 2)
    )
    
    return delta_modularity

def louvain_community_detection(graph):
    """
    Implement Louvain community detection algorithm
    """
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
    return list(node_communities.values())


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

    # Create a layout for the graph
    pos = nx.spring_layout(graph, seed=42)

    # Draw the graph with colored nodes
    plt.figure(figsize=(12, 8))
    nx.draw_networkx(
        graph,
        pos,
        node_color=colors,  # Color nodes by their community
        cmap=plt.cm.get_cmap('tab20', len(communities)),  # Color map with a different color for each community
        with_labels=True,
        node_size=500,
        font_size=10,
        edge_color='gray',
        width=0.5
    )

    # Title for the plot
    plt.title("Louvain Community Detection Visualization")
    plt.show()

if __name__ == "__main__":
    # Create a graph (for example, from some data or an existing NetworkX graph)
    G = nx.erdos_renyi_graph(30, 0.05)  # Example graph for testing
    
    # Apply Louvain community detection
    communities = louvain_community_detection(G)
    
    # Visualize the communities
    visualize_communities(G, communities)
