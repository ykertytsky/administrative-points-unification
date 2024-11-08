import matplotlib.pyplot as plt
import networkx as nx

def visualize_graph(graph, clusters=None):
    pos = nx.spring_layout(graph)  # or other layouts like `nx.kamada_kawai_layout`
    if clusters:
        node_colors = [clusters[node] for node in graph.nodes()]
    else:
        node_colors = 'blue'
    nx.draw(graph, pos, with_labels=True, node_color=node_colors, cmap=plt.cm.rainbow)
    plt.show()
