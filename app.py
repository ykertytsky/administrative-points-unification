import argparse
import numpy
import pandas as pd
import numpy as np


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

if __name__ == "__main__":
    data = readfile("data.csv")

    print(data)
