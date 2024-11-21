from sklearn.metrics import pairwise_distances
import numpy as np


def kmeans(adj_matrix, nodes, n_clusters):
    """
    Реалізація алгоритму K-Means для кластеризації.

    Параметри:
    adj_matrix (numpy.ndarray): Матриця суміжності.
    nodes (list): Список вузлів.
    n_clusters (int): Кількість кластерів.

    Повертає:
    dict: Мапа кластерів та відповідних вузлів.
    """
    # Ініціалізація кластерних центрів
    n_samples = adj_matrix.shape[0]
    centroids = adj_matrix[np.random.choice(n_samples, n_clusters, replace=False)]

    prev_labels = None

    while True:
        # Визначення кластера для кожної точки
        distances = pairwise_distances(adj_matrix, centroids)
        labels = np.argmin(distances, axis=1)

        for i, dist in enumerate(distances):
            if dist[0] == dist[1]:  # Якщо відстані рівні
                labels[i] = np.random.choice([0, 1])  # Випадковий вибір між двома кластерами

        # Перевірка на завершення: якщо кластери більше не змінюються
        if np.array_equal(labels, prev_labels):
            break

        prev_labels = labels

        # Оновлення кластерних центрів
        centroids = np.array([adj_matrix[labels == i].mean(axis=0) for i in range(n_clusters)])

    # Групування вузлів за кластерами
    clustered_nodes = {i: [] for i in range(n_clusters)}

    for node, cluster in zip(nodes, labels):
        clustered_nodes[cluster].append(node)

    return clustered_nodes
