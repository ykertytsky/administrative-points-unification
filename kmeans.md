## **1. Імпортування бібліотек**

`import networkx as nx`
`import numpy as np`
`import pandas as pd`
`import matplotlib.pyplot as plt`

- **`networkx` (nx)**: бібліотека для створення, маніпулювання та аналізу графів і мереж.
- **`numpy` (np)**: бібліотека для роботи з масивами та виконання числових обчислень.
- **`pandas` (pd)**: бібліотека для роботи з таблицями та аналізу даних у форматі `DataFrame`.
- **`matplotlib.pyplot` (plt)**: бібліотека для створення візуалізацій.

---

## **2. Функція `balanced_kmeans`**

`def balanced_kmeans(graph, k):`
    `# Step 1: Initialize centroids randomly`
    `nodes = list(graph.nodes)`
    `np.random.seed(42)  # Fix random seed for reproducibility`
    `centroids = np.random.choice(nodes, size=k, replace=False)`

- **`graph`** – це об'єкт графу `networkx.Graph`, який передається у функцію.
- **`k`** – кількість кластерів, на які потрібно поділити граф.
- **`nodes`** – список усіх вузлів графу (вершин).
- **`np.random.seed(42)`** – фіксує випадковий генератор для відтворюваності результатів (щоб алгоритм завжди генерував однакові випадкові значення).
- **`np.random.choice(nodes, size=k, replace=False)`** – випадковим чином вибирає `k` різних вузлів для ініціалізації центроїдів кластерів.

### Основний цикл (цикл кластеризації)

    while True:  # Infinite loop, will break on convergence
        # Step 2: Assign nodes to nearest centroid with even distribution
        clusters = {i: [] for i in range(k)}
        for node in nodes:
            distances = [
                nx.shortest_path_length(graph, source=node, target=centroid, weight="weight")
                for centroid in centroids
            ]
            assigned_cluster = np.argmin(distances)

- **`while True`**: Цикл продовжується, поки центроїди не перестануть змінюватися.
- **`clusters = {i: [] for i in range(k)}`**: Створюється словник для збереження вузлів кожного кластера.
- **`nx.shortest_path_length(graph, source=node, target=centroid, weight="weight")`**: Обчислює найкоротшу відстань між вузлом `node` та центроїдом `centroid`.
    - **`weight="weight"`** означає, що враховується вага ребра.
 - **`np.argmin(distances)`**: Знаходить індекс центроїда з мінімальною відстанню до вузла.

### Рівномірний розподіл вузлів

            # Enforce even distribution by limiting cluster size
            if len(clusters[assigned_cluster]) < len(nodes) // k:
                clusters[assigned_cluster].append(node)
            else:
                # If full, assign to the next closest cluster
                sorted_distances = sorted(enumerate(distances), key=lambda x: x[1])
                for idx, _ in sorted_distances:
                    if len(clusters[idx]) < len(nodes) // k:
                        clusters[idx].append(node)
                        break

- **`len(nodes) // k`**: максимальна кількість вузлів, яку можна додати в кластер для рівномірного розподілу.
- Якщо кластер заповнений, вузол додається до наступного найближчого кластеру.

### Оновлення центроїдів

        # Step 3: Update centroids
        new_centroids = []
        for i in range(k):
            subgraph = graph.subgraph(clusters[i])
            new_centroid = min(
                subgraph.nodes,
                key=lambda n: sum(nx.shortest_path_length(subgraph, source=n).values()),
            )
            new_centroids.append(new_centroid)

- **`graph.subgraph(clusters[i])`**: створює підграф для вузлів у кластері `i`.
- **`min(..., key=...)`**: вибирає вузол, сумарна відстань якого до інших вузлів мінімальна (вибір нового центроїда).

### Перевірка на завершення

        # Check for convergence: if centroids do not change, break the loop
        if set(new_centroids) == set(centroids):
            break

        centroids = new_centroids  # Update centroids for next iteration

Якщо центроїди не змінилися після ітерації, алгоритм завершує роботу.

---

## **3. Функція `visualize_clusters`**

`def visualize_clusters(graph, clusters, title="Balanced K-Means Clustering"):`
    `pos = nx.spring_layout(graph, seed=42)`
    `plt.figure(figsize=(12, 8))`
    `colors = plt.cm.get_cmap("tab10", len(clusters))`

    for cluster_id, nodes in clusters.items():
        nx.draw_networkx_nodes(
            graph,
            pos,
            nodelist=nodes,
            node_color=[colors(cluster_id)],
            label=f"Cluster {cluster_id}",
        )
    nx.draw_networkx_edges(graph, pos, alpha=0.5)
    nx.draw_networkx_labels(graph, pos, font_size=10)

    plt.title(title)
    plt.legend()
    plt.show()

- **`nx.spring_layout(graph)`**: генерує координати для розташування вузлів графу.
- **`plt.figure(figsize=(12, 8))`**: задає розмір графіку.
- **`colors = plt.cm.get_cmap("tab10", len(clusters))`**: генерує кольори для кластерів.
- **`nx.draw_networkx_nodes`**: відображає вузли графу.
- **`nx.draw_networkx_edges`**: відображає ребра графу.
- **`nx.draw_networkx_labels`**: відображає підписи вузлів.

---
