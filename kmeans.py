"""
K MEANS
"""

import pandas as pd
import numpy as np
import networkx as nx


def read_distance_data(file_path):
    """
    Зчитує дані з CSV-файлу у форматі, потрібному для програми.

    Args:
        file_path (str): Шлях до CSV-файлу.

    Returns:
        pd.DataFrame: DataFrame з даними про відстані.
    """
    try:
        df = pd.read_csv(file_path)

        # Перевірка наявності потрібних стовпців
        required_columns = {"Назва міста1", "Назва міста2", "Відстань (км)"}
        if not required_columns.issubset(df.columns):
            raise ValueError(
                f"CSV файл повинен містити такі стовпці: {', '.join(required_columns)}"
            )

        # Перевірка на відсутність порожніх значень
        if df.isnull().any().any():
            raise ValueError("CSV файл містить порожні значення. Перевірте дані.")

        # Перетворення типу стовпця 'Відстань (км)' на числовий
        df["Відстань (км)"] = pd.to_numeric(df["Відстань (км)"], errors="coerce")
        if df["Відстань (км)"].isnull().any():
            raise ValueError(
                "Стовпець 'Відстань (км)' повинен містити лише числові значення."
            )

        return df

    except FileNotFoundError:
        print(f"Помилка: Файл за адресою '{file_path}' не знайдено.")
    except ValueError as e:
        print(f"Помилка у даних: {e}")
    except Exception as e:
        print(f"Невідома помилка: {e}")


def create_distance_matrix(df):
    # Отримуємо унікальні міста
    cities = list(set(df["Назва міста1"].unique()) | set(df["Назва міста2"].unique()))
    n_cities = len(cities)

    # Створюємо словник для індексації міст
    city_to_idx = {city: idx for idx, city in enumerate(cities)}

    # Ініціалізуємо матрицю відстаней великими значеннями
    distances = np.full((n_cities, n_cities), np.inf)
    np.fill_diagonal(distances, 0)  # відстань від міста до себе = 0

    # Заповнюємо матрицю відомими відстанями
    for _, row in df.iterrows():
        city1_idx = city_to_idx[row["Назва міста1"]]
        city2_idx = city_to_idx[row["Назва міста2"]]
        distance = row["Відстань (км)"]
        distances[city1_idx, city2_idx] = distance
        distances[city2_idx, city1_idx] = distance

    # Алгоритм Флойда-Уоршелла для знаходження найкоротших шляхів
    for k in range(n_cities):
        for i in range(n_cities):
            for j in range(n_cities):
                if distances[i, k] != np.inf and distances[k, j] != np.inf:
                    distances[i, j] = min(
                        distances[i, j], distances[i, k] + distances[k, j]
                    )

    # Перевірка на наявність недосяжних міст
    if np.isinf(distances).any():
        print("Увага: Деякі міста не мають зв'язку між собою!")
        # Заміняємо inf на максимальне значення відстані
        max_distance = np.max(distances[~np.isinf(distances)])
        distances[np.isinf(distances)] = max_distance * 1.5

    return distances, cities


# Функція для кластеризації методом K-means
def kmeans_clustering(distances, cities, n_clusters):
    # Створюємо початкові центроїди випадковим чином з даних
    centroids = np.random.choice(
        range(distances.shape[0]), size=n_clusters, replace=False
    )
    centroids = distances[centroids]

    # Ініціалізуємо змінну для старих міток
    previous_labels = None

    while True:
        # Призначення точок до найближчого центроїду
        distances_to_centroids = np.array(
            [np.linalg.norm(distances - centroid, axis=1) for centroid in centroids]
        )
        labels = np.argmin(distances_to_centroids, axis=0)

        # Оновлення центроїдів
        new_centroids = np.array(
            [distances[labels == k].mean(axis=0) for k in range(n_clusters)]
        )

        # Оновлення центроїдів
        centroids = new_centroids

        # Перевірка стабільності кластерів
        if np.array_equal(labels, previous_labels):
            break

        # Оновлюємо попередні мітки
        previous_labels = labels

    # Return cities and their corresponding clusters
    city_clusters = dict(zip(cities, labels))

    return city_clusters


def clusters_to_nx_graph(city_clusters, cities, distances):
    """
    Створює граф NetworkX на основі кластерів, отриманих з K-Means.

    Args:
        city_clusters (dict): Словник з містами та їх класифікацією.
        cities (list): Список назв міст.
        distances (np.ndarray or list): Матриця відстаней між містами.

    Returns:
        nx.Graph: Об'єкт графа NetworkX.
    """
    G = nx.Graph()

    # adding nodes with info about cluster
    for city in cities:
        cluster = city_clusters.get(city)
        G.add_node(city, cluster=cluster)

    # Adding Edges
    for i, city1 in enumerate(cities):
        for j, city2 in enumerate(cities):
            if (
                city_clusters.get(city1) == city_clusters.get(city2) and i != j
            ):  # Only for cities in the same cluster
                weight = distances[i, j]
                G.add_edge(city1, city2, weight=weight)

    return G
