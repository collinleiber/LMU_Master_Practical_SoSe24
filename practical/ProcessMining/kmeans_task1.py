from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np


def k_means_scratch(data, n_clusters=2, max_iterations=100):
    # Step 1: Randomly initialize the centroids
    centroids = data[:n_clusters]
    clusters = []

    for _ in range(max_iterations):
        # Step 2: Assign each data point to the closest centroid
        clusters = [[] for _ in range(n_clusters)]
        for point in data:
            distances = [sum((point - centroid)**2) for centroid in centroids]
            closest_centroid = distances.index(min(distances))
            clusters[closest_centroid].append(point)

        # Step 3: Recalculate the centroids as the mean of all data points in the cluster
        for i, cluster in enumerate(clusters):
            centroids[i] = sum(cluster) / len(cluster)

    # Generate labels for the data points
    labels = [i for i, cluster in enumerate(clusters) for _ in cluster]

    visualize_clusters(data=data, labels=labels, centroids=centroids)
    return centroids, clusters


def k_means_lib(data: np.ndarray, n_clusters: int, init_centroids: np.ndarray = None) -> KMeans:
    # When centroids are not provided, the KMeans algorithm will initialize them randomly
    # -> result not deterministic / testable.
    if init_centroids is None:
        kmeans = KMeans(n_clusters=n_clusters)
    # Initialize centroids with provided values
    # -> only use for testing.
    else:
        if init_centroids.shape[0] != n_clusters:
            raise ValueError("Number of initial centroids must match the number of clusters.")
        kmeans = KMeans(n_clusters=n_clusters, init=init_centroids)
    kmeans.fit(data)
    visualize_clusters(data, kmeans.labels_, kmeans.cluster_centers_)
    return kmeans


def visualize_clusters(data, labels, centroids):
    # Create scatter plot of data, each point with cluster label's color
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis')
    # Plot centroids of clusters
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X')
    plt.show()

