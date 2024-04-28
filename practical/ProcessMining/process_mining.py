from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np


def calculate_k_means(data: np.ndarray, n_clusters: int, init_centroids: np.ndarray = None) -> KMeans:
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

