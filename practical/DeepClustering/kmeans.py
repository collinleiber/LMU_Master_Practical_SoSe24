import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.datasets import make_blobs
from scipy.spatial import distance


class KMeans:
    """
    Standard k-Means Algorithm
    """

    def __init__(self, n_clusters, distance=distance.euclidean):
        self.n_clusters: int = n_clusters
        self.distance: callable = distance
        self.centroids: [np.array] = []
        self.labels = None

    def pick_random_centroids(self, X) -> [np.ndarray]:
        """
        Returns n_clusters sized list of random data points from the dataset X.
        """
        n_data = np.shape(X)[0]
        return [X[i] for i in np.random.choice(range(n_data), size=self.n_clusters, replace=False)]

    def distances_to_centroids(self, x: np.array) -> [float]:
        """
        Calculates distances of the point x to the cluster centroids and returns them in a list (unsorted).
        """
        return [self.distance(x, centroid) for centroid in self.centroids]

    def calculate_new_centroids(self, X) -> [np.ndarray]:
        """
        Finds mean values of the clusters
        """
        return [np.mean(X[np.argwhere(self.labels == label)], axis=0)[0] for label in range(self.n_clusters)]

    def visualize(self, X, epoch=0):
        plt.scatter(X[:, 0], X[:, 1], c=self.labels)
        plt.title(f"Epoch: {epoch}")
        plt.show()

    def fit(self, X: iter, max_epoch: int, show=False):
        ## Setup
        X = np.asarray(X)
        n, d = np.shape(X)

        ## Initialize centroids randomly
        self.centroids = self.pick_random_centroids(X)

        ## Initialize array of labels
        self.labels = np.empty(n)

        for epoch in range(1, max_epoch + 1):
            print(epoch)
            ## Assign each point to the nearest centroid
            for i, x in enumerate(X):
                distances = self.distances_to_centroids(x)
                self.labels[i] = np.argmin(distances)

            if show:
                self.visualize(X, epoch)

            ## Update centroids
            new_centroids = self.calculate_new_centroids(X)
            if np.array_equal(new_centroids, self.centroids):
                # Converges if labels don't change
                break
            else:
                self.centroids = new_centroids

        print("Done!")
        return self.labels


class KMeansSGD(KMeans):
    pass


class KMeansMiniBatch(KMeans):
    pass


if __name__ == "__main__":
    X, y_true = make_blobs(n_samples=500, centers=4, cluster_std=0.60, random_state=0)
    plt.scatter(X[:, 0], X[:, 1])
    plt.show()

    kmeans = KMeans(10)
    kmeans.fit(X, max_epoch=30, show=True)
