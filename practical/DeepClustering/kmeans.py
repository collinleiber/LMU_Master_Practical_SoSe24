import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.datasets import make_blobs



class KMeans:
    """
    Standard k-Means Algorithm
    """

    @staticmethod
    def euclidian(x, y) -> float:
        """
        Calculates the Euclidian-distance of the two data points.
        """
        x = np.asarray(x)
        y = np.asarray(y)
        if len(x) == len(y):
            return np.linalg.norm((x - y))
        else:
            raise ValueError("Inputs must have the same dimension!\n")

    def __init__(self, data: iter, n_clusters, distance=euclidian):
        ## Initialize data
        self.X = np.asarray(data)
        self.n_data = np.shape(self.X)[0]
        self.n_clusters: int = n_clusters
        self.distance: callable = distance

        ## Available after fitting the data
        self.labels: {tuple: np.ndarray} = {}
        self.clusters: list = []
        self.centroids: list = []

    def pick_random_centroids(self) -> list:
        """
        Returns n_clusters sized list of random data points from the dataset X.
        """
        return [self.X[i] for i in np.random.choice(range(self.n_data), size=self.n_clusters, replace=False)]

    ## TODO:
    def get_closest_centroid(self, x: np.array) -> np.ndarray:
        """
        Calculates distances in a sorted list and returns the closest centroid.
        :param x: data point to be evaluated
        :return: the closest centroid to the point x
        """
        distances = [(centroid, self.distance(x, centroid)) for centroid in self.centroids]
        distances: sorted = distances.sort(key=lambda tpl: tpl[1])
        closest_centroid = distances[0]

        return closest_centroid


    def get_clusters(self) -> list:
        """
        Calculates the new clusters based on the current labels.
        :return: list of clusters, which are the list of data points of the cluster
        """
        clusters = []
        for centroid in self.centroids:
            cluster = []
            for x in self.labels.keys():
                if self.labels[x] == centroid:
                    point = np.asarray(x)
                    cluster.append(point)
            clusters.append(cluster)

        return clusters

    def get_centroids(self) -> list:
        """
        Finds mean values of the clusters
        """
        return [np.mean(cluster, axis=0) for cluster in self.clusters]

    def visualize(self, epoch=0):
        dim = self.X.shape[1] ## Dimension of each data point
        c_map = {centroid : i for i, centroid in enumerate(self.centroids)}
        colors = np.array([c_map[self.labels[x]] for x in self.X])

        ## 2D-Plots
        if dim == 2:
            plt.scatter(x=self.X[:, 0], y=self.X[:, 1], c=colors)
            plt.title(f"Clusters of epoch: {epoch}")
            plt.show()

        ## 3D-Plots
        if dim == 3:
            pass

    def fit(self, max_epoch: int, visuals=False):
        ## Initialize centroids randomly
        self.centroids = self.pick_random_centroids()

        for epoch in range(1, max_epoch):
            ## For each data point
            for x in self.X:
                ## Assign each point to the nearest centroid
                nearest_centroid = self.get_closest_centroid(x)
                self.labels[tuple(x)] = nearest_centroid

            ## Update clusters & centroids
            self.clusters = self.get_clusters()
            self.centroids = self.get_centroids()

            ## Visualize clusters
            if visuals:
                self.visualize()

        return self.labels, self.centroids


class KMeansSGD(KMeans):
    def __init__(self, data: iter, n_clusters: int):
        super(KMeansSGD, self).__init__(data, n_clusters)

    def fit(self, max_epoch: int, visuals=False):
        pass


class KMeansMiniBatch(KMeans):
    def __init__(self, data: iter, n_clusters: int, batch_size: int):
        super(KMeansMiniBatch, self).__init__(data, n_clusters)
        self.batch_size: int = batch_size

    def fit(self, max_epoch: int, visuals=False):

        pass


if __name__ == "__main__":
    X, y_true = make_blobs(n_samples=500, centers=5, cluster_std=0.60, random_state=0)
    # plt.scatter(X[:, 0], X[:, 1])
    # plt.show()

    kmeans = KMeans(X, n_clusters=5)
    print(kmeans.__dict__)

    kmeans.fit(max_epoch=5, visuals=True)