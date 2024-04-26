import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


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
            return (np.sum((x - y) ** 2)) ** (1 / len(x))
        else:
            raise ValueError("Inputs must have the same dimension!\n")

    def __init__(self, X, n_clusters, distance=euclidian):
        self.X = np.asarray(X)
        self.n_data: int = np.shape(self.X)[0]
        self.n_clusters: int = n_clusters
        self.distance: callable = distance

        ## Available after fitting the data
        self.labels: {tuple: np.ndarray} = {}
        self.centroids: list = []
        self.clusters: list = []

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
        dist = sorted([(centroid, self.distance(x, centroid))
                       for centroid in self.centroids], key=lambda tpl: tpl[1])
        centroid = dist[0][0]

        return centroid

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
        self.pick_random_centroids()

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
    def __init__(self, X: iter, n_clusters: int):
        super(KMeansSGD, self).__init__(X, n_clusters)

    def fit(self, max_epoch: int, visuals=False):
        pass


class KMeansMiniBatch(KMeans):
    def __init__(self, X: iter, n_clusters: int, batch_size: int):
        super(KMeansMiniBatch, self).__init__(X, n_clusters)
        self.batch_size: int = batch_size

    def fit(self, max_epoch: int, visuals=False):
        pass
