from typing import Iterator

import numpy as np

import torch
import torch.nn as nn
import torch.utils.data.dataloader
import torch.optim

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.datasets import make_blobs
from scipy.spatial import distance
from torch.nn import Parameter


class KMeans:
    """
    Standard k-Means Algorithm
    """

    def __init__(self, n_cluster, distance=distance.euclidean):
        self.n_cluster: int = n_cluster
        self.distance: callable = distance
        self.centroids: [np.array] = []
        self.labels = None

    def pick_random_centroids(self, X) -> [np.ndarray]:
        """
        Returns n_clusters sized list of random data points from the dataset X.
        """
        n_data = np.shape(X)[0]
        return [X[i] for i in np.random.choice(range(n_data), size=self.n_cluster, replace=False)]

    def distances_to_centroids(self, x: np.array) -> [float]:
        """
        Calculates distances of the point x to the cluster centroids and returns them in a list (unsorted).
        """
        return [self.distance(x, centroid) for centroid in self.centroids]

    def calculate_new_centroids(self, X) -> [np.ndarray]:
        """
        Finds mean values of the clusters
        """
        return [np.mean(X[np.argwhere(self.labels == label)], axis=0)[0] for label in range(self.n_cluster)]

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
        return self.labels, self.centroids



class MiniBatchKmeans(nn.Module):
    def __init__(self, n_cluster):
        super(MiniBatchKmeans, self).__init__()
        self.kmean = KMeans(n_cluster)

    def forward(self, X):
        labels, centroids = self.kmean.fit(X, 1)
        return centroids


def fit(X, n_cluster, max_epoch, batch_size):

    model = MiniBatchKmeans(n_cluster)
    train_data = torch.utils.data.DataLoader(dataset=X, batch_size=batch_size, shuffle=True)
    params = nn.Parameter(torch.tensor(model.kmean.centroids))
    optimizer = torch.optim.Adam(params)
    criterion = nn.MSELoss

    for epoch in range(max_epoch):
        for i, mb in enumerate(train_data):
            optimizer.zero_grad()

            pass
