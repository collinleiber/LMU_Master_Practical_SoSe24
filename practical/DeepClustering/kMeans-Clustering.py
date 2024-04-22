import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, in_dim: int, h_dim: int, z_dim: int):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(in_dim, h_dim)
        self.linear2 = nn.Linear(h_dim, z_dim)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu((self.linear2(x)))
        return x


class Decoder(nn.Module):
    def __init__(self, z_dim: int, h_dim: int, out_dim: int):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(z_dim, h_dim)
        self.linear2 = nn.Linear(h_dim, out_dim)

    def forward(self, z):
        z = F.relu(self.linear1(z))
        z = F.relu(self.linear2(z))
        return z


class Autoencoder(nn.Module):
    def __init__(self, in_dim, h_dim, z_dim):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(in_dim, h_dim, z_dim)
        self.decoder = Decoder(z_dim, h_dim, in_dim)

    def forward(self, x):
        z = self.encoder(x)
        x_rec = self.decoder(z)
        return x_rec


class KMeans:
    def __init__(self, n_clusters: int, centroids=None):
        self.n_clusters = n_clusters
        self.centroids = centroids
        self.random_gen = np.random.choice

    @staticmethod
    def euclidian_distance(x, y):
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        if len(x) == len(y):
            return (np.sum((x - y) ** 2))**(1/len(x))
        else:
            raise ValueError("Inputs must have the same dimension!\n")

    def _pick_random_centroids(self, X):
        pass

    def _plot_clustering(self):
        pass

    def fit(self, X, max_iter):
        pass
