import torch

class MiniBatchKMeans:
    def __init__(self, n_clusters: int, batch_size: int, max_iter: int):
        self.n_clusters = n_clusters
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.centroids = None

    def fit(self, X):
        # Initialize centroids randomly from the data points
        rand_indices = torch.randint(0, X.shape[0], (self.n_clusters,))
        self.centroids = X[rand_indices]

        for _ in range(self.max_iter):
            # Find a random batch
            batch_indices = torch.randint(0, X.shape[0], (self.batch_size,))
            batch = X[batch_indices]

            # Compute distances from the batch to the centroids
            distances = self._compute_distances(batch)

            # Update
            self._update_centroids(batch, distances)

    def predict(self, X):
        distances = self._compute_distances(X)
        return torch.argmin(distances, dim=1)

    def _compute_distances(self, X):
        return torch.cdist(X, self.centroids)

    def _update_centroids(self, X, distances):
        cluster_labels = torch.argmin(distances, dim=1)
        for i in range(self.n_clusters):
            points_in_cluster = X[cluster_labels == i]
            if points_in_cluster.shape[0] > 0:
                self.centroids[i] = points_in_cluster.mean(dim=0)
