import torch
import numpy as np

class MinibatchKMeans:
    def __init__(self, k, batch_size, max_iter=100):
        self.k = k
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.centers = None

    
    def fit(self, X):
        # randomly initialize the centers
        initial_indices = np.random.choice(len(X), self.k, replace=False)
        self.centers = X[initial_indices]

        for _ in range(self.max_iter):
            # mini-batch sampling
            indices = np.random.choice(len(X), self.batch_size, replace=False)
            minibatch = X[indices]

            # compute distances and assign points to clusters
            distances = torch.cdist(minibatch, self.centers)
            assignments = torch.argmin(distances, dim=1)

            # update the centers
            new_centers = torch.clone(self.centers)
            for i in range(self.k):
                # update the center only if there are points assigned to it
                cluster_points = minibatch[assignments == i]
                if len(cluster_points) > 0:
                    new_centers[i] = torch.mean(cluster_points, dim=0)

            # check for convergence
            if torch.allclose(self.centers, new_centers, atol=1e-4):
                break

            self.centers = new_centers

        return self.centers



