import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

from KMeans import KMeans

X, y_true = make_blobs(n_samples=500, centers=5, cluster_std=0.60, random_state=0)
plt.scatter(X[:, 0], X[:, 1])
plt.show()

kmeans = KMeans(X, n_clusters=5)
kmeans.fit(max_epoch=5, visuals=True)
