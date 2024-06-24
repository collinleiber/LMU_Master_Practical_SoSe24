import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class KMeans:
    def __init__(self, k, max_iter=300):
        self.k = k
        self.max_iter = max_iter
        self.centroids = None

    # def calc_distance(self, data, centroids):
    #     distances = []
    #     for point in data:
    #         diff = np.tile(point, (self.k, 1)) - centroids
    #         squared_diff = diff ** 2
    #         squared_dist = np.sum(squared_diff, axis=1)
    #         distance = np.sqrt(squared_dist)
    #         distances.append(distance)
    #     return np.array(distances)
    @staticmethod
    def calc_distance(x1, x2):
        return np.linalg.norm(x1 - x2)

    def classify(self, data, centroids):
        distances = np.zeros((len(data), self.k))
        for i, point in enumerate(data):
            for j, centroid in enumerate(centroids):
                distances[i, j] = self.calc_distance(point, centroid)
        min_dist_indices = np.argmin(distances, axis=1)
        new_centroids = pd.DataFrame(data).groupby(min_dist_indices).mean().values
        changed = new_centroids - centroids
        return changed, new_centroids, min_dist_indices

    def fit(self, data):
        self.centroids = np.array(random.sample(data, self.k))
        for iteration in range(self.max_iter):
            changed, new_centroids, min_dist_indices = self.classify(data, self.centroids)
            print(f"Iteration {iteration + 1}")
            for cluster_index in range(self.k):
                cluster_points = [data[i] for i in range(len(data)) if min_dist_indices[i] == cluster_index]
                print(f"Cluster {cluster_index}: {cluster_points}")
            if not np.any(changed):
                break
            self.centroids = new_centroids

        cluster = [[] for _ in range(self.k)]
        distances = np.zeros((len(data), self.k))
        for i, point in enumerate(data):
            for j, centroid in enumerate(self.centroids):
                distances[i, j] = self.calc_distance(point, centroid)
        min_dist_indices = np.argmin(distances, axis=1)
        for idx, cluster_idx in enumerate(min_dist_indices):
            cluster[cluster_idx].append(data[idx])

        return self.centroids, cluster, min_dist_indices

def create_dataset():
    return [[1.0, 1.0], [1.0, 2.0], [2.0, 1.0], [2.0, 2.0], [2.0, 6.0],
            [3.0, 1.0], [3.0, 7.6], [5.0, 6.0], [6.0, 5.0], [6.0, 6.0],
            [6.0, 7.0], [6.0, 8.0], [7.0, 8.0], [8.0, 8.0]]


if __name__ == '__main__':
    dataset = create_dataset()
    kmeans = KMeans(k=2)
    centroids, clusters, min_dist_indices = kmeans.fit(dataset)
    print('Centroids:', centroids)
    print('Clusters:', clusters)

    colors = ['green', 'blue']
    for idx, point in enumerate(dataset):
        plt.scatter(point[0], point[1], marker='o', color=colors[min_dist_indices[idx]], s=40)
    for centroid in centroids:
        plt.scatter(centroid[0], centroid[1], marker='x', color='red', s=50,
                    label='Centroid' if np.array_equal(centroid, centroids[0]) else "")

    plt.legend()
    plt.show()