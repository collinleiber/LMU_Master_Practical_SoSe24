import unittest
import numpy as np

from LMU_Master_Practical_SoSe24.practical.DeepClustering import kmeans


class TestKMeans(unittest.TestCase):

    def setUp(self):
        ## Random num_clusters and input dimensions
        self.K = np.random.randint(low=1, high=20)
        self.N = np.random.randint(low=20, high=400)
        self.d = np.random.randint(low=1, high=10)

        ## Random vectors of size d
        self.x = np.random.rand(self.d)
        self.y = np.random.rand(self.d)

        ## Kmean object with a random dataset
        self.X = np.random.rand(self.N, self.d)
        self.kmean = kmeans.KMeans(n_clusters=self.K)

    def tearDown(self):
        pass

    def test_pick_random_centroids(self):
        for centroid in self.kmean.pick_random_centroids():
            self.assertIn(centroid, self.X)

    def test_distances_to_centroids(self):
        pass

    def test_calculate_new_centroids(self):
        pass

    def test_fit(self):
        pass


if __name__ == '__main__':
    unittest.main()
