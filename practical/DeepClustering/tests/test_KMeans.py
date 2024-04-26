import unittest
import numpy as np

from LMU_Master_Practical_SoSe24.practical.DeepClustering.KMeans import KMeans


class TestKMeans(unittest.TestCase):

    def setUp(self):
        ## Random num_clusters and input dimensions
        K = np.random.randint(low=1, high=20)
        N = np.random.randint(low=20, high=400)
        d = np.random.randint(low=1, high=10)

        ## Random vectors of size d
        self.x = np.random.rand(d)
        self.y = np.random.rand(d)

        ## Kmean object with a random dataset
        X = np.random.rand(N, d)
        self.kmean = KMeans(X=X, n_clusters=K)


    def tearDown(self):
        pass

    def test_euclidian(self):

        dist = self.kmean.euclidian(self.x, self.y)

        self.assertEquals(dist, np.linalg.norm((self.x - self.y)))
        self.assertGreaterEqual(dist, 0)

    def test_pick_random_centroids(self):
        pass

    def test_get_clusters(self):
        pass

    def test_get_centroids(self):
        pass

    def test_visualize(self):
        pass

    def test_fit(self):
        pass

if __name__ == '__main__':
    TestKMeans.run()
