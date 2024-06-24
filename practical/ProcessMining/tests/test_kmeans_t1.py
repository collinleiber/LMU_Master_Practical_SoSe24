import unittest
import numpy as np
from practical.ProcessMining.kmeans_t1 import KMeans


class TestKMeans(unittest.TestCase):
    def setUp(self):
        self.dataset = [[1, 1], [1, 2], [2, 1], [6, 4], [6, 3], [5, 4]]
        self.kmeans = KMeans(k=2)

    def test_calc_distance(self):
        dist = self.kmeans.calc_distance(np.array([1, 1]), np.array([4, 5]))
        self.assertAlmostEqual(dist, 5.0, places=5)

    def test_classify(self):
        centroids = np.array([[1, 1], [6, 4]])
        _, new_centroids, min_dist_indices = self.kmeans.classify(self.dataset, centroids)
        expected_new_centroids = np.array([[1.33333333, 1.33333333], [5.66666667, 3.66666667]])
        np.testing.assert_almost_equal(new_centroids, expected_new_centroids, decimal=5)
        self.assertEqual(list(min_dist_indices), [0, 0, 0, 1, 1, 1])

    def test_fit(self):
        centroids, clusters, min_dist_indices = self.kmeans.fit(self.dataset)
        self.assertEqual(len(centroids), 2)
        self.assertEqual(len(clusters), 2)
        self.assertTrue(all(len(cluster) > 0 for cluster in clusters))


if __name__ == '__main__':
    unittest.main()
