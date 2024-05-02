import unittest
import torch
from task1_Yu_Zhou import MinibatchKMeans  

class TestMinibatchKMeans(unittest.TestCase):

    def test_initial_centers(self):
        """check if the initial centers are correctly initialized"""
        torch.manual_seed(0)
        X = torch.randn(50, 2) 
        kmeans = MinibatchKMeans(k=3, batch_size=10)
        kmeans.fit(X)
        self.assertEqual(kmeans.centers.shape, (3, 2)) 

    def test_convergence_within_iter(self):
        """test if the algorithm converges within max_iter iterations"""
        torch.manual_seed(1)
        X = torch.randn(100, 2)
        kmeans = MinibatchKMeans(k=3, batch_size=10, max_iter=100)
        centers = kmeans.fit(X)
        self.assertTrue(centers is not None)  

    def test_center_movement(self):
        """test if the centers are updated after the first fit"""
        torch.manual_seed(2)
        X = torch.randn(100, 2)
        kmeans = MinibatchKMeans(k=3, batch_size=10)
        kmeans.fit(X)  # first fit
        initial_centers = kmeans.centers.clone()
        kmeans.fit(X)  # second fit
        final_centers = kmeans.centers
        # check if the centers are updated
        self.assertFalse(torch.equal(initial_centers, final_centers))

if __name__ == '__main__':
    unittest.main()
