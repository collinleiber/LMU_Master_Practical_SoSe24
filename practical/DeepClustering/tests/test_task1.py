import torch
from practical.DeepClustering.task1 import MiniBatchKMeans

def test_mini_batch_k_means():

    X = torch.tensor([[1, 1], [1, 2], [2, 1], [10, 10], [10, 11], [11, 10]], dtype=torch.float)

    # Create and fit the model
    model = MiniBatchKMeans(n_clusters=2, batch_size=2, max_iter=100)
    model.fit(X)

    # Predict the cluster labels
    labels = model.predict(X)

    # Check that the model has learned the correct clusters
    assert (labels[:3] == labels[0]).all()
    assert (labels[3:] == labels[3]).all()
    assert (labels[0] != labels[3]).all()