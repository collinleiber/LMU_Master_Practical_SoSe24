from practical.ProcessMining.k_means import kMeans




def test_k_means ():
    data = [(2,3),(4,8),(8,7),(10,3),(5,7),(1,3),(3,11),(9,2),(8,4),(5,1)]
    generated_clusters = kMeans(data,2)
    assert len(generated_clusters) == 2
    assert len(data) == sum(len(cluster) for cluster in generated_clusters)
    print("All tests successful")
    return generated_clusters
