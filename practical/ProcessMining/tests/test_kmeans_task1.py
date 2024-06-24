from practical.ProcessMining.kmeans_task1 import k_means_lib, k_means_scratch
from sklearn.cluster import KMeans
import numpy as np


def test_case_random_centroids(clusters: int = 2) -> KMeans:
    data = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
    return k_means_lib(data, clusters)


def test_case_given_centroids(clusters: int = 2) -> KMeans:
    data = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
    # Generate evenly spaced centroid coordinates from 0 to max_val
    x_coords = np.linspace(0, np.max(data[:, 0]), clusters)
    y_coords = np.linspace(0, np.max(data[:, 1]), clusters)

    init_centroids = np.array(list(zip(x_coords, y_coords)))

    assert (init_centroids.size / 2) == clusters
    return k_means_lib(data, clusters, init_centroids)


def test_k_means():
    # Test cases
    cases = {
        'random1': test_case_random_centroids(),
        'random2': test_case_random_centroids(clusters=4),
        'static1': test_case_given_centroids(),
        'static2': test_case_given_centroids(clusters=6),
    }

    amount_clusters_matches_unique_labels(cases['random1'])
    amount_clusters_matches_unique_labels(cases['random2'])
    amount_clusters_matches_unique_labels(cases['static2'])

    deterministic_results_match_expected(result=cases['static1'],
                                              expected_labels=np.array([0, 1, 0, 1, 1, 0]))
    deterministic_results_match_expected(result=cases['static2'],
                                              expected_labels=np.array([2, 3, 1, 4, 5, 0]))


def amount_clusters_matches_unique_labels(result: KMeans):
    unique_labels = np.unique(result.labels_).size
    assert result.n_clusters == unique_labels, \
        f"Number of clusters ({result.n_clusters}) does not match unique labels ({unique_labels})."


def deterministic_results_match_expected(result: KMeans, expected_labels: np.ndarray):
    assert np.array_equal(result.labels_, expected_labels), \
        f"Expected labels {expected_labels} do not match result labels {result.labels_}."


def test_k_means_scratch():
    data = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
    clusters = 2

    centroids, clusters_result = k_means_scratch(data, clusters)

    # Check if the number of clusters is correct
    assert len(clusters_result) == clusters, \
        f"Expected number of clusters ({clusters}) does not match result ({len(clusters_result)})."

    # Check if all data points are assigned to a cluster
    all_points = [point for cluster in clusters_result for point in cluster]
    assert np.array_equal(np.sort(data, axis=0), np.sort(all_points, axis=0)), \
        "Not all data points are assigned to a cluster."

    # Check if centroids are in the correct range
    for centroid in centroids:
        assert np.min(data) <= centroid[0] <= np.max(data), \
            f"Centroid {centroid} is not in the correct range."
        assert np.min(data) <= centroid[1] <= np.max(data), \
            f"Centroid {centroid} is not in the correct range."
