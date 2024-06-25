import numpy as np


def kmeans_implementation(x, k, max_iters=100):
    # Select k random samples from X as the initial centroids
    centroids = x[np.random.choice(range(len(x)), k, replace=False)]
    labels = np.zeros(len(x))  # Initialize labels for each point in X

    for _ in range(max_iters):
        old_labels = labels.copy()    # Copy labels from the previous iteration to check for convergence later

        # Assignment phase: Assign each data point to the nearest centroid
        for i, point in enumerate(x):
            # Calculate the squared Euclidean distance from this point to each centroid
            labels[i] = np.argmin([np.linalg.norm(point - centroid) ** 2 for centroid in centroids])
            # Assign the closest centroid's index to the label of this point

        # Update phase: Update the centroids to the mean of the points assigned to them
        for j in range(k):
            # extract the points assigned to this centroid j
            points = x[labels == j]
            if points.any():  # Check if there are any points assigned to this centroid
                centroids[j] = np.mean(points, axis=0)  # Update the centroid to the mean of the points

        # Convergence check: If labels have not changed, stop the iteration
        if np.all(labels == old_labels):
            break

    return labels, centroids
