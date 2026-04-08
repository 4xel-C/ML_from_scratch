import numpy as np
from numpy.typing import NDArray


class KMeans:
    def __init__(self, k: int, max_iterations: int = 100, tol: float = 1e-4):
        self.k = k
        self.max_iterations = max_iterations
        self.tol = tol
        self.cluster = None
        self.is_fitted = None

    def fit(self, X: NDArray) -> NDArray:
        n = X.shape[0]

        # Initialize centroids using random points
        centroids_index = self._select_k_points(X, self.k)

        # Get the centroids X (n, 1, p)  Centroids (1, k, p)
        self.centroids = X[centroids_index]

        distances = np.zeros((len(X), self.k))
        iteration = 0

        while iteration < self.max_iterations:
            # compute the distance of all point to the centroids, axis=2 for the feature p
            distances = np.linalg.norm(
                X[:, np.newaxis, :] - self.centroids[np.newaxis, :, :], axis=2
            )

            # Assign each points de the closest cluster
            self.clusters = distances.argmin(axis=1)

            # update the centroids
            new_centroids = np.zeros_like(self.centroids)

            for i in range(self.k):
                new_centroids[i] = np.mean(X[self.clusters == i], axis=0)

            # Check if we have a modification of the cendroids
            delta = np.sum(np.linalg.norm(new_centroids - self.centroids, axis=1))

            if delta < self.tol:
                print("Algorithm converged!")
                break

            self.centroids = new_centroids.copy()

            iteration += 1

        self.is_fitted = True

        return self.clusters

    def predict(self, X: NDArray) -> NDArray:
        # Compute the distance to the clusters
        distances = np.linalg.norm(
            X[:, np.newaxis, :] - self.centroids[np.newaxis, :, :], axis=2
        )

        clusters = np.argmin(distances, axis=1)

        return clusters

    def _select_k_points(self, X, k):
        # Compute the first element randomly
        centroids_idx = np.array([np.random.choice(len(X))])

        while len(centroids_idx) <= k:
            # compute the distance of all points to the centroids
            distances = np.linalg.norm(
                X[:, np.newaxis, :] - X[centroids_idx][np.newaxis, :, :], axis=2
            )

            # take the minimum distance to each centroids
            min_distances = distances.min(axis=1) ** 2

            # Normalize to optain a probability -> the farthest a point is from the centroids,
            # the most probable it is to be picked as a new centroids
            # centroids being with a weight of 0
            probs = min_distances / sum(min_distances)

            # Choose a new centroids
            new_centroid = np.array([np.random.choice(len(X), p=probs)])

            # append the new centrod to the result array
            centroids_idx = np.hstack([centroids_idx, new_centroid])

        return centroids_idx
