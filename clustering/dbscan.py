import numpy as np
from numpy.typing import NDArray


class DBScan:
    def __init__(self, epsilon: float = 1, min_samples: int = 4):
        self.epsilon = epsilon
        self.min_samples = min_samples

    def fit_predict(self, X):
        # Initialization of the cluster registry (-1 = outliers)
        clusters = np.full(X.shape[0], -1)

        # Compute distance matrix
        dist_matrix = np.sqrt(
            np.sum((X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2, axis=2)
        )

        # Monitor the cluster number (increase when a cluster is fully populated with its members)
        cluster_number = 0

        for i, point in enumerate(dist_matrix):
            # skip if the point is already in a cluster or not a core point
            if clusters[i] != -1 or not self._is_core(point):
                continue

            # Initialize the stack to expand the core point
            core_points = list()

            # Append the current point if it's a core point
            core_points.append((i, point))

            while len(core_points) > 0:
                idx, core_point = core_points.pop()

                clusters[idx] = cluster_number

                # Expand the node
                neighbors_indices = np.where(core_point <= self.epsilon)[0]

                for neighbor_index in neighbors_indices:
                    neighbor = dist_matrix[neighbor_index]

                    # compute if we have a not visited core point, affet cluster and add it to the stack
                    if clusters[neighbor_index] == -1:
                        if self._is_core(neighbor):
                            core_points.append((neighbor_index, neighbor))
                        else:
                            # Add the neighbor to the cluster as it wil not go into the loop again
                            clusters[neighbor_index] = cluster_number

            cluster_number += 1

        return clusters

    def _is_core(self, point: NDArray):
        if len(point[(point <= self.epsilon) & (point > 0)]) < self.min_samples:
            return False
        else:
            return True
