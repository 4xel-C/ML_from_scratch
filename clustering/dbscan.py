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


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    import numpy as np
    from sklearn.cluster import DBSCAN as SklearnDBSCAN
    from sklearn.datasets import make_blobs
    from sklearn.metrics import adjusted_rand_score

    X, y_true = make_blobs(n_samples=200, centers=3, cluster_std=0.5, random_state=42)

    custom = DBScan(epsilon=0.8, min_samples=4)
    pred_custom = custom.fit_predict(X)

    sk = SklearnDBSCAN(eps=0.8, min_samples=4)
    pred_sk = sk.fit_predict(X)

    print("=== DBSCAN comparison ===")
    print(f"Custom ARI:  {adjusted_rand_score(y_true, pred_custom):.4f}")
    print(f"Sklearn ARI: {adjusted_rand_score(y_true, pred_sk):.4f}")
    print(f"Custom clusters found:  {len(set(pred_custom)) - (1 if -1 in pred_custom else 0)}")
    print(f"Sklearn clusters found: {len(set(pred_sk)) - (1 if -1 in pred_sk else 0)}")
