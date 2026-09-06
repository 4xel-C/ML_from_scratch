"""
UMAP (Uniform Manifold Approximation and Projection) implementation from scratch.

How to compute UMAP:

1) Compute the neighborhood graph with adpated weights
2) Randomly distribute the points in the reduced space
3) Optimize using the binary cross-entropy to rebuild the graph
"""

from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray


class UMAP:
    def __init__(
        self,
        n_neighbors: int,
        n_components: int,
        learning_rate: float,
        n_epochs: int,
        n_negative_sample: int,
        min_dist: float,
    ) -> None:
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.n_negative_sample = n_negative_sample
        self.min_dist = min_dist

    def fit_transform(self, X: NDArray) -> NDArray:
        # =============== KNN computing

        # initialize the variables to store the indices of the knn and the distances
        knn_indices: NDArray
        knn_distances: NDArray
        rho: NDArray

        # compute the distances using broadcasting
        distances: NDArray = np.sqrt(
            np.sum((X[np.newaxis, :, :] - X[:, np.newaxis, :]) ** 2, axis=2)
        )

        # find the knns, remove the proximity to self and get the n nearest neighbors
        knn_indices = np.argsort(distances, axis=1)[:, 1 : self.n_neighbors + 1]

        # get the distances from the adjacency matrix
        # np.arange(len(distances))[:, None] create a column of 1, 2, 3...n
        # knn_indices is then inexed using this indices to get the correct line of knn
        knn_distances = distances[np.arange(len(distances))[:, None], knn_indices]

        # ============== Computing probabilities in the original space (Weights of the graph)
        # Get vector rho (distance to the nearest neighbor to shift it's distance to 1)
        rho = knn_distances[:, 0]

        # apply the difference to all distances
        knn_dist_rho = knn_distances - rho[:, np.newaxis]

        # Compute the vector of sigma for each points using binary search
        sigmas = np.apply_along_axis(knn_dist_rho)

        print(knn_dist_rho)

    def _binary_search_sigma(
        self,
        distances_rho: NDArray,
        target_entropy: float,
        left_sigma: float,
        right_sigma: float,
        max_iterations: int = 50,
        tol: float = 0.1,
        current_iteration: int = 0,
    ) -> float:
        """Binary search to find the sigma generating maximum entropy on the gaussian distribution.
        Applied for each point on the distances vector.

        Args:
            distances_rho (NDArray): Vector of distances for the concerned point, already shifted by rho value
            target_entropy (float): The maximum possible entropy for a uniform distribution on K points
            left_sigma (float): left boundary of sigma value
            right_sigma (float): right boundary of sigma value

        Returns:
            float: The optimized sigma
        """

        # compute the mid value of sigma
        mid_sigma = (right_sigma + left_sigma) / 2

        if current_iteration == max_iterations:
            print("Max iteration reached ! The sigma search did not converge.")
            return mid_sigma

        probabilities: NDArray = np.exp(-(distances_rho) / mid_sigma)

        # compute the entropy for sigma
        H_mid = np.sum(-probabilities * np.log2(probabilities))

        # If we are inside the tolerance window: return the sigma
        if abs(H_mid - target_entropy) < tol:
            return mid_sigma

        # If not enough entropy, increase the sigma (to soften the distribution)
        elif H_mid < target_entropy:
            return self._binary_search_sigma(
                distances_rho,
                target_entropy,
                left_sigma=mid_sigma,
                right_sigma=right_sigma,
                max_iterations=max_iterations,
                tol=tol,
                current_iteration=current_iteration + 1,
            )

        # If too much entropy, decrease the sigma to sharpen the distribution
        else:
            return self._binary_search_sigma(
                distances_rho,
                target_entropy,
                left_sigma=left_sigma,
                right_sigma=mid_sigma,
                max_iterations=max_iterations,
                tol=tol,
                current_iteration=current_iteration + 1,
            )


if __name__ == "__main__":
    engine = UMAP(
        n_neighbors=10,
        n_components=2,
        learning_rate=0.01,
        n_epochs=100,
        n_negative_sample=20,
        min_dist=10,
    )

    X = np.array(
        [
            [12, 5, 8, 3, 17],
            [4, 21, 6, 14, 9],
            [18, 2, 15, 7, 11],
            [5, 13, 20, 1, 16],
            [10, 8, 3, 19, 22],
        ]
    )

    engine.fit_transform(X)
