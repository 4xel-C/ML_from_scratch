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
    ):
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.n_negative_sample = n_negative_sample
        self.min_dist = min_dist

    def fit_transform(self, X: NDArray) -> NDArray:
        # =============== KNN computing

        # initialize the variables to store the indices of the knn and the distances
        knn_indices: NDArray = np.full((len(X), self.n_neighbors), -1)
        knn_distances: NDArray = np.full((len(X), self.n_neighbors), -1)

        # compute the distances using broadcasting
        distances = np.sqrt(
            np.sum((X[np.newaxis, :, :] - X[:, np.newaxis, :]) ** 2, axis=2)
        )
