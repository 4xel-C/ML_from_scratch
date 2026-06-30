"""
t-SNE (t-Distributed Stochastic Neighbor Embedding) implementation.
See documentation for the full explainability.

The step are as follows:

- Compute the distance in the original space
- Compute the similarity and neighbor probabilities using the kernel gaussian and the binary search to match the perplexity (perplexity = 2^(H(Pi))
- Optimization loop:
    - Compute the probabilities using tstudent kernel in lower dimension with random initialized coordinates for points.
    - Compute the gradient (with respect to coordinates in lower dimension) using the loss' derivative (kullback-Leiber divergence) comparing gaussian probabilities and t-student probabilities in lower dimension.
    - Update the coordinates
"""

from typing import Tuple

import numpy as np
from numpy.typing import NDArray


class TNSE:
    def __init__(
        self,
        perplexity: float = 30,
        n_dimensions: int = 2,
        learning_rate: float = 200,
        tolerance: float = 1e-7,
        max_iterations: int = 1000,
    ):
        self.perplexity = perplexity
        self.n_dimensions = n_dimensions
        self.learning_rate = learning_rate
        self.tolerance = tolerance
        self.max_iterations = max_iterations

    def fit_transform(self, X: NDArray) -> NDArray: ...

    def _compute_distances(self, X: NDArray) -> NDArray:
        return np.sum((X[np.newaxis, :, :] - X[:, np.newaxis, :]) ** 2, axis=2)

    def _compute_perplexity(self, d: NDArray, sigma: float) -> Tuple[NDArray, float]:
        """Compute the perplexity for a point.

        Args:
            d (NDArray): The distances to all the other points (line of distance matrix)
            sigmai (float): The standart deviation used to compute the gaussian kernel

        Returns:
            Tuple[NDArray, float]: (Neighbor probabilities for the point to all other points, perplexity)
        """

        similarity = np.exp(-d / (2 * sigma**2))

        probabilities = similarity / np.sum(similarity)
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-15))
        perplexity = 2**entropy

        return probabilities, perplexity

    # To be implemented
    def _binary_search_sigma(
        self,
        d: NDArray,
        sigma_left: float,
        sigma_right: float,
        perplexity_target: float,
    ): ...
