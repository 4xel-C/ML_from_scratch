"""
t-SNE (t-Distributed Stochastic Neighbor Embedding) implementation.
See documentation for the full explainability.

The step are as follows:

- Compute the distance in the original space
- Compute the similarity and neighbor probabilities using the kernel gaussian and the binary search to match the perplexity (perplexity = 2^(H(Pi))
- Symmetrize the similarity matrix and normalize over the full matrix (distribution by pair of points)
- Optimization loop:
    - Compute the probabilities using tstudent kernel in lower dimension with random initialized coordinates for points.
    - Compute the gradient (with respect to coordinates in lower dimension) using the loss' derivative (kullback-Leiber divergence) comparing gaussian probabilities and t-student probabilities in lower dimension.
    - Update the coordinates
"""

from typing import Tuple

import numpy as np
from numpy.typing import NDArray


class TSNE:
    def __init__(
        self,
        perplexity: float = 30,
        n_dimensions: int = 2,
        learning_rate: float = 200,
        tolerance: float = 1e-7,
        max_iterations: int = 1000,
        binary_max_iterations: int = 50,
    ):
        self.perplexity = perplexity
        self.n_dimensions = n_dimensions
        self.learning_rate = learning_rate
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.binary_max_iterations = binary_max_iterations

    def fit_transform(self, X: NDArray) -> NDArray: ...

    def _compute_distances(self, X: NDArray) -> NDArray:
        return np.sum((X[np.newaxis, :, :] - X[:, np.newaxis, :]) ** 2, axis=2)

    def _compute_probas_perplexity(
        self, d: NDArray, sigma: float
    ) -> Tuple[NDArray, float]:
        """Compute the perplexity for a point.

        Args:
            d (NDArray): For a point, the distances to all the other points (line of distance matrix)
            sigmai (float): The standart deviation used to compute the gaussian kernel

        Returns:
            Tuple[NDArray, float]: (Neighbor probabilities for the point to all other points, perplexity)
        """

        similarity = np.exp(-d / (2 * sigma**2))

        probabilities = similarity / np.sum(similarity)
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-15))
        perplexity = 2**entropy

        return probabilities, perplexity

    def _binary_search_sigma_probas(
        self,
        d: NDArray,
        sigma_left: float,
        sigma_right: float,
        perplexity_target: float,
        iteration=0,
    ) -> Tuple[float, NDArray]:
        """Compute the binary serach to find the sigma value for a point.

        Args:
            d (NDArray): The distance from the point to all other points
            sigma_left (float): the left value for sigmae
            sigma_right (float): the right value for sigma
            perplexity_target (float): The perplexity we want to reach
            iteration (int, optional): The current iteration. Defaults to 0.

        Returns:
            Tuple[float, NDArray]: The sigma, The probabilities for the current point.
        """
        mid = (sigma_right + sigma_left) / 2

        probabilities, perplexity_mid = self._compute_probas_perplexity(d, mid)

        if (
            np.abs(perplexity_mid - perplexity_target) <= self.tolerance
            or iteration == self.binary_max_iterations
        ):
            return (mid, probabilities)

        # If perxiplity too low -> We have to increase entropy, and increase the variance, to have more sparse probabilities
        elif perplexity_mid < perplexity_target:
            return self._binary_search_sigma_probas(
                d, mid, sigma_right, perplexity_target, iteration=iteration + 1
            )

        # If perplexity too high, we have to reduce the perplexity, decrease entropy, thus decrease standart deviation to have sharper probabilities.
        else:
            return self._binary_search_sigma_probas(
                d, sigma_left, mid, perplexity_target, iteration=iteration + 1
            )

    def _computy_high_dimensionality_proba(self, D: NDArray) -> NDArray:
        """Compute the neighborhood probabilities for the original dimensionnal space.

        Args:
            D (NDArray): The distance matrix

        Returns:
            NDArray: The probabilities matrix for high dimensionnality.
        """
        # Ensure the diagonal has infinite value for the probability calculation (probability for a point to be neighbor to itself = 0)
        sigma_left = np.sqrt(max(D))
        np.fill_diagonal(D, np.inf)

        probas_matrix = np.zeros_like(D)

        for point_idx in range(len(D)):
            _, probas_matrix[point_idx, :] = self._binary_search_sigma_probas(
                D[point_idx, :], 1e-10, sigma_left, self.perplexity, 0
            )

        # Symetrization et generation de la distribution
        final_matrix = (probas_matrix + probas_matrix.T) / (2 * len(D))

        return final_matrix

    def _compute_low_dimensionnality_probas(self, Y: NDArray) -> NDArray:
        """Compute probabilities in lower dimensionnal space q.

        Args:
            Y (NDArray): The coordinates of each point in lower dimensional space

        Returns:
            NDArray: The neighborhood probabilites in reduced dimensionnality space.
        """
        D_low = self._compute_distances(Y)

        # Diagonal correction so the similarity of a point with itself worth nothing.
        np.fill_diagonal(D_low, np.inf)

        similarities = (1 + D_low) ** -1

        probas_q = similarities / np.sum(similarities)

        return probas_q
