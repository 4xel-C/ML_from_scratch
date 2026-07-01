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

    def fit_transform(self, X: NDArray) -> NDArray:
        n = len(X)

        D = self._compute_distances(X)
        P = self._computy_high_dimensionality_proba(D)

        # Initialize Y (low dimensionality coordinates randomly with normal distribution) -> keep the value small
        # to have a high loss function as the loss function dosn't penalize distant points that much.
        Y = np.random.randn(n, self.n_dimensions) * 1e-4

        # initialization of the loss
        loss = np.inf

        # optimization loop
        for i in range(self.max_iterations):
            # Compute probabilities in low dimension
            Q = self._compute_low_dimensionnality_probas(Y)

            # Compute the loss
            new_loss = np.sum(
                P * np.log(np.clip(P, 1e-10, None) / np.clip(Q, 1e-10, None))
            )

            # Stop condition with tolerance
            if np.abs(loss - new_loss) < self.tolerance:
                break
            else:
                loss = new_loss

            gradient = self._compute_gradient(Y, P, Q)

            # update Y
            Y += -self.learning_rate * gradient

            if i == self.max_iterations - 1:
                print("Algorithm did not converge!")

        return Y

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
        sigma_left = np.sqrt(np.max(D))
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

    def _compute_gradient(self, Y: NDArray, P: NDArray, Q: NDArray) -> NDArray:
        """Compute the Kullback Leiber divergence loss gradient.

        Args:
            Y (NDArray): The coordinates of the points in the lower dimension space
            P (NDArray): The neighborhood probability matrix in the higher dimension space
            Q (NDArray): The neighborhood probability in the lower dimension space
        """
        # shape (n, n)
        D_low = self._compute_distances(Y)

        # shape (n, n, p)
        difference = Y[:, np.newaxis, :] - Y[np.newaxis, :, :]

        # Sum on the j axis: gradient of shape (n, n_dimensions)
        gradient = 4 * np.sum(
            (P - Q)[:, :, np.newaxis]
            * (difference)
            * (1 + (D_low[:, :, np.newaxis])) ** -1,
            axis=1,
        )

        return gradient


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from sklearn.datasets import load_digits
    from sklearn.manifold import TSNE as SklearnTSNE
    from sklearn.preprocessing import StandardScaler

    # Load a subset of digits dataset (10 classes, 64 features)
    digits = load_digits()
    X, y = digits.data[:300], digits.target[:300]  # type: ignore
    X = StandardScaler().fit_transform(X)

    # Custom TSNE
    tsne = TSNE(perplexity=30, n_dimensions=2, learning_rate=200, max_iterations=500)
    Y_custom = tsne.fit_transform(X)

    # Sklearn TSNE
    Y_sklearn = SklearnTSNE(
        n_components=2, perplexity=30, learning_rate=200, max_iter=500, random_state=42
    ).fit_transform(X)

    # Plot side by side
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, Y, title in zip(
        axes, [Y_custom, Y_sklearn], ["Custom t-SNE", "Sklearn t-SNE"]
    ):
        scatter = ax.scatter(Y[:, 0], Y[:, 1], c=y, cmap="tab10", s=15, alpha=0.8)
        ax.set_title(title)
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        plt.colorbar(scatter, ax=ax, label="Digit class")

    plt.tight_layout()
    plt.savefig("tsne_comparison.png", dpi=150)
    plt.show()
    print("Done. Plot saved to tsne_comparison.png")
