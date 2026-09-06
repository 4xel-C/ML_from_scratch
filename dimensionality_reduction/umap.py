"""
UMAP (Uniform Manifold Approximation and Projection) implementation from scratch.

How to compute UMAP:

1) Compute the neighborhood graph with adpated weights
2) Randomly distribute the points in the reduced space
3) Optimize using the binary cross-entropy to rebuild the graph
"""

import numpy as np
import umap
from numpy.typing import NDArray
from scipy.optimize import curve_fit
from scipy.sparse import csr_matrix
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler


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

        n = len(X)

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
        target_entropy = np.log2(self.n_neighbors)

        sigmas = np.apply_along_axis(
            lambda row: self._binary_search_sigma(row, target_entropy, 0, 1000),
            1,
            knn_dist_rho,
        )

        # Compute the weights (similarities) of the graph with the sigmas computed
        w: NDArray = np.exp(-(knn_dist_rho) / sigmas[:, np.newaxis])

        # compute the sparse matrix
        adjacency_matrix = csr_matrix(
            (
                w.ravel(),
                ((np.repeat(np.arange(n), self.n_neighbors)), knn_indices.ravel()),
            ),
            shape=(n, n),
        )

        # Symmetrize the matrix
        P = (adjacency_matrix + adjacency_matrix.T) - (
            adjacency_matrix.multiply(adjacency_matrix.T)
        )

        # =============  Reduced space
        # randomly generates points with small variance in the reduced space
        Y = np.random.normal(0, scale=1e-4, size=(n, self.n_components))

        # find a, b for the student neighborhood probability formule
        a, b = self._find_ab(self.min_dist)

        # Get all the pairs of connected vertex from the adjancency matrix to compute the loss on connected edges
        rows, cols = adjacency_matrix.nonzero()

        # Optimize the position in reduced space
        for epoch in range(self.n_epochs):
            # Compute the distance matrix
            dist_reduced = np.sqrt(
                np.clip(
                    np.sum((Y[np.newaxis, :, :] - Y[:, np.newaxis, :]) ** 2, axis=2),
                    0,
                    None,
                )
            )

            # Compute the probabilities
            Q = 1 / (1 + a * dist_reduced ** (2 * b))

            # initialize the gradient
            gradients = np.zeros((n, self.n_components))

            # Compute the gradient for each point
            for i in range(n):
                # Handle the attraction on the connected pairs
                connected_to = cols[rows == i]
                d_connected = dist_reduced[i, connected_to]
                p_connected = (
                    P.getrow(i)[:, connected_to].toarray().ravel()
                )  # sparse matrix object, rebuild the row
                q_connected = Q[i, connected_to]

                grad = self._grad_calculation(
                    a, b, d_connected, p_connected, q_connected, Y[i], Y[connected_to]
                )

                gradients[i] = grad

                # negative sampling
                not_connected_to = np.random.choice(
                    np.setdiff1d(np.arange(n), connected_to), self.n_negative_sample
                )
                d_not_connected = dist_reduced[i, not_connected_to]
                p_not_connected = np.zeros(
                    self.n_negative_sample
                )  # Non neighbors pairs -> no probability

                q_not_connected = Q[i, not_connected_to]

                grad_repulsion = self._grad_calculation(
                    a,
                    b,
                    d_not_connected,
                    p_not_connected,
                    q_not_connected,
                    Y[i],
                    Y[not_connected_to],
                )

                gradients[i] += grad_repulsion

            # update the gradients
            Y -= self.learning_rate * gradients

        return Y

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

        if mid_sigma == 0:
            return right_sigma

        if current_iteration == max_iterations:
            print("Max iteration reached ! The sigma search did not converge.")
            return mid_sigma

        probabilities: NDArray = np.exp(-(distances_rho) / mid_sigma)

        # compute the entropy for sigma
        H_mid = np.sum(probabilities)

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

    def _find_ab(self, min_dist: float):
        """Numerical fitting to find a and b parameter in the personnalized student formula from the min_dist hyperparameter.
        Below min_dist, we want the points to distance 1, beyond min_dist, we want an exponential decrease in similarity.

        Args:
            min_dist (float): Min distance below which we normalize to 1
        """
        d_values = np.linspace(0, 3, 300)
        target = np.where(d_values < min_dist, 1, np.exp(-(d_values - self.min_dist)))

        def student_t_ab(x, a, b):
            return 1 / (1 + a * x ** (b * 2))

        params, covariance = curve_fit(student_t_ab, d_values, target)

        return params

    def _grad_calculation(
        self,
        a: float,
        b: float,
        d: NDArray,
        p: NDArray,
        q: NDArray,
        yi: NDArray,
        yj: NDArray,
    ) -> NDArray:
        """Compute the gradient using broadcasting on multiple pairs at once. m being the number of dimensions in
        the reduced space.

        Returns:
            An array containing the gradient of the components in the reduced space.
        """

        d = np.clip(d, 1e-10, None)

        # Shape [k]
        num_first_factor: NDArray = 2 * a * b * d ** (2 * b - 1)

        # shape [k]
        num_second_factor: NDArray = (p * (1 - q)) - (q * (1 - p))

        # shape [k, m]
        third_factor: NDArray = (yi[np.newaxis, :] - yj) / d[:, np.newaxis]

        # shape [k]
        denominator: NDArray = (
            np.clip(q * (1 - q), 1e-10, None) * (1 + a * d ** (2 * b)) ** 2
        )

        # shape k
        grad = num_first_factor * num_second_factor / denominator

        # shape [k, m]
        grad = grad[:, np.newaxis] * third_factor
        return np.sum(grad, axis=0)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from sklearn.datasets import make_blobs

    X, y = make_blobs(  # type: ignore
        n_samples=100,
        centers=[[-5, -5], [0, 5], [5, -5]],  # type: ignore
        cluster_std=2.1,
        random_state=42,
    )
    X = StandardScaler().fit_transform(X)

    # Notre implémentation
    engine = UMAP(
        n_neighbors=15,
        n_components=2,
        learning_rate=0.001,
        n_epochs=100,
        n_negative_sample=30,
        min_dist=0.01,
    )
    Y_ours = engine.fit_transform(X)

    ref = umap.UMAP(
        n_neighbors=15,
        n_components=2,
        min_dist=0.01,
        learning_rate=0.001,
        n_epochs=100,
        n_negative_sample=30,  # type: ignore
    )

    Y_ref = ref.fit_transform(X)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(Y_ours[:, 0], Y_ours[:, 1], c=y, cmap="tab10", s=10)
    axes[0].set_title("My UMAP Implementation")
    axes[1].scatter(Y_ref[:, 0], Y_ref[:, 1], c=y, cmap="tab10", s=10)  # type: ignore
    axes[1].set_title("UMAP Reference (umap-learn)")

    plt.tight_layout()
    plt.show()
