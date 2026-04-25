import numpy as np
from numpy.typing import NDArray


class PCA:
    def __init__(self, n_components=2):
        self.n_components = n_components

    def fit(self, X: NDArray) -> None:
        # normalization
        means = X.mean(axis=0)
        self.means = means

        X_norm = X - means

        # Compute covariance matrix
        cov_matrix = (1 / X.shape[0]) * X_norm.T @ X_norm

        # eigen decomposition
        eigen_values, eigen_vectors = np.linalg.eig(cov_matrix)

        # keep the total inertia
        self.total_inertia = eigen_values.sum()

        # sort the vectors
        order_indices = eigen_values.argsort()[::-1]
        eigen_values = eigen_values[order_indices]
        eigen_vectors = eigen_vectors[:, order_indices]

        # get the desired number of components
        self.eigen_values = eigen_values[: self.n_components]
        self.eigen_vectors = eigen_vectors[:, : self.n_components]

    def transform(self, X_new) -> NDArray:
        # center using the means of the data
        X_centered = X_new - self.means

        # project the data in the space using the eigen vectors
        z = X_centered @ self.eigen_vectors

        return z

    def explained_variance_ratio(self):
        return self.eigen_values / self.total_inertia
