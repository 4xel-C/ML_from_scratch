"""
Implementation of the GMM algorithm (Gaussian Mixed Model).

We try to clusterize data points with a soft assignation (probability of being member of all clustersa).
For this, we will compute cluster as Multivariate Gaussian.

We will optimize the likelihood p(x) = sum(p(x|k) * pk,
gaussinan likelihood times the mixing weight of the cluster (proportion of samples in cluster k)
With respect to sum(pk) = 1

We will use the EM algorithm:

Expectation:
    Each cluster has a 'reponsibility', being the probability of membership in one cluster for one point.

    r(i, k) = p(xi|k) * pk / (sum_k(xi|k) * pk) -> Baye's rule to compute the posterior given the parameter of the gaussian
    characterising the cluster k.
    sum_k(r(i, k)) = 1 -> Each point has a proportion in each cluster.

Maximization:
    After calculating the posterior, we recompute, for each gaussian, the centroids and their covariance matrix:
    - mu_k = sum_i(r(i, k) * x_i) / sum_i(r(i, k))
    - Sigma_k (covariance matrix) = 1/sum_i[r(i, k)] * sum_i[(xi - mui)(xi - mui)^T] (external product of the two vectors)

We stop the algorithm when reaching the max iterations threshold or when log(p(x)) (log likelihood) reach a plateau.

Initialisation with  neutral parameters:
    - k cluster
    - mu_k : Random points
    - Sigma_k: Identity matrix
    - p(k): 1/K
"""

import numpy as np
from numpy.typing import NDArray

from helpers import multivariate_gaussian_likelihood


class GMM:
    def __init__(self, k: int, max_iterations: int = 100, tol: float = 1e-4):
        self.k = k
        self.max_iterations = max_iterations
        self.tol = tol

    # TODO: Finish the implementation
    def fit(self, X: NDArray):
        # Dimension of the data matrix
        self.n = len(X)
        self.p = X.shape[1]

        centroids_idx = np.random.choice(self.n, size=self.k, replace=False)

        # starting parameters
        self.centroids = X[centroids_idx]

        # Creation of a tenso of size (k, p, p)
        self.covariances = np.array([np.identity(self.p) for _ in range(self.k)])

        # starting mixing weights
        self.pi_k = np.full(self.k, 1 / self.k)

        # Initialisation of the Rik matrix
        self.r = np.full((self.n, self.k), 0)

        # EM iterations
        for iterations in range(self.max_iterations):
            # Compute all likelihood for all points and clusters
            gaussian_likelihood = np.zeros((self.n, self.k))

            # Expectation
            for k in range(self.k):
                difference = X - self.centroids[k]

                mahalanobis = np.sum(
                    difference @ np.linalg.inv(self.covariances[k]) * difference, axis=1
                )

                gaussian_likelihood[:, k] = (
                    1
                    / np.sqrt(
                        ((2 * np.pi) ** self.p) * np.linalg.det(self.covariances[k])
                    )
                    * np.exp(-0.5 * mahalanobis)
                )

            self.r = (
                gaussian_likelihood
                * self.pi_k
                / (gaussian_likelihood @ self.pi_k)[:, np.newaxis]
            )

            # Maximization: update the clusters parameters
            for k in range(self.k):
                # Centroids
                self.centroids[k] = self.r[:, k] * X / (np.sum(self.r[:, k]))

                # Covariance matrix
                X_weighted = np.sqrt(self.r[:, k]) * (X - self.centroids[k])
                self.covariances[k] = (X_weighted @ X_weighted) / np.sum(self.r[:, k])

            ...

        ...

    def predict(self, X): ...


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    import numpy as np
    from sklearn.cluster import KMeans as SklearnKMeans
    from sklearn.datasets import make_blobs
    from sklearn.metrics import adjusted_rand_score

    X, y_true = make_blobs(n_samples=30, centers=3, cluster_std=0.8, random_state=42)  # type: ignore

    clust = GMM(k=3)

    print("Size of the data: ", X.shape)
    print()

    clust.fit(X)

    print(clust.pi_k)
