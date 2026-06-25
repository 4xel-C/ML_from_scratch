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

import sys
from pathlib import Path
from typing import Tuple

import numpy as np
from numpy.typing import NDArray

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


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

        self.global_likelihood = -np.inf

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

            # Compute the log likelihood of the model
            loglikelihood = np.sum(
                np.log(np.sum(gaussian_likelihood * self.pi_k, axis=1))
            )

            # Maximization: update the clusters parameters
            for k in range(self.k):
                # Centroids
                self.centroids[k] = (self.r[:, k, np.newaxis] * X).sum(axis=0) / np.sum(
                    self.r[:, k]
                )

                # Covariance matrix
                X_weighted = np.sqrt(self.r[:, k])[:, np.newaxis] * (
                    X - self.centroids[k]
                )
                self.covariances[k] = (X_weighted.T @ X_weighted) / np.sum(self.r[:, k])

            self.pi_k = self.r.sum(axis=0) / self.n

            # compute log likelihood: first sum the likelihood on all clusters for each point, and compute the total likelihood.
            if np.abs(loglikelihood - self.global_likelihood) < self.tol:
                break
            self.global_likelihood = loglikelihood

    def predict(self, X: NDArray) -> Tuple[NDArray, NDArray]:
        """To predict, we recompute the responsibility for each cluster for each point (Probability of cluster k knowing data point x)

        Args:
            X (NDArray): The data to clusterize

        Return:
            Tuple[NDArray, NDArray]: The clusters assignation, and the probabilities

        """

        # Compute all likelihood for all points and clusters
        gaussian_likelihood = np.zeros((len(X), self.k))

        # Expectation
        for k in range(self.k):
            difference = X - self.centroids[k]

            mahalanobis = np.sum(
                difference @ np.linalg.inv(self.covariances[k]) * difference, axis=1
            )

            gaussian_likelihood[:, k] = (
                1
                / np.sqrt(((2 * np.pi) ** self.p) * np.linalg.det(self.covariances[k]))
                * np.exp(-0.5 * mahalanobis)
            )

        probas = (
            gaussian_likelihood
            * self.pi_k
            / (gaussian_likelihood @ self.pi_k)[:, np.newaxis]
        )

        return np.argmax(probas, axis=1), probas


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    import numpy as np
    from sklearn.datasets import make_blobs
    from sklearn.metrics import (
        adjusted_rand_score,
        normalized_mutual_info_score,
        silhouette_score,
    )
    from sklearn.mixture import GaussianMixture

    # Make a harder dataset: overlapping + anisotropic clusters.
    np.random.seed(42)
    X, y_true = make_blobs(  # type: ignore
        n_samples=100,
        centers=3,
        cluster_std=[2.4, 1.9, 2.1],
        random_state=42,
    )

    transform = np.array([[0.7, -0.6], [0.45, 0.9]])
    X = X @ transform
    X += np.random.normal(loc=0.0, scale=0.3, size=X.shape)

    # Fit custom GMM.
    custom = GMM(k=3, max_iterations=300, tol=1e-6)
    custom.fit(X)
    custom_labels, custom_probas = custom.predict(X)

    # Fit sklearn GMM on the same data.
    sk = GaussianMixture(
        n_components=3,
        covariance_type="full",
        max_iter=300,
        tol=1e-6,
        random_state=42,
    )
    sk.fit(X)
    sk_labels = sk.predict(X)
    sk_probas = sk.predict_proba(X)

    # Quantitative comparison.
    custom_ari = adjusted_rand_score(y_true, custom_labels)
    sk_ari = adjusted_rand_score(y_true, sk_labels)

    custom_nmi = normalized_mutual_info_score(y_true, custom_labels)
    sk_nmi = normalized_mutual_info_score(y_true, sk_labels)

    custom_sil = silhouette_score(X, custom_labels)
    sk_sil = silhouette_score(X, sk_labels)

    print("Dataset shape:", X.shape)
    print("Hard setup: overlapping + anisotropic Gaussian clusters")
    print()
    print("=== Comparison vs sklearn GaussianMixture ===")
    print(
        f"Custom GMM  | ARI: {custom_ari:.4f} | NMI: {custom_nmi:.4f} | Silhouette: {custom_sil:.4f}"
    )
    print(
        f"sklearn GMM | ARI: {sk_ari:.4f} | NMI: {sk_nmi:.4f} | Silhouette: {sk_sil:.4f}"
    )
    print()
    print(f"Custom total log-likelihood:  {custom.global_likelihood:.3f}")
    print(f"sklearn total log-likelihood: {sk.lower_bound_ * len(X):.3f}")
    print()
    print("First 5 responsibilities (custom):")
    print(custom_probas[:5])
    print()
    print("First 5 responsibilities (sklearn):")
    print(sk_probas[:5])
