import numpy as np
from numpy.typing import NDArray


class KNN:
    def __init__(self, k: int):
        self.k = k

    def fit(self, X, y) -> None:
        self.X = X
        self.y = y

    def predict(self, X_test) -> NDArray:
        # Compute the distance matrix
        dist = np.linalg.norm(
            X_test[:, np.newaxis, :] - self.X[np.newaxis, :, :], axis=2
        )

        # get the k closest points
        knn = np.argsort(dist, axis=1)[:, : self.k]

        # k_labels
        k_labels = self.y[knn]

        # Compute the most frequent label
        prediction = np.apply_along_axis(
            lambda row: np.argmax(np.bincount(row)), axis=1, arr=k_labels
        )

        return prediction
