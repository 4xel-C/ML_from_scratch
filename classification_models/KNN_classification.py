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


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    import numpy as np
    from sklearn.datasets import make_classification
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier

    X, y = make_classification(n_samples=300, n_features=6, n_informative=4, n_redundant=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    custom = KNN(k=5)
    custom.fit(X_train, y_train)
    pred_custom = custom.predict(X_test)

    sk = KNeighborsClassifier(n_neighbors=5)
    sk.fit(X_train, y_train)
    pred_sk = sk.predict(X_test)

    print("=== KNN comparison ===")
    print(f"Custom accuracy:  {accuracy_score(y_test, pred_custom):.4f}")
    print(f"Sklearn accuracy: {accuracy_score(y_test, pred_sk):.4f}")
