"""
Implementation of the simple perceptron.
Apply a linear combination of the features for binary classification.
Classification 1 if z > 0 and 0 if z < 0. (Heaviside function).

Iterativly update the weights for each epoch using the feature value itself.
"""

import numpy as np
from numpy.typing import NDArray


class Perceptron:
    def __init__(self, learning_rate: float = 0.1, epochs: int = 1000):
        self.learning_rate = learning_rate
        self.epochs = epochs

    def fit(self, X: NDArray, y: NDArray):
        # Initialize the weights and bias to 0
        self.W = np.zeros_like(X[0])
        self.b = 0

        # Optimization loop
        for i in range(self.epochs):
            old_W = np.copy(self.W)
            old_b = np.copy(self.b)

            # update the weights sample by sample for convergence
            for j in range(len(X)):
                y_pred = (X[j, :] @ self.W) + self.b
                y_pred = np.where(y_pred > 0, 1, 0)

                # Compute the error
                error = y[j] - y_pred

                self.W = self.W + self.learning_rate * error * X[j, :]
                self.b = self.b + self.learning_rate * error

            # Check the convergence
            if all(old_W == self.W) and old_b == self.b:
                print(f"Converged at epoch {i}.")
                break
            elif i == self.epochs - 1:
                print("Algorithm did not converge")

    def predict(self, X: NDArray):
        predictions = (X @ self.W) + self.b
        predictions = np.where(predictions > 0, 1, 0)
        return predictions


if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.linear_model import Perceptron as SklearnPerceptron
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    # Linearly separable dataset (perceptron only converges on linearly separable data)
    X, y = make_classification(
        n_samples=500,
        n_features=4,
        n_informative=4,
        n_redundant=0,
        n_clusters_per_class=1,
        random_state=42,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Custom Perceptron
    custom = Perceptron(learning_rate=0.1, epochs=100)
    custom.fit(X_train, y_train)
    y_pred_custom = custom.predict(X_test)

    # Sklearn Perceptron
    sk = SklearnPerceptron(eta0=0.1, max_iter=100, random_state=42)
    sk.fit(X_train, y_train)
    y_pred_sk = sk.predict(X_test)

    print("\n=== Perceptron comparison ===")
    print(f"Custom  accuracy: {accuracy_score(y_test, y_pred_custom):.4f}")
    print(f"Sklearn accuracy: {accuracy_score(y_test, y_pred_sk):.4f}")
