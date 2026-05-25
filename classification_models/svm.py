"""
Implementation of the SVM Classifier using hard margin.

SVM -> hyperplan maximizin the margin between two classes.

hyperplan equation: W^Tx + b = 0

The margin == orthogonal distance from the hyperplane d =   (w^tx + b) / ||w||

We set w^tx + b = 1 | -1 for the margin (limit to the first support point)

we then have the total width of the margin of 2 / ||w||

We thus want to maximize this distance; minimizing ||w|| (or 1/2 ||w||^2 for pratical reason when calculating the gradien).
We also want all the points on outside of the margin:
yi * (w^tx_i +b) >= 1 (for any i)

We then introduce the Ei (slack variable) for a soft margin svm

we minimize: (1/2) ||w||^2 + C * sum(E_i), where C is the regularization parameter
satisfying: y_i * (w^tx_i +b) >= 1 - E_i for any i
and E_i > 0

when C is big: strong penalization of the margin violations, narrow margin
otherwise: more violations are tolerated, larger margin

We want E_i >= max(0, 1 - y_i * (w^tx_i +b)) (Constraint manipulation and positive)
-> E_i effective when the point is misclassified or in the margin, otherwise E_i = 0

L = (1/2) ||w||^2 + C * sum(max(0, 1 - y_i * (w^tx_i +b))


Gradient calculation:

If the point is correctly classified: y_i * (w^tx_i +b) >= 1, then L = (1/2) ||w||^2, the gradient dL / dw = w and dL / db = 0
If the point if misclassified: y_i * (w^tx_i +b) < 1, then gradient = w - C * yi * xi and dL / db = -C * yi
"""

import numpy as np
from numpy.typing import NDArray


class SVMClassifier:
    def __init__(
        self,
        C: float = 1,
        learning_rate: float = 0.1,
        n_iterations: int = 100,
        tol: float = 1e-4,
    ):
        self.C = C
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.tol = tol

        # Other attributes declaration
        self.fitted: bool = False

    def fit(self, X: NDArray, y: NDArray) -> None:
        """We suppose the class between 0 and 1"""

        # remap the y
        self.y = np.where(y == 0, -1, 1)

        # initialization of the weights
        self.w = np.zeros(X.shape[1])
        self.b = 0

        for i in range(self.n_iterations):
            # Compute and the distances to the hyperplane (shape n)
            dist = X @ self.w + self.b

            support_indices = np.where(self.y * dist < 1)

            # Compute the hinge loss for the violating point (no loss for the corrected classified)
            dhingedw = self.C * (self.y[support_indices].T @ X[support_indices])

            dhingedb = -self.C * np.sum(self.y[support_indices])

            # compute the gradients
            dldw = self.w - dhingedw
            dldb = 0 + dhingedb

            # Update the weights in the opposite direction of the gradient
            self.w = self.w - (self.learning_rate * dldw)
            self.b = self.b - (self.learning_rate * dldb)

            # Check convergence and break if the update loop if no notable evolution
            if (
                all(self.learning_rate * dldw <= self.tol)
                and self.learning_rate * dldb <= self.tol
            ):
                break

            if i == self.n_iterations - 1:
                print("Algorithm did not converge")

        self.fitted = True

    def predict(self, X: NDArray) -> NDArray:
        predictions = (X @ self.w) + self.b

        predictions = np.where(predictions > 0, 1, 0)

        return predictions
