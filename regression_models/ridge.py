"""Implementation of the linear regression using ridge as regularization method
Same as the linear regression, but adapting the ridge regularization to the loss function
L = MSE + lambda . ||w||^2

The derivative of the L2 norm is 2w. The factor 2, being absorbed by the regularization term lambda
"""

import numpy as np
from numpy.typing import NDArray

from loss_functions import gradient_mse, mse
from regression_models.linear_regression import LinearRegression


class Ridge(LinearRegression):
    def __init__(
        self, max_iterations: int = 1000, learning_rate: float = 0.01, l: float = 1
    ):
        # Appeler le constructeur parent
        super().__init__(max_iterations, learning_rate)

        self.l = l

    def fit(self, X: NDArray, y) -> None:
        iteration = 0
        loss = float("inf")

        # Adding a column of 1 to X for bias
        X_b = np.hstack([np.ones((X.shape[0], 1)), X])

        # Initialize the weights to 0
        self.w = np.zeros(X_b.shape[1])

        while iteration <= self.max_iterations and loss > 1e-6:
            iteration += 1

            prediction = np.dot(X_b, self.w)

            # compute the gradient (avoid computing the penalization for the bias term)
            gradient = gradient_mse(X_b, prediction, y)
            gradient[1:] += self.l * self.w[1:]

            # update the weights
            self.w = self.w - (self.learning_rate * gradient)

            # new predictions
            predictions = np.dot(X_b, self.w)

            # update the loss
            loss = mse(predictions, y)

        self.fitted = True
