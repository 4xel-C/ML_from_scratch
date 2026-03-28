"""
Implementation of linear regression:
Loss function: MSE. Convex loss function, with only 1 minima possible. Initialization of parameters at 0.
Optimization by gradient descent.

Equation : Ypred = X.w (bias added to the beginning of w with a virtual column of 1 in X)
"""

import numpy as np
from numpy.typing import NDArray

from loss_functions import mse


class NotFittedException(Exception): ...


class LinearRegression:
    def __init__(self, max_iterations: int = 1000, learning_rate: float = 0.01):
        # initialize fitted and parameters vector
        self.fitted: bool = False
        self.w: NDArray = np.array(0)

        # initialize hyperparameters
        self.max_iterations = max_iterations
        self.learning_rate = learning_rate

    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]):
        """fit the linear regression to the data. Initialize weights to 0, and proceed gradient descend until
        we reach threshold or max iteration.

        Args:
            X (NDArray[np.float64]): Features matrices
            y (NDArray[np.float64]): Target
        """
        # Adding a column of 1 at the beginning of X for bias
        X_b = np.hstack([np.ones((X.shape[0], 1)), X])
        n_features = X_b.shape[1]
        n = X.shape[0]

        # Initialize parameters to 0
        w = np.zeros(n_features)

        # Declare loss
        loss = float("inf")

        # step for gradient descent and iterations
        iteration = 0

        # gradient descent
        while iteration < self.max_iterations and loss > 1e-6:
            iteration += 1

            predictions = np.dot(X_b, w)
            loss = mse(predictions, y)

            # gradient calculation
            dw = 0.5 / n * np.dot((predictions - y), X_b)
            step_size = self.learning_rate * dw

            # updating the parameters
            w = w - step_size

        if iteration >= self.max_iterations:
            print("Algorithm did not converge.")

        # save the parameters
        self.fitted = True
        self.w = w

    def predict(self, X: NDArray) -> NDArray:
        # Add a biais column to X:
        X_b = np.hstack([np.ones((X.shape[0], 1)), X])

        if not self.fitted:
            print("Algorithm not fitted yet!")
            raise NotFittedException()

        return np.dot(X_b, self.w)
