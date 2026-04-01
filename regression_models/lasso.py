"""Implementation of the linear regression using lasso as regularization method
Same as the linear regression, but adapting the lasso regularization to the loss function
L = MSE + lambda . |w|

|w| derivative correspond to the sign of w:
    if w > 0: |w| = w and d|w|/dw = 1
    if w < 0: |w| = -w and d|w|/dw = -1
    corner case if |w| = 0: not possible -> return 0
"""

import numpy as np
from numpy.typing import NDArray

from loss_functions import gradient_mse
from regression_models import LinearRegression


class Lasso(LinearRegression):
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

            # compute the gradient
            gradient = gradient_mse(X_b, prediction, y) + self.l * np.sign(self.w)

            # update the weights
            self.w = self.w - (self.learning_rate * gradient)

        self.fitted = True
