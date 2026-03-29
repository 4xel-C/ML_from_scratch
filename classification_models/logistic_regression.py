"""
Implementation of the logistic regression for classification.
Linear regression for classification.
Equation: Logit = w^T . X (bias added as a supplementary 1 column to the features)
Sigmoid(Logit) -> Probability of predicting positive class
Loss function: Binary cross-entropy BCE = -sum(ylog(p) + (1-y)log(1-p))
"""

from typing import Optional

import numpy as np
from numpy.typing import NDArray

from helpers import NotFittedException, sigmoid
from loss_functions import bce, gradient_bce


class LogisticRegression:
    def __init__(self, learning_rate: float = 0.1, n_iterations: int = 1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.is_fitted = False
        self.w: NDArray[np.float64] = np.array(
            0
        )  # Initial value to single 0 until fit.

    def fit(self, X: NDArray, y: NDArray) -> None:
        # properties of matrix
        n_features = X.shape[1] + 1  # adding 1 column for the bias
        n = X.shape[0]

        # Add the bias column
        X_b = np.hstack([np.ones([n, 1]), X])

        # Initialize the weights to 0
        self.w = np.zeros(n_features)

        # Count the iteration
        iterations = 0

        # initialize the loss
        loss = float("inf")

        while iterations <= self.n_iterations and loss > 1e-6:
            iterations += 1

            # Make the predictions
            predictions = sigmoid(np.dot(X_b, self.w))

            # compute the initial loss
            loss = bce(predictions, y)

            # compute the gradient
            dbce = gradient_bce(X_b, predictions, y)

            # update the weights
            self.w = self.w - (self.learning_rate * dbce)

        if iterations >= self.n_iterations:
            print("Algorithm did not converge!")
        else:
            print("Algorithm successfully converged")

        self.is_fitted = True

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.int32]:
        if not self.is_fitted:
            raise NotFittedException

        # adding the bias column to the matrix
        X_b = np.hstack([np.ones((X.shape[0], 1)), X])

        proba = np.asarray(sigmoid(np.dot(X_b, self.w)), dtype=np.float64)
        return (proba > 0.5).astype(np.int32)

    def predict_proba(
        self, X: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        if not self.is_fitted:
            raise NotFittedException

        # adding the bias column to the matrix
        X_b = np.hstack([np.ones((X.shape[0], 1)), X])

        proba_target = sigmoid(np.dot(X_b, self.w))

        return (1 - proba_target, proba_target)
