"""
Implementation of linear regression:
Loss function: MSE. Convex loss function, with only 1 minima possible. Initialization of parameters at 0.
Optimization by gradient descent.

Equation : Ypred = X.w (bias added to the beginning of w with a virtual column of 1 in X)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from numpy.typing import NDArray

from helpers import NotFittedException
from loss_functions import gradient_mse, mse


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
            dw = gradient_mse(X_b, predictions, y)
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


if __name__ == "__main__":
    import numpy as np
    from sklearn.datasets import make_regression
    from sklearn.linear_model import LinearRegression as SklearnLR
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import train_test_split

    X, y = make_regression(n_samples=300, n_features=5, noise=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    custom = LinearRegression(max_iterations=2000, learning_rate=0.01)
    custom.fit(X_train, y_train)
    pred_custom = custom.predict(X_test)

    sk = SklearnLR()
    sk.fit(X_train, y_train)
    pred_sk = sk.predict(X_test)

    print("=== Linear Regression comparison ===")
    print(f"Custom MSE:  {mean_squared_error(y_test, pred_custom):.4f}")
    print(f"Sklearn MSE: {mean_squared_error(y_test, pred_sk):.4f}")
