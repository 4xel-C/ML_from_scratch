import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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

from loss_functions import gradient_mse, mse
from regression_models.linear_regression import LinearRegression


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

            # compute the gradient (avoid computing the penalization for the bias term)
            gradient = gradient_mse(X_b, prediction, y)
            gradient[1:] += self.l * np.sign(self.w[1:])

            # update the weights
            self.w = self.w - (self.learning_rate * gradient)

            # new predictions
            predictions = np.dot(X_b, self.w)

            # update the loss
            loss = mse(predictions, y)

        self.fitted = True


if __name__ == "__main__":
    import numpy as np
    from sklearn.datasets import make_regression
    from sklearn.linear_model import Lasso as SklearnLasso
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import train_test_split

    X, y = make_regression(n_samples=300, n_features=10, noise=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    custom = Lasso(max_iterations=2000, learning_rate=0.01, l=0.1)
    custom.fit(X_train, y_train)
    pred_custom = custom.predict(X_test)

    sk = SklearnLasso(alpha=0.1, max_iter=2000)
    sk.fit(X_train, y_train)
    pred_sk = sk.predict(X_test)

    print("=== Lasso comparison ===")
    print(f"Custom MSE:  {mean_squared_error(y_test, pred_custom):.4f}")
    print(f"Sklearn MSE: {mean_squared_error(y_test, pred_sk):.4f}")
