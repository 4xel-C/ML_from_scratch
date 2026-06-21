import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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


if __name__ == "__main__":
    import numpy as np
    from sklearn.datasets import load_diabetes
    from sklearn.linear_model import Ridge as SklearnRidge
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    X, y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    custom = Ridge(max_iterations=5000, learning_rate=0.01, l=1.0)
    custom.fit(X_train, y_train)
    pred_custom = custom.predict(X_test)

    sk = SklearnRidge(alpha=1.0)
    sk.fit(X_train, y_train)
    pred_sk = sk.predict(X_test)

    print("=== Ridge comparison ===")
    print(f"Custom MSE:  {mean_squared_error(y_test, pred_custom):.4f}")
    print(f"Sklearn MSE: {mean_squared_error(y_test, pred_sk):.4f}")
