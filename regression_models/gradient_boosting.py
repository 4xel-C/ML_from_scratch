import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import List, Optional

import numpy as np
from numpy.typing import NDArray

from regression_models import DecisionTreeRegressor


class GradientBoosting:
    def __init__(
        self, learning_rate: float = 0.1, n_estimators: int = 100, max_depth: int = 3
    ):
        self.eta = learning_rate
        self.n_estimators = n_estimators
        self.max_depth = max_depth  # Max_depth of the inner models

        # Declare the needed variables
        self.models: List[DecisionTreeRegressor] = list()
        self.mean: float
        self.fitted = False

    def fit(self, X: NDArray, y: NDArray):
        # Compute the mean of y
        self.mean = y.mean()

        # Initializse global predictions
        F = np.full(len(y), self.mean)

        # Compute the trees
        for i in range(self.n_estimators):
            # Compute residus
            res = y - F

            tree = DecisionTreeRegressor(max_depth=self.max_depth)

            # Fit on the residu
            tree.fit(X, res)

            # Prediction
            predictions = tree.predict(X)

            # Update the global prediction
            F = F + (self.eta * predictions)

            # Save the submodel
            self.models.append(tree)

        self.fitted = True

    def predict(self, X: NDArray) -> NDArray:
        # Compute the initial residus for all predictions with the means
        res = np.full(X.shape[0], self.mean)

        for tree in self.models:
            predictions = tree.predict(X)

            # refresh the residus
            res = res + (self.eta * predictions)

        return res


if __name__ == "__main__":
    import numpy as np
    from sklearn.datasets import make_regression
    from sklearn.ensemble import GradientBoostingRegressor as SklearnGB
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import train_test_split

    X, y = make_regression(n_samples=300, n_features=5, noise=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    custom = GradientBoosting(learning_rate=0.1, n_estimators=100, max_depth=3)
    custom.fit(X_train, y_train)
    pred_custom = custom.predict(X_test)

    sk = SklearnGB(learning_rate=0.1, n_estimators=100, max_depth=3, random_state=42)
    sk.fit(X_train, y_train)
    pred_sk = sk.predict(X_test)

    print("=== Gradient Boosting Regressor comparison ===")
    print(f"Custom MSE:  {mean_squared_error(y_test, pred_custom):.4f}")
    print(f"Sklearn MSE: {mean_squared_error(y_test, pred_sk):.4f}")
