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
