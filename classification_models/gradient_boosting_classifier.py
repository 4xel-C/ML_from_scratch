import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

"""
Implementaion of the gradient boosting classifier using cross_entropy as a loss function.
L = -y * log(p) - (1 - y)*log(1 - p) for a binary classification

where p represents the affectation probability of the class 1:
p = sigmoid(F) = 1  / (1 + e(-F)) where F is the logit computed by the model.

The gradient is then given by:
dL / dF = dL/dp * dp/dF (following the chain rule) (dp/ dF = sigmoid's derivative = p*(1-p))
        = (-y * (1/p) - (1-y) * (-1/(1-p))) * p*(1-p)
        = -y * (1/p) * p(1-p) - (1-y) * (-1/(1-p)) * p*(1-p)
        = -y * (1-p) - (1-y) * -p
        = -y * py - (-p + py)
        = -y * py +p -py
        = -y + p
        = p - y

We then have:
-dL / dF = y - p for optimization (gradient opposite)

Binary prediction
"""

from typing import List

import numpy as np
from numpy.typing import NDArray

from helpers import sigmoid
from regression_models import DecisionTreeRegressor


class GradientBoostingClassifier:
    def __init__(
        self, n_estimators: int = 100, learning_rate: float = 0.1, max_depth: int = 3
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth

        self.models: List[DecisionTreeRegressor] = list()  # All submodels of the GB
        self.fzero: (
            NDArray  # Initial prediction (logits of the class proportion (Prior))
        )
        self.fitted: bool

    def fit(self, X, y):
        # Compute the F initial prediction
        classes, counts = np.unique(y, return_counts=True)
        pzero = counts[1] / len(y)
        logit = np.full(
            len(y), np.log(pzero / (1 - pzero))
        )  # use the logit of the prior probability

        # Keep the initial logit prediction in memory (scalar)
        self.fzero = float(np.log(pzero / (1 - pzero)))

        # Compute the pseudo-residus using the cross entropy loss function gradient
        residus = (
            y - pzero
        )  # y contains classes 0 and 1 (Calculated from gradient of the crossentropy loss)

        for _ in range(self.n_estimators):
            # create a tree
            tree = DecisionTreeRegressor(max_depth=self.max_depth)

            # Fit on the data using the residus for prediction
            tree.fit(X, residus)

            # Compute the logit for each samples
            predictions = tree.predict(X)

            # Update the prediction f
            logit = logit + (self.learning_rate * predictions)

            # Recompute the residus
            residus = y - sigmoid(logit)

            # Save the model
            self.models.append(tree)

        self.fitted = True

    def predict(self, X):
        # Logits predicted (initialization)
        logits = self.fzero

        for tree in self.models:
            z = tree.predict(X)
            logits += self.learning_rate * z

        # Compute the prediction
        probas = sigmoid(logits)

        classes = np.where(probas >= 0.5, 1, 0)

        return classes

    def predict_probas(self, X):
        # Logits predicted (initialization)
        logits = self.fzero

        for tree in self.models:
            z = tree.predict(X)
            logits += self.learning_rate * z

        # Compute the prediction
        probas = sigmoid(logits)

        return probas


if __name__ == "__main__":
    import numpy as np
    from sklearn.datasets import make_classification
    from sklearn.ensemble import GradientBoostingClassifier as SklearnGBC
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split

    X, y = make_classification(n_samples=300, n_features=6, n_informative=4, n_redundant=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    custom = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
    custom.fit(X_train, y_train)
    pred_custom = custom.predict(X_test)

    sk = SklearnGBC(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    sk.fit(X_train, y_train)
    pred_sk = sk.predict(X_test)

    print("=== Gradient Boosting Classifier comparison ===")
    print(f"Custom accuracy:  {accuracy_score(y_test, pred_custom):.4f}")
    print(f"Sklearn accuracy: {accuracy_score(y_test, pred_sk):.4f}")
