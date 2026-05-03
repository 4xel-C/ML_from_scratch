from typing import List

import numpy as np
from numpy.typing import NDArray

from classification_models.decision_tree_classifier import DecisionTreeClassifier
from helpers import NotFittedException


class AdaBoostClassifier:
    def __init__(self, n_estimators: int = 100) -> None:
        self.n_estimators = n_estimators
        self.fitted = False

        # Declaration of the variables
        self.estimators: List[DecisionTreeClassifier]
        self.alpha: NDArray
        self.w: NDArray
        self.classes: NDArray

    def fit(self, X: NDArray, y: NDArray) -> None:
        models = list()

        self.alpha = np.zeros(self.n_estimators)  # Weights for learners normalize to 1

        self.w = np.full(
            X.shape[0], 1 / X.shape[0]
        )  # Initialize all equals, normalized to 1

        self.classes = np.unique(y)

        for i in range(self.n_estimators):
            # Create the weak learner
            tree = DecisionTreeClassifier(max_depth=1)  # Max depth of 1 for adaboost

            # Fit the data
            tree.fit(X, y, weights=self.w)

            # Make the prediction
            predictions = tree.predict(X)

            # Compute the weighted mean of the error (between 0 and 1)
            error = np.sum(self.w * (predictions != y) / sum(self.w))

            # update the model weight
            self.alpha[i] = 0.5 * np.log((1 - error) / error)

            # transform the error in -1 / +1
            predictions_check = np.where(predictions == y, 1, -1)

            # Update the samples weights using the tree weight (Misclassified samples weight more to focus on the error)
            self.w = self.w * np.exp(-self.alpha[i] * predictions_check)

            # Normalization to 1
            self.w = self.w / np.sum(self.w)

            # Add the tree to the model list
            models.append(tree)

        self.fitted = True
        self.estimators = models

    def predict(self, X) -> NDArray:
        if not self.fitted:
            raise NotFittedException()

        scores = np.zeros((len(X), len(self.classes)))

        for i, model in enumerate(self.estimators):
            predictions = model.predict(X)

            # Update for each sample the score for the predictions
            scores[np.arange(len(X)), np.searchsorted(self.classes, predictions)] += (
                self.alpha[i]
            )

        return self.classes[np.argmax(scores, axis=1)]
