from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from classification_models import DecisionTreeClassifier


class RandomForest:
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 20,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Optional[int] = None,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features

    def fit(self, X: NDArray, y: NDArray):
        # number of features
        p = X.shape[1]

        # number of line
        n = X.shape[0]

        if self.max_features is None:
            self.max_features = int(np.sqrt(p))

        # store the ensemble mode as tupple (classifier, NDArray of feature index)
        ensemble_model: List[Tuple[DecisionTreeClassifier, NDArray]] = list()

        for _ in range(self.n_estimators):
            # select the features
            k = min(self.max_features, p)
            selected_features = np.random.choice(p, size=k, replace=False)

            # bootstrap the samples
            selected_samples = np.random.choice(n, size=n, replace=True)

            X_selected = X[selected_samples][:, selected_features]
            y_selected = y[selected_samples]

            # Compute the submodel
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                min_samples_split=self.min_samples_split,
            )
            tree.fit(X_selected, y_selected)

            ensemble_model.append((tree, selected_features))

        # save the models
        self.ensemble_model = ensemble_model

    def predict(self, X: NDArray):
        # store the predictions of each models: shape (n_predictions, k_models)
        predictions_ensemble: NDArray = np.zeros((X.shape[0], self.n_estimators))

        for i, (tree, features) in enumerate(self.ensemble_model):
            # select the adequat subset of features
            X_subset = X[:, features]

            # Make the prediction
            predictions = tree.predict(X_subset)

            # add the prediction to result matrix
            predictions_ensemble[:, i] = predictions

        result_vector = np.zeros(X.shape[0])

        # Select the mode for each prediction
        for i, predicted_classes in enumerate(predictions_ensemble):
            classe, count = np.unique(predicted_classes, return_counts=True)

            result_vector[i] = classe[np.argmax(count)]

        return result_vector
