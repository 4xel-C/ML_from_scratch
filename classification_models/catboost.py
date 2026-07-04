"""
Implementation of the CatBoost algorithm.

Catboost is a gradient boosting algorithm, handling the multimodals categorical features.
It uses:
    - Ordered encoding: Encoding the categorical values taking into consideration the conditional means of the category
      using all the points predecessing the prediction point to avoid any leakage. Use a bayesian smoothing with a glboal
      prior to handle low supported categories and a smoothing parameter a.
    - Compute the residu (Ordering boosting ignored for this implementation)
    - Next tree predict on the residue
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


import numpy as np
from numpy.typing import NDArray

from classification_models import GradientBoostingClassifier
from helpers import NotFittedException


class CatboostClassifier(GradientBoostingClassifier):
    def __init__(
        self,
        cat_idx: NDArray,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        a: float = 1,
    ):
        """Constructor

        Args:
            cat_idx (NDArray): The index of the categorical values
            n_estimators (int, optional): Number of subtrees. Defaults to 100.
            learning_rate (float, optional): Learning rate to update the residus. Defaults to 0.1.
            max_depth (int, optional): Max depth of the subtrees. Defaults to 3.
            a (float, optional): Prior smoothing parameter
        """
        super().__init__(n_estimators, learning_rate, max_depth)
        self.cat_idx = cat_idx
        self.a = a

        # Initialize the dictionnaries to store the value of the modalities. (format: cat_values[col_idx][mod]) and the prior
        self.cat_values = dict()
        self.prior = 0
        self.fitted = False

    def fit(self, X: NDArray, y: NDArray):
        # Reset learned encodings on each fit.
        self.cat_values = {}

        # Encode the X matrix
        X_encoded = self._encode_categorical(X, y, self.cat_idx, self.a)

        # Fit on the X_encoded
        super().fit(X_encoded, y)
        self.fitted = True

    def predict(self, X: NDArray) -> NDArray:
        if not self.fitted:
            raise NotFittedException()

        X_encoded = np.zeros((X.shape[0], X.shape[1]), dtype=float)
        cat_idx_set = set(int(idx) for idx in self.cat_idx)

        # Keep non-categorical features as numeric values.
        for idx in range(X.shape[1]):
            if idx not in cat_idx_set:
                X_encoded[:, idx] = X[:, idx].astype(float)

        # Encode the X matrix
        for idx in self.cat_idx:
            cat_column = X[:, idx]

            # Encode the modatlities using the last modality encoding from the fit
            for mod in np.unique(cat_column):
                # Known modality
                if mod in self.cat_values[idx]:
                    X_encoded[cat_column == mod, idx] = self.cat_values[idx][mod]
                else:
                    # Replace by the prior if unknown modality
                    X_encoded[cat_column == mod, idx] = self.prior

        return super().predict(X_encoded)

    def _encode_categorical(
        self, X: NDArray, y: NDArray, cat_idx: NDArray, a: float
    ) -> NDArray:
        """Encode the categorial features of the X matrix using ordered target encoding.

        Args:
            X (NDArray): Matrix to encode
            y (NDArray): Targets array
            cat_idx (NDArray): The indices of the categorical features
            a (float): Smoothing parameter

        Returns:
            NDArray: The encoded matrix
        """

        # Compute the prior
        self.prior = np.mean(y)

        Xresult = np.zeros((X.shape[0], X.shape[1]), dtype=float)
        cat_idx_set = set(int(idx) for idx in cat_idx)

        # Keep non-categorical features as numeric values.
        for idx in range(X.shape[1]):
            if idx not in cat_idx_set:
                Xresult[:, idx] = X[:, idx].astype(float)

        # Generate a random permutation
        perm = np.random.permutation(len(X))

        # Compute the encoding for each feature
        for idx in cat_idx:
            # Count the categories
            cat_count = {mod: 0 for mod in np.unique(X[:, idx])}

            # Cumulative sum of the y value
            cat_cumsum = {mod: 0 for mod in np.unique(X[:, idx])}

            # extract the cat column
            cat_column = X[:, idx]

            # result
            result_col = np.zeros(cat_column.shape[0], dtype=float)

            # Iterate on permutation to compute the values of each modalities
            for i in perm:
                mod = cat_column[i]

                # Compute the point mean with bayesian smoothing
                point_value = (cat_cumsum[mod] + a * self.prior) / (cat_count[mod] + a)

                result_col[i] = point_value

                # Update the cumulative counts
                cat_count[mod] += 1
                cat_cumsum[mod] += int(y[i])

            # Update
            Xresult[:, idx] = result_col

            # save the last conditional means obtaines
            self.cat_values[idx] = {
                mod: float(cat_cumsum[mod] / cat_count[mod])
                for mod in np.unique(X[:, idx])
            }

        return Xresult


if __name__ == "__main__":
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import GradientBoostingClassifier as SklearnGBC
    from sklearn.metrics import accuracy_score, classification_report, f1_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    np.random.seed(42)

    # Synthetic dataset with only categorical features.
    n_samples = 80
    cities = np.random.choice(["paris", "lyon", "marseille", "lille"], size=n_samples)
    devices = np.random.choice(["mobile", "desktop", "tablet"], size=n_samples)
    plans = np.random.choice(["free", "basic", "premium"], size=n_samples)
    channels = np.random.choice(["ads", "seo", "email"], size=n_samples)

    X = np.column_stack([cities, devices, plans, channels])

    # Build a binary target from category interactions + small noise.
    logits = (
        0.8 * (plans == "premium").astype(float)
        + 0.6 * (devices == "desktop").astype(float)
        + 0.5 * (cities == "paris").astype(float)
        + 0.4 * ((plans == "basic") & (channels == "email")).astype(float)
        - 1.1
    )
    probas = 1 / (1 + np.exp(-logits))
    y = (np.random.rand(n_samples) < probas).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    cat_idx = np.array([0, 1, 2, 3])

    model = CatboostClassifier(
        cat_idx=cat_idx,
        n_estimators=120,
        learning_rate=0.1,
        max_depth=3,
        a=5,
    )
    model.fit(X_train, y_train)
    y_pred_custom = model.predict(X_test)

    # sklearn baseline: one-hot encoding + gradient boosting classifier.
    sklearn_pipeline = Pipeline(
        [
            (
                "preprocess",
                ColumnTransformer(
                    [
                        (
                            "cat",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                            [0, 1, 2, 3],
                        )
                    ],
                    remainder="drop",
                ),
            ),
            (
                "clf",
                SklearnGBC(
                    n_estimators=120,
                    learning_rate=0.1,
                    max_depth=3,
                    random_state=42,
                ),
            ),
        ]
    )
    sklearn_pipeline.fit(X_train, y_train)
    y_pred_sklearn = sklearn_pipeline.predict(X_test)

    acc_custom = accuracy_score(y_test, y_pred_custom)
    f1_custom = f1_score(y_test, y_pred_custom)
    acc_sklearn = accuracy_score(y_test, y_pred_sklearn)
    f1_sklearn = f1_score(y_test, y_pred_sklearn)

    print("=== CatBoost custom vs sklearn baseline ===")
    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")
    print(f"Custom   -> Accuracy: {acc_custom:.4f} | F1: {f1_custom:.4f}")
    print(f"sklearn  -> Accuracy: {acc_sklearn:.4f} | F1: {f1_sklearn:.4f}")
    print("\nClassification report (custom):")
    print(classification_report(y_test, y_pred_custom, digits=4))
    print("Classification report (sklearn):")
    print(classification_report(y_test, y_pred_sklearn, digits=4))
