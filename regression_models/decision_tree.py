from __future__ import annotations  # allow auto referencing class

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray


@dataclass
class Node:
    """Node of the decision tree"""

    level: int  # The level in the tree
    feature_idx: int  # The feature index on which the rule is created
    threshold: float  # The value of the threshold to split the data on
    left: Optional[Node] = None  # Left node in the tree
    right: Optional[Node] = None  # Right node in the tree
    value: Optional[float] = None  # The mean value stored in the leaf for y


class DecisionTreeRegressor:
    def __init__(
        self,
        max_depth: int = 5,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_variance: float = 1e-4,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_variance = min_variance

    def fit(self, X, y):
        # Get the first split
        first_feature, split = self._best_split(X, y)

        root_node: Node = Node(
            level=0,
            feature_idx=first_feature,
            threshold=split,
        )

        self._build_tree(X, y, root_node)

    def predict(self, X): ...

    def _build_tree(self, X, y, root: Node): ...

    def _best_split(self, X, y) -> Tuple[int, float]:
        # Get the best first feature to split on: for each featture, find the point or the category where we minimize the variance of y
        # Get the feature giving the smallest intraclass variance (minimize intraclass variability)
        intra_class_variance = np.zeros(
            (X.shape[1], 2)
        )  # for each feature: variance / split

        for feature_idx in range(X.shape[1]):
            feature = X[:, feature_idx]

            # sort the feature
            ordered_idx = np.argsort(feature)
            feature = feature[ordered_idx]
            y_sorted = y[ordered_idx]

            min_split = 0
            min_feature_var = float("inf")

            # iterate through all values
            for i in range(len(feature) - 1):
                split = (feature[i + 1] + feature[i]) / 2
                feature_var = np.var(y_sorted[np.where(feature < split)]) + np.var(
                    y_sorted[np.where(feature > split)]
                )

                if min_feature_var > feature_var:
                    min_split = split
                    min_feature_var = feature_var

            intra_class_variance[feature_idx, 0] = min_feature_var
            intra_class_variance[feature_idx, 1] = min_split

        best_feature = int(np.argmin(intra_class_variance[0]))
        best_split = intra_class_variance[best_feature, 1]

        return best_feature, best_split
