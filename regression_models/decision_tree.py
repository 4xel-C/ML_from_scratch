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
        root_node: Node = Node(
            level=0,
            feature_idx=0,  # placeholder value
            threshold=0,  # Placeholder value
        )

        self.tree = self._build_tree(X, y, root_node)

    # TODO: To be implemented
    def predict(self, X): ...

    def _build_tree(self, X, y, node: Node) -> Node:
        # If we reach the max depth of not enough samples to split, we are in a leaf
        if node.level == self.max_depth or len(X) < self.min_samples_split:
            node.value = np.mean(y)
            return node

        # Create the split
        node.feature_idx, node.threshold = self._best_split(X, y)

        # Split the data
        left_data_idx = X[:, node.feature_idx] < node.threshold
        right_data_idx = X[:, node.feature_idx] > node.threshold

        X_left = X[left_data_idx, :]
        y_left = y[left_data_idx]

        X_right = X[right_data_idx, :]
        y_right = y[right_data_idx]

        # Check if we can have enough samples in the potential leaves
        if len(X_left) < self.min_samples_leaf or len(X_right) < self.min_samples_leaf:
            # we have a leaf
            node.value = np.mean(y)
            return node

        # check variance improvement
        delta_var = np.var(y) - (np.var(y_left) + np.var(y_right))

        # If not enough gain, we have a leaf node
        if delta_var < self.min_variance:
            node.value = np.mean(y)
            return node

        # initialize left and right node
        left_node = None
        right_node = None

        left_node = Node(
            level=(node.level + 1),
            feature_idx=0,
            threshold=0,
        )

        right_node = Node(
            level=(node.level + 1),
            feature_idx=0,
            threshold=0,
        )

        # Crete the next node
        node.left = self._build_tree(X_left, y_left, left_node)

        node.right = self._build_tree(X_right, y_right, right_node)

        # return the node after updating left and right branch
        return node

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
                n = len(y)
                y_left = y_sorted[np.where(feature < split)]
                y_right = y_sorted[np.where(feature > split)]

                feature_var = (len(y_left) / n) * np.var(y_left) + (
                    len(y_right) / n
                ) * np.var(y_right)

                if min_feature_var > feature_var:
                    min_split = split
                    min_feature_var = feature_var

            intra_class_variance[feature_idx, 0] = min_feature_var
            intra_class_variance[feature_idx, 1] = min_split

        best_feature = int(np.argmin(intra_class_variance[:, 0]))
        best_split = intra_class_variance[best_feature, 1]

        return best_feature, best_split
