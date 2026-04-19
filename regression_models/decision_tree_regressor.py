from __future__ import annotations  # allow auto referencing class

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from bases import DecisionTreeBase, Node


class DecisionTreeRegressor(DecisionTreeBase):
    def __init__(
        self,
        max_depth: int = 5,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_variance: float = 1e-4,
    ):
        super().__init__(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
        )

        # Variance for regression
        self.min_variance = min_variance

    def _build_tree(self, X, y, node: Node) -> Node:
        # If we reach the max depth of not enough samples to split, we are in a leaf
        if node.level == self.max_depth or len(X) < self.min_samples_split:
            node.value = float(np.mean(y))
            return node

        # Create the split
        node.feature_idx, node.threshold, splitted_variance = self._best_split(X, y)

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
            node.value = float(np.mean(y))
            return node

        # check variance improvement
        delta_var = np.var(y) - splitted_variance

        # If not enough gain, we have a leaf node
        if delta_var < self.min_variance:
            node.value = float(np.mean(y))
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

    def _best_split(self, X, y) -> Tuple[int, float, float]:
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
        min_var = intra_class_variance[best_feature, 0]

        return best_feature, best_split, min_var
