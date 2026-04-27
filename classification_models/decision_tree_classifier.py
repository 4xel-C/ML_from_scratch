from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from bases import DecisionTreeBase, Node
from helpers import gini


class DecisionTreeClassifier(DecisionTreeBase):
    def __init__(
        self,
        max_depth: int = 5,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_gini: float = 1e-4,
    ):
        super().__init__(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
        )

        # for classification we will use Gini index
        self.min_gini = min_gini

    def _build_tree(self, X: NDArray, y: NDArray, node: Node) -> Node:
        # check the stopping parameters
        if node.level > self.max_depth or len(X) < self.min_samples_split:
            # find the mode (most frequent class)
            classes, counts = np.unique(y, return_counts=True)
            node.value = classes[np.argmax(counts)]
            return node

        # Update the currend node
        node.feature_idx, node.threshold, splitted_gini = self._best_split(X, y)

        # Split the data
        mask_left = X[:, node.feature_idx] < node.threshold
        mask_right = X[:, node.feature_idx] >= node.threshold

        X_left, y_left = X[mask_left, :], y[mask_left]
        X_right, y_right = X[mask_right, :], y[mask_right]

        # Confirm min_samples
        if len(X_left) < self.min_samples_leaf or len(X_right) < self.min_samples_leaf:
            # find the mode (most frequent class)
            classes, counts = np.unique(y, return_counts=True)
            node.value = classes[np.argmax(counts)]
            return node

        # Check if the gini delta if sufficient
        delta = gini(y) - splitted_gini

        if delta < self.min_gini:
            # find the mode (most frequent class)
            classes, counts = np.unique(y, return_counts=True)
            node.value = classes[np.argmax(counts)]
            return node

        # build the left node
        node_left = Node(
            level=node.level + 1,
        )

        node_right = Node(
            level=node.level + 1,
        )

        node.left = self._build_tree(X_left, y_left, node_left)
        node.right = self._build_tree(X_right, y_right, node_right)

        return node

    def _best_split(self, X: NDArray, y: NDArray) -> Tuple[int, float, float]:
        """Find the best feature and the optimized threshold to minimize the gini index (reduce entropy)

        Returns:
            Tuple[int, float, float]: A tuple containing the feature index, threshold, and gini value
        """
        # Initialize the gini index matrix size (p features, 2 columns for gini index, trheshold)
        feature_gini = np.zeros((X.shape[1], 2))

        for feature_index in range(X.shape[1]):
            x = X[:, feature_index]

            # sort both the feature and the target vector
            ordered_indices = np.argsort(x)
            x_sorted = x[ordered_indices]
            y_ordered = y[ordered_indices]

            # minimum value with corresponding threshold
            min_gini = float("inf")
            min_threshold = 0

            # Search for the best threshold
            for i in range(len(x_sorted) - 1):
                # threshold
                threshold = (x_sorted[i + 1] + x_sorted[i]) / 2

                # Compute the gini index
                y_left = y_ordered[x_sorted < threshold]
                y_right = y_ordered[x_sorted > threshold]

                # Compute the weighed gini index
                gini_value = (len(y_left) / len(y)) * gini(y_left) + (
                    len(y_right) / len(y)
                ) * gini(y_right)

                if gini_value < min_gini:
                    min_gini = gini_value
                    min_threshold = threshold

            # update the feature gini matrix
            feature_gini[feature_index, 0] = min_gini
            feature_gini[feature_index, 1] = min_threshold

        # Find the best feature
        best_feature = int(np.argmin(feature_gini[:, 0]))
        best_threshold = feature_gini[best_feature, 1]
        gini_value = feature_gini[best_feature, 0]

        return best_feature, best_threshold, gini_value
