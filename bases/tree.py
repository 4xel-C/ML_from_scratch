from __future__ import annotations  # allow auto referencing class

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray


@dataclass
class Node:
    """Node of the decision tree"""

    level: int  # The level in the tree
    feature_idx: int = 0  # The feature index on which the rule is created
    threshold: float = 0  # The value of the threshold to split the data on
    left: Optional[Node] = None  # Left node in the tree
    right: Optional[Node] = None  # Right node in the tree
    value: Optional[float] = None  # The mean value stored in the leaf for y


class DecisionTreeBase(ABC):
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

    def predict(self, X):
        return np.apply_along_axis(lambda x: self._traverse(x, self.tree), 1, X)

    def _traverse(self, x: NDArray, node: Node) -> float:
        """Function to recursively traverse the tree until a leaf is reached

        Args:
            x (_type_): One data point
            node (_type_): Node of the tree
        """

        if node.value is not None:
            return node.value

        x_value = x[node.feature_idx]

        if x_value < node.threshold:
            assert node.left is not None
            result = self._traverse(x, node.left)
        else:
            assert node.right is not None
            result = self._traverse(x, node.right)

        return result

    @abstractmethod
    def _build_tree(self, X: NDArray, y: NDArray, node: Node) -> Node: ...

    @abstractmethod
    def _best_split(self, X: NDArray, y: NDArray) -> Tuple[int, float, float]: ...
