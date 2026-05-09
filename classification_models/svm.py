"""
Implementation of the SVM Classifier using hard margin.
"""

import numpy as np
from numpy.typing import NDArray


class SVMClassifier:
    def __init__(self): ...

    def fit(self, X: NDArray, y: NDArray) -> None: ...

    def predict(self, X: NDArray) -> NDArray: ...
