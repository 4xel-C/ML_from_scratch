import numpy as np
from numpy.typing import NDArray


def bce(proba: NDArray, y_true: NDArray) -> np.float32:
    # protection if probability goes to 1 or 0
    proba = np.clip(proba, 1e-15, 1 - 1e-15)
    result = -np.mean((y_true * np.log(proba) + (1 - y_true) * np.log(1 - proba)))
    return result


def gradient_bce(X: NDArray, proba: NDArray, y_true: NDArray):
    return (1 / X.shape[0]) * (np.dot(np.transpose(X), (proba - y_true)))
