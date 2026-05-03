from typing import Optional

import numpy as np
from numpy.typing import NDArray


def gini(y: NDArray, weights: Optional[NDArray] = None) -> float:
    """Measure the gini criteria (entropy)

    Args:
        y (_type_): 1D array of classes
    """
    classes, counts = np.unique(y, return_counts=True)
    wk = np.zeros(len(counts))

    # recompute the
    if weights is not None:
        for i, k in enumerate(classes):
            wk[i] = sum(weights[np.where(y == k)]) / sum(weights)
    else:
        wk = counts / y.shape[0]

    gini = 1 - np.sum(wk**2)
    return gini


def find_mode(y: NDArray, weights: Optional[NDArray] = None):
    classes, counts = np.unique(y, return_counts=True)
    wk = np.zeros(len(counts))

    if weights is not None:
        for i, k in enumerate(classes):
            wk[i] = sum(weights[np.where(y == k)]) / sum(weights)
    else:
        wk = counts / y.shape[0]

    return classes[np.argmax(wk)]
