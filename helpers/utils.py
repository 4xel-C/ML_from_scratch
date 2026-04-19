import numpy as np
from numpy.typing import NDArray


def gini(y: NDArray) -> float:
    """Measure the gini criteria (entropy)

    Args:
        y (_type_): 1D array of classes
    """
    classes, counts = np.unique(y, return_counts=True)
    pj = counts / y.shape[0]

    gini = 1 - np.sum(pj**2)
    return gini
