from typing import overload

import numpy as np
from numpy.typing import NDArray


@overload
def sigmoid(x: float) -> float: ...


@overload
def sigmoid(x: NDArray[np.float64]) -> NDArray[np.float64]: ...


def sigmoid(x: float | NDArray[np.float64]) -> float | NDArray[np.float64]:
    """Apply the sigmoid transofrmation function.

    Args:
        x (float): Function entry.
    """
    return 1 / (1 + np.exp(-x))
