"""Mean Square Error function."""

import numpy as np
from numpy.typing import NDArray


def mse(y_pred: NDArray, y_true: NDArray) -> float:
    y_pred_arr = np.asarray(y_pred, dtype=np.float64)
    y_true_arr = np.asarray(y_true, dtype=np.float64)
    return float(np.mean((y_pred_arr - y_true_arr) ** 2))
