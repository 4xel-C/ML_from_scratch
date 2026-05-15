"""Simpler shap algorithm using the monte carlo sampling"""


import numpy as np
from numpy.typing import NDArray
from typing import Any
import random
import math


class Shap():

    def __init__(self, n_sampling: int = 50, background_size: int = 5):
        """Initialize the explainer

        Args:
            n_sampling (int): Number of monte carlo estimation.
            background_size (int): Number of reference points to compute unused features
        """
        self.n_sampling = n_sampling  
        self.m = background_size
    
    def explain(self, X: NDArray, model: Any):

        n_samples, n_features = X.shape
        shap_values = np.zeros((n_samples, n_features))

        # background dataset (imputation reference)
        Z = X[np.random.choice(n_samples, self.m, replace=False)]

        # Iterate over each points
        for point_idx in range(n_samples):

            # isolate the concerned point
            x = X[point_idx]
            point_shap = np.zeros(n_features)

            # Monte carlo sampling loop
            for _ in range(self.n_sampling):

                # 1. baseline sample (get a random background point)
                z = Z[np.random.randint(self.m)].copy()

                # 2. permutation of features (generate a permutation of all features)
                perm = np.random.permutation(n_features)

                # rebuild the input by adding features one by one according to the permutation, and compute marginal contribution of each feature
                x_curr = z.copy()

                # initial prediction
                v_prev = model.predict(x_curr.reshape(1, -1))[0]

                # 3. walk through permutation and rebuild x
                for j in perm:

                    # add feature j
                    x_curr[j] = x[j]

                    # compute new prediction
                    v_next = model.predict(x_curr.reshape(1, -1))[0]

                    # 4. marginal contribution of feature j
                    point_shap[j] += (v_next - v_prev)

                    # update state
                    v_prev = v_next

            # average over permutations
            shap_values[point_idx] = point_shap / self.n_sampling
        
        return shap_values
