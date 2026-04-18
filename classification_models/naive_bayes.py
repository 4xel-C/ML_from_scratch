"""Implementatrion of Naive Bayes theorem P(y|X) = P(X|y) * P(y) / p(X)
Posterior = Likelihood * Prior / Evidence

Evidence are considered a constant on a data set, and could be ignored to maximise the posterior.
Prior = Marginal probabilities of the class.
Likelihood = gaussian likelihood (use of mean and sigma)

For the prediction, we maximize the Log(posterior) = argmax(log(likelihood) + log(prior)) for each classes during the prediction
The maximum will determine the class attribution. (this is why we do not need to compute p(X) as we do not want an accurate probability but which class is more likely to fit our data).

Features are independant (naive algorithm), the likelihood of a point correspond to the product of the likelihood of each feature for the said point.
"""

from collections import defaultdict
from typing import Dict, Optional

import numpy as np
from numpy.typing import NDArray


class NaiveBayes:
    def fit(
        self, X: NDArray, y: NDArray, categorical_features: Optional[NDArray] = None
    ) -> None:
        # Find the numerical and categorical values
        self.cat_features = (
            np.where(categorical_features)[0]
            if categorical_features is not None
            else None
        )

        # get the classes and compute the prior
        self.classes, counts = np.unique(y, return_counts=True)
        self.prior = counts / len(y)

        # select the numerical values
        X_num = X[:, ~categorical_features] if categorical_features is not None else X

        # Initialize conditional means and conditional variances to 0
        self.means = np.zeros((len(self.classes), X_num.shape[1]))
        self.var = np.zeros((len(self.classes), X_num.shape[1]))

        # Iterate for each classes
        for i, k in enumerate(self.classes):
            self.means[i] = np.mean(X_num[y == k], axis=0)
            self.var[i] = np.var(X_num[y == k], axis=0)

        if categorical_features is not None:
            alpha = 1

            # Initialize the dictionnary of features: feature_index -> class_k -> modality -> proportion
            cat_proportions = defaultdict(lambda: defaultdict(dict))

            # select the categorical values
            for index in np.where(categorical_features)[0]:
                for k in self.classes:
                    modalities = np.unique(X[:, index])
                    nb_modalities = len(modalities)
                    nk = np.sum(y == k)

                    for c in modalities:
                        nkc = np.sum(X[y == k, index] == c)

                        # Compute the laplace smoothing
                        cat_proportions[index][k][c] = (nkc + alpha) / (
                            nk + alpha * nb_modalities
                        )

            # Saving the calculated proportions
            self.proportions = cat_proportions

    def predict(self, X_test: NDArray) -> NDArray:
        X_num = (
            X_test[:, ~self.cat_features] if self.cat_features is not None else X_test
        )

        # for each point, compute the posterior score (using broadcasting), give a shape of (n_test, k) with loglikelihood score
        posterior = (-0.5 * np.log(2 * np.pi * self.var[np.newaxis, :, :])) - (
            (X_num[:, np.newaxis, :] - self.means[np.newaxis, :, :]) ** 2
            / (2 * self.var[np.newaxis, :, :])
        )

        posterior = np.sum(posterior, axis=2)

        # Adding the log(proportion) to the posterior if categorical features present
        if self.cat_features is not None:
            for row_idx in range(posterior.shape[0]):
                for feature_idx in self.cat_features:
                    for i, k in enumerate(self.classes):
                        posterior[row_idx, i] += np.log(
                            self.proportions[feature_idx][k][
                                int(X_test[row_idx, feature_idx])
                            ]
                        )

        # Adding the log(prior) (broadcasting)
        posterior = posterior + np.log(self.prior)[np.newaxis, :]

        # Index of the chosen class
        classes_index = np.argmax(posterior, axis=1)
        predictions = self.classes[classes_index]

        return predictions
