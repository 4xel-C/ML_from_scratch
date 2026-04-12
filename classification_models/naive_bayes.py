"""Implementatrion of Naive Bayes theorem P(y|X) = P(X|y) * P(y) / p(X)
Posterior = Likelihood * Prior / Evidence

Evidence are considered a constant on a data set, and could be ignored to maximise the posterior.
Prior = Marginal probabilities of the class.
Likelihood = gaussian likelihood (use of mean and sigma)

For the prediction, we maximize the Log(posterior) = argmax(log(likelihood) + log(prior)) for each classes during the prediction
The maximum will determine the class attribution. (this is why we do not need to compute p(X) as we do not want an accurate probability but which class is more likely to fit our data).

Features are independant (naive algorithm), the likelihood of a point correspond to the product of the likelihood of each feature for the said point.
"""

import numpy as np
from numpy.typing import NDArray


class NaiveBayes:
    def fit(self, X: NDArray, y: NDArray) -> None:
        # get the classes and compute the prior
        self.classes, counts = np.unique(y, return_counts=True)
        self.prior = counts / len(y)

        # Initialize conditional means and conditional variances to 0
        self.means = np.zeros((len(self.classes), X.shape[1]))
        self.var = np.zeros((len(self.classes), X.shape[1]))

        # Iterate for each classes
        for i, k in enumerate(self.classes):
            self.means[i] = np.mean(X[y == k], axis=0)
            self.var[i] = np.var(X[y == k], axis=0)

    def predict(self, X_test: NDArray) -> NDArray:
        # for each point, compute the posterior score (using broadcasting), give a shape of (n_test, k) with loglikelihood score
        posterior = (-0.5 * np.log(2 * np.pi * self.var[np.newaxis, :, :])) - (
            (X_test[:, np.newaxis, :] - self.means[np.newaxis, :, :]) ** 2
            / (2 * self.var[np.newaxis, :, :])
        )

        posterior = np.sum(posterior, axis=2)

        # Adding the log(prior) (broadcasting)
        posterior = posterior + np.log(self.prior)[np.newaxis, :]

        # Index of the chosen class
        classes_index = np.argmax(posterior, axis=1)
        predictions = self.classes[classes_index]

        return predictions
