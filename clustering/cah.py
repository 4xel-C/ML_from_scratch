"""
Implémentation of the CAH algorithm. Will use 3 type of linkages,
Using Lance-Williams formula to recompute the distance of the new clusters
with the other points / clusters without recomputing the distance of each point.
"""

from dataclasses import dataclass
from typing import Dict, List, Literal, Set, Tuple

import numpy as np
from numpy.typing import NDArray


# TODO: create a dataclass for clusters
class CAH:
    def __init__(self, linkage: Literal["average", "ward", "complete", "single"]):
        if linkage not in ["average", "ward", "complete", "single"]:
            raise ValueError(
                "Wrong linkage type used. Please choose: " + "average, ",
                "ward, ",
                "complete, ",
                "single.",
            )

        self.linkage = linkage

    def fit(self, X: NDArray):
        n = len(X)

        # Initialise set of active clusters ids
        self.actives: Set[int] = set(i for i in range(n))

        # Clusters constitution, initialization each point to each clusters
        self.clusters: Dict[int, Dict] = dict()

        for i in range(len(X)):
            self.clusters[i]["points"] = [i]
            self.clusters[i]["barycenter"] = X[i]
            self.clusters[i]

        # History of aggregation
        self.history: List[Tuple] = list()

        # Compute the distance matrix
        self.D = X[:, np.newaxis, :] - X[np.newaxis, :, :]
        self.D = np.sum(self.D**2, axis=2)
        self.D = np.sqrt(self.D)

        # Fill diagonal with infinite values to compute the minimum
        np.fill_diagonal(self.D, np.inf)

        # Clustering loop
        while len(self.actives) != 1:
            # Get the minimum ignoring the diagonal and the aggregated clusters
            minimum_distance, idx1, idx2 = np.inf, -1, -1

            for member1 in self.actives:
                for member2 in self.actives:
                    if self.D[member1][member2] < minimum_distance:
                        minimum_distance = self.D[member1][member2]
                        idx1 = member1
                        idx2 = member2

            # Update the history
            self.history.append(
                (
                    idx1,
                    idx2,
                    minimum_distance,
                    len(self.clusters[idx1]["points"])
                    + len(self.clusters[idx2]["points"]),
                )
            )

            # Compupte the new distance
            new_distance = self._compute_distance(
                idx1, idx2, minimum_distance, method=self.linkage
            )

            # We consider idx2 as unactive and will compute only for idx1
            # Update the clusters
            self.clusters[idx1]["points"] = (
                self.clusters[idx1]["points"] + self.clusters[idx2]["points"]
            )

            new_barycenter = np.zeros(X.shape[1])

            for point in self.clusters[idx1]["points"]:
                new_barycenter += X[point]

            new_barycenter = new_barycenter / len(self.clusters[idx1]["points"])

            self.clusters[idx1]["barycenter"] = new_barycenter

            # Deactivate cluster 2
            self.actives.remove(idx2)

    def predict(self, height: float):
        """Predict the clusters.

        Args:
            height (float): Clustering height
        """

    # TODO: implement the calculation (for ward too using Lance-Williams formula)
    def _compute_distance(
        self, index1: int, index2: int, distance12: float, method: str
    ):
        n1 = len(self.clusters[index1])
        n2 = len(self.clusters[index2])

        if method == "complete":
            a = 1 / 2
            b = 1 / 2
            c = 0
            d = 1 / 2

        elif method == "single":
            a = 1 / 2
            b = 1 / 2
            c = 0
            d = -1 / 2

        elif method == "average":
            a = n1 / (n1 + n2)
            b = n2 / (n1 + n2)
            c = 0
            d = 0

        elif method == "ward":
            ...


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    import numpy as np
    from sklearn.cluster import KMeans as SklearnKMeans
    from sklearn.datasets import make_blobs
    from sklearn.metrics import adjusted_rand_score

    X, y_true = make_blobs(n_samples=10, centers=3, cluster_std=0.8, random_state=42)  # type: ignore

    cah = CAH("average")

    cah.fit(X)
