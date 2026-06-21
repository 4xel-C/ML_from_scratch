"""
Implémentation of the CAH algorithm. Will use 3 type of linkages,
Using Lance-Williams formula to recompute the distance of the new clusters
with the other points / clusters without recomputing the distance of each point.

The ward method is calculated considering the distance between the centroids of the cluster,
minimizing the variance intra cluster. Thus, because the centroid is computed by a weighed average
of the previous centroids, we can recursively compute the distances without storing the centroids,
as the first cenroids correspond to the data points themselves (1 data point = 1 cluster, each
being then centroid of their cluster.)
"""

from typing import Dict, List, Literal, Set, Tuple

import numpy as np
from numpy.typing import NDArray


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
        self.X = X
        n = len(X)

        # Initialise set of active clusters ids
        self.actives: Set[int] = set(i for i in range(n))

        # Clusters constitution, initialization each point to each clusters
        self.clusters: Dict[int, List] = dict()

        for i in range(len(X)):
            self.clusters[i] = [i]

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
                    len(self.clusters[idx1]) + len(self.clusters[idx2]),
                )
            )

            # Compupte the new distance
            self._compute_distances(idx1, idx2, minimum_distance, method=self.linkage)

            # We consider idx2 as unactive and will compute only for idx1
            # Merge the clusters and delete the index 2
            self.clusters[idx1] = self.clusters[idx1] + self.clusters.pop(idx2)

            # Deactivate cluster 2
            self.actives.remove(idx2)

    def cut(self, height: float):
        """Predict the clusters by history reconstruction. Stop to the cut height.

        Args:
            height (float): Clustering height
        """

        # Initialize the vector of cluster attributions with one different number for each clusters
        result = np.array(range(len(self.X)))

        # Initialise the cluster object to rebuild the clusters
        clusters: Dict[int, List] = {i: [i] for i in range(len(self.X))}

        for fusion in self.history:
            # Check if we go beyond the specified height
            if fusion[2] > height:
                break

            idx1, idx2 = fusion[0], fusion[1]

            clusters[idx1].extend(clusters[idx2])
            clusters.pop(idx2)

        for i, idx_cluster in enumerate(clusters):
            for point in clusters[idx_cluster]:
                # Cluster attribution with enumerate to have cluster from 0 to k
                result[point] = i

        return result

    def _compute_distances(
        self,
        index1: int,
        index2: int,
        distance12: float,
        method: str,
    ):
        """Compute distances between the two fusionned clusters and all other clusters

        Args:
            index1 (int): The first fusionned cluster
            index2 (int): The second fusionned cluster
            distance12 (float): The distance between cluster 1 and 2
            method (str): The linkage method, to correctly compute the distance
        """
        n1 = len(self.clusters[index1])
        n2 = len(self.clusters[index2])

        # Store the distance to all other clusters to refresh the distance matrix
        distances = np.zeros(self.D.shape[1])

        for index3 in self.actives:
            if index3 == index1 or index3 == index2:
                continue

            n3 = len(self.clusters[index3])

            # Compute the coeficients for the calculation
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
                a = (n1 + n3) / (n1 + n2 + n3)
                b = (n2 + n3) / (n1 + n2 + n3)
                c = -n3 / (n1 + n2 + n3)
                d = 0
            else:
                raise ValueError("Unknown linkage used for _compute_distances")

            # Calculation of the distance with the correct coefficients
            distances[index3] = (
                a * self.D[index1, index3]
                + b * self.D[index2, index3]
                + c * distance12
                + d * abs(self.D[index1, index3] - self.D[index2, index3])
            )

        # Update the distance matrix in row and column
        self.D[index1] = distances
        self.D[:, index1] = distances
        self.D[index1, index1] = np.inf


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    import numpy as np
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.datasets import make_blobs
    from sklearn.metrics import adjusted_rand_score

    X, y_true = make_blobs(n_samples=100, centers=3, cluster_std=0.8, random_state=42)  # type: ignore
    n, k = len(X), 3

    print("=== CAH comparison ===")
    for linkage in ["single", "complete", "average", "ward"]:
        cah = CAH(linkage)  # type: ignore
        cah.fit(X)

        # Find a cut height that yields exactly k clusters:
        # after (n - k) merges we have k clusters; cut between merge n-k-1 and n-k
        cut_height = (cah.history[n - k - 1][2] + cah.history[n - k][2]) / 2
        pred_custom = cah.cut(cut_height)

        sk = AgglomerativeClustering(n_clusters=k, linkage=linkage)  # type: ignore
        pred_sk = sk.fit_predict(X)

        print(f"\nLinkage: {linkage}")
        print(f"  Custom ARI:  {adjusted_rand_score(y_true, pred_custom):.4f}")
        print(f"  Sklearn ARI: {adjusted_rand_score(y_true, pred_sk):.4f}")
