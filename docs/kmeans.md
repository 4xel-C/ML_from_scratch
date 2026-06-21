# K-Means Clustering

**Type:** Unsupervised — Clustering  
**File:** `clustering/kmeans.py`

---

## Problem

Partition n points into k clusters by minimizing the total within-cluster variance (inertia):

$$\mathcal{L} = \sum_{k=1}^{K} \sum_{x \in C_k} \|x - \mu_k\|^2$$

---

## Algorithm — EM (Expectation-Maximization)

K-Means alternates between two steps:

**E-step (Assignment):** assign each point to its nearest centroid.

$$c_i = \arg\min_k \|x_i - \mu_k\|^2$$

**M-step (Update):** recompute each centroid as the mean of its assigned points.

$$\mu_k = \frac{1}{|C_k|} \sum_{x \in C_k} x$$

---

## Initialization — K-Means++

Naive random initialization can lead to poor clusters. K-Means++ spreads the initial centroids:

1. Pick the first centroid uniformly at random.
2. For each subsequent centroid, pick point $x$ with probability proportional to $d(x)^2$, where $d(x)$ is the distance to the nearest existing centroid.

This ensures centroids are spread out, reducing the chance of bad local minima.

---

## Vectorized Distance Computation

Distance matrix between all n points and k centroids in one NumPy operation:

```python
distances = np.linalg.norm(
    X[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2
)  # shape (n, k)
```

---

## Convergence

The algorithm stops when the total centroid displacement falls below tolerance:

$$\delta = \sum_k \|\mu_k^{\text{new}} - \mu_k^{\text{old}}\| < \epsilon$$

---

## Limitations

- Requires k to be specified in advance.
- Sensitive to outliers (mean is not robust).
- Assumes spherical clusters of similar size.
- Can converge to local minima (mitigated by K-Means++ initialization).
