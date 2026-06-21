# DBSCAN

**Type:** Unsupervised — Clustering  
**File:** `clustering/dbscan.py`

---

## Problem

Cluster points based on density rather than distance to a centroid. Unlike K-Means, DBSCAN:
- Does not require specifying k in advance.
- Can find clusters of arbitrary shape.
- Natively identifies **outliers** (noise points).

---

## Key Concepts

**epsilon (ε):** radius of the neighborhood around a point.

**min_samples:** minimum number of neighbors within ε for a point to be a core point.

**Three types of points:**

| Type | Definition |
|---|---|
| **Core point** | Has ≥ min_samples neighbors within ε (excluding itself) |
| **Border point** | Is within ε of a core point, but is not itself a core point |
| **Outlier** | Not within ε of any core point — labeled -1 |

---

## Algorithm — BFS Expansion

```
For each unvisited point i:
    if i is not a core point: skip
    start a new cluster
    push i onto a stack
    while stack is not empty:
        pop point p
        assign p to current cluster
        for each neighbor q of p within ε:
            if q is unvisited:
                assign q to current cluster
                if q is a core point: push q onto stack
    increment cluster number
```

---

## Distance Matrix

Computed once upfront for all pairs:

```python
dist_matrix = np.sqrt(
    np.sum((X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2, axis=2)
)  # shape (n, n)
```

---

## Core Point Check

```python
def _is_core(self, point):
    # Exclude the point itself (distance > 0) and count neighbors within epsilon
    return len(point[(point <= self.epsilon) & (point > 0)]) >= self.min_samples
```

---

## Effect of Hyperparameters

| Parameter | Small value | Large value |
|---|---|---|
| `epsilon` | Many small clusters, many outliers | Fewer large clusters, fewer outliers |
| `min_samples` | Many core points, fewer outliers | Few core points, many outliers |

---

## Limitations

- $O(n^2)$ memory for the distance matrix — not scalable to very large datasets without approximate neighbors.
- Struggles with clusters of varying density (use HDBSCAN instead).
