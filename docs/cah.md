# CAH — Classification Ascendante Hiérarchique

**Type:** Unsupervised — Clustering  
**File:** `clustering/cah.py`  
**Linkages:** `single`, `complete`, `average`, `ward`

---

## Problem

Cluster n points without specifying k in advance. The algorithm builds a full hierarchy of merges — the number of clusters is chosen **after the fact** by cutting the dendrogram at a chosen height.

---

## Algorithm — Bottom-Up Merging

Each point starts as its own cluster. At each step, the two closest clusters are merged. After n-1 merges, a single cluster remains.

```
Initialize: clusters = {i: [i] for i in range(n)}
            actives  = {0, 1, ..., n-1}

while len(actives) > 1:
    (idx1, idx2) = argmin D[i, j] over i, j in actives, i != j
    record fusion in history: (idx1, idx2, distance, new_size)
    update D using Lance-Williams formula
    merge idx2 into idx1
    remove idx2 from actives
```

**History** stores n-1 tuples `(idx1, idx2, distance, size)` — one per merge.

---

## Distance Matrix

Computed once at the start from the raw data:

```python
diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]   # shape (n, n, p)
D    = np.sqrt(np.sum(diff**2, axis=2))              # shape (n, n)
np.fill_diagonal(D, np.inf)                          # avoid self-merge
```

After each merge, D is updated in-place using Lance-Williams — no recomputation from raw points.

---

## Lance-Williams Formula

After merging clusters A and B into AB, the distance to any other cluster C is updated as:

```
d(AB, C) = a * d(A, C) + b * d(B, C) + c * d(A, B) + d * |d(A, C) - d(B, C)|
```

Coefficients by linkage (n1 = |A|, n2 = |B|, n3 = |C|):

| Linkage | a | b | c | d | Intuition |
|---|---|---|---|---|---|
| Single | 1/2 | 1/2 | 0 | -1/2 | d(AB,C) = min(d(A,C), d(B,C)) |
| Complete | 1/2 | 1/2 | 0 | +1/2 | d(AB,C) = max(d(A,C), d(B,C)) |
| Average | n1/(n1+n2) | n2/(n1+n2) | 0 | 0 | weighted mean of distances |
| Ward | (n1+n3)/(n1+n2+n3) | (n2+n3)/(n1+n2+n3) | -n3/(n1+n2+n3) | 0 | minimizes intra-cluster variance |

**Key insight for average linkage:** the weights n1/(n1+n2) and n2/(n1+n2) ensure each individual point contributes equally — not each cluster.

**Key insight for Ward:** Ward minimizes the increase in total intra-cluster variance after a merge. The variance increase when merging A and B is:

```
delta = (n1 * n2) / (n1 + n2) * ||m_A - m_B||^2
```

where m_A and m_B are the centroids of A and B. Lance-Williams lets us track this recursively without storing centroids explicitly, since each point is initially its own centroid.

---

## Matrix Update

After merging idx1 and idx2, the row and column of idx1 are updated, idx2 is deactivated:

```python
D[index1, :] = new_distances
D[:, index1] = new_distances   # keep matrix symmetric
D[index1, index1] = np.inf     # prevent self-merge
```

The row/column of idx2 becomes irrelevant since idx2 is removed from `actives`.

---

## Cutting the Dendrogram

To obtain k clusters, replay the merge history and stop when the merge distance exceeds the cut height:

```python
clusters = {i: [i] for i in range(n)}

for (idx1, idx2, distance, size) in history:
    if distance > height:
        break
    clusters[idx1] += clusters[idx2]
    del clusters[idx2]
```

**Choosing the cut height:** after (n - k) merges, exactly k clusters remain. A natural cut height is the midpoint between merge n-k-1 and merge n-k:

```python
cut_height = (history[n - k - 1][2] + history[n - k][2]) / 2
```

---

## Linkage Comparison

| Linkage | Cluster shape | Sensitivity to outliers |
|---|---|---|
| Single | Elongated, chaining effect | High |
| Complete | Compact, similar sizes | Low |
| Average | Compromise | Moderate |
| Ward | Compact, similar sizes | Low |

---

## Complexity

- Distance matrix: O(n^2) space and time to compute.
- Main loop: O(n^2) iterations, each scanning actives for the minimum — O(n^3) total.
- Not suited for very large datasets without approximations (e.g. scipy uses optimized algorithms).
