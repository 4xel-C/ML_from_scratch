# UMAP — Uniform Manifold Approximation and Projection

**Type:** Unsupervised — Dimensionality Reduction / Visualization  
**File:** `dimensionality_reduction/umap.py`

---

## Problem

Project data from high dimension to 2D (or low-D) for visualization, while preserving **both local and global structure** — clusters that are close in the original space remain close, and clusters that are far apart remain separated.

---

## Intuition

t-SNE models neighborhoods as probabilities and minimizes KL divergence — but it only preserves local structure and distorts global relationships.

UMAP models the data as a **weighted neighborhood graph** and optimizes 2D positions to reconstruct that graph via binary cross-entropy. The key differences:

- Distances are adapted **locally** per point (like t-SNE's perplexity, but via a different mechanism)
- Symmetrization uses **fuzzy set union** instead of averaging
- The loss function penalizes **both** false attractions and false repulsions → global structure is preserved
- **Negative sampling** makes optimization tractable without computing all n² pairs

---

## Mathematical Derivation

### Step 1 — k Nearest Neighbors

For each point i, find the k closest points (indices and distances).

Stored as two matrices of shape (n, k):
- `knn_indices[i]` — indices of the k neighbors of point i
- `knn_distances[i]` — corresponding Euclidean distances

---

### Step 2 — Local adaptive distances

For each point i, define:

rho_i = distance to the nearest neighbor of i

Shift all distances: d'_ij = d_ij - rho_i

This ensures the nearest neighbor always gets a weight of 1 (exp(0) = 1), regardless of local density.

---

### Step 3 — Graph weights via binary search on sigma

The weight of the edge (i, j) is:

w_ij = exp(-d'_ij / sigma_i)

sigma_i is calibrated per point by binary search so that:

sum_j w_ij = log2(k)

This constrains the effective number of neighbors to k — analogous to the perplexity constraint in t-SNE.

When sigma is small, all weights collapse to 0. When sigma grows, weights tend to 1 and their sum tends to k. The binary search finds the sigma such that the sum equals log2(k) within tolerance.

---

### Step 4 — Symmetrization via fuzzy union

w_ij != w_ji in general (asymmetric graph). Symmetrize using the **fuzzy set union**:

p_ij = w_ij + w_ji - w_ij * w_ji

This is the probability union formula P(A or B) = P(A) + P(B) - P(A)*P(B), assuming independence.

Interpretation: the edge (i,j) exists if either i considers j a neighbor, or j considers i a neighbor.

Computed as a sparse n x n matrix via `scipy.sparse.csr_matrix`.

---

### Step 5 — Initialization in reduced space

Y ~ N(0, 1e-4) of shape (n, n_components)

Small variance ensures a strong gradient signal from the start (same reasoning as t-SNE).

---

### Step 6 — Similarity in reduced space

UMAP uses a generalized t-Student kernel:

q_ij = 1 / (1 + a * d_ij^(2b))

Parameters a and b are found by fitting this curve to a target defined by min_dist:

- f(d) = 1 if d < min_dist
- f(d) = exp(-(d - min_dist)) if d >= min_dist

`scipy.optimize.curve_fit` finds a and b numerically. The effect: points within min_dist of each other in the reduced space have similarity 1 (considered identical); beyond min_dist, similarity decays.

---

### Step 7 — Loss: Binary Cross-Entropy on edges

For each pair (i, j):

L_ij = -p_ij * log(q_ij) - (1 - p_ij) * log(1 - q_ij)

This penalizes:
- p_ij high, q_ij low → neighbors in original space that are far in reduced space (attraction failure)
- p_ij low, q_ij high → non-neighbors that end up too close (repulsion failure)

Contrast with t-SNE's KL divergence, which only penalizes the first case — UMAP preserves global structure too.

---

### Step 8 — Gradient

Full gradient for each point i, summed over all pairs j:

dL/dy_i = sum_j [ 2ab * d^(2b-1) * (p*(1-q) - q*(1-p)) / (q*(1-q)*(1+a*d^(2b))^2) ] * (yi-yj)/d

Three factors:
- 2ab * d^(2b-1) / (1+a*d^(2b))^2 — derivative of q_ij with respect to d_ij
- (p*(1-q) - q*(1-p)) / (q*(1-q)) — BCE error signal
- (yi - yj) / d — unit direction vector from j to i

---

### Step 9 — Negative sampling

Computing the full gradient over all n² pairs is O(n²). Instead, for each point i per epoch:

1. Compute the **attractive gradient** over its connected neighbors (sparse, from P)
2. Sample `n_negative_samples` random non-neighbors and compute the **repulsive gradient** (p_ij = 0)

For repulsive pairs, p_ij = 0 so the BCE simplifies to: -log(1 - q_ij)

Update: Y -= learning_rate * gradients

---

## Hyperparameters

| Parameter | Role |
|---|---|
| `n_neighbors` | Number of neighbors per point — controls how much local vs global structure is preserved |
| `n_components` | Output dimension (usually 2 for visualization) |
| `min_dist` | Minimum distance between points in reduced space — controls cluster compactness |
| `learning_rate` | Gradient descent step size |
| `n_epochs` | Number of optimization iterations |
| `n_negative_sample` | Number of repulsive pairs sampled per point per epoch |

---

## Implementation Notes

- `fit_transform(X)` is the only public method — UMAP does not generalize to new points in this implementation.
- `_binary_search_sigma` is applied row-by-row via `np.apply_along_axis` — one sigma per point.
- The sparse matrix P is built with `csr_matrix` and symmetrized with `.multiply()` (element-wise), not `*` (matrix product).
- `_find_ab` uses `scipy.optimize.curve_fit` on a linspace grid — purely numerical, no closed form.
- `_grad_calculation` clips d to 1e-10 and the denominator to 1e-10 to avoid division by zero.
- t-SNE uses a normalized q_ij (sum over all pairs = 1); UMAP's q_ij is **not normalized** — each pair is independent.

---

## Pipeline

```
fit_transform(X):
    distances = euclidean(X)                     # (n, n)
    knn_indices, knn_distances = knn(distances)  # (n, k) each

    rho = knn_distances[:, 0]                    # (n,) nearest neighbor distance
    d_shifted = knn_distances - rho              # (n, k) shifted distances

    for each i:
        binary_search sigma_i → sum(exp(-d/sigma)) = log2(k)

    w = exp(-d_shifted / sigma)                  # (n, k) edge weights
    P = sparse(w)                                # (n, n) sparse graph
    P = P + P.T - P * P.T                        # fuzzy union symmetrization

    a, b = curve_fit(student_t_ab, min_dist)     # fit reduced space kernel
    Y = randn(n, n_components) * 1e-4            # random init

    for epoch in range(n_epochs):
        dist_reduced = euclidean(Y)              # (n, n)
        Q = 1 / (1 + a * dist^(2b))             # (n, n)

        for each i:
            grad_attract = _grad_calculation(connected neighbors, p > 0)
            grad_repulse = _grad_calculation(n_negative_sample random non-neighbors, p = 0)
            gradients[i] = grad_attract + grad_repulse

        Y -= lr * gradients

    return Y
```

---

## UMAP vs t-SNE

| | t-SNE | UMAP |
|---|---|---|
| Similarity model | Gaussian + t-Student | Fuzzy graph + generalized t-Student |
| Loss | KL divergence | Binary cross-entropy |
| Repulsion | Implicit (normalization of Q) | Explicit (negative sampling) |
| Global structure | Lost | Partially preserved |
| Speed | O(n²) | O(n * k) with negative sampling |
| New point projection | Not possible | Not possible (in this implementation) |
