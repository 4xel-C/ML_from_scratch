# PCA — Principal Component Analysis

**Type:** Unsupervised — Dimensionality Reduction  
**File:** `dimensionality_reduction/pca.py`

---

## Problem

Project X of shape (n, p) onto k dimensions (k < p) that capture the maximum variance. This reduces noise, memory usage, and computation while preserving the most informative structure.

---

## Intuition

PCA finds the k orthogonal directions (principal components) along which the data varies the most. Projecting onto these directions discards low-variance dimensions that are likely noise.

---

## Mathematical Derivation

**Step 1 — Center the data:**

$$X_{\text{norm}} = X - \bar{X}$$

Centering ensures that variance is not biased by the mean position.

**Step 2 — Covariance matrix:**

$$C = \frac{1}{n} X_{\text{norm}}^T X_{\text{norm}} \quad \text{shape } (p, p)$$

$C_{ij}$ measures how feature $i$ and feature $j$ vary together.

**Step 3 — Spectral decomposition:**

$$C = V \Lambda V^T$$

- $V$: matrix of eigenvectors (columns), shape $(p, p)$ — the principal directions.
- $\Lambda$: diagonal matrix of eigenvalues — the variance explained by each direction.

**Step 4 — Sort and select:**

Sort eigenvectors by decreasing eigenvalue. Keep the top k: $V_k$ of shape $(p, k)$.

**Step 5 — Project:**

$$Z = X_{\text{norm}} \cdot V_k \quad \text{shape } (n, k)$$

---

## Explained Variance Ratio

$$\text{ratio}_i = \frac{\lambda_i}{\sum_j \lambda_j}$$

Tells you how much of the total variance is captured by component $i$.

---

## Implementation Notes

- `np.linalg.eig` returns eigenvectors as columns — `eigen_vectors[:, i]` is the i-th direction.
- Eigenvectors are defined up to a sign: a component with the opposite sign as sklearn is mathematically equivalent.
- `fit` stores `self.means` and `self.eigen_vectors` (shape `(p, k)`).
- `transform` centers new data with training means before projecting — never recomputes the eigenvectors.

---

## Pipeline

```
fit(X):
    means = X.mean(axis=0)
    X_norm = X - means
    C = (1/n) * X_norm.T @ X_norm
    eigenvalues, eigenvectors = eig(C)
    sort by descending eigenvalue
    keep top k columns

transform(X_new):
    X_centered = X_new - means
    return X_centered @ eigen_vectors  # shape (n, k)
```
