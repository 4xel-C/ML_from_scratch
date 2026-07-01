# t-SNE — t-Distributed Stochastic Neighbor Embedding

**Type:** Unsupervised — Dimensionality Reduction / Visualization  
**File:** `dimensionality_reduction/tsne.py`

---

## Problem

Project data from high dimension (e.g. 64 features) to 2D for visualization, while preserving the **local neighborhood structure** — points that are close in the original space remain close in 2D.

---

## Intuition

PCA finds directions of maximum variance (linear, global). t-SNE instead models **neighborhood relationships as probability distributions** and minimizes the divergence between the original distribution (high dimension) and the reduced one (2D).

Key idea: convert distances into probabilities of being neighbors, then optimize 2D coordinates so those probabilities match.

---

## Mathematical Derivation

### Step 1 — Distances in original space

$$D_{ij} = \|x_i - x_j\|^2 \quad \text{shape } (n, n)$$

Squared Euclidean distances — used directly in the Gaussian kernel.

---

### Step 2 — Neighborhood probabilities in high dimension

For each point $i$, compute the conditional probability that $x_j$ is its neighbor using a **Gaussian kernel**:

$$p_{j|i} = \frac{\exp\left(-D_{ij} / 2\sigma_i^2\right)}{\sum_{k \neq i} \exp\left(-D_{ik} / 2\sigma_i^2\right)}$$

- $p_{i|i} = 0$ enforced by setting $D_{ii} = +\infty$ before computing the kernel.
- Each $\sigma_i$ is **specific to point $i$** — it adapts to local density.

---

### Step 3 — Finding $\sigma_i$ via binary search on perplexity

The perplexity controls the effective number of neighbors:

$$\text{Perplexity}(P_i) = 2^{H(P_i)} \quad \text{where} \quad H(P_i) = -\sum_j p_{j|i} \log_2 p_{j|i}$$

- High entropy → uniform distribution → many effective neighbors → large $\sigma_i$ (sparse zone)
- Low entropy → peaked distribution → few effective neighbors → small $\sigma_i$ (dense zone)

Binary search adjusts $\sigma_i$ until $\text{Perplexity}(P_i) \approx \text{perplexity target}$ (default: 30).  
Search bounds: $\sigma \in [10^{-10},\ \sqrt{\max(D)}]$ to adapt to the scale of the data.

---

### Step 4 — Symmetrization

$p_{j|i} \neq p_{i|j}$ in general (asymmetric: a point in a dense zone has many close neighbors so an isolated point matters little to it). Symmetrize into a **joint distribution over all pairs**:

$$p_{ij} = \frac{p_{j|i} + p_{i|j}}{2n}$$

The $2n$ normalization ensures $\sum_{i,j} p_{ij} = 1$ over the full matrix — $P$ is a valid global probability distribution over pairs.

---

### Step 5 — Probabilities in low dimension (t-Student kernel)

Initialize 2D coordinates $Y \sim \mathcal{N}(0, 10^{-4})$ (small scale to ensure strong initial gradient signal).

Compute neighborhood probabilities using a **t-Student kernel with 1 degree of freedom** (Cauchy distribution):

$$q_{ij} = \frac{(1 + \|y_i - y_j\|^2)^{-1}}{\sum_{k \neq l}(1 + \|y_k - y_l\|^2)^{-1}}$$

$q_{ii} = 0$ enforced by setting $D_{ii} = +\infty$.

**Why t-Student instead of Gaussian?**  
The t-Student has heavier tails: it decays much more slowly than $\exp(-d^2)$. In 2D there is less space than in high dimension (crowding problem) — the heavy tails allow t-SNE to push clusters far apart without being penalized, while still keeping local neighbors tightly grouped.

Comparison at $d = 3$: Gaussian gives $e^{-9} \approx 0.0001$, t-Student gives $(1+9)^{-1} = 0.1$.

---

### Step 6 — Loss function: KL Divergence

Measure the divergence between $P$ (fixed, high-dim) and $Q$ (variable, low-dim):

$$C = KL(P \| Q) = \sum_{i,j} p_{ij} \log \frac{p_{ij}}{q_{ij}}$$

This divergence **heavily penalizes** cases where $p_{ij}$ is large but $q_{ij}$ is small (close points in high dimension that end up far in 2D). It penalizes less the reverse case — which is coherent with the goal of preserving local structure.

---

### Step 7 — Gradient and optimization

Gradient of the KL divergence with respect to $y_i$:

$$\frac{\partial C}{\partial y_i} = 4 \sum_j (p_{ij} - q_{ij})(y_i - y_j)(1 + \|y_i - y_j\|^2)^{-1}$$

Three factors:
- $(p_{ij} - q_{ij})$: the error — should we attract or repel point $j$?
- $(y_i - y_j)$: the direction to move in 2D space
- $(1 + \|y_i - y_j\|^2)^{-1}$: the t-Student kernel — attenuates the influence of distant points

**Vectorized implementation** using broadcasting:

```python
D_low   = distances(Y)                          # (n, n)
diff    = Y[:, newaxis, :] - Y[newaxis, :, :]   # (n, n, d)
grad    = 4 * sum(
    (P - Q)[:, :, newaxis]                      # (n, n, 1)
    * diff                                       # (n, n, d)
    * (1 + D_low[:, :, newaxis])**-1,            # (n, n, 1)
    axis=1                                       # sum over j → (n, d)
)
```

Update rule (gradient descent):

$$Y \leftarrow Y - \eta \cdot \nabla_Y C$$

Convergence: stop when $|L_t - L_{t-1}| < \text{tol}$.

---

## Hyperparameters

| Parameter | Default | Role |
|---|---|---|
| `perplexity` | 30.0 | Effective number of neighbors per point. Typical range: 5–50 |
| `n_dimensions` | 2 | Output space dimension (usually 2 for visualization) |
| `learning_rate` | 200.0 | Step size for gradient descent |
| `max_iterations` | 1000 | Max optimization steps |
| `binary_max_iterations` | 50 | Max iterations for binary search on $\sigma_i$ |
| `tolerance` | 1e-7 | Convergence threshold on KL divergence variation |

---

## Implementation Notes

- `fit_transform(X)` is the only public method — t-SNE does not generalize to new points.
- `_compute_distances` is reused for both high-dim (`X`) and low-dim (`Y`) distance matrices.
- The `probas_matrix` in `_computy_high_dimensionality_proba` modifies `D` in place via `np.fill_diagonal` — do not reuse `D` after this call.
- `sigma_right = np.sqrt(np.max(D))` computed **before** `fill_diagonal` to avoid picking up `inf`.
- t-SNE is stochastic (random init of `Y`) — results vary across runs.
- t-SNE complexity is $O(n^2)$ in memory and time — not suited for very large datasets (n > 10 000).

---

## Pipeline

```
fit_transform(X):
    D = squared_euclidean_distances(X)           # (n, n)
    P = high_dim_probabilities(D)                # (n, n), symmetric, sums to 1
        for each i:
            binary_search sigma_i → target perplexity
        symmetrize: (P + P.T) / 2n

    Y = randn(n, d) * 1e-4                       # random init in low dim

    for iteration in range(max_iterations):
        Q = low_dim_probabilities(Y)             # t-Student kernel
        loss = KL(P || Q)
        if |loss - prev_loss| < tol: break
        gradient = 4 * sum((P-Q) * diff * kernel, axis=1)
        Y -= lr * gradient

    return Y
```
