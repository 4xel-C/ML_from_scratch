# SVM — Support Vector Machine

**Type:** Supervised — Classification  
**File:** `classification_models/svm.py`  
**Classes:** `SVMClassifier` (linear), `KernelSVM` (RBF)

---

## Geometric Intuition

The SVM finds the hyperplane $w^Tx + b = 0$ that **maximizes the margin** between two classes. The margin is the distance between the two support hyperplanes $w^Tx + b = \pm 1$, which equals $\frac{2}{\|w\|}$.

Labels are remapped to $\{-1, +1\}$ so that correctly classified points satisfy:

$$y_i(w^Tx_i + b) \geq 1$$

---

## Hard Margin (Primal)

$$\min_{w, b} \frac{1}{2}\|w\|^2 \quad \text{s.t.} \quad y_i(w^Tx_i + b) \geq 1 \quad \forall i$$

This requires the data to be linearly separable.

---

## Soft Margin

For non-separable data, introduce slack variables $\xi_i \geq 0$:

$$\min_{w, b, \xi} \frac{1}{2}\|w\|^2 + C \sum_i \xi_i \quad \text{s.t.} \quad y_i(w^Tx_i + b) \geq 1 - \xi_i$$

Substituting the optimal $\xi_i = \max(0, 1 - y_i(w^Tx_i + b))$ gives the **hinge loss**:

$$\mathcal{L} = \frac{1}{2}\|w\|^2 + C \sum_i \max(0, 1 - y_i(w^Tx_i + b))$$

**C large:** strong penalty on violations → narrow margin.  
**C small:** more violations tolerated → wider margin.

---

## SVMClassifier — Gradient Descent on Hinge Loss

**Gradients:**

| Condition | $\partial \mathcal{L}/\partial w$ | $\partial \mathcal{L}/\partial b$ |
|---|---|---|
| $y_i(w^Tx_i+b) \geq 1$ (correct) | $w$ | $0$ |
| $y_i(w^Tx_i+b) < 1$ (violation) | $w - C \sum y_i x_i$ | $-C \sum y_i$ |

The update is vectorized over all violating points:

```python
mask = np.where(y * dist < 1)
dw = w - C * (y[mask] @ X[mask])
db = -C * np.sum(y[mask])
```

---

## Kernel SVM — Dual Formulation

### Why the Dual?

For non-linear boundaries, we project data into a high-dimensional space $\phi(x)$. The dual shows that data only appears as **dot products** $\phi(x_i)^T\phi(x_j)$, which can be replaced by a kernel function $K(x_i, x_j)$ without computing $\phi$ explicitly.

### Dual Problem

From the KKT conditions: $w = \sum_i \alpha_i y_i x_i$, substituting into the Lagrangian gives:

$$\max_\alpha \sum_i \alpha_i - \frac{1}{2} \sum_{i,j} \alpha_i \alpha_j y_i y_j K(x_i, x_j)$$

$$\text{s.t.} \quad 0 \leq \alpha_i \leq C, \quad \sum_i \alpha_i y_i = 0$$

### RBF Kernel

$$K(x_i, x_j) = \exp(-\gamma \|x_i - x_j\|^2)$$

Measures similarity: close to 1 for nearby points, close to 0 for distant points. $\gamma$ controls the width of the Gaussian — large $\gamma$ = narrow, flexible boundary.

### Gram Matrix

$$K_{ij} = K(x_i, x_j) \quad \text{shape } (n, n)$$

Computed as:
```python
diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]  # (n1, n2, p)
gram = np.exp(-gamma * np.sum(diff**2, axis=2))       # (n1, n2)
```

### QP Formulation (cvxopt)

| Variable | Value |
|---|---|
| P | $(y \cdot y^T) \odot K$ |
| q | $-\mathbf{1}$ |
| G | $\begin{pmatrix}-I \\ I\end{pmatrix}$ |
| h | $\begin{pmatrix}\mathbf{0} \\ C \cdot \mathbf{1}\end{pmatrix}$ |
| A | $y^T$ |
| b | $0$ |

### Bias and Prediction

After solving for $\alpha$, support vectors are points where $\alpha_i > 10^{-5}$.

For each support vector $i$:
$$b_i = y_i - \sum_j \alpha_j y_j K(x_j, x_i)$$
$$b = \text{mean}(b_i)$$

Prediction for a new point $x$:
$$f(x) = \sum_j \alpha_j y_j K(x_j, x) + b$$
$$\hat{y} = \text{sign}(f(x))$$
