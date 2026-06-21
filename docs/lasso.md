# Lasso (L1 Regularization)

**Type:** Supervised — Regression  
**File:** `regression_models/lasso.py`  
**Inherits from:** `LinearRegression`

---

## Problem

Same as linear regression, but with a penalty on the absolute values of weights to promote **sparsity** — many weights are pushed exactly to zero.

---

## Loss Function

$$\mathcal{L}(w) = \underbrace{\frac{1}{n}\|Xw - y\|^2}_{\text{MSE}} + \lambda \|w\|_1$$

where $\|w\|_1 = \sum_j |w_j|$.

---

## Gradient

The L1 norm is not differentiable at 0, but we use the subgradient:

$$\frac{\partial \mathcal{L}}{\partial w} = \frac{1}{n} X^T(Xw - y) + \lambda \cdot \text{sign}(w)$$

**Important:** the bias term is not regularized:

```python
gradient[1:] += lambda * np.sign(w[1:])
```

---

## Why L1 Produces Sparsity

The L1 penalty ball is a **diamond** (rhombus) with corners on the axes. When the MSE contours intersect the L1 ball, contact happens at a corner where one or more weights are exactly 0. By contrast, the L2 ball is a sphere with no corners, so weights are only reduced, never zeroed.

| | Lasso (L1) | Ridge (L2) |
|---|---|---|
| Penalty | $\lambda \|w\|_1$ | $\lambda \|w\|_2^2$ |
| Gradient | $\lambda \cdot \text{sign}(w)$ | $\lambda \cdot w$ |
| Effect | Forces weights to 0 (sparsity) | Shrinks weights |

---

## Optimization

Identical to linear regression gradient descent, with the regularization gradient added at each step. The bias coefficient `w[0]` is excluded from regularization.
