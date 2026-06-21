# Ridge (L2 Regularization)

**Type:** Supervised — Regression  
**File:** `regression_models/ridge.py`  
**Inherits from:** `LinearRegression`

---

## Problem

Same as linear regression, but with a penalty on the squared norm of weights to reduce overfitting without forcing weights to zero.

---

## Loss Function

$$\mathcal{L}(w) = \underbrace{\frac{1}{n}\|Xw - y\|^2}_{\text{MSE}} + \lambda \|w\|_2^2$$

where $\|w\|_2^2 = \sum_j w_j^2$.

---

## Gradient

$$\frac{\partial \mathcal{L}}{\partial w} = \frac{1}{n} X^T(Xw - y) + 2\lambda w$$

The factor 2 is absorbed into $\lambda$ in practice. The bias is excluded from regularization:

```python
gradient[1:] += lambda * w[1:]
```

---

## Why L2 Does Not Produce Sparsity

The L2 penalty ball is a **sphere** — smooth with no corners. The MSE contours always intersect the sphere at a point where all weights are non-zero. Weights are shrunk toward zero but never reach it exactly.

---

## Effect of Lambda

- **Large λ:** strong regularization, weights close to zero, high bias, low variance.
- **Small λ:** weak regularization, weights close to OLS solution.
- **λ = 0:** equivalent to plain linear regression.

---

## Practical Note

Ridge is sensitive to feature scale. Features should be standardized with `StandardScaler` before fitting, otherwise features with large scales dominate the gradient and gradient descent diverges.
