# Linear Regression

**Type:** Supervised — Regression  
**File:** `regression_models/linear_regression.py`

---

## Problem

Given a dataset of n samples with p features, find the linear function that best predicts a continuous target y from input X.

---

## Model

$$\hat{y} = Xw$$

The bias is absorbed into the weight vector by prepending a column of ones to X:

$$X_b = \begin{bmatrix} 1 & x_1 \\ 1 & x_2 \\ \vdots & \vdots \end{bmatrix}, \quad w = \begin{bmatrix} b \\ w_1 \\ \vdots \end{bmatrix}$$

---

## Loss Function — MSE

$$\mathcal{L}(w) = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)^2 = \frac{1}{n} \|Xw - y\|^2$$

MSE is **convex** — there is a single global minimum, so gradient descent is guaranteed to converge.

---

## Gradient

$$\frac{\partial \mathcal{L}}{\partial w} = \frac{2}{n} X^T(Xw - y)$$

The factor 2 is absorbed into the learning rate in practice.

---

## Optimization — Gradient Descent

```
Initialize w = 0
while iteration < max_iterations and loss > tol:
    y_hat = X @ w
    loss  = MSE(y_hat, y)
    dw    = (1/n) * X^T @ (y_hat - y)
    w     = w - lr * dw
```

**Stopping conditions:**
- `loss < tol` (1e-6)
- `iteration >= max_iterations`

**Weight initialization:** zeros — valid because MSE is convex (single minimum).

---

## Sklearn comparison

Sklearn's `LinearRegression` uses the closed-form OLS solution `w = (X^T X)^{-1} X^T y`. Our gradient descent converges to the same solution within ~1% MSE on standard datasets.
