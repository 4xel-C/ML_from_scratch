# Logistic Regression

**Type:** Supervised — Classification  
**File:** `classification_models/logistic_regression.py`

---

## Problem

Binary classification: predict $P(y=1 \mid X) \in [0, 1]$ from a linear combination of features.

---

## Model

A linear combination (logit) is squashed through the sigmoid function to produce a probability:

$$\text{logit} = Xw$$

$$\hat{p} = \sigma(\text{logit}) = \frac{1}{1 + e^{-Xw}}$$

The sigmoid maps any real value to $(0, 1)$, making it interpretable as a probability.

**Sigmoid derivative:**

$$\sigma'(z) = \sigma(z)(1 - \sigma(z))$$

---

## Loss Function — Binary Cross-Entropy (BCE)

$$\mathcal{L}(w) = -\frac{1}{n}\left[y^T \log(\hat{p}) + (1-y)^T \log(1 - \hat{p})\right]$$

BCE penalizes confident wrong predictions heavily (log goes to $-\infty$ when predicting 0 for a true 1).

**Numerical stability:** probabilities are clipped to $[10^{-15}, 1 - 10^{-15}]$ to avoid $\log(0)$.

---

## Gradient

Despite using a different loss than linear regression, the gradient has the same form:

$$\frac{\partial \mathcal{L}}{\partial w} = \frac{1}{n} X^T(\hat{p} - y)$$

This comes from the chain rule: $\frac{\partial \text{BCE}}{\partial \hat{p}} \cdot \frac{\partial \hat{p}}{\partial \text{logit}} = \hat{p} - y$, with the $\sigma'$ terms canceling.

---

## Optimization — Gradient Descent

```
Initialize w = 0
while iteration < n_iterations and loss > tol:
    p    = sigmoid(X @ w)
    loss = BCE(p, y)
    dw   = (1/n) * X^T @ (p - y)
    w    = w - lr * dw
```

---

## Prediction

$$\hat{y} = \mathbb{1}[\hat{p} \geq 0.5]$$

The decision boundary is the hyperplane where $\hat{p} = 0.5$, i.e., where $Xw = 0$.
