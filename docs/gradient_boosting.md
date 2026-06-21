# Gradient Boosting

**Type:** Supervised — Regression & Classification  
**Files:**
- `regression_models/gradient_boosting.py`
- `classification_models/gradient_boosting_classifier.py`  
**Depends on:** `DecisionTreeRegressor`

---

## Core Idea

Build an additive model by sequentially fitting new trees to the **negative gradient of the loss** with respect to the current predictions. Each tree corrects the residual error of the ensemble so far.

$$F_m(x) = F_{m-1}(x) + \eta \cdot h_m(x)$$

where $h_m$ is trained on the pseudo-residuals and $\eta$ is the learning rate.

---

## Gradient Boosting Regressor

**Loss:** MSE

**Pseudo-residual:** $r_i = y_i - F_{m-1}(x_i)$ (ordinary residuals)

**Pipeline:**
```
F = mean(y)                          # initialization
for m in range(n_estimators):
    r = y - F                        # residuals
    h_m = DecisionTreeRegressor.fit(X, r)
    F = F + eta * h_m.predict(X)
```

**Predict:**
```
F = mean(y_train)
for each tree h_m:
    F = F + eta * h_m.predict(X)
return F
```

---

## Gradient Boosting Classifier

**Loss:** Binary cross-entropy

**Model:** F represents **log-odds** — $p = \sigma(F)$

**Initialization:**

$$F_0 = \log\left(\frac{p_0}{1 - p_0}\right) \quad \text{where } p_0 = \frac{\sum y_i}{n}$$

**Pseudo-residual derivation:**

$$\frac{\partial \mathcal{L}}{\partial F} = p - y \quad \Rightarrow \quad r = -\frac{\partial \mathcal{L}}{\partial F} = y - p = y - \sigma(F)$$

**Pipeline:**
```
F = log(p0 / (1 - p0))              # scalar initialization
r = y - sigmoid(F)
for m in range(n_estimators):
    h_m = DecisionTreeRegressor.fit(X, r)
    F = F + eta * h_m.predict(X)
    r = y - sigmoid(F)              # recompute residuals
```

**Predict:**
```
F = fzero
for each tree: F += eta * tree.predict(X)
p = sigmoid(F)
return (p >= 0.5).astype(int)
```

---

## Comparison

| | GB Regressor | GB Classifier |
|---|---|---|
| Loss | MSE | Cross-entropy |
| Pseudo-residual | $y - F$ | $y - \sigma(F)$ |
| Initialization $F_0$ | $\bar{y}$ | $\log(p_0 / (1-p_0))$ |
| Final prediction | $F$ directly | $\sigma(F) \geq 0.5$ |

---

## Effect of Hyperparameters

| Parameter | Small value | Large value |
|---|---|---|
| `learning_rate` (η) | More trees needed, less overfitting | Fewer trees, risk of overfitting |
| `n_estimators` | Underfitting | Overfitting (if η is too large) |
| `max_depth` | High bias, low variance | Low bias, high variance per tree |

---

## Why Trees Regress on Residuals

Each tree approximates the negative gradient of the loss in function space. For MSE, the negative gradient at step m is exactly $y - F_m$, i.e., the residuals. For cross-entropy, it is $y - \sigma(F_m)$. Gradient boosting generalizes this idea to any differentiable loss.
