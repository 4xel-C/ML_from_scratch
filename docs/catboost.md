# CatBoost — Categorical Boosting

**Type:** Supervised — Classification  
**File:** `classification_models/catboost.py`  
**Inherits from:** `GradientBoostingClassifier`

---

## Problem

Standard gradient boosting algorithms have two limitations with categorical features:

1. **Encoding problem** — trees cannot split on raw string categories. The standard fix (one-hot encoding) explodes dimensionality and loses inter-category relationships.
2. **Target leakage in encoding** — naive target encoding (replacing a category with the mean of $y$ for that category) uses $y_i$ to encode $x_i$ itself, causing overfitting.

CatBoost solves both with **Ordered Target Encoding**.

---

## Core Idea: Ordered Target Encoding

Instead of encoding each category with the global mean of $y$ (which leaks the target), CatBoost treats the dataset as an **ordered stream**: to encode point $x_i$, only points that appeared **before** $i$ in a random permutation are used.

This guarantees that $y_i$ is **never used** to encode $x_i$.

---

## Mathematical Derivation

### Ordered Target Encoding formula

For point $x_i$, feature column $k$, with category $c = x_i^k$:

$$\hat{x}_i^k = \frac{\displaystyle\sum_{j < i} \mathbf{1}[x_j^k = c] \cdot y_j + a \cdot p}{\displaystyle\sum_{j < i} \mathbf{1}[x_j^k = c] + a}$$

Where:
- $\sum_{j < i} \mathbf{1}[x_j^k = c] \cdot y_j$ — cumulative sum of $y$ for points with category $c$ seen before $i$
- $\sum_{j < i} \mathbf{1}[x_j^k = c]$ — count of points with category $c$ seen before $i$
- $a$ — smoothing parameter (prior strength)
- $p = \mathbb{E}[y]$ — global prior (mean of $y$ over the full training set)

### Bayesian smoothing intuition

The formula is a **weighted average** between the empirical mean of the category and the global prior:

$$\hat{x}_i^k = \frac{n_c \cdot \bar{y}_c + a \cdot p}{n_c + a}$$

Where $n_c$ is the number of observations of category $c$ seen before $i$.

| Case | Result |
|---|---|
| $n_c = 0$ (never seen) | $\hat{x}_i^k = p$ — pure prior |
| $n_c \gg a$ (many observations) | $\hat{x}_i^k \approx \bar{y}_c$ — data dominates |
| $n_c$ small | Prior regularizes the estimate |

The parameter $a$ represents the **weight of the prior** — equivalent to $a$ fictitious observations all with value $p$.

### Why a random permutation?

Without a permutation, the encoding is deterministic and the model could learn to exploit the ordering artifact. The random permutation ensures:
- No point uses its own $y_i$ in its encoding
- The order is unpredictable, preventing memorization

In full CatBoost, a **different permutation is used per tree** (Ordered Boosting). In this implementation, a single permutation is used for all trees to keep complexity manageable.

---

## Gradient Boosting (inherited)

After encoding, the algorithm runs standard gradient boosting (inherited from `GradientBoostingClassifier`):

$$F_0 = \log\frac{p_0}{1 - p_0} \quad \text{(logit of class proportion)}$$

At each iteration $m$:

$$r_i = y_i - \sigma(F_{m-1}(x_i)) \quad \text{(pseudo-residuals, gradient of cross-entropy)}$$

$$F_m = F_{m-1} + \eta \cdot h_m(x) \quad \text{(update with new tree)}$$

$$\hat{y} = \mathbf{1}[\sigma(F_M(x)) \geq 0.5]$$

---

## Predict: encoding new data

At prediction time, $y$ is not available. The encoding uses the **final conditional means** stored during `fit`:

$$\hat{x}^k = \frac{\text{catcumsum}[c]}{\text{catcount}[c]}$$

For an **unknown category** (never seen in training): fall back to the prior $p$.

---

## What this implementation omits

**Ordered Boosting** — in real CatBoost, residuals for point $x_i$ are computed using a model trained only on points $j < i$, removing gradient leakage. This requires $O(n \cdot M)$ models in memory and is not implemented here. We use standard gradient boosting on the encoded matrix instead.

---

## Hyperparameters

| Parameter | Default | Role |
|---|---|---|
| `cat_idx` | required | Indices of categorical feature columns |
| `n_estimators` | 100 | Number of boosting trees |
| `learning_rate` | 0.1 | Step size for boosting updates |
| `max_depth` | 3 | Max depth of each decision tree |
| `a` | 1.0 | Bayesian smoothing strength — higher = more prior weight |

---

## Implementation Notes

- `_encode_categorical` uses a single random permutation generated once at `fit` time.
- Non-categorical columns are copied as-is to `Xresult` (converted to `float`).
- `self.cat_values[idx][mod]` stores the final conditional mean per column/category — used at predict time.
- `self.prior` = `np.mean(y)` — stored during `fit`, used as fallback for unknown categories at predict time.
- The encoded matrix `X_encoded` is float — the original `X` may contain strings in categorical columns.

---

## Pipeline

```
fit(X, y):
    prior = mean(y)
    perm = random_permutation(n)
    for each categorical column idx:
        for i in perm:
            encode x_i using cumulative stats of points before i
            update cat_count[mod], cat_cumsum[mod]
        store final conditional means in cat_values[idx]
    GradientBoostingClassifier.fit(X_encoded, y)

predict(X):
    for each categorical column idx:
        replace each category c with cat_values[idx][c]
        unknown category → prior
    GradientBoostingClassifier.predict(X_encoded)
```
