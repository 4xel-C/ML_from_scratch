# SHAP Values (Monte Carlo Permutation Sampling)

**Type:** Explainability — Model-agnostic  
**File:** `explainability/shap.py`

---

## Problem

Explain the contribution of each feature to a specific prediction. Unlike global feature importance, SHAP values give a **local, sample-level** explanation.

---

## Foundation — Shapley Values (Game Theory)

A Shapley value is borrowed from cooperative game theory. For a coalition of "players" (features), it measures each player's fair average marginal contribution across all possible coalitions.

$$\phi_i(x) = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!(|F|-|S|-1)!}{|F|!} \left[ f(S \cup \{i\}, x) - f(S, x) \right]$$

where:
- $F$: set of all features
- $S$: a coalition not containing feature $i$
- $f(S, x)$: model prediction when only features in $S$ take the values of $x$, and all other features are drawn from a background distribution

The weight before score difference shows the probability of have the feature F to this specifice position considering all possible permutations

---

## Estimating $f(S, x)$ with a Background Dataset

We cannot set "unused features to nothing" — the model needs a full input vector. Instead, for a background point $z$:

$$f(S, x) \approx \text{model.predict}(\text{hybrid}_{S,x,z})$$

where the hybrid vector takes feature values from $x$ for features in $S$, and from $z$ for features outside $S$.

---

## Monte Carlo Estimation via Permutations

Rather than enumerating all $2^p$ coalitions (exponential), we use random permutations:

**Each permutation defines a coalition:** if feature $i$ appears at position $k$ in the permutation, then $S$ = all features appearing before position $k$.

**Algorithm for one sample $x$:**

```
for _ in range(n_sampling):
    z    = random background point
    perm = random permutation of features
    x_curr = z.copy()
    v_prev = model.predict(x_curr)

    for j in perm:
        x_curr[j] = x[j]              # add feature j
        v_next     = model.predict(x_curr)
        shap[j]   += v_next - v_prev  # marginal contribution of j
        v_prev     = v_next

shap /= n_sampling
```

---

## Why Permutations = Coalitions

Each permutation implicitly defines a coalition $S$ = {features appearing before $i$}. Averaging the marginal contribution of $i$ over all random permutations is equivalent to averaging over all coalitions with the exact Shapley weight $\frac{|S|!(|F|-|S|-1)!}{|F|!}$.

---

## Complexity

$$O(n_\text{samples} \times n_\text{sampling} \times n_\text{features})$$

Linear in all dimensions — tractable even for moderately large datasets.

---

## Properties of Shapley Values

| Property | Meaning |
|---|---|
| **Efficiency** | $\sum_i \phi_i = f(x) - f(\text{baseline})$ |
| **Symmetry** | Features contributing equally get equal SHAP values |
| **Dummy** | A feature that never changes predictions gets SHAP = 0 |
| **Additivity** | For ensemble models, SHAP values add across trees |

---

## Usage

```python
shap = Shap(n_sampling=50, background_size=20)
shap_values = shap.explain(X, model)  # shape (n_samples, n_features)
```

For classification, pass a model with `.predict_proba` or use `.predict` if it returns probabilities.

---

## Convergence

Increase `n_sampling` and `background_size` to reduce variance in the estimates. The ranking of feature importances stabilizes faster than the exact values.
