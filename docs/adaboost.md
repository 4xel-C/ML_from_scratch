# AdaBoost

**Type:** Supervised — Classification  
**File:** `classification_models/adaboost.py`  
**Depends on:** `DecisionTreeClassifier` (with sample weights)

---

## Problem

Combine many weak learners (shallow trees) into a strong classifier by iteratively focusing on the samples that previous learners got wrong.

---

## Key Idea

Each sample has a weight $w_i$. Misclassified samples get their weight increased, so the next learner focuses on the hard cases. The final prediction is a weighted vote of all learners.

---

## Algorithm

**Initialization:**
$$w_i = \frac{1}{n} \quad \forall i$$

**For each estimator m:**

1. Train a `DecisionTreeClassifier(max_depth=1)` with sample weights $w$.

2. Compute the **weighted error:**
$$\epsilon_m = \frac{\sum_i w_i \cdot \mathbb{1}[\hat{y}_i \neq y_i]}{\sum_i w_i}$$

3. Compute the **learner weight:**
$$\alpha_m = \frac{1}{2} \ln\left(\frac{1 - \epsilon_m}{\epsilon_m}\right)$$

   A learner with error close to 0 gets a large positive weight; error close to 0.5 gets weight near 0.

4. **Update sample weights:**
$$w_i \leftarrow w_i \cdot \exp(-\alpha_m \cdot \text{check}_i)$$

   where $\text{check}_i = +1$ if correctly classified, $-1$ if misclassified. Misclassified samples get their weight multiplied by $e^{\alpha_m} > 1$.

5. **Normalize:** $w_i \leftarrow w_i / \sum_i w_i$

---

## Prediction — SAMME (Multiclass)

Accumulate a score matrix of shape `(n_samples, n_classes)`:

$$\text{score}[i, \hat{y}] \mathrel{+}= \alpha_m$$

Final prediction: class with the highest accumulated score.

```python
scores[np.arange(n), np.searchsorted(classes, predictions)] += alpha[m]
return classes[np.argmax(scores, axis=1)]
```

---

## Modifications to DecisionTreeClassifier

To support sample weights:
- `fit(X, y, weights)` — weights propagated through `_build_tree` and `_best_split`.
- **Weighted Gini:** $p_k = \frac{\sum_{i: y_i=k} w_i}{\sum_i w_i}$ instead of $\frac{n_k}{n}$.
- **Weighted leaf value:** class with the largest total weight (not count).

---

## Comparison with Gradient Boosting

| | AdaBoost | Gradient Boosting |
|---|---|---|
| What the next learner fixes | Misclassified samples (reweighting) | Residual errors (pseudo-residuals) |
| Learner weight | Computed from error (α) | Fixed learning rate |
| Typical base learner | Stump (depth 1) | Shallow tree (depth 3–5) |
