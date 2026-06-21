# Random Forest

**Type:** Supervised — Classification  
**File:** `classification_models/random_forest.py`  
**Depends on:** `DecisionTreeClassifier`

---

## Problem

A single decision tree has high variance — small changes in training data lead to very different trees. Random Forest reduces variance by averaging many decorrelated trees.

---

## Key Idea — Bagging + Feature Randomness

**Bagging (Bootstrap AGGregatING):** each tree is trained on a different bootstrap sample (n samples drawn with replacement from the training set). On average, each bootstrap sample contains ~63.2% unique samples.

**Feature randomness:** at each split, only a random subset of $\sqrt{p}$ features is considered. This decorrelates the trees — they can't all rely on the same dominant feature.

---

## Algorithm

**Fit:**
```
for each estimator:
    1. Sample n rows with replacement (bootstrap)
    2. Sample sqrt(p) features without replacement
    3. Train a DecisionTreeClassifier on this subset
    4. Store (tree, selected_features)
```

**Predict:**
```
for each (tree, features):
    pred_i = tree.predict(X[:, features])
collect all predictions -> majority vote per sample
```

---

## Why It Works

- Each tree sees a different training set → different errors → averaging reduces variance.
- Feature randomness prevents all trees from using the same split → trees are decorrelated → averaging is more effective.
- Bias stays the same as a single deep tree, but variance decreases with the number of trees.

$$\text{Var}(\bar{X}) = \rho \sigma^2 + \frac{1-\rho}{n_\text{trees}} \sigma^2$$

where $\rho$ is the correlation between trees. Feature randomness reduces $\rho$.

---

## Hyperparameters

| Parameter | Effect |
|---|---|
| `n_estimators` | More trees → lower variance, diminishing returns beyond ~100 |
| `max_depth` | Deeper trees → lower bias, higher variance per tree |
| `max_features` | Fewer features → more decorrelation, but higher bias per tree |

---

## Majority Vote

```python
classes, counts = np.unique(predictions_per_tree, return_counts=True)
prediction = classes[np.argmax(counts)]
```
