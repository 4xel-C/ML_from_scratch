# Decision Tree

**Type:** Supervised — Regression & Classification  
**Files:**
- `regression_models/decision_tree_regressor.py`
- `classification_models/decision_tree_classifier.py`

---

## Problem

Recursively partition the feature space into rectangular regions. Each region predicts a constant value (regression) or a class (classification).

---

## Structure

The tree is a binary structure of `Node` objects:

```python
@dataclass
class Node:
    level: int
    feature_idx: int
    threshold: float
    value: Optional[float]   # set only for leaf nodes
    left:  Optional[Node]
    right: Optional[Node]
```

---

## Building the Tree — Recursive Algorithm

```
_build_tree(X, y, node):
    if stopping condition:
        node.value = leaf_value(y)
        return node
    feature, threshold = best_split(X, y)
    X_left,  y_left  = X[X[:, feature] <  threshold]
    X_right, y_right = X[X[:, feature] >= threshold]
    node.left  = _build_tree(X_left,  y_left,  left_node)
    node.right = _build_tree(X_right, y_right, right_node)
    return node
```

---

## Split Criterion

### Regressor — Weighted Variance

$$\text{variance\_split} = \frac{n_L}{n} \text{Var}(y_L) + \frac{n_R}{n} \text{Var}(y_R)$$

$$\Delta\text{var} = \text{Var}(y) - \text{variance\_split}$$

A split is accepted only if $\Delta\text{var} > \text{min\_variance}$.

**Leaf value:** mean of y in the node.

### Classifier — Weighted Gini Impurity

$$\text{Gini}(y) = 1 - \sum_k p_k^2$$

$$\text{Gini\_split} = \frac{n_L}{n} \text{Gini}(y_L) + \frac{n_R}{n} \text{Gini}(y_R)$$

Gini = 0 for a pure node, 0.5 for a 50/50 binary split.

**Leaf value:** mode of y (most frequent class) in the node.

---

## Finding the Best Split

For each feature, sort the values and evaluate every midpoint as a candidate threshold:

```
threshold = (x[i] + x[i+1]) / 2
```

Keep the threshold that minimizes the weighted criterion. Then compare across all features.

---

## Stopping Conditions

| Condition | Description |
|---|---|
| `node.level == max_depth` | Maximum tree depth reached |
| `len(X) < min_samples_split` | Too few samples to split |
| `len(X_left) < min_samples_leaf` | Would produce too-small leaf |
| `len(X_right) < min_samples_leaf` | Would produce too-small leaf |
| `delta < min_variance / min_gini` | No meaningful gain from splitting |

---

## Prediction

Traverse the tree from root to leaf for each sample:

```python
def _traverse(x, node):
    if node.value is not None:    # leaf
        return node.value
    if x[node.feature_idx] < node.threshold:
        return _traverse(x, node.left)
    else:
        return _traverse(x, node.right)
```
