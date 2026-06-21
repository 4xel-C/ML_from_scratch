# K-Nearest Neighbors (KNN)

**Type:** Supervised — Classification  
**File:** `classification_models/KNN_classification.py`

---

## Problem

Classify a new point by looking at the k most similar points in the training set and taking a majority vote.

---

## Key Property — Lazy Learner

KNN has **no training phase**. It simply memorizes the entire training set. All computation happens at prediction time. This makes it:
- **Fast to train** (just store data)
- **Slow to predict** (must compute distances to all training points)

---

## Algorithm

**Fit:**
```
Store X_train, y_train
```

**Predict for a test set X_test:**
1. Compute the distance matrix between X_test and X_train.
2. For each test point, find the k nearest neighbors.
3. Predict the majority class among those k neighbors.

---

## Vectorized Distance Matrix

All pairwise distances in one operation:

```python
dist = np.linalg.norm(
    X_test[:, np.newaxis, :] - X_train[np.newaxis, :, :], axis=2
)  # shape (n_test, n_train)
```

---

## Finding k Nearest Neighbors

```python
knn_indices = np.argsort(dist, axis=1)[:, :k]  # shape (n_test, k)
k_labels    = y_train[knn_indices]              # shape (n_test, k)
```

---

## Majority Vote

```python
prediction = np.apply_along_axis(
    lambda row: np.argmax(np.bincount(row)), axis=1, arr=k_labels
)
```

---

## Effect of k

- **Small k (k=1):** very flexible, captures local structure, high variance, prone to overfitting.
- **Large k:** smoother decision boundary, higher bias, more robust to noise.
- **k = n:** always predicts the majority class (maximum bias).
