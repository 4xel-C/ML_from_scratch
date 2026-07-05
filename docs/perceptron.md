# Perceptron

**Type:** Supervised — Binary Classification  
**File:** `classification_models/perceptron.py`

---

## Problem

Classify a point $x \in \mathbb{R}^p$ into one of two classes $\{0, 1\}$ using a linear decision boundary.

---

## Intuition

The perceptron is the simplest neural network — a single neuron. It computes a weighted sum of the features, adds a bias, and applies a threshold function to produce a binary prediction. It is the historical precursor to all modern neural networks (Rosenblatt, 1957).

**Key difference from Logistic Regression:**
- Logistic Regression minimizes a loss (BCE) via gradient descent — the update uses the gradient computed globally.
- The Perceptron has no explicit loss function — it updates weights **only when it makes a mistake**, point by point.

---

## Mathematical Derivation

### Step 1 — Linear combination

$$z = w^T x + b = \sum_{k=1}^{p} w_k x_k + b$$

### Step 2 — Activation: Heaviside function

$$\hat{y} = \begin{cases} 1 & \text{if } z \geq 0 \\ 0 & \text{if } z < 0 \end{cases}$$

Unlike the sigmoid (which outputs a continuous probability), the Heaviside function outputs a **hard binary decision**. It is not differentiable at 0 — which is why gradient descent cannot be used directly.

### Step 3 — Perceptron update rule

For each point $(x_i, y_i)$, compute the error:

$$e_i = y_i - \hat{y}_i \in \{-1, 0, +1\}$$

Then update:

$$w \leftarrow w + \eta \cdot e_i \cdot x_i$$
$$b \leftarrow b + \eta \cdot e_i$$

Where $\eta$ is the learning rate.

### Update rule — three cases

| Case | $e_i = y_i - \hat{y}_i$ | Effect on $w$ |
|---|---|---|
| Correct prediction | $0$ | No update |
| False positive ($\hat{y}=1$, $y=0$) | $-1$ | $w \leftarrow w - \eta \cdot x_i$ |
| False negative ($\hat{y}=0$, $y=1$) | $+1$ | $w \leftarrow w + \eta \cdot x_i$ |

**Intuition:** on a false negative, the perceptron pushes $w$ in the direction of $x_i$ to increase $z = w^T x_i$ and flip the prediction to 1. On a false positive, it does the opposite.

---

## Convergence Theorem

The perceptron converges in a **finite number of steps** if and only if the data is **linearly separable** — i.e., there exists a hyperplane $w^T x + b = 0$ that perfectly separates the two classes.

If the data is not linearly separable (e.g., XOR), the perceptron loops indefinitely without converging. This fundamental limitation led to the development of the MLP (multi-layer perceptron).

---

## Why initialize $w = 0$?

For the perceptron, $w = 0$ is a valid neutral starting point:
- There is only one neuron — no **symmetry breaking** problem (unlike MLP where all neurons in a layer would learn identically if initialized to the same value).
- The update rule is error-driven, not gradient-driven — the first erroneous prediction immediately breaks the symmetry.

---

## Comparison with Logistic Regression

| | Perceptron | Logistic Regression |
|---|---|---|
| Activation | Heaviside (step) | Sigmoid |
| Output | $\{0, 1\}$ (hard) | $[0, 1]$ (probability) |
| Update | Error-driven, online | Gradient descent, batch/online |
| Loss | None explicit | Binary Cross-Entropy |
| Convergence | Only if linearly separable | Always (convex loss) |

---

## Hyperparameters

| Parameter | Default | Role |
|---|---|---|
| `learning_rate` | 0.1 | Step size for weight updates |
| `epochs` | 1000 | Max number of passes over the training data |

---

## Implementation Notes

- Updates are **online** (point by point, not batch) — required for the convergence theorem to hold.
- The inner loop uses index `j`, outer loop uses `i` — never confuse the two or `X[i]` silently replaces `X[j]`.
- Convergence check: `all(old_W == W) and old_b == b` — no weight changed during the epoch.
- `StandardScaler` recommended before training — large feature scales cause large weight updates that destabilize convergence.

---

## Pipeline

```
fit(X, y):
    W = zeros(p), b = 0
    for each epoch:
        old_W, old_b = copy(W), copy(b)
        for each point j:
            z = X[j] @ W + b
            y_pred = 1 if z >= 0 else 0
            error = y[j] - y_pred
            W += lr * error * X[j]
            b += lr * error
        if no weight changed: stop (converged)

predict(X):
    z = X @ W + b
    return 1 if z >= 0 else 0
```
