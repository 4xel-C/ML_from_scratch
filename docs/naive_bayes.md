# Naive Bayes

**Type:** Supervised — Classification  
**File:** `classification_models/naive_bayes.py`

---

## Problem

Probabilistic classification using Bayes' theorem. Predict the class with the highest posterior probability given the features.

---

## Bayes' Theorem

$$P(y \mid X) = \frac{P(X \mid y) \cdot P(y)}{P(X)}$$

- **Prior** $P(y)$: marginal probability of each class.
- **Likelihood** $P(X \mid y)$: probability of observing X given class y.
- **Evidence** $P(X)$: constant for all classes, so ignored in argmax.

**Decision rule:**

$$\hat{y} = \arg\max_k \left[ \log P(y=k) + \sum_j \log P(x_j \mid y=k) \right]$$

---

## Naive Assumption

Features are assumed **conditionally independent** given the class:

$$P(X \mid y=k) = \prod_j P(x_j \mid y=k)$$

This is rarely true in practice but dramatically simplifies computation and works surprisingly well.

---

## Gaussian Likelihood (Continuous Features)

Each feature $j$ in class $k$ is modeled as a Gaussian:

$$\log P(x_j \mid y=k) = -\frac{1}{2}\log(2\pi\sigma_{jk}^2) - \frac{(x_j - \mu_{jk})^2}{2\sigma_{jk}^2}$$

**Fit:** compute $\mu_{jk}$ and $\sigma_{jk}^2$ for each feature $j$ and class $k$.

---

## Categorical Likelihood (Discrete Features)

For categorical features, use the observed proportions with **Laplace smoothing** to avoid zero probabilities:

$$P(c \mid y=k) = \frac{\text{count}(c, k) + \alpha}{N_k + \alpha \cdot |\mathcal{V}|}$$

where $\alpha=1$ (add-one smoothing) and $|\mathcal{V}|$ is the number of unique modalities.

---

## Mixed Features

The implementation supports a mix of continuous and categorical features via a boolean mask `categorical_features`. The log-posterior combines:
- Gaussian log-likelihood for continuous features
- Log-proportion for categorical features

---

## Vectorized Posterior

The log-posterior for all n_test points and k classes is computed in one operation:

```python
# shape: (n_test, n_classes, n_features) -> sum -> (n_test, n_classes)
posterior = -0.5 * log(2π * var) - (X - means)^2 / (2 * var)
posterior = posterior.sum(axis=2) + log(prior)
```
