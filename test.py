import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import Lasso as SklearnLasso
from sklearn.linear_model import Ridge as SklearnRidge
from sklearn.preprocessing import StandardScaler

from regression_models import Lasso, Ridge

# ── Données ──────────────────────────────────────────────────────────────────
X, y = make_regression(n_samples=200, n_features=10, noise=10, random_state=42)

# Normalisation importante pour Ridge/Lasso — les pénalités sont sensibles aux échelles
scaler = StandardScaler()
X = scaler.fit_transform(X)

# ── Hyperparamètres ───────────────────────────────────────────────────────────
LAMBDA = 0.1
LR = 0.01
ITERATIONS = 2000

# ── Tes modèles ───────────────────────────────────────────────────────────────
ridge = Ridge(max_iterations=ITERATIONS, learning_rate=LR, l=LAMBDA)
lasso = Lasso(max_iterations=ITERATIONS, learning_rate=LR, l=LAMBDA)
ridge.fit(X, y)
lasso.fit(X, y)

# ── Modèles sklearn (alpha = lambda) ─────────────────────────────────────────
sk_ridge = SklearnRidge(alpha=LAMBDA)
sk_lasso = SklearnLasso(alpha=LAMBDA, max_iter=ITERATIONS)
sk_ridge.fit(X, y)
sk_lasso.fit(X, y)

# ── Comparaison des coefficients ──────────────────────────────────────────────
print("=" * 55)
print(
    f"{'Feature':<12} {'Ridge':>10} {'sklearn R':>12} {'Lasso':>10} {'sklearn L':>12}"
)
print("=" * 55)
print(
    f"{'biais':<12} {ridge.w[0]:>10.4f} {sk_ridge.intercept_:>12.4f} {lasso.w[0]:>10.4f} {sk_lasso.intercept_:>12.4f}"
)
for i in range(X.shape[1]):
    print(
        f"{'w' + str(i+1):<12} {ridge.w[i+1]:>10.4f} {sk_ridge.coef_[i]:>12.4f} {lasso.w[i+1]:>10.4f} {sk_lasso.coef_[i]:>12.4f}"
    )

# ── Comparaison des prédictions ───────────────────────────────────────────────
ridge_preds = ridge.predict(X)
lasso_preds = lasso.predict(X)
sk_ridge_preds = sk_ridge.predict(X)
sk_lasso_preds = sk_lasso.predict(X)

ridge_mse = np.mean((ridge_preds - sk_ridge_preds) ** 2)
lasso_mse = np.mean((lasso_preds - sk_lasso_preds) ** 2)

print("=" * 55)
print(f"MSE prédictions Ridge vs sklearn  : {ridge_mse:.6f}")
print(f"MSE prédictions Lasso vs sklearn  : {lasso_mse:.6f}")

# ── Vérification sparsité Lasso ───────────────────────────────────────────────
n_zeros_lasso = np.sum(np.abs(lasso.w[1:]) < 1e-3)
n_zeros_sk_lasso = np.sum(np.abs(sk_lasso.coef_) < 1e-3)
print("=" * 55)
print(f"Poids ~ 0 (Lasso)         : {n_zeros_lasso} / {X.shape[1]}")
print(f"Poids ~ 0 (sklearn Lasso) : {n_zeros_sk_lasso} / {X.shape[1]}")
