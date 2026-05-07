import numpy as np
from sklearn.datasets import make_regression
from sklearn.ensemble import GradientBoostingRegressor as SklearnGBM
from sklearn.metrics import mean_squared_error

from regression_models import GradientBoosting

# ─────────────────────────────────────────────
# TEST 1 : Régression simple (1 feature)
# ─────────────────────────────────────────────
print("=" * 50)
print("TEST 1 : Regression simple (1 feature)")
print("=" * 50)

X, y = make_regression(n_samples=300, n_features=1, noise=10, random_state=42)

my_model = GradientBoosting(learning_rate=0.1, n_estimators=100, max_depth=3)
my_model.fit(X, y)
my_mse = mean_squared_error(y, my_model.predict(X))

sk_model = SklearnGBM(learning_rate=0.1, n_estimators=100, max_depth=3)
sk_model.fit(X, y)
sk_mse = mean_squared_error(y, sk_model.predict(X))

print(f"Custom  MSE : {my_mse:.3f}")
print(f"Sklearn MSE : {sk_mse:.3f}")

# ─────────────────────────────────────────────
# TEST 2 : Régression multi-features
# ─────────────────────────────────────────────
print()
print("=" * 50)
print("TEST 2 : Regression multi-features (10 features)")
print("=" * 50)

X2, y2 = make_regression(n_samples=500, n_features=10, noise=15, random_state=42)

my_model2 = GradientBoosting(learning_rate=0.1, n_estimators=100, max_depth=3)
my_model2.fit(X2, y2)
my_mse2 = mean_squared_error(y2, my_model2.predict(X2))

sk_model2 = SklearnGBM(learning_rate=0.1, n_estimators=100, max_depth=3)
sk_model2.fit(X2, y2)
sk_mse2 = mean_squared_error(y2, sk_model2.predict(X2))

print(f"Custom  MSE : {my_mse2:.3f}")
print(f"Sklearn MSE : {sk_mse2:.3f}")
