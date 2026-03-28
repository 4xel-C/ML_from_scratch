import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression as SklearnLR

from regression_models import LinearRegression

# Générer des données
X, y = make_regression(n_samples=200, n_features=3, noise=10, random_state=42)

# Ton modèle
model = LinearRegression(learning_rate=0.1)
model.fit(X, y)
pred = model.predict(X)

# Sklearn
sk_model = SklearnLR()
sk_model.fit(X, y)
sk_pred = sk_model.predict(X)

# Comparer
print("Tes coefs    :", model.w)
print("Sklearn coefs:", np.append(sk_model.intercept_, sk_model.coef_))
