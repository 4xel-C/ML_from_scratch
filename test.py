import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB, GaussianNB
from sklearn.tree import DecisionTreeClassifier as SklearnDTC
from sklearn.tree import DecisionTreeRegressor as SklearnDTR

from classification_models import DecisionTreeClassifier, NaiveBayes
from regression_models import DecisionTreeRegressor

# ─────────────────────────────────────────────
# TEST 1 : Features purement continues (comme avant)
# ─────────────────────────────────────────────
print("=" * 50)
print("TEST 1 : Features continues uniquement")
print("=" * 50)

X, y = make_classification(n_samples=300, n_features=4, random_state=42)

model = NaiveBayes()
model.fit(X, y)
preds = model.predict(X)

sk = GaussianNB()
sk.fit(X, y)
sk_preds = sk.predict(X)

print(f"Ton accuracy    : {accuracy_score(y, preds):.4f}")
print(f"Sklearn accuracy: {accuracy_score(y, sk_preds):.4f}")

# ─────────────────────────────────────────────
# TEST 2 : Features mixtes (continues + catégorielles)
# ─────────────────────────────────────────────
print()
print("=" * 50)
print("TEST 2 : Features mixtes")
print("=" * 50)

np.random.seed(42)
n = 500

# 2 features continues
X_cont = np.random.randn(n, 2)

# 2 features catégorielles (entiers représentant des catégories)
X_cat_1 = np.random.randint(0, 3, size=(n, 1))  # 3 modalités
X_cat_2 = np.random.randint(0, 4, size=(n, 1))  # 4 modalités

X_mixed = np.hstack([X_cont, X_cat_1, X_cat_2])

# Labels : influencés par les features continues
y_mixed = (X_cont[:, 0] + X_cont[:, 1] + np.random.randn(n) * 0.5 > 0).astype(int)

# Masque booléen : False=continue, True=catégorielle
cat_mask = np.array([False, False, True, True])

# Ton modèle
model_mixed = NaiveBayes()
model_mixed.fit(X_mixed, y_mixed, categorical_features=cat_mask)
preds_mixed = model_mixed.predict(X_mixed)

# Sklearn : GaussianNB sur les continues + CategoricalNB sur les catégorielles
# (comparaison approximative — sklearn n'a pas de modèle mixte natif)
sk_gauss = GaussianNB()
sk_gauss.fit(X_cont, y_mixed)
sk_preds_cont = sk_gauss.predict(X_cont)

sk_cat = CategoricalNB()
sk_cat.fit(X_mixed[:, 2:].astype(int), y_mixed)
sk_preds_cat = sk_cat.predict(X_mixed[:, 2:].astype(int))

print(f"Ton accuracy (mixte)         : {accuracy_score(y_mixed, preds_mixed):.4f}")
print(f"Sklearn GaussianNB (continu) : {accuracy_score(y_mixed, sk_preds_cont):.4f}")
print(f"Sklearn CategoricalNB (cat)  : {accuracy_score(y_mixed, sk_preds_cat):.4f}")
print()
print(
    "Note : ton modèle combine les deux — il devrait faire mieux que chacun séparément."
)

# ─────────────────────────────────────────────
# TEST 3 : Decision Tree Regressor
# ─────────────────────────────────────────────
print()
print("=" * 50)
print("TEST 3 : Decision Tree Regressor")
print("=" * 50)

X_reg, y_reg = make_regression(n_samples=300, n_features=5, noise=10, random_state=42)

model_dt = DecisionTreeRegressor(max_depth=5)
model_dt.fit(X_reg, y_reg)
preds_dt = model_dt.predict(X_reg)

sk_dt = SklearnDTR(max_depth=5)
sk_dt.fit(X_reg, y_reg)
sk_preds_dt = sk_dt.predict(X_reg)

print(f"Ton MSE    : {mean_squared_error(y_reg, preds_dt):.4f}")
print(f"Sklearn MSE: {mean_squared_error(y_reg, sk_preds_dt):.4f}")

# ─────────────────────────────────────────────
# TEST 4 : Decision Tree Classifier
# ─────────────────────────────────────────────
print()
print("=" * 50)
print("TEST 4 : Decision Tree Classifier")
print("=" * 50)

X_iris, y_iris = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X_iris, y_iris, random_state=42)

model_clf = DecisionTreeClassifier(max_depth=5)
model_clf.fit(X_train, y_train)
preds_clf = model_clf.predict(X_test)

sk_clf = SklearnDTC(max_depth=5)
sk_clf.fit(X_train, y_train)
sk_preds_clf = sk_clf.predict(X_test)

print(f"Ton accuracy    : {accuracy_score(y_test, preds_clf):.4f}")
print(f"Sklearn accuracy: {accuracy_score(y_test, sk_preds_clf):.4f}")
