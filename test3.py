import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier as SklearnGBC
from sklearn.metrics import accuracy_score

from classification_models import GradientBoostingClassifier

# ─────────────────────────────────────────────
# TEST 1 : Classification binaire simple
# ─────────────────────────────────────────────
print("=" * 50)
print("TEST 1 : Classification binaire simple (2 features)")
print("=" * 50)

X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, random_state=42)

my_model = GradientBoostingClassifier(learning_rate=0.1, n_estimators=100, max_depth=3)
my_model.fit(X, y)
my_acc = accuracy_score(y, my_model.predict(X))

sk_model = SklearnGBC(learning_rate=0.1, n_estimators=100, max_depth=3)
sk_model.fit(X, y)
sk_acc = accuracy_score(y, sk_model.predict(X))

print(f"Custom  Accuracy : {my_acc:.3f}")
print(f"Sklearn Accuracy : {sk_acc:.3f}")

# ─────────────────────────────────────────────
# TEST 2 : Classification binaire multi-features
# ─────────────────────────────────────────────
print()
print("=" * 50)
print("TEST 2 : Classification binaire multi-features (10 features)")
print("=" * 50)

X2, y2 = make_classification(n_samples=100, n_features=10, random_state=42)

my_model2 = GradientBoostingClassifier(learning_rate=0.1, n_estimators=100, max_depth=3)
my_model2.fit(X2, y2)
my_acc2 = accuracy_score(y2, my_model2.predict(X2))

sk_model2 = SklearnGBC(learning_rate=0.1, n_estimators=100, max_depth=3)
sk_model2.fit(X2, y2)
sk_acc2 = accuracy_score(y2, sk_model2.predict(X2))

print(f"Custom  Accuracy : {my_acc2:.3f}")
print(f"Sklearn Accuracy : {sk_acc2:.3f}")
