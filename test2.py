import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import AdaBoostClassifier as SklearnAdaBoost
from sklearn.metrics import accuracy_score

from classification_models import AdaBoostClassifier

# ─────────────────────────────────────────────
# TEST 1 : Binaire
# ─────────────────────────────────────────────
print("=" * 50)
print("TEST 1 : Classification binaire")
print("=" * 50)

X, y = make_classification(n_samples=500, n_features=10, random_state=42)

my_model = AdaBoostClassifier(n_estimators=50)
my_model.fit(X, y)
my_acc = accuracy_score(y, my_model.predict(X))

sk_model = SklearnAdaBoost(n_estimators=50, algorithm="SAMME")
sk_model.fit(X, y)
sk_acc = accuracy_score(y, sk_model.predict(X))

print(f"Custom  : {my_acc:.3f}")
print(f"Sklearn : {sk_acc:.3f}")

# ─────────────────────────────────────────────
# TEST 2 : Multiclasse
# ─────────────────────────────────────────────
print()
print("=" * 50)
print("TEST 2 : Classification multiclasse")
print("=" * 50)

X3, y3 = make_classification(
    n_samples=500, n_features=10, n_classes=3, n_informative=5, random_state=42
)

my_model3 = AdaBoostClassifier(n_estimators=50)
my_model3.fit(X3, y3)
my_acc3 = accuracy_score(y3, my_model3.predict(X3))

sk_model3 = SklearnAdaBoost(n_estimators=50, algorithm="SAMME")
sk_model3.fit(X3, y3)
sk_acc3 = accuracy_score(y3, sk_model3.predict(X3))

print(f"Custom  : {my_acc3:.3f}")
print(f"Sklearn : {sk_acc3:.3f}")
