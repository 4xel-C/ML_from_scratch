import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier as SklearnGBC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from classification_models import GradientBoostingClassifier

# ─────────────────────────────────────────────
# TEST 1 : Classification binaire simple
# ─────────────────────────────────────────────
print("=" * 50)
print("TEST 1 : Classification binaire simple (2 features)")
print("=" * 50)

X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

my_model = GradientBoostingClassifier(learning_rate=0.1, n_estimators=100, max_depth=3)
my_model.fit(X_train, y_train)

sk_model = SklearnGBC(learning_rate=0.1, n_estimators=100, max_depth=3)
sk_model.fit(X_train, y_train)

print(
    f"Custom  Train Accuracy : {accuracy_score(y_train, my_model.predict(X_train)):.3f}"
)
print(
    f"Custom  Test  Accuracy : {accuracy_score(y_test,  my_model.predict(X_test)):.3f}"
)
print(
    f"Sklearn Train Accuracy : {accuracy_score(y_train, sk_model.predict(X_train)):.3f}"
)
print(
    f"Sklearn Test  Accuracy : {accuracy_score(y_test,  sk_model.predict(X_test)):.3f}"
)

# ─────────────────────────────────────────────
# TEST 2 : Classification binaire multi-features
# ─────────────────────────────────────────────
print()
print("=" * 50)
print("TEST 2 : Classification binaire multi-features (10 features)")
print("=" * 50)

X2, y2 = make_classification(n_samples=100, n_features=10, random_state=42)
X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2, y2, test_size=0.2, random_state=42
)

my_model2 = GradientBoostingClassifier(learning_rate=0.1, n_estimators=100, max_depth=3)
my_model2.fit(X2_train, y2_train)

sk_model2 = SklearnGBC(learning_rate=0.1, n_estimators=100, max_depth=3)
sk_model2.fit(X2_train, y2_train)

print(
    f"Custom  Train Accuracy : {accuracy_score(y2_train, my_model2.predict(X2_train)):.3f}"
)
print(
    f"Custom  Test  Accuracy : {accuracy_score(y2_test,  my_model2.predict(X2_test)):.3f}"
)
print(
    f"Sklearn Train Accuracy : {accuracy_score(y2_train, sk_model2.predict(X2_train)):.3f}"
)
print(
    f"Sklearn Test  Accuracy : {accuracy_score(y2_test,  sk_model2.predict(X2_test)):.3f}"
)
