import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import shap

from explainability import Shap

data = load_diabetes()
X, y = data.data[:100], data.target[:100]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=10, random_state=42)
model.fit(X_train, y_train)

# Our implementation
explainer_ours = Shap(n_sampling=500, background_size=20)
shap_ours = explainer_ours.explain(X_test, model)

# shap KernelExplainer reference
background = X_train[:20]
explainer_ref = shap.KernelExplainer(model.predict, background)
shap_ref = explainer_ref.shap_values(X_test)

mean_ours = np.abs(shap_ours).mean(axis=0)
mean_ref = np.abs(shap_ref).mean(axis=0)

print("=== Mean absolute SHAP per feature (ours) ===")
print(np.round(mean_ours, 3))

print("\n=== Mean absolute SHAP per feature (shap lib) ===")
print(np.round(mean_ref, 3))

print("\n=== Mean absolute difference per feature ===")
print(np.round(np.abs(shap_ours - shap_ref).mean(axis=0), 3))

print("\n=== Feature ranking (ours) ===")
print(np.argsort(mean_ours)[::-1])

print("\n=== Feature ranking (shap lib) ===")
print(np.argsort(mean_ref)[::-1])
