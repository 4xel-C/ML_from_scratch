<<<<<<< HEAD
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

from classification_models import KNN

X, y = make_classification(n_samples=300, n_features=4, random_state=42)

# Ton modèle
model = KNN(k=5)
model.fit(X, y)
preds = model.predict(X)

# Sklearn
sk = KNeighborsClassifier(n_neighbors=5)
sk.fit(X, y)
sk_preds = sk.predict(X)

print(f"Ton accuracy    : {accuracy_score(y, preds):.4f}")
print(f"Sklearn accuracy: {accuracy_score(y, sk_preds):.4f}")

print(y)
=======
import numpy as np
from sklearn.cluster import KMeans as SklearnKMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

from clustering import KMeans

# Données avec 3 clusters bien séparés
X, y_true = make_blobs(n_samples=300, centers=3, cluster_std=0.8, random_state=42)

# Ton modèle
model = KMeans(k=3, max_iterations=100, tol=1e-4)
model.fit(X)
labels = model.predict(X)

# Sklearn
sk = SklearnKMeans(n_clusters=3, n_init=10, random_state=42)
sk.fit(X)

# Adjusted Rand Score — mesure la similarité des clusterings (1.0 = parfait)
score = adjusted_rand_score(sk.labels_, labels)
print(f"Adjusted Rand Score vs sklearn : {score:.4f}")

# Comparer les centroïdes (triés pour compenser l'ordre arbitraire)
print("\nTes centroïdes (triés) :")
print(np.sort(model.centroids, axis=0).round(3))
print("\nSklearn centroïdes (triés) :")
print(np.sort(sk.cluster_centers_, axis=0).round(3))
>>>>>>> cf83662788dd6ac36cb3c5b23cab26589b2cea70
