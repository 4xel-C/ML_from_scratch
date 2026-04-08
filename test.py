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
