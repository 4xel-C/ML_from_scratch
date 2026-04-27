import numpy as np
from sklearn.cluster import DBSCAN as SklearnDBSCAN
from sklearn.datasets import make_blobs, make_moons
from sklearn.metrics import adjusted_rand_score

from clustering import DBScan

# ─────────────────────────────────────────────
# TEST 1 : Clusters sphériques (cas simple)
# ─────────────────────────────────────────────
print("=" * 50)
print("TEST 1 : Clusters sphériques")
print("=" * 50)

X_blobs, y_blobs = make_blobs(n_samples=300, centers=3, cluster_std=0.5, random_state=42)

my_db = DBScan(epsilon=0.8, min_samples=5)
my_labels = my_db.fit_predict(X_blobs)

sk_db = SklearnDBSCAN(eps=0.8, min_samples=5)
sk_labels = sk_db.fit_predict(X_blobs)

print(f"Ton ARI vs sklearn : {adjusted_rand_score(sk_labels, my_labels):.4f}  (1.0 = identique)")
print(f"Clusters trouvés   — toi: {len(set(my_labels)) - (1 if -1 in my_labels else 0)}, sklearn: {len(set(sk_labels)) - (1 if -1 in sk_labels else 0)}")
print(f"Outliers           — toi: {(my_labels == -1).sum()}, sklearn: {(sk_labels == -1).sum()}")

# ─────────────────────────────────────────────
# TEST 2 : Clusters en forme de lune (non convexes)
# ─────────────────────────────────────────────
print()
print("=" * 50)
print("TEST 2 : Clusters en lune (non convexes)")
print("=" * 50)

X_moons, y_moons = make_moons(n_samples=200, noise=0.05, random_state=42)

my_db2 = DBScan(epsilon=0.2, min_samples=5)
my_labels2 = my_db2.fit_predict(X_moons)

sk_db2 = SklearnDBSCAN(eps=0.2, min_samples=5)
sk_labels2 = sk_db2.fit_predict(X_moons)

print(f"Ton ARI vs sklearn : {adjusted_rand_score(sk_labels2, my_labels2):.4f}  (1.0 = identique)")
print(f"Clusters trouvés   — toi: {len(set(my_labels2)) - (1 if -1 in my_labels2 else 0)}, sklearn: {len(set(sk_labels2)) - (1 if -1 in sk_labels2 else 0)}")
print(f"Outliers           — toi: {(my_labels2 == -1).sum()}, sklearn: {(sk_labels2 == -1).sum()}")

# ─────────────────────────────────────────────
# TEST 3 : Données avec outliers explicites
# ─────────────────────────────────────────────
print()
print("=" * 50)
print("TEST 3 : Données avec outliers")
print("=" * 50)

np.random.seed(42)
X_core, _ = make_blobs(n_samples=200, centers=2, cluster_std=0.3, random_state=42)
X_outliers = np.random.uniform(low=-6, high=6, size=(20, 2))
X_with_outliers = np.vstack([X_core, X_outliers])

my_db3 = DBScan(epsilon=0.5, min_samples=5)
my_labels3 = my_db3.fit_predict(X_with_outliers)

sk_db3 = SklearnDBSCAN(eps=0.5, min_samples=5)
sk_labels3 = sk_db3.fit_predict(X_with_outliers)

print(f"Ton ARI vs sklearn : {adjusted_rand_score(sk_labels3, my_labels3):.4f}  (1.0 = identique)")
print(f"Clusters trouvés   — toi: {len(set(my_labels3)) - (1 if -1 in my_labels3 else 0)}, sklearn: {len(set(sk_labels3)) - (1 if -1 in sk_labels3 else 0)}")
print(f"Outliers           — toi: {(my_labels3 == -1).sum()}, sklearn: {(sk_labels3 == -1).sum()}")
