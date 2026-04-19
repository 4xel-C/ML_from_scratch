# ML From Scratch — Guide & Algorithmes Implémentés

## Philosophie pédagogique

L'objectif est de **comprendre** les algorithmes, pas seulement de les coder.
Chaque implémentation suit ce processus :

1. **Comprendre le problème** — avant d'écrire une ligne de code
2. **Dériver les formules** — à la main si possible
3. **Coder progressivement** — par petites étapes validées
4. **Tester et valider** — comparaison avec sklearn

---

## Consignes pédagogiques

### Ce que le guide fait
- Pose des **questions** plutôt que de donner les réponses directement
- Signale les bugs **sans les corriger** — juste des indices
- Demande de **dériver les formules** avant de coder
- Encourage la **vectorisation NumPy** plutôt que les boucles Python
- Valide chaque étape avant de passer à la suivante

### Ce que l'apprenant doit faire
- Donner l'algorithme et les formules à utiliser
- Décomposer le fonctionnement
- Réfléchir aux **dimensions** (shapes) avant d'écrire du code
- Dériver les **gradients** à la main
- Tester son implémentation contre **sklearn**
- Comprendre **pourquoi** avant de coder **comment**

---

## Algorithmes implémentés

### 1. Régression Linéaire
**Type** : Supervisé — Régression  
**Fichier** : `regression_models/linear_regression.py`

**Concepts clés**
- Modèle : $\hat{y} = Xw$ (biais intégré via colonne de 1)
- Loss : MSE $= \frac{1}{n} \sum (\hat{y} - y)^2$
- Optimisation : descente de gradient
- Gradient : $\frac{\partial MSE}{\partial w} = \frac{1}{n} X^T(\hat{y} - y)$
- Initialisation de `w` à 0 (MSE convexe → un seul minimum global)

**Points importants**
- Ajouter une colonne de 1 à `X` pour le biais : `np.hstack([np.ones((n,1)), X])`
- Condition d'arrêt : `iteration < max_iterations and loss > tol`

---

### 2. Lasso (Régularisation L1)
**Type** : Supervisé — Régression  
**Fichier** : `regression_models/lasso.py`  
**Hérite de** : `LinearRegression`

**Concepts clés**
- Loss : $MSE + \lambda \|w\|_1$
- Gradient : $\frac{1}{n} X^T(\hat{y} - y) + \lambda \cdot \text{sign}(w)$
- Propriété clé : **sparsité** — certains poids tombent exactement à 0
- Ne pas régulariser le biais : `gradient[1:] += lambda * sign(w[1:])`

**Pourquoi L1 produit de la sparsité**
- La boule L1 est un losange avec des **coins sur les axes**
- Le contact avec les contours du MSE se fait souvent sur un coin → $w_j = 0$
- La pénalité $\lambda \cdot \text{sign}(w)$ est **constante** quelle que soit la valeur de $w$

---

### 3. Ridge (Régularisation L2)
**Type** : Supervisé — Régression  
**Fichier** : `regression_models/ridge.py`  
**Hérite de** : `LinearRegression`

**Concepts clés**
- Loss : $MSE + \lambda \|w\|_2^2$
- Gradient : $\frac{1}{n} X^T(\hat{y} - y) + \lambda \cdot w$
- Propriété clé : réduit les poids **sans les annuler**
- Ne pas régulariser le biais : `gradient[1:] += lambda * w[1:]`

**Différence L1 vs L2**
| | Lasso (L1) | Ridge (L2) |
|---|---|---|
| Gradient régularisation | $\lambda \cdot \text{sign}(w)$ | $\lambda \cdot w$ |
| Effet sur les poids | Force à 0 (sparsité) | Réduit sans annuler |
| Boule | Losange (coins) | Cercle (lisse) |

---

### 4. Régression Logistique
**Type** : Supervisé — Classification  
**Fichier** : `regression_models/logistic_regression.py`

**Concepts clés**
- Modèle : $\hat{y} = \sigma(Xw)$ où $\sigma(z) = \frac{1}{1+e^{-z}}$
- Loss : BCE $= -\frac{1}{n}[y^T \log(\hat{y}) + (1-y)^T \log(1-\hat{y})]$
- Gradient : $\frac{1}{n} X^T(\hat{y} - y)$ (identique à la régression linéaire !)
- Sortie : probabilité $P(y=1 \mid X) \in [0, 1]$

**Points importants**
- Le **logit** $= Xw$ est la combinaison linéaire avant sigmoid
- Seuil de décision : $\hat{y} \geq 0.5 \Rightarrow$ classe 1
- Stabilité numérique BCE : `np.clip(proba, 1e-15, 1-1e-15)`
- Dérivée sigmoid : $\sigma'(z) = \sigma(z)(1-\sigma(z))$

---

### 5. K-Means
**Type** : Non supervisé — Clustering  
**Fichier** : `clustering_models/kmeans.py`

**Concepts clés**
- Algorithme EM : alterner **assignation** et **mise à jour des centroïdes**
- Distance euclidienne vectorisée : `np.linalg.norm(X[:,np.newaxis,:] - centroids[np.newaxis,:,:], axis=2)`
- Assignation : `distances.argmin(axis=1)`
- Convergence : $\delta = \sum_k \|c_k^{new} - c_k^{old}\| < \epsilon$

**K-Means++**
- Initialisation intelligente pour éviter les centroïdes trop proches
- Choisir chaque centroïde avec probabilité $\propto d^2$ au plus proche centroïde existant

---

### 6. KNN (K-Nearest Neighbors)
**Type** : Supervisé — Classification  
**Fichier** : `classification_models/knn.py`

**Concepts clés**
- **Lazy learner** : pas d'entraînement, mémorisation de `X` et `y`
- Matrice de distances : `np.linalg.norm(X_test[:,np.newaxis,:] - X_train[np.newaxis,:,:], axis=2)`
- K voisins : `np.argsort(dist, axis=1)[:, :k]`
- Vote majoritaire : `np.argmax(np.bincount(row))` sur chaque ligne

---

### 7. Naive Bayes Gaussien (+ Catégoriel mixte)
**Type** : Supervisé — Classification
**Fichier** : `classification_models/naive_bayes.py`

**Concepts clés**
- Théorème de Bayes : $P(y \mid X) \propto P(X \mid y) \cdot P(y)$
- Hypothèse **naïve** : features indépendantes entre elles
- Vraisemblance gaussienne : $\log P(x_j \mid y=k) = -\frac{1}{2}\log(2\pi\sigma^2) - \frac{(x-\mu)^2}{2\sigma^2}$
- Log-posterior : $\log P(y=k) + \sum_j \log P(x_j \mid y=k)$

**`fit`** : calculer prior, moyennes et variances par classe
**`predict`** : calculer log-posterior vectorisé, `argmax` sur les classes

**Extension : features mixtes (continues + catégorielles)**
- Paramètre optionnel `categorical_features` : masque booléen sur les colonnes catégorielles
- Features catégorielles : loi **Categorical** — proportions par modalité et par classe
- **Laplace smoothing** : $P(c \mid y=k) = \frac{count(c,k) + \alpha}{N_k + \alpha \cdot |\mathcal{V}|}$ pour éviter les probabilités nulles
- Stockage : `defaultdict(lambda: defaultdict(dict))` — `proportions[feature_idx][class][modalité]`
- Log-posterior final = contribution gaussienne (continues) + log-proportion (catégorielles)

---

## Broadcasting NumPy — Règles clés

| Opération | Pattern | Résultat |
|---|---|---|
| Distance points vs centroïdes | `X[:,np.newaxis,:] - C[np.newaxis,:,:]` | `(n, k, p)` |
| Moyenne par groupe | `sums / counts[:, np.newaxis]` | `(k, p)` |
| Masque par classe | `classes[:,np.newaxis] == y[np.newaxis,:]` | `(n_classes, n)` |

**Règle** : deux dimensions sont compatibles si elles sont **égales** ou si l'une vaut **1**.

---

---

### 8. Decision Tree Regressor
**Type** : Supervisé — Régression
**Fichier** : `regression_models/decision_tree.py`

**Concepts clés**
- Construction récursive : `_build_tree` appelle `_best_split` puis se rappelle sur les sous-arbres gauche et droit
- Critère de split : minimiser la **variance pondérée** des deux groupes
  `variance_split = (n_left/n) * var(y_left) + (n_right/n) * var(y_right)`
- Gain de variance : `delta_var = var(y) - variance_split` — doit être > `min_variance`
- Prédiction en feuille : **moyenne de y** des points du nœud
- Traversée : `_traverse(x, node)` récursif, `np.apply_along_axis` pour vectoriser sur X

**Conditions d'arrêt**
- `node.level == max_depth`
- `len(X) < min_samples_split`
- `len(X_left) < min_samples_leaf` ou `len(X_right) < min_samples_leaf`
- `delta_var < min_variance`

**Structure de données**
- `@dataclass Node` avec `left`, `right`, `value` optionnels
- `from __future__ import annotations` pour l'auto-référencement dans le dataclass

---

### 9. Decision Tree Classifier
**Type** : Supervisé — Classification
**Fichier** : `classification_models/decision_tree_classifier.py`
**Hérite de** : `DecisionTreeBase` (dans `bases/`)

**Concepts clés**
- Même structure récursive que le Regressor
- Critère de split : minimiser le **Gini pondéré** des deux groupes
  `Gini_split = (n_left/n) * Gini(y_left) + (n_right/n) * Gini(y_right)`
- Impureté de Gini : $Gini = 1 - \sum_k p_k^2$ — vaut 0 si nœud pur, 0.5 si 50/50
- Gain de Gini : `delta = Gini(parent) - Gini_split` — doit être > `min_gini`
- Valeur en feuille : **mode de y** (classe la plus fréquente) via `np.unique` + `argmax`

**Différence clé avec le Regressor**
| | Regressor | Classifier |
|---|---|---|
| Critère de split | Variance pondérée | Gini pondéré |
| Valeur en feuille | Moyenne de y | Mode de y |

---

## Prochains algorithmes suggérés

### Niveau 2
- **Decision Tree Classifier** — même structure, critère Gini/Entropy ✅
- **PCA** — réduction de dimension, valeurs propres
- **DBSCAN** — clustering par densité

### Niveau 3
- **Random Forest** — bagging, ensemble de Decision Trees
- **Gradient Boosting** — boosting séquentiel
- **Neural Network** — backpropagation, couches
