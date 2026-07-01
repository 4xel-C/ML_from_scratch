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
- Ne donne des **indices** que si l'apprenant est explicitement bloqué — laisser chercher d'abord
- Encourage la **vectorisation NumPy** plutôt que les boucles Python
- Valide chaque étape avant de passer à la suivante
- **Décompose chaque étape en sous-étapes** — une seule question à la fois, jamais plusieurs concepts d'un coup
- **Ne jamais donner la réponse directement** — même pour une explication, guider par des questions successives
- Si l'apprenant demande une explication, poser des questions pour qu'il reconstruise le raisonnement lui-même
- Progresse **linéairement** : problème → formalisation → formules → gradients → code → validation
- Ne jamais donner le pseudo-code ou la structure du code — laisser l'apprenant proposer

### Ce que l'apprenant doit faire
- Donner l'algorithme et les formules à utiliser
- Décomposer le fonctionnement
- Réfléchir aux **dimensions** (shapes) avant d'écrire du code
- Dériver les **gradients** à la main
- Tester son implémentation contre **sklearn**
- Comprendre **pourquoi** avant de coder **comment**

---

## Broadcasting NumPy — Règles clés

| Opération | Pattern | Résultat |
|---|---|---|
| Distance points vs centroïdes | `X[:,np.newaxis,:] - C[np.newaxis,:,:]` | `(n, k, p)` |
| Moyenne par groupe | `sums / counts[:, np.newaxis]` | `(k, p)` |
| Masque par classe | `classes[:,np.newaxis] == y[np.newaxis,:]` | `(n_classes, n)` |

**Règle** : deux dimensions sont compatibles si elles sont **égales** ou si l'une vaut **1**.

---

## Algorithmes implémentés

### 1. Régression Linéaire
**Fichier** : `regression_models/linear_regression.py` | **Doc** : `docs/linear_regression.md`
- Modèle : y_hat = Xw (biais via colonne de 1) | Loss : MSE | Gradient : (1/n) X^T(y_hat - y)
- Initialisation w=0 (MSE convexe) | Arrêt : loss < tol ou max_iterations

### 2. Lasso (L1)
**Fichier** : `regression_models/lasso.py` | **Doc** : `docs/lasso.md`
- Loss : MSE + lambda * |w| | Gradient[1:] += lambda * sign(w[1:]) (biais exclu)
- Sparsité : boule L1 = losange avec coins sur les axes → poids forcés à 0

### 3. Ridge (L2)
**Fichier** : `regression_models/ridge.py` | **Doc** : `docs/ridge.md`
- Loss : MSE + lambda * ||w||^2 | Gradient[1:] += lambda * w[1:] (biais exclu)
- Réduit les poids sans les annuler | Sensible à l'échelle → StandardScaler recommandé

### 4. Régression Logistique
**Fichier** : `classification_models/logistic_regression.py` | **Doc** : `docs/logistic_regression.md`
- Modèle : sigmoid(Xw) | Loss : BCE | Gradient : (1/n) X^T(p - y) (même forme que linéaire)
- Seuil : p >= 0.5 → classe 1 | Clip BCE : np.clip(p, 1e-15, 1-1e-15)

### 5. K-Means
**Fichier** : `clustering/kmeans.py` | **Doc** : `docs/kmeans.md`
- EM : assignation (argmin distance) + mise à jour centroïdes (moyenne)
- K-Means++ : init avec proba proportionnelle à d^2 | Convergence : delta centroïdes < tol

### 6. KNN
**Fichier** : `classification_models/KNN_classification.py` | **Doc** : `docs/knn.md`
- Lazy learner : mémorise X, y | Distance vectorisée shape (n_test, n_train)
- K voisins : argsort[:, :k] | Vote majoritaire : bincount + argmax

### 7. Naive Bayes Gaussien
**Fichier** : `classification_models/naive_bayes.py` | **Doc** : `docs/naive_bayes.md`
- P(y|X) ∝ P(X|y) * P(y) | Hypothèse naïve : features indépendantes
- Log-posterior vectorisé shape (n_test, k) | Laplace smoothing pour features catégorielles

### 8. Decision Tree Regressor
**Fichier** : `regression_models/decision_tree_regressor.py` | **Doc** : `docs/decision_tree.md`
- Split : minimiser variance pondérée | Feuille : moyenne de y
- Structure : @dataclass Node (level, feature_idx, threshold, value, left, right)
- Arrêt : max_depth, min_samples_split, min_samples_leaf, delta_var < min_variance

### 9. Decision Tree Classifier
**Fichier** : `classification_models/decision_tree_classifier.py` | **Doc** : `docs/decision_tree.md`
- Split : minimiser Gini pondéré | Gini = 1 - sum(p_k^2) | Feuille : mode de y
- Hérite de DecisionTreeBase | Supporte les poids (pour AdaBoost)

### 10. PCA
**Fichier** : `dimensionality_reduction/pca.py` | **Doc** : `docs/pca.md`
- Centrer X → covariance C = (1/n) X^T X → décomposition spectrale → garder k vecteurs propres
- Projection : X_reduit = X_centré @ V_k | Signe des vecteurs propres défini à ± près

### 11. Random Forest
**Fichier** : `classification_models/random_forest.py` | **Doc** : `docs/random_forest.md`
- Bagging : bootstrap samples + sqrt(p) features aléatoires par arbre
- Stockage : liste de tuples (tree, selected_features) | Prédiction : vote majoritaire

### 12. DBSCAN
**Fichier** : `clustering/dbscan.py` | **Doc** : `docs/dbscan.md`
- Core point : >= min_samples voisins dans epsilon (excluant lui-même)
- Expansion BFS depuis chaque core non visité | Outliers : restent à -1
- np.where retourne un tuple → [0] indispensable

### 13. AdaBoost
**Fichier** : `classification_models/adaboost.py` | **Doc** : `docs/adaboost.md`
- Poids samples w_i = 1/n init | alpha = 0.5 * log((1-error)/error)
- Mise à jour : w *= exp(-alpha * check) puis normalisation
- Predict SAMME : scores shape (n_samples, n_classes), argmax final

### 14. Gradient Boosting Regressor
**Fichier** : `regression_models/gradient_boosting.py` | **Doc** : `docs/gradient_boosting.md`
- F_0 = mean(y) | Résidus : r = y - F | F += eta * h_m(X)
- self.mean stocké comme float scalaire pour predict

### 15. Gradient Boosting Classifier
**Fichier** : `classification_models/gradient_boosting_classifier.py` | **Doc** : `docs/gradient_boosting.md`
- F représente les log-odds | p = sigmoid(F) | Pseudo-résidu : r = y - p
- F_0 = log(p0/(1-p0)) stocké comme scalaire | Prédiction : sigmoid(F) >= 0.5

### 16. SVM Linéaire + Kernel RBF
**Fichier** : `classification_models/svm.py` | **Doc** : `docs/svm.md`

**SVMClassifier (descente de gradient)**
- Hinge loss : (1/2)||w||^2 + C * sum(max(0, 1 - y*(wx+b)))
- Gradients sur points violants : dw = w - C*sum(y_i*x_i), db = -C*sum(y_i)

**KernelSVM (dual + cvxopt)**
- Kernel RBF : K(xi, xj) = exp(-gamma * ||xi-xj||^2)
- Dual : max sum(alpha) - (1/2)(alpha*y)^T K (alpha*y) | 0 <= alpha <= C, sum(alpha*y) = 0
- QP : P = yy^T * K, q = -1, G/h pour contraintes alpha, A = y^T, b = 0
- Support vectors : alpha > 1e-5 | Biais : mean(y_i - K[sv,:] @ (alpha*y))
- cvxopt : tc='d' obligatoire | b de shape (1,1)

### 17. SHAP Values (Monte Carlo)
**Fichier** : `explainability/shap.py` | **Doc** : `docs/shap.md`
- Shapley values : contribution marginale moyenne de chaque feature sur toutes les permutations
- f(S, x) estimé via vecteur hybride : features S=valeurs de x, reste=valeurs d'un point background z
- Complexité : O(n_samples * n_sampling * n_features)

### 18. CAH (Classification Ascendante Hiérarchique)
**Fichier** : `clustering/cah.py` | **Doc** : `docs/cah.md`
- Ascendant : chaque point = cluster init, fusion progressive des deux plus proches
- Pas de k à spécifier — hauteur de coupe choisie après sur le dendrogramme

**Formule de Lance-Williams** : d(AB,C) = a*d(A,C) + b*d(B,C) + c*d(A,B) + d*|d(A,C)-d(B,C)|

| Linkage | a | b | c | d |
|---|---|---|---|---|
| Single | 1/2 | 1/2 | 0 | -1/2 |
| Complete | 1/2 | 1/2 | 0 | +1/2 |
| Average | n1/(n1+n2) | n2/(n1+n2) | 0 | 0 |
| Ward | (n1+n3)/(n1+n2+n3) | (n2+n3)/(n1+n2+n3) | -n3/(n1+n2+n3) | 0 |

- history : liste de tuples (idx1, idx2, distance, taille) — n-1 fusions
- Après mise à jour D : forcer D[idx1,idx1] = inf (évite auto-fusion)
- cut(height) : rejouer history, s'arrêter quand distance > height

### 19. GMM (Gaussian Mixture Model)
**Fichier** : `clustering/gmm.py` | **Doc** : `docs/gmm.md`
- Pendant probabiliste de K-Means : assignation soft (probabilités) vs hard
- p(x) = Σ π_k · N(x | μ_k, Σ_k) | Paramètres : μ_k, Σ_k, π_k par cluster
- EM : E-step r(i,k) = π_k·N(x_i|μ_k,Σ_k) / Σ_j π_j·N(x_i|μ_j,Σ_j) (Bayes)
- M-step : μ_k = barycentre pondéré, Σ_k = covariance pondérée, π_k = Σ_i r(i,k)/n
- Mahalanobis vectorisé : `np.sum(diff @ inv(Σ_k) * diff, axis=1)` shape (n,)
- Covariance via sqrt trick : `X_w.T @ X_w` avec X_w = sqrt(r[:,k])[:,newaxis] * diff
- Convergence : |L_new - L_old| < tol sur log-vraisemblance L = Σ_i log(Σ_k π_k·N(x_i|μ_k,Σ_k))
- Init : μ_k = points aléatoires, Σ_k = I_p, π_k = 1/K

### 20. t-SNE (t-Distributed Stochastic Neighbor Embedding)
**Fichier** : `dimensionality_reduction/tsne.py` | **Doc** : `docs/tsne.md`
- Réduction non-linéaire : préserve la structure locale (petites distances) vs PCA (variance globale)
- Espace original : p_{j|i} = softmax gaussien avec σ_i adapté par perplexité (recherche binaire)
- Perplexité = 2^H(P_i) — contrôle le nombre effectif de voisins, σ_i adapté à la densité locale
- Symétrisation : p_{ij} = (p_{j|i} + p_{i|j}) / 2n — distribution globale sur les paires (somme = 1)
- Espace réduit : q_{ij} = noyau t-Student (1+||yi-yj||²)^{-1} / Σ_{k≠l} — queues lourdes vs gaussienne
- Crowding problem : queues lourdes forcent les clusters à s'écarter en 2D sans être trop pénalisés
- Loss : KL(P||Q) = Σ_{ij} p_{ij} log(p_{ij}/q_{ij}) — pénalise surtout les voisins proches mal représentés
- Gradient : 4 Σ_j (p_{ij}-q_{ij})(yi-yj)(1+||yi-yj||²)^{-1} — shape (n, n_dimensions) via broadcasting
- Init : Y ~ N(0, 1e-4) (valeurs petites pour fort gradient initial) | sigma_right = sqrt(max(D))

---

## Prochains algorithmes suggérés

### Niveau 3 — Classifieurs
- **Perceptron** — brique de base des réseaux de neurones

### Niveau 3 — Réseaux de neurones
- **MLP (Multi-Layer Perceptron)** — backpropagation, couches denses
- **CNN (Convolutional Neural Network)** — convolutions, pooling
