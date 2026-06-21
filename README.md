# ML from Scratch

> Reimplementing machine learning algorithms from the ground up — as a learning exercise to sharpen mathematical intuition and build a deep understanding of each algorithm.

---

## Objective

The goal of this repository is to **re-implement popular machine learning algorithms from scratch**, without relying on high-level libraries such as scikit-learn or PyTorch for the core logic. Each implementation serves as a hands-on exercise to:

- **Strengthen mathematical foundations** — linear algebra, calculus, probability and statistics.
- **Develop a deep understanding** of how each algorithm works internally.

Implementations are written in **Python**, using only low-level numerical libraries (NumPy) so that the focus remains on the algorithm itself. Each file can be run directly (`python path/to/file.py`) to compare results against scikit-learn.

---

## Algorithms

| # | Algorithm | Category | File | Docs |
|---|-----------|----------|------|------|
| 1 | Linear Regression | Supervised — Regression | `regression_models/linear_regression.py` | [doc](docs/linear_regression.md) |
| 2 | Lasso (L1) | Supervised — Regression | `regression_models/lasso.py` | [doc](docs/lasso.md) |
| 3 | Ridge (L2) | Supervised — Regression | `regression_models/ridge.py` | [doc](docs/ridge.md) |
| 4 | Logistic Regression | Supervised — Classification | `classification_models/logistic_regression.py` | [doc](docs/logistic_regression.md) |
| 5 | K-Means | Unsupervised — Clustering | `clustering/kmeans.py` | [doc](docs/kmeans.md) |
| 6 | KNN | Supervised — Classification | `classification_models/KNN_classification.py` | [doc](docs/knn.md) |
| 7 | Naive Bayes | Supervised — Classification | `classification_models/naive_bayes.py` | [doc](docs/naive_bayes.md) |
| 8 | Decision Tree Regressor | Supervised — Regression | `regression_models/decision_tree_regressor.py` | [doc](docs/decision_tree.md) |
| 9 | Decision Tree Classifier | Supervised — Classification | `classification_models/decision_tree_classifier.py` | [doc](docs/decision_tree.md) |
| 10 | PCA | Unsupervised — Dimensionality Reduction | `dimensionality_reduction/pca.py` | [doc](docs/pca.md) |
| 11 | Random Forest | Supervised — Classification | `classification_models/random_forest.py` | [doc](docs/random_forest.md) |
| 12 | DBSCAN | Unsupervised — Clustering | `clustering/dbscan.py` | [doc](docs/dbscan.md) |
| 13 | AdaBoost | Supervised — Classification | `classification_models/adaboost.py` | [doc](docs/adaboost.md) |
| 14 | Gradient Boosting Regressor | Supervised — Regression | `regression_models/gradient_boosting.py` | [doc](docs/gradient_boosting.md) |
| 15 | Gradient Boosting Classifier | Supervised — Classification | `classification_models/gradient_boosting_classifier.py` | [doc](docs/gradient_boosting.md) |
| 16 | SVM (Linear + Kernel RBF) | Supervised — Classification | `classification_models/svm.py` | [doc](docs/svm.md) |
| 17 | SHAP Values | Explainability | `explainability/shap.py` | [doc](docs/shap.md) |

---

## Repository Structure

```
ML_from_scratch/
├── regression_models/         # Linear, Lasso, Ridge, Decision Tree, Gradient Boosting
├── classification_models/     # Logistic, KNN, Naive Bayes, Decision Tree, Random Forest,
│                              # AdaBoost, Gradient Boosting, SVM
├── clustering/                # KMeans, DBSCAN
├── dimensionality_reduction/  # PCA
├── explainability/            # SHAP
├── bases/                     # Shared base classes (DecisionTreeBase, Node)
├── helpers/                   # Activation functions, utilities, exceptions
├── loss_functions/            # MSE, Binary Cross-Entropy
└── docs/                      # Theory and implementation notes for each algorithm
```

---

## Getting Started

### Prerequisites

- Python 3.8+
- NumPy, scikit-learn (for benchmarking), cvxopt (for Kernel SVM)

### Installation

```bash
git clone https://github.com/4xel-C/ML_from_scratch.git
cd ML_from_scratch

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install numpy scikit-learn cvxopt
```

### Running an implementation

Each file can be executed directly to benchmark against scikit-learn:

```bash
python regression_models/linear_regression.py
python classification_models/svm.py
python clustering/kmeans.py
# etc.
```

---

## License

This project is open-source and available under the [MIT License](LICENSE).
