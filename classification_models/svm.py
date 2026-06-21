"""
Binary linear SVM (soft-margin version) optimized with gradient descent.

General idea
- The model looks for a separating hyperplane: w^T x + b = 0
- The objective is to maximize the margin between classes.
- Margin width is proportional to 2 / ||w||, so we minimize 0.5 * ||w||^2.

Notation
- x_i: feature vector of sample i
- y_i: label of sample i in {-1, +1}
- w: weight vector
- b: bias
- C: regularization coefficient

Soft-margin constraints
- Ideal margin condition: y_i * (w^T x_i + b) >= 1
- With slack variables xi_i >= 0:
    y_i * (w^T x_i + b) >= 1 - xi_i

Objective function (primal)
- Minimize: 0.5 * ||w||^2 + C * sum(xi_i)
- Equivalent hinge-loss form:
    L = 0.5 * ||w||^2 + C * sum(max(0, 1 - y_i * (w^T x_i + b)))

Effect of C
- Large C: stronger penalty on violations, narrower margin
- Small C: more tolerated violations, wider margin

Gradients used here (on violating points)
- If y_i * (w^T x_i + b) >= 1:
    dL/dw = w
    dL/db = 0
- If y_i * (w^T x_i + b) < 1:
    dL/dw = w - C * sum(y_i * x_i)
    dL/db = -C * sum(y_i)

Prediction
- Score: s = w^T x + b
- Predicted class: 1 if s > 0, else 0

Kernel SVM
- Idea: project points into a higher-dimensional space with phi(x),
    then perform a linear separation in that space.
- We start from the same soft-margin problem:
    minimize 0.5 * ||w||^2 + C * sum(xi_i)
    subject to xi_i >= 0 and y_i * (w^T x_i + b) >= 1 - xi_i.

- We build the Lagrangian with multipliers alpha_i and lambda_i:
    L(w, b, xi, alpha, lambda)
    = 0.5 * ||w||^2 + C * sum(xi_i)
        - sum(alpha_i * (y_i * (w^T x_i + b) - 1 + xi_i))
        - sum(lambda_i * xi_i)

- Important stationary condition:
    dL/dw = 0  =>  w = sum(alpha_i * y_i * x_i)

- After replacing primal variables in the Lagrangian and applying KKT,
    we obtain the dual problem:
    maximize sum(alpha_i) - 0.5 * sum_i sum_j(alpha_i * alpha_j * y_i * y_j * x_i^T x_j)

- The kernel trick replaces x_i^T x_j with K(x_i, x_j) = phi(x_i)^T phi(x_j),
    which enables a nonlinear decision boundary without explicitly computing phi.
    K_ij = K(x_i, x_j) = phi(x_i)^T phi(x_j)
    It is positive semidefinite, so all eigenvalues are nonnegative. (Convex)

Common kernels
- Polynomial: K(x, z) = (x^T z + c)^d
- Gaussian (RBF): K(x, z) = exp(-gamma * ||x - z||^2): Measure similarity between 2 points, with gamma controlling the width of the Gaussian.


We will implement the gaussian kernel SVM. For the algorithm to work, we have to compute the
Gram matrix, measuring the proximity bet all pairs of points.

We then solve the QP problem, finding alpha. Knowing that w = sum(alpha * yi * xi) from the derivative,
we can reinjected this equation into the decision function:
f(x) = wx + b = sum(alphai yi xi) * x + b; raplacing all saclar products <xi, x> by the gaussian kernel
"""

import numpy as np
from cvxopt import matrix, solvers
from numpy.typing import NDArray


class SVMClassifier:
    def __init__(
        self,
        C: float = 1,
        learning_rate: float = 0.1,
        n_iterations: int = 100,
        tol: float = 1e-4,
    ):
        self.C = C
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.tol = tol

        # Other attributes declaration
        self.fitted: bool = False

    def fit(self, X: NDArray, y: NDArray) -> None:
        """We suppose the class between 0 and 1"""

        # remap the y
        self.y = np.where(y == 0, -1, 1)

        # initialization of the weights
        self.w = np.zeros(X.shape[1])
        self.b = 0

        for i in range(self.n_iterations):
            # Compute and the distances to the hyperplane (shape n)
            dist = X @ self.w + self.b

            support_indices = np.where(self.y * dist < 1)

            # Compute the hinge loss for the violating point (no loss for the corrected classified)
            dhingedw = self.C * (self.y[support_indices].T @ X[support_indices])

            dhingedb = -self.C * np.sum(self.y[support_indices])

            # compute the gradients
            dldw = self.w - dhingedw
            dldb = 0 + dhingedb

            # Update the weights in the opposite direction of the gradient
            self.w = self.w - (self.learning_rate * dldw)
            self.b = self.b - (self.learning_rate * dldb)

            # Check convergence and break if the update loop if no notable evolution
            if (
                all(self.learning_rate * dldw <= self.tol)
                and self.learning_rate * dldb <= self.tol
            ):
                break

            if i == self.n_iterations - 1:
                print("Algorithm did not converge")

        self.fitted = True

    def predict(self, X: NDArray) -> NDArray:
        predictions = (X @ self.w) + self.b

        predictions = np.where(predictions > 0, 1, 0)

        return predictions


class KernelSVM:
    """Implementation of the kernel SVM."""

    def __init__(self, C: float, gamma: float):
        """Initialize the KernelSVM.

        Args:
            C (float): The penalty strength (on slack variables)
            gamma(float): The width of the kernel gaussian
        """
        self.C = C
        self.gamma = gamma

    def _rbf_kernel(self, X1: NDArray, X2: NDArray):
        """Compute the Gram matrix using the kernel"""

        # For each point, compute the difference with all other points
        substraction = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]

        # shape npoints proximity with ksupport points (sum on feature axis)
        gram = np.sum(substraction**2, axis=2)

        return np.exp(-self.gamma * gram)

    def fit(self, X: NDArray, y: NDArray):
        # Remapping y -1 / 1
        self.y = y.copy()
        self.y[self.y == 0] = -1

        self.X = X
        n = len(X)

        # Gram calculation
        self.gram = self._rbf_kernel(X, X)

        # Solve the QP problem: detail of each problem member (adapted for cvxopt)
        P = self.y[:, np.newaxis] * self.y[np.newaxis, :] * self.gram
        q = -np.ones(n)
        A = self.y.reshape(1, n)
        b = np.zeros((1, 1))

        # We stack constraints on alpha (>= 0 and <= C)
        G = np.vstack([-np.eye(n), np.eye(n)])
        h = np.hstack([np.zeros(n), np.ones(n) * self.C])

        # Solve the QP problem
        P = matrix(P, tc="d")
        q = matrix(q, tc="d")
        G = matrix(G, tc="d")
        h = matrix(h, tc="d")
        A = matrix(A, tc="d")
        b = matrix(b, tc="d")

        sol = solvers.qp(P, q, G, h, A, b)
        self.alpha = np.ravel(sol["x"])

        # Find the support vectors
        supports = self.alpha > 0

        # Compute the bias only for the support points
        predictions = self.gram[supports, :] @ (self.alpha * self.y)
        self.b = np.mean(self.y[supports] - predictions)

    def predict(self, X: NDArray):
        # Compute the kernel for the new points
        gram = self._rbf_kernel(X, self.X)

        predictions = np.sign(gram @ (self.alpha * self.y) + self.b)
        predictions = np.where(predictions == -1, 0, 1)

        return predictions


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    from sklearn.datasets import make_classification
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC

    np.random.seed(42)

    # Linear benchmark dataset
    X_lin, y_lin = make_classification(
        n_samples=300,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        class_sep=1.5,
        random_state=42,
    )
    Xl_train, Xl_test, yl_train, yl_test = train_test_split(
        X_lin, y_lin, test_size=0.25, random_state=42
    )

    custom_linear = SVMClassifier(C=1.0, learning_rate=0.001, n_iterations=1500)
    custom_linear.fit(Xl_train, yl_train)
    pred_custom_linear = custom_linear.predict(Xl_test)

    sk_linear = SVC(kernel="linear", C=1.0, random_state=42)
    sk_linear.fit(Xl_train, yl_train)
    pred_sk_linear = sk_linear.predict(Xl_test)

    # Nonlinear benchmark dataset
    X_rbf, y_rbf = make_classification(
        n_samples=220,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_clusters_per_class=1,
        class_sep=0.9,
        random_state=7,
    )
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(
        X_rbf, y_rbf, test_size=0.25, random_state=42
    )

    custom_kernel = KernelSVM(C=1.0, gamma=0.8)
    custom_kernel.fit(Xr_train, yr_train)
    pred_custom_kernel = custom_kernel.predict(Xr_test)

    sk_kernel = SVC(kernel="rbf", C=1.0, gamma=0.8, random_state=42)
    sk_kernel.fit(Xr_train, yr_train)
    pred_sk_kernel = sk_kernel.predict(Xr_test)

    print("=== Linear SVM comparison ===")
    print(f"Custom SVM accuracy:  {accuracy_score(yl_test, pred_custom_linear):.4f}")
    print(f"Sklearn accuracy:     {accuracy_score(yl_test, pred_sk_linear):.4f}")

    print("\n=== Kernel SVM (RBF) comparison ===")
    print(f"Custom Kernel accuracy: {accuracy_score(yr_test, pred_custom_kernel):.4f}")
    print(f"Sklearn accuracy:       {accuracy_score(yr_test, pred_sk_kernel):.4f}")
