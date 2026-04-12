from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB

from classification_models import NaiveBayes

X, y = make_classification(n_samples=300, n_features=4, random_state=42)

model = NaiveBayes()
model.fit(X, y)
preds = model.predict(X)

sk = GaussianNB()
sk.fit(X, y)
sk_preds = sk.predict(X)

print(f"Ton accuracy    : {accuracy_score(y, preds):.4f}")
print(f"Sklearn accuracy: {accuracy_score(y, sk_preds):.4f}")
