import numpy as np

from sklearn.neural_network import MLPClassifier

# for testing
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class MLPEnsemble():
    def __init__(self, n_classifiers=1, n_samples=1.0, clf_args={}):
        self.classifiers = []
        self.n_classifiers = n_classifiers
        self.n_samples = n_samples
        for _ in range(self.n_classifiers):
            self.classifiers.append(MLPClassifier(**clf_args))

    def fit(self, X, y):
        self.classifiers_ = []
        for clf in self.classifiers:
            # get random sample with replacement
            n_samples = int(X.shape[0] * self.n_samples)
            indices = np.random.choice(X.shape[0], n_samples)
            X_, y_ = X[indices], y[indices]

            fitted_clf = clf.fit(X_, y_)
            self.classifiers_.append(fitted_clf)
        return self

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_proba(self, X):
        probas = np.asarray([clf.predict_proba(X)
            for clf in self.classifiers_])
        avg_proba = np.average(probas, axis=0)
        return avg_proba


data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target)
mlp = MLPEnsemble(n_classifiers=100, n_samples=0.5)
mlp.fit(X_train, y_train)
print(accuracy_score(y_test, mlp.predict(X_test)))
