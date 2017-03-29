import numpy as np

from sklearn.neural_network import MLPClassifier

# for testing
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

class MLPEnsemble():
    def __init__(self, n_classifiers=5, clf_args={}):
        self.classifiers = []
        self.n_classifiers = n_classifiers
        for _ in range(self.n_classifiers):
            self.classifiers.append(MLPClassifier(**clf_args))

    def fit(self, X, y):
        self.classifiers_ = []
        for clf in self.classifiers:
            fitted_clf = clf.fit(X, y)
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
