import numpy as np

class MeanBaseline:
    def fit(self, X, y):
        self.mean = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean)