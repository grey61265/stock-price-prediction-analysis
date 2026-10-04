import numpy as np


class LinearRegressionScratch:

    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.w = None
        self.b = 0
        self.losses = []

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        m, n = X.shape

        self.w = np.zeros(n)
        self.b = 0
        self.losses = []

        for _ in range(self.epochs):
            y_pred = X @ self.w + self.b
            error = y_pred - y
            self.losses.append(np.mean(error ** 2))

            dw = (2 / m) * (X.T @ error)
            db = (2 / m) * np.sum(error)

            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return X @ self.w + self.b