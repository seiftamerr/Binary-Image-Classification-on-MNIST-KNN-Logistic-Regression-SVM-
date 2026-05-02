import numpy as np

class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)

        n_samples, n_features = X.shape

        # Convert labels to -1 and 1
        y_ = np.where(y <= 0, -1, 1)

        # Initialize weights and bias
        self.w = np.zeros(n_features, dtype=np.float32)
        self.b = 0.0

        # Gradient Descent
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1

                if condition:
                    dw = 2 * self.lambda_param * self.w
                    db = 0
                else:
                    dw = 2 * self.lambda_param * self.w - y_[idx] * x_i
                    db = y_[idx]

                self.w -= self.lr * dw
                self.b -= self.lr * db

    def decision_function(self, X):
        X = np.asarray(X, dtype=np.float32)
        return np.dot(X, self.w) - self.b

    def predict(self, X):
        linear_output = self.decision_function(X)
        return np.where(linear_output >= 0, 1, 0)

    def accuracy(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)