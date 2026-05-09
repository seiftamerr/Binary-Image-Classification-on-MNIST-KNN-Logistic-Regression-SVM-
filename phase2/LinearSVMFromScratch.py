# =========================
# 13. Linear SVM From Scratch
# =========================
import numpy as np
class LinearSVMFromScratch:
    def __init__(self, learning_rate=0.001, num_iterations=1000, reg_strength=0.001):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.reg_strength = reg_strength
        self.weights = None
        self.bias = None
        self.loss_history = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        num_classes = len(np.unique(y))

        self.weights = 0.001 * np.random.randn(n_features, num_classes)
        self.bias = np.zeros(num_classes)

        for iteration in range(self.num_iterations):
            scores = np.dot(X, self.weights) + self.bias
            correct_scores = scores[np.arange(n_samples), y].reshape(-1, 1)

            margins = np.maximum(0, scores - correct_scores + 1)
            margins[np.arange(n_samples), y] = 0

            loss = np.mean(np.sum(margins, axis=1)) + self.reg_strength * np.sum(self.weights ** 2)
            self.loss_history.append(loss)

            binary = (margins > 0).astype(float)
            row_sum = np.sum(binary, axis=1)
            binary[np.arange(n_samples), y] = -row_sum

            dw = (1 / n_samples) * np.dot(X.T, binary) + 2 * self.reg_strength * self.weights
            db = (1 / n_samples) * np.sum(binary, axis=0)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            if iteration % 500 == 0:
                print(f"Iteration {iteration}: Loss = {loss:.4f}")

    def predict(self, X):
        scores = np.dot(X, self.weights) + self.bias
        return np.argmax(scores, axis=1)