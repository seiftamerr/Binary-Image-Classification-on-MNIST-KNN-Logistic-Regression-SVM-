import numpy as np

class MultiClassLinearSVM:
    def __init__(self, learning_rate=0.001, num_iterations=1000, regularization_strength=0.001):
        # Hyperparameters
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.regularization_strength = regularization_strength

        # Model parameters
        self.weights = None
        self.bias = None

        # Store loss values during training
        self.loss_history = []

    def train_model(self, X, y):
        # Get dataset dimensions
        num_samples, num_features = X.shape
        num_classes = len(np.unique(y))

        # Initialize weights with small random values
        self.weights = 0.001 * np.random.randn(num_features, num_classes)

        # Initialize bias for each class
        self.bias = np.zeros(num_classes)

        # Training loop
        for iteration in range(self.num_iterations):

            # Step 1: Compute class scores
            class_scores = np.dot(X, self.weights) + self.bias

            # Step 2: Extract correct class scores
            true_class_scores = class_scores[np.arange(num_samples), y].reshape(-1, 1)

            # Step 3: Compute multiclass hinge loss margins
            margins = np.maximum(0, class_scores - true_class_scores + 1)

            # Ignore correct class margins
            margins[np.arange(num_samples), y] = 0

            # Step 4: Compute total loss
            data_loss = np.mean(np.sum(margins, axis=1))
            regularization_loss = self.regularization_strength * np.sum(self.weights ** 2)
            total_loss = data_loss + regularization_loss

            # Store loss history
            self.loss_history.append(total_loss)

            # Step 5: Create margin violation matrix
            margin_violation_matrix = (margins > 0).astype(float)

            # Count violations for each sample
            violation_count = np.sum(margin_violation_matrix, axis=1)

            # Assign negative count to correct class
            margin_violation_matrix[np.arange(num_samples), y] = -violation_count

            # Step 6: Compute gradients
            weight_gradient = (
                (1 / num_samples) * np.dot(X.T, margin_violation_matrix)
                + 2 * self.regularization_strength * self.weights
            )

            bias_gradient = (1 / num_samples) * np.sum(margin_violation_matrix, axis=0)

            # Step 7: Update parameters
            self.weights -= self.learning_rate * weight_gradient
            self.bias -= self.learning_rate * bias_gradient

            # Print training progress
            if iteration % 500 == 0:
                print(f"Iteration {iteration}: Loss = {total_loss:.4f}")

    def predict_classes(self, X):
        class_scores = np.dot(X, self.weights) + self.bias
        return np.argmax(class_scores, axis=1)