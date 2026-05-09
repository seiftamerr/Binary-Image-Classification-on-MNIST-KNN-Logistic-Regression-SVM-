import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# =====================================================
# MULTICLASS LOGISTIC REGRESSION IMPLEMENTATION
# =====================================================

def stable_softmax(scores):
    """
    Softmax with numerical stabilization.
    Prevents overflow issues during exponentiation.
    """
    shifted_scores = scores - np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(shifted_scores)

    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


def compute_gradient(weights, X_input, Y_encoded):
    """
    Computes gradient for multiclass logistic regression.
    """

    total_samples = X_input.shape[0]

    logits = X_input @ weights
    probabilities = stable_softmax(logits)

    difference = probabilities - Y_encoded

    gradient = (X_input.T @ difference) / total_samples

    return gradient


def train_softmax_regression(
        X_train,
        Y_train,
        learning_rate=0.01,
        max_iterations=5000,
        tolerance=1e-7,
        regularization=0.0
):
    """
    Trains multinomial logistic regression using gradient descent.
    """

    num_classes = Y_train.shape[1]

    # Add bias column
    X_bias = np.c_[np.ones((X_train.shape[0], 1)), X_train]

    weights = np.zeros((X_bias.shape[1], num_classes))

    for step in range(max_iterations):

        grad = compute_gradient(weights, X_bias, Y_train)

        # Weight update
        weights -= learning_rate * (grad + regularization * weights)

        # Early stopping condition
        if np.linalg.norm(grad) < tolerance:
            print(f"Optimization stopped early at iteration {step + 1}")
            break

        # Display progress occasionally
        if (step + 1) % 1000 == 0:

            logits = X_bias @ weights
            probs = stable_softmax(logits)

            cross_entropy = -np.mean(
                np.sum(Y_train * np.log(probs + 1e-12), axis=1)
            )

            reg_term = (regularization / 2) * np.sum(weights ** 2)

            total_loss = cross_entropy + reg_term

            print(f"[Iteration {step + 1}/{max_iterations}] Loss = {total_loss:.4f}")

    return weights


def generate_predictions(X_data, trained_weights):
    """
    Predicts digit classes using trained model weights.
    """

    X_bias = np.c_[np.ones((X_data.shape[0], 1)), X_data]

    prediction_scores = stable_softmax(X_bias @ trained_weights)

    return np.argmax(prediction_scores, axis=1)


# =====================================================
# DATASET PREPARATION
# =====================================================

print("Downloading MNIST dataset...")
X, y = fetch_openml(
    'mnist_784',
    version=1,
    return_X_y=True,
    as_frame=False
)

print("Dataset loaded successfully.")

# Convert labels to integers
y = y.astype(int)

# One-hot encoding
Y_encoded = np.eye(10)[y]

# -----------------------------------------------------
# Dataset Splitting
# -----------------------------------------------------

print("Creating train / validation / test splits...")

X_remaining, X_test, y_remaining, y_test, Y_remaining, Y_test = train_test_split(
    X,
    y,
    Y_encoded,
    test_size=10000,
    random_state=42
)

X_train, X_validation, y_train, y_validation, Y_train, Y_validation = train_test_split(
    X_remaining,
    y_remaining,
    Y_remaining,
    test_size=10000,
    random_state=42
)

# -----------------------------------------------------
# Feature Scaling
# -----------------------------------------------------

print("Applying feature normalization...")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_validation_scaled = scaler.transform(X_validation)

X_test_scaled = scaler.transform(X_test)

# =====================================================
# HYPERPARAMETER SEARCH
# =====================================================

pca_options = [50, 100, 200]

learning_rate_options = [0.01, 0.1]

regularization_options = [0.1, 1.0]

epochs = 5000

best_validation_score = 0

best_configuration = {}

optimal_weights = None

optimal_pca = None

print("\nBeginning hyperparameter experiments...")
print("===================================================")

for pca_size in pca_options:

    for lr in learning_rate_options:

        for reg in regularization_options:

            print(
                f"Current setup -> PCA Components: {pca_size}, "
                f"Learning Rate: {lr}, Regularization: {reg}"
            )

            # PCA transformation
            pca_model = PCA(
                n_components=pca_size,
                random_state=42
            )

            X_train_pca = pca_model.fit_transform(X_train_scaled)

            X_validation_pca = pca_model.transform(X_validation_scaled)

            # Model training
            trained_weights = train_softmax_regression(
                X_train_pca,
                Y_train,
                learning_rate=lr,
                max_iterations=epochs,
                regularization=reg
            )

            # Validation predictions
            validation_predictions = generate_predictions(
                X_validation_pca,
                trained_weights
            )

            validation_accuracy = accuracy_score(
                y_validation,
                validation_predictions
            )

            print(
                f"Validation Accuracy: "
                f"{validation_accuracy * 100:.2f}%\n"
            )

            # Store best configuration
            if validation_accuracy > best_validation_score:

                best_validation_score = validation_accuracy

                best_configuration = {
                    "PCA": pca_size,
                    "Learning Rate": lr,
                    "Regularization": reg
                }

                optimal_weights = trained_weights

                optimal_pca = pca_model

print("===================================================")
print("Hyperparameter search completed.")
print(f"Best Validation Accuracy: {best_validation_score * 100:.2f}%")
print(f"Optimal Settings: {best_configuration}")
print("===================================================")

# =====================================================
# FINAL MODEL TESTING
# =====================================================

print("\nRunning final evaluation on test data...")

X_test_pca = optimal_pca.transform(X_test_scaled)

X_train_best_pca = optimal_pca.transform(X_train_scaled)

train_predictions = generate_predictions(
    X_train_best_pca,
    optimal_weights
)

test_predictions = generate_predictions(
    X_test_pca,
    optimal_weights
)

# =====================================================
# PERFORMANCE METRICS
# =====================================================

train_accuracy = accuracy_score(y_train, train_predictions)

test_accuracy = accuracy_score(y_test, test_predictions)

macro_precision = precision_score(
    y_test,
    test_predictions,
    average='macro'
)

macro_recall = recall_score(
    y_test,
    test_predictions,
    average='macro'
)

macro_f1 = f1_score(
    y_test,
    test_predictions,
    average='macro'
)

weighted_precision = precision_score(
    y_test,
    test_predictions,
    average='weighted'
)

weighted_recall = recall_score(
    y_test,
    test_predictions,
    average='weighted'
)

weighted_f1 = f1_score(
    y_test,
    test_predictions,
    average='weighted'
)

conf_matrix = confusion_matrix(
    y_test,
    test_predictions,
    labels=np.arange(10)
)

# =====================================================
# RESULTS SUMMARY
# =====================================================

print(f"\nTraining Accuracy : {train_accuracy * 100:.2f}%")

print(f"Testing Accuracy  : {test_accuracy * 100:.2f}%")

print("\n===== Macro Metrics =====")

print(f"Precision : {macro_precision * 100:.2f}%")

print(f"Recall    : {macro_recall * 100:.2f}%")

print(f"F1 Score  : {macro_f1 * 100:.2f}%")

print("\n===== Weighted Metrics =====")

print(f"Precision : {weighted_precision * 100:.2f}%")

print(f"Recall    : {weighted_recall * 100:.2f}%")

print(f"F1 Score  : {weighted_f1 * 100:.2f}%")

print("\nDetailed Classification Report")

print(classification_report(
    y_test,
    test_predictions,
    digits=4
))

print("\nConfusion Matrix")

print("Rows -> Actual Class")
print("Columns -> Predicted Class")

print(conf_matrix)