import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import fetch_openml
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from sklearn.model_selection import (
    train_test_split,
    KFold
)

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score

from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier
)


# =========================================================
# OPTIMIZED CUSTOM KNN IMPLEMENTATION
# =========================================================

class CustomKNN:

    def __init__(self, k=3):
        self.k = k

    def fit(self, X_train, y_train):

        self.X_train = np.asarray(
            X_train,
            dtype=np.float32
        )

        self.y_train = np.asarray(
            y_train,
            dtype=np.int64
        )

        # Precompute squared norms
        self.train_norms = np.sum(
            self.X_train ** 2,
            axis=1
        )

    def _compute_distances(self, sample):

        sample_norm = np.sum(sample ** 2)

        distances = (
            self.train_norms
            + sample_norm
            - 2 * np.dot(self.X_train, sample)
        )

        return distances

    def _predict_single(self, sample):

        distances = self._compute_distances(sample)

        # Faster than full sorting
        nearest_indices = np.argpartition(
            distances,
            self.k
        )[:self.k]

        nearest_labels = self.y_train[
            nearest_indices
        ]

        unique_labels, counts = np.unique(
            nearest_labels,
            return_counts=True
        )

        prediction = unique_labels[
            np.argmax(counts)
        ]

        return prediction

    def predict(self, X_test):

        X_test = np.asarray(
            X_test,
            dtype=np.float32
        )

        predictions = []

        for sample in X_test:

            prediction = self._predict_single(
                sample
            )

            predictions.append(prediction)

        return np.array(predictions)

    def accuracy(self, X_test, y_test):

        predictions = self.predict(X_test)

        accuracy = np.mean(
            predictions == y_test
        )

        return accuracy


# =========================================================
# LOAD MNIST DATASET
# =========================================================

print("Downloading MNIST dataset...")

X, y = fetch_openml(
    'mnist_784',
    version=1,
    return_X_y=True,
    as_frame=False
)

y = y.astype(np.int64)

print("Dataset loaded successfully.")

print("Dataset shape:", X.shape)


# =========================================================
# PREPROCESSING
# =========================================================

print("\nStarting preprocessing...")

# Normalize pixel values
X = X / 255.0

# Standardization
scaler = StandardScaler()

X = scaler.fit_transform(X)

print("Preprocessing completed")


# =========================================================
# TRAIN / TEST SPLIT
# =========================================================

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Validation split
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.2,
    random_state=42,
    stratify=y_train_full
)

print("\nData splitting completed")

print("Training samples  :", X_train.shape[0])
print("Validation samples:", X_val.shape[0])
print("Testing samples   :", X_test.shape[0])


# =========================================================
# BASELINE CUSTOM KNN
# =========================================================

print("\n==============================")
print("BASELINE CUSTOM KNN")
print("==============================")

baseline_model = CustomKNN(k=3)

baseline_model.fit(X_train, y_train)

baseline_accuracy = baseline_model.accuracy(
    X_val,
    y_val
)

print("Validation Accuracy:", baseline_accuracy)


# =========================================================
# HYPERPARAMETER TUNING
# CROSS VALIDATION
# =========================================================

print("\n==============================")
print("HYPERPARAMETER TUNING")
print("CROSS VALIDATION")
print("==============================")

k_values = [1, 3, 5, 7, 9]

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

best_k = None
best_score = 0

for current_k in k_values:

    fold_scores = []

    print(f"\nTesting k = {current_k}")

    for fold_number, (train_index, val_index) in enumerate(
        kf.split(X_train),
        start=1
    ):

        print(f"Fold {fold_number}/5")

        X_fold_train = X_train[train_index]
        X_fold_val = X_train[val_index]

        y_fold_train = y_train[train_index]
        y_fold_val = y_train[val_index]

        model = CustomKNN(k=current_k)

        model.fit(
            X_fold_train,
            y_fold_train
        )

        score = model.accuracy(
            X_fold_val,
            y_fold_val
        )

        fold_scores.append(score)

    average_score = np.mean(fold_scores)

    print(
        f"Average Accuracy = "
        f"{average_score:.4f}"
    )

    if average_score > best_score:

        best_score = average_score
        best_k = current_k


print("\nBest k value:", best_k)

print(
    "Best Cross Validation Accuracy:",
    best_score
)


# =========================================================
# BIAS - VARIANCE ANALYSIS
# =========================================================

print("\n==============================")
print("BIAS - VARIANCE ANALYSIS")
print("==============================")

print("""
Small k values:
- Low bias
- High variance
- Sensitive to noise
- Risk of overfitting

Large k values:
- Higher bias
- Lower variance
- Smoother decision boundaries
- Risk of underfitting
""")


# =========================================================
# FINAL CUSTOM KNN
# =========================================================

print("\n==============================")
print("FINAL CUSTOM KNN")
print("==============================")

final_knn = CustomKNN(k=best_k)

final_knn.fit(X_train, y_train)

knn_predictions = final_knn.predict(X_test)

knn_accuracy = accuracy_score(
    y_test,
    knn_predictions
)

knn_precision = precision_score(
    y_test,
    knn_predictions,
    average='weighted'
)

knn_recall = recall_score(
    y_test,
    knn_predictions,
    average='weighted'
)

knn_f1 = f1_score(
    y_test,
    knn_predictions,
    average='weighted'
)

knn_conf_matrix = confusion_matrix(
    y_test,
    knn_predictions
)

print(
    "Custom KNN Test Accuracy:",
    knn_accuracy
)

print(
    "Precision:",
    knn_precision
)

print(
    "Recall:",
    knn_recall
)

print(
    "F1 Score:",
    knn_f1
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        knn_predictions
    )
)

print("\nConfusion Matrix:")

print(knn_conf_matrix)


# =========================================================
# RANDOM FOREST
# =========================================================

print("\n==============================")
print("RANDOM FOREST")
print("==============================")

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)

rf_accuracy = accuracy_score(
    y_test,
    rf_predictions
)

print(
    "Random Forest Accuracy:",
    rf_accuracy
)


# =========================================================
# BOOSTING
# =========================================================

print("\n==============================")
print("ADABOOST")
print("==============================")

from sklearn.tree import DecisionTreeClassifier

weak_learner = DecisionTreeClassifier(
    max_depth=2,
    random_state=42
)

boost_model = AdaBoostClassifier(
    estimator=weak_learner,
    n_estimators=200,
    learning_rate=0.5,
    algorithm='SAMME',
    random_state=42
)

boost_model.fit(X_train, y_train)

boost_predictions = boost_model.predict(X_test)

boost_accuracy = accuracy_score(
    y_test,
    boost_predictions
)

print(
    "AdaBoost Accuracy:",
    boost_accuracy
)
# =========================================================
# REGULARIZATION
# =========================================================

print("\n==============================")
print("REGULARIZATION")
print("==============================")

# L1 Regularization
l1_model = LogisticRegression(
    penalty='l1',
    solver='saga',
    max_iter=1000,
    random_state=42
)

l1_model.fit(X_train, y_train)

l1_predictions = l1_model.predict(X_test)

l1_accuracy = accuracy_score(
    y_test,
    l1_predictions
)

print(
    "L1 Regularization Accuracy:",
    l1_accuracy
)

# L2 Regularization
l2_model = LogisticRegression(
    penalty='l2',
    solver='lbfgs',
    max_iter=1000,
    random_state=42
)

l2_model.fit(X_train, y_train)

l2_predictions = l2_model.predict(X_test)

l2_accuracy = accuracy_score(
    y_test,
    l2_predictions
)

print(
    "L2 Regularization Accuracy:",
    l2_accuracy
)

# =========================================================
# FINAL MODEL COMPARISON
# =========================================================

print("\n==============================")
print("FINAL MODEL COMPARISON")
print("==============================")

print(
    f"Custom KNN Accuracy   : "
    f"{knn_accuracy:.4f}"
)

print(
    f"Custom KNN Precision  : "
    f"{knn_precision:.4f}"
)

print(
    f"Custom KNN Recall     : "
    f"{knn_recall:.4f}"
)

print(
    f"Custom KNN F1-Score   : "
    f"{knn_f1:.4f}"
)

print()

print(
    f"Random Forest Accuracy: "
    f"{rf_accuracy:.4f}"
)

print()

print(
    f"AdaBoost Accuracy     : "
    f"{boost_accuracy:.4f}"
)

print()

print(
    f"L1 Regularization     : "
    f"{l1_accuracy:.4f}"
)

print(
    f"L2 Regularization     : "
    f"{l2_accuracy:.4f}"
)


# =========================================================
# BEST MODEL
# =========================================================

results = {
    "Custom KNN": knn_accuracy,
    "Random Forest": rf_accuracy,
    "AdaBoost": boost_accuracy,
    "L1 Logistic Regression": l1_accuracy,
    "L2 Logistic Regression": l2_accuracy
}

best_model = max(
    results,
    key=results.get
)

print(
    "\nBest Performing Model:",
    best_model
)

print(
    "Best Accuracy:",
    results[best_model]
)