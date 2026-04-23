"""Classification model factory.

Covers required topics:
  - Logistic Regression Classifier
  - K-Nearest Neighbors Classifier
  - Non-parametric baseline (Decision Tree Classifier)
  - Baseline (Gaussian Naive Bayes)
"""
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB


def build_classification_models() -> dict:
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            multi_class="multinomial",
            random_state=42,
        ),
        "knn_classifier": KNeighborsClassifier(n_neighbors=7, weights="distance"),
        "decision_tree_classifier": DecisionTreeClassifier(
            max_depth=6, min_samples_leaf=5, random_state=42
        ),
        "naive_bayes": GaussianNB(),
    }
