"""Regression model factory.

Covers required topics:
  - Multivariate Linear Regression (LinearRegression)
  - Regularized variants (Ridge)
  - Non-parametric models (KNN Regressor, Decision Tree Regressor)
"""
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor


def build_regression_models() -> dict:
    """Return a dict of name -> fresh model instance."""
    return {
        "linear_regression": LinearRegression(),
        "ridge_regression": Ridge(alpha=1.0, random_state=42),
        "knn_regressor": KNeighborsRegressor(n_neighbors=7, weights="distance"),
        "decision_tree_regressor": DecisionTreeRegressor(
            max_depth=6, min_samples_leaf=5, random_state=42
        ),
    }
