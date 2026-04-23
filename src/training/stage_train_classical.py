"""DVC stage 3: train all classical models.

Trains:
  - Regression: LinearRegression, Ridge, KNN Regressor, DecisionTree Regressor
  - Classification: Logistic Regression, KNN Classifier, Decision Tree, Gaussian NB
  - Clustering: K-Means with segment labels

Persists models + metrics JSON + plots.
"""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from src.config import SETTINGS, resolve_path
from src.preprocessing.preprocessor import Preprocessor
from src.models.regression import build_regression_models
from src.models.classification import build_classification_models
from src.models.clustering import build_kmeans, SEGMENT_NAMES
from src.evaluation.metrics import regression_metrics, classification_metrics
from src.evaluation.cross_validation import cv_classification, cv_regression
from src.evaluation.plots import (
    plot_confusion_matrix,
    plot_residuals,
    plot_metric_bars,
    plot_cluster_scatter,
)
from src.utils.helpers import ensure_dir, write_json
from src.utils.logging import get_logger

log = get_logger(__name__)

PROCESSED = resolve_path(SETTINGS["paths"]["data_processed"])
MODELS = resolve_path(SETTINGS["paths"]["models"])
REPORTS = ensure_dir(resolve_path(SETTINGS["paths"]["reports"]))

LABELS = ["High", "Medium", "Low"]


def train_regression(train_df, test_df, pre: Preprocessor):
    log.info("=== Training regression models ===")
    out_dir = ensure_dir(MODELS / "regression")
    metrics_all = {}
    cv_results = {}
    rmse_bars = {}

    X_train = pre.transform(train_df)
    X_test = pre.transform(test_df)
    y_train = train_df["G3"].values
    y_test = test_df["G3"].values

    for name, model in build_regression_models().items():
        log.info(f"Training {name}")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = regression_metrics(y_test, y_pred)
        cv = cv_regression(model, X_train, y_train)
        metrics_all[name] = metrics
        cv_results[name] = cv
        rmse_bars[name] = metrics["rmse"]

        joblib.dump(model, out_dir / f"{name}.pkl")
        plot_residuals(
            y_test, y_pred, f"Residuals — {name}", f"residuals_{name}.png"
        )

    plot_metric_bars(
        rmse_bars, "Regression RMSE Comparison (test set)", "regression_rmse.png", "RMSE"
    )
    write_json(REPORTS / "regression_metrics.json", metrics_all)
    write_json(REPORTS / "regression_cv.json", cv_results)
    log.info(f"Regression results: {json.dumps(rmse_bars, indent=2)}")
    return metrics_all


def train_classification(train_df, test_df, pre: Preprocessor):
    log.info("=== Training classification models ===")
    out_dir = ensure_dir(MODELS / "classification")
    metrics_all = {}
    cv_results = {}
    f1_bars = {}

    X_train = pre.transform(train_df)
    X_test = pre.transform(test_df)
    y_train = train_df["risk_class"].values
    y_test = test_df["risk_class"].values

    for name, model in build_classification_models().items():
        log.info(f"Training {name}")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = None
        if hasattr(model, "predict_proba"):
            try:
                y_proba = model.predict_proba(X_test)
            except Exception:
                y_proba = None
        metrics = classification_metrics(y_test, y_pred, y_proba, labels=LABELS)
        cv = cv_classification(model, X_train, y_train)
        metrics_all[name] = metrics
        cv_results[name] = cv
        f1_bars[name] = metrics["f1_macro"]

        joblib.dump(model, out_dir / f"{name}.pkl")
        plot_confusion_matrix(
            metrics["confusion_matrix"],
            LABELS,
            f"Confusion Matrix — {name}",
            f"cm_{name}.png",
        )

    plot_metric_bars(
        f1_bars,
        "Classification Macro F1 Comparison (test set)",
        "classification_f1.png",
        "Macro F1",
    )
    write_json(REPORTS / "classification_metrics.json", metrics_all)
    write_json(REPORTS / "classification_cv.json", cv_results)
    log.info(f"Classification results: {json.dumps(f1_bars, indent=2)}")
    return metrics_all


def train_clustering(train_df, pre: Preprocessor):
    log.info("=== Training K-Means clustering ===")
    out_dir = ensure_dir(MODELS / "clustering")
    X = pre.transform(train_df)

    kmeans = build_kmeans(n_clusters=4)
    labels = kmeans.fit_predict(X)
    sil = float(silhouette_score(X, labels)) if len(set(labels)) > 1 else 0.0

    joblib.dump(kmeans, out_dir / "kmeans.pkl")

    # PCA for visualization
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X)
    joblib.dump(pca, out_dir / "pca.pkl")

    plot_cluster_scatter(
        X_2d,
        labels,
        f"Student Segments (silhouette={sil:.3f})",
        "cluster_scatter.png",
    )

    write_json(
        REPORTS / "clustering_metrics.json",
        {
            "silhouette_score": sil,
            "n_clusters": 4,
            "segment_names": SEGMENT_NAMES,
            "cluster_sizes": {
                str(i): int((labels == i).sum()) for i in range(4)
            },
        },
    )
    log.info(f"Clustering silhouette: {sil:.3f}")


def main():
    train_df = pd.read_parquet(PROCESSED / "static_train.parquet")
    test_df = pd.read_parquet(PROCESSED / "static_test.parquet")
    pre = Preprocessor.load()

    train_regression(train_df, test_df, pre)
    train_classification(train_df, test_df, pre)
    train_clustering(train_df, pre)


if __name__ == "__main__":
    main()
