"""Cross-validation utilities."""
from typing import Dict, Any
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold

from src.config import SETTINGS


def cv_classification(model, X, y) -> Dict[str, Any]:
    folds = SETTINGS["training"]["cv_folds"]
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=skf, scoring="f1_macro", n_jobs=-1)
    return {
        "cv_mean_f1_macro": float(scores.mean()),
        "cv_std_f1_macro": float(scores.std()),
        "cv_scores": [float(s) for s in scores],
    }


def cv_regression(model, X, y) -> Dict[str, Any]:
    folds = SETTINGS["training"]["cv_folds"]
    kf = KFold(n_splits=folds, shuffle=True, random_state=42)
    neg_mse = cross_val_score(
        model, X, y, cv=kf, scoring="neg_mean_squared_error", n_jobs=-1
    )
    rmse_scores = np.sqrt(-neg_mse)
    return {
        "cv_mean_rmse": float(rmse_scores.mean()),
        "cv_std_rmse": float(rmse_scores.std()),
        "cv_scores": [float(s) for s in rmse_scores],
    }
