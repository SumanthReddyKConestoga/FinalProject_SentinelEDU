"""Model registry loader — loads all production models at startup."""
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib

from src.config import SETTINGS, resolve_path
from src.utils.helpers import read_json
from src.utils.logging import get_logger

log = get_logger(__name__)

MODELS_DIR = resolve_path(SETTINGS["paths"]["models"])
ARTIFACTS_DIR = resolve_path(SETTINGS["paths"]["artifacts"])
REGISTRY_PATH = resolve_path("config/model_registry.json")


def _load_keras(path: Path) -> Any:
    import tensorflow as tf
    return tf.keras.models.load_model(str(path))


class ModelRegistry:
    """Loads and holds production model artifacts."""

    def __init__(self):
        self.preprocessor = None
        self.regression_model = None
        self.classification_model = None
        self.classification_kind: str = "sklearn"  # "sklearn" | "keras"
        self.cnn_model = None
        self.clustering_model = None
        self._registry: dict = {}
        self._loaded = False

    def load(self) -> "ModelRegistry":
        if self._loaded:
            return self
        from src.preprocessing.preprocessor import Preprocessor

        self._registry = read_json(REGISTRY_PATH)
        prod = self._registry.get("production", {})

        log.info("Loading preprocessor …")
        self.preprocessor = Preprocessor.load()

        reg_name = prod.get("regression") or "ridge_regression"
        log.info("Loading regression model: %s", reg_name)
        self.regression_model = joblib.load(MODELS_DIR / "regression" / f"{reg_name}.pkl")

        clf_name = prod.get("classification") or "logistic_regression"
        log.info("Loading classification model: %s", clf_name)
        clf_pkl = MODELS_DIR / "classification" / f"{clf_name}.pkl"
        clf_keras = MODELS_DIR / "deep" / f"{clf_name}.keras"
        if clf_pkl.exists():
            self.classification_model = joblib.load(clf_pkl)
            self.classification_kind = "sklearn"
        elif clf_keras.exists():
            self.classification_model = _load_keras(clf_keras)
            self.classification_kind = "keras"
        else:
            log.warning("No model file found for %s; falling back to logistic_regression", clf_name)
            self.classification_model = joblib.load(
                MODELS_DIR / "classification" / "logistic_regression.pkl"
            )
            self.classification_kind = "sklearn"

        log.info("Loading CNN model …")
        cnn_keras = MODELS_DIR / "cnn" / "cnn1d.keras"
        if cnn_keras.exists():
            self.cnn_model = _load_keras(cnn_keras)
        else:
            log.warning("CNN model not found; CNN inference will be unavailable.")
            self.cnn_model = None

        log.info("Loading clustering model …")
        self.clustering_model = joblib.load(MODELS_DIR / "clustering" / "kmeans.pkl")

        self._loaded = True
        log.info("ModelRegistry ready.")
        return self

    @property
    def registry_info(self) -> dict:
        return self._registry


@lru_cache(maxsize=1)
def get_registry() -> ModelRegistry:
    return ModelRegistry().load()
