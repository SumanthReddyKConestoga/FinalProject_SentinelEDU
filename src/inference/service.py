"""InferenceService — thread-safe prediction wrapper around loaded models."""
import threading
from typing import Tuple

import numpy as np
import pandas as pd

from src.inference.registry import get_registry
from src.models.clustering import SEGMENT_NAMES
from src.utils.logging import get_logger

log = get_logger(__name__)

_LABELS = ["High", "Low", "Medium"]  # alphabetical order from sklearn LabelEncoder
_KERAS_LABELS = ["High", "Medium", "Low"]  # order used during Keras training

_lock = threading.Lock()


class InferenceService:
    """Wraps the registry models with a clean prediction API."""

    def __init__(self):
        self._reg = get_registry()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _to_array(self, feature_row: dict) -> np.ndarray:
        df = pd.DataFrame([feature_row])
        # Fill any columns the preprocessor expects but are absent (imputers handle NaN)
        pre = self._reg.preprocessor
        for col in pre.numeric_cols + pre.nominal_cols:
            if col not in df.columns:
                df[col] = np.nan
        return pre.transform(df)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict_regression(self, feature_row: dict) -> float:
        with _lock:
            X = self._to_array(feature_row)
            return float(self._reg.regression_model.predict(X)[0])

    def predict_classification(self, feature_row: dict) -> Tuple[str, float]:
        with _lock:
            X = self._to_array(feature_row)
            kind = self._reg.classification_kind
            if kind == "sklearn":
                model = self._reg.classification_model
                label = str(model.predict(X)[0])
                if hasattr(model, "predict_proba"):
                    confidence = float(model.predict_proba(X).max())
                else:
                    confidence = 0.5
            else:
                proba = self._reg.classification_model.predict(X, verbose=0)[0]
                idx = int(np.argmax(proba))
                label = _KERAS_LABELS[idx] if idx < len(_KERAS_LABELS) else "Low"
                confidence = float(proba[idx])
            return label, confidence

    def predict_cnn(self, sequence: np.ndarray) -> Tuple[str, float]:
        """sequence shape: (window, n_features)."""
        if self._reg.cnn_model is None:
            return "Low", 0.5
        with _lock:
            X = sequence[np.newaxis, :, :]  # (1, window, n_features)
            proba = self._reg.cnn_model.predict(X, verbose=0)[0]
            idx = int(np.argmax(proba))
            label = _KERAS_LABELS[idx] if idx < len(_KERAS_LABELS) else "Low"
            return label, float(proba[idx])

    def assign_segment(self, feature_row: dict) -> str:
        with _lock:
            X = self._to_array(feature_row)
            cluster_id = int(self._reg.clustering_model.predict(X)[0])
            return SEGMENT_NAMES.get(cluster_id, f"Segment {cluster_id}")


_service: InferenceService | None = None
_service_lock = threading.Lock()


def get_service() -> InferenceService:
    global _service
    if _service is None:
        with _service_lock:
            if _service is None:
                _service = InferenceService()
    return _service
