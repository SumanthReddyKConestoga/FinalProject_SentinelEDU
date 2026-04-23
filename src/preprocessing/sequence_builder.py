"""Sequence builder for the 1D CNN.

Takes the weekly-events long-form dataframe and builds per-student sliding
windows of shape (window_size, n_features).
"""
from typing import Tuple
import numpy as np
import pandas as pd

from src.config import SETTINGS
from src.utils.logging import get_logger

log = get_logger(__name__)

SEQ_FEATURES = SETTINGS["data"]["sequence_features"]
WINDOW = SETTINGS["data"]["sequence_window"]


def build_sequences(
    weekly_df: pd.DataFrame,
    static_df: pd.DataFrame,
    label_col: str = "risk_class",
    window: int = WINDOW,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (X, y, student_ids).

    X: shape (n_samples, window, n_features)
    y: shape (n_samples,)  (string labels)
    student_ids: shape (n_samples,)
    """
    label_map = static_df.set_index("student_id")[label_col].to_dict()

    X_list, y_list, sid_list = [], [], []

    for sid, grp in weekly_df.groupby("student_id"):
        grp = grp.sort_values("week")
        values = grp[SEQ_FEATURES].values
        if len(values) < window:
            continue
        # Sliding windows with stride 1
        for i in range(len(values) - window + 1):
            window_matrix = values[i : i + window]
            X_list.append(window_matrix)
            y_list.append(label_map.get(sid))
            sid_list.append(sid)

    X = np.asarray(X_list, dtype=np.float32)
    y = np.asarray(y_list)
    sids = np.asarray(sid_list)

    # Per-feature z-score normalization
    if X.size > 0:
        mean = X.reshape(-1, X.shape[-1]).mean(axis=0)
        std = X.reshape(-1, X.shape[-1]).std(axis=0) + 1e-8
        X = (X - mean) / std

    log.info(
        f"Built sequences: X={X.shape}, y={y.shape}, unique students={len(set(sids))}"
    )
    return X, y, sids
