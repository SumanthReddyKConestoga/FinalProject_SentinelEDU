"""FeatureStore — per-student rolling windows in memory."""
import collections
import threading
from typing import Dict, List, Optional

import numpy as np

from src.config import SETTINGS
from src.utils.logging import get_logger

log = get_logger(__name__)

WINDOW = SETTINGS["data"]["sequence_window"]
SEQ_FEATURES = SETTINGS["data"]["sequence_features"]
# Maps event keys → internal feature names
_EVENT_KEY_MAP = {
    "attendance_pct": "weekly_attendance_pct",
    "quiz_score": "weekly_quiz_score",
    "submission_rate": "weekly_submission_rate",
    "lms_logins": "weekly_lms_logins",
    "late_count": "weekly_late_count",
}


class FeatureStore:
    """Maintains per-student sliding windows of weekly features."""

    def __init__(self):
        self._windows: Dict[str, collections.deque] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------

    def _ensure(self, student_id: str):
        if student_id not in self._windows:
            self._windows[student_id] = collections.deque(maxlen=WINDOW)

    def hydrate_from_db(self, session) -> None:
        """Pre-populate windows from weekly_records table at startup."""
        from src.db.models import WeeklyRecord
        records = (
            session.query(WeeklyRecord)
            .order_by(WeeklyRecord.student_id, WeeklyRecord.week)
            .all()
        )
        with self._lock:
            for r in records:
                sid = r.student_id
                self._ensure(sid)
                self._windows[sid].append({
                    "weekly_attendance_pct": r.attendance_pct,
                    "weekly_quiz_score": r.quiz_score,
                    "weekly_submission_rate": r.submission_rate,
                    "weekly_lms_logins": float(r.lms_logins),
                    "weekly_late_count": float(r.late_count),
                })
        log.info(f"FeatureStore hydrated: {len(self._windows)} students")

    def update(self, student_id: str, event: dict) -> None:
        """Push a new event observation into the student's rolling window."""
        entry = {}
        for event_k, feat_k in _EVENT_KEY_MAP.items():
            if event_k in event:
                entry[feat_k] = float(event[event_k])
            elif feat_k in event:
                entry[feat_k] = float(event[feat_k])
        if not entry:
            return
        with self._lock:
            self._ensure(student_id)
            self._windows[student_id].append(entry)

    def get_window(self, student_id: str) -> Optional[np.ndarray]:
        """Return (window, n_features) array, or None if insufficient data."""
        with self._lock:
            deque = self._windows.get(student_id)
            if deque is None or len(deque) < 2:
                return None
            rows = list(deque)
        # Build matrix in SEQ_FEATURES order
        matrix = np.array(
            [[row.get(f, 0.0) for f in SEQ_FEATURES] for row in rows],
            dtype=np.float32,
        )
        # Pad to WINDOW if shorter
        if len(matrix) < WINDOW:
            pad = np.zeros((WINDOW - len(matrix), len(SEQ_FEATURES)), dtype=np.float32)
            matrix = np.vstack([pad, matrix])
        return matrix[-WINDOW:]

    def compute_derived(self, student_id: str) -> dict:
        """Return derived features: means, latest values, slopes."""
        window = self.get_window(student_id)
        if window is None:
            return {}
        result = {}
        for i, feat in enumerate(SEQ_FEATURES):
            col = window[:, i]
            result[f"{feat}_mean"] = float(np.mean(col))
            result[feat] = float(col[-1])  # latest value as plain name
            if len(col) >= 3:
                slope = float(np.polyfit(range(len(col)), col, 1)[0])
                result[f"{feat.replace('weekly_', '')}_slope_3"] = slope
        # Convenience aliases expected by alert rules
        result["weekly_attendance_pct"] = result.get("weekly_attendance_pct", 0.0)
        result["weekly_quiz_score"] = result.get("weekly_quiz_score", 0.0)
        result["weekly_late_count"] = result.get("weekly_late_count", 0.0)
        result["quiz_score_slope_3"] = result.get("quiz_score_slope_3", 0.0)
        return result

    def count_consecutive_high_risk(self, student_id: str, predictions: List[str]) -> int:
        """Count consecutive trailing 'High' predictions."""
        count = 0
        for p in reversed(predictions):
            if p == "High":
                count += 1
            else:
                break
        return count

    @property
    def student_ids(self) -> List[str]:
        with self._lock:
            return list(self._windows.keys())


_store: Optional[FeatureStore] = None
_store_lock = threading.Lock()


def get_store() -> FeatureStore:
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = FeatureStore()
    return _store
