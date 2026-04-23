"""StreamConsumer — processes events, runs inference, writes DB rows."""
import queue
import threading
from datetime import datetime
from typing import Optional

from src.config import SETTINGS
from src.utils.logging import get_logger

log = get_logger(__name__)


class StreamConsumer(threading.Thread):
    """Pulls events from queue, runs inference pipeline, writes DB."""

    def __init__(self, event_queue: queue.Queue):
        super().__init__(daemon=True, name="StreamConsumer")
        self.q = event_queue
        self._stop_event = threading.Event()
        self.events_processed = 0
        self._recent_cnn_labels: dict[str, list] = {}  # student_id → list of recent CNN labels

    def stop(self) -> None:
        self._stop_event.set()

    @property
    def is_running(self) -> bool:
        return self.is_alive() and not self._stop_event.is_set()

    def run(self) -> None:
        from src.db.models import SessionLocal
        from src.streaming.feature_store import get_store
        from src.inference.service import get_service
        from src.alerts.engine import AlertEngine
        from src.recommendations.engine import RecommendationEngine

        log.info("Consumer started.")
        store = get_store()
        service = get_service()
        alert_engine = AlertEngine()
        rec_engine = RecommendationEngine()

        while not self._stop_event.is_set():
            try:
                event = self.q.get(timeout=1.0)
            except queue.Empty:
                continue

            try:
                self._process(event, store, service, alert_engine, rec_engine)
                self.events_processed += 1
            except Exception as exc:
                log.error(f"Consumer error on event {event}: {exc}", exc_info=True)
            finally:
                self.q.task_done()

        log.info(f"Consumer stopped. Events processed: {self.events_processed}")

    def _process(self, event: dict, store, service, alert_engine, rec_engine) -> None:
        from src.db.models import SessionLocal, Student, WeeklyRecord, Prediction
        from src.utils.helpers import now_iso

        student_id = event.get("student_id")
        if not student_id:
            return

        # 1. Update rolling window
        store.update(student_id, event)

        session = SessionLocal()
        try:
            student = session.query(Student).filter_by(id=student_id).first()
            if student is None:
                session.close()
                return

            # 2. Write weekly record
            week = int(event.get("week", 0))
            existing = (
                session.query(WeeklyRecord)
                .filter_by(student_id=student_id, week=week)
                .first()
            )
            if existing is None:
                session.add(WeeklyRecord(
                    student_id=student_id,
                    week=week,
                    attendance_pct=float(event.get("weekly_attendance_pct", 0)),
                    quiz_score=float(event.get("weekly_quiz_score", 0)),
                    submission_rate=float(event.get("weekly_submission_rate", 0)),
                    lms_logins=int(event.get("weekly_lms_logins", 0)),
                    late_count=int(event.get("weekly_late_count", 0)),
                ))

            # 3. Build static feature row for regression/classification
            static_feats = dict(student.static_features or {})
            static_feats.update({
                "weekly_attendance_pct": float(event.get("weekly_attendance_pct", 0)),
                "weekly_quiz_score": float(event.get("weekly_quiz_score", 0)),
                "weekly_submission_rate": float(event.get("weekly_submission_rate", 0)),
                "weekly_lms_logins": float(event.get("weekly_lms_logins", 0)),
                "weekly_late_count": float(event.get("weekly_late_count", 0)),
                "mean_attendance_pct": float(event.get("weekly_attendance_pct", 0)),
                "mean_quiz_score": float(event.get("weekly_quiz_score", 0)),
                "attendance_trend_slope": 0.0,
                "quiz_score_trend_slope": 0.0,
            })

            # 4. Regression prediction
            reg_pred = service.predict_regression(static_feats)

            # 5. Classification prediction
            clf_label, clf_conf = service.predict_classification(static_feats)

            # 6. CNN prediction from rolling window
            window = store.get_window(student_id)
            cnn_label, cnn_conf = "Low", 0.5
            if window is not None:
                cnn_label, cnn_conf = service.predict_cnn(window)

            # Track consecutive high-risk CNN labels
            history = self._recent_cnn_labels.setdefault(student_id, [])
            history.append(cnn_label)
            if len(history) > 10:
                history.pop(0)
            cnn_consecutive = 0
            for lbl in reversed(history):
                if lbl == "High":
                    cnn_consecutive += 1
                else:
                    break

            # 7. Write predictions
            session.add(Prediction(
                student_id=student_id,
                model_name="regression_stream",
                prediction_value=reg_pred,
                week=week,
                created_at=datetime.utcnow(),
            ))
            session.add(Prediction(
                student_id=student_id,
                model_name="classification_stream",
                prediction_value=clf_conf,
                predicted_class=clf_label,
                confidence=clf_conf,
                week=week,
                created_at=datetime.utcnow(),
            ))

            # 8. Update student current state
            student.current_risk_class = clf_label
            student.current_predicted_g3 = reg_pred

            # 9. Build feature map for alert engine
            derived = store.compute_derived(student_id)
            feature_map = {
                **derived,
                "regression_prediction": reg_pred,
                "classifier_label": clf_label,
                "classifier_confidence": clf_conf,
                "cnn_high_risk_consecutive": cnn_consecutive,
            }

            session.commit()

            # 10. Alert evaluation (uses its own session commit internally)
            new_alerts = alert_engine.evaluate(student_id, feature_map, session)

            # 11. Recommendations for each new alert
            for alert in new_alerts:
                rec_engine.generate(alert, student, session)

            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()


# Module-level singletons
_consumer: Optional[StreamConsumer] = None
_consumer_lock = threading.Lock()


def get_consumer() -> Optional[StreamConsumer]:
    return _consumer


def start_consumer(event_queue: queue.Queue) -> StreamConsumer:
    global _consumer
    with _consumer_lock:
        if _consumer is not None and _consumer.is_alive():
            log.info("Consumer already running.")
            return _consumer
        _consumer = StreamConsumer(event_queue)
        _consumer.start()
        return _consumer


def stop_consumer() -> None:
    global _consumer
    with _consumer_lock:
        if _consumer and _consumer.is_alive():
            _consumer.stop()
            _consumer.join(timeout=3)
            log.info("Consumer stopped.")


def consumer_status() -> dict:
    global _consumer
    return {
        "running": _consumer is not None and _consumer.is_alive(),
        "events_processed": _consumer.events_processed if _consumer else 0,
    }
