"""Seed the SQLite DB from processed parquet files + trained models.

Run after DVC pipeline finishes. Populates:
  - students (static features + assigned segment)
  - weekly_records (historical weeks so dashboard has data immediately)
  - predictions (one baseline regression + one classification per student)
"""
import json
import joblib
import pandas as pd
import numpy as np

from src.config import SETTINGS, resolve_path
from src.db.models import (
    SessionLocal,
    Student,
    WeeklyRecord,
    Prediction,
    init_db,
)
from src.preprocessing.preprocessor import Preprocessor
from src.models.clustering import SEGMENT_NAMES
from src.utils.helpers import read_json, now_iso
from src.utils.logging import get_logger

log = get_logger(__name__)

PROCESSED = resolve_path(SETTINGS["paths"]["data_processed"])
MODELS = resolve_path(SETTINGS["paths"]["models"])
REGISTRY_PATH = resolve_path("config/model_registry.json")


def main():
    init_db()
    session = SessionLocal()

    # Clear existing rows for a clean seed
    for tbl in [Prediction, WeeklyRecord, Student]:
        session.query(tbl).delete()
    session.commit()

    static_all = pd.read_parquet(PROCESSED / "static_all.parquet")
    weekly = pd.read_parquet(PROCESSED / "weekly.parquet")

    registry = read_json(REGISTRY_PATH)
    reg_name = registry["production"].get("regression") or "linear_regression"
    clf_name = registry["production"].get("classification") or "logistic_regression"

    # Load models
    pre = Preprocessor.load()
    reg_model = joblib.load(MODELS / "regression" / f"{reg_name}.pkl")
    # classifier may be classical OR deep (.keras)
    clf_path_pkl = MODELS / "classification" / f"{clf_name}.pkl"
    clf_path_keras = MODELS / "deep" / f"{clf_name}.keras"
    clf_path_cnn = MODELS / "cnn" / "cnn1d.keras"
    if clf_path_pkl.exists():
        clf_model = joblib.load(clf_path_pkl)
        clf_kind = "sklearn"
    elif clf_path_keras.exists():
        import tensorflow as tf
        clf_model = tf.keras.models.load_model(clf_path_keras)
        clf_kind = "keras"
    else:
        clf_model = joblib.load(MODELS / "classification" / "logistic_regression.pkl")
        clf_kind = "sklearn"

    kmeans = joblib.load(MODELS / "clustering" / "kmeans.pkl")

    # Compute features + predictions for all students
    X = pre.transform(static_all)
    reg_preds = reg_model.predict(X)
    segments = kmeans.predict(X)

    if clf_kind == "sklearn":
        clf_preds = clf_model.predict(X)
        if hasattr(clf_model, "predict_proba"):
            clf_proba = clf_model.predict_proba(X).max(axis=1)
        else:
            clf_proba = np.ones(len(clf_preds)) * 0.5
    else:
        proba_mat = clf_model.predict(X, verbose=0)
        idx = np.argmax(proba_mat, axis=1)
        labels = ["High", "Medium", "Low"]
        clf_preds = np.array([labels[i] for i in idx])
        clf_proba = proba_mat.max(axis=1)

    # Insert students
    for i, row in static_all.iterrows():
        stu = Student(
            id=row["student_id"],
            age=int(row.get("age", 17)),
            sex=str(row.get("sex", "F")),
            school=str(row.get("school", "GP")),
            program="General",
            segment=SEGMENT_NAMES.get(int(segments[i]), f"Segment {int(segments[i])}"),
            current_risk_class=str(clf_preds[i]),
            current_predicted_g3=float(reg_preds[i]),
            static_features={
                "Medu": int(row.get("Medu", 0)),
                "Fedu": int(row.get("Fedu", 0)),
                "studytime": int(row.get("studytime", 2)),
                "failures": int(row.get("failures", 0)),
                "absences": int(row.get("absences", 0)),
                "G1": int(row.get("G1", 0)),
                "G2": int(row.get("G2", 0)),
                "G3": int(row.get("G3", 0)),
                "mean_attendance_pct": float(row.get("mean_attendance_pct", 0.0)),
                "mean_quiz_score": float(row.get("mean_quiz_score", 0.0)),
                "attendance_trend_slope": float(row.get("attendance_trend_slope", 0.0)),
                "quiz_score_trend_slope": float(row.get("quiz_score_trend_slope", 0.0)),
            },
        )
        session.add(stu)
        session.add(
            Prediction(
                student_id=row["student_id"],
                model_name=reg_name,
                prediction_value=float(reg_preds[i]),
                predicted_class=None,
                confidence=None,
            )
        )
        session.add(
            Prediction(
                student_id=row["student_id"],
                model_name=clf_name,
                prediction_value=float(clf_proba[i]),
                predicted_class=str(clf_preds[i]),
                confidence=float(clf_proba[i]),
            )
        )

    # Insert weekly records
    for _, row in weekly.iterrows():
        session.add(
            WeeklyRecord(
                student_id=row["student_id"],
                week=int(row["week"]),
                attendance_pct=float(row["weekly_attendance_pct"]),
                quiz_score=float(row["weekly_quiz_score"]),
                submission_rate=float(row["weekly_submission_rate"]),
                lms_logins=int(row["weekly_lms_logins"]),
                late_count=int(row["weekly_late_count"]),
            )
        )

    session.commit()
    log.info(
        f"Seeded DB: students={session.query(Student).count()}, "
        f"weekly_records={session.query(WeeklyRecord).count()}, "
        f"predictions={session.query(Prediction).count()}"
    )
    session.close()


if __name__ == "__main__":
    main()
