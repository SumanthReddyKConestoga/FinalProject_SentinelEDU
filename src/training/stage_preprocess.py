"""DVC stage 2: augment + preprocess + split.

Outputs:
  data/processed/static_train.parquet
  data/processed/static_val.parquet
  data/processed/static_test.parquet
  data/processed/weekly.parquet
  data/streaming/events.jsonl   (hold-out streaming events)
  artifacts/preprocessor.pkl
"""
from pathlib import Path
import json
import pandas as pd

from src.config import SETTINGS, resolve_path
from src.data.loader import DataLoader
from src.data.augmentor import Augmentor
from src.preprocessing.feature_engineer import derive_risk_class, split
from src.preprocessing.preprocessor import Preprocessor
from src.utils.helpers import ensure_dir
from src.utils.logging import get_logger

log = get_logger(__name__)

PROCESSED = ensure_dir(resolve_path(SETTINGS["paths"]["data_processed"]))
STREAMING = ensure_dir(resolve_path(SETTINGS["paths"]["data_streaming"]))


def main():
    # 1. Load
    df = DataLoader().load()

    # 2. Augment (synthesize behavioral features + weekly events)
    aug = Augmentor(n_weeks=12)
    static_df, weekly_df = aug.synthesize(df)

    # 3. Derive target class
    static_df = derive_risk_class(static_df)

    # 4. Hold out ~20% of students for streaming simulation
    rng_sample = static_df.sample(
        frac=0.2, random_state=SETTINGS["data"]["random_state"]
    )
    stream_ids = set(rng_sample["student_id"])
    train_pool = static_df[~static_df["student_id"].isin(stream_ids)].copy()

    # 5. Train/Val/Test split on the remaining
    train, val, test = split(train_pool)

    # 6. Save parquet
    train.to_parquet(PROCESSED / "static_train.parquet", index=False)
    val.to_parquet(PROCESSED / "static_val.parquet", index=False)
    test.to_parquet(PROCESSED / "static_test.parquet", index=False)
    static_df.to_parquet(PROCESSED / "static_all.parquet", index=False)
    weekly_df.to_parquet(PROCESSED / "weekly.parquet", index=False)

    # 7. Fit preprocessor on train only
    pre = Preprocessor()
    pre.fit(train)
    pre.save()

    # 8. Persist streaming events (long-form) as JSONL
    stream_weekly = weekly_df[weekly_df["student_id"].isin(stream_ids)].sort_values(
        ["week", "student_id"]
    )
    events_path = STREAMING / "events.jsonl"
    with open(events_path, "w") as f:
        for _, row in stream_weekly.iterrows():
            event = {
                "student_id": row["student_id"],
                "week": int(row["week"]),
                "weekly_attendance_pct": float(row["weekly_attendance_pct"]),
                "weekly_quiz_score": float(row["weekly_quiz_score"]),
                "weekly_submission_rate": float(row["weekly_submission_rate"]),
                "weekly_lms_logins": int(row["weekly_lms_logins"]),
                "weekly_late_count": int(row["weekly_late_count"]),
            }
            f.write(json.dumps(event) + "\n")

    # 9. Save static records for streaming students (for dashboard lookups)
    stream_static = static_df[static_df["student_id"].isin(stream_ids)]
    stream_static.to_parquet(PROCESSED / "static_streaming.parquet", index=False)

    log.info(f"Preprocessing done. Streaming events: {events_path}")


if __name__ == "__main__":
    main()
