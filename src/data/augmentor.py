"""Augmentor: synthesizes LMS-style behavioral features.

The UCI dataset lacks streaming-friendly behavioral data, so we synthesize
weekly attendance percentages, quiz scores per week, submission counts,
and LMS login counts — all correlated with G3 + noise.

This is the honest truth of the project and we disclose it in the report.
"""
from typing import Tuple
import numpy as np
import pandas as pd

from src.config import SETTINGS
from src.utils.logging import get_logger

log = get_logger(__name__)


class Augmentor:
    """Adds synthesized behavioral and temporal features."""

    def __init__(
        self,
        n_weeks: int = 12,
        random_state: int = SETTINGS["data"]["random_state"],
    ):
        self.n_weeks = n_weeks
        self.rng = np.random.default_rng(random_state)

    def synthesize(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Return (static_features_df, weekly_events_df).

        static_features_df: one row per student with aggregate behavioral
            features + original UCI columns.
        weekly_events_df: long-form dataframe with 12 weeks per student,
            used for streaming and CNN sequences.
        """
        df = df.reset_index(drop=True).copy()
        df["student_id"] = df.index.map(lambda i: f"S{i:04d}")

        # Per-student bias — a latent "conscientiousness" that drives behaviors
        target = df["G3"].astype(float)
        norm_target = (target - target.min()) / (target.max() - target.min() + 1e-9)

        weekly_rows = []
        agg_rows = []

        for idx, row in df.iterrows():
            sid = row["student_id"]
            bias = norm_target.iloc[idx]
            # Trend: some students decline over weeks, some improve
            trend_direction = self.rng.choice(
                [-1, 0, 1], p=[0.25, 0.5, 0.25]
            )
            trend_strength = self.rng.uniform(0.0, 1.5)

            attendance_series = []
            quiz_series = []
            submission_series = []
            logins_series = []
            late_series = []

            for week in range(1, self.n_weeks + 1):
                week_factor = (week - 1) / max(self.n_weeks - 1, 1)
                drift = trend_direction * trend_strength * week_factor

                attendance = np.clip(
                    70 + 25 * bias - 8 * drift + self.rng.normal(0, 5), 20, 100
                )
                quiz = np.clip(
                    8 + 10 * bias - 2.0 * drift + self.rng.normal(0, 1.5), 0, 20
                )
                submissions = np.clip(
                    0.6 + 0.35 * bias - 0.1 * drift + self.rng.normal(0, 0.1),
                    0,
                    1,
                )
                logins = int(
                    np.clip(
                        2 + 10 * bias - 1.5 * drift + self.rng.normal(0, 1.5),
                        0,
                        20,
                    )
                )
                lates = int(
                    np.clip(
                        max(0, 2 - 4 * bias + 1.5 * drift + self.rng.normal(0, 0.6)),
                        0,
                        5,
                    )
                )

                attendance_series.append(attendance)
                quiz_series.append(quiz)
                submission_series.append(submissions)
                logins_series.append(logins)
                late_series.append(lates)

                weekly_rows.append(
                    {
                        "student_id": sid,
                        "week": week,
                        "weekly_attendance_pct": round(attendance, 2),
                        "weekly_quiz_score": round(quiz, 2),
                        "weekly_submission_rate": round(submissions, 3),
                        "weekly_lms_logins": logins,
                        "weekly_late_count": lates,
                    }
                )

            agg_rows.append(
                {
                    "student_id": sid,
                    "mean_attendance_pct": float(np.mean(attendance_series)),
                    "mean_quiz_score": float(np.mean(quiz_series)),
                    "mean_submission_rate": float(np.mean(submission_series)),
                    "mean_lms_logins": float(np.mean(logins_series)),
                    "mean_late_count": float(np.mean(late_series)),
                    "attendance_trend_slope": self._slope(attendance_series),
                    "quiz_score_trend_slope": self._slope(quiz_series),
                    "quiz_score_volatility": float(np.std(quiz_series)),
                    "engagement_decay": self._decay(logins_series),
                    "lateness_ratio": float(
                        np.sum(late_series) / (np.sum(late_series) + 1e-5)
                    )
                    if np.sum(late_series) > 0
                    else 0.0,
                }
            )

        agg_df = pd.DataFrame(agg_rows)
        weekly_df = pd.DataFrame(weekly_rows)
        enriched = df.merge(agg_df, on="student_id", how="left")

        log.info(
            f"Augmented dataset: static={enriched.shape}, weekly={weekly_df.shape}"
        )
        return enriched, weekly_df

    @staticmethod
    def _slope(values) -> float:
        x = np.arange(len(values))
        y = np.asarray(values)
        if len(x) < 2:
            return 0.0
        slope = np.polyfit(x, y, 1)[0]
        return float(slope)

    @staticmethod
    def _decay(values) -> float:
        """Difference between first-half and second-half averages."""
        arr = np.asarray(values)
        if len(arr) < 4:
            return 0.0
        half = len(arr) // 2
        return float(np.mean(arr[:half]) - np.mean(arr[half:]))
