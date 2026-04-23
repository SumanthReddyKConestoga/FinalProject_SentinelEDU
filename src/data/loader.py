"""Data loading module.

Loads UCI Student Performance dataset. Falls back to a bundled synthetic
generator if the UCI download fails (e.g. offline demo).
"""
from pathlib import Path
import pandas as pd
import numpy as np

from src.config import SETTINGS, resolve_path
from src.utils.logging import get_logger
from src.utils.helpers import ensure_dir

log = get_logger(__name__)

RAW_DIR = resolve_path(SETTINGS["paths"]["data_raw"])
PROCESSED_DIR = resolve_path(SETTINGS["paths"]["data_processed"])


class DataLoader:
    """Loads and validates the student performance dataset."""

    def __init__(self, raw_dir: Path = RAW_DIR):
        self.raw_dir = ensure_dir(raw_dir)

    def load_uci(self) -> pd.DataFrame:
        """Try the UCI repo; fall back to synthetic if unavailable."""
        try:
            from ucimlrepo import fetch_ucirepo  # type: ignore

            log.info("Fetching UCI Student Performance dataset (id=320)...")
            dataset = fetch_ucirepo(id=320)
            df = pd.concat(
                [dataset.data.features, dataset.data.targets], axis=1
            )
            df.to_csv(self.raw_dir / "student_performance.csv", index=False)
            log.info(f"Loaded UCI dataset: shape={df.shape}")
            return df
        except Exception as exc:  # network-less fallback
            log.warning(f"UCI fetch failed ({exc}); using synthetic fallback.")
            return self._synthetic_fallback()

    def _synthetic_fallback(self) -> pd.DataFrame:
        """Generate a UCI-schema-compatible synthetic dataset.

        Used only when the real UCI download fails. Provides ~500 rows with
        realistic correlations between features and G3.
        """
        rng = np.random.default_rng(SETTINGS["data"]["random_state"])
        n = 500

        df = pd.DataFrame(
            {
                "school": rng.choice(["GP", "MS"], n, p=[0.7, 0.3]),
                "sex": rng.choice(["F", "M"], n),
                "age": rng.integers(15, 22, n),
                "address": rng.choice(["U", "R"], n, p=[0.75, 0.25]),
                "famsize": rng.choice(["GT3", "LE3"], n, p=[0.7, 0.3]),
                "Pstatus": rng.choice(["T", "A"], n, p=[0.9, 0.1]),
                "Medu": rng.integers(0, 5, n),
                "Fedu": rng.integers(0, 5, n),
                "studytime": rng.integers(1, 5, n),
                "failures": rng.choice([0, 1, 2, 3], n, p=[0.77, 0.13, 0.06, 0.04]),
                "schoolsup": rng.choice(["yes", "no"], n, p=[0.12, 0.88]),
                "famsup": rng.choice(["yes", "no"], n, p=[0.6, 0.4]),
                "paid": rng.choice(["yes", "no"], n, p=[0.45, 0.55]),
                "activities": rng.choice(["yes", "no"], n, p=[0.5, 0.5]),
                "internet": rng.choice(["yes", "no"], n, p=[0.83, 0.17]),
                "romantic": rng.choice(["yes", "no"], n, p=[0.33, 0.67]),
                "freetime": rng.integers(1, 6, n),
                "goout": rng.integers(1, 6, n),
                "Dalc": rng.integers(1, 6, n),
                "Walc": rng.integers(1, 6, n),
                "health": rng.integers(1, 6, n),
                "absences": rng.integers(0, 30, n),
            }
        )
        # Build G1, G2, G3 with realistic correlations
        base = (
            10
            + 1.2 * df["studytime"]
            - 2.0 * df["failures"]
            - 0.08 * df["absences"]
            + 0.5 * df["Medu"]
            + rng.normal(0, 2.0, n)
        )
        df["G1"] = np.clip(base + rng.normal(0, 1.5, n), 0, 20).round().astype(int)
        df["G2"] = np.clip(
            0.3 * df["G1"] + 0.7 * base + rng.normal(0, 1.2, n), 0, 20
        ).round().astype(int)
        df["G3"] = np.clip(
            0.2 * df["G1"] + 0.3 * df["G2"] + 0.5 * base + rng.normal(0, 1.0, n),
            0,
            20,
        ).round().astype(int)

        df.to_csv(self.raw_dir / "student_performance.csv", index=False)
        log.info(f"Synthetic dataset created: shape={df.shape}")
        return df

    def load(self) -> pd.DataFrame:
        """Load data, preferring cached CSV if already downloaded."""
        cached = self.raw_dir / "student_performance.csv"
        if cached.exists():
            log.info(f"Loading cached dataset from {cached}")
            return pd.read_csv(cached)
        return self.load_uci()


if __name__ == "__main__":
    loader = DataLoader()
    df = loader.load()
    print(df.head())
    print(df.shape)
