"""Preprocessor: fit-once, transform-many.

Responsibilities:
  - split features into numeric / nominal / ordinal
  - impute missing values
  - encode categoricals (OneHot for nominal)
  - scale numerics (StandardScaler)
  - persist fitted pipeline for streaming inference

Invariant: streaming layer NEVER refits. It loads the persisted pipeline
and calls .transform() only.
"""
from pathlib import Path
from typing import List, Tuple
import joblib
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from src.config import SETTINGS, resolve_path
from src.utils.logging import get_logger
from src.utils.helpers import ensure_dir

log = get_logger(__name__)

ARTIFACTS_DIR = ensure_dir(resolve_path(SETTINGS["paths"]["artifacts"]))

# Features that should be excluded from the model input
EXCLUDE = {"student_id", "G1", "G2", "G3", "risk_class"}

# The UCI categorical columns
NOMINAL = [
    "school",
    "sex",
    "address",
    "famsize",
    "Pstatus",
    "schoolsup",
    "famsup",
    "paid",
    "activities",
    "internet",
    "romantic",
]


class Preprocessor:
    """Fit + transform with sklearn ColumnTransformer under the hood."""

    def __init__(self):
        self.pipeline: Pipeline | None = None
        self.numeric_cols: List[str] = []
        self.nominal_cols: List[str] = []
        self.feature_names_out_: List[str] = []

    def _split_columns(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        cols = [c for c in df.columns if c not in EXCLUDE]
        nominal = [c for c in cols if c in NOMINAL]
        numeric = [c for c in cols if c not in nominal]
        return numeric, nominal

    def fit(self, df: pd.DataFrame) -> "Preprocessor":
        self.numeric_cols, self.nominal_cols = self._split_columns(df)
        num_pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        cat_pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "onehot",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
            ]
        )
        transformer = ColumnTransformer(
            [
                ("num", num_pipe, self.numeric_cols),
                ("cat", cat_pipe, self.nominal_cols),
            ],
            remainder="drop",
        )
        self.pipeline = Pipeline([("ct", transformer)])
        self.pipeline.fit(df)
        self._compute_feature_names()
        log.info(
            "Preprocessor fit: numeric=%d, nominal=%d, output_features=%d",
            len(self.numeric_cols), len(self.nominal_cols), len(self.feature_names_out_),
        )
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError("Preprocessor not fitted.")
        return self.pipeline.transform(df)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)

    def _compute_feature_names(self):
        ct: ColumnTransformer = self.pipeline.named_steps["ct"]
        numeric = self.numeric_cols
        cat_pipe = ct.named_transformers_["cat"]
        onehot: OneHotEncoder = cat_pipe.named_steps["onehot"]
        try:
            cat_names = onehot.get_feature_names_out(self.nominal_cols).tolist()
        except AttributeError:
            cat_names = []
        self.feature_names_out_ = numeric + cat_names

    def save(self, path: Path | None = None) -> Path:
        path = Path(path) if path else ARTIFACTS_DIR / "preprocessor.pkl"
        ensure_dir(path.parent)
        joblib.dump(
            {
                "pipeline": self.pipeline,
                "numeric_cols": self.numeric_cols,
                "nominal_cols": self.nominal_cols,
                "feature_names_out": self.feature_names_out_,
            },
            path,
        )
        log.info(f"Preprocessor saved to {path}")
        return path

    @classmethod
    def load(cls, path: Path | None = None) -> "Preprocessor":
        path = Path(path) if path else ARTIFACTS_DIR / "preprocessor.pkl"
        data = joblib.load(path)
        inst = cls()
        inst.pipeline = data["pipeline"]
        inst.numeric_cols = data["numeric_cols"]
        inst.nominal_cols = data["nominal_cols"]
        inst.feature_names_out_ = data["feature_names_out"]
        return inst
