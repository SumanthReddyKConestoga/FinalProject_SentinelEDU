"""Feature engineer.

Derives:
  - risk_class (3-level) from G3
  - train/val/test splits stratified on risk_class
"""
from typing import Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from src.config import SETTINGS
from src.utils.logging import get_logger

log = get_logger(__name__)


def derive_risk_class(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    high_max = SETTINGS["targets"]["risk_bins"]["high_max"]
    medium_max = SETTINGS["targets"]["risk_bins"]["medium_max"]
    bins = [-np.inf, high_max, medium_max, np.inf]
    labels = ["High", "Medium", "Low"]
    df["risk_class"] = pd.cut(
        df["G3"], bins=bins, labels=labels, right=False
    ).astype(str)
    return df


def split(
    df: pd.DataFrame, target_col: str = "risk_class"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    test_size = SETTINGS["data"]["test_size"]
    val_size = SETTINGS["data"]["val_size"]
    rs = SETTINGS["data"]["random_state"]

    train_val, test = train_test_split(
        df,
        test_size=test_size,
        stratify=df[target_col],
        random_state=rs,
    )
    # adjust val size relative to remaining
    relative_val = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=relative_val,
        stratify=train_val[target_col],
        random_state=rs,
    )
    log.info(
        f"Split: train={len(train)}, val={len(val)}, test={len(test)}"
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)
