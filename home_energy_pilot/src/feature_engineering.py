"""Feature engineering and supervised sample construction."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


TIME_FEATURE_COLUMNS = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend"]


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create cyclical hour/day features plus weekend flag."""
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        raise TypeError("Input dataframe must have DatetimeIndex.")

    hour = out.index.hour
    dow = out.index.dayofweek
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    out["is_weekend"] = (dow >= 5).astype(int)
    return out


def build_and_save_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    processed_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate and save train/val/test feature tables."""
    train_feat = add_time_features(train_df)
    val_feat = add_time_features(val_df)
    test_feat = add_time_features(test_df)

    cols = ["load_kwh"] + TIME_FEATURE_COLUMNS
    train_feat = train_feat[cols]
    val_feat = val_feat[cols]
    test_feat = test_feat[cols]

    train_feat.to_csv(processed_dir / "train_features.csv", index=True, index_label="timestamp")
    val_feat.to_csv(processed_dir / "val_features.csv", index=True, index_label="timestamp")
    test_feat.to_csv(processed_dir / "test_features.csv", index=True, index_label="timestamp")
    return train_feat, val_feat, test_feat


def create_supervised_samples(
    df: pd.DataFrame,
    window: int = 24,
    horizon: int = 1,
    mode: str = "load_only",
) -> Tuple[np.ndarray, np.ndarray, List[pd.Timestamp]]:
    """
    Convert sequence data to supervised samples.

    Parameters:
    - mode='load_only': input uses only `load_kwh`
    - mode='with_time': input uses load + time features
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if window < 1:
        raise ValueError("window must be >= 1")

    if mode == "load_only":
        feature_cols = ["load_kwh"]
    elif mode == "with_time":
        feature_cols = ["load_kwh"] + TIME_FEATURE_COLUMNS
    else:
        raise ValueError("mode must be one of {'load_only', 'with_time'}")

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    data_x = df[feature_cols].values
    data_y = df["load_kwh"].values
    index = df.index

    X, y, ts = [], [], []
    max_i = len(df) - horizon + 1
    for i in range(window, max_i):
        X.append(data_x[i - window : i])
        y.append(data_y[i + horizon - 1])
        ts.append(index[i + horizon - 1])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32).reshape(-1, 1), ts

