"""Data loading, cleaning, hourly aggregation, and temporal split."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from config import ProjectConfig


def load_raw_uci_data(raw_path: Path) -> pd.DataFrame:
    """
    Load the UCI household dataset with datetime index.

    Expected input columns:
    - Date
    - Time
    - Global_active_power
    """
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw data file not found: {raw_path}. "
            "Please place household_power_consumption.txt in data/raw/."
        )

    df = pd.read_csv(
        raw_path,
        sep=";",
        usecols=["Date", "Time", "Global_active_power"],
        na_values=["?"],
        low_memory=False,
    )
    dt = pd.to_datetime(df["Date"] + " " + df["Time"], dayfirst=True, errors="coerce")
    df = pd.DataFrame({"datetime": dt, "Global_active_power": df["Global_active_power"]})
    df = df.dropna(subset=["datetime"]).set_index("datetime").sort_index()
    return df


def clean_load_data(df: pd.DataFrame) -> pd.DataFrame:
    """Convert load to numeric and fill missing values with time interpolation."""
    clean_df = df.copy()
    clean_df["Global_active_power"] = pd.to_numeric(
        clean_df["Global_active_power"], errors="coerce"
    )
    clean_df = clean_df[clean_df["Global_active_power"] >= 0]
    clean_df.loc[clean_df["Global_active_power"] > 20, "Global_active_power"] = pd.NA

    clean_df["Global_active_power"] = (
        clean_df["Global_active_power"].interpolate(method="time").ffill().bfill()
    )
    clean_df = clean_df.dropna(subset=["Global_active_power"])
    return clean_df


def resample_to_hourly(df: pd.DataFrame, rule: str = "1H") -> pd.DataFrame:
    """
    Resample minute-level power (kW) to hourly average power.

    For this coursework setup, hourly average kW * 1h is approximated as hourly kWh.
    Final column name is standardized as load_kwh.
    """
    hourly = df.resample(rule).mean(numeric_only=True)
    hourly = hourly.rename(columns={"Global_active_power": "load_kwh"})
    hourly = hourly.dropna(subset=["load_kwh"])
    return hourly


def split_train_val_test(
    hourly_df: pd.DataFrame, train_ratio: float, val_ratio: float, test_ratio: float
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Time-ordered split into train/val/test."""
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-9:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    n = len(hourly_df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = hourly_df.iloc[:train_end].copy()
    val_df = hourly_df.iloc[train_end:val_end].copy()
    test_df = hourly_df.iloc[val_end:].copy()
    return train_df, val_df, test_df


def run_preprocessing(cfg: ProjectConfig) -> Dict[str, pd.DataFrame]:
    """Execute full preprocessing pipeline and save outputs."""
    cfg.ensure_directories()

    raw_df = load_raw_uci_data(cfg.raw_data_path)
    raw_count = len(raw_df)
    raw_missing = int(raw_df["Global_active_power"].isna().sum())

    clean_df = clean_load_data(raw_df)
    clean_count = len(clean_df)
    clean_missing = int(clean_df["Global_active_power"].isna().sum())

    hourly_df = resample_to_hourly(clean_df, cfg.resample_rule)
    train_df, val_df, test_df = split_train_val_test(
        hourly_df, cfg.train_ratio, cfg.val_ratio, cfg.test_ratio
    )

    # Save data splits
    hourly_df.to_csv(cfg.processed_dir / "full_hourly.csv", index=True, index_label="timestamp")
    train_df.to_csv(cfg.processed_dir / "train.csv", index=True, index_label="timestamp")
    val_df.to_csv(cfg.processed_dir / "val.csv", index=True, index_label="timestamp")
    test_df.to_csv(cfg.processed_dir / "test.csv", index=True, index_label="timestamp")

    logs = {
        "raw_samples": raw_count,
        "clean_samples": clean_count,
        "raw_missing_global_active_power": raw_missing,
        "clean_missing_global_active_power": clean_missing,
        "hourly_samples": len(hourly_df),
        "train_start": str(train_df.index.min()),
        "train_end": str(train_df.index.max()),
        "val_start": str(val_df.index.min()),
        "val_end": str(val_df.index.max()),
        "test_start": str(test_df.index.min()),
        "test_end": str(test_df.index.max()),
    }

    print("[Preprocessing] Raw samples:", raw_count)
    print("[Preprocessing] Clean samples:", clean_count)
    print("[Preprocessing] Missing (raw):", raw_missing)
    print("[Preprocessing] Missing (clean):", clean_missing)
    print("[Preprocessing] Train range:", logs["train_start"], "->", logs["train_end"])
    print("[Preprocessing] Val range:", logs["val_start"], "->", logs["val_end"])
    print("[Preprocessing] Test range:", logs["test_start"], "->", logs["test_end"])

    with open(cfg.metrics_dir / "preprocessing_log.json", "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2)

    return {
        "full": hourly_df,
        "train": train_df,
        "val": val_df,
        "test": test_df,
    }

