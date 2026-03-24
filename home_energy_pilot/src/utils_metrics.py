"""Metric utilities for forecasting and battery dispatch evaluation."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd


def mae(y_true, y_pred) -> float:
    """Mean absolute error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred) -> float:
    """Root mean squared error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true, y_pred, eps: float = 1e-6) -> float:
    """Mean absolute percentage error in percent."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.where(np.abs(y_true) < eps, eps, np.abs(y_true))
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)


def forecast_metrics(y_true, y_pred) -> Dict[str, float]:
    """Return MAE/RMSE/MAPE dict."""
    return {"mae": mae(y_true, y_pred), "rmse": rmse(y_true, y_pred), "mape": mape(y_true, y_pred)}


def dispatch_metrics(
    traj_df: pd.DataFrame,
    baseline_peak: Optional[float] = None,
) -> Dict[str, float]:
    """
    Compute dispatch KPIs from trajectory dataframe.

    Required columns:
    - grid_import
    - price
    Optional:
    - charge_power
    - discharge_power
    - step_cost
    """
    required = ["grid_import", "price"]
    missing = [c for c in required if c not in traj_df.columns]
    if missing:
        raise ValueError(f"Trajectory missing required columns: {missing}")

    df = traj_df.copy()
    if "step_cost" not in df.columns:
        df["step_cost"] = df["grid_import"] * df["price"]
    if "charge_power" not in df.columns:
        df["charge_power"] = 0.0
    if "discharge_power" not in df.columns:
        df["discharge_power"] = 0.0

    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"])
    else:
        ts = pd.to_datetime(df.index)
    peak_mask = (ts.dt.hour >= 16) & (ts.dt.hour < 21)

    total_cost = float(df["step_cost"].sum())
    peak_grid_import = float(df["grid_import"].max())
    avg_peak_hours_grid_import = float(df.loc[peak_mask, "grid_import"].mean())
    throughput = float(df["charge_power"].sum() + df["discharge_power"].sum())
    load_factor = float(df["grid_import"].mean() / max(df["grid_import"].max(), 1e-8))

    metrics = {
        "total_cost": total_cost,
        "peak_grid_import": peak_grid_import,
        "avg_peak_hours_grid_import": avg_peak_hours_grid_import,
        "battery_throughput": throughput,
        "load_factor": load_factor,
    }
    if baseline_peak is not None and baseline_peak > 0:
        metrics["peak_reduction_ratio"] = float((baseline_peak - peak_grid_import) / baseline_peak)
    return metrics


def metrics_dict_to_df(metrics: Dict[str, float], strategy_name: str) -> pd.DataFrame:
    """Convert metric dict into a one-row dataframe with strategy label."""
    row = {"strategy": strategy_name}
    row.update(metrics)
    return pd.DataFrame([row])

