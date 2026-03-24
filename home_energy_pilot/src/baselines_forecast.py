"""Classical baseline forecasting models (persistence and moving average)."""

from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd

from config import ProjectConfig
from utils_metrics import forecast_metrics


def persistence_forecast(full_df: pd.DataFrame, target_index: pd.DatetimeIndex) -> pd.Series:
    """L_hat(t+1)=L(t), aligned to target timestamps."""
    pred = full_df["load_kwh"].shift(1)
    return pred.reindex(target_index)


def moving_average_forecast(
    full_df: pd.DataFrame, target_index: pd.DatetimeIndex, window: int = 24
) -> pd.Series:
    """Predict by rolling average of the previous `window` hours."""
    pred = full_df["load_kwh"].rolling(window=window, min_periods=window).mean().shift(1)
    return pred.reindex(target_index)


def _assemble_pred_df(actual: pd.Series, pred: pd.Series) -> pd.DataFrame:
    out = pd.DataFrame({"actual_load": actual, "pred_load": pred}).dropna()
    out.index.name = "timestamp"
    return out


def run_baseline_forecasts(
    cfg: ProjectConfig,
    full_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Run baselines, save predictions + metrics, and return them."""
    cfg.ensure_directories()

    actual = test_df["load_kwh"]
    pred_persistence = persistence_forecast(full_df, actual.index)
    pred_ma = moving_average_forecast(full_df, actual.index, cfg.moving_average_window)

    persistence_df = _assemble_pred_df(actual, pred_persistence)
    moving_average_df = _assemble_pred_df(actual, pred_ma)

    persistence_df.to_csv(cfg.predictions_dir / "persistence_test_pred.csv")
    moving_average_df.to_csv(cfg.predictions_dir / "moving_average_test_pred.csv")

    p_metrics = forecast_metrics(persistence_df["actual_load"], persistence_df["pred_load"])
    ma_metrics = forecast_metrics(moving_average_df["actual_load"], moving_average_df["pred_load"])

    metrics_df = pd.DataFrame(
        [
            {"model": "Persistence", **p_metrics},
            {"model": f"MovingAverage_{cfg.moving_average_window}", **ma_metrics},
        ]
    )
    metrics_df.to_csv(cfg.metrics_dir / "forecast_baselines_metrics.csv", index=False)
    return metrics_df, {"Persistence": persistence_df, "MovingAverage": moving_average_df}

