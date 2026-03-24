"""Plot helpers for forecast and dispatch visualizations."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import pandas as pd


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def plot_forecast_window(
    forecast_map: Dict[str, pd.DataFrame],
    output_path: Path,
    max_points: int = 24 * 7,
) -> None:
    """Plot actual vs predicted load for multiple forecasting models."""
    _ensure_parent(output_path)
    plt.figure(figsize=(12, 5))

    first = next(iter(forecast_map.values()))
    first = first.sort_index().iloc[:max_points]
    plt.plot(first.index, first["actual_load"], label="Actual", linewidth=2)

    for name, df in forecast_map.items():
        sub = df.sort_index().iloc[:max_points]
        plt.plot(sub.index, sub["pred_load"], label=name, alpha=0.85)

    plt.title("Forecast Comparison on Test Window")
    plt.xlabel("Time")
    plt.ylabel("Load (kWh)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_dispatch_grid_import(
    no_batt_df: pd.DataFrame,
    rule_df: pd.DataFrame,
    output_path: Path,
    max_points: int = 24 * 7,
) -> None:
    """Plot grid import timeseries for no-battery and rule-based baselines."""
    _ensure_parent(output_path)
    nb = no_batt_df.copy()
    rb = rule_df.copy()
    nb["timestamp"] = pd.to_datetime(nb["timestamp"])
    rb["timestamp"] = pd.to_datetime(rb["timestamp"])
    nb = nb.iloc[:max_points]
    rb = rb.iloc[:max_points]

    plt.figure(figsize=(12, 5))
    plt.plot(nb["timestamp"], nb["grid_import"], label="No battery")
    plt.plot(rb["timestamp"], rb["grid_import"], label="Rule-based")
    plt.title("Grid Import Comparison")
    plt.xlabel("Time")
    plt.ylabel("Grid Import (kWh)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_rule_soc(rule_df: pd.DataFrame, output_path: Path, max_points: int = 24 * 7) -> None:
    """Plot SOC trajectory for rule-based policy."""
    _ensure_parent(output_path)
    df = rule_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.iloc[:max_points]

    plt.figure(figsize=(12, 4))
    plt.plot(df["timestamp"], df["soc"], label="SOC", color="tab:green")
    plt.title("Rule-based Battery SOC")
    plt.xlabel("Time")
    plt.ylabel("SOC")
    plt.ylim(0, 1.05)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_cost_peak_bars(metrics_df: pd.DataFrame, cost_out: Path, peak_out: Path) -> None:
    """Plot cost and peak bar charts for dispatch strategy comparison."""
    _ensure_parent(cost_out)
    _ensure_parent(peak_out)
    df = metrics_df.copy()

    plt.figure(figsize=(7, 4))
    plt.bar(df["strategy"], df["total_cost"], color=["tab:blue", "tab:orange"])
    plt.title("Total Cost Comparison")
    plt.ylabel("Cost (GBP)")
    plt.tight_layout()
    plt.savefig(cost_out, dpi=150)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.bar(df["strategy"], df["peak_grid_import"], color=["tab:blue", "tab:orange"])
    plt.title("Peak Grid Import Comparison")
    plt.ylabel("Peak Grid Import (kWh)")
    plt.tight_layout()
    plt.savefig(peak_out, dpi=150)
    plt.close()


def plot_week_window(
    no_batt_df: pd.DataFrame,
    rule_df: pd.DataFrame,
    output_path: Path,
    start_time: Optional[str] = None,
) -> None:
    """Plot one-week local comparison window."""
    _ensure_parent(output_path)
    nb = no_batt_df.copy()
    rb = rule_df.copy()
    nb["timestamp"] = pd.to_datetime(nb["timestamp"])
    rb["timestamp"] = pd.to_datetime(rb["timestamp"])
    nb = nb.sort_values("timestamp")
    rb = rb.sort_values("timestamp")

    if start_time is None:
        start = nb["timestamp"].iloc[0]
    else:
        start = pd.to_datetime(start_time)
    end = start + pd.Timedelta(days=7)

    nb_w = nb[(nb["timestamp"] >= start) & (nb["timestamp"] < end)]
    rb_w = rb[(rb["timestamp"] >= start) & (rb["timestamp"] < end)]

    plt.figure(figsize=(12, 5))
    plt.plot(nb_w["timestamp"], nb_w["grid_import"], label="No battery")
    plt.plot(rb_w["timestamp"], rb_w["grid_import"], label="Rule-based")
    plt.title("Weekly Grid Import Window")
    plt.xlabel("Time")
    plt.ylabel("Grid Import (kWh)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
