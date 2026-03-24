"""Plotting utilities for DQN experiments (standalone; does not modify core utils_plot)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# English-only figure text; avoid locale-dependent CJK fonts on some systems
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica", "Liberation Sans", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _format_datetime_xaxis(ax) -> None:
    """Use numeric date labels so tick text stays English on any locale."""
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
    for label in ax.get_xticklabels():
        label.set_ha("right")
    ax.figure.autofmt_xdate()


def plot_reward_convergence(
    series: Dict[str, Sequence[float]],
    output_path: Path,
    window: int = 5,
) -> None:
    """Plot raw and moving-average episode returns for one or more runs."""
    _ensure_parent(output_path)
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, returns in series.items():
        y = np.asarray(returns, dtype=float)
        ax.plot(y, alpha=0.35, label=f"{name} (raw)")
        if len(y) >= window:
            kernel = np.ones(window) / window
            ma = np.convolve(y, kernel, mode="valid")
            ax.plot(range(window - 1, len(y)), ma, linewidth=2, label=f"{name} MA({window})")
    ax.set_title("DQN Training: Episode Return Convergence")
    ax.set_xlabel("Episode index")
    ax.set_ylabel("Episode return (sum of step rewards)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_soc_curves(
    traj_map: Dict[str, pd.DataFrame],
    output_path: Path,
    max_points: int = 24 * 14,
) -> None:
    """Plot SOC trajectories for multiple strategies."""
    _ensure_parent(output_path)
    fig, ax = plt.subplots(figsize=(12, 5))
    for name, df in traj_map.items():
        sub = df.copy()
        sub["timestamp"] = pd.to_datetime(sub["timestamp"])
        sub = sub.sort_values("timestamp").iloc[:max_points]
        ax.plot(sub["timestamp"], sub["soc"], label=name, linewidth=1.6)
    ax.set_title("Battery State of Charge (SOC) Comparison")
    ax.set_xlabel("Time")
    ax.set_ylabel("State of charge")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(alpha=0.3)
    _format_datetime_xaxis(ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_total_cost_bars(metrics_df: pd.DataFrame, output_path: Path) -> None:
    """Bar chart of total electricity cost by strategy."""
    _ensure_parent(output_path)
    df = metrics_df.copy()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(df["strategy"], df["total_cost"], color=plt.cm.tab10(np.linspace(0, 0.9, len(df))))
    ax.set_title("Total Electricity Cost by Strategy")
    ax.set_ylabel("Total cost (GBP)")
    ax.set_xlabel("Strategy")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_peak_grid_import_bars(metrics_df: pd.DataFrame, output_path: Path) -> None:
    """Bar chart of peak grid import by strategy."""
    _ensure_parent(output_path)
    df = metrics_df.copy()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(df["strategy"], df["peak_grid_import"], color=plt.cm.tab10(np.linspace(0, 0.9, len(df))))
    ax.set_title("Peak Grid Import by Strategy")
    ax.set_ylabel("Peak grid import (kWh)")
    ax.set_xlabel("Strategy")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_grid_import_curves(
    traj_map: Dict[str, pd.DataFrame],
    output_path: Path,
    max_points: int = 24 * 14,
) -> None:
    """Overlay grid import time series for multiple strategies."""
    _ensure_parent(output_path)
    fig, ax = plt.subplots(figsize=(12, 5))
    for name, df in traj_map.items():
        sub = df.copy()
        sub["timestamp"] = pd.to_datetime(sub["timestamp"])
        sub = sub.sort_values("timestamp").iloc[:max_points]
        ax.plot(sub["timestamp"], sub["grid_import"], label=name, linewidth=1.4, alpha=0.9)
    ax.set_title("Grid Import Time Series Comparison")
    ax.set_xlabel("Time")
    ax.set_ylabel("Grid import (kWh)")
    ax.legend()
    ax.grid(alpha=0.3)
    _format_datetime_xaxis(ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_weekly_grid_window(
    traj_map: Dict[str, pd.DataFrame],
    output_path: Path,
    start_time: str | None = None,
) -> None:
    """One-week zoom for grid import curves."""
    _ensure_parent(output_path)
    first = next(iter(traj_map.values())).copy()
    first["timestamp"] = pd.to_datetime(first["timestamp"])
    if start_time is None:
        start = first["timestamp"].iloc[0]
    else:
        start = pd.to_datetime(start_time)
    end = start + pd.Timedelta(days=7)

    fig, ax = plt.subplots(figsize=(12, 5))
    for name, df in traj_map.items():
        sub = df.copy()
        sub["timestamp"] = pd.to_datetime(sub["timestamp"])
        sub = sub.sort_values("timestamp")
        w = sub[(sub["timestamp"] >= start) & (sub["timestamp"] < end)]
        ax.plot(w["timestamp"], w["grid_import"], label=name, linewidth=1.4)
    ax.set_title("Weekly Grid Import (7-Day Window)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Grid import (kWh)")
    ax.legend()
    ax.grid(alpha=0.3)
    _format_datetime_xaxis(ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
