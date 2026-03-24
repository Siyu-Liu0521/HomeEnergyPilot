"""Main one-click pipeline for HomeEnergyPilot."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from baselines_forecast import run_baseline_forecasts
from config import ProjectConfig
from data_preprocessing import run_preprocessing
from feature_engineering import build_and_save_features
from lstm_forecast import run_lstm_experiments
from run_dqn_experiments import main as run_dqn
from simulate_baselines import run_dispatch_baselines
from utils_plot import (
    plot_cost_peak_bars,
    plot_dispatch_grid_import,
    plot_forecast_window,
    plot_rule_soc,
    plot_week_window,
)


def _load_csv_with_dt(path: Path) -> pd.DataFrame:
    """Read CSV and parse timestamp column to DatetimeIndex."""
    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")
    return df


def main() -> None:
    """Run all project steps end-to-end."""
    cfg = ProjectConfig()
    cfg.ensure_directories()

    print("\n========== HomeEnergyPilot Pipeline ==========")

    # Step 1. Data preprocessing
    print("\n[Step 1/9] Data preprocessing...")
    data = run_preprocessing(cfg)
    train_df, val_df, test_df, full_df = (
        data["train"],
        data["val"],
        data["test"],
        data["full"],
    )

    # Step 2. Feature engineering
    print("[Step 2/9] Feature engineering...")
    train_feat, val_feat, test_feat = build_and_save_features(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        processed_dir=cfg.processed_dir,
    )

    # Step 3. Forecast baselines
    print("[Step 3/9] Forecast baselines...")
    baseline_metrics_df, baseline_pred_map = run_baseline_forecasts(cfg, full_df, test_df)

    # Step 4. LSTM forecast
    print("[Step 4/9] LSTM experiments...")
    lstm_results = run_lstm_experiments(cfg, train_feat, val_feat, test_feat)

    # Step 5. Forecast sequence for RL
    print("[Step 5/9] RL forecast sequence generated at outputs/predictions/test_forecast_sequence.csv")

    # Step 6. Dispatch baselines
    print("[Step 6/9] Dispatch baseline simulations...")
    dispatch_metrics_df = run_dispatch_baselines(cfg, train_df, test_df)

    # Step 7. DQN dispatch experiments
    print("[Step 7/9] DQN experiments...")
    run_dqn()

    # Step 8. Figures
    print("[Step 8/9] Plotting figures...")
    pred_map = dict(baseline_pred_map)
    pred_map["LSTM_load_only"] = lstm_results["LSTM_load_only"].pred_df
    pred_map["LSTM_with_time"] = lstm_results["LSTM_with_time"].pred_df
    plot_forecast_window(pred_map, cfg.figures_dir / "forecast_comparison_test_window.png")

    no_batt_ts = pd.read_csv(cfg.predictions_dir / "no_battery_timeseries.csv")
    rule_ts = pd.read_csv(cfg.predictions_dir / "rule_based_timeseries.csv")
    plot_dispatch_grid_import(no_batt_ts, rule_ts, cfg.figures_dir / "dispatch_grid_import_compare.png")
    plot_rule_soc(rule_ts, cfg.figures_dir / "rule_based_soc.png")
    plot_cost_peak_bars(
        dispatch_metrics_df,
        cfg.figures_dir / "dispatch_cost_comparison.png",
        cfg.figures_dir / "dispatch_peak_comparison.png",
    )
    plot_week_window(no_batt_ts, rule_ts, cfg.figures_dir / "dispatch_week_window.png")

    # Step 9. Summary
    print("[Step 9/9] Summary")
    print("\n--- Forecast performance summary ---")
    print(baseline_metrics_df.to_string(index=False))
    print(pd.read_csv(cfg.metrics_dir / "lstm_metrics.csv").to_string(index=False))

    print("\n--- Dispatch baseline summary ---")
    print(dispatch_metrics_df.to_string(index=False))

    print("\nOutput directories:")
    print("Processed:", cfg.processed_dir)
    print("Predictions:", cfg.predictions_dir)
    print("Metrics:", cfg.metrics_dir)
    print("Figures:", cfg.figures_dir)
    print("Models:", cfg.models_dir)
    print("\nPipeline complete.")


if __name__ == "__main__":
    main()

