"""Dispatch baseline simulations: no battery and rule-based."""

from __future__ import annotations

import json
from typing import Dict, Tuple

import pandas as pd

from battery_env import HomeBatteryEnv
from config import ProjectConfig, get_tou_price
from rule_based_controller import simulate_rule_based
from utils_metrics import dispatch_metrics, metrics_dict_to_df


def simulate_no_battery(cfg: ProjectConfig, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """No-control baseline where grid import equals load."""
    ts = test_df.index
    out = pd.DataFrame(index=ts)
    out.index.name = "timestamp"
    out["load_kwh"] = test_df["load_kwh"].values
    out["price"] = [get_tou_price(t.hour, cfg) for t in ts]
    out["action"] = 0
    out["soc"] = cfg.init_soc
    out["charge_power"] = 0.0
    out["discharge_power"] = 0.0
    out["grid_import"] = out["load_kwh"]
    out["step_cost"] = out["grid_import"] * out["price"]

    metrics = dispatch_metrics(out.reset_index())
    return out.reset_index(), metrics


def simulate_rule_based_baseline(
    cfg: ProjectConfig,
    test_df: pd.DataFrame,
    g_cap: float,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Rule-based policy simulation through the battery environment."""
    env = HomeBatteryEnv(
        df=test_df,
        cfg=cfg,
        use_forecast=False,
        g_cap=g_cap,
        lambda_peak=cfg.lambda_peak,
        mu_action=cfg.mu_action,
    )
    _, traj = simulate_rule_based(env)
    metrics = dispatch_metrics(traj)
    return traj, metrics


def run_dispatch_baselines(
    cfg: ProjectConfig,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> pd.DataFrame:
    """Run and save no-battery and rule-based dispatch baselines."""
    cfg.ensure_directories()
    g_cap = cfg.compute_g_cap(train_df["load_kwh"])

    no_batt_ts, no_batt_metrics = simulate_no_battery(cfg, test_df)
    rule_ts, rule_metrics = simulate_rule_based_baseline(cfg, test_df, g_cap=g_cap)

    # add comparable metric
    baseline_peak = no_batt_metrics["peak_grid_import"]
    rule_metrics = dispatch_metrics(rule_ts, baseline_peak=baseline_peak)

    no_batt_ts.to_csv(cfg.predictions_dir / "no_battery_timeseries.csv", index=False)
    rule_ts.to_csv(cfg.predictions_dir / "rule_based_timeseries.csv", index=False)

    no_batt_df = metrics_dict_to_df(no_batt_metrics, "No battery")
    rule_df = metrics_dict_to_df(rule_metrics, "Rule-based")
    all_df = pd.concat([no_batt_df, rule_df], ignore_index=True)
    all_df.to_csv(cfg.metrics_dir / "dispatch_baselines_metrics.csv", index=False)

    no_batt_df.to_csv(cfg.metrics_dir / "no_battery_metrics.csv", index=False)
    rule_df.to_csv(cfg.metrics_dir / "rule_based_metrics.csv", index=False)

    with open(cfg.metrics_dir / "dispatch_baselines_metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "No battery": no_batt_metrics,
                "Rule-based": rule_metrics,
            },
            f,
            indent=2,
        )

    return all_df

