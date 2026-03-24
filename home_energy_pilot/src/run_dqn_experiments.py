"""DQN training and evaluation (new module; does not modify existing project files)."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from battery_env import HomeBatteryEnv
from config import ProjectConfig
from dqn_agent import DQNAgent, DQNHyperParams
from dqn_plotting import (
    plot_grid_import_curves,
    plot_peak_grid_import_bars,
    plot_reward_convergence,
    plot_soc_curves,
    plot_total_cost_bars,
    plot_weekly_grid_window,
)
from simulate_baselines import simulate_no_battery, simulate_rule_based_baseline
from utils_metrics import dispatch_metrics, metrics_dict_to_df


def parse_args() -> argparse.Namespace:
    """CLI for full runs and quick multi-seed validation (English help text)."""
    p = argparse.ArgumentParser(description="Train and evaluate DQN dispatch policies.")
    p.add_argument(
        "--quick",
        action="store_true",
        help="Quick validation: shorter training, multi-seed table, separate CSV outputs (no main report overwrite).",
    )
    p.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Environment steps per training run (default: 120000 full, 10000 quick).",
    )
    p.add_argument(
        "--reward-scale",
        type=float,
        default=None,
        help="Scale rewards stored in replay buffer (default: 1.0 full, 0.01 quick). Unchanged env + metrics.",
    )
    p.add_argument(
        "--seeds",
        type=str,
        default=None,
        help=(
            "Comma-separated RNG seeds. Quick default: 42,43,44. Full default: config random_seed. "
            "Full multi-seed: runs all seeds, writes dqn_full_multiseed_*.csv, "
            "then uses the seed with lowest RL+forecast test total_cost for main figures/CSVs."
        ),
    )
    p.add_argument(
        "--no-double-dqn",
        action="store_true",
        help="Disable Double DQN (use vanilla max Q from target network for bootstrap).",
    )
    p.add_argument(
        "--norm-samples",
        type=int,
        default=None,
        help="Random rollout steps to fit observation normalizer (default: 25000 full, 5000 quick).",
    )
    p.add_argument(
        "--min-replay",
        type=int,
        default=None,
        help="Minimum replay size before gradient updates (default: 4000 full, 2000 quick).",
    )
    return p.parse_args()


def _parse_seed_list(s: str) -> List[int]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return [int(x) for x in parts]


def _summarize_by_variant(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """Mean/std of numeric metrics grouped by variant (English column names)."""
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    metric_cols = [c for c in df.columns if c not in ("seed", "variant")]
    out_rows: List[Dict[str, Any]] = []
    for variant, g in df.groupby("variant"):
        row: Dict[str, Any] = {"variant": variant, "n_seeds": int(len(g))}
        for c in metric_cols:
            if pd.api.types.is_numeric_dtype(g[c]):
                row[f"{c}_mean"] = float(np.nanmean(g[c].astype(float)))
                row[f"{c}_std"] = float(np.nanstd(g[c].astype(float), ddof=1)) if len(g) > 1 else 0.0
        out_rows.append(row)
    return pd.DataFrame(out_rows)


class StateNormalizer:
    """Z-score normalization for environment observations."""

    def __init__(self, mean: np.ndarray, std: np.ndarray) -> None:
        self.mean = mean.astype(np.float32)
        self.std = np.where(std.astype(np.float32) < 1e-6, 1.0, std.astype(np.float32))

    def transform(self, obs: np.ndarray) -> np.ndarray:
        x = (obs.astype(np.float32) - self.mean) / self.std
        return x.astype(np.float32)


def fit_state_normalizer(
    env: HomeBatteryEnv,
    rng: np.random.Generator,
    n_steps: int,
    desc: str = "Collecting rollouts for observation normalization",
) -> StateNormalizer:
    """Collect random rollouts to estimate observation statistics."""
    obs, _ = env.reset()
    buf: list[np.ndarray] = []
    for _ in tqdm(range(n_steps), desc=desc, unit="step", leave=False):
        buf.append(obs.astype(np.float32))
        a = int(rng.integers(0, 3))
        next_obs, _, done, _, _ = env.step(a)
        if done:
            obs, _ = env.reset()
        else:
            obs = next_obs
    mat = np.stack(buf, axis=0)
    mean = mat.mean(axis=0)
    std = mat.std(axis=0)
    return StateNormalizer(mean, std)


def make_train_forecast_df(train_df: pd.DataFrame) -> pd.DataFrame:
    """Training-side forecast channel: use actual load as stand-in for forecast column."""
    out = train_df.copy()
    out["forecast_load"] = out["load_kwh"].astype(float)
    return out


def load_aligned_test_forecast(cfg: ProjectConfig, test_df: pd.DataFrame) -> pd.DataFrame:
    """Load LSTM test forecast sequence and align to test index."""
    path = cfg.predictions_dir / "test_forecast_sequence.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run the main pipeline first so LSTM exports test_forecast_sequence.csv."
        )
    fc = pd.read_csv(path, parse_dates=["timestamp"])
    fc = fc.set_index("timestamp").sort_index()
    idx = test_df.index.intersection(fc.index)
    if len(idx) == 0:
        raise ValueError("No overlapping timestamps between test split and forecast sequence.")
    out = test_df.loc[idx].copy()
    out["forecast_load"] = fc.loc[idx, "forecast_load"].astype(float).values
    return out


def train_dqn(
    agent: DQNAgent,
    env: HomeBatteryEnv,
    normalizer: StateNormalizer,
    max_train_steps: int,
    seed: int,
    desc: str = "DQN training",
    reward_scale: float = 1.0,
) -> list[float]:
    """Train DQN with epsilon decay; returns unscaled episode returns (for logging)."""
    rng = np.random.default_rng(seed)
    obs_raw, _ = env.reset()
    obs_n = normalizer.transform(obs_raw)
    global_step = 0
    episode_returns: list[float] = []
    ep_return = 0.0
    ep_idx = 0

    pbar = tqdm(
        total=max_train_steps,
        desc=desc,
        unit="step",
        dynamic_ncols=True,
    )
    while global_step < max_train_steps:
        eps = agent.epsilon_by_step(global_step)
        action = agent.select_action(obs_n, eps)
        next_raw, reward, done, _, _ = env.step(action)
        ep_return += float(reward)

        if done:
            next_n = np.zeros_like(obs_n, dtype=np.float32)
        else:
            next_n = normalizer.transform(next_raw)

        scaled_r = float(reward) * float(reward_scale)
        agent.buffer.push(obs_n, action, scaled_r, next_n, done)

        if len(agent.buffer) >= agent.hp.min_replay_size:
            agent.train_from_buffer()

        global_step += 1
        pbar.update(1)
        buf_len = len(agent.buffer)
        training = buf_len >= agent.hp.min_replay_size
        pbar.set_postfix(
            epsilon=f"{eps:.3f}",
            ep=f"{ep_idx}",
            ep_ret=f"{ep_return:.1f}",
            buffer=buf_len,
            train="on" if training else "off",
            rscale=f"{reward_scale:g}",
        )

        if done:
            episode_returns.append(ep_return)
            ep_return = 0.0
            ep_idx += 1
            obs_raw, _ = env.reset()
            obs_n = normalizer.transform(obs_raw)
        else:
            obs_raw = next_raw
            obs_n = next_n

    pbar.close()
    return episode_returns


def rollout_greedy(
    agent: DQNAgent,
    env: HomeBatteryEnv,
    normalizer: StateNormalizer,
    desc: str = "Greedy policy evaluation on test split",
) -> pd.DataFrame:
    """Run a full greedy episode and return trajectory dataframe."""
    obs_raw, _ = env.reset()
    obs_n = normalizer.transform(obs_raw)
    done = False
    n_steps = len(env.df)
    pbar = tqdm(total=n_steps, desc=desc, unit="step", leave=False, dynamic_ncols=True)
    while not done:
        action = agent.select_action(obs_n, epsilon=0.0)
        next_raw, _, done, _, _ = env.step(action)
        pbar.update(1)
        if done:
            break
        obs_n = normalizer.transform(next_raw)
    pbar.close()
    return env.get_history_df()


def run_dqn_pair_for_seed(
    cfg: ProjectConfig,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_fc_df: pd.DataFrame,
    test_fc_df: pd.DataFrame,
    g_cap: float,
    baseline_peak: float,
    hp: DQNHyperParams,
    max_train_steps: int,
    reward_scale: float,
    norm_samples: int,
    seed: int,
) -> Dict[str, Any]:
    """Train/eval RL-only and RL+forecast for one seed; returns metrics, trajectories, agents, returns."""
    rng = np.random.default_rng(seed)

    env_train_rl = HomeBatteryEnv(df=train_df, cfg=cfg, use_forecast=False, g_cap=g_cap)
    norm_rl = fit_state_normalizer(
        env_train_rl,
        rng,
        n_steps=norm_samples,
        desc="Random rollouts: RL-only (normalizer)",
    )
    agent_rl = DQNAgent(5, n_actions=3, hp=hp, seed=seed)
    returns_rl = train_dqn(
        agent_rl,
        env_train_rl,
        norm_rl,
        max_train_steps=max_train_steps,
        seed=seed,
        desc="DQN training: RL-only",
        reward_scale=reward_scale,
    )
    env_test_rl = HomeBatteryEnv(df=test_df, cfg=cfg, use_forecast=False, g_cap=g_cap)
    traj_rl = rollout_greedy(agent_rl, env_test_rl, norm_rl, desc="Greedy eval: RL-only on test")
    metrics_rl = dispatch_metrics(traj_rl, baseline_peak=baseline_peak)

    rng_fc = np.random.default_rng(seed + 9_999)
    env_train_fc = HomeBatteryEnv(df=train_fc_df, cfg=cfg, use_forecast=True, g_cap=g_cap)
    norm_fc = fit_state_normalizer(
        env_train_fc,
        rng_fc,
        n_steps=norm_samples,
        desc="Random rollouts: RL+forecast (normalizer)",
    )
    agent_fc = DQNAgent(6, n_actions=3, hp=hp, seed=seed + 1)
    returns_fc = train_dqn(
        agent_fc,
        env_train_fc,
        norm_fc,
        max_train_steps=max_train_steps,
        seed=seed + 1,
        desc="DQN training: RL+forecast",
        reward_scale=reward_scale,
    )
    env_test_fc = HomeBatteryEnv(df=test_fc_df, cfg=cfg, use_forecast=True, g_cap=g_cap)
    traj_fc = rollout_greedy(agent_fc, env_test_fc, norm_fc, desc="Greedy eval: RL+forecast on test")
    metrics_fc = dispatch_metrics(traj_fc, baseline_peak=baseline_peak)

    return {
        "metrics_rl": metrics_rl,
        "metrics_fc": metrics_fc,
        "returns_rl": returns_rl,
        "returns_fc": returns_fc,
        "traj_rl": traj_rl,
        "traj_fc": traj_fc,
        "agent_rl": agent_rl,
        "agent_fc": agent_fc,
    }


def main() -> None:
    args = parse_args()
    cfg = ProjectConfig()
    cfg.ensure_directories()

    train_path = cfg.processed_dir / "train.csv"
    test_path = cfg.processed_dir / "test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Missing processed splits. Run data preprocessing first: {train_path}, {test_path}"
        )

    train_df = pd.read_csv(train_path, index_col="timestamp", parse_dates=True).sort_index()
    test_df = pd.read_csv(test_path, index_col="timestamp", parse_dates=True).sort_index()
    g_cap = cfg.compute_g_cap(train_df["load_kwh"])

    quick = bool(args.quick)
    max_train_steps = args.steps if args.steps is not None else (10_000 if quick else 120_000)
    reward_scale = args.reward_scale if args.reward_scale is not None else (0.01 if quick else 1.0)
    seeds = _parse_seed_list(args.seeds) if args.seeds else ([42, 43, 44] if quick else [cfg.random_seed])
    norm_samples = args.norm_samples if args.norm_samples is not None else (5_000 if quick else 25_000)
    min_replay = args.min_replay if args.min_replay is not None else (2_000 if quick else 4_000)
    epsilon_decay = max_train_steps if quick else 120_000
    replay_cap = min(200_000, max(50_000, max_train_steps * 4))

    hp = DQNHyperParams(
        gamma=0.99,
        learning_rate=1e-3,
        replay_capacity=replay_cap,
        batch_size=128,
        min_replay_size=min_replay,
        target_update_every=500,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_steps=epsilon_decay,
        hidden_units=(128, 128),
        gradient_clip=10.0,
        use_double_dqn=not bool(args.no_double_dqn),
    )

    no_batt_ts, no_batt_metrics = simulate_no_battery(cfg, test_df)
    rule_ts, rule_metrics = simulate_rule_based_baseline(cfg, test_df, g_cap=g_cap)
    baseline_peak = no_batt_metrics["peak_grid_import"]
    rule_metrics = dispatch_metrics(rule_ts, baseline_peak=baseline_peak)

    train_fc_df = make_train_forecast_df(train_df)
    test_fc_df = load_aligned_test_forecast(cfg, test_df)

    print("\n=== DQN experiments (progress bars in English) ===\n")
    algo_name = "Double DQN" if hp.use_double_dqn else "Vanilla DQN (max target Q)"
    print(f"Algorithm: {algo_name}\n")
    if quick:
        print(
            "Quick validation mode: shorter training, reward scaling in replay buffer, multiple seeds.\n"
            f"steps={max_train_steps}, reward_scale={reward_scale}, seeds={seeds}, "
            f"norm_samples={norm_samples}, min_replay={min_replay}, epsilon_decay_steps={epsilon_decay}\n"
        )

    if quick and len(seeds) >= 1:
        rows: List[Dict[str, Any]] = []
        for sd in seeds:
            out = run_dqn_pair_for_seed(
                cfg=cfg,
                train_df=train_df,
                test_df=test_df,
                train_fc_df=train_fc_df,
                test_fc_df=test_fc_df,
                g_cap=g_cap,
                baseline_peak=baseline_peak,
                hp=hp,
                max_train_steps=max_train_steps,
                reward_scale=reward_scale,
                norm_samples=norm_samples,
                seed=sd,
            )
            mr = out["metrics_rl"]
            mf = out["metrics_fc"]
            row_rl = {"seed": sd, "variant": "DQN (RL-only)"}
            row_rl.update(mr)
            row_fc = {"seed": sd, "variant": "DQN (RL+forecast)"}
            row_fc.update(mf)
            rows.append(row_rl)
            rows.append(row_fc)

        runs_df = pd.DataFrame(rows)
        runs_path = cfg.metrics_dir / "dqn_quick_validate_runs.csv"
        runs_df.to_csv(runs_path, index=False)

        summary_df = _summarize_by_variant(rows)
        summary_path = cfg.metrics_dir / "dqn_quick_validate_summary.csv"
        summary_df.to_csv(summary_path, index=False)

        with open(cfg.metrics_dir / "dqn_quick_validate_config.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "quick": True,
                    "hyperparams": asdict(hp),
                    "max_train_steps": max_train_steps,
                    "reward_scale": reward_scale,
                    "seeds": seeds,
                    "norm_samples": norm_samples,
                    "g_cap": g_cap,
                },
                f,
                indent=2,
            )

        print("Quick validation finished.")
        print("Per-seed metrics:", runs_path)
        print("Summary (mean/std):", summary_path)
        print("\n--- Summary (English) ---")
        print(summary_df.to_string(index=False))
        return

    # --- Full run: single or multi-seed (main report + figures) ---
    selected_seed: int
    best_seed_rl_forecast_cost: Optional[float] = None
    if len(seeds) > 1:
        print(
            f"Full multi-seed mode: training {len(seeds)} seeds {seeds}. "
            "Selecting best seed by minimum RL+forecast test total_cost for main artifacts.\n"
        )
        rows_full: List[Dict[str, Any]] = []
        outs_by_seed: Dict[int, Dict[str, Any]] = {}
        for sd in seeds:
            o = run_dqn_pair_for_seed(
                cfg=cfg,
                train_df=train_df,
                test_df=test_df,
                train_fc_df=train_fc_df,
                test_fc_df=test_fc_df,
                g_cap=g_cap,
                baseline_peak=baseline_peak,
                hp=hp,
                max_train_steps=max_train_steps,
                reward_scale=reward_scale,
                norm_samples=norm_samples,
                seed=sd,
            )
            outs_by_seed[sd] = o
            mr = o["metrics_rl"]
            mf = o["metrics_fc"]
            row_rl = {"seed": sd, "variant": "DQN (RL-only)"}
            row_rl.update(mr)
            row_fc = {"seed": sd, "variant": "DQN (RL+forecast)"}
            row_fc.update(mf)
            rows_full.extend([row_rl, row_fc])

        runs_full_df = pd.DataFrame(rows_full)
        runs_full_df.to_csv(cfg.metrics_dir / "dqn_full_multiseed_runs.csv", index=False)
        summary_full_df = _summarize_by_variant(rows_full)
        summary_full_df.to_csv(cfg.metrics_dir / "dqn_full_multiseed_summary.csv", index=False)
        print("Full multi-seed per-run metrics:", cfg.metrics_dir / "dqn_full_multiseed_runs.csv")
        print("Full multi-seed summary (mean/std):", cfg.metrics_dir / "dqn_full_multiseed_summary.csv")

        selected_seed = min(seeds, key=lambda s: float(outs_by_seed[s]["metrics_fc"]["total_cost"]))
        best_seed_rl_forecast_cost = float(outs_by_seed[selected_seed]["metrics_fc"]["total_cost"])
        out = outs_by_seed[selected_seed]
        print(
            f"\nSelected seed for dispatch CSVs / figures / saved models: {selected_seed} "
            f"(RL+forecast test total_cost={best_seed_rl_forecast_cost:.6f})\n"
        )
    else:
        selected_seed = seeds[0]
        out = run_dqn_pair_for_seed(
            cfg=cfg,
            train_df=train_df,
            test_df=test_df,
            train_fc_df=train_fc_df,
            test_fc_df=test_fc_df,
            g_cap=g_cap,
            baseline_peak=baseline_peak,
            hp=hp,
            max_train_steps=max_train_steps,
            reward_scale=reward_scale,
            norm_samples=norm_samples,
            seed=selected_seed,
        )

    metrics_dqn_rl = out["metrics_rl"]
    metrics_dqn_fc = out["metrics_fc"]
    returns_rl = out["returns_rl"]
    returns_fc = out["returns_fc"]
    traj_dqn_rl = out["traj_rl"]
    traj_dqn_fc = out["traj_fc"]
    agent_rl = out["agent_rl"]
    agent_fc = out["agent_fc"]

    agent_rl.online_q.save(str(cfg.models_dir / "dqn_rl_only.keras"))
    agent_fc.online_q.save(str(cfg.models_dir / "dqn_rl_forecast.keras"))

    traj_dqn_rl.to_csv(cfg.predictions_dir / "dqn_rl_only_timeseries.csv", index=False)
    traj_dqn_fc.to_csv(cfg.predictions_dir / "dqn_rl_forecast_timeseries.csv", index=False)

    row_no = metrics_dict_to_df(no_batt_metrics, "No battery")
    row_rule = metrics_dict_to_df(rule_metrics, "Rule-based")
    row_dqn_rl = metrics_dict_to_df(metrics_dqn_rl, "DQN (RL-only)")
    row_dqn_fc = metrics_dict_to_df(metrics_dqn_fc, "DQN (RL+forecast)")

    three_df = pd.concat([row_no, row_rule, row_dqn_rl], ignore_index=True)
    three_df.to_csv(cfg.metrics_dir / "dispatch_three_strategies_metrics.csv", index=False)

    all_df = pd.concat([row_no, row_rule, row_dqn_rl, row_dqn_fc], ignore_index=True)
    all_df.to_csv(cfg.metrics_dir / "dispatch_all_strategies_metrics.csv", index=False)

    ablation_df = pd.concat([row_dqn_rl, row_dqn_fc], ignore_index=True)
    ablation_df.to_csv(cfg.metrics_dir / "dqn_rl_vs_forecast_metrics.csv", index=False)

    cfg_payload: Dict[str, Any] = {
        "hyperparams": asdict(hp),
        "max_train_steps": max_train_steps,
        "reward_scale": reward_scale,
        "seeds": seeds,
        "selected_seed": selected_seed,
        "g_cap": g_cap,
    }
    if len(seeds) > 1:
        cfg_payload["selection_rule"] = "min_RL_plus_forecast_test_total_cost"
        cfg_payload["selected_rl_forecast_total_cost"] = best_seed_rl_forecast_cost
    with open(cfg.metrics_dir / "dqn_training_config.json", "w", encoding="utf-8") as f:
        json.dump(cfg_payload, f, indent=2)

    max_ep = max(len(returns_rl), len(returns_fc))
    pad_rl = returns_rl + [float("nan")] * (max_ep - len(returns_rl))
    pad_fc = returns_fc + [float("nan")] * (max_ep - len(returns_fc))
    pd.DataFrame(
        {
            "episode": range(1, max_ep + 1),
            "return_rl_only": pad_rl,
            "return_rl_forecast": pad_fc,
        }
    ).to_csv(cfg.metrics_dir / "dqn_episode_returns.csv", index=False)

    fig_dir = cfg.figures_dir
    plot_reward_convergence(
        {"RL-only": returns_rl, "RL+forecast": returns_fc},
        fig_dir / "dqn_reward_convergence.png",
    )

    soc_map = {
        "Rule-based": rule_ts,
        "DQN (RL-only)": traj_dqn_rl,
        "DQN (RL+forecast)": traj_dqn_fc,
    }
    plot_soc_curves(soc_map, fig_dir / "dqn_soc_comparison.png")

    plot_total_cost_bars(all_df, fig_dir / "dqn_total_cost_bars.png")
    plot_peak_grid_import_bars(all_df, fig_dir / "dqn_peak_grid_import_bars.png")

    grid_map = {
        "No battery": no_batt_ts,
        "Rule-based": rule_ts,
        "DQN (RL-only)": traj_dqn_rl,
        "DQN (RL+forecast)": traj_dqn_fc,
    }
    plot_grid_import_curves(grid_map, fig_dir / "dqn_grid_import_comparison.png")
    plot_weekly_grid_window(grid_map, fig_dir / "dqn_grid_import_week_window.png")

    print("\nDQN experiments complete.")
    print("Metrics (three strategies):", cfg.metrics_dir / "dispatch_three_strategies_metrics.csv")
    print("Metrics (all):", cfg.metrics_dir / "dispatch_all_strategies_metrics.csv")
    print("Figures:", fig_dir)


if __name__ == "__main__":
    main()
