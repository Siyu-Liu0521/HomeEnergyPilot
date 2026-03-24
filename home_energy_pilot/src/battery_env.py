"""Gym-style home battery environment for dispatch simulation and RL."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import ProjectConfig, get_tou_price


@dataclass
class StepResult:
    """Container for one environment transition."""

    next_state: np.ndarray
    reward: float
    done: bool
    truncated: bool
    info: Dict


class HomeBatteryEnv:
    """
    Simple discrete-action battery environment.

    Action space:
    - 0: idle
    - 1: charge (+Pmax)
    - 2: discharge (-Pmax)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        cfg: Optional[ProjectConfig] = None,
        use_forecast: bool = False,
        g_cap: Optional[float] = None,
        lambda_peak: Optional[float] = None,
        mu_action: Optional[float] = None,
    ) -> None:
        self.cfg = cfg or ProjectConfig()
        if "load_kwh" not in df.columns:
            raise ValueError("Environment dataframe must contain `load_kwh` column.")
        if use_forecast and "forecast_load" not in df.columns:
            raise ValueError("use_forecast=True requires dataframe column `forecast_load`.")

        self.df = df.copy()
        if "timestamp" in self.df.columns:
            self.df["timestamp"] = pd.to_datetime(self.df["timestamp"])
            self.df = self.df.set_index("timestamp")
        if not isinstance(self.df.index, pd.DatetimeIndex):
            raise TypeError("Environment dataframe must use DatetimeIndex or have `timestamp` column.")
        self.df = self.df.sort_index()

        self.use_forecast = use_forecast
        self.g_cap = g_cap if g_cap is not None else self.cfg.compute_g_cap(self.df["load_kwh"])
        self.lambda_peak = self.cfg.lambda_peak if lambda_peak is None else lambda_peak
        self.mu_action = self.cfg.mu_action if mu_action is None else mu_action

        self.current_step = 0
        self.soc = self.cfg.init_soc
        self.prev_action = 0
        self.history: List[Dict] = []

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """Reset episode to first time step."""
        if seed is not None:
            np.random.seed(seed)
        self.current_step = 0
        self.soc = self.cfg.init_soc
        self.prev_action = 0
        self.history = []
        return self._get_obs(), {"timestamp": str(self.df.index[self.current_step])}

    def _get_obs(self) -> np.ndarray:
        """Build numeric state vector."""
        ts = self.df.index[self.current_step]
        load = float(self.df.iloc[self.current_step]["load_kwh"])
        price = float(get_tou_price(ts.hour, self.cfg))
        hour_sin = np.sin(2 * np.pi * ts.hour / 24)
        hour_cos = np.cos(2 * np.pi * ts.hour / 24)
        obs = [load, price, float(self.soc), float(hour_sin), float(hour_cos)]
        if self.use_forecast:
            obs.append(float(self.df.iloc[self.current_step]["forecast_load"]))
        return np.asarray(obs, dtype=np.float32)

    def _apply_action(self, action: int) -> Tuple[float, float]:
        """
        Apply action with SOC and efficiency constraints.

        Returns:
        - charge_power (kWh in current 1h step)
        - discharge_power (kWh in current 1h step)
        """
        if action not in (0, 1, 2):
            raise ValueError("Action must be 0, 1, or 2.")

        charge_power = 0.0
        discharge_power = 0.0
        pmax = self.cfg.battery_pmax_kw
        c = self.cfg.battery_capacity_kwh
        eta = self.cfg.battery_eta

        if action == 1:  # charge
            max_charge_by_soc = max(0.0, (self.cfg.soc_max - self.soc) * c / eta)
            charge_power = min(pmax, max_charge_by_soc)
            self.soc += (charge_power * eta) / c
        elif action == 2:  # discharge
            max_discharge_by_soc = max(0.0, (self.soc - self.cfg.soc_min) * c * eta)
            discharge_power = min(pmax, max_discharge_by_soc)
            self.soc -= (discharge_power / eta) / c

        self.soc = float(np.clip(self.soc, self.cfg.soc_min, self.cfg.soc_max))
        return float(charge_power), float(discharge_power)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Run one environment transition."""
        if self.current_step >= len(self.df):
            raise RuntimeError("Episode is done. Call reset() before step().")

        ts = self.df.index[self.current_step]
        load = float(self.df.iloc[self.current_step]["load_kwh"])
        price = float(get_tou_price(ts.hour, self.cfg))
        charge_power, discharge_power = self._apply_action(action)

        grid_import = max(0.0, load + charge_power - discharge_power)
        step_cost = price * grid_import
        peak_penalty = self.lambda_peak * max(0.0, grid_import - self.g_cap)
        action_penalty = self.mu_action * (1.0 if action != 0 else 0.0)
        reward = -(step_cost + peak_penalty + action_penalty)

        info = {
            "timestamp": ts,
            "load_kwh": load,
            "price": price,
            "action": int(action),
            "soc": float(self.soc),
            "charge_power": charge_power,
            "discharge_power": discharge_power,
            "grid_import": float(grid_import),
            "step_cost": float(step_cost),
            "peak_penalty": float(peak_penalty),
            "action_penalty": float(action_penalty),
        }
        self.history.append(info)
        self.prev_action = int(action)

        self.current_step += 1
        done = self.current_step >= len(self.df)
        truncated = False
        next_state = self._get_obs() if not done else np.zeros(5 + int(self.use_forecast), dtype=np.float32)
        return next_state, float(reward), done, truncated, info

    def get_history_df(self) -> pd.DataFrame:
        """Return trajectory dataframe for all executed steps."""
        return pd.DataFrame(self.history)


def run_random_policy_debug(
    env: HomeBatteryEnv,
    n_steps: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """Quick random rollout helper for environment debugging."""
    rng = np.random.default_rng(seed)
    env.reset(seed=seed)
    done = False
    step_count = 0
    while not done and step_count < n_steps:
        action = int(rng.integers(0, 3))
        _, _, done, _, _ = env.step(action)
        step_count += 1
    return env.get_history_df()
