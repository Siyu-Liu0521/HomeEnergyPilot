"""Rule-based control policy for HomeBatteryEnv."""

from __future__ import annotations

from typing import List, Tuple

import pandas as pd

from battery_env import HomeBatteryEnv


def choose_rule_action(
    hour: int,
    soc: float,
    load: float,
    g_cap: float,
    soc_min: float,
    soc_max: float,
) -> int:
    """
    Time-of-use rule policy.

    - Off-peak (00-06): charge when possible.
    - Peak (16-21): discharge when possible.
    - Mid period: idle by default; discharge if load exceeds cap and SOC allows.
    """
    if 0 <= hour < 6 and soc < soc_max - 1e-6:
        return 1
    if 16 <= hour < 21 and soc > soc_min + 1e-6:
        return 2
    if load > g_cap and soc > soc_min + 1e-6:
        return 2
    return 0


def simulate_rule_based(env: HomeBatteryEnv) -> Tuple[List[int], pd.DataFrame]:
    """Run full episode with rule-based actions and return action list + trajectory."""
    actions: List[int] = []
    env.reset()
    done = False
    while not done:
        ts = env.df.index[env.current_step]
        row = env.df.iloc[env.current_step]
        action = choose_rule_action(
            hour=ts.hour,
            soc=env.soc,
            load=float(row["load_kwh"]),
            g_cap=env.g_cap,
            soc_min=env.cfg.soc_min,
            soc_max=env.cfg.soc_max,
        )
        actions.append(action)
        _, _, done, _, _ = env.step(action)
    return actions, env.get_history_df()

