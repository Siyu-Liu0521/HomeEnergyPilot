"""Central project configuration for HomeEnergyPilot."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

import numpy as np


@dataclass
class ProjectConfig:
    """Configuration container used by all modules."""

    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1])

    # Paths
    raw_data_path: Path = field(init=False)
    processed_dir: Path = field(init=False)
    outputs_dir: Path = field(init=False)
    figures_dir: Path = field(init=False)
    models_dir: Path = field(init=False)
    predictions_dir: Path = field(init=False)
    metrics_dir: Path = field(init=False)

    # Data / forecasting setup
    resample_rule: str = "1H"
    window_size: int = 24
    horizon: int = 1
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    moving_average_window: int = 24

    # LSTM setup
    lstm_hidden_size: int = 64
    lstm_layers: int = 1
    lstm_dropout: float = 0.2
    lstm_loss: str = "mse"
    lstm_optimizer: str = "adam"
    batch_size: int = 64
    epochs: int = 30
    patience: int = 5
    random_seed: int = 42

    # Battery environment setup
    battery_capacity_kwh: float = 10.0
    battery_pmax_kw: float = 2.0
    battery_eta: float = 0.95
    init_soc: float = 0.5
    soc_min: float = 0.1
    soc_max: float = 1.0
    g_cap_quantile: float = 0.9
    lambda_peak: float = 0.1
    mu_action: float = 0.01

    # TOU prices (GBP/kWh)
    tou_prices: Dict[str, float] = field(
        default_factory=lambda: {"off_peak": 0.15, "mid": 0.25, "peak": 0.45}
    )

    def __post_init__(self) -> None:
        self.raw_data_path = self.project_root / "data" / "raw" / "household_power_consumption.txt"
        self.processed_dir = self.project_root / "data" / "processed"
        self.outputs_dir = self.project_root / "outputs"
        self.figures_dir = self.outputs_dir / "figures"
        self.models_dir = self.outputs_dir / "models"
        self.predictions_dir = self.outputs_dir / "predictions"
        self.metrics_dir = self.outputs_dir / "metrics"

    def ensure_directories(self) -> None:
        """Create required output and processed directories."""
        for d in [
            self.processed_dir,
            self.outputs_dir,
            self.figures_dir,
            self.models_dir,
            self.predictions_dir,
            self.metrics_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)

    def compute_g_cap(self, load_series) -> float:
        """Compute default grid-cap threshold from load quantile."""
        return float(np.quantile(load_series, self.g_cap_quantile))


def get_tou_price(hour: int, cfg: ProjectConfig) -> float:
    """Return TOU electricity price by hour (0-23)."""
    if 0 <= hour < 6:
        return cfg.tou_prices["off_peak"]
    if 6 <= hour < 16:
        return cfg.tou_prices["mid"]
    if 16 <= hour < 21:
        return cfg.tou_prices["peak"]
    return cfg.tou_prices["mid"]
