# HomeEnergyPilot

HomeEnergyPilot is a coursework project for end-to-end household energy analytics and control:

- short-term load forecasting (classical baselines + LSTM),
- battery dispatch simulation under time-of-use electricity tariffs,
- DQN-based dispatch policy learning (RL-only and RL+forecast variants).

## Repository Layout

```text
home_energy_pilot/
├─ data/
│  ├─ raw/
│  └─ processed/
├─ outputs/
│  ├─ figures/
│  ├─ metrics/
│  ├─ models/
│  └─ predictions/
├─ src/
│  ├─ main.py
│  ├─ config.py
│  ├─ data_preprocessing.py
│  ├─ feature_engineering.py
│  ├─ baselines_forecast.py
│  ├─ lstm_forecast.py
│  ├─ battery_env.py
│  ├─ rule_based_controller.py
│  ├─ simulate_baselines.py
│  ├─ dqn_q_network.py
│  ├─ replay_buffer.py
│  ├─ dqn_agent.py
│  ├─ dqn_plotting.py
│  ├─ run_dqn_experiments.py
│  ├─ utils_metrics.py
│  └─ utils_plot.py
├─ requirements.txt
└─ README.md
```

## Requirements

- Python 3.10 or newer
- Recommended: create and activate a virtual environment before installing dependencies

Install dependencies from the project root:

```bash
pip install -r requirements.txt
```

## Dataset Setup

Place the UCI data file at:

`data/raw/household_power_consumption.txt`

The preprocessing module expects columns `Date`, `Time`, and `Global_active_power`.

If your data path is different, update `raw_data_path` in `src/config.py`.

## How To Run

### 1) Full End-to-End Pipeline (recommended)

From the project root:

```bash
python src/main.py
```

This runs the complete workflow, including DQN experiments.

### 2) Run DQN Module Only

If processed splits and `test_forecast_sequence.csv` already exist:

```bash
python src/run_dqn_experiments.py
```

Useful options:

```bash
python src/run_dqn_experiments.py --quick
python src/run_dqn_experiments.py --steps 30000 --seeds 42,43,44
python src/run_dqn_experiments.py --no-double-dqn
```

## Pipeline Stages (`src/main.py`)

1. Data preprocessing from raw UCI file
2. Time-feature engineering for train/val/test
3. Baseline forecasting (Persistence, Moving Average)
4. LSTM forecasting (`load_only`, `with_time`)
5. RL forecast sequence export (`test_forecast_sequence.csv`)
6. Dispatch baseline simulation (No battery, Rule-based)
7. DQN dispatch experiments (`run_dqn_experiments.main`)
8. Figure generation
9. Console summary of key metrics and output directories

## Core Modules

- `data_preprocessing.py`: cleaning, hourly resampling, chronological split, preprocessing logs
- `feature_engineering.py`: cyclical time features and supervised sequence helpers
- `baselines_forecast.py`: persistence and moving-average forecasting baselines
- `lstm_forecast.py`: two LSTM variants, model training, metrics, and RL forecast sequence export
- `battery_env.py`: discrete-action battery environment for simulation and RL
- `simulate_baselines.py`: no-battery and rule-based dispatch baselines
- `run_dqn_experiments.py`: DQN training/evaluation pipeline with optional quick and multi-seed modes
- `utils_metrics.py`: forecasting and dispatch KPI functions
- `utils_plot.py` and `dqn_plotting.py`: baseline and DQN figure generation

## Main Outputs

### Forecast Artifacts

- `outputs/metrics/forecast_baselines_metrics.csv`
- `outputs/metrics/lstm_metrics.csv`
- `outputs/predictions/persistence_test_pred.csv`
- `outputs/predictions/moving_average_test_pred.csv`
- `outputs/predictions/lstm_load_only_test_pred.csv`
- `outputs/predictions/lstm_with_time_test_pred.csv`
- `outputs/predictions/test_forecast_sequence.csv`

### Baseline Dispatch Artifacts

- `outputs/predictions/no_battery_timeseries.csv`
- `outputs/predictions/rule_based_timeseries.csv`
- `outputs/metrics/dispatch_baselines_metrics.csv`
- `outputs/metrics/no_battery_metrics.csv`
- `outputs/metrics/rule_based_metrics.csv`

### DQN Dispatch Artifacts

- `outputs/predictions/dqn_rl_only_timeseries.csv`
- `outputs/predictions/dqn_rl_forecast_timeseries.csv`
- `outputs/metrics/dispatch_three_strategies_metrics.csv`
- `outputs/metrics/dispatch_all_strategies_metrics.csv`
- `outputs/metrics/dqn_rl_vs_forecast_metrics.csv`
- `outputs/metrics/dqn_episode_returns.csv`
- `outputs/metrics/dqn_training_config.json`
- `outputs/models/dqn_rl_only.keras`
- `outputs/models/dqn_rl_forecast.keras`

Quick or multi-seed modes may also produce:

- `outputs/metrics/dqn_quick_validate_runs.csv`
- `outputs/metrics/dqn_quick_validate_summary.csv`
- `outputs/metrics/dqn_quick_validate_config.json`
- `outputs/metrics/dqn_full_multiseed_runs.csv`
- `outputs/metrics/dqn_full_multiseed_summary.csv`

### Figures

- `outputs/figures/forecast_comparison_test_window.png`
- `outputs/figures/lstm_training_curve.png`
- `outputs/figures/dispatch_grid_import_compare.png`
- `outputs/figures/rule_based_soc.png`
- `outputs/figures/dispatch_cost_comparison.png`
- `outputs/figures/dispatch_peak_comparison.png`
- `outputs/figures/dispatch_week_window.png`
- `outputs/figures/dqn_reward_convergence.png`
- `outputs/figures/dqn_soc_comparison.png`
- `outputs/figures/dqn_total_cost_bars.png`
- `outputs/figures/dqn_peak_grid_import_bars.png`
- `outputs/figures/dqn_grid_import_comparison.png`
- `outputs/figures/dqn_grid_import_week_window.png`

## Notes

- The project computes electricity price by TOU schedule defined in `src/config.py`.
- `g_cap` is derived from a train-load quantile and used in dispatch penalty design.
- DQN can run in RL-only mode (without forecast input) or RL+forecast mode.
- For reproducibility, seeds and major DQN hyperparameters are saved to output metrics JSON files.


