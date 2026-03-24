"""LSTM forecasting module with two input variants."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout, LSTM

from config import ProjectConfig
from feature_engineering import TIME_FEATURE_COLUMNS
from utils_metrics import forecast_metrics


@dataclass
class LSTMResult:
    """Container for one LSTM experiment result."""

    model_name: str
    history: Dict[str, List[float]]
    pred_df: pd.DataFrame
    metrics: Dict[str, float]
    model_path: str


def _build_model(input_shape: Tuple[int, int], cfg: ProjectConfig) -> tf.keras.Model:
    """Build a compact LSTM model."""
    model = Sequential(name="lstm_forecaster")
    if cfg.lstm_layers == 1:
        model.add(LSTM(cfg.lstm_hidden_size, input_shape=input_shape))
        model.add(Dropout(cfg.lstm_dropout))
    else:
        model.add(LSTM(cfg.lstm_hidden_size, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(cfg.lstm_dropout))
        model.add(LSTM(cfg.lstm_hidden_size))
        model.add(Dropout(cfg.lstm_dropout))

    model.add(Dense(1))
    model.compile(optimizer=cfg.lstm_optimizer, loss=cfg.lstm_loss)
    return model


def _build_full_sequences(
    full_feat_scaled: np.ndarray,
    full_target_scaled: np.ndarray,
    full_index: pd.DatetimeIndex,
    window: int,
    horizon: int,
) -> Tuple[np.ndarray, np.ndarray, List[pd.Timestamp]]:
    """Generate rolling sequences from full scaled arrays."""
    X, y, ts = [], [], []
    max_i = len(full_index) - horizon + 1
    for i in range(window, max_i):
        X.append(full_feat_scaled[i - window : i])
        y.append(full_target_scaled[i + horizon - 1])
        ts.append(full_index[i + horizon - 1])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), ts


def _slice_by_timestamps(
    X: np.ndarray,
    y: np.ndarray,
    timestamps: List[pd.Timestamp],
    target_index: pd.DatetimeIndex,
) -> Tuple[np.ndarray, np.ndarray, List[pd.Timestamp]]:
    """Filter pre-built sequences by target timestamp set."""
    target_set = set(target_index)
    idx = [i for i, ts in enumerate(timestamps) if ts in target_set]
    return X[idx], y[idx], [timestamps[i] for i in idx]


def _prepare_lstm_data(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: ProjectConfig,
    use_time_features: bool,
):
    """Scale data using train-only fit, then create split sequences."""
    features = ["load_kwh"] + TIME_FEATURE_COLUMNS if use_time_features else ["load_kwh"]
    full_df = pd.concat([train_df, val_df, test_df], axis=0)

    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    x_scaler.fit(train_df[features])
    y_scaler.fit(train_df[["load_kwh"]])

    full_x_scaled = x_scaler.transform(full_df[features])
    full_y_scaled = y_scaler.transform(full_df[["load_kwh"]]).reshape(-1)

    X_all, y_all, ts_all = _build_full_sequences(
        full_x_scaled, full_y_scaled, full_df.index, cfg.window_size, cfg.horizon
    )

    X_train, y_train, _ = _slice_by_timestamps(X_all, y_all, ts_all, train_df.index)
    X_val, y_val, _ = _slice_by_timestamps(X_all, y_all, ts_all, val_df.index)
    X_test, y_test, ts_test = _slice_by_timestamps(X_all, y_all, ts_all, test_df.index)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test, ts_test), y_scaler


def _train_one_lstm(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: ProjectConfig,
    model_name: str,
    use_time_features: bool,
    model_filename: str,
) -> LSTMResult:
    """Train and evaluate a single LSTM configuration."""
    tf.random.set_seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)

    (X_train, y_train), (X_val, y_val), (X_test, y_test, ts_test), y_scaler = _prepare_lstm_data(
        train_df, val_df, test_df, cfg, use_time_features=use_time_features
    )
    if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
        raise ValueError(
            f"Insufficient sequences for {model_name}. "
            "Check split sizes or reduce window size."
        )

    model = _build_model((X_train.shape[1], X_train.shape[2]), cfg)
    model_path = str(cfg.models_dir / model_filename)

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=cfg.patience, restore_best_weights=True),
        ModelCheckpoint(model_path, monitor="val_loss", save_best_only=True),
    ]
    hist = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        callbacks=callbacks,
        verbose=0,
    )

    y_pred_scaled = model.predict(X_test, verbose=0).reshape(-1, 1)
    y_test_scaled = y_test.reshape(-1, 1)

    y_pred = y_scaler.inverse_transform(y_pred_scaled).reshape(-1)
    y_true = y_scaler.inverse_transform(y_test_scaled).reshape(-1)

    pred_df = pd.DataFrame({"actual_load": y_true, "pred_load": y_pred}, index=pd.to_datetime(ts_test))
    pred_df.index.name = "timestamp"
    metrics = forecast_metrics(y_true, y_pred)

    return LSTMResult(
        model_name=model_name,
        history={k: list(v) for k, v in hist.history.items()},
        pred_df=pred_df,
        metrics=metrics,
        model_path=model_path,
    )


def make_forecast_sequence(
    pred_df: pd.DataFrame,
    output_path: str | None = None,
) -> pd.DataFrame:
    """
    Build RL-ready forecast sequence from LSTM prediction output.

    Output columns:
    - timestamp
    - load_kwh
    - forecast_load
    """
    seq = pred_df.reset_index().rename(
        columns={"actual_load": "load_kwh", "pred_load": "forecast_load"}
    )
    seq = seq[["timestamp", "load_kwh", "forecast_load"]]
    if output_path:
        seq.to_csv(output_path, index=False)
    return seq


def plot_lstm_training_curve(histories: Dict[str, Dict[str, List[float]]], output_path: str) -> None:
    """Plot train/val loss curves for one or more LSTM runs."""
    plt.figure(figsize=(10, 5))
    for name, h in histories.items():
        if "loss" in h:
            plt.plot(h["loss"], label=f"{name} train")
        if "val_loss" in h:
            plt.plot(h["val_loss"], label=f"{name} val")
    plt.title("LSTM Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def run_lstm_experiments(
    cfg: ProjectConfig,
    train_feat: pd.DataFrame,
    val_feat: pd.DataFrame,
    test_feat: pd.DataFrame,
) -> Dict[str, LSTMResult]:
    """Run both LSTM experiments and save artifacts."""
    cfg.ensure_directories()

    run_a = _train_one_lstm(
        train_feat,
        val_feat,
        test_feat,
        cfg,
        model_name="LSTM_load_only",
        use_time_features=False,
        model_filename="lstm_load_only.keras",
    )
    run_b = _train_one_lstm(
        train_feat,
        val_feat,
        test_feat,
        cfg,
        model_name="LSTM_with_time",
        use_time_features=True,
        model_filename="lstm_with_time.keras",
    )

    run_a.pred_df.to_csv(cfg.predictions_dir / "lstm_load_only_test_pred.csv")
    run_b.pred_df.to_csv(cfg.predictions_dir / "lstm_with_time_test_pred.csv")

    metrics_df = pd.DataFrame(
        [
            {"model": run_a.model_name, **run_a.metrics},
            {"model": run_b.model_name, **run_b.metrics},
        ]
    )
    metrics_df.to_csv(cfg.metrics_dir / "lstm_metrics.csv", index=False)

    plot_lstm_training_curve(
        {
            run_a.model_name: run_a.history,
            run_b.model_name: run_b.history,
        },
        str(cfg.figures_dir / "lstm_training_curve.png"),
    )

    # Default RL forecast source: richer model with time features.
    make_forecast_sequence(
        run_b.pred_df,
        output_path=str(cfg.predictions_dir / "test_forecast_sequence.csv"),
    )

    return {run_a.model_name: run_a, run_b.model_name: run_b}

