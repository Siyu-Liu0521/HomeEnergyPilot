"""Q-network builders for DQN."""

from __future__ import annotations

from typing import Sequence

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, InputLayer


def build_q_network(
    state_dim: int,
    n_actions: int = 3,
    hidden_units: Sequence[int] = (128, 128),
    name: str = "q_online",
) -> tf.keras.Model:
    """Build a fully-connected Q-network with linear output heads per action."""
    layers = [InputLayer(input_shape=(state_dim,), name=f"{name}_in")]
    for i, h in enumerate(hidden_units):
        layers.append(Dense(int(h), activation="relu", name=f"{name}_h{i}"))
    layers.append(Dense(int(n_actions), activation="linear", name=f"{name}_out"))
    model = Sequential(layers, name=name)
    return model


def clone_q_network(online: tf.keras.Model) -> tf.keras.Model:
    """Duplicate architecture and weights for target network initialization."""
    target = tf.keras.models.clone_model(online)
    target.set_weights(online.get_weights())
    return target
