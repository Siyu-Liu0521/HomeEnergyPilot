"""Experience replay buffer for DQN."""

from __future__ import annotations

from typing import Tuple

import numpy as np


class ReplayBuffer:
    """Fixed-capacity circular replay buffer storing transitions."""

    def __init__(self, capacity: int, state_dim: int) -> None:
        self.capacity = int(capacity)
        self.state_dim = int(state_dim)
        self._states = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self._next_states = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self._actions = np.zeros((self.capacity,), dtype=np.int32)
        self._rewards = np.zeros((self.capacity,), dtype=np.float32)
        self._dones = np.zeros((self.capacity,), dtype=np.float32)
        self._pos = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self._states[self._pos] = state
        self._actions[self._pos] = int(action)
        self._rewards[self._pos] = float(reward)
        self._next_states[self._pos] = next_state
        self._dones[self._pos] = 1.0 if done else 0.0
        self._pos = (self._pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        if self._size < batch_size:
            raise ValueError("Not enough transitions to sample.")
        idx = rng.integers(0, self._size, size=batch_size, dtype=np.int64)
        return (
            self._states[idx],
            self._actions[idx],
            self._rewards[idx],
            self._next_states[idx],
            self._dones[idx],
        )
