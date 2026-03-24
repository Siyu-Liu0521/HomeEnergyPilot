"""DQN agent with target network and epsilon-greedy exploration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import tensorflow as tf

from dqn_q_network import build_q_network, clone_q_network
from replay_buffer import ReplayBuffer


@dataclass
class DQNHyperParams:
    gamma: float = 0.99
    learning_rate: float = 1e-3
    replay_capacity: int = 200_000
    batch_size: int = 128
    min_replay_size: int = 5_000
    target_update_every: int = 500
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 100_000
    hidden_units: Tuple[int, ...] = (128, 128)
    gradient_clip: float = 10.0
    use_double_dqn: bool = True


class DQNAgent:
    """DQN with online Q, target Q, replay buffer, epsilon-greedy, and optional Double DQN targets."""

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hp: DQNHyperParams,
        seed: int = 42,
    ) -> None:
        self.state_dim = int(state_dim)
        self.n_actions = int(n_actions)
        self.hp = hp
        self.rng = np.random.default_rng(seed)
        tf.random.set_seed(seed)

        self.online_q = build_q_network(
            self.state_dim,
            self.n_actions,
            hidden_units=hp.hidden_units,
            name="q_online",
        )
        self.target_q = clone_q_network(self.online_q)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=hp.learning_rate)
        self.buffer = ReplayBuffer(hp.replay_capacity, self.state_dim)
        self.train_step_count = 0

    def epsilon_by_step(self, global_step: int) -> float:
        """Linear decay from epsilon_start to epsilon_end."""
        t = min(1.0, float(global_step) / max(1, self.hp.epsilon_decay_steps))
        return self.hp.epsilon_start + t * (self.hp.epsilon_end - self.hp.epsilon_start)

    def select_action(self, state: np.ndarray, epsilon: float) -> int:
        """Epsilon-greedy action selection."""
        if self.rng.random() < epsilon:
            return int(self.rng.integers(0, self.n_actions))
        q = self.online_q(state[None, :], training=False)
        return int(np.argmax(q.numpy()[0]))

    def soft_update_target(self) -> None:
        """Hard-copy weights from online Q to target Q."""
        self.target_q.set_weights(self.online_q.get_weights())

    def train_from_buffer(self) -> float:
        """Sample one batch and perform one gradient step."""
        s, a, r, s_next, done = self.buffer.sample(self.hp.batch_size, self.rng)
        gamma = float(self.hp.gamma)
        states = tf.constant(s, dtype=tf.float32)
        actions = tf.constant(a, dtype=tf.int32)
        rewards = tf.constant(r, dtype=tf.float32)
        next_states = tf.constant(s_next, dtype=tf.float32)
        dones = tf.constant(done, dtype=tf.float32)
        with tf.GradientTape() as tape:
            q_online = self.online_q(states, training=True)
            idx = tf.stack([tf.range(tf.shape(actions)[0]), actions], axis=1)
            q_sa = tf.gather_nd(q_online, idx)
            if self.hp.use_double_dqn:
                q_online_next = self.online_q(next_states, training=False)
                best_a = tf.cast(tf.argmax(q_online_next, axis=1), tf.int32)
                bi = tf.range(tf.shape(best_a)[0], dtype=tf.int32)
                idx_next = tf.stack([bi, best_a], axis=1)
                q_next = tf.gather_nd(self.target_q(next_states, training=False), idx_next)
            else:
                q_next = tf.reduce_max(self.target_q(next_states, training=False), axis=1)
            targets = rewards + (1.0 - dones) * gamma * q_next
            loss = tf.reduce_mean(tf.square(q_sa - targets))
        grads = tape.gradient(loss, self.online_q.trainable_variables)
        if self.hp.gradient_clip > 0:
            grads, _ = tf.clip_by_global_norm(grads, self.hp.gradient_clip)
        self.optimizer.apply_gradients(zip(grads, self.online_q.trainable_variables))
        self.train_step_count += 1
        if self.train_step_count % self.hp.target_update_every == 0:
            self.soft_update_target()
        return float(loss.numpy())
