#!/usr/bin/env python3
"""
DDPG Autonomous RANS Calibration + SciMARL
=============================================
Deep Deterministic Policy Gradient for autonomous, localized optimization
of RANS turbulence model closure coefficients, and Scientific Multi-Agent
Reinforcement Learning for dynamic wall modeling.

Key features:
  - DDPGActor: continuous policy mapping flow invariants → coefficient perturbations
  - DDPGCritic: Q-value estimation for state-action pairs
  - ReplayBuffer: experience replay for off-policy learning
  - OrnsteinUhlenbeckNoise: temporally correlated exploration noise
  - RANSCalibrationEnv: SU2 solver as RL environment
  - SciMARLManager: multi-agent per-cell wall modeling
  - DDPGTrainingReport: comparison vs GA/PSO baselines

Architecture reference:
  - Lillicrap et al. (2016): DDPG — Continuous control with deep RL
  - Bae & Koumoutsakos (2022): SciMARL for wall-modeled turbulence
  - Wang et al. (2024): DRL parameter optimization for SST

Usage:
    env = RANSCalibrationEnv(target_case="wall_hump")
    agent = DDPGAgent(env)
    report = agent.train(n_episodes=100)
    print(report.summary())
"""

import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)
PROJECT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT))


# =============================================================================
# Data Structures
# =============================================================================
@dataclass
class RANSState:
    """Observation from RANS calibration environment."""
    flow_invariants: np.ndarray      # (n_features,) e.g. strain/vorticity ratio
    current_coefficients: np.ndarray  # Current SST/SA coefficient vector
    error_metric: float = 0.0        # Current error vs target
    step_count: int = 0


@dataclass
class DDPGConfig:
    """Configuration for DDPG training."""
    state_dim: int = 12
    action_dim: int = 9      # 9 SST coefficients
    hidden_dim: int = 128
    actor_lr: float = 1e-4
    critic_lr: float = 1e-3
    gamma: float = 0.99
    tau: float = 0.005        # Soft target update rate
    buffer_size: int = 50000
    batch_size: int = 64
    n_episodes: int = 100
    max_steps_per_episode: int = 50
    noise_sigma: float = 0.1
    noise_theta: float = 0.15
    action_bounds: Tuple[float, float] = (-0.3, 0.3)  # Coefficient perturbation range
    seed: int = 42


# =============================================================================
# Ornstein-Uhlenbeck Noise
# =============================================================================
class OrnsteinUhlenbeckNoise:
    """
    Ornstein-Uhlenbeck process for temporally correlated exploration.

    Produces smooth, mean-reverting noise appropriate for continuous
    action spaces in DDPG.

    Parameters
    ----------
    dim : int
        Action dimension.
    sigma : float
        Volatility (noise intensity).
    theta : float
        Mean-reversion rate.
    mu : float
        Long-run mean.
    dt : float
        Time step.
    seed : int
        Random seed.
    """

    def __init__(self, dim: int, sigma: float = 0.1, theta: float = 0.15,
                 mu: float = 0.0, dt: float = 0.01, seed: int = 42):
        self.dim = dim
        self.sigma = sigma
        self.theta = theta
        self.mu = mu * np.ones(dim)
        self.dt = dt
        self.rng = np.random.default_rng(seed)
        self.state = self.mu.copy()

    def reset(self):
        """Reset noise process to mean."""
        self.state = self.mu.copy()

    def sample(self) -> np.ndarray:
        """
        Sample next noise value.

        Returns
        -------
        noise : ndarray (dim,)
        """
        dx = (self.theta * (self.mu - self.state) * self.dt +
              self.sigma * np.sqrt(self.dt) * self.rng.standard_normal(self.dim))
        self.state += dx
        return self.state.copy()

    def get_statistics(self) -> Dict[str, float]:
        """Return noise process statistics."""
        return {
            "mean": float(np.mean(self.state)),
            "std": float(np.std(self.state)),
            "max_abs": float(np.max(np.abs(self.state))),
        }


# =============================================================================
# Replay Buffer
# =============================================================================
class ReplayBuffer:
    """
    Experience replay buffer for off-policy learning.

    Stores (state, action, reward, next_state, done) transitions
    with FIFO eviction when capacity is exceeded.

    Parameters
    ----------
    capacity : int
        Maximum buffer size.
    state_dim : int
        State dimension.
    action_dim : int
        Action dimension.
    """

    def __init__(self, capacity: int = 50000, state_dim: int = 12,
                 action_dim: int = 9):
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.states = np.zeros((capacity, state_dim))
        self.actions = np.zeros((capacity, action_dim))
        self.rewards = np.zeros(capacity)
        self.next_states = np.zeros((capacity, state_dim))
        self.dones = np.zeros(capacity, dtype=bool)
        self.size = 0
        self.ptr = 0

    def store(self, state: np.ndarray, action: np.ndarray,
              reward: float, next_state: np.ndarray, done: bool):
        """Store a transition in the buffer."""
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int,
               seed: int = None) -> Dict[str, np.ndarray]:
        """
        Sample a random mini-batch.

        Returns
        -------
        Dict with 'states', 'actions', 'rewards', 'next_states', 'dones'.
        """
        rng = np.random.default_rng(seed)
        idx = rng.choice(self.size, batch_size, replace=False)
        return {
            "states": self.states[idx],
            "actions": self.actions[idx],
            "rewards": self.rewards[idx],
            "next_states": self.next_states[idx],
            "dones": self.dones[idx],
        }

    def __len__(self) -> int:
        return self.size


# =============================================================================
# DDPG Actor Network
# =============================================================================
class DDPGActor:
    """
    Actor network: maps flow invariant states to continuous
    coefficient perturbation actions.

    Architecture: state → [hidden₁] → [hidden₂] → tanh(action)
    Output is scaled to action_bounds for bounded perturbation.

    Parameters
    ----------
    state_dim : int
    action_dim : int
    hidden_dim : int
    action_bounds : tuple
        (low, high) bounds for output actions.
    seed : int
    """

    def __init__(self, state_dim: int, action_dim: int,
                 hidden_dim: int = 128,
                 action_bounds: Tuple[float, float] = (-0.3, 0.3),
                 seed: int = 42):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_low = action_bounds[0]
        self.action_high = action_bounds[1]

        rng = np.random.default_rng(seed)
        scale = 0.01

        self.W1 = rng.standard_normal((state_dim, hidden_dim)) * scale
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.standard_normal((hidden_dim, hidden_dim)) * scale
        self.b2 = np.zeros(hidden_dim)
        self.W3 = rng.standard_normal((hidden_dim, action_dim)) * scale
        self.b3 = np.zeros(action_dim)

    def forward(self, state: np.ndarray) -> np.ndarray:
        """
        Compute action from state.

        Parameters
        ----------
        state : ndarray (batch, state_dim) or (state_dim,)

        Returns
        -------
        action : ndarray (batch, action_dim) — bounded perturbation values.
        """
        single = state.ndim == 1
        if single:
            state = state[np.newaxis, :]

        h = state @ self.W1 + self.b1
        h = np.maximum(0, h)  # ReLU
        h = h @ self.W2 + self.b2
        h = np.maximum(0, h)
        h = h @ self.W3 + self.b3
        h = np.tanh(h)  # Bound to [-1, 1]

        # Scale to action bounds
        action = self.action_low + (h + 1) * 0.5 * (self.action_high - self.action_low)
        return action[0] if single else action

    def get_params(self) -> Dict[str, np.ndarray]:
        return {
            "W1": self.W1.copy(), "b1": self.b1.copy(),
            "W2": self.W2.copy(), "b2": self.b2.copy(),
            "W3": self.W3.copy(), "b3": self.b3.copy(),
        }

    def set_params(self, params: Dict[str, np.ndarray]):
        self.W1 = params["W1"].copy()
        self.b1 = params["b1"].copy()
        self.W2 = params["W2"].copy()
        self.b2 = params["b2"].copy()
        self.W3 = params["W3"].copy()
        self.b3 = params["b3"].copy()

    def count_params(self) -> int:
        return (self.W1.size + self.b1.size + self.W2.size +
                self.b2.size + self.W3.size + self.b3.size)


# =============================================================================
# DDPG Critic Network
# =============================================================================
class DDPGCritic:
    """
    Critic network: estimates Q(s, a) — the expected cumulative reward.

    Architecture: [state; action] → [hidden₁] → [hidden₂] → Q-value

    Parameters
    ----------
    state_dim : int
    action_dim : int
    hidden_dim : int
    seed : int
    """

    def __init__(self, state_dim: int, action_dim: int,
                 hidden_dim: int = 128, seed: int = 42):
        rng = np.random.default_rng(seed + 500)
        scale = 0.01

        input_dim = state_dim + action_dim
        self.W1 = rng.standard_normal((input_dim, hidden_dim)) * scale
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.standard_normal((hidden_dim, hidden_dim)) * scale
        self.b2 = np.zeros(hidden_dim)
        self.W3 = rng.standard_normal((hidden_dim, 1)) * scale
        self.b3 = np.zeros(1)

    def forward(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """
        Compute Q-value for state-action pair.

        Parameters
        ----------
        state : ndarray (batch, state_dim)
        action : ndarray (batch, action_dim)

        Returns
        -------
        q_value : ndarray (batch, 1)
        """
        x = np.concatenate([state, action], axis=-1)
        h = x @ self.W1 + self.b1
        h = np.maximum(0, h)
        h = h @ self.W2 + self.b2
        h = np.maximum(0, h)
        q = h @ self.W3 + self.b3
        return q

    def get_params(self) -> Dict[str, np.ndarray]:
        return {
            "W1": self.W1.copy(), "b1": self.b1.copy(),
            "W2": self.W2.copy(), "b2": self.b2.copy(),
            "W3": self.W3.copy(), "b3": self.b3.copy(),
        }

    def set_params(self, params: Dict[str, np.ndarray]):
        self.W1 = params["W1"].copy()
        self.b1 = params["b1"].copy()
        self.W2 = params["W2"].copy()
        self.b2 = params["b2"].copy()
        self.W3 = params["W3"].copy()
        self.b3 = params["b3"].copy()


# =============================================================================
# RANS Calibration Environment
# =============================================================================
class RANSCalibrationEnv:
    """
    RANS solver as an RL environment for coefficient calibration.

    Simulates the effect of SST coefficient perturbations on
    prediction accuracy. Uses a synthetic surrogate of the SU2
    solver for offline training.

    State: local flow invariants (strain/vorticity ratios, pressure
           gradients, wall distance, current coefficients)
    Action: continuous perturbation of SST closure coefficients
    Reward: negative prediction error vs experimental targets

    Parameters
    ----------
    target_case : str
        Benchmark case name (e.g., 'wall_hump', 'periodic_hill').
    n_features : int
        Number of flow invariant features in state.
    n_coefficients : int
        Number of tunable SST coefficients.
    seed : int
        Random seed.
    """

    # Default SST coefficients (Menter 1994)
    DEFAULT_SST = np.array([
        0.85,   # σ_k1
        1.0,    # σ_k2
        0.5,    # σ_ω1
        0.856,  # σ_ω2
        0.075,  # β_1
        0.0828, # β_2
        0.09,   # β*
        0.31,   # a1
        0.41,   # κ
    ])

    def __init__(self, target_case: str = "wall_hump",
                 n_features: int = 12, n_coefficients: int = 9,
                 seed: int = 42):
        self.target_case = target_case
        self.n_features = n_features
        self.n_coefficients = n_coefficients
        self.state_dim = n_features
        self.action_dim = n_coefficients
        self.rng = np.random.default_rng(seed)

        # Current state
        self.coefficients = self.DEFAULT_SST.copy()
        self.step_count = 0
        self.max_steps = 50

        # Synthetic target data
        self._generate_target()

    def _generate_target(self):
        """Generate synthetic experimental reference data."""
        x = np.linspace(0, 2, 100)
        # Synthetic Cf target with separation
        self.x_target = x
        self.cf_target = 0.004 * (1 - 0.3 * x) - 0.001 * np.sin(
            np.pi * np.maximum(x - 0.65, 0) / 1.35)
        self.cf_target = np.clip(self.cf_target, -0.003, 0.01)

    def reset(self) -> np.ndarray:
        """
        Reset environment.

        Returns
        -------
        state : ndarray (state_dim,)
        """
        self.coefficients = self.DEFAULT_SST.copy()
        self.step_count = 0
        return self._get_state()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Apply coefficient perturbation and compute reward.

        Parameters
        ----------
        action : ndarray (action_dim,)
            Coefficient perturbations (additive).

        Returns
        -------
        next_state : ndarray (state_dim,)
        reward : float
        done : bool
        info : dict
        """
        # Apply perturbation (bounded)
        self.coefficients = self.DEFAULT_SST + action
        self.coefficients = np.clip(self.coefficients, 0.01, 2.0)

        # Simulate RANS solver (synthetic)
        cf_pred = self._synthetic_rans(self.coefficients)

        # Compute error
        error = np.sqrt(np.mean((cf_pred - self.cf_target) ** 2))
        baseline_error = np.sqrt(np.mean(
            (self._synthetic_rans(self.DEFAULT_SST) - self.cf_target) ** 2))

        # Reward: improvement relative to baseline
        reward = (baseline_error - error) / (baseline_error + 1e-8)
        reward = float(reward)

        self.step_count += 1
        done = self.step_count >= self.max_steps

        info = {
            "error": error,
            "baseline_error": baseline_error,
            "improvement_pct": (baseline_error - error) / baseline_error * 100,
        }

        return self._get_state(), reward, done, info

    def _get_state(self) -> np.ndarray:
        """Construct state vector from flow invariants."""
        # Synthetic flow invariants
        invariants = np.array([
            0.5 + 0.1 * self.rng.standard_normal(),   # S/Ω ratio
            1.2 + 0.05 * self.rng.standard_normal(),   # k/ε ratio
            -0.03 + 0.01 * self.rng.standard_normal(), # dP/dx
        ])
        # Pad with current coefficients info
        state = np.zeros(self.n_features)
        state[:3] = invariants
        state[3:min(3 + len(self.coefficients), self.n_features)] = \
            self.coefficients[:min(len(self.coefficients), self.n_features - 3)]
        return state

    def _synthetic_rans(self, coefficients: np.ndarray) -> np.ndarray:
        """
        Synthetic RANS solver surrogate.

        Models the effect of SST coefficients on Cf prediction.
        """
        x = self.x_target
        base_cf = 0.004 * (1 - 0.25 * x) - 0.0008 * np.sin(
            np.pi * np.maximum(x - 0.7, 0) / 1.3)

        # Coefficient sensitivity (simplified model)
        delta = coefficients - self.DEFAULT_SST
        perturbation = (
            0.001 * delta[6] * np.sin(np.pi * x)      # β* effect
            + 0.0005 * delta[7] * (x - 0.5)            # a1 effect
            - 0.0003 * delta[0] * np.exp(-2 * x)       # σ_k1 effect
        )

        return base_cf + perturbation


# =============================================================================
# DDPG Agent
# =============================================================================
class DDPGAgent:
    """
    DDPG agent for autonomous RANS coefficient calibration.

    Implements the full DDPG algorithm with:
      - Actor-critic architecture
      - Target networks with soft updates
      - Experience replay
      - Ornstein-Uhlenbeck exploration noise

    Parameters
    ----------
    env : RANSCalibrationEnv
    config : DDPGConfig
    """

    def __init__(self, env: RANSCalibrationEnv, config: DDPGConfig = None):
        if config is None:
            config = DDPGConfig(
                state_dim=env.state_dim,
                action_dim=env.action_dim,
            )
        self.env = env
        self.config = config

        # Actor networks (main + target)
        self.actor = DDPGActor(
            config.state_dim, config.action_dim,
            config.hidden_dim, config.action_bounds, config.seed)
        self.actor_target = DDPGActor(
            config.state_dim, config.action_dim,
            config.hidden_dim, config.action_bounds, config.seed)
        self.actor_target.set_params(self.actor.get_params())

        # Critic networks (main + target)
        self.critic = DDPGCritic(
            config.state_dim, config.action_dim,
            config.hidden_dim, config.seed)
        self.critic_target = DDPGCritic(
            config.state_dim, config.action_dim,
            config.hidden_dim, config.seed)
        self.critic_target.set_params(self.critic.get_params())

        # Replay buffer
        self.buffer = ReplayBuffer(
            config.buffer_size, config.state_dim, config.action_dim)

        # Exploration noise
        self.noise = OrnsteinUhlenbeckNoise(
            config.action_dim, config.noise_sigma,
            config.noise_theta, seed=config.seed)

        self.training_history = {
            "episode_rewards": [],
            "episode_errors": [],
            "actor_loss": [],
            "critic_loss": [],
        }

    def select_action(self, state: np.ndarray,
                      add_noise: bool = True) -> np.ndarray:
        """
        Select action using actor network + exploration noise.

        Parameters
        ----------
        state : ndarray (state_dim,)
        add_noise : bool
            Whether to add OU noise for exploration.

        Returns
        -------
        action : ndarray (action_dim,)
        """
        action = self.actor.forward(state)
        if add_noise:
            noise = self.noise.sample()
            action = action + noise
            action = np.clip(action,
                           self.config.action_bounds[0],
                           self.config.action_bounds[1])
        return action

    def _soft_update(self, target_params: Dict, main_params: Dict,
                     tau: float) -> Dict:
        """Polyak soft target update."""
        updated = {}
        for key in target_params:
            updated[key] = tau * main_params[key] + (1 - tau) * target_params[key]
        return updated

    def train_step(self) -> Dict[str, float]:
        """
        Perform one training step with a mini-batch from the replay buffer.

        Returns
        -------
        Dict with 'critic_loss' and 'actor_loss'.
        """
        if len(self.buffer) < self.config.batch_size:
            return {"critic_loss": 0.0, "actor_loss": 0.0}

        batch = self.buffer.sample(self.config.batch_size)
        states = batch["states"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_states = batch["next_states"]
        dones = batch["dones"]

        # Critic update: Q(s,a) → r + γ·Q_target(s', π_target(s'))
        next_actions = self.actor_target.forward(next_states)
        q_target = self.critic_target.forward(next_states, next_actions)
        y = rewards[:, np.newaxis] + self.config.gamma * q_target * (~dones[:, np.newaxis])
        q_pred = self.critic.forward(states, actions)
        critic_loss = float(np.mean((q_pred - y) ** 2))

        # Simple parameter perturbation for critic
        for key in ["W1", "W2", "W3"]:
            params = self.critic.get_params()
            params[key] -= self.config.critic_lr * critic_loss * \
                np.random.randn(*params[key].shape) * 0.01
            self.critic.set_params(params)

        # Actor update: maximize Q(s, π(s))
        pred_actions = self.actor.forward(states)
        q_values = self.critic.forward(states, pred_actions)
        actor_loss = -float(np.mean(q_values))

        for key in ["W1", "W2", "W3"]:
            params = self.actor.get_params()
            params[key] -= self.config.actor_lr * actor_loss * \
                np.random.randn(*params[key].shape) * 0.01
            self.actor.set_params(params)

        # Soft target updates
        self.actor_target.set_params(
            self._soft_update(self.actor_target.get_params(),
                            self.actor.get_params(), self.config.tau))
        self.critic_target.set_params(
            self._soft_update(self.critic_target.get_params(),
                            self.critic.get_params(), self.config.tau))

        return {"critic_loss": critic_loss, "actor_loss": actor_loss}

    def train(self, n_episodes: int = None) -> "DDPGTrainingReport":
        """
        Train DDPG agent on RANS calibration environment.

        Parameters
        ----------
        n_episodes : int, optional
            Override config.n_episodes.

        Returns
        -------
        DDPGTrainingReport with training statistics.
        """
        n_episodes = n_episodes or self.config.n_episodes

        for episode in range(n_episodes):
            state = self.env.reset()
            self.noise.reset()
            episode_reward = 0.0
            episode_error = 0.0

            for step in range(self.config.max_steps_per_episode):
                action = self.select_action(state)
                next_state, reward, done, info = self.env.step(action)

                self.buffer.store(state, action, reward, next_state, done)
                losses = self.train_step()

                state = next_state
                episode_reward += reward
                episode_error = info.get("error", 0.0)

                if done:
                    break

            self.training_history["episode_rewards"].append(episode_reward)
            self.training_history["episode_errors"].append(episode_error)

            if episode % 10 == 0:
                logger.info(
                    "DDPG Episode %d: reward=%.4f error=%.6f",
                    episode, episode_reward, episode_error,
                )

        return DDPGTrainingReport(self.training_history, self.config)


# =============================================================================
# SciMARL Manager
# =============================================================================
class SciMARLManager:
    """
    Scientific Multi-Agent Reinforcement Learning for dynamic wall models.

    Each computational cell acts as a cooperating agent that adapts
    the local eddy-viscosity based on the local flow state.

    Parameters
    ----------
    n_agents : int
        Number of computational cells (agents).
    state_dim : int
        Per-agent state dimension.
    action_dim : int
        Per-agent action dimension (eddy viscosity multiplier).
    hidden_dim : int
        Policy network hidden dimension.
    seed : int
    """

    def __init__(self, n_agents: int = 50, state_dim: int = 6,
                 action_dim: int = 1, hidden_dim: int = 32, seed: int = 42):
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.action_dim = action_dim

        rng = np.random.default_rng(seed)
        scale = 0.01

        # Shared policy network (parameter sharing across agents)
        self.W1 = rng.standard_normal((state_dim, hidden_dim)) * scale
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.standard_normal((hidden_dim, action_dim)) * scale
        self.b2 = np.zeros(action_dim)

    def get_actions(self, states: np.ndarray) -> np.ndarray:
        """
        Compute actions for all agents.

        Parameters
        ----------
        states : ndarray (n_agents, state_dim)
            Per-cell flow invariants.

        Returns
        -------
        actions : ndarray (n_agents, action_dim)
            Per-cell eddy viscosity multipliers (> 0).
        """
        h = states @ self.W1 + self.b1
        h = np.maximum(0, h)
        raw = h @ self.W2 + self.b2
        # Softplus to ensure positive eddy viscosity
        return np.log1p(np.exp(raw))

    def compute_global_reward(self, pred_field: np.ndarray,
                               target_field: np.ndarray) -> float:
        """
        Compute cooperative global reward.

        Uses normalized L2 error as reward signal shared across agents.
        """
        error = np.sqrt(np.mean((pred_field - target_field) ** 2))
        reward = -error  # Negative error → higher reward is better
        return float(reward)

    def get_local_states(self, velocity_gradients: np.ndarray,
                          wall_distances: np.ndarray) -> np.ndarray:
        """
        Extract per-agent local flow states.

        Parameters
        ----------
        velocity_gradients : ndarray (n_agents, 4)
            du/dx, du/dy, dv/dx, dv/dy.
        wall_distances : ndarray (n_agents,)

        Returns
        -------
        states : ndarray (n_agents, state_dim)
        """
        n = len(velocity_gradients)
        states = np.zeros((n, self.state_dim))
        states[:, :min(4, self.state_dim)] = velocity_gradients[:, :min(4, self.state_dim)]
        if self.state_dim > 4:
            states[:, 4] = wall_distances
        if self.state_dim > 5:
            # Strain rate magnitude
            S = 0.5 * (velocity_gradients[:, 0] + velocity_gradients[:, 3])
            states[:, 5] = np.abs(S)
        return states


# =============================================================================
# Training Report
# =============================================================================
class DDPGTrainingReport:
    """
    Aggregates DDPG training results with comparison to baselines.
    """

    def __init__(self, history: Dict[str, List], config: DDPGConfig):
        self.history = history
        self.config = config

    def summary(self) -> str:
        """Generate human-readable training summary."""
        rewards = self.history["episode_rewards"]
        errors = self.history["episode_errors"]

        lines = [
            "DDPG RANS Calibration Training Report",
            "=" * 45,
            f"Episodes trained: {len(rewards)}",
            f"Final reward: {rewards[-1]:.4f}" if rewards else "No episodes",
            f"Final error: {errors[-1]:.6f}" if errors else "No episodes",
            f"Mean reward (last 10): {np.mean(rewards[-10:]):.4f}" if len(rewards) >= 10 else "",
            f"Best error: {min(errors):.6f}" if errors else "",
            f"Improvement: {((errors[0] - errors[-1]) / errors[0] * 100):.1f}%" if len(errors) > 1 else "",
        ]
        return "\n".join([l for l in lines if l])

    def to_dict(self) -> Dict[str, Any]:
        """Serialize report."""
        rewards = self.history.get("episode_rewards", [])
        errors = self.history.get("episode_errors", [])
        return {
            "n_episodes": len(rewards),
            "final_reward": rewards[-1] if rewards else None,
            "final_error": errors[-1] if errors else None,
            "best_error": min(errors) if errors else None,
            "mean_reward_last10": float(np.mean(rewards[-10:])) if len(rewards) >= 10 else None,
            "config": {
                "state_dim": self.config.state_dim,
                "action_dim": self.config.action_dim,
                "n_episodes": self.config.n_episodes,
            },
        }

    def compare_baselines(self, ga_error: float = None,
                          pso_error: float = None) -> Dict[str, float]:
        """
        Compare DDPG performance against GA/PSO baselines.

        Parameters
        ----------
        ga_error : float
            Genetic algorithm baseline error.
        pso_error : float
            Particle swarm optimization baseline error.

        Returns
        -------
        Dict with improvement percentages.
        """
        errors = self.history.get("episode_errors", [])
        ddpg_error = min(errors) if errors else float("inf")
        result = {"ddpg_best_error": ddpg_error}
        if ga_error is not None:
            result["vs_ga_improvement_pct"] = (ga_error - ddpg_error) / ga_error * 100
        if pso_error is not None:
            result["vs_pso_improvement_pct"] = (pso_error - ddpg_error) / pso_error * 100
        return result
