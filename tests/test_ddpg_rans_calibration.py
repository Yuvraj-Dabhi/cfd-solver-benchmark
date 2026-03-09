#!/usr/bin/env python3
"""
Tests for DDPG RANS Calibration Module
==========================================
Validates actor/critic networks, replay buffer, noise process,
environment, DDPG agent training, and SciMARL multi-agent policies.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ml_augmentation.ddpg_rans_calibration import (
    DDPGActor,
    DDPGCritic,
    DDPGAgent,
    DDPGConfig,
    DDPGTrainingReport,
    OrnsteinUhlenbeckNoise,
    RANSCalibrationEnv,
    ReplayBuffer,
    SciMARLManager,
)


# =========================================================================
# TestOrnsteinUhlenbeckNoise
# =========================================================================
class TestOrnsteinUhlenbeckNoise:
    """Tests for OU exploration noise."""

    def test_sample_shape(self):
        noise = OrnsteinUhlenbeckNoise(dim=9)
        sample = noise.sample()
        assert sample.shape == (9,)

    def test_mean_reverting(self):
        """Over many samples, mean should stay near mu=0."""
        noise = OrnsteinUhlenbeckNoise(dim=5, sigma=0.1, theta=0.5)
        samples = [noise.sample() for _ in range(1000)]
        mean = np.mean(samples, axis=0)
        assert np.all(np.abs(mean) < 0.5)

    def test_reset(self):
        noise = OrnsteinUhlenbeckNoise(dim=4)
        for _ in range(10):
            noise.sample()
        noise.reset()
        assert np.allclose(noise.state, 0.0)

    def test_statistics(self):
        noise = OrnsteinUhlenbeckNoise(dim=3)
        noise.sample()
        stats = noise.get_statistics()
        assert "mean" in stats
        assert "std" in stats
        assert "max_abs" in stats


# =========================================================================
# TestReplayBuffer
# =========================================================================
class TestReplayBuffer:
    """Tests for experience replay buffer."""

    def test_store_and_size(self):
        buf = ReplayBuffer(capacity=100, state_dim=4, action_dim=2)
        for i in range(10):
            buf.store(np.zeros(4), np.zeros(2), 1.0, np.zeros(4), False)
        assert len(buf) == 10

    def test_fifo_eviction(self):
        """Buffer should evict oldest entries when full."""
        buf = ReplayBuffer(capacity=5, state_dim=2, action_dim=1)
        for i in range(10):
            buf.store(np.array([i, i]), np.array([i]),
                      float(i), np.array([i, i]), False)
        assert len(buf) == 5
        # First entry should have been evicted
        assert buf.states[0, 0] == 5.0  # Index 0 has the 6th entry (overwritten)

    def test_sample_shape(self):
        buf = ReplayBuffer(capacity=100, state_dim=4, action_dim=2)
        for i in range(20):
            buf.store(np.random.randn(4), np.random.randn(2),
                      1.0, np.random.randn(4), False)
        batch = buf.sample(8)
        assert batch["states"].shape == (8, 4)
        assert batch["actions"].shape == (8, 2)
        assert batch["rewards"].shape == (8,)
        assert batch["dones"].shape == (8,)


# =========================================================================
# TestDDPGActor
# =========================================================================
class TestDDPGActor:
    """Tests for DDPG actor network."""

    def test_output_shape(self):
        actor = DDPGActor(state_dim=12, action_dim=9)
        state = np.random.randn(12)
        action = actor.forward(state)
        assert action.shape == (9,)

    def test_batch_output_shape(self):
        actor = DDPGActor(state_dim=12, action_dim=9)
        states = np.random.randn(8, 12)
        actions = actor.forward(states)
        assert actions.shape == (8, 9)

    def test_output_bounded(self):
        """Actions should be within specified bounds."""
        actor = DDPGActor(state_dim=6, action_dim=4,
                          action_bounds=(-0.5, 0.5))
        for _ in range(10):
            state = np.random.randn(6) * 5  # Large inputs
            action = actor.forward(state)
            assert np.all(action >= -0.5)
            assert np.all(action <= 0.5)

    def test_param_count(self):
        actor = DDPGActor(state_dim=12, action_dim=9, hidden_dim=128)
        assert actor.count_params() > 0

    def test_get_set_params(self):
        actor = DDPGActor(state_dim=8, action_dim=4)
        params = actor.get_params()
        state = np.random.randn(8)
        out1 = actor.forward(state)
        # Perturb
        for k in params:
            params[k] += 0.1
        actor.set_params(params)
        out2 = actor.forward(state)
        assert not np.allclose(out1, out2)


# =========================================================================
# TestDDPGCritic
# =========================================================================
class TestDDPGCritic:
    """Tests for DDPG critic network."""

    def test_output_shape(self):
        critic = DDPGCritic(state_dim=12, action_dim=9)
        s = np.random.randn(4, 12)
        a = np.random.randn(4, 9)
        q = critic.forward(s, a)
        assert q.shape == (4, 1)

    def test_different_actions_different_q(self):
        critic = DDPGCritic(state_dim=6, action_dim=3, seed=7)
        s = np.random.randn(2, 6)
        a1 = np.ones((2, 3))
        a2 = -np.ones((2, 3))
        q1 = critic.forward(s, a1)
        q2 = critic.forward(s, a2)
        assert not np.allclose(q1, q2)


# =========================================================================
# TestRANSCalibrationEnv
# =========================================================================
class TestRANSCalibrationEnv:
    """Tests for RANS calibration environment."""

    def test_reset(self):
        env = RANSCalibrationEnv()
        state = env.reset()
        assert state.shape == (env.state_dim,)
        assert np.all(np.isfinite(state))

    def test_step(self):
        env = RANSCalibrationEnv()
        env.reset()
        action = np.zeros(env.action_dim)
        next_state, reward, done, info = env.step(action)
        assert next_state.shape == (env.state_dim,)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert "error" in info

    def test_episode_terminates(self):
        env = RANSCalibrationEnv()
        env.reset()
        done = False
        steps = 0
        while not done:
            action = np.zeros(env.action_dim)
            _, _, done, _ = env.step(action)
            steps += 1
        assert steps == env.max_steps

    def test_zero_perturbation_baseline(self):
        """Zero perturbation should give a baseline performance."""
        env = RANSCalibrationEnv()
        env.reset()
        _, _, _, info = env.step(np.zeros(env.action_dim))
        assert info["error"] > 0
        # With zero perturbation, improvement should be near zero
        assert abs(info["improvement_pct"]) < 10


# =========================================================================
# TestDDPGAgent
# =========================================================================
class TestDDPGAgent:
    """Tests for the full DDPG agent."""

    def test_training_runs(self):
        env = RANSCalibrationEnv(n_features=8, n_coefficients=9)
        config = DDPGConfig(
            state_dim=8, action_dim=9, hidden_dim=32,
            n_episodes=3, max_steps_per_episode=5,
            buffer_size=500, batch_size=4,
        )
        agent = DDPGAgent(env, config)
        report = agent.train()
        assert len(report.history["episode_rewards"]) == 3

    def test_action_selection(self):
        env = RANSCalibrationEnv(n_features=6, n_coefficients=3)
        config = DDPGConfig(state_dim=6, action_dim=3, hidden_dim=16)
        agent = DDPGAgent(env, config)
        state = env.reset()
        action = agent.select_action(state, add_noise=True)
        assert action.shape == (3,)
        assert np.all(np.isfinite(action))

    def test_deterministic_action(self):
        env = RANSCalibrationEnv(n_features=6, n_coefficients=3)
        config = DDPGConfig(state_dim=6, action_dim=3, hidden_dim=16)
        agent = DDPGAgent(env, config)
        state = env.reset()
        # Without noise, same state should give same action
        a1 = agent.select_action(state, add_noise=False)
        a2 = agent.select_action(state, add_noise=False)
        np.testing.assert_array_equal(a1, a2)


# =========================================================================
# TestSciMARL
# =========================================================================
class TestSciMARL:
    """Tests for SciMARL multi-agent wall modeling."""

    def test_action_shape(self):
        manager = SciMARLManager(n_agents=50, state_dim=6, action_dim=1)
        states = np.random.randn(50, 6)
        actions = manager.get_actions(states)
        assert actions.shape == (50, 1)

    def test_positive_eddy_viscosity(self):
        """Eddy viscosity should be strictly positive."""
        manager = SciMARLManager(n_agents=20)
        states = np.random.randn(20, 6) * 3
        actions = manager.get_actions(states)
        assert np.all(actions > 0)

    def test_global_reward(self):
        manager = SciMARLManager()
        pred = np.random.randn(100)
        target = pred + 0.1
        reward = manager.compute_global_reward(pred, target)
        assert reward < 0  # Negative error

    def test_local_states(self):
        manager = SciMARLManager(n_agents=30, state_dim=6)
        vel_grad = np.random.randn(30, 4)
        wall_dist = np.abs(np.random.randn(30))
        states = manager.get_local_states(vel_grad, wall_dist)
        assert states.shape == (30, 6)


# =========================================================================
# TestDDPGTrainingReport
# =========================================================================
class TestDDPGTrainingReport:
    """Tests for training report generation."""

    def test_summary(self):
        history = {
            "episode_rewards": [0.1, 0.2, 0.3],
            "episode_errors": [0.05, 0.04, 0.03],
            "actor_loss": [],
            "critic_loss": [],
        }
        config = DDPGConfig()
        report = DDPGTrainingReport(history, config)
        s = report.summary()
        assert "DDPG" in s
        assert "Episodes" in s

    def test_to_dict(self):
        history = {
            "episode_rewards": [0.1, 0.2],
            "episode_errors": [0.05, 0.04],
        }
        report = DDPGTrainingReport(history, DDPGConfig())
        d = report.to_dict()
        assert "n_episodes" in d
        assert d["n_episodes"] == 2

    def test_compare_baselines(self):
        history = {
            "episode_rewards": [0.1, 0.5],
            "episode_errors": [0.05, 0.02],
        }
        report = DDPGTrainingReport(history, DDPGConfig())
        comp = report.compare_baselines(ga_error=0.04, pso_error=0.03)
        assert "vs_ga_improvement_pct" in comp
        assert "vs_pso_improvement_pct" in comp


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
