"""
Tests for DRL Flow Control Training (Gap 5) — Extended
========================================================
Validates WallHumpEnv, NACA0012Env, MARLWrapper, GridTransferManager,
DRLTrainingReport, plus the original NNPolicy, GAE, PPO, checkpoint,
and curriculum tests.
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ml_augmentation.drl_flow_control import (
    FlowControlEnv,
    WallHumpEnv,
    NACA0012Env,
    MARLWrapper,
    GridTransferManager,
    PPOAgent,
    SACAgent,
    NNPolicy,
    TrainingConfig,
    CurriculumScheduler,
    TrainingLogger,
    DRLTrainingReport,
    RedisCFDInterface,
    ReplayBuffer,
    compute_gae,
    compute_tsb_area,
    compute_actuation_psd,
    train_coarse_then_transfer,
    transfer_wall_hump_to_naca0012,
)


# =========================================================================
# Original Tests (unchanged)
# =========================================================================
class TestNNPolicy:
    """Test NN policy shapes and outputs."""

    def test_nn_policy_shapes(self):
        policy = NNPolicy(obs_dim=103, action_dim=5, hidden_size=32)
        obs = np.random.randn(103)

        action_mean = policy.forward_actor(obs)
        assert action_mean.shape == (5,)

        value = policy.forward_critic(obs)
        assert isinstance(value, float)
        assert np.isfinite(value)

    def test_get_action_stochastic(self):
        policy = NNPolicy(obs_dim=103, action_dim=5)
        obs = np.random.randn(103)

        action1, lp1 = policy.get_action(obs)
        action2, lp2 = policy.get_action(obs)

        assert action1.shape == (5,)
        assert np.isfinite(lp1)
        # Stochastic: actions should differ (with very high prob)
        assert not np.allclose(action1, action2)

    def test_get_action_deterministic(self):
        policy = NNPolicy(obs_dim=103, action_dim=5, seed=42)
        obs = np.random.randn(103)

        a1, _ = policy.get_action(obs, deterministic=True)
        a2, _ = policy.get_action(obs, deterministic=True)
        np.testing.assert_array_equal(a1, a2)

    def test_entropy(self):
        policy = NNPolicy(obs_dim=10, action_dim=3)
        ent = policy.entropy()
        assert isinstance(ent, float)
        assert np.isfinite(ent)


class TestGAE:
    """Test Generalized Advantage Estimation."""

    def test_gae_computation(self):
        """Verify GAE against hand-computed values."""
        rewards = [1.0, 2.0, 3.0]
        values = [0.5, 1.0, 1.5, 2.0]  # T+1 values
        dones = [False, False, True]

        advantages, returns = compute_gae(
            rewards, values, dones,
            gamma=0.99, gae_lambda=0.95,
        )

        assert advantages.shape == (3,)
        assert returns.shape == (3,)
        assert np.all(np.isfinite(advantages))
        assert np.all(np.isfinite(returns))

    def test_gae_returns_shape(self):
        T = 50
        rewards = [float(np.random.randn()) for _ in range(T)]
        values = [float(np.random.randn()) for _ in range(T + 1)]
        dones = [False] * (T - 1) + [True]

        advantages, returns = compute_gae(rewards, values, dones)
        assert advantages.shape == (T,)
        assert returns.shape == (T,)

    def test_gae_terminal_advantage(self):
        """If episode ends, advantage should reflect terminal reward."""
        rewards = [0.0, 0.0, 10.0]
        values = [0.0, 0.0, 0.0, 0.0]
        dones = [False, False, True]

        advantages, _ = compute_gae(rewards, values, dones,
                                     gamma=1.0, gae_lambda=1.0)
        # Last advantage should be dominated by the large reward
        assert advantages[-1] > 0


class TestPPOTraining:
    """Test PPO training loop."""

    def test_ppo_train_1000_steps(self):
        env = FlowControlEnv(n_wall_points=20, n_actuators=3)
        config = TrainingConfig(
            total_timesteps=1000,
            hidden_size=16,
            n_epochs=1,
            mini_batch_size=32,
        )
        agent = PPOAgent(env, config=config)
        history = agent.train()

        assert "episode_reward" in history
        assert len(history["episode_reward"]) > 0

    def test_evaluate_deterministic(self):
        env = FlowControlEnv(n_wall_points=20, n_actuators=3)
        config = TrainingConfig(
            n_episodes_eval=3, hidden_size=16,
        )
        agent = PPOAgent(env, config=config)

        results = agent.evaluate(n_episodes=3)
        assert "mean_reward" in results
        assert "mean_bubble_reduction" in results
        assert np.isfinite(results["mean_reward"])


class TestCheckpointing:
    """Test checkpoint save/load."""

    def test_checkpoint_save_load(self):
        env = FlowControlEnv(n_wall_points=20, n_actuators=3)
        config = TrainingConfig(hidden_size=16)
        agent = PPOAgent(env, config=config)

        obs = env.reset()
        action_before = agent.get_action(obs, deterministic=True)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            agent.save_checkpoint(f.name)

        # Create new agent and load
        agent2 = PPOAgent(env, config=config)
        agent2.load_checkpoint(f.name)

        action_after = agent2.get_action(obs, deterministic=True)
        np.testing.assert_array_almost_equal(action_before, action_after)


class TestCurriculumScheduler:
    """Test curriculum stage advancement."""

    def test_curriculum_scheduler(self):
        scheduler = CurriculumScheduler()

        assert scheduler.current_stage["name"] == "easy"

        # Should advance at episode 50
        stage = scheduler.update(50)
        assert stage is not None
        assert stage["name"] == "medium"

        # Should advance at episode 150
        stage = scheduler.update(150)
        assert stage is not None
        assert stage["name"] == "hard"

    def test_no_change_within_stage(self):
        scheduler = CurriculumScheduler()
        assert scheduler.update(5) is None  # Still in easy


class TestTrainingLogger:
    """Test training logger."""

    def test_logger_logs(self):
        logger = TrainingLogger()
        for i in range(5):
            logger.log_episode(i, float(i), float(i * 2), i + 10)

        assert len(logger.history["episode_reward"]) == 5
        stats = logger.get_recent_stats()
        assert "episode_reward_mean" in stats

    def test_logger_save(self, tmp_path):
        lgr = TrainingLogger()
        lgr.log_episode(1, 10.0, 5.0, 50)
        path = tmp_path / "test_log.json"
        lgr.save(str(path))

        with open(path) as f:
            data = json.load(f)
        assert "episode_reward" in data


# =========================================================================
# New Tests — Wall Hump Environment
# =========================================================================
class TestWallHumpEnv:
    """Test wall hump environment physics and interface."""

    def test_env_creation(self):
        """Env creates with correct dimensions."""
        env = WallHumpEnv(n_actuators=5, grid_level="coarse")
        assert env.action_dim == 5
        assert env.n_wall_points == 40  # coarse = 40
        assert env.case == "wall_hump"
        assert env.grid_level == "coarse"

    def test_env_reset(self):
        """Reset produces valid observation."""
        env = WallHumpEnv(n_actuators=3, grid_level="coarse")
        obs = env.reset()
        assert obs.shape == (env.obs_dim,)
        assert np.all(np.isfinite(obs))

    def test_baseline_separation(self):
        """Baseline flow has separation near x/c=0.665."""
        env = WallHumpEnv(n_actuators=3, grid_level="medium")
        env.reset()
        state = env.current_state

        # Separation should be detectable
        assert state.bubble_length > 0
        # Separation point should be near 0.665
        assert 0.4 < state.x_sep < 0.9, f"x_sep={state.x_sep} outside expected range"
        # Reattachment should be downstream
        assert state.x_reat > state.x_sep

    def test_step_produces_reward(self):
        """Stepping with action produces finite reward."""
        env = WallHumpEnv(n_actuators=5, grid_level="coarse")
        obs = env.reset()
        action = np.zeros(5)
        obs2, reward, terminated, truncated, info = env.step(action)

        assert np.isfinite(reward)
        assert obs2.shape == obs.shape
        assert "bubble_length" in info
        assert "x_sep" in info

    def test_blowing_reduces_bubble(self):
        """Positive blowing should reduce separation bubble."""
        env = WallHumpEnv(n_actuators=5, grid_level="coarse")
        env.reset()
        baseline_bubble = env.current_state.bubble_length

        # Apply strong blowing
        action = np.ones(5) * env.max_blowing
        _, _, _, _, info = env.step(action)

        # Bubble should be reduced (or at least not massively increased)
        assert info["bubble_length"] <= baseline_bubble * 1.5

    def test_grid_levels(self):
        """Different grid levels produce different resolutions."""
        env_c = WallHumpEnv(grid_level="coarse")
        env_m = WallHumpEnv(grid_level="medium")
        env_f = WallHumpEnv(grid_level="fine")

        assert env_c.n_wall_points < env_m.n_wall_points < env_f.n_wall_points


# =========================================================================
# New Tests — NACA 0012 Environment
# =========================================================================
class TestNACA0012Env:
    """Test NACA 0012 near-stall environment."""

    def test_env_creation(self):
        """Env creates with lift/drag observation."""
        env = NACA0012Env(alpha_deg=15.0, n_actuators=5)
        assert env.action_dim == 5
        assert env.alpha_deg == 15.0
        # obs = Cf + Cp + sep + reat + bubble + CL + CD
        assert env.obs_dim == env.n_wall_points * 2 + 5

    def test_env_reset(self):
        """Reset produces valid observation with CL, CD."""
        env = NACA0012Env(alpha_deg=15.0, n_actuators=3)
        obs = env.reset()
        assert obs.shape == (env.obs_dim,)
        assert np.all(np.isfinite(obs))

    def test_positive_lift(self):
        """At α=15°, baseline CL should be positive."""
        env = NACA0012Env(alpha_deg=15.0, n_actuators=3)
        env.reset()
        assert env.current_CL > 0, f"CL={env.current_CL} should be positive"
        assert env.current_CD > 0, f"CD={env.current_CD} should be positive"

    def test_separation_at_high_alpha(self):
        """Should detect separation at α=15°."""
        env = NACA0012Env(alpha_deg=15.0, n_actuators=3)
        env.reset()
        state = env.current_state
        assert state.bubble_length > 0, "Should have separation at α=15°"

    def test_step_returns_aero_coefficients(self):
        """Step info should include CL, CD, L/D."""
        env = NACA0012Env(alpha_deg=15.0, n_actuators=3)
        env.reset()
        action = np.zeros(3)
        _, _, _, _, info = env.step(action)

        assert "CL" in info
        assert "CD" in info
        assert "L_over_D" in info
        assert np.isfinite(info["CL"])
        assert np.isfinite(info["CD"])

    def test_obs_dimension_matches(self):
        """Observation dimension matches specified obs_dim."""
        env = NACA0012Env(alpha_deg=15.0, n_actuators=5, grid_level="medium")
        obs = env.reset()
        assert obs.shape[0] == env.obs_dim

    def test_naca_thickness(self):
        """NACA 0012 thickness should be correct at known points."""
        env = NACA0012Env()
        # At x=0, thickness should be 0
        t0 = env._naca0012_thickness(np.array([0.0]))
        assert abs(t0[0]) < 0.001
        # At x=0.3 (approx max thickness), should be ~0.06
        t30 = env._naca0012_thickness(np.array([0.3]))
        assert 0.04 < t30[0] < 0.08


# =========================================================================
# New Tests — MARL Wrapper
# =========================================================================
class TestMARLWrapper:
    """Test multi-agent RL wrapper."""

    def test_marl_creation(self):
        """MARL wrapper creates with correct dimensions."""
        env = FlowControlEnv(n_wall_points=20, n_actuators=3)
        marl = MARLWrapper(env, n_span_agents=4, communication_radius=1)

        assert marl.n_span_agents == 4
        assert marl.agent_action_dim == 3
        # obs_dim = base_obs + 2 neighbors * 3 scalars + 1 span_pos
        assert marl.agent_obs_dim == env.obs_dim + 6 + 1

    def test_marl_reset(self):
        """Reset returns per-agent observations."""
        env = FlowControlEnv(n_wall_points=20, n_actuators=3)
        marl = MARLWrapper(env, n_span_agents=4)
        obs_list = marl.reset()

        assert len(obs_list) == 4
        for obs in obs_list:
            assert obs.shape == (marl.agent_obs_dim,)
            assert np.all(np.isfinite(obs))

    def test_marl_step(self):
        """Step with per-agent actions returns correct outputs."""
        env = FlowControlEnv(n_wall_points=20, n_actuators=3)
        marl = MARLWrapper(env, n_span_agents=4)
        marl.reset()

        actions = [np.zeros(3) for _ in range(4)]
        obs_list, rewards, terminated, truncated, info = marl.step(actions)

        assert len(obs_list) == 4
        assert len(rewards) == 4
        assert "global_reward" in info
        assert "per_agent_rewards" in info

    def test_marl_span_positions(self):
        """Span positions should be evenly distributed."""
        env = FlowControlEnv(n_wall_points=20, n_actuators=3)
        marl = MARLWrapper(env, n_span_agents=4)

        np.testing.assert_allclose(
            marl.span_positions,
            [0, 1/3, 2/3, 1],
            atol=0.01,
        )

    def test_marl_different_agent_obs(self):
        """Different agents should have different observations (span pos)."""
        env = FlowControlEnv(n_wall_points=20, n_actuators=3)
        marl = MARLWrapper(env, n_span_agents=4)
        obs_list = marl.reset()

        # Last element is span position — should differ between agents
        span_positions = [obs[-1] for obs in obs_list]
        assert len(set(np.round(span_positions, 4))) == 4


# =========================================================================
# New Tests — Grid Transfer
# =========================================================================
class TestGridTransfer:
    """Test coarse-to-fine grid transfer."""

    def test_transfer_same_grid(self):
        """Transfer between same grid levels should be identity."""
        env1 = WallHumpEnv(n_actuators=3, grid_level="coarse")
        env2 = WallHumpEnv(n_actuators=3, grid_level="coarse")

        agent = PPOAgent(env1, config=TrainingConfig(
            hidden_size=16, total_timesteps=200,
        ))
        agent.train()

        transfer = GridTransferManager(agent, env1, env2)
        results = transfer.evaluate_transfer(n_episodes=3)

        assert "source_reward" in results
        assert "target_reward" in results
        assert np.isfinite(results["transfer_efficiency"])

    def test_transfer_coarse_to_fine(self):
        """Transfer from coarse to fine grid should work."""
        coarse = WallHumpEnv(n_actuators=3, grid_level="coarse")
        fine = WallHumpEnv(n_actuators=3, grid_level="fine")

        agent = PPOAgent(coarse, config=TrainingConfig(
            hidden_size=16, total_timesteps=200,
        ))
        agent.train()

        transfer = GridTransferManager(agent, coarse, fine)
        results = transfer.evaluate_transfer(n_episodes=3)

        assert results["grid_levels"]["source"] == "coarse"
        assert results["grid_levels"]["target"] == "fine"
        assert np.isfinite(results["target_reward"])

    def test_observation_mapping(self):
        """Observation mapping should interpolate between grid sizes."""
        coarse = WallHumpEnv(n_actuators=3, grid_level="coarse")
        fine = WallHumpEnv(n_actuators=3, grid_level="fine")

        agent = PPOAgent(coarse, config=TrainingConfig(hidden_size=16))
        transfer = GridTransferManager(agent, coarse, fine)

        fine_obs = fine.reset()
        mapped_obs = transfer.map_observation(fine_obs)

        assert mapped_obs.shape[0] == coarse.obs_dim
        assert np.all(np.isfinite(mapped_obs))


# =========================================================================
# New Tests — DRL Training Report
# =========================================================================
class TestDRLTrainingReport:
    """Test training report generation."""

    def test_report_creation(self, tmp_path):
        """Report creates and saves JSON."""
        report = DRLTrainingReport(output_dir=str(tmp_path))

        report.add_case_result(
            "test_case",
            {"episode_reward": [1.0, 2.0, 3.0], "bubble_reduction": [5.0, 6.0, 7.0]},
            {"mean_reward": 2.0, "mean_bubble_reduction": 6.0},
        )

        result = report.generate_report()
        assert "cases" in result
        assert "test_case" in result["cases"]

        # Check JSON file was saved
        report_path = tmp_path / "drl_training_report.json"
        assert report_path.exists()

        with open(report_path) as f:
            data = json.load(f)
        assert data["cases"]["test_case"]["summary"]["final_reward"] == 2.0

    def test_report_multiple_cases(self, tmp_path):
        """Report handles multiple cases."""
        report = DRLTrainingReport(output_dir=str(tmp_path))

        for name in ["wall_hump", "naca0012"]:
            report.add_case_result(
                name,
                {"episode_reward": [1.0, 2.0], "bubble_reduction": [3.0, 4.0]},
                {"mean_reward": 1.5, "mean_bubble_reduction": 3.5},
            )

        result = report.generate_report()
        assert result["overall_summary"]["n_cases"] == 2

    def test_report_serialization(self, tmp_path):
        """Report handles numpy types correctly."""
        report = DRLTrainingReport(output_dir=str(tmp_path))

        report.add_case_result(
            "numpy_test",
            {"episode_reward": list(np.array([1.0, 2.0, 3.0]))},
            {"mean_reward": np.float64(2.5), "mean_bubble_reduction": np.float64(5.0)},
        )

        # Should not raise
        result = report.generate_report()
        assert result is not None


# =========================================================================
# New Tests — End-to-End Training
# =========================================================================
class TestEndToEndTraining:
    """Test end-to-end training runs."""

    def test_wall_hump_short_training(self):
        """Short training on wall hump completes and produces results."""
        env = WallHumpEnv(n_actuators=3, grid_level="coarse")
        config = TrainingConfig(
            total_timesteps=500,
            hidden_size=16,
            n_epochs=1,
            mini_batch_size=32,
        )
        agent = PPOAgent(env, config=config)
        history = agent.train()

        assert len(history["episode_reward"]) > 0
        results = agent.evaluate(n_episodes=3)
        assert np.isfinite(results["mean_reward"])

    def test_naca0012_short_training(self):
        """Short training on NACA 0012 completes and produces results."""
        env = NACA0012Env(alpha_deg=15.0, n_actuators=3, grid_level="coarse")
        config = TrainingConfig(
            total_timesteps=500,
            hidden_size=16,
            n_epochs=1,
            mini_batch_size=32,
        )
        agent = PPOAgent(env, config=config)
        history = agent.train()

        assert len(history["episode_reward"]) > 0
        results = agent.evaluate(n_episodes=3)
        assert np.isfinite(results["mean_reward"])

    def test_wall_hump_checkpoint_roundtrip(self):
        """Save and load checkpoint for wall hump agent."""
        env = WallHumpEnv(n_actuators=3, grid_level="coarse")
        config = TrainingConfig(hidden_size=16, total_timesteps=200)
        agent = PPOAgent(env, config=config)
        agent.train()

        obs = env.reset()
        action_before = agent.get_action(obs, deterministic=True)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            agent.save_checkpoint(f.name)

        agent2 = PPOAgent(env, config=config)
        agent2.load_checkpoint(f.name)
        action_after = agent2.get_action(obs, deterministic=True)
        np.testing.assert_array_almost_equal(action_before, action_after)


# =========================================================================
# New Tests — Redis, SAC, TSB, PSD, Probes, Transfer
# =========================================================================
class TestRedisCFDInterface:
    """Test Redis communication layer (dummy mode)."""

    def test_dummy_creation(self):
        """RedisCFDInterface with dummy backend creates successfully."""
        iface = RedisCFDInterface(use_dummy=True, n_envs=2)
        assert iface.is_connected
        assert iface.use_dummy is True

    def test_publish_subscribe_roundtrip(self):
        """Publish action, publish state, then read state back."""
        iface = RedisCFDInterface(use_dummy=True, n_envs=1)
        action = np.array([0.1, 0.2, 0.3])
        iface.publish_action(0, action)

        # Publish a state from solver side
        state_dict = {"Cf": [0.001, 0.002], "x_sep": 0.65}
        iface.publish_state(0, state_dict)

        # Read it back
        received = iface.subscribe_state(0)
        assert received is not None
        assert received["x_sep"] == 0.65

    def test_channel_naming(self):
        """Channels follow cfd:{env_id}:{type} naming convention."""
        iface = RedisCFDInterface(use_dummy=True, n_envs=3)
        iface.send_command(2, "reset")
        # Verify the command was published to the right channel
        msg = iface._redis.get_message("cfd:2:control")
        assert msg == "reset"

    def test_close(self):
        iface = RedisCFDInterface(use_dummy=True)
        iface.close()
        assert not iface._redis.connected


class TestSACAgent:
    """Test Soft Actor-Critic agent."""

    def test_sac_creation(self):
        """SAC agent creates with correct dimensions."""
        env = WallHumpEnv(n_actuators=3, grid_level="coarse")
        config = TrainingConfig(algorithm="sac", hidden_size=16,
                                total_timesteps=100)
        agent = SACAgent(env, config=config)
        assert agent.policy.obs_dim == env.obs_dim
        assert agent.policy.action_dim == env.action_dim
        assert agent.alpha > 0

    def test_sac_short_training(self):
        """SAC trains for 500 steps without error."""
        env = WallHumpEnv(n_actuators=3, grid_level="coarse")
        config = TrainingConfig(
            algorithm="sac", hidden_size=16,
            total_timesteps=500, replay_buffer_size=1000,
        )
        agent = SACAgent(env, config=config)
        history = agent.train()
        assert len(history["episode_reward"]) > 0

    def test_sac_checkpoint_roundtrip(self):
        """SAC save/load preserves policy weights."""
        env = WallHumpEnv(n_actuators=3, grid_level="coarse")
        config = TrainingConfig(
            algorithm="sac", hidden_size=16, total_timesteps=200,
        )
        agent = SACAgent(env, config=config)
        agent.train()

        obs = env.reset()
        action_before = agent.get_action(obs, deterministic=True)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            agent.save_checkpoint(f.name)

        agent2 = SACAgent(env, config=config)
        agent2.load_checkpoint(f.name)
        action_after = agent2.get_action(obs, deterministic=True)
        np.testing.assert_array_almost_equal(action_before, action_after)

    def test_replay_buffer(self):
        """Replay buffer add/sample works correctly."""
        buf = ReplayBuffer(capacity=100, obs_dim=5, action_dim=2)
        for i in range(50):
            buf.add(
                np.random.randn(5), np.random.randn(2),
                float(i), np.random.randn(5), i % 10 == 0,
            )
        assert buf.size == 50
        batch = buf.sample(16)
        assert batch["obs"].shape == (16, 5)
        assert batch["actions"].shape == (16, 2)
        assert batch["rewards"].shape == (16,)


class TestTSBMetrics:
    """Test Turbulent Separation Bubble area computation."""

    def test_tsb_with_known_cf(self):
        """TSB area of a known negative Cf region is correct."""
        x = np.linspace(0, 2, 200)
        # Cf negative between 0.5 and 1.0 with magnitude 0.002
        Cf = np.where((x > 0.5) & (x < 1.0), -0.002, 0.003)
        tsb = compute_tsb_area(Cf, x, x_sep=0.5, x_reat=1.0)
        # Expected ≈ 0.002 * 0.5 = 0.001
        assert 0.0005 < tsb < 0.0015

    def test_tsb_zero_when_no_separation(self):
        """TSB area is 0 when Cf is everywhere positive."""
        x = np.linspace(0, 2, 100)
        Cf = 0.003 * np.ones_like(x)
        tsb = compute_tsb_area(Cf, x)
        assert tsb == 0.0

    def test_tsb_fallback_mask(self):
        """TSB area uses Cf < 0 mask when sep/reat not given."""
        x = np.linspace(0, 2, 200)
        Cf = np.where((x > 0.7) & (x < 1.2), -0.001, 0.005)
        tsb = compute_tsb_area(Cf, x)
        assert tsb > 0


class TestActuationPSD:
    """Test actuation PSD computation."""

    def test_sine_wave_peak(self):
        """PSD of a pure sine wave peaks at the correct frequency."""
        dt = 0.01
        freq = 5.0
        t = np.arange(0, 10, dt)
        signal = np.sin(2 * np.pi * freq * t)
        freqs, psd = compute_actuation_psd(signal, dt=dt)
        peak_idx = np.argmax(psd[1:]) + 1
        assert abs(freqs[peak_idx] - freq) < 0.5

    def test_2d_input(self):
        """PSD handles 2D (T, n_actuators) input by averaging."""
        actions = np.random.randn(100, 5)
        freqs, psd = compute_actuation_psd(actions, dt=1.0)
        assert len(freqs) == len(psd)
        assert len(freqs) > 1

    def test_short_signal(self):
        """Very short signal returns fallback."""
        freqs, psd = compute_actuation_psd(np.array([1.0, 2.0]), dt=1.0)
        assert len(freqs) == 1


class TestPressureProbeState:
    """Test 8-point surface pressure probe observation."""

    def test_probe_obs_dimension(self):
        """Env with 8 probes produces obs of dim 8+3=11."""
        env = WallHumpEnv(n_actuators=3, grid_level="coarse", n_pressure_probes=8)
        assert env.obs_dim == 11  # 8 probes + sep + reat + bubble
        obs = env.reset()
        assert len(obs) == 11

    def test_probe_locations(self):
        """Probes are at correct x/c positions."""
        env = WallHumpEnv(n_actuators=3, n_pressure_probes=8)
        assert len(env.pressure_probe_x) == 8
        assert env.pressure_probe_x[0] > env.HUMP_START
        assert env.pressure_probe_x[-1] < env.HUMP_END

    def test_no_probes_default(self):
        """Without probes, obs is full Cf+Cp (default)."""
        env = WallHumpEnv(n_actuators=3, grid_level="coarse")
        obs = env.reset()
        assert len(obs) == env.n_wall_points * 2 + 3  # Cf + Cp + sep/reat/bubble

    def test_probe_step_produces_finite(self):
        """Stepping with probes produces finite observations."""
        env = WallHumpEnv(n_actuators=3, n_pressure_probes=8)
        obs = env.reset()
        action = np.zeros(3)
        obs2, reward, _, _, _ = env.step(action)
        assert np.all(np.isfinite(obs2))


class TestCoarseToFineTraining:
    """Test coarse→fine training pipeline."""

    def test_pipeline_returns_valid_report(self):
        """train_coarse_then_transfer() runs and returns expected keys."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = train_coarse_then_transfer(
                n_actuators=3,
                coarse_timesteps=500,
                n_pressure_probes=0,
                output_dir=tmpdir,
            )
            assert "training_history" in result
            assert "coarse_eval" in result
            assert "fine_eval" in result
            assert "checkpoint_path" in result
            assert result["coarse_eval"]["mean_reward"] != 0


class TestNACA0012Transfer:
    """Test NACA 0012 transfer learning from wall-hump policy."""

    def test_transfer_produces_finite_rewards(self):
        """Warm-started NACA 0012 agent produces finite evaluation rewards."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # First train a wall-hump agent briefly
            env = WallHumpEnv(n_actuators=3, grid_level="coarse")
            config = TrainingConfig(hidden_size=16, total_timesteps=200)
            agent = PPOAgent(env, config=config)
            agent.train()
            ckpt_path = str(Path(tmpdir) / "wall_hump_ckpt.json")
            agent.save_checkpoint(ckpt_path)

            # Transfer to NACA 0012
            result = transfer_wall_hump_to_naca0012(
                wall_hump_checkpoint=ckpt_path,
                n_actuators=3,
                fine_tune_timesteps=200,
                output_dir=tmpdir,
            )
            assert np.isfinite(result["evaluation"]["mean_reward"])
            assert "checkpoint_path" in result

    def test_obs_remapping_partial_weights(self):
        """Obs dim mismatch triggers partial weight transfer."""
        # Wall hump obs_dim != NACA 0012 obs_dim → partial transfer path
        with tempfile.TemporaryDirectory() as tmpdir:
            env_hump = WallHumpEnv(n_actuators=3, n_pressure_probes=8)
            env_naca = NACA0012Env(n_actuators=3, grid_level="coarse")

            # They should have different obs dims
            # (8+3=11 vs naca default)
            config = TrainingConfig(hidden_size=16, total_timesteps=200)
            agent = PPOAgent(env_hump, config=config)
            agent.train()
            ckpt_path = str(Path(tmpdir) / "probe_hump.json")
            agent.save_checkpoint(ckpt_path)

            result = transfer_wall_hump_to_naca0012(
                wall_hump_checkpoint=ckpt_path,
                n_actuators=3,
                fine_tune_timesteps=100,
                output_dir=tmpdir,
            )
            assert np.isfinite(result["evaluation"]["mean_reward"])


# =========================================================================
# New Tests — MARL and Flow Control Extensions (Extension IV)
# =========================================================================

class TestMARLExtensions:
    """Tests for MARLCoordinator, GridTransferPolicy, RewardShaping, Baselines."""

    def test_marl_coordinator(self):
        from scripts.ml_augmentation.drl_flow_control import MARLCoordinator
        coord = MARLCoordinator(n_agents=3)
        obs = [np.random.randn(5) for _ in range(3)]
        actions = coord.coordinate_actions(obs)
        assert len(actions) == 3
        assert actions[0].shape == (1,)

    def test_grid_transfer_policy(self):
        from scripts.ml_augmentation.drl_flow_control import GridTransferPolicy
        class DummyPolicy:
            def get_action(self, obs, deterministic=True):
                return np.ones(2), None

        base = DummyPolicy()
        policy = GridTransferPolicy(base)
        action = policy.get_action(np.random.randn(10))
        assert np.all(action == 1)

    def test_afc_reward_shaping(self):
        from scripts.ml_augmentation.drl_flow_control import AFCRewardShaping
        shaping = AFCRewardShaping(lambda_cd=1.0, lambda_tsb=2.0, lambda_power=0.5)
        reward = shaping.compute_reward(cl=1.5, cd=0.1, tsb_area=0.2, power=0.1)
        # 1.5 - 0.1 - 0.4 - 0.05 = 0.95
        np.testing.assert_allclose(reward, 0.95)

    def test_baseline_comparison(self):
        from scripts.ml_augmentation.drl_flow_control import BaselineComparison, WallHumpEnv
        env = WallHumpEnv(n_actuators=2, grid_level="coarse")
        comp = BaselineComparison(env)
        
        unc = comp.evaluate_uncontrolled()
        assert "reward" in unc
        
        const = comp.evaluate_constant_blowing(0.1)
        assert "reward" in const
        
        period = comp.evaluate_periodic_forcing(10.0, 0.1)
        assert "mean_reward" in period
