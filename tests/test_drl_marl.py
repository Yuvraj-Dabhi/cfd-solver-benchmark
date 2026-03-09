import pytest
import numpy as np

from scripts.ml_augmentation.drl_flow_control import (
    NACA0012Env, BeVERLIHillEnv, MARLWrapper
)

class TestNACA0012Env:
    def test_initialization(self):
        env = NACA0012Env(alpha_deg=15.0, n_actuators=3)
        assert env.case == "naca0012"
        assert env.action_dim == 3
        # Should have CL and CD
        obs = env.reset()
        assert len(obs) == env.obs_dim
        
    def test_step_logic(self):
        env = NACA0012Env()
        obs = env.reset()
        assert env.current_CL > 0.0
        
        # Take action
        action = np.array([0.05, 0.05, 0.05, 0.05, 0.05])
        next_obs, reward, terminated, truncated, info = env.step(action)
        assert "CL" in info
        assert "CD" in info
        assert "L_over_D" in info

class TestBeVERLIHillEnv:
    def test_initialization_and_step(self):
        env = BeVERLIHillEnv(n_actuators=4)
        obs = env.reset()
        assert len(obs) == env.obs_dim
        
        action = np.array([0.1, 0.1, 0.1, 0.1])
        next_obs, reward, terminated, truncated, info = env.step(action)
        assert isinstance(reward, float)
        assert info["bubble_length"] >= 0

class TestMARLWrapper:
    def test_wrapper_dimensions(self):
        base_env = BeVERLIHillEnv(n_actuators=3)
        marl = MARLWrapper(base_env, n_span_agents=4, communication_radius=1)
        assert marl.n_span_agents == 4
        # Observaton has local + neighbors (max 2) + span_pos
        expected_obs_dim = base_env.obs_dim + 3 * 2 + 1
        assert marl.agent_obs_dim == expected_obs_dim
        
    def test_reset_and_step(self):
        base_env = BeVERLIHillEnv(n_actuators=3)
        marl = MARLWrapper(base_env, n_span_agents=4, communication_radius=1)
        
        obs_list = marl.reset()
        assert len(obs_list) == 4
        assert len(obs_list[0]) == marl.agent_obs_dim
        
        actions = [np.array([0.05, 0.05, 0.05]) for _ in range(4)]
        next_obs_list, rewards, terminated, truncated, info = marl.step(actions)
        
        assert len(next_obs_list) == 4
        assert len(rewards) == 4
        assert "global_reward" in info
        assert "per_agent_rewards" in info
