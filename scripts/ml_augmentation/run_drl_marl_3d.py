import json
import logging
from pathlib import Path
import numpy as np

from scripts.ml_augmentation.drl_flow_control import (
    WallHumpEnv,
    NACA0012Env,
    BeVERLIHillEnv,
    MARLWrapper,
    PPOAgent,
    TrainingConfig,
    transfer_wall_hump_to_naca0012
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class MARLPPOAgent(PPOAgent):
    """PPO Agent adapted to handle MARLWrapper list-based transitions."""
    def train(self, total_timesteps=None):
        cfg = self.config
        total_timesteps = total_timesteps or cfg.total_timesteps
        total_steps = 0
        episode = 0

        while total_steps < total_timesteps:
            obs_list = []
            action_list = []
            reward_list = []
            value_list = []
            done_list = []

            obs = self.env.reset()
            episode_rewards = [0] * self.env.n_span_agents

            for t in range(self.env.max_steps):
                actions = []
                values = []
                for i in range(self.env.n_span_agents):
                    a, _ = self.policy.get_action(obs[i])
                    a = np.clip(a, -self.env.max_blowing, self.env.max_blowing)
                    v = self.policy.forward_critic(obs[i])
                    actions.append(a)
                    values.append(v)
                    
                    obs_list.append(obs[i].copy())
                    action_list.append(a.copy())
                    
                next_obs, rewards, terminated, truncated, info = self.env.step(actions)
                done = terminated or truncated
                
                for i in range(self.env.n_span_agents):
                    reward_list.append(rewards[i])
                    value_list.append(values[i])
                    done_list.append(done)
                    episode_rewards[i] += rewards[i]
                    
                obs = next_obs
                total_steps += 1
                if done:
                    break
                    
            # Bootstrap
            for i in range(self.env.n_span_agents):
                value_list.append(self.policy.forward_critic(obs[i]))

            # Simplified Returns calculation for placeholder MARL
            returns = np.array(reward_list) + cfg.gamma * (1 - np.array(done_list)) * np.array(value_list[self.env.n_span_agents:])
            advantages = returns - np.array(value_list[:-self.env.n_span_agents])
            
            if len(advantages) > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
            self._ppo_update(
                np.array(obs_list), 
                np.array(action_list), 
                np.array(advantages), 
                np.array(returns), 
                cfg
            )
            episode += 1
            if episode % 10 == 0:
                logger.info(f"MARL Episode {episode}, Mean Reward: {np.mean(episode_rewards):.2f}")

        return {"mean_reward": [float(np.mean(episode_rewards))]}


def main():
    out_dir = Path("results/drl_marl_3d")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 2D Wall Hump Baseline
    logger.info("=== Phase 1: 2D Wall Hump PPO ===")
    cfg = TrainingConfig(total_timesteps=500, n_epochs=1, batch_size=100)
    wh_env = WallHumpEnv(n_actuators=3)
    wh_agent = PPOAgent(wh_env, config=cfg)
    wh_agent.train()
    wh_ckpt = out_dir / "wall_hump_quick.json"
    wh_agent.save_checkpoint(str(wh_ckpt))
    
    # 2. Transfer to NACA 0012
    logger.info("=== Phase 2: Transfer to NACA 0012 ===")
    transfer_res = transfer_wall_hump_to_naca0012(
        str(wh_ckpt), n_actuators=3, fine_tune_timesteps=500, 
        output_dir=str(out_dir)
    )
    
    # 3. 3D BeVERLI Hill MARL
    logger.info("=== Phase 3: 3D BeVERLI Hill MARL ===")
    bev_env = BeVERLIHillEnv(n_actuators=3)
    marl_env = MARLWrapper(bev_env, n_span_agents=4, communication_radius=1)
    
    marl_cfg = TrainingConfig(total_timesteps=500, n_epochs=1)
    marl_agent = MARLPPOAgent(marl_env, config=marl_cfg)
    marl_agent.train()
    
    # Evaluate
    obs = marl_env.reset()
    ep_reward = 0
    for _ in range(marl_env.max_steps):
        acts = [marl_agent.get_action(o, deterministic=True) for o in obs]
        obs, rewards, done, _, info = marl_env.step(acts)
        ep_reward += info["global_reward"]
        if done: break
        
    logger.info(f"MARL Final Global Reward: {ep_reward:.2f}")
    
    benchmark_report = {
        "naca0012_transfer": transfer_res["evaluation"],
        "beverli_marl": {"final_global_reward": float(ep_reward)}
    }
    with open(out_dir / "drl_marl_benchmark.json", "w") as f:
        json.dump(benchmark_report, f, indent=2)
        
    logger.info("Difficulty Ladder complete! Benchmark saved.")

if __name__ == "__main__":
    main()
