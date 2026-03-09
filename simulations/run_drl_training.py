#!/usr/bin/env python3
"""
DRL Active Flow Control — Complete Training Runner
====================================================
Trains PPO agents on Wall Hump and NACA 0012 environments,
evaluates grid transfer (coarse→fine), and generates results.

Based on:
  - Font et al., Nature Communications 16, 1422 (Feb 2025)
  - Montalà et al., arXiv 2509.10185 (Sep 2025)

Outputs:
  - Trained checkpoints in results/drl_flow_control/checkpoints/
  - Training report JSON in results/drl_flow_control/
  - Training curve plots in results/drl_flow_control/

Usage:
    python run_drl_training.py
    python run_drl_training.py --case wall_hump --timesteps 20000
    python run_drl_training.py --case naca0012 --timesteps 20000
    python run_drl_training.py --all
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

# Project imports
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ml_augmentation.drl_flow_control import (
    WallHumpEnv,
    NACA0012Env,
    MARLWrapper,
    GridTransferManager,
    PPOAgent,
    TrainingConfig,
    DRLTrainingReport,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = PROJECT_ROOT / "results" / "drl_flow_control"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"


def train_wall_hump(
    total_timesteps: int = 10000,
    hidden_size: int = 32,
) -> dict:
    """
    Train PPO on Wall Hump environment with coarse→fine grid transfer.

    Returns dict with training history, eval results, and transfer metrics.
    """
    logger.info("=" * 60)
    logger.info("WALL HUMP DRL TRAINING")
    logger.info("=" * 60)

    # --- Stage 1: Train on coarse grid ---
    logger.info("Stage 1: Training on coarse grid (%d timesteps)", total_timesteps)
    coarse_env = WallHumpEnv(n_actuators=5, grid_level="coarse")
    config = TrainingConfig(
        total_timesteps=total_timesteps,
        hidden_size=hidden_size,
        n_epochs=2,
        mini_batch_size=64,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        log_dir=str(RESULTS_DIR / "logs" / "wall_hump"),
    )

    agent = PPOAgent(coarse_env, config=config)
    t0 = time.time()
    history = agent.train()
    train_time = time.time() - t0

    logger.info("Training completed in %.1f seconds", train_time)

    # Save checkpoint
    ckpt_path = CHECKPOINT_DIR / "wall_hump_coarse_ppo.json"
    agent.save_checkpoint(str(ckpt_path))

    # --- Stage 2: Evaluate on coarse grid ---
    logger.info("Stage 2: Evaluating on coarse grid")
    eval_results = agent.evaluate(n_episodes=5)
    logger.info(
        "Coarse eval: reward=%.2f, bubble_reduction=%.1f%%",
        eval_results["mean_reward"],
        eval_results["mean_bubble_reduction"],
    )

    # --- Stage 3: Zero-shot transfer to fine grid ---
    logger.info("Stage 3: Zero-shot transfer to fine grid")
    fine_env = WallHumpEnv(n_actuators=5, grid_level="fine")
    transfer = GridTransferManager(agent, coarse_env, fine_env)
    transfer_results = transfer.evaluate_transfer(n_episodes=5)

    logger.info(
        "Transfer: coarse_reward=%.2f → fine_reward=%.2f (efficiency=%.1f%%)",
        transfer_results["source_reward"],
        transfer_results["target_reward"],
        transfer_results["transfer_efficiency"] * 100,
    )

    return {
        "history": history,
        "eval": eval_results,
        "transfer": transfer_results,
        "config": config,
        "train_time_s": train_time,
    }


def train_naca0012(
    total_timesteps: int = 10000,
    hidden_size: int = 32,
) -> dict:
    """
    Train PPO on NACA 0012 at α=15° with lift/drag reward.

    Returns dict with training history, eval results, and MARL metrics.
    """
    logger.info("=" * 60)
    logger.info("NACA 0012 DRL TRAINING (α=15°)")
    logger.info("=" * 60)

    # --- Stage 1: Train single-agent on coarse grid ---
    logger.info("Stage 1: Training single-agent on coarse grid")
    env = NACA0012Env(
        alpha_deg=15.0, n_actuators=5, grid_level="coarse",
    )
    config = TrainingConfig(
        total_timesteps=total_timesteps,
        hidden_size=hidden_size,
        n_epochs=2,
        mini_batch_size=64,
        lr=3e-4,
        log_dir=str(RESULTS_DIR / "logs" / "naca0012"),
    )

    agent = PPOAgent(env, config=config)
    t0 = time.time()
    history = agent.train()
    train_time = time.time() - t0

    logger.info("Training completed in %.1f seconds", train_time)

    # Save checkpoint
    ckpt_path = CHECKPOINT_DIR / "naca0012_ppo.json"
    agent.save_checkpoint(str(ckpt_path))

    # --- Stage 2: Evaluate ---
    logger.info("Stage 2: Evaluating trained agent")
    eval_results = agent.evaluate(n_episodes=5)
    logger.info(
        "Eval: reward=%.2f, bubble_reduction=%.1f%%",
        eval_results["mean_reward"],
        eval_results["mean_bubble_reduction"],
    )

    # --- Stage 3: MARL evaluation ---
    logger.info("Stage 3: MARL spanwise evaluation")
    marl_env = MARLWrapper(env, n_span_agents=4, communication_radius=1)
    marl_obs = marl_env.reset()

    # Use trained policy (shared) for all span agents
    marl_policy = PPOAgent(
        # Create a temporary env with MARL obs dim
        type('TempEnv', (), {
            'obs_dim': marl_env.obs_dim,
            'action_dim': marl_env.action_dim,
            'max_blowing': marl_env.max_blowing,
            'max_steps': marl_env.max_steps,
        })(),
        config=TrainingConfig(hidden_size=hidden_size),
    )

    # Run a few MARL episodes
    marl_rewards = []
    for ep in range(3):
        obs_list = marl_env.reset()
        ep_reward = 0
        for t in range(marl_env.max_steps):
            actions = [
                marl_policy.get_action(obs, deterministic=True)
                for obs in obs_list
            ]
            obs_list, rewards, terminated, truncated, info = marl_env.step(actions)
            ep_reward += info["global_reward"]
            if terminated or truncated:
                break
        marl_rewards.append(ep_reward)

    marl_results = {
        "mean_reward": float(np.mean(marl_rewards)),
        "n_span_agents": marl_env.n_span_agents,
        "communication_radius": marl_env.comm_radius,
    }
    logger.info("MARL eval: mean_reward=%.2f (4 span agents)", marl_results["mean_reward"])

    # --- Stage 4: Grid transfer ---
    logger.info("Stage 4: Zero-shot transfer to fine grid")
    fine_env = NACA0012Env(alpha_deg=15.0, n_actuators=5, grid_level="fine")
    transfer = GridTransferManager(agent, env, fine_env)
    transfer_results = transfer.evaluate_transfer(n_episodes=5)

    logger.info(
        "Transfer: coarse_reward=%.2f → fine_reward=%.2f (efficiency=%.1f%%)",
        transfer_results["source_reward"],
        transfer_results["target_reward"],
        transfer_results["transfer_efficiency"] * 100,
    )

    eval_results["marl"] = marl_results

    return {
        "history": history,
        "eval": eval_results,
        "transfer": transfer_results,
        "config": config,
        "train_time_s": train_time,
    }


def main():
    parser = argparse.ArgumentParser(
        description="DRL Active Flow Control Training",
    )
    parser.add_argument(
        "--case", type=str, default="all",
        choices=["wall_hump", "naca0012", "all"],
        help="Which case to train (default: all)",
    )
    parser.add_argument(
        "--timesteps", type=int, default=10000,
        help="Total training timesteps per case (default: 10000)",
    )
    parser.add_argument(
        "--hidden-size", type=int, default=32,
        help="Hidden layer size for policy network (default: 32)",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    report = DRLTrainingReport(output_dir=str(RESULTS_DIR))

    logger.info("DRL Active Flow Control Training")
    logger.info("Cases: %s | Timesteps: %d | Hidden: %d",
                args.case, args.timesteps, args.hidden_size)

    # --- Wall Hump ---
    if args.case in ("wall_hump", "all"):
        wh_results = train_wall_hump(
            total_timesteps=args.timesteps,
            hidden_size=args.hidden_size,
        )
        report.add_case_result(
            "wall_hump",
            wh_results["history"],
            wh_results["eval"],
            transfer_results=wh_results["transfer"],
            config=wh_results["config"],
        )

    # --- NACA 0012 ---
    if args.case in ("naca0012", "all"):
        naca_results = train_naca0012(
            total_timesteps=args.timesteps,
            hidden_size=args.hidden_size,
        )
        report.add_case_result(
            "naca0012",
            naca_results["history"],
            naca_results["eval"],
            transfer_results=naca_results["transfer"],
            config=naca_results["config"],
        )

    # --- Generate report ---
    final_report = report.generate_report()
    report.plot_training_curves()

    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)

    # Print summary
    for case_name, case_res in final_report.get("cases", {}).items():
        s = case_res.get("summary", {})
        logger.info(
            "  %s: reward=%.2f, bubble_reduction=%.1f%%, episodes=%d",
            case_name,
            s.get("final_reward", 0),
            s.get("bubble_reduction_pct", 0),
            s.get("n_episodes_trained", 0),
        )

    logger.info("Results saved to %s", RESULTS_DIR)
    return final_report


if __name__ == "__main__":
    main()
