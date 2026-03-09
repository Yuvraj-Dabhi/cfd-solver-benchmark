#!/usr/bin/env python3
"""
DRL Active Flow Control Harness
===============================
Trains a Deep Reinforcement Learning (PPO) agent to suppress
flow separation over the NASA wall-mounted hump using synthetic jets.

This script demonstrates zero-shot transfer:
1. Trains on a coarse grid (for computational speed).
2. Transfers the policy zero-shot to a fine grid.
3. Evaluates and plots the results against baselines.
"""

import sys
import logging
import argparse
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.ml_augmentation.drl_flow_control import (
    WallHumpEnv,
    PPOAgent,
    TrainingConfig,
    CurriculumScheduler,
    GridTransferManager,
    BaselineComparison
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="DRL Flow Control Benchmark")
    parser.add_argument("--timesteps", type=int, default=5000,
                        help="Total training timesteps (default: 5000 for demo)")
    parser.add_argument("--actuators", type=int, default=5,
                        help="Number of ZNMF synthetic jets")
    parser.add_argument("--baseline-only", action="store_true",
                        help="Skip DRL training and just run baselines")
    return parser.parse_args()


def run_training_and_transfer():
    args = parse_args()
    output_dir = Path("results/drl_hump")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best_ppo_model.json"
    
    # 1. Setup Coarse Environment for Training
    logger.info("Initializing Coarse Grid Environment...")
    env_coarse = WallHumpEnv(n_actuators=args.actuators, grid_level="coarse")
    
    if not args.baseline_only:
        # Curriculum: start with easy reward scaling, move to strict
        curriculum = CurriculumScheduler([
            {"name": "warmup", "episode_threshold": 0, "params": {"reward_scaling": 0.1}},
            {"name": "strict", "episode_threshold": 20, "params": {"reward_scaling": 1.0}}
        ])
        
        config = TrainingConfig(
            algorithm="ppo",
            total_timesteps=args.timesteps,
            batch_size=1024,
            mini_batch_size=256,
            n_epochs=4,
            lr=3e-4,
            checkpoint_freq=10,
            log_dir=str(output_dir)
        )
        
        logger.info(f"Starting PPO Agent Training ({args.timesteps} timesteps)...")
        agent = PPOAgent(env_coarse, config=config, seed=42)
        
        # In a real environment, you'd integrate the curriculum into the training loop,
        # but the agent will handle basic training.
        history = agent.train()
        
        logger.info(f"Saving checkpoint to {checkpoint_path}")
        agent.save_checkpoint(str(checkpoint_path))
    else:
        logger.info("Skipping DRL training...")
        if not checkpoint_path.exists():
            logger.error("No checkpoint found for evaluation!")
            return
        
        # Load existing policy
        agent = PPOAgent(env_coarse, config=TrainingConfig(total_timesteps=0))
        agent.load_checkpoint(str(checkpoint_path))

    # 2. Setup Fine Environment for Zero-Shot Transfer
    logger.info("Initializing Fine Grid Environment for Evaluation...")
    env_fine = WallHumpEnv(n_actuators=args.actuators, grid_level="fine")
    
    # 3. Perform Grid Transfer Validation
    logger.info("Evaluating zero-shot transfer onto fine grid...")
    transfer_mgr = GridTransferManager(agent, source_env=env_coarse, target_env=env_fine)
    transfer_results = transfer_mgr.evaluate_transfer(n_episodes=3)
    
    logger.info(f"Transfer Results: mean_reward={transfer_results['target_reward']:.3f}, bubble_reduction={transfer_results['target_bubble_reduction']:.3f}")
    
    # 4. Compare Against Classical Baselines
    logger.info("Running Baseline Comparison...")
    baseline_comp = BaselineComparison(env_fine)
    res_unc = baseline_comp.evaluate_uncontrolled()
    res_const = baseline_comp.evaluate_constant_blowing()
    res_period = baseline_comp.evaluate_periodic_forcing(frequency=0.1)
    
    from tabulate import tabulate
    
    table_data = [
        ["Uncontrolled", res_unc.get("bubble_length", 0)],
        ["Constant Blowing", res_const.get("bubble_length", 0)],
        ["Periodic Forcing", res_period.get("bubble_length", 0)],
        ["DRL (Zero-Shot)", transfer_results.get("target_bubble_reduction", 0)]
    ]
    
    print("\n" + "="*50)
    print("      Turbulent Separation Bubble (TSB) Length")
    print("="*50)
    print(tabulate(table_data, headers=["Strategy", "TSB Length"], floatfmt=".4f"))
    print("="*50 + "\n")
    
    # Save comparison data
    with open(output_dir / "baseline_comparison.json", "w") as f:
        import json
        out_data = {
            "transfer": transfer_results,
            "baselines": {
                "uncontrolled": res_unc,
                "constant_blowing": res_const,
                "periodic_forcing": res_period
            }
        }
        json.dump(out_data, f, indent=2)

if __name__ == "__main__":
    run_training_and_transfer()
