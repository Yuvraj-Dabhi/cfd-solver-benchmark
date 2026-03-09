#!/usr/bin/env python3
"""
Deep Reinforcement Learning (DRL) Active Flow Control Benchmark
=============================================================
A concrete case study demonstrating active flow control of a 2D separated 
boundary layer (NASA Wall-Mounted Hump proxy) using PPO or SAC.

This script benchmarks a trained DRL policy against explicit open-loop
baseline strategies:
1. Baseline (No control)
2. Constant Blowing (Naive)
3. Periodic Forcing (Zero-Net-Mass-Flux Synthetic Jets)
4. DRL Agent (Closed-loop feedback — PPO or SAC)

The results are reported using established literature metrics:
- Bubble length reduction
- Turbulent Separation Bubble (TSB) area (Font et al. 2025)
- Reattachment position (x_reat) optimization
- Skin friction distribution (Cf)
- Actuation Power Spectral Density (PSD)

Optional:
  --algorithm {ppo,sac}  Select DRL algorithm (default: ppo)
  --transfer             Run coarse→fine + NACA 0012 transfer learning
  --fast                 Quick validation (20 episodes)
"""

import logging
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Attempt imports gracefully
try:
    from scripts.ml_augmentation.drl_flow_control import (
        WallHumpEnv, PPOAgent, SACAgent, TrainingConfig, ControlAction,
        compute_tsb_area, compute_actuation_psd,
        train_coarse_then_transfer, transfer_wall_hump_to_naca0012,
        RedisCFDInterface, DRLTrainingReport,
    )
    HAS_DRL = True
except ImportError as e:
    HAS_DRL = False
    print(f"Warning: DRL modules not found or missing dependencies: {e}")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("drl_benchmark")


class ConstantBlowingPolicy:
    """Naive open-loop strategy: constant maximum blowing from all actuators."""
    def __init__(self, action_dim: int, max_velocity: float = 1.0):
        self.action_dim = action_dim
        self.max_velocity = max_velocity

    def get_action(self, obs: np.ndarray, deterministic: bool = True) -> tuple:
        # PPO outputs in [-1, 1], env maps to physical limits.
        # Constant max blowing -> action = +1.0 for all actuators
        action = np.ones(self.action_dim, dtype=np.float32)
        return action, None


class PeriodicForcingPolicy:
    """
    Zero-Net-Mass-Flux (ZNMF) Synthetic Jet strategy.
    Translates a sine wave across the actuator array.
    """
    def __init__(self, action_dim: int, frequency: float = 0.5):
        self.action_dim = action_dim
        self.frequency = frequency
        self.step_idx = 0

    def get_action(self, obs: np.ndarray, deterministic: bool = True) -> tuple:
        # Time-varying sine wave based on step count
        phase = 2 * np.pi * self.frequency * self.step_idx * 0.1
        action = np.sin(phase) * np.ones(self.action_dim, dtype=np.float32)
        self.step_idx += 1
        return action, None


def evaluate_policy(
    env: "WallHumpEnv", policy, episodes: int = 5, name: str = "Policy",
    collect_actions: bool = False,
) -> dict:
    """Runs the environment for N episodes, returns aggregate metrics + TSB area."""
    bubble_lengths = []
    x_reats = []
    cf_mins = []
    rewards = []
    tsb_areas = []
    all_actions = []

    for ep in range(episodes):
        obs_reset = env.reset()
        obs = obs_reset[0] if isinstance(obs_reset, tuple) else obs_reset
        done = False
        ep_reward = 0.0
        ep_actions = []
        
        # Reset periodic step if applicable
        if hasattr(policy, 'step_idx'):
            policy.step_idx = 0
            
        while not done:
            action, _ = policy.get_action(obs, deterministic=True)
            ep_actions.append(action.copy() if hasattr(action, 'copy') else action)
            step_result = env.step(action)
            if len(step_result) == 5:
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                obs, reward, done, info = step_result
            ep_reward += reward
            
        # Terminal state metrics
        bubble_lengths.append(env.current_state.bubble_length)
        x_reats.append(env.current_state.x_reat)
        cf_mins.append(np.min(env.current_state.Cf))
        rewards.append(ep_reward)
        tsb_areas.append(env.current_state.tsb_area)
        if collect_actions and ep_actions:
            all_actions.extend(ep_actions)

    metrics = {
        "mean_bubble_length": float(np.mean(bubble_lengths)),
        "mean_x_reat": float(np.mean(x_reats)),
        "mean_cf_min": float(np.mean(cf_mins)),
        "mean_reward": float(np.mean(rewards)),
        "mean_tsb_area": float(np.mean(tsb_areas)),
    }

    if collect_actions and all_actions:
        action_array = np.array(all_actions)
        freqs, psd = compute_actuation_psd(action_array, dt=1.0)
        if len(freqs) > 1:
            peak_idx = np.argmax(psd[1:]) + 1  # Skip DC
            metrics["psd_peak_freq"] = float(freqs[peak_idx])
            metrics["psd_peak_power"] = float(psd[peak_idx])
        metrics["_action_history"] = action_array
    
    logger.info(f"[{name}] L_bubble: {metrics['mean_bubble_length']:.4f} "
                f"| TSB: {metrics['mean_tsb_area']:.6f} "
                f"| X_reat: {metrics['mean_x_reat']:.4f} "
                f"| Reward: {metrics['mean_reward']:.2f}")
    
    return metrics


def plot_learning_curve(history: dict, output_path: str, algorithm: str = "PPO"):
    """Plot training convergence."""
    plt.figure(figsize=(10, 6))
    if 'rewards' in history:
        plt.plot(history['episodes'], history['rewards'], 'b-', alpha=0.3)
        # Moving average
        window = 10
        if len(history['rewards']) >= window:
            running_mean = np.convolve(history['rewards'], np.ones(window)/window, mode='valid')
            plt.plot(history['episodes'][window-1:], running_mean, 'b-', linewidth=2, label="10-Ep Moving Avg")
            
    plt.title(f"DRL Active Flow Control — {algorithm.upper()} Learning Curve (Wall Hump)")
    plt.xlabel("Episodes")
    plt.ylabel("Cumulative Reward")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Only add legend if plotting moving average
    if len(history.get('rewards', [])) >= 10:
        plt.legend()
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved learning curve to {output_path}")


def plot_control_comparison(env, baseline_state, const_state, drl_state, out_dir):
    """Plot the final converged Cf profiles under different strategies."""
    plt.figure(figsize=(10, 5))
    
    x = env.x_wall
    plt.plot(x, baseline_state.Cf, 'k--', linewidth=2, label="Baseline (No Control)")
    plt.plot(x, const_state.Cf, 'r-', alpha=0.6, label="Constant Blowing")
    plt.plot(x, drl_state.Cf, 'b-', linewidth=2, label="DRL Policy")
    
    plt.axhline(0, color='gray', linestyle=':')
    
    # Mark actuators
    for act_x in env.actuator_locations:
        plt.axvline(act_x, color='g', linestyle=':', alpha=0.3)
    
    plt.xlim(0.4, 1.2)
    plt.xlabel("x/c")
    plt.ylabel("Skin Friction Coefficient ($C_f$)")
    plt.title("Impact of Active Flow Control on Wall Hump Separation")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path = out_dir / "control_comparison.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    logger.info(f"Saved Cf comparison plot to {plot_path}")


def plot_tsb_comparison(all_metrics: dict, out_dir: Path):
    """Bar chart of TSB area across control strategies."""
    names = list(all_metrics.keys())
    tsb_values = [m["mean_tsb_area"] for m in all_metrics.values()]

    colors = ['#2c3e50', '#e74c3c', '#3498db', '#27ae60'][:len(names)]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(names, tsb_values, color=colors, edgecolor='white', linewidth=1.2)

    # Value labels on bars
    for bar, val in zip(bars, tsb_values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.00005,
                 f"{val:.5f}", ha='center', va='bottom', fontsize=9)

    plt.ylabel("TSB Area (∫|Cf| dx)")
    plt.title("Turbulent Separation Bubble Area — Control Strategy Comparison")
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    plot_path = out_dir / "tsb_comparison.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    logger.info(f"Saved TSB comparison plot to {plot_path}")


def plot_actuation_psd(drl_metrics: dict, periodic_metrics: dict, out_dir: Path):
    """PSD of DRL vs periodic forcing actuation signals."""
    plt.figure(figsize=(10, 5))

    for label, metrics, color in [
        ("DRL Agent", drl_metrics, 'tab:blue'),
        ("Periodic Forcing", periodic_metrics, 'tab:orange'),
    ]:
        if "_action_history" in metrics:
            freqs, psd = compute_actuation_psd(metrics["_action_history"], dt=1.0)
            if len(freqs) > 1:
                plt.semilogy(freqs[1:], psd[1:], label=label, color=color, linewidth=1.5)

    plt.xlabel("Frequency (1/Δt)")
    plt.ylabel("Power Spectral Density")
    plt.title("Actuation Signal PSD — DRL vs Periodic Forcing")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_path = out_dir / "actuation_psd.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    logger.info(f"Saved actuation PSD plot to {plot_path}")


def main():
    parser = argparse.ArgumentParser(description="DRL Flow Control Benchmark")
    parser.add_argument("--fast", action="store_true", help="Run short training for quick validation")
    parser.add_argument("--episodes", type=int, default=500, help="Number of training episodes")
    parser.add_argument("--algorithm", choices=["ppo", "sac"], default="ppo",
                        help="DRL algorithm (default: ppo)")
    parser.add_argument("--transfer", action="store_true",
                        help="Run coarse→fine + NACA 0012 transfer learning")
    args = parser.parse_args()

    if not HAS_DRL:
        logger.error("Cannot run DRL benchmark: Environment/Agent dependencies missing.")
        sys.exit(1)

    algo_name = args.algorithm.upper()
    logger.info("Initializing NASA Wall-Mounted Hump Proxy Environment (5 Actuators)...")
    env = WallHumpEnv(n_actuators=5, grid_level="coarse", n_pressure_probes=8)
    
    # Keep track of terminal state for plotting
    states = {}
    all_metrics = {}
    
    # Evaluate Baseline (No control)
    logger.info("--- Evaluating Zero-Control Baseline ---")
    class ZeroPolicy:
        def get_action(self, obs, deterministic=True):
            return np.zeros(5, dtype=np.float32), None
    baseline_metrics = evaluate_policy(env, ZeroPolicy(), name="Baseline (No Control)")
    states['baseline'] = env.current_state
    all_metrics["No Control"] = baseline_metrics

    # Evaluate Constant Blowing
    logger.info("--- Evaluating Constant Blowing Strategy ---")
    const_policy = ConstantBlowingPolicy(action_dim=5)
    const_metrics = evaluate_policy(env, const_policy, name="Constant Blowing")
    states['constant'] = env.current_state
    all_metrics["Const. Blowing"] = const_metrics

    # Evaluate Periodic Forcing (Synthetic Jet)
    logger.info("--- Evaluating ZNMF Periodic Forcing ---")
    freq_policy = PeriodicForcingPolicy(action_dim=5, frequency=0.5)
    freq_metrics = evaluate_policy(env, freq_policy, name="Periodic Forcing",
                                   collect_actions=True)
    states['periodic'] = env.current_state
    all_metrics["Periodic (ZNMF)"] = freq_metrics

    # Train DRL Agent
    episodes = 20 if args.fast else args.episodes
    total_timesteps = episodes * 50
    
    logger.info(f"--- Training {algo_name} Agent ({total_timesteps} timesteps) ---")
    config = TrainingConfig(
        algorithm=args.algorithm,
        total_timesteps=total_timesteps,
        batch_size=min(512, total_timesteps),
        mini_batch_size=64,
        n_epochs=4,
        lr=5e-4,
    )
    
    if args.algorithm == "sac":
        agent = SACAgent(env, config=config)
    else:
        agent = PPOAgent(env, config=config)

    history = agent.train()
    
    # Save checkpoint
    out_dir = PROJECT_ROOT / "results" / "drl_wall_hump"
    out_dir.mkdir(parents=True, exist_ok=True)
    chkpt_path = out_dir / f"{args.algorithm}_wall_hump_policy.json"
    agent.save_checkpoint(str(chkpt_path))
    logger.info(f"Saved agent checkpoint to {chkpt_path}")

    # Evaluate Trained Agent
    logger.info(f"--- Evaluating Trained {algo_name} Agent ---")
    drl_metrics = evaluate_policy(env, agent.policy, name=f"Trained {algo_name}",
                                  collect_actions=True)
    states['drl'] = env.current_state
    all_metrics[f"DRL ({algo_name})"] = drl_metrics
    
    # Plotting
    plot_path = out_dir / "learning_curve.png"
    plot_learning_curve(history, str(plot_path), algorithm=algo_name)
    plot_control_comparison(env, states['baseline'], states['constant'], states['drl'], out_dir)
    plot_tsb_comparison(all_metrics, out_dir)
    plot_actuation_psd(drl_metrics, freq_metrics, out_dir)

    # Summary Table
    print("\n" + "=" * 100)
    print(" DRL Flow Control Benchmark Results (NASA Wall Hump)")
    print("=" * 100)
    header = (f"{'Strategy':<25} | {'Bubble Length':<14} | {'TSB Area':<14} | "
              f"{'X_reatt':<10} | {'Cf_min':<10} | {'Reward':<10}")
    print(header)
    print("-" * 100)
    
    for name, m in all_metrics.items():
        psd_info = f" | PSD_peak={m.get('psd_peak_freq', 'N/A')}" if 'psd_peak_freq' in m else ""
        row = (f"{name:<25} | {m['mean_bubble_length']:<14.4f} | "
               f"{m['mean_tsb_area']:<14.6f} | {m['mean_x_reat']:<10.4f} | "
               f"{m['mean_cf_min']:<10.5f} | {m['mean_reward']:<10.2f}{psd_info}")
        print(row)
    
    print("=" * 100)
    
    base_bubble = baseline_metrics["mean_bubble_length"]
    drl_bubble = drl_metrics["mean_bubble_length"]
    reduction = max(0, ((base_bubble - drl_bubble) / base_bubble) * 100)
    base_tsb = baseline_metrics["mean_tsb_area"]
    drl_tsb = drl_metrics["mean_tsb_area"]
    tsb_reduction = max(0, ((base_tsb - drl_tsb) / base_tsb) * 100) if base_tsb > 0 else 0
    print(f">> DRL achieved a {reduction:.1f}% reduction in separation bubble length.")
    print(f">> DRL achieved a {tsb_reduction:.1f}% reduction in TSB area.")

    # Generate JSON report
    report = DRLTrainingReport(output_dir=str(out_dir))
    report.add_case_result("wall_hump_benchmark", history, drl_metrics,
                           config=config)
    report.generate_report()

    # === Optional: Transfer Learning ===
    if args.transfer:
        print("\n" + "=" * 100)
        print(" Running Coarse→Fine Transfer + NACA 0012 Transfer Learning")
        print("=" * 100)

        transfer_timesteps = 1000 if args.fast else 5000
        transfer_results = train_coarse_then_transfer(
            n_actuators=5,
            coarse_timesteps=transfer_timesteps,
            n_pressure_probes=8,
            output_dir=str(out_dir),
        )
        print(f"\n  Coarse eval:  reward={transfer_results['coarse_eval']['mean_reward']:.2f},  "
              f"bubble_red={transfer_results['coarse_eval']['mean_bubble_reduction']:.1f}%")
        print(f"  Fine transfer: reward={transfer_results['fine_eval']['target_reward']:.2f},  "
              f"efficiency={transfer_results['fine_eval']['transfer_efficiency']:.2f}")

        # NACA 0012 transfer
        naca_results = transfer_wall_hump_to_naca0012(
            wall_hump_checkpoint=str(chkpt_path),
            n_actuators=5,
            fine_tune_timesteps=transfer_timesteps,
            output_dir=str(out_dir),
        )
        print(f"\n  NACA 0012 eval: reward={naca_results['evaluation']['mean_reward']:.2f},  "
              f"bubble_red={naca_results['evaluation']['mean_bubble_reduction']:.1f}%")
        print("=" * 100)


if __name__ == "__main__":
    main()
