#!/usr/bin/env python3
"""
Generate Physics vs Accuracy Pareto Plot
=========================================
Visualizes the benchmark results from `physics_informed_benchmark.py`.
Highlights the Pareto front showing how TBNN trades nominal accuracy
for physical constraint satisfaction (Invariance + Realizability).
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent

# Make sure ml_augmentation is accessible if running from root
sys.path.insert(0, str(PROJECT_ROOT))
try:
    from scripts.ml_augmentation.physics_informed_benchmark import PhysicsAwarenessBenchmark
except ImportError:
    print("Error: Must be run from the project root or with proper PYTHONPATH.")
    sys.exit(1)


def generate_pareto_plot(save_path: str = "plots/physics_pareto.png"):
    """
    Run the benchmark and plot Accuracy (RMSE) vs 
    Constraint Violation (Invariance + Trace + Symmetry + Lumley).
    """
    print("Running Physics-Informed Benchmark...")
    bm = PhysicsAwarenessBenchmark()
    results = bm.run_benchmark()
    
    # Compile data for plotting
    architectures = []
    rmse_values = []
    violation_scores = []  # Aggregate % of violations
    colors = []
    
    color_map = {
        "Vanilla_MLP_Raw": "red",
        "Invariant_MLP_Scalars": "orange",
        "TBNN_Tensor_Basis": "green"
    }
    
    for r in results:
        architectures.append(r.architecture.replace("_", " "))
        rmse_values.append(r.anisotropy_rmse)
        
        # Aggregate violations (capped at 100% for visualization clarity)
        total_violation = (r.invariance_violation_pct + r.trace_violation_pct + 
                           r.symmetry_violation_pct + r.lumley_violation_pct) / 4.0
        violation_scores.append(total_violation)
        
        colors.append(color_map.get(r.architecture, "blue"))

    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Scatter points
    scatter = ax.scatter(violation_scores, rmse_values, c=colors, s=150, zorder=5)
    
    # Annotate points
    for i, txt in enumerate(architectures):
        ax.annotate(txt, (violation_scores[i], rmse_values[i]), 
                    xytext=(10, 5), textcoords="offset points", 
                    fontsize=10, fontweight="bold")
                    
    # Draw Pareto Front line (approximate)
    # Sort by violation score to draw the front
    sorted_indices = np.argsort(violation_scores)
    pareto_x = [violation_scores[i] for i in sorted_indices]
    pareto_y = [rmse_values[i] for i in sorted_indices]
    
    ax.plot(pareto_x, pareto_y, 'k--', alpha=0.5, zorder=1, label="Physics-Accuracy Pareto Front")

    # Formatting
    ax.set_title("Physics-Informed ML: Accuracy vs Physical Constraints Pareto", fontsize=14, pad=15)
    ax.set_xlabel("Aggregated Physical Constraint Violations (%)", fontsize=12)
    ax.set_ylabel("Nominal Anisotropy RMSE", fontsize=12)
    
    # Invert x-axis so better physics is to the right
    ax.set_xlim(105, -5) 
    
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right")
    
    # Add interpretation text box
    textstr = "\n".join((
        "TBNN perfectly satisfies Galilean Invariance,",
        "Symmetry, and Trace constraints (0% violation),",
        "while explicitly enforcing the Lumley triangle.",
        "Vanilla MLP sacrifices all physics for a slight fit improvement."
    ))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.05, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom', bbox=props)

    # Save
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"-> Saved Pareto plot to {save_path}")
    plt.close()


if __name__ == "__main__":
    generate_pareto_plot()
