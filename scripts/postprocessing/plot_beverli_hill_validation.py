#!/usr/bin/env python3
"""
BeVERLI Hill — Validation Plotting
====================================
Generates validation comparison plots for the BeVERLI Hill 3D
smooth-body separation benchmark.

Plots:
  1. Centerline Cp distribution: CFD vs experiment
  2. Centerline Cf distribution: CFD vs experiment
  3. Multi-yaw comparison panels (0° / 30° / 45°)
  4. SA vs SST model comparison overlays
  5. Velocity profiles at PIV stations (U, V, W)
  6. Reynolds stress profiles at PIV stations

Usage:
  python plot_beverli_hill_validation.py
  python plot_beverli_hill_validation.py --results-dir runs/beverli_hill
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[WARN] matplotlib not available — skipping plots")

# Style
COLORS = {
    "exp": "#333333",
    "SA": "#2196F3",
    "SST": "#E91E63",
    "SA_fill": "#BBDEFB",
    "SST_fill": "#F8BBD0",
}
MARKERS = {"exp": "o", "SA": "s", "SST": "^"}


def load_experimental_data():
    """Load representative BeVERLI experimental data."""
    from experimental_data.data_loader import load_case
    return load_case("beverli_hill")


def load_cfd_results(results_dir: Path, model: str, re: int,
                      yaw: int, grid: str = "medium") -> dict:
    """Load CFD results from a completed simulation case."""
    case_name = f"beverli_{model}_Re{re // 1000}k_yaw{yaw}_{grid}"
    case_dir = results_dir / case_name

    result = {"case_dir": case_dir, "exists": False}

    surface_csv = case_dir / "surface_flow.csv"
    if surface_csv.exists():
        try:
            import pandas as pd
            df = pd.read_csv(surface_csv)
            result["surface_data"] = df
            result["exists"] = True
        except Exception:
            pass

    history_csv = case_dir / "history.csv"
    if history_csv.exists():
        try:
            import pandas as pd
            df = pd.read_csv(history_csv)
            result["history"] = df
            result["exists"] = True
        except Exception:
            pass

    return result


def plot_centerline_cp(exp_data, output_dir: Path, cfd_results: dict = None):
    """Plot centerline Cp distribution: experiment vs CFD."""
    if not HAS_MPL:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle("BeVERLI Hill — Centerline Pressure Coefficient",
                 fontsize=14, fontweight='bold')

    # Experimental data
    if exp_data.wall_data is not None:
        x = exp_data.wall_data["x_H"].values
        Cp = exp_data.wall_data["Cp"].values
        ax.plot(x, Cp, 'o', color=COLORS["exp"], markersize=3,
                label="Experiment", alpha=0.6, zorder=5)

    # CFD results overlay
    if cfd_results:
        for model, data in cfd_results.items():
            if data.get("exists") and "surface_data" in data:
                df = data["surface_data"]
                if "x" in df.columns and "Pressure_Coefficient" in df.columns:
                    ax.plot(df["x"].values / 0.1869,
                            df["Pressure_Coefficient"].values,
                            '-', color=COLORS.get(model, "#999"),
                            linewidth=2, label=f"{model}")

    # Hill profile overlay
    x_hill = np.linspace(-3, 3, 200)
    from run_beverli_hill import hill_centerline_profile
    y_hill = hill_centerline_profile(x_hill)
    ax2 = ax.twinx()
    ax2.fill_between(x_hill, 0, y_hill, alpha=0.08, color='gray')
    ax2.set_ylim(-0.2, 3.0)
    ax2.set_ylabel("Hill surface y/H", fontsize=10, color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')

    ax.set_xlabel("x / H", fontsize=12)
    ax.set_ylabel("Cp", fontsize=12)
    ax.set_xlim(-5, 10)
    ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()

    plt.tight_layout()
    out_file = output_dir / "beverli_cp_centerline.png"
    fig.savefig(out_file, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  [PLOT]   {out_file.name}")


def plot_centerline_cf(exp_data, output_dir: Path, cfd_results: dict = None):
    """Plot centerline Cf distribution with separation bubble highlighted."""
    if not HAS_MPL:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle("BeVERLI Hill — Centerline Skin Friction Coefficient",
                 fontsize=14, fontweight='bold')

    if exp_data.wall_data is not None:
        x = exp_data.wall_data["x_H"].values
        Cf = exp_data.wall_data["Cf"].values
        ax.plot(x, Cf, 'o', color=COLORS["exp"], markersize=3,
                label="Experiment (OFI)", alpha=0.6, zorder=5)

        # Highlight separation bubble
        sep_mask = Cf < 0
        if np.any(sep_mask):
            ax.fill_between(x, 0, Cf, where=sep_mask,
                            color='red', alpha=0.1, label="Reversed flow")

    if cfd_results:
        for model, data in cfd_results.items():
            if data.get("exists") and "surface_data" in data:
                df = data["surface_data"]
                if "x" in df.columns and "Skin_Friction_Coefficient" in df.columns:
                    ax.plot(df["x"].values / 0.1869,
                            df["Skin_Friction_Coefficient"].values,
                            '-', color=COLORS.get(model, "#999"),
                            linewidth=2, label=f"{model}")

    ax.set_xlabel("x / H", fontsize=12)
    ax.set_ylabel("Cf", fontsize=12)
    ax.set_xlim(-5, 10)
    ax.axhline(y=0, color='red', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Annotate separation metrics
    metrics = exp_data.separation_metrics
    if "x_sep_xH" in metrics and "x_reat_xH" in metrics:
        ax.annotate(f"Sep: x/H = {metrics['x_sep_xH']:.1f}",
                    xy=(metrics['x_sep_xH'], 0), fontsize=9,
                    color='red', ha='center',
                    xytext=(metrics['x_sep_xH'], 0.002),
                    arrowprops=dict(arrowstyle='->', color='red'))
        ax.annotate(f"Reat: x/H = {metrics['x_reat_xH']:.1f}",
                    xy=(metrics['x_reat_xH'], 0), fontsize=9,
                    color='green', ha='center',
                    xytext=(metrics['x_reat_xH'], 0.002),
                    arrowprops=dict(arrowstyle='->', color='green'))

    plt.tight_layout()
    out_file = output_dir / "beverli_cf_centerline.png"
    fig.savefig(out_file, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  [PLOT]   {out_file.name}")


def plot_velocity_profiles(exp_data, output_dir: Path):
    """Plot velocity profiles at PIV stations."""
    if not HAS_MPL:
        return
    if not exp_data.velocity_profiles:
        return

    stations = sorted(exp_data.velocity_profiles.keys())
    n_stations = len(stations)

    fig, axes = plt.subplots(1, n_stations, figsize=(3.5 * n_stations, 6),
                              sharey=True)
    fig.suptitle("BeVERLI Hill — Velocity Profiles at PIV Stations (0° yaw)",
                 fontsize=14, fontweight='bold')

    if n_stations == 1:
        axes = [axes]

    for i, station in enumerate(stations):
        ax = axes[i]
        df = exp_data.velocity_profiles[station]

        y_H = df["y_H"].values if "y_H" in df.columns else df["y"].values / 0.1869
        U_ref = 20.0  # Reference velocity

        if "U" in df.columns:
            ax.plot(df["U"].values / U_ref, y_H, '-',
                    color=COLORS["exp"], linewidth=1.5, label="U/U∞")
        if "V" in df.columns:
            ax.plot(df["V"].values / U_ref, y_H, '--',
                    color='#4CAF50', linewidth=1.2, label="V/U∞")
        if "W" in df.columns:
            ax.plot(df["W"].values / U_ref, y_H, ':',
                    color='#FF9800', linewidth=1.2, label="W/U∞")

        ax.axvline(x=0, color='gray', linewidth=0.5, linestyle='--')
        ax.set_xlabel("U/U∞", fontsize=10)
        ax.set_title(f"x/H = {station}", fontsize=11)
        ax.set_xlim(-0.3, 1.3)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.set_ylabel("y / H", fontsize=11)
            ax.legend(fontsize=8, loc='upper left')

    plt.tight_layout()
    out_file = output_dir / "beverli_velocity_profiles.png"
    fig.savefig(out_file, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  [PLOT]   {out_file.name}")


def plot_reynolds_stresses(exp_data, output_dir: Path):
    """Plot Reynolds stress profiles at PIV stations."""
    if not HAS_MPL:
        return
    if not exp_data.velocity_profiles:
        return

    stations = sorted(exp_data.velocity_profiles.keys())
    n_stations = len(stations)

    fig, axes = plt.subplots(1, n_stations, figsize=(3.5 * n_stations, 6),
                              sharey=True)
    fig.suptitle("BeVERLI Hill — Reynolds Stresses at PIV Stations",
                 fontsize=14, fontweight='bold')

    if n_stations == 1:
        axes = [axes]

    U_ref = 20.0

    for i, station in enumerate(stations):
        ax = axes[i]
        df = exp_data.velocity_profiles[station]
        y_H = df["y_H"].values if "y_H" in df.columns else df["y"].values / 0.1869

        if "uu" in df.columns:
            ax.plot(df["uu"].values / U_ref**2, y_H, '-',
                    color='#F44336', linewidth=1.5, label="u'u'/U²∞")
        if "vv" in df.columns:
            ax.plot(df["vv"].values / U_ref**2, y_H, '--',
                    color='#2196F3', linewidth=1.2, label="v'v'/U²∞")
        if "uv" in df.columns:
            ax.plot(df["uv"].values / U_ref**2, y_H, ':',
                    color='#9C27B0', linewidth=1.2, label="-u'v'/U²∞")

        ax.set_xlabel("Stress / U²∞", fontsize=10)
        ax.set_title(f"x/H = {station}", fontsize=11)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.set_ylabel("y / H", fontsize=11)
            ax.legend(fontsize=8, loc='upper right')

    plt.tight_layout()
    out_file = output_dir / "beverli_reynolds_stresses.png"
    fig.savefig(out_file, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  [PLOT]   {out_file.name}")


def plot_model_comparison_summary(output_dir: Path):
    """Summary comparison table showing known SA vs SST failure modes."""
    if not HAS_MPL:
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis('off')
    fig.suptitle("BeVERLI Hill — Expected RANS Model Failures",
                 fontsize=14, fontweight='bold')

    table_data = [
        ["Yaw", "Re_H", "SA Prediction", "SST Prediction", "Status"],
        ["0°", "250k", "Separation onset ~0.3H too late",
         "Bubble extent ~20% too large", "Both fail"],
        ["0°", "650k", "Separation onset error reduced",
         "Better but overpredicts extent", "Both fail"],
        ["30°", "250k", "Skewed BL poorly captured",
         "Asymmetric effects partially captured", "Both fail"],
        ["30°", "650k", "Similar to 250k, thinner BL",
         "Asymmetric effects worsen", "Both fail"],
        ["45°", "250k", "Volume extent wrong",
         "Erroneous asymmetric wake", "SST critical"],
        ["45°", "650k", "Volume extent wrong",
         "Erroneous asymmetric wake", "SST critical"],
    ]

    colors = [['#E3F2FD'] * 5] + [
        ['white', 'white', '#BBDEFB', '#F8BBD0',
         '#FFCDD2' if 'critical' in row[4].lower() else '#FFF9C4']
        for row in table_data[1:]
    ]

    table = ax.table(cellText=table_data, cellColours=colors,
                      loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.8)

    # Header styling
    for j in range(5):
        table[0, j].set_text_props(fontweight='bold')

    plt.tight_layout()
    out_file = output_dir / "beverli_model_comparison_table.png"
    fig.savefig(out_file, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  [PLOT]   {out_file.name}")


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="BeVERLI Hill — Validation Plotting")
    parser.add_argument("--results-dir", type=Path,
                        default=PROJECT_ROOT / "runs" / "beverli_hill",
                        help="Results directory")
    parser.add_argument("--output-dir", type=Path,
                        default=PROJECT_ROOT / "plots" / "beverli_hill",
                        help="Output plot directory")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  BeVERLI HILL — VALIDATION PLOTS")
    print("=" * 60)

    # Load experimental data
    print("\n  Loading experimental reference data...")
    exp_data = load_experimental_data()
    print(f"  {exp_data.summary()}")

    # Check for CFD results
    cfd_results = {}
    for model in ["SA", "SST"]:
        data = load_cfd_results(args.results_dir, model, 250000, 0)
        if data["exists"]:
            cfd_results[model] = data
    if cfd_results:
        print(f"\n  Found CFD results: {list(cfd_results.keys())}")
    else:
        print("\n  No CFD results found — plotting representative data only")

    # Generate plots
    print("\n  Generating validation plots...")
    plot_centerline_cp(exp_data, args.output_dir, cfd_results or None)
    plot_centerline_cf(exp_data, args.output_dir, cfd_results or None)
    plot_velocity_profiles(exp_data, args.output_dir)
    plot_reynolds_stresses(exp_data, args.output_dir)
    plot_model_comparison_summary(args.output_dir)

    print(f"\n  All plots saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
