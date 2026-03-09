"""
Publication-Quality Visualization
==================================
Matplotlib-based figure generation for benchmarking papers.
Produces publication-ready figures matching journal standards:
- Computers & Fluids
- AIAA Journal
- Flow, Turbulence and Combustion

All figures use consistent styling, proper axis labels, and SI units.
"""

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# =============================================================================
# Global Style
# =============================================================================
STYLE = {
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "lines.linewidth": 1.5,
    "lines.markersize": 5,
    "axes.grid": True,
    "grid.alpha": 0.3,
}
plt.rcParams.update(STYLE)

# Color palette (colorblind-friendly)
COLORS = {
    "SA": "#1f77b4",
    "SST": "#ff7f0e",
    "kEpsilon": "#2ca02c",
    "v2f": "#d62728",
    "RSM": "#9467bd",
    "DDES": "#8c564b",
    "WMLES": "#e377c2",
    "experiment": "#000000",
    "DNS": "#17becf",
}

MARKERS = {
    "SA": "o", "SST": "s", "kEpsilon": "^", "v2f": "D",
    "RSM": "v", "DDES": "P", "WMLES": "X",
    "experiment": "o", "DNS": "*",
}


def _get_color(model: str) -> str:
    return COLORS.get(model, "#7f7f7f")


def _get_marker(model: str) -> str:
    return MARKERS.get(model, "o")


# =============================================================================
# 1. Cf with Uncertainty Bands
# =============================================================================
def plot_cf_comparison(
    x_exp: np.ndarray,
    cf_exp: np.ndarray,
    cf_exp_unc: np.ndarray,
    model_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    case_name: str = "NASA Hump",
    x_label: str = "x/c",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot Cf distribution with experimental uncertainty bands.

    Parameters
    ----------
    x_exp, cf_exp, cf_exp_unc : ndarray
        Experimental x, Cf, and uncertainty.
    model_data : dict
        {model_name: (x_cfd, cf_cfd)}
    """
    fig, ax = plt.subplots(figsize=(8, 4))

    # Experimental data with uncertainty band
    ax.fill_between(x_exp, cf_exp - cf_exp_unc, cf_exp + cf_exp_unc,
                    alpha=0.2, color="gray", label="Exp. uncertainty")
    ax.plot(x_exp, cf_exp, "ko", markersize=3, label="Experiment", zorder=5)

    # CFD models
    for model, (x_cfd, cf_cfd) in model_data.items():
        ax.plot(x_cfd, cf_cfd, color=_get_color(model), label=model,
                linewidth=1.5)

    # Zero line for separation
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)

    ax.set_xlabel(x_label)
    ax.set_ylabel(r"$C_f$")
    ax.set_title(f"Skin Friction Coefficient — {case_name}")
    ax.legend(ncol=2, loc="best", framealpha=0.8)

    if save_path:
        fig.savefig(save_path)
    return fig


# =============================================================================
# 2. Velocity Profiles at Key Stations
# =============================================================================
def plot_velocity_profiles(
    stations: List[float],
    exp_data: Dict[float, Tuple[np.ndarray, np.ndarray]],
    model_data: Dict[str, Dict[float, Tuple[np.ndarray, np.ndarray]]],
    case_name: str = "BFS",
    x_label: str = "x/H",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot velocity profiles at multiple streamwise stations.

    Parameters
    ----------
    stations : list of float
        Station locations (x/H, x/c, etc.).
    exp_data : dict
        {station: (y, U)}
    model_data : dict
        {model: {station: (y, U)}}
    """
    n_stations = len(stations)
    fig, axes = plt.subplots(1, n_stations, figsize=(3 * n_stations, 5),
                             sharey=True)
    if n_stations == 1:
        axes = [axes]

    for i, station in enumerate(stations):
        ax = axes[i]

        # Experiment
        if station in exp_data:
            y_exp, U_exp = exp_data[station]
            ax.plot(U_exp, y_exp, "ko", markersize=3, label="Exp." if i == 0 else "")

        # CFD models
        for model, data in model_data.items():
            if station in data:
                y_cfd, U_cfd = data[station]
                ax.plot(U_cfd, y_cfd, color=_get_color(model),
                        label=model if i == 0 else "")

        ax.set_xlabel(r"$U/U_\infty$")
        ax.set_title(f"{x_label}={station}")

    axes[0].set_ylabel(r"$y/H$")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle(f"Velocity Profiles — {case_name}", y=1.02)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path)
    return fig


# =============================================================================
# 3. Law of the Wall
# =============================================================================
def plot_law_of_wall(
    yplus: np.ndarray,
    uplus: np.ndarray,
    model: str = "SST",
    case_name: str = "Flat Plate",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot U⁺ vs y⁺ with analytical law-of-wall references.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    # Analytical curves
    yp_vis = np.linspace(0.1, 11, 100)
    yp_log = np.linspace(11, 1e4, 200)
    ax.plot(yp_vis, yp_vis, "k--", linewidth=0.8, label=r"$U^+ = y^+$")
    ax.plot(yp_log, 2.5 * np.log(yp_log) + 5.0, "k:", linewidth=0.8,
            label=r"$U^+ = 2.5\ln y^+ + 5.0$")

    # CFD data
    ax.plot(yplus, uplus, color=_get_color(model), marker=_get_marker(model),
            markersize=3, linewidth=0, label=f"{model} (CFD)")

    ax.set_xscale("log")
    ax.set_xlim(0.1, 1e4)
    ax.set_ylim(0, 30)
    ax.set_xlabel(r"$y^+$")
    ax.set_ylabel(r"$U^+$")
    ax.set_title(f"Law of the Wall — {case_name}")
    ax.legend()

    if save_path:
        fig.savefig(save_path)
    return fig


# =============================================================================
# 4. Grid Convergence Visualization
# =============================================================================
def plot_grid_convergence(
    grid_sizes: List[int],
    phi_values: List[float],
    phi_exact: Optional[float] = None,
    quantity_name: str = r"$x_R/H$",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot quantity vs grid size with Richardson extrapolation.
    """
    fig, ax = plt.subplots(figsize=(6, 4))

    h = [1.0 / n ** 0.5 for n in grid_sizes]  # Representative grid spacing
    ax.plot(h, phi_values, "bo-", markersize=8, label="CFD")

    if phi_exact is not None:
        ax.axhline(y=phi_exact, color="r", linestyle="--",
                   label=f"Richardson extrap. = {phi_exact:.4f}")

    ax.set_xlabel(r"$h$ (representative grid spacing)")
    ax.set_ylabel(quantity_name)
    ax.set_title("Grid Convergence Study")
    ax.legend()
    ax.invert_xaxis()

    if save_path:
        fig.savefig(save_path)
    return fig


# =============================================================================
# 5. Model Comparison Heatmap
# =============================================================================
def plot_mape_heatmap(
    mape_data: pd.DataFrame,
    title: str = "MAPE (%) by Model and Case",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot heatmap of MAPE values across models and cases.

    Parameters
    ----------
    mape_data : DataFrame
        Rows = models, columns = cases, values = MAPE (%).
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    im = ax.imshow(mape_data.values, cmap="RdYlGn_r", aspect="auto",
                   vmin=0, vmax=50)
    cbar = plt.colorbar(im, ax=ax, label="MAPE (%)")

    ax.set_xticks(range(len(mape_data.columns)))
    ax.set_xticklabels(mape_data.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(mape_data.index)))
    ax.set_yticklabels(mape_data.index)
    ax.set_title(title)

    # Annotate cells
    for i in range(len(mape_data.index)):
        for j in range(len(mape_data.columns)):
            val = mape_data.values[i, j]
            color = "white" if val > 25 else "black"
            ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                    color=color, fontsize=8)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


# =============================================================================
# 6. Accuracy vs Cost Scatter
# =============================================================================
def plot_accuracy_vs_cost(
    models: List[str],
    mape: List[float],
    wall_time: List[float],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Scatter plot of accuracy (MAPE) vs computational cost (wall time).
    Lower-left = best (accurate + fast).
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    for i, model in enumerate(models):
        ax.scatter(wall_time[i], mape[i], s=100, color=_get_color(model),
                  marker=_get_marker(model), label=model, zorder=5)

    ax.set_xlabel("Wall Time [hours]")
    ax.set_ylabel("MAPE (%)")
    ax.set_title("Accuracy vs Computational Cost")
    ax.set_xscale("log")
    ax.legend(ncol=2, loc="upper right")

    # Ideal region indicator
    ax.annotate("Ideal", xy=(ax.get_xlim()[0] * 2, 5),
               fontsize=12, color="green", alpha=0.5)

    if save_path:
        fig.savefig(save_path)
    return fig


# =============================================================================
# 7. Streamline + Contour Plot
# =============================================================================
def plot_contour_with_streamlines(
    x: np.ndarray,
    y: np.ndarray,
    U: np.ndarray,
    V: np.ndarray,
    scalar: np.ndarray,
    scalar_name: str = r"$U/U_\infty$",
    case_name: str = "",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot scalar contours overlaid with streamlines.

    Parameters
    ----------
    x, y : ndarray (2D meshgrid)
    U, V : ndarray (2D velocity fields)
    scalar : ndarray (2D scalar field to contour)
    """
    fig, ax = plt.subplots(figsize=(10, 4))

    # Contour
    levels = np.linspace(np.min(scalar), np.max(scalar), 30)
    cf = ax.contourf(x, y, scalar, levels=levels, cmap="coolwarm")
    plt.colorbar(cf, ax=ax, label=scalar_name, shrink=0.8)

    # Streamlines
    speed = np.sqrt(U ** 2 + V ** 2)
    lw = 1.5 * speed / (speed.max() + 1e-10)
    ax.streamplot(x, y, U, V, color="k", linewidth=lw, density=1.5,
                  arrowsize=0.8)

    ax.set_xlabel(r"$x/H$")
    ax.set_ylabel(r"$y/H$")
    ax.set_title(f"Flow Field — {case_name}")
    ax.set_aspect("equal")

    if save_path:
        fig.savefig(save_path)
    return fig


# =============================================================================
# 8. Sensitivity Analysis Bar Chart
# =============================================================================
def plot_sobol_indices(
    names: List[str],
    S1: np.ndarray,
    ST: np.ndarray,
    title: str = "Sobol Sensitivity Indices",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Bar chart of first-order and total-order Sobol indices.
    """
    fig, ax = plt.subplots(figsize=(8, 4))

    x = np.arange(len(names))
    width = 0.35

    ax.bar(x - width / 2, S1, width, label=r"$S_1$ (first-order)", color="#1f77b4")
    ax.bar(x + width / 2, ST, width, label=r"$S_T$ (total-order)", color="#ff7f0e")

    ax.axhline(y=0.1, color="red", linestyle="--", linewidth=0.8,
               label="Critical threshold (0.1)")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("Sensitivity Index")
    ax.set_title(title)
    ax.legend()

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


# =============================================================================
# Close all figures utility
# =============================================================================
def close_all():
    """Close all matplotlib figures."""
    plt.close("all")
