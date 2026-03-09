"""
Cross-Solver Comparison & Visualization
========================================
Model ranking, ANOVA/Tukey HSD statistical significance testing,
workshop-style scatter bands, heatmaps, and publication-quality plots.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import Dict, List, Optional, Tuple

# Publication style
mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


# =============================================================================
# Model Ranking
# =============================================================================
def rank_models(
    results: Dict[str, Dict[str, float]],
    metric: str = "MAPE",
) -> pd.DataFrame:
    """
    Rank turbulence models by a given metric across cases.

    Parameters
    ----------
    results : dict
        {case_name: {model_name: metric_value}} for each case.
    metric : str
        Metric name for ranking (used as column label).

    Returns
    -------
    DataFrame with models ranked per case and overall.
    """
    rows = []
    for case, model_scores in results.items():
        for model, score in model_scores.items():
            rows.append({"Case": case, "Model": model, metric: score})

    df = pd.DataFrame(rows)

    # Rank per case
    df["Rank"] = df.groupby("Case")[metric].rank(method="min")

    # Overall average rank
    avg_rank = df.groupby("Model")["Rank"].mean().sort_values()

    return df, avg_rank


# =============================================================================
# ANOVA / Tukey HSD
# =============================================================================
def anova_model_comparison(
    results: Dict[str, Dict[str, float]],
    metric: str = "MAPE",
) -> Dict:
    """
    One-way ANOVA + Tukey HSD post-hoc test for model significance.

    Parameters
    ----------
    results : dict
        {case_name: {model_name: metric_value}}.

    Returns
    -------
    dict with F-statistic, p-value, and Tukey HSD pairwise comparisons.
    """
    from scipy import stats

    # Organize data: list of arrays, one per model
    model_data: Dict[str, List[float]] = {}
    for case, model_scores in results.items():
        for model, score in model_scores.items():
            model_data.setdefault(model, []).append(score)

    groups = list(model_data.values())
    model_names = list(model_data.keys())

    if len(groups) < 2:
        return {"error": "Need at least 2 models for ANOVA"}

    # One-way ANOVA
    f_stat, p_value = stats.f_oneway(*groups)

    result = {
        "F_statistic": float(f_stat),
        "p_value": float(p_value),
        "significant": p_value < 0.05,
        "n_models": len(model_names),
        "model_names": model_names,
    }

    # Tukey HSD (if significant)
    if p_value < 0.05:
        try:
            from statsmodels.stats.multicomp import pairwise_tukeyhsd

            all_values = []
            all_labels = []
            for name, values in model_data.items():
                all_values.extend(values)
                all_labels.extend([name] * len(values))

            tukey = pairwise_tukeyhsd(all_values, all_labels, alpha=0.05)
            result["tukey_summary"] = str(tukey)
            result["tukey_reject"] = tukey.reject.tolist()
        except ImportError:
            result["tukey_summary"] = "statsmodels not installed"

    return result


# =============================================================================
# Visualization
# =============================================================================
def plot_model_comparison_heatmap(
    results: Dict[str, Dict[str, float]],
    metric: str = "MAPE",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Heatmap of MAPE × model × case.
    """
    # Build matrix
    cases = sorted(results.keys())
    all_models = sorted(set(m for scores in results.values() for m in scores))

    matrix = np.full((len(cases), len(all_models)), np.nan)
    for i, case in enumerate(cases):
        for j, model in enumerate(all_models):
            matrix[i, j] = results[case].get(model, np.nan)

    fig, ax = plt.subplots(figsize=(max(8, len(all_models) * 1.2), max(4, len(cases) * 0.6)))
    im = ax.imshow(matrix, cmap="RdYlGn_r", aspect="auto")

    ax.set_xticks(range(len(all_models)))
    ax.set_xticklabels(all_models, rotation=45, ha="right")
    ax.set_yticks(range(len(cases)))
    ax.set_yticklabels(cases)

    # Annotate cells
    for i in range(len(cases)):
        for j in range(len(all_models)):
            val = matrix[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                        fontsize=9, color="white" if val > 20 else "black")

    plt.colorbar(im, label=f"{metric} (%)")
    ax.set_title(f"Turbulence Model Comparison: {metric}")

    if save_path:
        fig.savefig(save_path)
    return fig


def plot_accuracy_vs_cost(
    model_accuracy: Dict[str, float],
    model_walltime: Dict[str, float],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Scatter plot of accuracy vs computational cost for model selection guidance.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    models = sorted(model_accuracy.keys())
    acc = [model_accuracy[m] for m in models]
    cost = [model_walltime[m] for m in models]

    scatter = ax.scatter(cost, acc, s=100, c=acc, cmap="RdYlGn_r",
                         edgecolors="black", zorder=5)

    for m, x, y in zip(models, cost, acc):
        ax.annotate(m, (x, y), textcoords="offset points",
                    xytext=(5, 5), fontsize=9)

    ax.set_xlabel("Wall-Clock Time (relative)")
    ax.set_ylabel("MAPE (%)")
    ax.set_title("Accuracy vs. Computational Cost")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, label="MAPE (%)")

    if save_path:
        fig.savefig(save_path)
    return fig


def plot_profile_comparison(
    y: np.ndarray,
    profiles: Dict[str, np.ndarray],
    exp_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    exp_uncertainty: Optional[np.ndarray] = None,
    xlabel: str = "U/U_ref",
    ylabel: str = "y/H",
    title: str = "",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot velocity profiles for multiple models vs experiment.
    """
    fig, ax = plt.subplots(figsize=(6, 8))

    # Experimental data with uncertainty band
    if exp_data is not None:
        y_exp, u_exp = exp_data
        ax.plot(u_exp, y_exp, "ko", markersize=4, label="Experiment", zorder=10)
        if exp_uncertainty is not None:
            ax.fill_betweenx(y_exp, u_exp - exp_uncertainty, u_exp + exp_uncertainty,
                             alpha=0.15, color="gray", label="Exp. uncertainty")

    # Model results
    colors = plt.cm.tab10(np.linspace(0, 1, len(profiles)))
    for (name, u_profile), color in zip(profiles.items(), colors):
        ax.plot(u_profile, y, "-", color=color, linewidth=1.5, label=name)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path)
    return fig


def plot_cf_comparison(
    x: np.ndarray,
    cf_models: Dict[str, np.ndarray],
    cf_exp: Optional[np.ndarray] = None,
    x_exp: Optional[np.ndarray] = None,
    title: str = "Skin Friction Coefficient",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot Cf distribution for multiple models vs experiment.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    if cf_exp is not None and x_exp is not None:
        ax.plot(x_exp, cf_exp, "ko", markersize=3, label="Experiment", zorder=10)

    colors = plt.cm.tab10(np.linspace(0, 1, len(cf_models)))
    for (name, cf), color in zip(cf_models.items(), colors):
        ax.plot(x, cf, "-", color=color, linewidth=1.5, label=name)

    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_xlabel("x / ref. length")
    ax.set_ylabel("Cf")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path)
    return fig
