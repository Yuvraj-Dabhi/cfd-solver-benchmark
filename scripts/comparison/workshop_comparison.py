"""
Workshop Comparison Framework
==============================
Compare CFD results against published scatter bands from AIAA workshops:
  - Drag Prediction Workshop (DPW-4/5/6/7)
  - High-Lift Prediction Workshop (HiLiftPW-1/2/3)

These workshops provide the gold standard for assessing CFD solver
accuracy in the context of multi-code, multi-grid, multi-modeler benchmarks.

References
----------
- Tinoco et al. (2018), J. Aircraft 55(4), DOI:10.2514/1.C034409
- Levy et al. (2014), J. Aircraft 51(4), DOI:10.2514/1.C032389
- Rumsey et al. (2011), J. Aircraft 48(6), DOI:10.2514/1.C031447
"""

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Resolve project root
_THIS_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = _THIS_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


# =============================================================================
# Scatter Band Data Structures
# =============================================================================
@dataclass
class ScatterBand:
    """
    Workshop participant scatter band for one metric.

    Encodes the min/max/mean/std from published workshop results,
    allowing users to see where their results fall within the
    community distribution.
    """
    metric: str
    mean: float
    std: float
    minimum: float
    maximum: float
    n_participants: int = 0
    source: str = ""

    @property
    def range(self) -> float:
        return self.maximum - self.minimum

    def z_score(self, value: float) -> float:
        """Compute z-score of a value relative to the scatter band."""
        if self.std == 0:
            return 0.0
        return (value - self.mean) / self.std

    def percentile(self, value: float) -> float:
        """Estimate percentile rank assuming normal distribution."""
        from scipy.stats import norm
        return float(norm.cdf(self.z_score(value)) * 100)

    def within_band(self, value: float) -> bool:
        """Check if value falls within the scatter band."""
        return self.minimum <= value <= self.maximum


# =============================================================================
# DPW Scatter Bands (Published Workshop Results)
# =============================================================================
"""
DPW scatter band data from published workshop summaries.
Values represent the range of results submitted by participants
using various solvers, grids, and turbulence models.
"""

DPW_SCATTER: Dict[str, Dict[str, ScatterBand]] = {
    # DPW-5: CRM Wing-Body at M=0.85, Re=5e6, CL=0.5
    "DPW5_WB": {
        "CL": ScatterBand(
            metric="CL", mean=0.500, std=0.010,
            minimum=0.470, maximum=0.530,
            n_participants=25,
            source="Levy et al. (2014), J. Aircraft 51(4)",
        ),
        "CD_counts": ScatterBand(
            metric="CD (counts)", mean=256.0, std=8.0,
            minimum=235.0, maximum=280.0,
            n_participants=25,
            source="Levy et al. (2014), J. Aircraft 51(4)",
        ),
        "CM": ScatterBand(
            metric="CM", mean=-0.095, std=0.012,
            minimum=-0.130, maximum=-0.060,
            n_participants=25,
            source="Levy et al. (2014), J. Aircraft 51(4)",
        ),
        "alpha_deg": ScatterBand(
            metric="alpha (deg)", mean=2.20, std=0.15,
            minimum=1.80, maximum=2.60,
            n_participants=25,
            source="Levy et al. (2014), J. Aircraft 51(4)",
        ),
    },
    # DPW-6: CRM Wing-Body-Tail at M=0.85
    "DPW6_WBT": {
        "CL": ScatterBand(
            metric="CL", mean=0.500, std=0.008,
            minimum=0.480, maximum=0.520,
            n_participants=18,
            source="Tinoco et al. (2018), J. Aircraft 55(4)",
        ),
        "CD_counts": ScatterBand(
            metric="CD (counts)", mean=260.0, std=6.0,
            minimum=245.0, maximum=275.0,
            n_participants=18,
            source="Tinoco et al. (2018), J. Aircraft 55(4)",
        ),
        "CD_pressure": ScatterBand(
            metric="CD_p (counts)", mean=130.0, std=4.0,
            minimum=120.0, maximum=140.0,
            n_participants=18,
            source="Tinoco et al. (2018), J. Aircraft 55(4)",
        ),
    },
}


# =============================================================================
# HiLiftPW Scatter Bands
# =============================================================================
HILIFTPW_SCATTER: Dict[str, Dict[str, ScatterBand]] = {
    # HiLiftPW-1: Trap Wing, Config 1 (slat=30°, flap=25°)
    "HLPW1_TrapWing": {
        "CL_max": ScatterBand(
            metric="CL_max", mean=2.70, std=0.15,
            minimum=2.35, maximum=3.15,
            n_participants=20,
            source="Rumsey et al. (2011), J. Aircraft 48(6)",
        ),
        "alpha_stall_deg": ScatterBand(
            metric="alpha_stall (deg)", mean=32.0, std=3.0,
            minimum=26.0, maximum=38.0,
            n_participants=20,
            source="Rumsey et al. (2011), J. Aircraft 48(6)",
        ),
        "CD_at_CLmax": ScatterBand(
            metric="CD at CLmax", mean=0.080, std=0.015,
            minimum=0.050, maximum=0.120,
            n_participants=20,
            source="Rumsey et al. (2011), J. Aircraft 48(6)",
        ),
    },
    # HiLiftPW-3: CRM-HL
    "HLPW3_CRM_HL": {
        "CL_max": ScatterBand(
            metric="CL_max", mean=2.40, std=0.12,
            minimum=2.10, maximum=2.75,
            n_participants=15,
            source="HiLiftPW-3 Summary (2017)",
        ),
    },
}


# =============================================================================
# Comparison Functions
# =============================================================================
def compare_to_workshop(
    user_results: Dict[str, float],
    workshop: str = "DPW5_WB",
) -> Dict[str, Dict[str, Any]]:
    """
    Compare user CFD results against workshop scatter bands.

    Parameters
    ----------
    user_results : dict
        {metric_name: value} for metrics to compare.
    workshop : str
        Workshop identifier (e.g., "DPW5_WB", "HLPW1_TrapWing").

    Returns
    -------
    dict
        Comparison results for each metric, including z-score,
        percentile, and whether within the scatter band.
    """
    # Select scatter band source
    if workshop in DPW_SCATTER:
        bands = DPW_SCATTER[workshop]
    elif workshop in HILIFTPW_SCATTER:
        bands = HILIFTPW_SCATTER[workshop]
    else:
        raise ValueError(
            f"Unknown workshop: {workshop}. "
            f"Available DPW: {list(DPW_SCATTER.keys())}, "
            f"HiLiftPW: {list(HILIFTPW_SCATTER.keys())}"
        )

    results = {}
    for metric, value in user_results.items():
        if metric not in bands:
            logger.warning(f"Metric '{metric}' not available in {workshop}")
            continue

        band = bands[metric]
        z = band.z_score(value)
        results[metric] = {
            "value": value,
            "mean": band.mean,
            "std": band.std,
            "z_score": z,
            "within_band": band.within_band(value),
            "min": band.minimum,
            "max": band.maximum,
            "n_participants": band.n_participants,
            "source": band.source,
            "rating": _z_to_rating(z),
        }

    return results


def _z_to_rating(z: float) -> str:
    """Convert z-score to qualitative rating."""
    az = abs(z)
    if az <= 0.5:
        return "EXCELLENT (within ±0.5σ)"
    elif az <= 1.0:
        return "GOOD (within ±1σ)"
    elif az <= 2.0:
        return "ACCEPTABLE (within ±2σ)"
    else:
        return "OUTLIER (beyond ±2σ)"


def compute_workshop_ranking(
    user_results: Dict[str, float],
    workshop: str = "DPW5_WB",
) -> Dict[str, Any]:
    """
    Rank user results within workshop participant distribution.

    Parameters
    ----------
    user_results : dict
        {metric_name: value}
    workshop : str
        Workshop identifier.

    Returns
    -------
    dict with overall ranking score and per-metric rankings.
    """
    comparison = compare_to_workshop(user_results, workshop)

    if not comparison:
        return {"overall_score": 0.0, "metrics": {}, "rank_label": "NO DATA"}

    # Composite score: average of (1 - |z|/3), capped at [0, 1]
    scores = []
    for metric, result in comparison.items():
        score = max(0, 1 - abs(result["z_score"]) / 3.0)
        comparison[metric]["score"] = score
        scores.append(score)

    overall = float(np.mean(scores)) if scores else 0.0

    if overall >= 0.8:
        rank_label = "TOP-TIER"
    elif overall >= 0.6:
        rank_label = "COMPETITIVE"
    elif overall >= 0.4:
        rank_label = "AVERAGE"
    else:
        rank_label = "BELOW AVERAGE"

    return {
        "overall_score": overall,
        "rank_label": rank_label,
        "metrics": comparison,
        "workshop": workshop,
    }


# =============================================================================
# Visualization
# =============================================================================
def plot_workshop_comparison(
    user_results: Dict[str, float],
    workshop: str = "DPW5_WB",
    save_path: Optional[str] = None,
) -> Optional[Any]:
    """
    Create scatter-band comparison plot.

    Overlays user results on workshop scatter bands with
    color-coded z-score regions.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("Matplotlib not available for plotting")
        return None

    comparison = compare_to_workshop(user_results, workshop)
    if not comparison:
        return None

    metrics = list(comparison.keys())
    n = len(metrics)

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 5), squeeze=False)
    axes = axes[0]

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        result = comparison[metric]

        band_min = result["min"]
        band_max = result["max"]
        mean = result["mean"]
        std = result["std"]
        value = result["value"]

        # Draw scatter band regions
        ax.axhspan(mean - 2*std, mean + 2*std, alpha=0.1, color="blue", label="±2σ")
        ax.axhspan(mean - std, mean + std, alpha=0.2, color="blue", label="±1σ")
        ax.axhline(mean, color="blue", linestyle="--", linewidth=1, label="Mean")
        ax.axhline(band_min, color="gray", linestyle=":", linewidth=0.5)
        ax.axhline(band_max, color="gray", linestyle=":", linewidth=0.5)

        # User result
        color = "green" if result["within_band"] else "red"
        ax.plot(0, value, "D", color=color, markersize=12, zorder=5)
        ax.annotate(f"{value:.4f}", (0, value), textcoords="offset points",
                    xytext=(15, 0), fontsize=9, color=color, fontweight="bold")

        ax.set_title(f"{metric}\nz={result['z_score']:.2f} ({result['rating'].split('(')[0].strip()})",
                     fontsize=10)
        ax.set_xlim(-1, 1)
        ax.set_ylabel(metric)
        ax.tick_params(bottom=False, labelbottom=False)

    fig.suptitle(f"Results vs {workshop} Workshop Scatter", fontsize=13,
                 fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved workshop comparison plot: {save_path}")

    return fig


# =============================================================================
# Report Generation
# =============================================================================
def generate_workshop_report(
    user_results: Dict[str, float],
    workshops: Optional[List[str]] = None,
) -> str:
    """
    Generate a markdown report comparing results to workshops.

    Parameters
    ----------
    user_results : dict
        {metric: value}
    workshops : list, optional
        Workshop identifiers. Default: all available.

    Returns
    -------
    str
        Markdown-formatted report.
    """
    if workshops is None:
        workshops = list(DPW_SCATTER.keys()) + list(HILIFTPW_SCATTER.keys())

    lines = ["# Workshop Comparison Report\n"]

    for ws in workshops:
        try:
            ranking = compute_workshop_ranking(user_results, ws)
        except ValueError:
            continue

        if not ranking["metrics"]:
            continue

        lines.append(f"## {ws}\n")
        lines.append(
            f"**Overall Score**: {ranking['overall_score']:.2f} "
            f"({ranking['rank_label']})\n"
        )

        lines.append("| Metric | Your Value | Workshop Mean±σ | z-score | Rating |")
        lines.append("|--------|-----------|-----------------|---------|--------|")

        for metric, data in ranking["metrics"].items():
            lines.append(
                f"| {metric} "
                f"| {data['value']:.4f} "
                f"| {data['mean']:.4f}±{data['std']:.4f} "
                f"| {data['z_score']:+.2f} "
                f"| {data['rating'].split('(')[0].strip()} |"
            )
        lines.append("")

    return "\n".join(lines)


# =============================================================================
# CLI
# =============================================================================
if __name__ == "__main__":
    # Example usage
    print("Available Workshop Scatter Bands:")
    print(f"  DPW:      {list(DPW_SCATTER.keys())}")
    print(f"  HiLiftPW: {list(HILIFTPW_SCATTER.keys())}")
    print()

    # Example comparison
    example_results = {"CL": 0.505, "CD_counts": 258.0}
    ranking = compute_workshop_ranking(example_results, "DPW5_WB")
    print(f"Example DPW5 ranking: {ranking['rank_label']} "
          f"(score={ranking['overall_score']:.2f})")
    for metric, data in ranking["metrics"].items():
        print(f"  {metric}: z={data['z_score']:+.2f}, {data['rating']}")
