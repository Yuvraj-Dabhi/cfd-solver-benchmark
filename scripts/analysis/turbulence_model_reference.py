"""
Turbulence Model Reference Library
====================================
Centralized reference data for turbulence model performance on benchmark
cases, drawn from published literature. Enables:
  - Comparison of user CFD results against published accuracy ranges
  - Evidence-based model selection recommendations
  - Literature-backed error baselines for the NASA 40% Challenge

Sources
-------
- Menter (1994), AIAA J. 32(8), DOI:10.2514/3.12149
- Spalart & Allmaras (1992), AIAA Paper 92-0439
- Rumsey (2018), AIAA-2018-3319 (Juncture Flow)
- NASA TMR: https://turbmodels.larc.nasa.gov/
- DPW-6: Tinoco et al. (2018), J. Aircraft 55(4)
- Breuer et al. (2009), Computers & Fluids 38(2)
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
from config import SeparationCategory

logger = logging.getLogger(__name__)


# =============================================================================
# Reference Performance Data
# =============================================================================
@dataclass
class ReferenceMetric:
    """Published performance metric for a model on a case."""
    metric_name: str
    value: float
    unit: str = ""
    error_pct: float = 0.0   # Error relative to experiment
    source: str = ""          # Publication DOI or citation
    notes: str = ""


# Format: REFERENCE_PERFORMANCE[model][case] = list of ReferenceMetric
REFERENCE_PERFORMANCE: Dict[str, Dict[str, List[ReferenceMetric]]] = {
    # =========================================================================
    # Spalart-Allmaras (SA)
    # =========================================================================
    "SA": {
        "backward_facing_step": [
            ReferenceMetric(
                metric_name="x_reat_xH",
                value=6.10,
                error_pct=2.6,
                source="NASA TMR 2DBFS",
                notes="Slightly underpredicts reattachment (exp=6.26)",
            ),
        ],
        "nasa_hump": [
            ReferenceMetric(
                metric_name="bubble_length_xc",
                value=0.60,
                error_pct=35,
                source="NASA TMR 2DWMH",
                notes="Overpredicts bubble length (~35% too long)",
            ),
            ReferenceMetric(
                metric_name="x_reat_xc",
                value=1.27,
                error_pct=14,
                source="CFDVAL2004 Workshop",
                notes="Late reattachment (exp=1.11)",
            ),
        ],
        "periodic_hill": [
            ReferenceMetric(
                metric_name="x_reat_xh",
                value=5.80,
                error_pct=23,
                source="Breuer et al. (2009), Comp. Fluids 38(2)",
                notes="Massive overprediction (DNS=4.72)",
            ),
        ],
        "bachalo_johnson": [
            ReferenceMetric(
                metric_name="bubble_length_xc",
                value=0.28,
                error_pct=27,
                source="NASA TMR ATB",
                notes="Overpredicts separation bubble (~27%)",
            ),
        ],
        "juncture_flow": [
            ReferenceMetric(
                metric_name="corner_bubble",
                value=0,
                source="Rumsey et al. (2018), AIAA-2018-3319",
                notes="Completely misses corner separation bubble",
            ),
        ],
    },

    # =========================================================================
    # SA with QCR
    # =========================================================================
    "SA-QCR": {
        "juncture_flow": [
            ReferenceMetric(
                metric_name="corner_bubble",
                value=1,
                source="Rumsey et al. (2018), AIAA-2018-3319",
                notes="Captures corner separation bubble",
            ),
            ReferenceMetric(
                metric_name="bubble_size_accuracy",
                value=75,
                unit="%",
                source="Rumsey et al. (2020), AIAA J. 58",
                notes="Captures ~75% of measured corner flow",
            ),
        ],
    },

    # =========================================================================
    # Menter SST (k-ω SST)
    # =========================================================================
    "SST": {
        "backward_facing_step": [
            ReferenceMetric(
                metric_name="x_reat_xH",
                value=6.20,
                error_pct=1.0,
                source="NASA TMR 2DBFS",
                notes="Excellent BFS prediction (exp=6.26)",
            ),
        ],
        "nasa_hump": [
            ReferenceMetric(
                metric_name="bubble_length_xc",
                value=0.53,
                error_pct=20,
                source="NASA TMR 2DWMH / Menter (1994)",
                notes="Best RANS for hump (~20% overpredict)",
            ),
            ReferenceMetric(
                metric_name="Cp_MAPE",
                value=8.3,
                unit="%",
                source="NASA TMR 2DWMH",
                notes="Mean Absolute Percentage Error in Cp",
            ),
        ],
        "periodic_hill": [
            ReferenceMetric(
                metric_name="x_reat_xh",
                value=5.20,
                error_pct=10,
                source="Breuer et al. (2009), Comp. Fluids 38(2)",
                notes="Moderate overprediction (DNS=4.72)",
            ),
        ],
        "bachalo_johnson": [
            ReferenceMetric(
                metric_name="bubble_length_xc",
                value=0.25,
                error_pct=14,
                source="NASA TMR ATB",
                notes="Better than SA for transonic separation",
            ),
        ],
    },

    # =========================================================================
    # k-ε (Standard / Realizable)
    # =========================================================================
    "k-epsilon": {
        "backward_facing_step": [
            ReferenceMetric(
                metric_name="x_reat_xH",
                value=5.50,
                error_pct=12,
                source="Launder & Sharma (1974)",
                notes="Underpredicts reattachment significantly",
            ),
        ],
        "nasa_hump": [
            ReferenceMetric(
                metric_name="bubble_length_xc",
                value=0.70,
                error_pct=57,
                source="NASA TMR 2DWMH",
                notes="Very poor: massive bubble overpredict",
            ),
        ],
    },

    # =========================================================================
    # RSM (Reynolds Stress Model)
    # =========================================================================
    "RSM": {
        "periodic_hill": [
            ReferenceMetric(
                metric_name="x_reat_xh",
                value=4.90,
                error_pct=4,
                source="Jakirlic & Maduta (2015)",
                notes="Best RANS for periodic hills",
            ),
        ],
        "juncture_flow": [
            ReferenceMetric(
                metric_name="horseshoe_vortex",
                value=1,
                source="Rumsey et al. (2018)",
                notes="Best for horseshoe vortex structure",
            ),
        ],
    },

    # =========================================================================
    # DDES (Delayed Detached-Eddy Simulation)
    # =========================================================================
    "DDES": {
        "nasa_hump": [
            ReferenceMetric(
                metric_name="bubble_length_xc",
                value=0.46,
                error_pct=3,
                source="Shur et al. (2008)",
                notes="Near-DNS accuracy for hump when well-resolved",
            ),
        ],
        "periodic_hill": [
            ReferenceMetric(
                metric_name="x_reat_xh",
                value=4.80,
                error_pct=2,
                source="Fröhlich et al. (2005)",
                notes="Excellent agreement with DNS (4.72)",
            ),
        ],
    },
}


# =============================================================================
# Model Recommendation Engine
# =============================================================================

# Recommended models by separation category (evidence-based from literature)
MODEL_RECOMMENDATIONS: Dict[str, Dict[str, Any]] = {
    "smooth_body_2d": {
        "primary": "SST",
        "alternative": "SA",
        "hybrid": "DDES",
        "rationale": "SST provides best RANS accuracy for smooth-body separation "
                     "(MAPE ~8.3% on hump). DDES further reduces to ~3%.",
        "source": "NASA TMR 2DWMH, Menter (1994)",
    },
    "geometric": {
        "primary": "SST",
        "alternative": "SA",
        "hybrid": "DDES",
        "rationale": "Both SA and SST perform well for geometry-forced separation "
                     "(BFS x_R error <5%). SST slightly better.",
        "source": "NASA TMR 2DBFS",
    },
    "curvature": {
        "primary": "RSM",
        "alternative": "SST",
        "hybrid": "DDES",
        "rationale": "RSM captures curvature effects best (periodic hill x_R error ~4%). "
                     "SST acceptable (~10% error).",
        "source": "Breuer et al. (2009)",
    },
    "shock_induced": {
        "primary": "SST",
        "alternative": "SA",
        "hybrid": "DDES",
        "rationale": "SST better than SA for shock-induced separation "
                     "(bubble error 14% vs 27%). Both struggle with shock position.",
        "source": "NASA TMR ATB, Bachalo & Johnson (1986)",
    },
    "corner_3d": {
        "primary": "SA-QCR",
        "alternative": "RSM",
        "hybrid": "DDES",
        "rationale": "SA-QCR captures corner separation bubbles that SA misses entirely. "
                     "RSM best for horseshoe vortex. DPW mandates SA-QCR.",
        "source": "Rumsey et al. (2018), DPW-6",
    },
    "trailing_edge": {
        "primary": "SST",
        "alternative": "SA",
        "hybrid": "DDES",
        "rationale": "SST handles adverse PG trailing-edge separation better. "
                     "All RANS struggle near stall.",
        "source": "NASA TMR 2DN44, Coles & Wadcock (1979)",
    },
}


def get_model_recommendation(
    separation_type: str,
    constraints: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Get evidence-based turbulence model recommendation.

    Parameters
    ----------
    separation_type : str
        Type of separation. Must match a key in MODEL_RECOMMENDATIONS
        or a SeparationCategory enum name (case-insensitive).
    constraints : dict, optional
        Optional constraints like {"cost": "low", "3D": True}.

    Returns
    -------
    dict with recommended model(s) and supporting evidence.
    """
    # Normalize separation type
    sep_key = separation_type.lower().replace(" ", "_")

    # Try direct match first
    if sep_key not in MODEL_RECOMMENDATIONS:
        # Try mapping from SeparationCategory
        category_map = {
            "smooth_body_2d": "smooth_body_2d",
            "geometric": "geometric",
            "curvature": "curvature",
            "shock_induced": "shock_induced",
            "corner_3d": "corner_3d",
            "trailing_edge": "trailing_edge",
            "verification": "geometric",  # Fallback
        }
        sep_key = category_map.get(sep_key, "smooth_body_2d")

    rec = MODEL_RECOMMENDATIONS[sep_key]

    result = {
        "separation_type": sep_key,
        "primary_model": rec["primary"],
        "alternative_model": rec["alternative"],
        "hybrid_model": rec["hybrid"],
        "rationale": rec["rationale"],
        "source": rec["source"],
    }

    # Apply constraints
    if constraints:
        if constraints.get("cost") == "low":
            result["note"] = (
                f"For low cost, prefer {rec['primary']} (RANS) "
                f"over {rec['hybrid']} (hybrid)."
            )
        if constraints.get("3D") and sep_key != "corner_3d":
            result["note"] = (
                "For 3D flows, consider SA-QCR or RSM in addition to "
                f"the recommended {rec['primary']}."
            )

    return result


# =============================================================================
# Literature Comparison
# =============================================================================
def compare_to_literature(
    model: str,
    case: str,
    user_metrics: Dict[str, float],
) -> Dict[str, Any]:
    """
    Compare user CFD results against published literature values.

    Parameters
    ----------
    model : str
        Turbulence model name (e.g., "SA", "SST").
    case : str
        Benchmark case name (e.g., "nasa_hump").
    user_metrics : dict
        {metric_name: user_value}

    Returns
    -------
    dict with comparison results for each metric.
    """
    if model not in REFERENCE_PERFORMANCE:
        return {"error": f"No reference data for model '{model}'"}
    if case not in REFERENCE_PERFORMANCE[model]:
        return {"error": f"No reference data for model '{model}' on case '{case}'"}

    refs = REFERENCE_PERFORMANCE[model][case]
    comparisons = {}

    for ref in refs:
        if ref.metric_name in user_metrics:
            user_val = user_metrics[ref.metric_name]
            if ref.value != 0:
                error_vs_ref = abs(user_val - ref.value) / abs(ref.value) * 100
            else:
                error_vs_ref = abs(user_val - ref.value) * 100

            comparisons[ref.metric_name] = {
                "user_value": user_val,
                "reference_value": ref.value,
                "reference_error_pct": ref.error_pct,
                "user_vs_reference_pct": error_vs_ref,
                "consistent": error_vs_ref < max(ref.error_pct * 1.5, 10),
                "source": ref.source,
                "notes": ref.notes,
            }
        else:
            comparisons[ref.metric_name] = {
                "reference_value": ref.value,
                "reference_error_pct": ref.error_pct,
                "source": ref.source,
                "notes": ref.notes,
                "user_value": None,
            }

    return comparisons


def get_literature_baseline(model: str, case: str) -> Dict[str, float]:
    """
    Get published performance values for a model/case combination.

    Returns
    -------
    dict
        {metric_name: published_value}
    """
    if model not in REFERENCE_PERFORMANCE or case not in REFERENCE_PERFORMANCE[model]:
        return {}
    return {
        ref.metric_name: ref.value
        for ref in REFERENCE_PERFORMANCE[model][case]
    }


def list_available_references() -> Dict[str, List[str]]:
    """List all models and their available case references."""
    return {
        model: list(cases.keys())
        for model, cases in REFERENCE_PERFORMANCE.items()
    }


# =============================================================================
# CLI
# =============================================================================
if __name__ == "__main__":
    print("=== Turbulence Model Reference Library ===\n")

    print("Available references:")
    for model, cases in list_available_references().items():
        print(f"  {model}: {', '.join(cases)}")

    print("\n--- Model Recommendations by Separation Type ---")
    for sep_type in MODEL_RECOMMENDATIONS:
        rec = get_model_recommendation(sep_type)
        print(f"\n  {sep_type}:")
        print(f"    Primary: {rec['primary_model']}")
        print(f"    Hybrid:  {rec['hybrid_model']}")
        print(f"    Source:  {rec['source']}")

    print("\n--- Example: SST on NASA Hump ---")
    baseline = get_literature_baseline("SST", "nasa_hump")
    print(f"  Published baselines: {baseline}")
