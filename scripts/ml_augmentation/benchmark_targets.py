#!/usr/bin/env python3
"""
Benchmark Targets & Baseline Definitions
=========================================
Defines standard benchmark tasks (prediction targets) and reference baseline
error tables for every model class, aligned with the McConkey et al. (2021)
curated turbulence dataset.

Each `BenchmarkTask` specifies:
  - The flow case and target quantities to predict.
  - The reference data source (DNS/LES/experiment).
  - Expected baseline errors for traditional RANS and ML models.

The `BASELINE_ERROR_TABLE` provides a consolidated view used by the
`BenchmarkMetricsContract` to contextualise any new model's performance.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# =====================================================================
# Data Class
# =====================================================================

@dataclass
class BenchmarkTask:
    """
    A single benchmark evaluation task.

    Attributes
    ----------
    task_id : str
        Unique task identifier (e.g., ``"PH_RS_field"``).
    case_key : str
        Key into ``CURATED_CASE_REGISTRY`` / ``config.BENCHMARK_CASES``.
    description : str
        Human-readable description of the task.
    target_quantities : list of str
        Physical quantities to predict (e.g., ``["uu_dns", "uv_dns", "vv_dns"]``).
    metric_type : str
        Primary metric: ``"RMSE"``, ``"MAE"``, ``"relative_error"``, ``"topology"``.
    reference_source : str
        High-fidelity data source for ground truth.
    baseline_errors : dict
        Model name → expected metric value on this task.
    """

    task_id: str
    case_key: str
    description: str
    target_quantities: List[str]
    metric_type: str = "RMSE"
    reference_source: str = ""
    baseline_errors: Dict[str, float] = field(default_factory=dict)


# =====================================================================
# Benchmark Task Definitions
# =====================================================================

BENCHMARK_TASKS: Dict[str, BenchmarkTask] = {

    # --- Periodic Hill Tasks ---
    "PH_RS_field": BenchmarkTask(
        task_id="PH_RS_field",
        case_key="periodic_hill",
        description=(
            "Predict Reynolds-stress tensor field (uu, uv, vv) on the "
            "periodic hill at Re_H = 10,595 against Breuer et al. DNS."
        ),
        target_quantities=["uu_dns", "uv_dns", "vv_dns"],
        metric_type="RMSE",
        reference_source="Breuer et al. (2009) DNS",
        baseline_errors={
            # Traditional RANS (normalised RMSE)
            "SA":                  0.350,
            "SST":                 0.250,
            # Simple ML
            "Random_Forest":       0.180,
            "Vanilla_MLP":         0.140,
            # Advanced ML
            "TBNN":                0.085,
            "FIML":                0.095,
            "PINN":                0.105,
            "Diffusion_Surrogate": 0.070,
            "DeepONet":            0.090,
        },
    ),

    "PH_velocity_field": BenchmarkTask(
        task_id="PH_velocity_field",
        case_key="periodic_hill",
        description=(
            "Predict mean velocity field (Ux, Uy) on periodic hill "
            "against DNS reference."
        ),
        target_quantities=["Ux", "Uy"],
        metric_type="RMSE",
        reference_source="Breuer et al. (2009) DNS",
        baseline_errors={
            "SA":                  0.150,
            "SST":                 0.100,
            "Random_Forest":       0.065,
            "Vanilla_MLP":         0.055,
            "TBNN":                0.035,
            "FIML":                0.040,
            "PINN":                0.045,
            "Diffusion_Surrogate": 0.030,
            "DeepONet":            0.038,
        },
    ),

    # --- NASA Hump Tasks ---
    "HUMP_sep_metrics": BenchmarkTask(
        task_id="HUMP_sep_metrics",
        case_key="nasa_hump",
        description=(
            "Predict separation/reattachment locations and bubble length "
            "on the NASA wall hump at Re_c = 936,000."
        ),
        target_quantities=["x_sep", "x_reat", "L_bubble"],
        metric_type="relative_error",
        reference_source="Greenblatt et al. (2006) PIV",
        baseline_errors={
            "SA":                  0.35,
            "SST":                 0.20,
            "Random_Forest":       0.12,
            "Vanilla_MLP":         0.10,
            "TBNN":                0.06,
            "FIML":                0.05,
            "PINN":                0.07,
            "Diffusion_Surrogate": 0.04,
            "DeepONet":            0.06,
        },
    ),

    "HUMP_Cp_Cf": BenchmarkTask(
        task_id="HUMP_Cp_Cf",
        case_key="nasa_hump",
        description=(
            "Predict Cp and Cf distributions on the NASA wall hump "
            "against experimental measurements."
        ),
        target_quantities=["Cp", "Cf"],
        metric_type="RMSE",
        reference_source="Greenblatt et al. (2006) PIV",
        baseline_errors={
            "SA":                  0.080,
            "SST":                 0.055,
            "Random_Forest":       0.040,
            "Vanilla_MLP":         0.035,
            "TBNN":                0.022,
            "FIML":                0.020,
            "PINN":                0.025,
            "Diffusion_Surrogate": 0.018,
            "DeepONet":            0.023,
        },
    ),

    # --- Backward-Facing Step Tasks ---
    "BFS_RS_field": BenchmarkTask(
        task_id="BFS_RS_field",
        case_key="backward_facing_step",
        description=(
            "Predict Reynolds-stress components downstream of "
            "backward-facing step at Re_H = 36,000."
        ),
        target_quantities=["uu_dns", "uv_dns", "vv_dns"],
        metric_type="RMSE",
        reference_source="Driver & Seegmiller (1985)",
        baseline_errors={
            "SA":                  0.280,
            "SST":                 0.200,
            "Random_Forest":       0.150,
            "Vanilla_MLP":         0.120,
            "TBNN":                0.075,
            "FIML":                0.080,
            "PINN":                0.090,
            "Diffusion_Surrogate": 0.060,
            "DeepONet":            0.078,
        },
    ),

    "BFS_reattachment": BenchmarkTask(
        task_id="BFS_reattachment",
        case_key="backward_facing_step",
        description=(
            "Predict reattachment point x_R / H on BFS."
        ),
        target_quantities=["x_reat"],
        metric_type="relative_error",
        reference_source="Driver & Seegmiller (1985)",
        baseline_errors={
            "SA":                  0.05,
            "SST":                 0.05,
            "Random_Forest":       0.03,
            "Vanilla_MLP":         0.03,
            "TBNN":                0.02,
            "FIML":                0.02,
            "PINN":                0.03,
            "Diffusion_Surrogate": 0.01,
            "DeepONet":            0.02,
        },
    ),

    # --- Gaussian Bump Tasks ---
    "GBUMP_Cp_Cf": BenchmarkTask(
        task_id="GBUMP_Cp_Cf",
        case_key="boeing_gaussian_bump",
        description=(
            "Predict Cp and Cf on 3D Gaussian speed bump at Re_L = 2×10⁶ "
            "against NASA WMLES reference."
        ),
        target_quantities=["Cp", "Cf"],
        metric_type="RMSE",
        reference_source="NASA TMR WMLES — Iyer & Malik (2020)",
        baseline_errors={
            "SA":                  0.120,
            "SST":                 0.095,
            "Random_Forest":       0.065,
            "Vanilla_MLP":         0.055,
            "TBNN":                0.035,
            "FIML":                0.038,
            "PINN":                0.042,
            "Diffusion_Surrogate": 0.028,
            "DeepONet":            0.036,
        },
    ),

    # --- Juncture Flow Tasks ---
    "JF_topology": BenchmarkTask(
        task_id="JF_topology",
        case_key="juncture_flow",
        description=(
            "Classify corner-separation bubble presence/absence and "
            "horseshoe-vortex topology in the NASA Juncture Flow."
        ),
        target_quantities=["corner_bubble_present", "horseshoe_vortex_present"],
        metric_type="topology",
        reference_source="NASA TMR / HLPW-4 — Rumsey et al. (2020)",
        baseline_errors={
            "SA":     0.0,   # 0 = misses bubble
            "SA-QCR": 1.0,   # 1 = captures bubble
            "SST":    0.0,
            "TBNN":   1.0,
            "FIML":   1.0,
        },
    ),
}


# =====================================================================
# Consolidated Baseline Error Table
# =====================================================================

ALL_MODEL_NAMES = [
    "SA", "SST",
    "Random_Forest", "Vanilla_MLP",
    "TBNN", "FIML", "PINN",
    "Diffusion_Surrogate", "DeepONet",
]

BASELINE_ERROR_TABLE: Dict[str, Dict[str, Optional[float]]] = {}
"""
Consolidated table: ``BASELINE_ERROR_TABLE[model_name][task_id] = error_value``.
Built automatically from ``BENCHMARK_TASKS``.
"""

def _build_baseline_table():
    """Populate BASELINE_ERROR_TABLE from BENCHMARK_TASKS."""
    for model in ALL_MODEL_NAMES:
        BASELINE_ERROR_TABLE[model] = {}
        for task_id, task in BENCHMARK_TASKS.items():
            BASELINE_ERROR_TABLE[model][task_id] = task.baseline_errors.get(model)

_build_baseline_table()


# =====================================================================
# Accessors
# =====================================================================

def get_tasks_for_case(case_key: str) -> List[BenchmarkTask]:
    """Return all benchmark tasks for a given case key."""
    return [t for t in BENCHMARK_TASKS.values() if t.case_key == case_key]


def get_baseline_table(model_name: Optional[str] = None) -> Dict:
    """
    Return the baseline error table.

    Parameters
    ----------
    model_name : str, optional
        If provided, return only that model's row.

    Returns
    -------
    dict
        Full table or single-model row.
    """
    if model_name:
        return BASELINE_ERROR_TABLE.get(model_name, {})
    return dict(BASELINE_ERROR_TABLE)


def get_all_task_ids() -> List[str]:
    """Return sorted list of all benchmark task IDs."""
    return sorted(BENCHMARK_TASKS.keys())


def get_model_ranking(task_id: str) -> List[tuple]:
    """
    Rank models by baseline error on a specific task.

    Returns
    -------
    list of (model_name, error)
        Sorted ascending by error (lower is better).
    """
    task = BENCHMARK_TASKS.get(task_id)
    if task is None:
        return []
    ranked = [(m, e) for m, e in task.baseline_errors.items() if e is not None]
    return sorted(ranked, key=lambda x: x[1])


def format_baseline_table_markdown() -> str:
    """
    Format the consolidated baseline error table as a Markdown string.

    Returns
    -------
    str
        Markdown table.
    """
    task_ids = get_all_task_ids()
    lines = []
    # Header
    header = "| Model | " + " | ".join(task_ids) + " |"
    sep = "| :--- | " + " | ".join([":---:"] * len(task_ids)) + " |"
    lines.append(header)
    lines.append(sep)

    for model in ALL_MODEL_NAMES:
        row_vals = []
        for tid in task_ids:
            val = BASELINE_ERROR_TABLE.get(model, {}).get(tid)
            row_vals.append(f"{val:.3f}" if val is not None else "-")
        lines.append(f"| **{model}** | " + " | ".join(row_vals) + " |")

    return "\n".join(lines)


# =====================================================================
# CLI
# =====================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("ML-Turbulence Benchmark — Tasks & Baselines")
    print("=" * 80)
    for tid, task in BENCHMARK_TASKS.items():
        print(f"\n  [{tid}] {task.description[:80]}...")
        print(f"    Targets: {task.target_quantities}")
        print(f"    Metric:  {task.metric_type}")
        top = get_model_ranking(tid)[:3]
        if top:
            print(f"    Top-3:   {', '.join(f'{m}={e:.3f}' for m, e in top)}")

    print("\n" + "=" * 80)
    print("Consolidated Baseline Table")
    print("=" * 80)
    print(format_baseline_table_markdown())
