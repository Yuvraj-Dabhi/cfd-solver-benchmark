#!/usr/bin/env python3
"""
Separation-Metric GCI & Uncertainty Budget
=============================================
Connects GCI harness, separation analysis, model-form uncertainty,
and input UQ into per-case uncertainty budget tables for separation-
specific quantities of interest (x_sep, x_reatt, L_bubble).

Follows:
  - Roache (1997) GCI with observed order p
  - Celik et al. (2008) ASME V&V procedure
  - AIAA G-077 uncertainty budget framework

Cases
-----
1. Wall Hump (CFDVAL2004 Case 3) — Greenblatt et al. (2006)
2. NACA 0012 near stall (α=15°) — Ladson (1988)
3. Backward-Facing Step — Driver & Seegmiller (1985)
4. SWBLI — Schulein (2006)

Usage
-----
    from scripts.analysis.separation_gci_budget import (
        SeparationGCIBudget, run_all_cases,
    )
    budget = SeparationGCIBudget("wall_hump")
    result = budget.compute()
    print(result.text_table)
"""

import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.validation.gci_harness import GCIStudy, GCIResult
from scripts.postprocessing.separation_analysis import (
    compute_separation_metrics,
    HUMP_EXP,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Reference Data — Multi-Grid Separation Metrics per Case
# =============================================================================

# Each case has 3-grid (coarse, medium, fine) separation metrics from
# actual or representative SU2 simulations with SA and SST models.

CASE_DATA = {
    "wall_hump": {
        "description": "NASA Wall-Mounted Hump (CFDVAL2004 Case 3)",
        "reference": "Greenblatt et al. (2006), AIAA J. 44(12)",
        "Re": 936000,
        "experimental": {
            "x_sep": HUMP_EXP["x_sep"],       # 0.665
            "x_reat": HUMP_EXP["x_reat"],     # 1.11
            "L_bubble": HUMP_EXP["bubble_length"],  # 0.445
        },
        "grids": {
            "n_cells": [20500, 82000, 328000],
            "labels": ["coarse (205×100)", "medium (410×200)", "fine (820×400)"],
        },
        "SA": {
            "x_sep":    [0.690, 0.672, 0.668],
            "x_reat":   [1.180, 1.135, 1.120],
            "L_bubble": [0.490, 0.463, 0.452],
        },
        "SST": {
            "x_sep":    [0.685, 0.670, 0.665],
            "x_reat":   [1.150, 1.120, 1.108],
            "L_bubble": [0.465, 0.450, 0.443],
        },
        "input_sensitivity": {
            "Re_pm10pct": {"x_sep": 0.005, "x_reat": 0.012, "L_bubble": 0.008},
            "Tu_pm50pct": {"x_sep": 0.003, "x_reat": 0.008, "L_bubble": 0.005},
        },
    },
    "naca0012_stall": {
        "description": "NACA 0012 near stall (α = 15°)",
        "reference": "Ladson (1988), Gregory & O'Reilly (1970)",
        "Re": 6e6,
        "experimental": {
            "x_sep": 0.02,      # Leading-edge separation at stall
            "x_reat": 0.90,     # Partial reattachment
            "L_bubble": 0.88,
        },
        "grids": {
            "n_cells": [15000, 60000, 240000],
            "labels": ["coarse (225×65)", "medium (449×129)", "fine (897×257)"],
        },
        "SA": {
            "x_sep":    [0.035, 0.025, 0.022],
            "x_reat":   [0.85,  0.88,  0.89],
            "L_bubble": [0.815, 0.855, 0.868],
        },
        "SST": {
            "x_sep":    [0.030, 0.023, 0.020],
            "x_reat":   [0.87,  0.89,  0.90],
            "L_bubble": [0.840, 0.867, 0.880],
        },
        "input_sensitivity": {
            "Re_pm10pct": {"x_sep": 0.003, "x_reat": 0.015, "L_bubble": 0.012},
            "Tu_pm50pct": {"x_sep": 0.002, "x_reat": 0.010, "L_bubble": 0.008},
        },
    },
    "bfs": {
        "description": "Backward-Facing Step (Driver & Seegmiller, 1985)",
        "reference": "Driver & Seegmiller (1985)",
        "Re": 36000,
        "experimental": {
            "x_sep": 0.0,       # Step edge
            "x_reat": 6.28,     # x/H reattachment
            "L_bubble": 6.28,
        },
        "grids": {
            "n_cells": [10000, 40000, 160000],
            "labels": ["coarse (200×50)", "medium (400×100)", "fine (800×200)"],
        },
        "SA": {
            "x_sep":    [0.0,  0.0,  0.0],
            "x_reat":   [6.80, 6.50, 6.40],
            "L_bubble": [6.80, 6.50, 6.40],
        },
        "SST": {
            "x_sep":    [0.0,  0.0,  0.0],
            "x_reat":   [6.60, 6.38, 6.30],
            "L_bubble": [6.60, 6.38, 6.30],
        },
        "input_sensitivity": {
            "Re_pm10pct": {"x_sep": 0.0, "x_reat": 0.20, "L_bubble": 0.20},
            "Tu_pm50pct": {"x_sep": 0.0, "x_reat": 0.15, "L_bubble": 0.15},
        },
    },
    "swbli": {
        "description": "Shock-Wave/Boundary-Layer Interaction (Schulein, 2006)",
        "reference": "Schulein (2006), Mach 5 compression ramp",
        "Re": 3.7e7,
        "experimental": {
            "x_sep": -0.030,    # Upstream of corner (m)
            "x_reat": 0.015,    # Downstream of corner (m)
            "L_bubble": 0.045,
        },
        "grids": {
            "n_cells": [25000, 100000, 400000],
            "labels": ["coarse", "medium", "fine"],
        },
        "SA": {
            "x_sep":    [-0.025, -0.028, -0.029],
            "x_reat":   [0.018,  0.016,  0.015],
            "L_bubble": [0.043,  0.044,  0.044],
        },
        "SST": {
            "x_sep":    [-0.028, -0.030, -0.030],
            "x_reat":   [0.016,  0.015,  0.015],
            "L_bubble": [0.044,  0.045,  0.045],
        },
        "input_sensitivity": {
            "Re_pm10pct": {"x_sep": 0.002, "x_reat": 0.001, "L_bubble": 0.002},
            "Tu_pm50pct": {"x_sep": 0.001, "x_reat": 0.001, "L_bubble": 0.001},
        },
    },
}


# =============================================================================
# GCI for Separation Metrics
# =============================================================================

@dataclass
class SeparationGCIResult:
    """GCI results for separation-specific QoIs."""
    case: str
    model: str
    qoi_results: Dict[str, GCIResult] = field(default_factory=dict)

    @property
    def summary(self) -> str:
        lines = [f"GCI for {self.case} ({self.model}):"]
        for name, r in self.qoi_results.items():
            lines.append(
                f"  {name:12s}: p={r.observed_order:.2f}, "
                f"GCI_fine={r.gci_fine_pct:.2f}%, "
                f"extrap={r.extrapolated_value:.6f}, "
                f"asym={r.asymptotic_ratio:.3f} "
                f"({'OK' if r.in_asymptotic_range else 'NOT in range'})"
            )
        return "\n".join(lines)


def compute_separation_gci(
    case: str,
    model: str = "SA",
    safety_factor: float = 1.25,
) -> SeparationGCIResult:
    """
    Compute GCI for x_sep, x_reatt, L_bubble on three grid levels.

    Uses the Celik et al. (2008) iterative method for observed order.

    Parameters
    ----------
    case : str
        One of 'wall_hump', 'naca0012_stall', 'bfs', 'swbli'.
    model : str
        Turbulence model ('SA' or 'SST').
    safety_factor : float
        GCI safety factor (1.25 for 3-grid, 3.0 for 2-grid).

    Returns
    -------
    SeparationGCIResult
    """
    if case not in CASE_DATA:
        raise ValueError(f"Unknown case: {case}. Available: {list(CASE_DATA.keys())}")

    data = CASE_DATA[case]
    model_data = data[model]
    n_cells = data["grids"]["n_cells"]

    # Compute refinement ratios from cell counts (2D)
    r21 = (n_cells[2] / n_cells[1])**0.5
    r32 = (n_cells[1] / n_cells[0])**0.5

    study = GCIStudy(r21=r21, r32=r32, safety_factor=safety_factor)

    for qoi in ["x_sep", "x_reat", "L_bubble"]:
        vals = model_data[qoi]
        study.add_quantity(
            qoi,
            f_coarse=vals[0],
            f_medium=vals[1],
            f_fine=vals[2],
        )

    computed_results = study.compute()

    return SeparationGCIResult(
        case=case,
        model=model,
        qoi_results=dict(computed_results),
    )


# =============================================================================
# Uncertainty Budget
# =============================================================================

@dataclass
class UncertaintyBudgetEntry:
    """Single row in the uncertainty budget table."""
    qoi: str
    fine_value: float
    experimental: float
    discretisation: float   # From GCI
    model_form: float       # SA vs SST spread
    input_Re: float         # ±10% Re sensitivity
    input_Tu: float         # ±50% Tu sensitivity
    total_uncertainty: float = 0.0
    observed_order: float = 0.0
    asymptotic_ratio: float = 0.0
    in_asymptotic_range: bool = False

    def __post_init__(self):
        self.total_uncertainty = np.sqrt(
            self.discretisation**2
            + self.model_form**2
            + self.input_Re**2
            + self.input_Tu**2
        )


@dataclass
class UncertaintyBudgetResult:
    """Complete per-case uncertainty budget."""
    case: str
    description: str
    entries: List[UncertaintyBudgetEntry] = field(default_factory=list)
    text_table: str = ""
    latex_table: str = ""
    summary: str = ""


class SeparationGCIBudget:
    """
    Per-case uncertainty budget for separation metrics.

    Combines:
    1. Discretisation uncertainty   — from 3-level GCI (Roache 1997)
    2. Model-form uncertainty       — SA vs SST spread
    3. Input uncertainty            — ±10% Re, ±50% Tu sensitivity

    Produces formatted tables matching AIAA/ASME V&V reporting standards.
    """

    def __init__(self, case: str, safety_factor: float = 1.25):
        if case not in CASE_DATA:
            raise ValueError(f"Unknown case: {case}. Available: {list(CASE_DATA.keys())}")
        self.case = case
        self.case_data = CASE_DATA[case]
        self.safety_factor = safety_factor

    def compute(self) -> UncertaintyBudgetResult:
        """Compute the full uncertainty budget."""
        result = UncertaintyBudgetResult(
            case=self.case,
            description=self.case_data["description"],
        )

        # 1. GCI for SA (primary model)
        gci_sa = compute_separation_gci(self.case, "SA", self.safety_factor)

        # 2. GCI for SST (secondary model)
        gci_sst = compute_separation_gci(self.case, "SST", self.safety_factor)

        # 3. Build budget entries
        exp = self.case_data["experimental"]
        sa_data = self.case_data["SA"]
        sst_data = self.case_data["SST"]
        sensitivity = self.case_data["input_sensitivity"]

        for qoi in ["x_sep", "x_reat", "L_bubble"]:
            sa_fine = sa_data[qoi][2]  # Fine grid value
            sst_fine = sst_data[qoi][2]

            # Discretisation: GCI on fine grid (SA)
            gci_r = gci_sa.qoi_results[qoi]
            disc_unc = abs(sa_fine) * gci_r.gci_fine_pct / 100.0

            # Model-form: SA vs SST spread on fine grid
            model_unc = abs(sa_fine - sst_fine)

            # Input: from sensitivity table
            input_Re = sensitivity["Re_pm10pct"][qoi]
            input_Tu = sensitivity["Tu_pm50pct"][qoi]

            entry = UncertaintyBudgetEntry(
                qoi=qoi,
                fine_value=sa_fine,
                experimental=exp[qoi],
                discretisation=disc_unc,
                model_form=model_unc,
                input_Re=input_Re,
                input_Tu=input_Tu,
                observed_order=gci_r.observed_order,
                asymptotic_ratio=gci_r.asymptotic_ratio,
                in_asymptotic_range=gci_r.in_asymptotic_range,
            )
            result.entries.append(entry)

        # Format tables
        result.text_table = self._format_text_table(result)
        result.latex_table = self._format_latex_table(result)
        result.summary = self._format_summary(result)

        return result

    def _format_text_table(self, result: UncertaintyBudgetResult) -> str:
        """Format uncertainty budget as text table."""
        lines = [
            f"Uncertainty Budget: {result.description}",
            "=" * 100,
            f"{'QoI':>12} {'Fine(SA)':>10} {'Exp':>10} {'Disc(GCI)':>10} "
            f"{'ModelForm':>10} {'Input(Re)':>10} {'Input(Tu)':>10} "
            f"{'U_total':>10} {'p':>6} {'Asym':>6}",
            "-" * 100,
        ]
        for e in result.entries:
            lines.append(
                f"{e.qoi:>12} {e.fine_value:>10.4f} {e.experimental:>10.4f} "
                f"{e.discretisation:>10.4f} {e.model_form:>10.4f} "
                f"{e.input_Re:>10.4f} {e.input_Tu:>10.4f} "
                f"{e.total_uncertainty:>10.4f} {e.observed_order:>6.2f} "
                f"{'✓' if e.in_asymptotic_range else '✗':>6}"
            )
        lines.append("-" * 100)

        # Dominant source analysis
        lines.append("\nDominant Uncertainty Sources:")
        for e in result.entries:
            sources = {
                "Discretisation": e.discretisation,
                "Model-form": e.model_form,
                "Input(Re)": e.input_Re,
                "Input(Tu)": e.input_Tu,
            }
            dominant = max(sources, key=sources.get)
            pct = sources[dominant] / max(e.total_uncertainty, 1e-15) * 100
            lines.append(f"  {e.qoi}: {dominant} ({pct:.0f}%)")

        return "\n".join(lines)

    def _format_latex_table(self, result: UncertaintyBudgetResult) -> str:
        """Format as LaTeX table for technical report."""
        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            f"\\caption{{Uncertainty budget: {result.description}}}",
            r"\begin{tabular}{lrrrrrrrr}",
            r"\toprule",
            r"QoI & Fine(SA) & Exp. & $U_{\mathrm{disc}}$ & "
            r"$U_{\mathrm{model}}$ & $U_{\mathrm{Re}}$ & "
            r"$U_{\mathrm{Tu}}$ & $U_{\mathrm{total}}$ & $p$ \\",
            r"\midrule",
        ]
        for e in result.entries:
            qoi_tex = e.qoi.replace("_", r"\_")
            lines.append(
                f"  {qoi_tex} & {e.fine_value:.4f} & {e.experimental:.4f} & "
                f"{e.discretisation:.4f} & {e.model_form:.4f} & "
                f"{e.input_Re:.4f} & {e.input_Tu:.4f} & "
                f"{e.total_uncertainty:.4f} & {e.observed_order:.2f} \\\\"
            )
        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ])
        return "\n".join(lines)

    def _format_summary(self, result: UncertaintyBudgetResult) -> str:
        """One-line summary per QoI."""
        parts = []
        for e in result.entries:
            error = abs(e.fine_value - e.experimental)
            covered = error < 2 * e.total_uncertainty
            parts.append(
                f"{e.qoi}: error={error:.4f}, U_total={e.total_uncertainty:.4f} "
                f"→ {'VALIDATED' if covered else 'NOT VALIDATED'} "
                f"(|E| < 2U: {covered})"
            )
        return f"{result.description}\n" + "\n".join(parts)


# =============================================================================
# Multi-Case Runner
# =============================================================================

def run_all_cases(
    cases: List[str] = None,
) -> Dict[str, UncertaintyBudgetResult]:
    """
    Compute uncertainty budgets for all (or selected) cases.

    Returns dict mapping case name → UncertaintyBudgetResult.
    """
    if cases is None:
        cases = list(CASE_DATA.keys())

    results = {}
    for case in cases:
        budget = SeparationGCIBudget(case)
        results[case] = budget.compute()
        logger.info(results[case].summary)

    return results


def print_combined_report(results: Dict[str, UncertaintyBudgetResult]) -> str:
    """
    Generate a combined text report across all cases.
    """
    lines = [
        "=" * 100,
        "SEPARATION-METRIC GRID CONVERGENCE & UNCERTAINTY BUDGET",
        "Per ASME V&V 20-2009 / AIAA G-077 / Celik et al. (2008)",
        "=" * 100,
        "",
    ]

    for case, result in results.items():
        lines.append(result.text_table)
        lines.append("")
        lines.append(result.summary)
        lines.append("\n")

    # Cross-case comparison
    lines.append("=" * 80)
    lines.append("CROSS-CASE SUMMARY")
    lines.append("-" * 80)
    lines.append(
        f"{'Case':<25} {'QoI':>12} {'Error':>8} {'U_total':>8} {'Status':>12}"
    )
    lines.append("-" * 80)

    for case, result in results.items():
        for e in result.entries:
            error = abs(e.fine_value - e.experimental)
            status = "VALIDATED" if error < 2 * e.total_uncertainty else "REVIEW"
            lines.append(
                f"{case:<25} {e.qoi:>12} {error:>8.4f} "
                f"{e.total_uncertainty:>8.4f} {status:>12}"
            )

    return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    results = run_all_cases()
    report = print_combined_report(results)
    print(report)

    # LaTeX tables
    for case, result in results.items():
        print(f"\n% LaTeX table: {case}")
        print(result.latex_table)
