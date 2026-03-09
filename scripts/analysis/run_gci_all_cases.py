#!/usr/bin/env python3
"""
Multi-Case GCI Orchestrator
==============================
Runs 3-level Grid Convergence Index analysis for every benchmark case
that defines 3+ mesh levels in config.BENCHMARK_CASES.

Produces a master GCI summary table and per-case detailed reports,
following Celik et al. (2008) and NASA/AIAA guidelines.

Usage
-----
    python run_gci_all_cases.py                  # All eligible cases
    python run_gci_all_cases.py --case nasa_hump # Single case
    python run_gci_all_cases.py --json           # Export JSON
    python run_gci_all_cases.py --latex          # Export LaTeX table
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import BENCHMARK_CASES, RESULTS_DIR
from scripts.validation.gci_harness import GCIStudy, compute_from_cell_counts


# =============================================================================
# Synthetic GCI Data
# =============================================================================
# For each case, define the primary QoI and grid-convergence behavior.
# In production, replace with actual solver outputs on 3 grids.
CASE_GCI_DATA = {
    "flat_plate": {
        "quantities": {
            "Cf_x1": {"fine": 0.00297, "medium": 0.00301, "coarse": 0.00315},
        },
        "ndim": 2,
        "description": "Skin friction at x=1.0",
    },
    "nasa_hump": {
        "quantities": {
            "x_reat_xc": {"fine": 1.11, "medium": 1.14, "coarse": 1.22},
            "Cp_min": {"fine": -0.82, "medium": -0.79, "coarse": -0.72},
        },
        "ndim": 2,
        "description": "Reattachment location and suction peak",
    },
    "backward_facing_step": {
        "quantities": {
            "x_reat_xH": {"fine": 6.10, "medium": 6.22, "coarse": 6.55},
        },
        "ndim": 2,
        "description": "Reattachment length",
    },
    "periodic_hill": {
        "quantities": {
            "x_reat_xh": {"fine": 4.72, "medium": 4.85, "coarse": 5.20},
        },
        "ndim": 2,
        "description": "Reattachment on hill",
    },
    "naca_0012_stall": {
        "quantities": {
            "CL_alpha10": {"fine": 1.0912, "medium": 1.0890, "coarse": 1.0830},
            "CD_alpha10": {"fine": 0.01222, "medium": 0.01235, "coarse": 0.01280},
        },
        "ndim": 2,
        "description": "Force coefficients at alpha=10",
    },
    "beverli_hill": {
        "quantities": {
            "x_sep_xH": {"fine": 0.95, "medium": 1.02, "coarse": 1.18},
        },
        "ndim": 3,
        "description": "Separation onset on 3D hill",
    },
    "boeing_gaussian_bump": {
        "quantities": {
            "x_sep_xL": {"fine": 0.82, "medium": 0.85, "coarse": 0.92},
            "bubble_len": {"fine": 0.33, "medium": 0.30, "coarse": 0.22},
        },
        "ndim": 3,
        "description": "3D Gaussian bump separation",
    },
    "bump_3d_channel": {
        "quantities": {
            "Cf_crest": {"fine": -0.00050, "medium": -0.00035, "coarse": -0.00010},
        },
        "ndim": 3,
        "description": "Centerline Cf at bump crest",
    },
    "axisymmetric_jet": {
        "quantities": {
            "core_length_xD": {"fine": 5.20, "medium": 5.05, "coarse": 4.70},
        },
        "ndim": 2,
        "description": "Potential core length",
    },
    "bachalo_johnson": {
        "quantities": {
            "x_sep_xc": {"fine": 0.68, "medium": 0.70, "coarse": 0.75},
        },
        "ndim": 2,
        "description": "Shock-induced separation onset",
    },
}


# =============================================================================
# GCI Orchestrator
# =============================================================================
@dataclass
class CaseGCIResult:
    """GCI results for a single benchmark case."""
    case_name: str
    case_label: str
    description: str
    quantities: Dict  # {qty_name: GCIResult}
    grid_levels: Dict[str, int]
    refinement_ratios: Dict[str, float]
    all_in_asymptotic_range: bool = True


def get_eligible_cases() -> List[str]:
    """Return case names with 3+ mesh levels and GCI data defined."""
    eligible = []
    for name, case in BENCHMARK_CASES.items():
        if (hasattr(case, 'mesh_levels') and case.mesh_levels
                and len(case.mesh_levels) >= 3 and name in CASE_GCI_DATA):
            eligible.append(name)
    return sorted(eligible)


def run_gci_for_case(case_name: str) -> CaseGCIResult:
    """
    Run 3-level GCI for a single benchmark case.

    Uses the three finest grids available.
    """
    case = BENCHMARK_CASES[case_name]
    gci_data = CASE_GCI_DATA[case_name]
    ndim = gci_data["ndim"]

    # Get 3 finest grids
    levels = sorted(case.mesh_levels.items(), key=lambda x: x[1])
    if len(levels) < 3:
        raise ValueError(f"{case_name}: need 3+ grids, have {len(levels)}")

    # Use finest 3
    coarse_name, N_coarse = levels[-3]
    medium_name, N_medium = levels[-2]
    fine_name, N_fine = levels[-1]

    r21, r32 = compute_from_cell_counts(N_coarse, N_medium, N_fine, ndim=ndim)

    study = GCIStudy(r21=r21, r32=r32)

    for qty_name, values in gci_data["quantities"].items():
        study.add_quantity(
            qty_name,
            f_coarse=values["coarse"],
            f_medium=values["medium"],
            f_fine=values["fine"],
        )

    results = study.compute()

    all_asymptotic = all(r.in_asymptotic_range for r in results.values())

    return CaseGCIResult(
        case_name=case_name,
        case_label=case.name,
        description=gci_data["description"],
        quantities=results,
        grid_levels={coarse_name: N_coarse, medium_name: N_medium, fine_name: N_fine},
        refinement_ratios={"r21": r21, "r32": r32},
        all_in_asymptotic_range=all_asymptotic,
    )


def run_all_cases(
    cases: Optional[List[str]] = None,
) -> Dict[str, CaseGCIResult]:
    """Run GCI for all eligible cases."""
    if cases is None:
        cases = get_eligible_cases()

    results = {}
    for name in cases:
        try:
            results[name] = run_gci_for_case(name)
            print(f"  [OK] {name}: {len(CASE_GCI_DATA[name]['quantities'])} quantities")
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")

    return results


# =============================================================================
# Summary Table
# =============================================================================
def master_summary_table(results: Dict[str, CaseGCIResult]) -> str:
    """Generate master GCI summary table across all cases."""
    lines = [
        "=" * 95,
        "Grid Convergence Index (GCI) — All-Case Summary",
        "=" * 95,
        f"  Method: Celik et al. (2008) iterative order estimation, Fs=1.25",
        "",
        f"{'Case':<25} {'Quantity':<18} {'f_fine':>10} {'f_extrap':>10} "
        f"{'p':>6} {'GCI%':>8} {'Asymp?':>7}",
        "-" * 95,
    ]

    total_quantities = 0
    in_asymptotic = 0

    for case_name, cr in results.items():
        first = True
        for qty_name, r in cr.quantities.items():
            total_quantities += 1
            if r.in_asymptotic_range:
                in_asymptotic += 1

            label = cr.case_label[:24] if first else ""
            asymp = "YES" if r.in_asymptotic_range else "NO"
            lines.append(
                f"{label:<25} {qty_name:<18} {r.f_fine:>10.5f} "
                f"{r.extrapolated_value:>10.5f} {r.observed_order:>6.2f} "
                f"{r.gci_fine_pct:>7.2f}% {asymp:>7}"
            )
            first = False

    lines.extend([
        "-" * 95,
        f"  Total quantities: {total_quantities}",
        f"  In asymptotic range: {in_asymptotic}/{total_quantities} "
        f"({100*in_asymptotic/max(total_quantities,1):.0f}%)",
        "",
        "  Reference: Celik et al. (2008), J. Fluids Eng. 130(7), 078001",
        "  NASA/AIAA GCI standard: GCI_fine < 5% recommended for validation",
    ])

    return "\n".join(lines)


def to_json(results: Dict[str, CaseGCIResult], output_path: Path) -> None:
    """Export GCI results to JSON."""
    data = {}
    for case_name, cr in results.items():
        data[case_name] = {
            "case_label": cr.case_label,
            "description": cr.description,
            "grid_levels": cr.grid_levels,
            "refinement_ratios": cr.refinement_ratios,
            "all_in_asymptotic_range": cr.all_in_asymptotic_range,
            "quantities": {
                qty_name: {
                    "f_fine": r.f_fine,
                    "f_medium": r.f_medium,
                    "f_coarse": r.f_coarse,
                    "extrapolated": r.extrapolated_value,
                    "observed_order": r.observed_order,
                    "gci_fine_pct": r.gci_fine_pct,
                    "gci_medium_pct": r.gci_medium_pct,
                    "convergence_type": r.convergence_type,
                    "in_asymptotic_range": r.in_asymptotic_range,
                }
                for qty_name, r in cr.quantities.items()
            },
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2))
    print(f"GCI results saved to {output_path}")


def to_latex(results: Dict[str, CaseGCIResult]) -> str:
    """Generate LaTeX table snippet for GCI results."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Grid Convergence Index for all benchmark cases}",
        r"\label{tab:gci_all}",
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Case & Quantity & $f_\text{fine}$ & $f_\text{extrap}$ & $p$ & GCI\textsubscript{fine} (\%) & Asymptotic? \\",
        r"\midrule",
    ]

    for case_name, cr in results.items():
        first = True
        for qty_name, r in cr.quantities.items():
            label = cr.case_label.replace("&", r"\&")[:30] if first else ""
            asymp = r"\checkmark" if r.in_asymptotic_range else r"$\times$"
            lines.append(
                f"  {label} & {qty_name} & {r.f_fine:.5f} & "
                f"{r.extrapolated_value:.5f} & {r.observed_order:.2f} & "
                f"{r.gci_fine_pct:.2f} & {asymp} \\\\"
            )
            first = False

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)


# =============================================================================
# CLI
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Multi-Case GCI Orchestrator")
    parser.add_argument("--case", type=str, default=None,
                        help="Run GCI for single case")
    parser.add_argument("--json", action="store_true", help="Export JSON")
    parser.add_argument("--latex", action="store_true", help="Export LaTeX table")
    parser.add_argument("--list", action="store_true", help="List eligible cases")
    args = parser.parse_args()

    if args.list:
        eligible = get_eligible_cases()
        print(f"Eligible cases ({len(eligible)}):")
        for c in eligible:
            case = BENCHMARK_CASES[c]
            n_grids = len(case.mesh_levels)
            print(f"  {c:<30} {n_grids} grids")
        return

    cases = [args.case] if args.case else None
    print("Running GCI analysis...")
    results = run_all_cases(cases)

    report = master_summary_table(results)
    print(f"\n{report}")

    output_dir = RESULTS_DIR / "gci_all_cases"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "gci_summary.txt").write_text(report)

    if args.json:
        to_json(results, output_dir / "gci_all_cases.json")

    if args.latex:
        latex = to_latex(results)
        (output_dir / "gci_table.tex").write_text(latex)
        print(f"LaTeX table saved to {output_dir / 'gci_table.tex'}")


if __name__ == "__main__":
    main()
