#!/usr/bin/env python3
"""
Periodic Hill Benchmark Runner (ERCOFTAC Case #081)
=====================================================
DNS-validated curvature-driven separation on a periodically constricted channel.

This is a **genuinely challenging** separation case that exposes fundamental
RANS closure failures:
  - SA overpredicts separation extent
  - SST shows >15% reattachment error
  - k-ε can fail entirely on curved-wall separation

Breuer et al. (2009) DNS provides ground truth at Re_h = 10,595.

Flow Physics
------------
  - Curvature-driven separation on the lee side of the hill
  - Large recirculation zone (x_sep/h ~ 0.22, x_reat/h ~ 4.72)
  - Strong adverse pressure gradient recovery
  - Periodic streamwise domain: 9h × 3.036h

References
----------
  - Breuer et al. (2009), Computers & Fluids 38(2), pp. 433–457
  - ERCOFTAC Classic Database #081

Usage
-----
    python run_periodic_hill.py [--model SA|SST|KE] [--grid coarse|medium|fine]
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import BENCHMARK_CASES, RESULTS_DIR


# =============================================================================
# Configuration
# =============================================================================

CASE_CONFIG = BENCHMARK_CASES["periodic_hill"]

GRID_LEVELS = {
    "coarse":  {"nx": 100, "ny": 50,  "cells": 50_000},
    "medium":  {"nx": 200, "ny": 100, "cells": 150_000},
    "fine":    {"nx": 400, "ny": 200, "cells": 400_000},
}

# DNS reference data (Breuer et al. 2009)
DNS_REFERENCE = {
    "Re_h": 10_595,
    "x_sep_xh": 0.22,       # Separation point
    "x_reat_xh": 4.72,      # Reattachment point
    "bubble_length_xh": 4.50,
    "domain": "9h × 3.036h (periodic)",
}

# Profile extraction stations (x/h)
PROFILE_STATIONS = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]


# =============================================================================
# SU2 Configuration Generator
# =============================================================================

def generate_su2_config(
    case_dir: Path,
    mesh_file: str,
    model: str = "SA",
    n_iter: int = 10000,
) -> Path:
    """
    Generate SU2 configuration for periodic hill.

    Parameters
    ----------
    case_dir : Path
        Output directory for config and results.
    mesh_file : str
        Path to mesh file.
    model : str
        Turbulence model: "SA", "SST", or "KE".
    n_iter : int
        Number of iterations.

    Returns
    -------
    Path to the config file.
    """
    Re_h = DNS_REFERENCE["Re_h"]
    # Reference quantities
    U_ref = 1.0  # Reference velocity (non-dimensional)
    h = 1.0      # Hill height (reference length)
    nu = U_ref * h / Re_h  # Kinematic viscosity

    model_map = {
        "SA": "SA",
        "SST": "SST",
        "KE": "KE",
    }

    config = f"""\
% ============================================================
% Periodic Hill — ERCOFTAC #081 / Breuer et al. (2009) DNS
% Re_h = {Re_h}, Domain = 9h x 3.036h, Periodic BC
% Model: {model}
% ============================================================

SOLVER = RANS
KIND_TURB_MODEL = {model_map.get(model, "SA")}
MATH_PROBLEM = DIRECT

% --- Flow conditions ---
MACH_NUMBER = 0.1
AOA = 0.0
REYNOLDS_NUMBER = {Re_h}
REYNOLDS_LENGTH = {h}
FREESTREAM_TEMPERATURE = 300.0
FREESTREAM_TURBULENCEINTENSITY = 0.05
VISCOSITY_MODEL = CONSTANT_VISCOSITY
MU_CONSTANT = {nu * 1.0:.8e}

% --- Reference values ---
REF_ORIGIN_MOMENT_X = 0.0
REF_ORIGIN_MOMENT_Y = 0.0
REF_ORIGIN_MOMENT_Z = 0.0
REF_LENGTH = {h}
REF_AREA = {h}

% --- Boundary conditions ---
MARKER_HEATFLUX = ( wall, 0.0 )
MARKER_PERIODIC = ( inlet, outlet, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 9.0, 0.0, 0.0 )
MARKER_PLOTTING = ( wall )
MARKER_MONITORING = ( wall )

% --- Numerical methods ---
NUM_METHOD_GRAD = GREEN_GAUSS
CFL_NUMBER = 10.0
CFL_ADAPT = YES
CFL_ADAPT_PARAM = ( 0.5, 1.5, 1.0, 50.0 )

CONV_NUM_METHOD_FLOW = ROE
MUSCL_FLOW = YES
SLOPE_LIMITER_FLOW = VENKATAKRISHNAN
VENKAT_LIMITER_COEFF = 0.05

CONV_NUM_METHOD_TURB = SCALAR_UPWIND
MUSCL_TURB = NO

TIME_DISCRE_FLOW = EULER_IMPLICIT
TIME_DISCRE_TURB = EULER_IMPLICIT

% --- Linear solver ---
LINEAR_SOLVER = FGMRES
LINEAR_SOLVER_PREC = ILU
LINEAR_SOLVER_ERROR = 1e-10
LINEAR_SOLVER_ITER = 20

% --- Convergence ---
ITER = {n_iter}
CONV_RESIDUAL_MINVAL = -10
CONV_STARTITER = 10
CONV_FIELD = RMS_DENSITY

% --- Input/Output ---
MESH_FILENAME = {mesh_file}
MESH_FORMAT = SU2
SOLUTION_FILENAME = solution.dat
RESTART_FILENAME = restart.dat
CONV_FILENAME = history
VOLUME_FILENAME = flow
SURFACE_FILENAME = surface_flow

OUTPUT_FILES = (RESTART, PARAVIEW, SURFACE_PARAVIEW)
OUTPUT_WRT_FREQ = 1000
SCREEN_OUTPUT = (INNER_ITER, RMS_DENSITY, RMS_MOMENTUM-X, RMS_MOMENTUM-Y, RMS_TKE, LIFT, DRAG)
HISTORY_OUTPUT = (ITER, RMS_RES, AERO_COEFF, FLOW_COEFF)
"""

    config_path = case_dir / f"periodic_hill_{model}.cfg"
    case_dir.mkdir(parents=True, exist_ok=True)
    config_path.write_text(config)
    return config_path


# =============================================================================
# Results Parser
# =============================================================================

def parse_results(case_dir: Path, model: str) -> Dict:
    """
    Parse SU2 results and extract separation/reattachment metrics.

    Returns dict with Cf distribution, separation/reattachment locations,
    and comparison to DNS reference.
    """
    result = {
        "model": model,
        "case_dir": str(case_dir),
        "converged": False,
        "x_sep": None,
        "x_reat": None,
        "bubble_length": None,
        "dns_comparison": {},
    }

    # Check for surface output
    surface_file = case_dir / "surface_flow.csv"
    if not surface_file.exists():
        surface_file = case_dir / "surface_flow.vtu"
    if not surface_file.exists():
        result["error"] = "No surface output found"
        return result

    result["converged"] = True

    # Parse Cf from surface data (if CSV available)
    try:
        import pandas as pd
        if surface_file.suffix == ".csv":
            df = pd.read_csv(surface_file)
            x = df.get("x", df.iloc[:, 0]).values
            cf = df.get("Skin_Friction_Coefficient_X",
                         df.get("Cf_x", None))
            if cf is not None:
                cf = cf.values
                # Find separation (Cf crosses zero, + → -)
                for i in range(1, len(cf)):
                    if cf[i-1] > 0 and cf[i] <= 0:
                        result["x_sep"] = float(x[i])
                        break
                # Find reattachment (Cf crosses zero, - → +)
                for i in range(len(cf)-1, 0, -1):
                    if cf[i-1] <= 0 and cf[i] > 0:
                        result["x_reat"] = float(x[i])
                        break

                if result["x_sep"] and result["x_reat"]:
                    result["bubble_length"] = result["x_reat"] - result["x_sep"]

                    # DNS comparison
                    result["dns_comparison"] = {
                        "x_sep_error_pct": abs(result["x_sep"] - DNS_REFERENCE["x_sep_xh"]) /
                                            DNS_REFERENCE["x_sep_xh"] * 100,
                        "x_reat_error_pct": abs(result["x_reat"] - DNS_REFERENCE["x_reat_xh"]) /
                                             DNS_REFERENCE["x_reat_xh"] * 100,
                        "bubble_error_pct": abs(result["bubble_length"] - DNS_REFERENCE["bubble_length_xh"]) /
                                             DNS_REFERENCE["bubble_length_xh"] * 100,
                    }

    except Exception as e:
        result["parse_warning"] = str(e)

    return result


# =============================================================================
# Main Runner
# =============================================================================

def run_case(
    model: str = "SA",
    grid_level: str = "medium",
    n_iter: int = 10000,
    dry_run: bool = False,
) -> Dict:
    """
    Run a single periodic hill simulation.

    Parameters
    ----------
    model : str
        Turbulence model ("SA", "SST", "KE").
    grid_level : str
        Grid level ("coarse", "medium", "fine").
    n_iter : int
        Number of iterations.
    dry_run : bool
        If True, only generate config without running.

    Returns
    -------
    dict with results and DNS comparison.
    """
    runs_dir = PROJECT_ROOT / "runs" / "periodic_hill"
    case_name = f"ph_{model}_{grid_level}"
    case_dir = runs_dir / case_name
    case_dir.mkdir(parents=True, exist_ok=True)

    # Grid info
    grid = GRID_LEVELS.get(grid_level, GRID_LEVELS["medium"])

    print(f"\n{'='*60}")
    print(f"  Periodic Hill: {model} on {grid_level} grid ({grid['cells']:,} cells)")
    print(f"  Re_h = {DNS_REFERENCE['Re_h']:,}")
    print(f"  DNS ref: x_sep/h = {DNS_REFERENCE['x_sep_xh']}, "
          f"x_reat/h = {DNS_REFERENCE['x_reat_xh']}")
    print(f"{'='*60}")

    # Generate config
    mesh_file = f"periodic_hill_{grid_level}.su2"
    config_path = generate_su2_config(case_dir, mesh_file, model, n_iter)
    print(f"  Config written: {config_path}")

    if dry_run:
        print("  [DRY RUN] Skipping simulation")
        return {"model": model, "grid": grid_level, "config": str(config_path)}

    # Find SU2 executable
    su2_exe = None
    for exe_name in ["SU2_CFD", "SU2_CFD.exe"]:
        try:
            result = subprocess.run([exe_name, "--version"],
                                     capture_output=True, timeout=5)
            su2_exe = exe_name
            break
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue

    if su2_exe is None:
        print("  [WARN] SU2_CFD not found. Config generated but simulation skipped.")
        return {"model": model, "grid": grid_level, "config": str(config_path),
                "error": "SU2 not found"}

    # Run simulation
    print(f"  Running SU2 ({n_iter} iterations)...")
    try:
        proc = subprocess.run(
            [su2_exe, str(config_path)],
            cwd=str(case_dir),
            capture_output=True,
            text=True,
            timeout=3600,
        )

        if proc.returncode != 0:
            print(f"  [ERROR] SU2 returned code {proc.returncode}")
            return {"model": model, "grid": grid_level, "error": proc.stderr[-500:]}

    except subprocess.TimeoutExpired:
        return {"model": model, "grid": grid_level, "error": "Timeout after 3600s"}

    # Parse results
    results = parse_results(case_dir, model)
    results["grid_level"] = grid_level
    results["n_cells"] = grid["cells"]

    # Save results
    results_file = case_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Print summary
    if results.get("x_sep") and results.get("x_reat"):
        print(f"\n  Results:")
        print(f"    x_sep/h  = {results['x_sep']:.3f} (DNS: {DNS_REFERENCE['x_sep_xh']})")
        print(f"    x_reat/h = {results['x_reat']:.3f} (DNS: {DNS_REFERENCE['x_reat_xh']})")
        print(f"    Bubble   = {results['bubble_length']:.3f} "
              f"(DNS: {DNS_REFERENCE['bubble_length_xh']})")
        if results.get("dns_comparison"):
            dc = results["dns_comparison"]
            print(f"    Errors: sep={dc['x_sep_error_pct']:.1f}%, "
                  f"reat={dc['x_reat_error_pct']:.1f}%, "
                  f"bubble={dc['bubble_error_pct']:.1f}%")
    else:
        print("  [WARN] Could not extract separation metrics from results")

    return results


def run_model_comparison(
    grid_level: str = "medium",
    models: List[str] = None,
    n_iter: int = 10000,
    dry_run: bool = False,
):
    """Run SA, SST, and k-ε on the same grid for model comparison."""
    if models is None:
        models = ["SA", "SST", "KE"]

    all_results = {}
    for model in models:
        result = run_case(model, grid_level, n_iter, dry_run)
        all_results[model] = result

    # Summary table
    print(f"\n{'='*70}")
    print(f"  MODEL COMPARISON — Periodic Hill ({grid_level} grid)")
    print(f"{'='*70}")
    print(f"  {'Model':<8} {'x_sep/h':<12} {'x_reat/h':<12} {'Bubble Err%':<12}")
    print(f"  {'-'*44}")
    print(f"  {'DNS':<8} {DNS_REFERENCE['x_sep_xh']:<12.3f} "
          f"{DNS_REFERENCE['x_reat_xh']:<12.3f} {'—':<12}")

    for model, res in all_results.items():
        x_s = f"{res.get('x_sep', 'N/A')}"
        x_r = f"{res.get('x_reat', 'N/A')}"
        b_err = res.get("dns_comparison", {}).get("bubble_error_pct", "N/A")
        if isinstance(x_s, float):
            x_s = f"{x_s:.3f}"
        if isinstance(x_r, float):
            x_r = f"{x_r:.3f}"
        if isinstance(b_err, float):
            b_err = f"{b_err:.1f}%"
        print(f"  {model:<8} {x_s:<12} {x_r:<12} {b_err:<12}")

    # Save summary
    summary_path = PROJECT_ROOT / "results" / "periodic_hill_comparison.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved: {summary_path}")

    return all_results


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Periodic Hill Benchmark Runner (ERCOFTAC #081)"
    )
    parser.add_argument("--model", default="SA", choices=["SA", "SST", "KE", "ALL"],
                        help="Turbulence model (default: SA)")
    parser.add_argument("--grid", default="medium", choices=["coarse", "medium", "fine"],
                        help="Grid level (default: medium)")
    parser.add_argument("--iter", type=int, default=10000,
                        help="Number of iterations (default: 10000)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Generate config only, don't run simulations")
    parser.add_argument("--comparison", action="store_true",
                        help="Run SA, SST, k-ε comparison")
    args = parser.parse_args()

    if args.comparison or args.model == "ALL":
        run_model_comparison(args.grid, n_iter=args.iter, dry_run=args.dry_run)
    else:
        run_case(args.model, args.grid, args.iter, args.dry_run)


if __name__ == "__main__":
    main()
