#!/usr/bin/env python3
"""
Backward-Facing Step Benchmark Runner (Driver & Seegmiller 1985)
==================================================================
Canonical geometry-forced separation with extensive experimental validation.

This case deliberately contrasts model performance:
  - SA and SST: <5% reattachment error (good RANS cases)
  - k-ε: systematic -20% underprediction of x_R (demonstrating known deficiency)

Including k-ε here is intentional — it shows a quantifiable failure mode that
motivates the need for ML-augmented closures.

Flow Physics
------------
  - Re_H = 36,000 based on step height H = 0.0127 m
  - Expansion ratio 1.125 (step in 8H channel)
  - Fixed separation at step edge
  - Reattachment at x/H = 6.26 ± 0.10

References
----------
  - Driver & Seegmiller (1985), AIAA J. 23(2), pp. 163–171
  - NASA TMR: https://turbmodels.larc.nasa.gov/backstep_val.html

Usage
-----
    python run_backward_facing_step.py [--model SA|SST|KE] [--grid coarse|medium|fine]
"""

import argparse
import json
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

CASE_CONFIG = BENCHMARK_CASES["backward_facing_step"]

GRID_LEVELS = {
    "coarse":  {"nx": 200, "ny": 80,  "cells": 40_000},
    "medium":  {"nx": 400, "ny": 120, "cells": 90_000},
    "fine":    {"nx": 600, "ny": 200, "cells": 200_000},
    "xfine":   {"nx": 800, "ny": 300, "cells": 450_000},
}

# Experimental reference (Driver & Seegmiller 1985)
EXP_REFERENCE = {
    "Re_H": 36_000,
    "U_ref": 44.2,          # m/s
    "H": 0.0127,            # Step height (m)
    "expansion_ratio": 1.125,
    "x_reat_xH": 6.26,      # Reattachment point
    "x_reat_uncertainty": 0.10,
    "upstream_length_H": 50,
    "downstream_length_H": 200,
}

# Profile stations (x/H from step)
PROFILE_STATIONS = [1, 4, 6, 10]
PROFILE_QUANTITIES = ["U", "uu", "vv", "uv"]

# Expected RANS model performance
EXPECTED_PERFORMANCE = {
    "SA":  {"x_reat_xH": 6.26, "error_pct": "<5%"},
    "SST": {"x_reat_xH": 6.26, "error_pct": "<5%"},
    "KE":  {"x_reat_xH": 5.00, "error_pct": "~-20%"},
}


# =============================================================================
# SU2 Configuration Generator
# =============================================================================

def generate_su2_config(
    case_dir: Path,
    mesh_file: str,
    model: str = "SA",
    n_iter: int = 10000,
) -> Path:
    """Generate SU2 configuration for backward-facing step."""

    Re_H = EXP_REFERENCE["Re_H"]
    U_ref = EXP_REFERENCE["U_ref"]
    H = EXP_REFERENCE["H"]
    T_ref = 300.0
    nu = U_ref * H / Re_H

    model_map = {"SA": "SA", "SST": "SST", "KE": "KE"}

    config = f"""\
% ============================================================
% Backward-Facing Step — Driver & Seegmiller (1985)
% Re_H = {Re_H}, H = {H} m, U_ref = {U_ref} m/s
% Model: {model}
% ============================================================

SOLVER = RANS
KIND_TURB_MODEL = {model_map.get(model, "SA")}
MATH_PROBLEM = DIRECT

% --- Flow conditions ---
MACH_NUMBER = 0.128
AOA = 0.0
REYNOLDS_NUMBER = {Re_H}
REYNOLDS_LENGTH = {H}
FREESTREAM_TEMPERATURE = {T_ref}
FREESTREAM_TURBULENCEINTENSITY = 0.03
VISCOSITY_MODEL = CONSTANT_VISCOSITY
MU_CONSTANT = {nu * 1.225:.8e}

% --- Reference values ---
REF_ORIGIN_MOMENT_X = 0.0
REF_ORIGIN_MOMENT_Y = 0.0
REF_ORIGIN_MOMENT_Z = 0.0
REF_LENGTH = {H}
REF_AREA = {H}

% --- Boundary conditions ---
MARKER_HEATFLUX = ( wall_upper, 0.0, wall_lower, 0.0, step, 0.0 )
INC_INLET_TYPE = VELOCITY_INLET
MARKER_INLET = ( inlet, {T_ref}, {U_ref}, 1.0, 0.0, 0.0 )
MARKER_OUTLET = ( outlet, 0.0 )
MARKER_PLOTTING = ( wall_lower, step )
MARKER_MONITORING = ( wall_lower, step )

% --- Numerical methods ---
NUM_METHOD_GRAD = GREEN_GAUSS
CFL_NUMBER = 15.0
CFL_ADAPT = YES
CFL_ADAPT_PARAM = ( 0.5, 1.5, 1.0, 50.0 )

CONV_NUM_METHOD_FLOW = ROE
MUSCL_FLOW = YES
SLOPE_LIMITER_FLOW = VENKATAKRISHNAN
VENKAT_LIMITER_COEFF = 0.03

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
SCREEN_OUTPUT = (INNER_ITER, RMS_DENSITY, RMS_MOMENTUM-X, RMS_MOMENTUM-Y, RMS_TKE)
HISTORY_OUTPUT = (ITER, RMS_RES, FLOW_COEFF)
"""

    config_path = case_dir / f"bfs_{model}.cfg"
    case_dir.mkdir(parents=True, exist_ok=True)
    config_path.write_text(config)
    return config_path


# =============================================================================
# Results Parser
# =============================================================================

def parse_results(case_dir: Path, model: str) -> Dict:
    """Parse BFS results and extract reattachment location."""

    result = {
        "model": model,
        "case_dir": str(case_dir),
        "converged": False,
        "x_reat_xH": None,
        "x_reat_error_pct": None,
    }

    surface_file = case_dir / "surface_flow.csv"
    if not surface_file.exists():
        result["error"] = "No surface output found"
        return result

    result["converged"] = True

    try:
        import pandas as pd
        df = pd.read_csv(surface_file)
        x = df.get("x", df.iloc[:, 0]).values
        cf = df.get("Skin_Friction_Coefficient_X",
                     df.get("Cf_x", None))

        if cf is not None:
            cf = cf.values
            H = EXP_REFERENCE["H"]

            # Reattachment: Cf crosses zero (- → +) downstream of step
            x_norm = x / H  # Normalize by step height

            for i in range(1, len(cf)):
                if x_norm[i] > 0.5:  # Only look downstream of step
                    if cf[i-1] <= 0 and cf[i] > 0:
                        result["x_reat_xH"] = float(x_norm[i])
                        break

            if result["x_reat_xH"]:
                result["x_reat_error_pct"] = float(
                    (result["x_reat_xH"] - EXP_REFERENCE["x_reat_xH"]) /
                    EXP_REFERENCE["x_reat_xH"] * 100
                )
                result["within_uncertainty"] = (
                    abs(result["x_reat_xH"] - EXP_REFERENCE["x_reat_xH"]) <=
                    EXP_REFERENCE["x_reat_uncertainty"]
                )

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
    """Run a single backward-facing step simulation."""

    runs_dir = PROJECT_ROOT / "runs" / "backward_facing_step"
    case_name = f"bfs_{model}_{grid_level}"
    case_dir = runs_dir / case_name
    case_dir.mkdir(parents=True, exist_ok=True)

    grid = GRID_LEVELS.get(grid_level, GRID_LEVELS["medium"])

    print(f"\n{'='*60}")
    print(f"  Backward-Facing Step: {model} on {grid_level} grid ({grid['cells']:,} cells)")
    print(f"  Re_H = {EXP_REFERENCE['Re_H']:,}, U = {EXP_REFERENCE['U_ref']} m/s")
    print(f"  Exp ref: x_R/H = {EXP_REFERENCE['x_reat_xH']} ± "
          f"{EXP_REFERENCE['x_reat_uncertainty']}")
    print(f"  Expected {model}: {EXPECTED_PERFORMANCE.get(model, {}).get('error_pct', 'N/A')}")
    print(f"{'='*60}")

    mesh_file = f"bfs_{grid_level}.su2"
    config_path = generate_su2_config(case_dir, mesh_file, model, n_iter)
    print(f"  Config written: {config_path}")

    if dry_run:
        print("  [DRY RUN] Skipping simulation")
        return {"model": model, "grid": grid_level, "config": str(config_path)}

    # Find SU2
    su2_exe = None
    for exe_name in ["SU2_CFD", "SU2_CFD.exe"]:
        try:
            subprocess.run([exe_name, "--version"], capture_output=True, timeout=5)
            su2_exe = exe_name
            break
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue

    if su2_exe is None:
        print("  [WARN] SU2_CFD not found. Config generated but simulation skipped.")
        return {"model": model, "grid": grid_level, "config": str(config_path),
                "error": "SU2 not found"}

    # Run
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

    except subprocess.TimeoutExpired:
        return {"model": model, "grid": grid_level, "error": "Timeout"}

    results = parse_results(case_dir, model)
    results["grid_level"] = grid_level
    results["n_cells"] = grid["cells"]

    # Save
    with open(case_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    if results.get("x_reat_xH"):
        print(f"\n  Results:")
        print(f"    x_R/H = {results['x_reat_xH']:.3f} "
              f"(Exp: {EXP_REFERENCE['x_reat_xH']} ± {EXP_REFERENCE['x_reat_uncertainty']})")
        print(f"    Error = {results['x_reat_error_pct']:.1f}%")
        print(f"    Within uncertainty: {'YES' if results['within_uncertainty'] else 'NO'}")

    return results


def run_model_comparison(
    grid_level: str = "medium",
    models: List[str] = None,
    n_iter: int = 10000,
    dry_run: bool = False,
):
    """
    Run SA, SST, and k-ε comparison.

    The k-ε result is deliberately included to demonstrate the systematic
    -20% underprediction of x_R — a known closure deficiency that motivates
    ML augmentation.
    """
    if models is None:
        models = ["SA", "SST", "KE"]

    all_results = {}
    for model in models:
        result = run_case(model, grid_level, n_iter, dry_run)
        all_results[model] = result

    print(f"\n{'='*70}")
    print(f"  MODEL COMPARISON — Backward-Facing Step ({grid_level} grid)")
    print(f"{'='*70}")
    print(f"  {'Model':<8} {'x_R/H':<10} {'Error%':<12} {'Note'}")
    print(f"  {'-'*55}")
    print(f"  {'Exp':<8} {EXP_REFERENCE['x_reat_xH']:<10.2f} {'—':<12} "
          f"Driver & Seegmiller (1985)")

    for model, res in all_results.items():
        xr = res.get("x_reat_xH", "N/A")
        err = res.get("x_reat_error_pct", "N/A")
        note = EXPECTED_PERFORMANCE.get(model, {}).get("error_pct", "")
        if isinstance(xr, float):
            xr = f"{xr:.2f}"
        if isinstance(err, float):
            err = f"{err:+.1f}%"
        print(f"  {model:<8} {xr:<10} {err:<12} expected: {note}")

    summary_path = PROJECT_ROOT / "results" / "bfs_comparison.json"
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
        description="Backward-Facing Step Benchmark Runner (Driver & Seegmiller 1985)"
    )
    parser.add_argument("--model", default="SA", choices=["SA", "SST", "KE", "ALL"],
                        help="Turbulence model (default: SA)")
    parser.add_argument("--grid", default="medium",
                        choices=["coarse", "medium", "fine", "xfine"],
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
