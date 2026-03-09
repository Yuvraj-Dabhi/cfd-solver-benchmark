#!/usr/bin/env python3
"""
Extended Wall Hump Run — 409×109 xfine Grid + Automated GCI
==============================================================
Generates SU2 INC_RANS config for SA on the xfine (409×109) grid,
then computes 3-level GCI on coarse/medium/fine results.

Usage
-----
    python run_wall_hump_xfine.py [--dry-run] [--gci-only]
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


# Grid specifications
GRIDS = {
    "coarse":  {"dims": (103, 28), "n_cells": 2884},
    "medium":  {"dims": (205, 55), "n_cells": 11275},
    "fine":    {"dims": (409, 109), "n_cells": 44581},
}

CASE_DIR = PROJECT_ROOT / "runs" / "wall_hump"
EXP_DATA = {
    "x_sep": 0.665,
    "x_reat": 1.11,
    "bubble_length": 0.445,
}


def generate_xfine_config(case_dir: Path, n_iter: int = 15000) -> Path:
    """Generate SU2 config for SA on the xfine (409×109) wall hump grid."""
    run_dir = case_dir / "hump_SA_fine"
    run_dir.mkdir(parents=True, exist_ok=True)

    config = f"""\
% ============================================================
% NASA Wall-Mounted Hump — SA on 409×109 (xfine) Grid
% Target: residuals to 1e-10+, thorough grid convergence
% ============================================================

SOLVER= INC_RANS
KIND_TURB_MODEL= SA
MATH_PROBLEM= DIRECT
RESTART_SOL= NO

% --- Freestream ---
INC_DENSITY_MODEL= CONSTANT
INC_DENSITY_INIT= 1.225
INC_VELOCITY_INIT= (34.6, 0.0, 0.0)
INC_TEMPERATURE_INIT= 300.0
VISCOSITY_MODEL= CONSTANT
MU_CONSTANT= 1.7894e-05
FREESTREAM_NU_FACTOR= 3.0

% --- Reference ---
REF_LENGTH= 0.420
REYNOLDS_NUMBER= 936000

% --- Boundary Conditions ---
MARKER_HEATFLUX= ( wall, 0.0 )
MARKER_INLET= ( inlet, 300.0, 34.6, 1.0, 0.0, 0.0 )
MARKER_OUTLET= ( outlet, 0.0 )
MARKER_SYM= ( top )

% --- Numerics ---
NUM_METHOD_GRAD= GREEN_GAUSS
CFL_NUMBER= 25.0
CFL_ADAPT= YES
CFL_ADAPT_PARAM= ( 0.1, 2.0, 25.0, 1e10 )

CONV_NUM_METHOD_FLOW= FDS
MUSCL_FLOW= YES
SLOPE_LIMITER_FLOW= VENKATAKRISHNAN
VENKAT_LIMITER_COEFF= 0.05

CONV_NUM_METHOD_TURB= SCALAR_UPWIND
MUSCL_TURB= NO

% --- Convergence (strict for V&V) ---
ITER= {n_iter}
CONV_RESIDUAL_MINVAL= -12
CONV_FIELD= RMS_PRESSURE, RMS_NU_TILDE
CONV_STARTITER= 100

% --- Output ---
MESH_FILENAME= mesh_409x109.su2
OUTPUT_FILES= RESTART, PARAVIEW_MULTIBLOCK, SURFACE_PARAVIEW
VOLUME_OUTPUT= RESIDUAL, PRIMITIVE, TURBULENT
SURFACE_OUTPUT= SKIN_FRICTION, PRESSURE_COEFFICIENT
OUTPUT_WRT_FREQ= 500
CONV_FILENAME= history
RESTART_FILENAME= restart_flow.dat
SOLUTION_FILENAME= solution_flow.dat
"""

    config_path = run_dir / "wall_hump_xfine.cfg"
    config_path.write_text(config)
    print(f"Generated xfine config: {config_path}")
    return config_path


def run_gci_analysis():
    """Run 3-level GCI analysis on existing wall hump results."""
    from scripts.validation.gci_harness import GCIStudy, compute_from_cell_counts

    print("=" * 60)
    print("WALL HUMP 3-LEVEL GCI ANALYSIS (SA)")
    print("=" * 60)

    # Compute refinement ratios from cell counts
    n_c = GRIDS["coarse"]["n_cells"]
    n_m = GRIDS["medium"]["n_cells"]
    n_f = GRIDS["fine"]["n_cells"]
    r21, r32 = compute_from_cell_counts(n_c, n_m, n_f, ndim=2)

    print(f"\nGrid cells: coarse={n_c}, medium={n_m}, fine={n_f}")
    print(f"Refinement ratios: r21={r21:.4f}, r32={r32:.4f}")

    # Try to extract data from VTU files
    data = {}
    for level in ["coarse", "medium", "fine"]:
        vtu_path = CASE_DIR / f"hump_SA_{level}" / "surface_flow.vtu"
        if vtu_path.exists():
            try:
                from scripts.validation.gci_harness import extract_quantities_from_vtu
                surface = extract_quantities_from_vtu(str(vtu_path))
                x = surface["x"]
                cf = surface.get("Skin_Friction_Coefficient", np.zeros_like(x))

                # Find separation and reattachment
                x_sep = _find_zero_crossing(x, cf, direction="down", x_range=(0.6, 0.75))
                x_reat = _find_zero_crossing(x, cf, direction="up", x_range=(0.9, 1.3))

                data[level] = {
                    "x_sep": x_sep,
                    "x_reat": x_reat,
                    "bubble_length": x_reat - x_sep if x_sep and x_reat else None,
                    "cf_min": float(np.min(cf)) if len(cf) > 0 else None,
                }
                print(f"\n{level}: x_sep={x_sep}, x_reat={x_reat}")
            except Exception as e:
                print(f"  Warning: Could not extract {level} data: {e}")
        else:
            print(f"  VTU not found for {level}: {vtu_path}")

    # If we have data from all 3 levels, compute GCI
    if len(data) == 3 and all(data[l].get("x_sep") for l in data):
        study = GCIStudy(r21=r21, r32=r32)

        for qty in ["x_sep", "x_reat", "bubble_length"]:
            vals = [data[l][qty] for l in ["coarse", "medium", "fine"]]
            if all(v is not None for v in vals):
                study.add_quantity(qty, *vals)

        study.compute()
        print("\n" + study.summary_table())

        # Save results
        output_path = PROJECT_ROOT / "plots" / "wall_hump" / "gci_xfine_results.json"
        study.to_json(output_path)
        print(f"\nAll converged (GCI < 5%): {study.all_converged}")
    else:
        print("\nInsufficient data for GCI analysis. Run simulations first.")
        # Print what data we have
        print("\nTo complete GCI, run SU2 on all three grid levels:")
        for level in ["coarse", "medium", "fine"]:
            dims = GRIDS[level]["dims"]
            print(f"  {level}: {dims[0]}×{dims[1]} ({GRIDS[level]['n_cells']} cells)")


def _find_zero_crossing(x, cf, direction="down", x_range=None):
    """Find zero crossing of Cf in the specified x range."""
    if x_range:
        mask = (x >= x_range[0]) & (x <= x_range[1])
        x_sub = x[mask]
        cf_sub = cf[mask]
    else:
        x_sub, cf_sub = x, cf

    if len(x_sub) < 2:
        return None

    sort_idx = np.argsort(x_sub)
    x_sub = x_sub[sort_idx]
    cf_sub = cf_sub[sort_idx]

    for i in range(len(cf_sub) - 1):
        if direction == "down" and cf_sub[i] > 0 and cf_sub[i + 1] <= 0:
            # Linear interpolation
            frac = cf_sub[i] / (cf_sub[i] - cf_sub[i + 1])
            return float(x_sub[i] + frac * (x_sub[i + 1] - x_sub[i]))
        elif direction == "up" and cf_sub[i] < 0 and cf_sub[i + 1] >= 0:
            frac = -cf_sub[i] / (cf_sub[i + 1] - cf_sub[i])
            return float(x_sub[i] + frac * (x_sub[i + 1] - x_sub[i]))

    return None


def main():
    parser = argparse.ArgumentParser(description="Wall Hump xfine grid run + GCI")
    parser.add_argument("--dry-run", action="store_true", help="Generate config only")
    parser.add_argument("--gci-only", action="store_true", help="Run GCI analysis only")
    parser.add_argument("--n-iter", type=int, default=15000, help="Number of iterations")
    args = parser.parse_args()

    if args.gci_only:
        run_gci_analysis()
        return

    # Generate config
    config_path = generate_xfine_config(CASE_DIR, args.n_iter)

    if args.dry_run:
        print("\nDry run complete. Config generated but SU2 not executed.")
        return

    # Run SU2
    run_dir = config_path.parent
    print(f"\nRunning SU2 in {run_dir}...")
    try:
        result = subprocess.run(
            ["SU2_CFD", str(config_path)],
            cwd=str(run_dir),
            capture_output=True,
            text=True,
            timeout=7200,  # 2 hour timeout
        )
        if result.returncode == 0:
            print("SU2 completed successfully.")
            run_gci_analysis()
        else:
            print(f"SU2 failed with return code {result.returncode}")
            print(result.stderr[-500:] if result.stderr else "")
    except FileNotFoundError:
        print("SU2_CFD not found. Install SU2 and add to PATH.")
    except subprocess.TimeoutExpired:
        print("SU2 timed out after 2 hours.")


if __name__ == "__main__":
    main()
