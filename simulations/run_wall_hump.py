#!/usr/bin/env python3
"""
NASA Wall-Mounted Hump — TMR Validation Runner
================================================
End-to-end runner for the NASA wall-mounted hump separated flow
validation case (TMR 2DWMH).

Case specification:
  - Geometry: Glauert-Goldschmied body on flat plate (no plenum)
  - Mach = 0.1, Re_c = 936,000 (chord-based, c = 420 mm)
  - Freestream velocity ≈ 34.6 m/s
  - Fully turbulent, SA or SST model
  - BCs: Inflow subsonic, outflow back-pressure, upper wall inviscid,
          lower wall adiabatic no-slip

Usage:
  python run_wall_hump.py --dry-run               Setup only
  python run_wall_hump.py --model SA -t 7          Run with OpenMP
  python run_wall_hump.py --validate-only          Post-process only
  python run_wall_hump.py --grid fine --model SST  Run SST on fine

Reference: https://turbmodels.larc.nasa.gov/nasahump_val.html
"""

import argparse
import csv as csv_mod
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent

# ============================================================================
# Flow Conditions (TMR specification)
# ============================================================================

CASE_CONFIG = {
    "mach":                    0.1,
    "reynolds":                936000.0,
    "chord":                   1.0,      # normalised
    "chord_physical_mm":       420.0,
    "pressure_freestream":     101325.0, # Pa
    "temperature_freestream":  300.0,    # K  (approx, to match Re)
    "gamma":                   1.4,
    "gas_constant":            287.058,  # J/(kg·K)
    "prandtl_lam":             0.72,
    "prandtl_turb":            0.9,
    # Sutherland's Law
    "mu_ref":                  1.716e-5,
    "mu_t_ref":                273.11,
    "sutherland_constant":     110.33,
    # Turbulence inflow
    "mu_t_ratio":              3.0,
    "turbulence_intensity":    0.03,
}

# ============================================================================
# Grid Levels (no-plenum variant)
# ============================================================================

GRID_LEVELS = {
    "coarse": {
        "plot3d": "hump2newtop_noplenumZ103x28.p2dfmt.gz",
        "dims":   (103, 28),
        "cells":  2754,
    },
    "medium": {
        "plot3d": "hump2newtop_noplenumZ205x55.p2dfmt.gz",
        "dims":   (205, 55),
        "cells":  11016,
    },
    "fine": {
        "plot3d": "hump2newtop_noplenumZ409x109.p2dfmt.gz",
        "dims":   (409, 109),
        "cells":  44064,
    },
    "xfine": {
        "plot3d": "hump2newtop_noplenumZ817x217.p2dfmt.gz",
        "dims":   (817, 217),
        "cells":  176256,
    },
    "ultra": {
        "plot3d": "hump2newtop_noplenumZ1633x433.p2dfmt.gz",
        "dims":   (1633, 433),
        "cells":  705024,
    },
}

# SU2 turbulence model mapping
SU2_MODELS = {
    "SA": {
        "KIND_TURB_MODEL": "SA",
        "description": "Spalart-Allmaras (TMR primary, MRR Level 4)",
    },
    "SST": {
        "KIND_TURB_MODEL": "SST",
        "description": "Menter SST k-omega (MRR Level 3)",
    },
}


def sutherlands_law(T: float) -> float:
    """Compute dynamic viscosity via Sutherland's law."""
    mu_0 = CASE_CONFIG['mu_ref']
    T_0 = CASE_CONFIG['mu_t_ref']
    S = CASE_CONFIG['sutherland_constant']
    return mu_0 * (T / T_0) ** 1.5 * (T_0 + S) / (T + S)


# ============================================================================
# SU2 Config Generator
# ============================================================================

def generate_su2_config(case_dir: Path, mesh_file: str,
                         model: str = "SA", n_iter: int = 20000,
                         restart: bool = False) -> Path:
    """
    Generate SU2 configuration file for wall-mounted hump.

    Key differences from NACA 0012:
    - Internal flow with upper wall (inviscid/slip BC)
    - No farfield BC; uses inlet/outlet BCs
    - Low Mach number (M=0.1)
    - Separation focus (adverse pressure gradient)
    """
    cfg = CASE_CONFIG
    model_cfg = SU2_MODELS[model]

    T = cfg['temperature_freestream']
    mu = sutherlands_law(T)

    # Compute total (stagnation) conditions from isentropic relations
    M = cfg['mach']
    gamma = cfg['gamma']
    P_static = cfg['pressure_freestream']
    T_static = cfg['temperature_freestream']
    # P_total / P_static = (1 + (gamma-1)/2 * M^2)^(gamma/(gamma-1))
    P_total = P_static * (1 + (gamma - 1) / 2 * M**2) ** (gamma / (gamma - 1))
    T_total = T_static * (1 + (gamma - 1) / 2 * M**2)

    # Model-dependent config blocks
    if model == 'SA':
        turb_inflow_block = f"FREESTREAM_NU_FACTOR= {cfg['mu_t_ratio']}"
        turb_screen_vars = "RMS_NU_TILDE"
    elif model == 'SST':
        turb_inflow_block = (
            f"FREESTREAM_TURBULENCEINTENSITY= {cfg['turbulence_intensity']}\n"
            f"FREESTREAM_TURB2LAMVISCRATIO= {cfg['mu_t_ratio']}"
        )
        turb_screen_vars = "RMS_TKE, RMS_DISSIPATION"
    else:
        turb_inflow_block = f"FREESTREAM_NU_FACTOR= {cfg['mu_t_ratio']}"
        turb_screen_vars = "RMS_NU_TILDE"

    config_content = f"""\
% =============================================================================
% NASA Wall-Mounted Hump — TMR Validation Case (2DWMH)
% Turbulence model: {model} ({model_cfg['description']})
% M = {cfg['mach']}, Re_c = {cfg['reynolds']:.0f}
% Generated by run_wall_hump.py
%
% BCs: Inlet = subsonic, Outlet = back-pressure,
%       Lower wall = adiabatic no-slip, Upper wall = inviscid (slip)
% =============================================================================

% ------------- SOLVER --------------------------------------------------------
SOLVER= RANS
KIND_TURB_MODEL= {model_cfg['KIND_TURB_MODEL']}
MATH_PROBLEM= DIRECT
RESTART_SOL= {'YES' if restart else 'NO'}

% ------------- COMPRESSIBLE FREE-STREAM DEFINITION --------------------------
MACH_NUMBER= {cfg['mach']}
AOA= 0.0
SIDESLIP_ANGLE= 0.0
FREESTREAM_OPTION= TEMPERATURE_FS
FREESTREAM_PRESSURE= {cfg['pressure_freestream']}
FREESTREAM_TEMPERATURE= {cfg['temperature_freestream']}
REYNOLDS_NUMBER= {cfg['reynolds']:.1f}
REYNOLDS_LENGTH= {cfg['chord']}

% ------------- FLUID MODEL ---------------------------------------------------
GAMMA_VALUE= {cfg['gamma']}
GAS_CONSTANT= {cfg['gas_constant']}

% ------------- VISCOSITY — Sutherland's Law -----------------------------------
VISCOSITY_MODEL= SUTHERLAND
MU_REF= {cfg['mu_ref']}
MU_T_REF= {cfg['mu_t_ref']}
SUTHERLAND_CONSTANT= {cfg['sutherland_constant']}

% ------------- THERMAL CONDUCTIVITY ------------------------------------------
CONDUCTIVITY_MODEL= CONSTANT_PRANDTL
PRANDTL_LAM= {cfg['prandtl_lam']}
PRANDTL_TURB= {cfg['prandtl_turb']}

% ------------- TURBULENCE ----------------------------------------------------
{turb_inflow_block}

% ------------- REFERENCE VALUES -----------------------------------------------
REF_ORIGIN_MOMENT_X= 0.0
REF_ORIGIN_MOMENT_Y= 0.0
REF_ORIGIN_MOMENT_Z= 0.0
REF_LENGTH= {cfg['chord']}
REF_AREA= {cfg['chord']}
REF_DIMENSIONALIZATION= DIMENSIONAL

% ------------- BOUNDARY CONDITIONS -------------------------------------------
% Lower wall (hump + flat plate): Adiabatic no-slip
% Upper wall: Inviscid (Euler / slip wall)
% Inflow: Subsonic inlet (total conditions)
% Outflow: Back-pressure outlet
MARKER_HEATFLUX= ( lower_wall, 0.0 )
MARKER_EULER= ( upper_wall )
MARKER_INLET= ( inflow, {T_total:.3f}, {P_total:.2f}, 1.0, 0.0, 0.0 )
MARKER_OUTLET= ( outflow, {P_static} )
MARKER_PLOTTING= ( lower_wall )
MARKER_MONITORING= ( lower_wall )

% ------------- MULTIGRID -----------------------------------------------------
MGLEVEL= 2
MGCYCLE= V_CYCLE
MG_PRE_SMOOTH= ( 1, 2, 3 )
MG_POST_SMOOTH= ( 0, 0, 0 )
MG_CORRECTION_SMOOTH= ( 0, 0, 0 )
MG_DAMP_RESTRICTION= 0.5
MG_DAMP_PROLONGATION= 0.5

% ------------- NUMERICAL METHOD -----------------------------------------------
NUM_METHOD_GRAD= WEIGHTED_LEAST_SQUARES
CFL_NUMBER= 5.0
CFL_ADAPT= YES
CFL_ADAPT_PARAM= ( 0.5, 1.5, 5.0, 20.0 )

% ------------- LINEAR SOLVER --------------------------------------------------
LINEAR_SOLVER= FGMRES
LINEAR_SOLVER_PREC= LU_SGS
LINEAR_SOLVER_ERROR= 1e-10
LINEAR_SOLVER_ITER= 5

% ------------- FLOW NUMERICAL METHOD ------------------------------------------
CONV_NUM_METHOD_FLOW= ROE
MUSCL_FLOW= YES
SLOPE_LIMITER_FLOW= VENKATAKRISHNAN
VENKAT_LIMITER_COEFF= 0.03
TIME_DISCRE_FLOW= EULER_IMPLICIT

% ------------- TURBULENCE NUMERICAL METHOD ------------------------------------
CONV_NUM_METHOD_TURB= SCALAR_UPWIND
MUSCL_TURB= NO
TIME_DISCRE_TURB= EULER_IMPLICIT

% ------------- CONVERGENCE ----------------------------------------------------
ITER= {n_iter}
CONV_RESIDUAL_MINVAL= -14
CONV_STARTITER= 500
CONV_FIELD= RMS_DENSITY
CONV_CAUCHY_ELEMS= 200
CONV_CAUCHY_EPS= 1e-8

% ------------- INPUT/OUTPUT ---------------------------------------------------
MESH_FILENAME= {mesh_file}
MESH_FORMAT= SU2
SOLUTION_FILENAME= restart_flow.dat
RESTART_FILENAME= restart_flow.dat
CONV_FILENAME= history
VOLUME_FILENAME= flow
SURFACE_FILENAME= surface_flow
OUTPUT_WRT_FREQ= 500
HISTORY_WRT_FREQ_INNER= 1
OUTPUT_FILES= (RESTART, PARAVIEW, SURFACE_PARAVIEW, SURFACE_CSV)
SCREEN_OUTPUT= (INNER_ITER, RMS_DENSITY, RMS_ENERGY, LIFT, DRAG, {turb_screen_vars})
HISTORY_OUTPUT= (ITER, RMS_RES, AERO_COEFF, CAUCHY)
"""

    config_path = case_dir / "wall_hump.cfg"
    config_path.write_text(config_content)
    return config_path


# ============================================================================
# Grid Conversion
# ============================================================================

def convert_hump_grid(grid_level: str, grids_dir: Path) -> Optional[Path]:
    """Convert PLOT3D hump grid to SU2 format."""
    grid_cfg = GRID_LEVELS[grid_level]
    dims = grid_cfg["dims"]
    su2_name = f"wall_hump_{dims[0]}x{dims[1]}.su2"
    su2_path = grids_dir / su2_name

    if su2_path.exists():
        print(f"  [OK]     SU2 mesh already exists: {su2_name}")
        return su2_path

    plot3d_file = grids_dir / grid_cfg["plot3d"]
    if not plot3d_file.exists():
        print(f"  [WARN]   Grid file not found: {grid_cfg['plot3d']}")
        print(f"           Run: python experimental_data/wall_hump/wall_hump_tmr_data.py")
        return None

    try:
        from scripts.preprocessing.plot3d_to_su2 import read_plot3d_2d
    except ImportError:
        print(f"  [WARN]   plot3d_to_su2 module not found")
        print(f"           Grid conversion requires scripts/preprocessing/plot3d_to_su2.py")
        return None

    print(f"  [CONV]   Converting {grid_cfg['plot3d']} -> {su2_name}")
    x, y, idim, jdim = read_plot3d_2d(plot3d_file)

    # read_plot3d_2d returns shape (idim, jdim).
    # We need flat arrays indexed as idx = j*idim + i, i.e. x_flat[j*idim+i] = x[i,j].
    # Transpose to (jdim, idim) then flatten in C-order gives the right mapping.
    x_flat = x.T.ravel()  # shape (jdim*idim,)
    y_flat = y.T.ravel()

    # Hump grid boundary identification:
    # - j=0: lower wall (hump + flat plate)
    # - j=jmax: upper wall (contoured, inviscid)
    # - i=0: inflow
    # - i=imax: outflow
    n_pts = idim * jdim
    n_elems = (idim - 1) * (jdim - 1)

    with open(su2_path, 'w', encoding='utf-8') as f:
        f.write(f"% SU2 mesh - Wall-mounted hump ({idim}x{jdim})\n")
        f.write(f"% Converted from TMR PLOT3D grid\n")
        f.write(f"NDIME= 2\n")

        # Elements (quads)
        f.write(f"NELEM= {n_elems}\n")
        elem_id = 0
        for j in range(jdim - 1):
            for i in range(idim - 1):
                n0 = j * idim + i
                n1 = j * idim + (i + 1)
                n2 = (j + 1) * idim + (i + 1)
                n3 = (j + 1) * idim + i
                f.write(f"9 {n0} {n1} {n2} {n3} {elem_id}\n")
                elem_id += 1

        # Points
        f.write(f"NPOIN= {n_pts}\n")
        for idx in range(n_pts):
            f.write(f"{x_flat[idx]:.15e} {y_flat[idx]:.15e} {idx}\n")

        # Markers
        f.write(f"NMARK= 4\n")

        # Lower wall (j=0, all i)
        f.write(f"MARKER_TAG= lower_wall\n")
        f.write(f"MARKER_ELEMS= {idim - 1}\n")
        for i in range(idim - 1):
            n0 = i
            n1 = i + 1
            f.write(f"3 {n0} {n1}\n")

        # Upper wall (j=jmax)
        f.write(f"MARKER_TAG= upper_wall\n")
        f.write(f"MARKER_ELEMS= {idim - 1}\n")
        for i in range(idim - 1):
            n0 = (jdim - 1) * idim + i
            n1 = (jdim - 1) * idim + (i + 1)
            f.write(f"3 {n0} {n1}\n")

        # Inflow (i=0, all j)
        f.write(f"MARKER_TAG= inflow\n")
        f.write(f"MARKER_ELEMS= {jdim - 1}\n")
        for j in range(jdim - 1):
            n0 = j * idim
            n1 = (j + 1) * idim
            f.write(f"3 {n0} {n1}\n")

        # Outflow (i=imax)
        f.write(f"MARKER_TAG= outflow\n")
        f.write(f"MARKER_ELEMS= {jdim - 1}\n")
        for i_col in [idim - 1]:
            for j in range(jdim - 1):
                n0 = j * idim + i_col
                n1 = (j + 1) * idim + i_col
                f.write(f"3 {n0} {n1}\n")

    size_mb = su2_path.stat().st_size / (1024 * 1024)
    print(f"  [OK]     Written {su2_name} ({size_mb:.1f} MB)")
    print(f"           Nodes: {n_pts}, Elements: {n_elems}")
    print(f"           Markers: lower_wall, upper_wall, inflow, outflow")
    return su2_path


# ============================================================================
# Case Setup
# ============================================================================

def setup_case(model: str, grid: str, runs_dir: Path,
                grids_dir: Path, n_iter: int = 20000,
                restart: bool = False) -> Path:
    """Set up a single wall-mounted hump simulation case."""
    case_name = f"hump_{model}_{grid}"
    case_dir = runs_dir / case_name
    case_dir.mkdir(parents=True, exist_ok=True)

    # Convert grid
    su2_mesh = convert_hump_grid(grid, grids_dir)
    if su2_mesh is None:
        return case_dir

    # Copy mesh to case directory
    mesh_in_case = case_dir / "mesh.su2"
    if not mesh_in_case.exists():
        shutil.copy2(su2_mesh, mesh_in_case)

    # Generate SU2 config
    generate_su2_config(case_dir, "mesh.su2", model, n_iter=n_iter,
                         restart=restart)

    return case_dir


# ============================================================================
# Simulation Runner
# ============================================================================

def check_solver():
    """Check if SU2_CFD is available."""
    result = shutil.which("SU2_CFD") or shutil.which("SU2_CFD.exe")
    if result:
        print(f"  [OK]     SU2_CFD found: {result}")
        return True
    else:
        print(f"  [!!]     SU2_CFD not found on PATH")
        return False


def run_simulation(case_dir: Path, config_file: Path,
                    n_procs: int = 1, n_threads: int = 1,
                    timeout: int = 14400) -> Dict:
    """Run SU2_CFD for a single hump case."""
    result = {
        "case_dir": str(case_dir),
        "config": str(config_file),
        "converged": False,
        "iterations": 0,
        "wall_time_s": 0.0,
        "CL": None,
        "CD": None,
        "final_rms_density": None,
        "error": None,
    }

    su2_exe = shutil.which("SU2_CFD") or shutil.which("SU2_CFD.exe") or "SU2_CFD"
    cmd = [su2_exe]
    if n_threads > 1:
        cmd.extend(["-t", str(n_threads)])
    cmd.append(str(config_file.name))

    if n_procs > 1:
        mpi_cmd = shutil.which("mpiexec") or shutil.which("mpirun")
        if mpi_cmd:
            cmd = [mpi_cmd, "-np", str(n_procs)] + cmd

    log_file = case_dir / "su2_log.txt"

    try:
        print(f"  [RUN]    {' '.join(cmd[:6])}")
        start = time.time()
        with open(log_file, 'w') as log:
            proc = subprocess.run(
                cmd, cwd=str(case_dir), stdout=log, stderr=subprocess.STDOUT,
                timeout=timeout
            )
        result["wall_time_s"] = time.time() - start

        if proc.returncode != 0:
            # Capture last lines of log for diagnostics
            err_detail = ""
            if log_file.exists():
                lines = log_file.read_text(errors='replace').splitlines()
                tail = lines[-20:] if len(lines) > 20 else lines
                err_detail = "\n".join(tail)
            result["error"] = (f"SU2_CFD exited with code {proc.returncode}\n"
                               f"--- Last 20 lines of log ---\n{err_detail}")
            print(f"  [FAIL]   Return code {proc.returncode}")
            for line in err_detail.splitlines()[-5:]:
                print(f"           {line}")
            return result

        # Parse history and validate
        history = parse_su2_history(case_dir)
        if history:
            result["CL"] = history.get("CL")
            result["CD"] = history.get("CD")
            result["iterations"] = history.get("iterations", 0)

            # Validate: check history.csv has reasonable iteration count
            hist_file = case_dir / "history.csv"
            if hist_file.exists():
                n_lines = sum(1 for _ in open(hist_file, errors='replace')) - 1
                if n_lines < 10:
                    result["error"] = (f"Only {n_lines} iterations in history.csv "
                                       f"(expected ~{result['iterations']}). "
                                       f"Possible early termination.")
                    print(f"  [WARN]   Only {n_lines} history entries!")
                else:
                    print(f"  [OK]     {n_lines} iterations logged")

                # Extract final residual
                try:
                    last_line = hist_file.read_text(errors='replace').strip().split('\n')[-1]
                    cols = last_line.split(',')
                    if len(cols) > 3:
                        result["final_rms_density"] = float(cols[3].strip().strip('"'))
                except (ValueError, IndexError):
                    pass

            result["converged"] = True
            if result["final_rms_density"] is not None:
                rms = result["final_rms_density"]
                print(f"  [CONV]   Final rms[Rho] = {rms:.4f} "
                      f"({'GOOD' if rms < -6 else 'FAIR' if rms < -4 else 'POOR'})")

    except subprocess.TimeoutExpired:
        result["error"] = f"Timeout after {timeout}s"
    except FileNotFoundError:
        result["error"] = "SU2_CFD not found"
    except Exception as e:
        result["error"] = str(e)

    return result


def parse_su2_history(case_dir: Path) -> Optional[Dict]:
    """Parse SU2 history.csv for final CL, CD values."""
    for name in ["history.csv", "history.dat"]:
        history_file = case_dir / name
        if history_file.exists():
            break
    else:
        return None

    try:
        csv_mod.field_size_limit(10 * 1024 * 1024)
        with open(history_file, 'r') as f:
            reader = csv_mod.reader(f)
            headers = None
            last_row = None
            for row in reader:
                if not row:
                    continue
                # Detect header row: first field is non-numeric
                # (csv.reader strips quotes, so row[0] is e.g. 'Time_Iter')
                first = row[0].strip().strip('"')
                try:
                    float(first)
                except ValueError:
                    headers = [h.strip().strip('"') for h in row]
                    continue
                last_row = row

            if headers and last_row:
                data = {}
                for h, v in zip(headers, last_row):
                    try:
                        data[h] = float(v.strip().strip('"'))
                    except (ValueError, IndexError):
                        pass
                return {
                    "CL": data.get("CL", data.get("Lift", None)),
                    "CD": data.get("CD", data.get("Drag", None)),
                    "iterations": int(data.get("Inner_Iter",
                                      data.get("Iteration", 0))),
                }
    except Exception as e:
        print(f"  [WARN]   Could not parse history: {e}")
    return None


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="NASA Wall-Mounted Hump — TMR Validation Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python run_wall_hump.py --dry-run                    Setup only
  python run_wall_hump.py --model SA -t 7              Run with OpenMP
  python run_wall_hump.py --model SA SST --grid fine   Multiple models
  python run_wall_hump.py --validate-only              Post-process only
"""
    )
    parser.add_argument("--model", nargs="+", default=["SA"],
                        choices=list(SU2_MODELS.keys()),
                        help="Turbulence models (default: SA)")
    parser.add_argument("--grid", default="medium",
                        choices=list(GRID_LEVELS.keys()),
                        help="Grid level (default: medium)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Setup cases without running solver")
    parser.add_argument("--validate-only", action="store_true",
                        help="Only validate existing results")
    parser.add_argument("--n-procs", type=int, default=1,
                        help="Number of MPI processes")
    parser.add_argument("--n-threads", "-t", type=int, default=1,
                        help="OpenMP threads (SU2 v8+)")
    parser.add_argument("--n-iter", type=int, default=20000,
                        help="Max iterations (default: 20000)")
    parser.add_argument("--timeout", type=int, default=14400,
                        help="Timeout in seconds (default: 14400)")
    parser.add_argument("--restart", action="store_true",
                        help="Restart from existing solution")
    parser.add_argument("--runs-dir", type=Path, default=None,
                        help="Output directory for runs")
    args = parser.parse_args()

    print("=" * 60)
    print("  NASA WALL-MOUNTED HUMP — TMR VALIDATION (2DWMH)")
    print("  Turbulence Modeling Resource")
    print("=" * 60)
    print(f"\n  Mach = {CASE_CONFIG['mach']}, Re = {CASE_CONFIG['reynolds']:.0f}")
    dims = GRID_LEVELS[args.grid]['dims']
    print(f"  Grid: {args.grid} ({dims[0]}×{dims[1]})")
    print(f"  Models: {', '.join(args.model)}")

    runs_dir = args.runs_dir or PROJECT_ROOT / "runs" / "wall_hump"
    runs_dir.mkdir(parents=True, exist_ok=True)
    grids_dir = PROJECT_ROOT / "experimental_data" / "wall_hump" / "grids"
    data_dir = PROJECT_ROOT / "experimental_data" / "wall_hump"

    # --- Step 1: Download data ---
    if not (data_dir / "csv" / "hump_case_reference.json").exists():
        print("\n  [STEP 1] Downloading TMR data...")
        sys.path.insert(0, str(data_dir))
        from experimental_data.wall_hump.wall_hump_tmr_data import download_all
        download_all()
    else:
        print("\n  [STEP 1] TMR data already downloaded [OK]")

    # --- Step 2: Validate only? ---
    if args.validate_only:
        print("\n  [VALIDATE] Post-processing existing results...")
        return

    # --- Step 3: Setup cases ---
    print(f"\n  [STEP 2] Setting up {len(args.model)} cases...")
    case_dirs = {}
    for model in args.model:
        case_dir = setup_case(model, args.grid, runs_dir, grids_dir,
                               n_iter=args.n_iter, restart=args.restart)
        case_dirs[model] = case_dir
        print(f"  [OK]     {case_dir.name}")

    if args.dry_run:
        print("\n  [DRY-RUN] Cases set up. No solver executed.")
        print(f"\n  Case directories created in: {runs_dir}")
        for model, case_dir in case_dirs.items():
            print(f"\n  To run {model}:")
            print(f"    cd {case_dir}")
            if args.n_threads > 1:
                print(f"    SU2_CFD -t {args.n_threads} wall_hump.cfg")
            else:
                print(f"    SU2_CFD wall_hump.cfg")
        return

    # --- Step 4: Run simulations ---
    if not check_solver():
        print("\n  [!!]     Cannot run without SU2. Use --dry-run.")
        return

    print(f"\n  [STEP 3] Running simulations...")
    all_results = {}
    for model, case_dir in case_dirs.items():
        config = case_dir / "wall_hump.cfg"
        print(f"\n  --- model={model} ---")
        result = run_simulation(
            case_dir, config,
            n_procs=args.n_procs, n_threads=args.n_threads,
            timeout=args.timeout
        )
        all_results[model] = result
        if result["error"]:
            print(f"  [FAIL]   {result['error']}")
        elif result.get("CD") is not None:
            print(f"  [OK]     CD={result['CD']:.5f}, "
                  f"time={result['wall_time_s']:.1f}s")

    # Save results summary
    summary_file = runs_dir / "results_summary.json"
    summary = {
        "case": "NASA Wall-Mounted Hump (2DWMH)",
        "grid": args.grid,
        "grid_dims": GRID_LEVELS[args.grid]["dims"],
        "conditions": CASE_CONFIG,
    }
    for model, result in all_results.items():
        summary[f"model_{model}"] = {
            "CD": result.get("CD"),
            "converged": result.get("converged", False),
            "wall_time_s": result.get("wall_time_s", 0),
        }
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Results saved: {summary_file}")


if __name__ == "__main__":
    main()
