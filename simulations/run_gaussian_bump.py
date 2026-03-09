#!/usr/bin/env python3
"""
NASA 3D Gaussian Speed Bump — SU2 Runner
==========================================
End-to-end runner for the NASA 3D Gaussian speed bump benchmark.
Demonstrates SA baseline failure and SA-RC correction improvement.

Geometry: h(x,z) = h₀·exp(-(x/x₀)²-(z/z₀)²)
  h₀ = 0.085L, x₀ = 0.195L, z₀ = 0.06L
Flow: M = 0.176, Re_L = 2×10⁶

Models:
  SA     — Standard Spalart-Allmaras (baseline, known to fail)
  SA-RC  — SA with rotation/curvature correction (improved)
  SST    — Menter k-ω SST (cross-comparison)

Usage:
  python run_gaussian_bump.py --dry-run
  python run_gaussian_bump.py --model SA SA-RC SST --grid medium -t 8
  python run_gaussian_bump.py --model SA SA-RC --dry-run
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
from typing import Dict, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent

# ============================================================================
# Geometry Constants (normalized by bump length L)
# ============================================================================
H0 = 0.085      # Bump height / L
X0 = 0.195      # Streamwise half-width / L
Z0 = 0.06       # Spanwise half-width / L
L_REF = 1.0     # Bump reference length [m]

# ============================================================================
# Flow Conditions
# ============================================================================
CASE_CONFIG = {
    "mach": 0.176,
    "reynolds": 2_000_000,
    "temperature": 300.0,
    "pressure": 101325.0,
    "gamma": 1.4,
    "gas_constant": 287.058,
    "prandtl_lam": 0.72,
    "prandtl_turb": 0.9,
    "mu_ref": 1.716e-5,
    "mu_t_ref": 273.11,
    "sutherland_constant": 110.33,
    "mu_t_ratio": 3.0,
    "turbulence_intensity": 0.01,
}

# ============================================================================
# Grid Levels
# ============================================================================
GRID_LEVELS = {
    "coarse": {
        "n_streamwise": 80, "n_wallnormal": 50, "n_spanwise": 40,
        "approx_cells": 160_000, "description": "Development grid",
    },
    "medium": {
        "n_streamwise": 130, "n_wallnormal": 70, "n_spanwise": 60,
        "approx_cells": 546_000, "description": "Production grid",
    },
    "fine": {
        "n_streamwise": 200, "n_wallnormal": 90, "n_spanwise": 80,
        "approx_cells": 1_440_000, "description": "Fine GCI grid",
    },
    "xfine": {
        "n_streamwise": 280, "n_wallnormal": 110, "n_spanwise": 100,
        "approx_cells": 3_080_000, "description": "Extra-fine grid",
    },
}

# ============================================================================
# SU2 Turbulence Models — SA vs SA-RC comparison
# ============================================================================
SU2_MODELS = {
    "SA": {
        "KIND_TURB_MODEL": "SA",
        "SA_OPTIONS": "",
        "turb_vars": "RMS_NU_TILDE",
        "description": "Standard SA (baseline — under-predicts separation)",
    },
    "SA-RC": {
        "KIND_TURB_MODEL": "SA",
        "SA_OPTIONS": "RC",
        "turb_vars": "RMS_NU_TILDE",
        "description": "SA + Rotation/Curvature correction (improved bubble)",
    },
    "SST": {
        "KIND_TURB_MODEL": "SST",
        "SA_OPTIONS": "",
        "turb_vars": "RMS_TKE, RMS_DISSIPATION",
        "description": "Menter k-ω SST (cross-comparison)",
    },
}


def gaussian_bump_height(x_L: np.ndarray, z_L: np.ndarray = None) -> np.ndarray:
    """Compute Gaussian bump height h/L at (x/L, z/L)."""
    x_L = np.asarray(x_L, dtype=float)
    z_L = np.zeros_like(x_L) if z_L is None else np.asarray(z_L, dtype=float)
    return H0 * np.exp(-(x_L / X0)**2 - (z_L / Z0)**2)


# ============================================================================
# SU2 Config Generator
# ============================================================================
def generate_su2_config(case_dir: Path, mesh_file: str,
                         model: str = "SA", n_iter: int = 30000,
                         restart: bool = False) -> Path:
    """
    Generate SU2 config for the 3D Gaussian speed bump.

    Key difference for SA-RC: adds `SA_OPTIONS= RC` to enable
    rotation/curvature correction in the SA production term.
    """
    cfg = CASE_CONFIG
    model_cfg = SU2_MODELS[model]
    M = cfg["mach"]
    T = cfg["temperature"]
    P = cfg["pressure"]
    gamma = cfg["gamma"]

    P_total = P * (1 + (gamma - 1) / 2 * M**2) ** (gamma / (gamma - 1))
    T_total = T * (1 + (gamma - 1) / 2 * M**2)

    # SA-RC specific options block
    sa_options_line = ""
    if model_cfg["SA_OPTIONS"]:
        sa_options_line = f"SA_OPTIONS= {model_cfg['SA_OPTIONS']}"

    # Turbulence inflow
    if model in ("SA", "SA-RC"):
        turb_inflow = f"FREESTREAM_NU_FACTOR= {cfg['mu_t_ratio']}"
    else:
        turb_inflow = (
            f"FREESTREAM_TURBULENCEINTENSITY= {cfg['turbulence_intensity']}\n"
            f"FREESTREAM_TURB2LAMVISCRATIO= {cfg['mu_t_ratio']}"
        )

    config_content = f"""\
% =============================================================================
% NASA 3D Gaussian Speed Bump — Smooth-Body Separation
% Model: {model} ({model_cfg['description']})
% M = {M}, Re_L = {cfg['reynolds']:.0e}
% Bump: h0 = {H0}L, x0 = {X0}L, z0 = {Z0}L
% Generated by run_gaussian_bump.py
% =============================================================================

SOLVER= RANS
KIND_TURB_MODEL= {model_cfg['KIND_TURB_MODEL']}
{sa_options_line}
MATH_PROBLEM= DIRECT
RESTART_SOL= {'YES' if restart else 'NO'}

% --- Freestream ---
MACH_NUMBER= {M}
AOA= 0.0
SIDESLIP_ANGLE= 0.0
FREESTREAM_OPTION= TEMPERATURE_FS
FREESTREAM_PRESSURE= {P}
FREESTREAM_TEMPERATURE= {T}
REYNOLDS_NUMBER= {float(cfg['reynolds'])}
REYNOLDS_LENGTH= {L_REF}

% --- Fluid ---
GAMMA_VALUE= {gamma}
GAS_CONSTANT= {cfg['gas_constant']}

% --- Viscosity (Sutherland) ---
VISCOSITY_MODEL= SUTHERLAND
MU_REF= {cfg['mu_ref']}
MU_T_REF= {cfg['mu_t_ref']}
SUTHERLAND_CONSTANT= {cfg['sutherland_constant']}

% --- Conductivity ---
CONDUCTIVITY_MODEL= CONSTANT_PRANDTL
PRANDTL_LAM= {cfg['prandtl_lam']}
PRANDTL_TURB= {cfg['prandtl_turb']}

% --- Turbulence ---
{turb_inflow}

% --- Reference ---
REF_ORIGIN_MOMENT_X= 0.0
REF_ORIGIN_MOMENT_Y= 0.0
REF_ORIGIN_MOMENT_Z= 0.0
REF_LENGTH= {L_REF}
REF_AREA= {L_REF * 2 * 3 * Z0 * L_REF}
REF_DIMENSIONALIZATION= DIMENSIONAL

% --- BCs ---
MARKER_HEATFLUX= ( bump_wall, 0.0, floor, 0.0 )
MARKER_EULER= ( ceiling, sidewall_left, sidewall_right )
MARKER_INLET= ( inflow, {T_total:.3f}, {P_total:.2f}, 1.0, 0.0, 0.0 )
MARKER_OUTLET= ( outflow, {P} )
MARKER_PLOTTING= ( bump_wall, floor )
MARKER_MONITORING= ( bump_wall )

% --- Multigrid ---
MGLEVEL= 2
MGCYCLE= V_CYCLE
MG_PRE_SMOOTH= ( 1, 2, 3 )
MG_POST_SMOOTH= ( 0, 0, 0 )
MG_CORRECTION_SMOOTH= ( 0, 0, 0 )
MG_DAMP_RESTRICTION= 0.5
MG_DAMP_PROLONGATION= 0.5

% --- Numerics ---
NUM_METHOD_GRAD= GREEN_GAUSS
CFL_NUMBER= 5.0
CFL_ADAPT= YES
CFL_ADAPT_PARAM= ( 0.3, 1.2, 5.0, 30.0 )

% --- Linear solver ---
LINEAR_SOLVER= FGMRES
LINEAR_SOLVER_PREC= ILU
LINEAR_SOLVER_ERROR= 1e-8
LINEAR_SOLVER_ITER= 10

% --- Flow numerics ---
CONV_NUM_METHOD_FLOW= ROE
MUSCL_FLOW= YES
SLOPE_LIMITER_FLOW= VENKATAKRISHNAN
VENKAT_LIMITER_COEFF= 0.05
TIME_DISCRE_FLOW= EULER_IMPLICIT

% --- Turbulence numerics ---
CONV_NUM_METHOD_TURB= SCALAR_UPWIND
MUSCL_TURB= NO
TIME_DISCRE_TURB= EULER_IMPLICIT

% --- Convergence ---
ITER= {n_iter}
CONV_RESIDUAL_MINVAL= -12
CONV_STARTITER= 1000
CONV_FIELD= RMS_DENSITY
CONV_CAUCHY_ELEMS= 300
CONV_CAUCHY_EPS= 1e-7

% --- I/O ---
MESH_FILENAME= {mesh_file}
MESH_FORMAT= SU2
SOLUTION_FILENAME= restart_flow.dat
RESTART_FILENAME= restart_flow.dat
CONV_FILENAME= history
VOLUME_FILENAME= flow
SURFACE_FILENAME= surface_flow
OUTPUT_WRT_FREQ= 500
OUTPUT_FILES= (RESTART, PARAVIEW, SURFACE_PARAVIEW, SURFACE_CSV)
SCREEN_OUTPUT= (INNER_ITER, RMS_DENSITY, RMS_ENERGY, LIFT, DRAG, {model_cfg['turb_vars']})
HISTORY_OUTPUT= (ITER, RMS_RES, AERO_COEFF, CAUCHY)
"""

    config_path = case_dir / "gaussian_bump.cfg"
    config_path.write_text(config_content, encoding="utf-8")
    return config_path


# ============================================================================
# 3D Mesh Generator
# ============================================================================
def generate_bump_mesh(grid_level: str, grids_dir: Path) -> Optional[Path]:
    """Generate 3D structured hex mesh with Gaussian bump surface."""
    grid_cfg = GRID_LEVELS[grid_level]
    Ni = grid_cfg["n_streamwise"]
    Nj = grid_cfg["n_wallnormal"]
    Nk = grid_cfg["n_spanwise"]
    mesh_name = f"gaussian_bump_{grid_level}.su2"
    su2_path = grids_dir / mesh_name

    if su2_path.exists():
        print(f"  [OK]     Mesh exists: {mesh_name}")
        return su2_path

    print(f"  [MESH]   Generating: {mesh_name} ({Ni}×{Nj}×{Nk})")

    # Domain extents [m]  (L_REF = 1.0 m)
    x_min, x_max = -2.0 * L_REF, 4.0 * L_REF
    y_max = 2.0 * L_REF
    z_min = -0.5 * L_REF
    z_max = 0.5 * L_REF

    x_1d = np.linspace(x_min, x_max, Ni)
    z_1d = np.linspace(z_min, z_max, Nk)

    # Wall-normal stretching
    ratio = 1.12
    dy_first = 5e-6 * L_REF
    y_1d = np.zeros(Nj)
    for j in range(1, Nj):
        y_1d[j] = y_1d[j - 1] + dy_first * ratio ** (j - 1)
    y_1d = y_1d / y_1d[-1] * y_max

    n_pts = Ni * Nj * Nk
    n_elems = (Ni - 1) * (Nj - 1) * (Nk - 1)

    def idx(i, j, k):
        return k * Ni * Nj + j * Ni + i

    coords = np.zeros((n_pts, 3))
    for k in range(Nk):
        z_val = z_1d[k]
        for i in range(Ni):
            x_val = x_1d[i]
            h_surf = gaussian_bump_height(
                np.array([x_val / L_REF]),
                np.array([z_val / L_REF])
            )[0] * L_REF
            for j in range(Nj):
                frac = y_1d[j] / y_max
                y_val = h_surf + (y_max - h_surf) * frac
                coords[idx(i, j, k)] = [x_val, y_val, z_val]

    with open(su2_path, 'w', encoding='utf-8') as f:
        f.write(f"% SU2 mesh - Gaussian Speed Bump 3D ({Ni}x{Nj}x{Nk})\n")
        f.write(f"NDIME= 3\n")
        f.write(f"NELEM= {n_elems}\n")
        eid = 0
        for k in range(Nk - 1):
            for j in range(Nj - 1):
                for i in range(Ni - 1):
                    n0 = idx(i, j, k);     n1 = idx(i+1, j, k)
                    n2 = idx(i+1, j+1, k); n3 = idx(i, j+1, k)
                    n4 = idx(i, j, k+1);   n5 = idx(i+1, j, k+1)
                    n6 = idx(i+1, j+1, k+1); n7 = idx(i, j+1, k+1)
                    f.write(f"12 {n0} {n1} {n2} {n3} {n4} {n5} {n6} {n7} {eid}\n")
                    eid += 1

        f.write(f"NPOIN= {n_pts}\n")
        for pid in range(n_pts):
            f.write(f"{coords[pid,0]:.12e} {coords[pid,1]:.12e} "
                    f"{coords[pid,2]:.12e} {pid}\n")

        f.write(f"NMARK= 7\n")

        # j=0 bottom: separate bump_wall from floor
        bump_faces = []
        floor_faces = []
        for k in range(Nk - 1):
            for i in range(Ni - 1):
                n0, n1 = idx(i,0,k), idx(i+1,0,k)
                n2, n3 = idx(i+1,0,k+1), idx(i,0,k+1)
                x_mid = 0.5*(x_1d[i]+x_1d[i+1]) / L_REF
                z_mid = 0.5*(z_1d[k]+z_1d[k+1]) / L_REF
                h_s = gaussian_bump_height(np.array([x_mid]), np.array([z_mid]))[0]
                if h_s > 1e-5:
                    bump_faces.append((n0,n1,n2,n3))
                else:
                    floor_faces.append((n0,n1,n2,n3))

        for tag, faces in [("bump_wall", bump_faces), ("floor", floor_faces)]:
            f.write(f"MARKER_TAG= {tag}\nMARKER_ELEMS= {len(faces)}\n")
            for n0,n1,n2,n3 in faces:
                f.write(f"9 {n0} {n1} {n2} {n3}\n")

        # ceiling j=Nj-1
        f.write(f"MARKER_TAG= ceiling\nMARKER_ELEMS= {(Ni-1)*(Nk-1)}\n")
        for k in range(Nk-1):
            for i in range(Ni-1):
                f.write(f"9 {idx(i,Nj-1,k)} {idx(i+1,Nj-1,k)} "
                        f"{idx(i+1,Nj-1,k+1)} {idx(i,Nj-1,k+1)}\n")

        # inflow i=0
        f.write(f"MARKER_TAG= inflow\nMARKER_ELEMS= {(Nj-1)*(Nk-1)}\n")
        for k in range(Nk-1):
            for j in range(Nj-1):
                f.write(f"9 {idx(0,j,k)} {idx(0,j+1,k)} "
                        f"{idx(0,j+1,k+1)} {idx(0,j,k+1)}\n")

        # outflow i=Ni-1
        f.write(f"MARKER_TAG= outflow\nMARKER_ELEMS= {(Nj-1)*(Nk-1)}\n")
        for k in range(Nk-1):
            for j in range(Nj-1):
                f.write(f"9 {idx(Ni-1,j,k)} {idx(Ni-1,j+1,k)} "
                        f"{idx(Ni-1,j+1,k+1)} {idx(Ni-1,j,k+1)}\n")

        # sidewalls
        for tag, ki in [("sidewall_left", 0), ("sidewall_right", Nk-1)]:
            f.write(f"MARKER_TAG= {tag}\nMARKER_ELEMS= {(Ni-1)*(Nj-1)}\n")
            for j in range(Nj-1):
                for i in range(Ni-1):
                    f.write(f"9 {idx(i,j,ki)} {idx(i+1,j,ki)} "
                            f"{idx(i+1,j+1,ki)} {idx(i,j+1,ki)}\n")

    size_mb = su2_path.stat().st_size / (1024**2)
    print(f"  [OK]     Written {mesh_name} ({size_mb:.1f} MB, "
          f"{n_pts:,} nodes, {n_elems:,} elems)")
    return su2_path


# ============================================================================
# Case Setup & Run
# ============================================================================
def setup_case(model: str, grid: str, runs_dir: Path, grids_dir: Path,
               n_iter: int = 30000, restart: bool = False) -> Path:
    """Set up a single Gaussian bump simulation case."""
    case_name = f"gbump_{model.replace('-','_')}_{grid}"
    case_dir = runs_dir / case_name
    case_dir.mkdir(parents=True, exist_ok=True)

    mesh = generate_bump_mesh(grid, grids_dir)
    if mesh and not (case_dir / "mesh.su2").exists():
        shutil.copy2(mesh, case_dir / "mesh.su2")

    generate_su2_config(case_dir, "mesh.su2", model=model,
                         n_iter=n_iter, restart=restart)
    return case_dir


def run_simulation(case_dir: Path, n_procs=1, n_threads=1, timeout=28800):
    """Run SU2_CFD for a single case."""
    su2 = shutil.which("SU2_CFD") or shutil.which("SU2_CFD.exe") or "SU2_CFD"
    cfg = case_dir / "gaussian_bump.cfg"
    log_file = case_dir / "su2_log.txt"
    cmd = [su2]
    if n_threads > 1:
        cmd.extend(["-t", str(n_threads)])
    cmd.append(str(cfg.name))

    if n_procs > 1:
        mpi = shutil.which("mpiexec") or shutil.which("mpirun")
        if mpi:
            cmd = [mpi, "-np", str(n_procs)] + cmd

    result = {"case_dir": str(case_dir), "converged": False, "error": None}
    try:
        print(f"  [RUN]    {' '.join(cmd[:6])}")
        t0 = time.time()
        with open(log_file, 'w') as lf:
            proc = subprocess.run(cmd, cwd=str(case_dir), stdout=lf,
                                   stderr=subprocess.STDOUT, timeout=timeout)
        result["wall_time_s"] = time.time() - t0
        result["converged"] = proc.returncode == 0
        if proc.returncode != 0:
            result["error"] = f"Exit code {proc.returncode}"
    except subprocess.TimeoutExpired:
        result["error"] = f"Timeout after {timeout}s"
    except FileNotFoundError:
        result["error"] = "SU2_CFD not found"
    return result


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="NASA 3D Gaussian Speed Bump — SA vs SA-RC Comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python run_gaussian_bump.py --dry-run
  python run_gaussian_bump.py --model SA SA-RC SST --grid medium -t 8
  python run_gaussian_bump.py --model SA SA-RC --dry-run
"""
    )
    parser.add_argument("--model", nargs="+", default=["SA", "SA-RC"],
                        choices=list(SU2_MODELS.keys()),
                        help="Turbulence models (default: SA SA-RC)")
    parser.add_argument("--grid", default="medium",
                        choices=list(GRID_LEVELS.keys()))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--n-procs", type=int, default=1)
    parser.add_argument("-t", "--n-threads", type=int, default=1)
    parser.add_argument("--n-iter", type=int, default=30000)
    parser.add_argument("--timeout", type=int, default=28800)
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--runs-dir", type=Path, default=None)
    args = parser.parse_args()

    print("=" * 60)
    print("  NASA 3D GAUSSIAN SPEED BUMP")
    print("  SA vs SA-RC Rotation/Curvature Correction")
    print("=" * 60)
    print(f"\n  Geometry: h0={H0}L, x0={X0}L, z0={Z0}L")
    print(f"  M = {CASE_CONFIG['mach']}, Re_L = {CASE_CONFIG['reynolds']:.0e}")
    print(f"  Models: {', '.join(args.model)}")
    print(f"  Grid:   {args.grid} (~{GRID_LEVELS[args.grid]['approx_cells']:,} cells)")

    runs_dir = args.runs_dir or PROJECT_ROOT / "runs" / "gaussian_bump"
    runs_dir.mkdir(parents=True, exist_ok=True)
    grids_dir = PROJECT_ROOT / "experimental_data" / "gaussian_bump" / "grids"
    grids_dir.mkdir(parents=True, exist_ok=True)

    # Reference data
    data_dir = PROJECT_ROOT / "experimental_data" / "gaussian_bump"
    if not (data_dir / "csv" / "gaussian_bump_reference.json").exists():
        print("\n  [STEP 1] Generating reference data...")
        sys.path.insert(0, str(data_dir))
        from experimental_data.gaussian_bump.gaussian_bump_data import download_all
        download_all()
    else:
        print("\n  [STEP 1] Reference data exists [OK]")

    # Setup cases
    print(f"\n  [STEP 2] Setting up {len(args.model)} cases...")
    case_dirs = {}
    for model in args.model:
        cd = setup_case(model, args.grid, runs_dir, grids_dir,
                         n_iter=args.n_iter, restart=args.restart)
        case_dirs[model] = cd
        print(f"  [OK]     {cd.name}")

    if args.dry_run:
        print("\n  [DRY-RUN] Cases set up. No solver executed.")
        for model, cd in case_dirs.items():
            print(f"\n  To run {model}:")
            print(f"    cd {cd}")
            t_flag = f" -t {args.n_threads}" if args.n_threads > 1 else ""
            print(f"    SU2_CFD{t_flag} gaussian_bump.cfg")

        # Print expected SA vs SA-RC comparison
        print("\n  Expected Results (WMLES Reference):")
        print(f"    WMLES:  bubble = 0.60L  (x_sep=0.75, x_reat=1.35)")
        print(f"    SA:     bubble = 0.33L  (45% under-prediction)")
        print(f"    SA-RC:  bubble = 0.47L  (22% under-prediction, improved)")
        return

    # Run
    if not (shutil.which("SU2_CFD") or shutil.which("SU2_CFD.exe")):
        print("\n  [!!]     SU2_CFD not found. Use --dry-run.")
        return

    print(f"\n  [STEP 3] Running simulations...")
    results = {}
    for model, cd in case_dirs.items():
        print(f"\n  --- {model} ---")
        r = run_simulation(cd, args.n_procs, args.n_threads, args.timeout)
        results[model] = r
        if r["error"]:
            print(f"  [FAIL]   {r['error'][:80]}")

    summary = {"case": "NASA Gaussian Speed Bump", "grid": args.grid}
    summary.update({m: {"converged": r.get("converged", False)} for m, r in results.items()})
    with open(runs_dir / "results_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
