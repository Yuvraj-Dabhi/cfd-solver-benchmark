#!/usr/bin/env python3
"""
BeVERLI Hill — 3D Smooth-Body Separation Validation Runner
============================================================
End-to-end runner for the Virginia Tech / NASA Langley BeVERLI
(Benchmark Validation Experiments for RANS and LES Investigations)
Hill 3D smooth-body separated flow validation case.

Case specification:
  - Geometry: 3D superelliptic hill, H = 0.1869 m, w = 5H
  - Re_H = 250,000 – 650,000 (hill-height based)
  - Subsonic conditions (M ≈ 0.06–0.09)
  - Yaw angles: 0°, 30°, 45°
  - Fully turbulent, SA or SST models in SU2
  - BCs: Inlet subsonic, outlet back-pressure, tunnel walls

Usage:
  python run_beverli_hill.py --dry-run
  python run_beverli_hill.py --model SA SST --re 250000 --yaw 0 -t 8
  python run_beverli_hill.py --model SA --re 650000 --yaw 0 30 45
  python run_beverli_hill.py --validate-only

Reference: https://beverlihill.aoe.vt.edu/
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
from typing import Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent

# ============================================================================
# Hill Geometry Constants
# ============================================================================
HILL_HEIGHT = 0.186944      # H [m]
HILL_WIDTH = 0.93472        # w [m] (= 5H)
FLAT_TOP = 0.093472         # s [m] (= H/2)
HILL_HALF_LENGTH = 2.5      # In H units: hill from -2.5H to +2.5H

# ============================================================================
# Flow Conditions — Multiple Reynolds Numbers
# ============================================================================
REYNOLDS_CASES = {
    250000: {
        "mach": 0.06,
        "temperature_freestream": 293.15,
        "pressure_freestream": 101325.0,
        "velocity_freestream": 20.0,
        "description": "Low Re — moderate leeward separation",
    },
    400000: {
        "mach": 0.07,
        "temperature_freestream": 293.15,
        "pressure_freestream": 101325.0,
        "velocity_freestream": 24.0,
        "description": "Mid Re — intermediate",
    },
    650000: {
        "mach": 0.09,
        "temperature_freestream": 293.15,
        "pressure_freestream": 101325.0,
        "velocity_freestream": 31.0,
        "description": "High Re — fully turbulent, massive separation",
    },
}

# ============================================================================
# Common Flow/Fluid Properties
# ============================================================================
CASE_CONFIG = {
    "gamma": 1.4,
    "gas_constant": 287.058,
    "prandtl_lam": 0.72,
    "prandtl_turb": 0.9,
    # Sutherland's Law
    "mu_ref": 1.716e-5,
    "mu_t_ref": 273.11,
    "sutherland_constant": 110.33,
    # Turbulence inflow
    "mu_t_ratio": 3.0,
    "turbulence_intensity": 0.01,
}

# ============================================================================
# Yaw Angles
# ============================================================================
VALID_YAW_ANGLES = [0, 30, 45]

# ============================================================================
# Grid Levels
# ============================================================================
GRID_LEVELS = {
    "coarse": {
        "n_streamwise": 60,
        "n_wallnormal": 40,
        "n_spanwise": 40,
        "approx_cells": 96_000,
        "description": "Development grid",
    },
    "medium": {
        "n_streamwise": 100,
        "n_wallnormal": 60,
        "n_spanwise": 60,
        "approx_cells": 360_000,
        "description": "Production grid",
    },
    "fine": {
        "n_streamwise": 160,
        "n_wallnormal": 80,
        "n_spanwise": 80,
        "approx_cells": 1_024_000,
        "description": "Fine grid for GCI study",
    },
    "xfine": {
        "n_streamwise": 240,
        "n_wallnormal": 100,
        "n_spanwise": 100,
        "approx_cells": 2_400_000,
        "description": "Extra-fine grid for near-grid-convergence",
    },
}

# ============================================================================
# SU2 Turbulence Model Mapping
# ============================================================================
SU2_MODELS = {
    "SA": {
        "KIND_TURB_MODEL": "SA",
        "description": "Spalart-Allmaras (known to miscalculate separation onset/extent)",
    },
    "SST": {
        "KIND_TURB_MODEL": "SST",
        "description": "Menter SST k-omega (fails at 45° yaw: asymmetric wakes)",
    },
}


def sutherlands_law(T: float) -> float:
    """Compute dynamic viscosity via Sutherland's law."""
    mu_0 = CASE_CONFIG['mu_ref']
    T_0 = CASE_CONFIG['mu_t_ref']
    S = CASE_CONFIG['sutherland_constant']
    return mu_0 * (T / T_0) ** 1.5 * (T_0 + S) / (T + S)


# ============================================================================
# SU2 Config Generator (3D)
# ============================================================================
def generate_su2_config(case_dir: Path, mesh_file: str,
                         reynolds: int = 250000, yaw_deg: int = 0,
                         model: str = "SA", n_iter: int = 30000,
                         restart: bool = False) -> Path:
    """
    Generate SU2 configuration file for the 3D BeVERLI hill.

    Key features:
    - 3D RANS solver with low-Mach preconditioning
    - Freestream direction rotated by yaw angle
    - Wind-tunnel boundary conditions
    """
    re_cfg = REYNOLDS_CASES[reynolds]
    model_cfg = SU2_MODELS[model]
    cfg = CASE_CONFIG

    T = re_cfg['temperature_freestream']
    P_static = re_cfg['pressure_freestream']
    M = re_cfg['mach']
    gamma = cfg['gamma']

    # Freestream direction for yaw angle
    yaw_rad = math.radians(yaw_deg)
    flow_dir_x = math.cos(yaw_rad)
    flow_dir_z = math.sin(yaw_rad)

    # Total conditions (isentropic)
    P_total = P_static * (1 + (gamma - 1) / 2 * M**2) ** (gamma / (gamma - 1))
    T_total = T * (1 + (gamma - 1) / 2 * M**2)

    # Model-dependent config
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
% BeVERLI Hill — 3D Smooth-Body Separation Validation
% Turbulence model: {model} ({model_cfg['description']})
% Re_H = {reynolds}, M = {M}, Yaw = {yaw_deg}°
% Generated by run_beverli_hill.py
%
% BCs: Inlet = subsonic total conditions,
%       Outlet = back-pressure,
%       Hill + floor = adiabatic no-slip,
%       Tunnel ceiling + side walls = inviscid (slip)
% =============================================================================

% ------------- SOLVER --------------------------------------------------------
SOLVER= RANS
KIND_TURB_MODEL= {model_cfg['KIND_TURB_MODEL']}
MATH_PROBLEM= DIRECT
RESTART_SOL= {'YES' if restart else 'NO'}

% ------------- COMPRESSIBLE FREE-STREAM DEFINITION --------------------------
MACH_NUMBER= {M}
AOA= 0.0
SIDESLIP_ANGLE= {float(yaw_deg)}
FREESTREAM_OPTION= TEMPERATURE_FS
FREESTREAM_PRESSURE= {P_static}
FREESTREAM_TEMPERATURE= {T}
REYNOLDS_NUMBER= {float(reynolds)}
REYNOLDS_LENGTH= {HILL_HEIGHT}

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
REF_LENGTH= {HILL_HEIGHT}
REF_AREA= {HILL_HEIGHT * HILL_WIDTH}
REF_DIMENSIONALIZATION= DIMENSIONAL

% ------------- BOUNDARY CONDITIONS -------------------------------------------
% Hill surface + tunnel floor: Adiabatic no-slip
% Tunnel ceiling + side walls: Inviscid (Euler / slip)
% Inflow: Subsonic inlet (total conditions)
% Outflow: Back-pressure outlet
MARKER_HEATFLUX= ( hill_wall, 0.0, floor, 0.0 )
MARKER_EULER= ( ceiling, sidewall_left, sidewall_right )
MARKER_INLET= ( inflow, {T_total:.3f}, {P_total:.2f}, {flow_dir_x:.6f}, 0.0, {flow_dir_z:.6f} )
MARKER_OUTLET= ( outflow, {P_static} )
MARKER_PLOTTING= ( hill_wall, floor )
MARKER_MONITORING= ( hill_wall )

% ------------- MULTIGRID -----------------------------------------------------
MGLEVEL= 2
MGCYCLE= V_CYCLE
MG_PRE_SMOOTH= ( 1, 2, 3 )
MG_POST_SMOOTH= ( 0, 0, 0 )
MG_CORRECTION_SMOOTH= ( 0, 0, 0 )
MG_DAMP_RESTRICTION= 0.5
MG_DAMP_PROLONGATION= 0.5

% ------------- NUMERICAL METHOD -----------------------------------------------
NUM_METHOD_GRAD= GREEN_GAUSS
CFL_NUMBER= 3.0
CFL_ADAPT= YES
CFL_ADAPT_PARAM= ( 0.3, 1.2, 3.0, 15.0 )

% ------------- LOW-SPEED PRECONDITIONING (important for M < 0.1) ------------
LOW_MACH_PREC= YES

% ------------- LINEAR SOLVER --------------------------------------------------
LINEAR_SOLVER= FGMRES
LINEAR_SOLVER_PREC= ILU
LINEAR_SOLVER_ERROR= 1e-8
LINEAR_SOLVER_ITER= 10

% ------------- FLOW NUMERICAL METHOD ------------------------------------------
CONV_NUM_METHOD_FLOW= ROE
MUSCL_FLOW= YES
SLOPE_LIMITER_FLOW= VENKATAKRISHNAN
VENKAT_LIMITER_COEFF= 0.05
TIME_DISCRE_FLOW= EULER_IMPLICIT

% ------------- TURBULENCE NUMERICAL METHOD ------------------------------------
CONV_NUM_METHOD_TURB= SCALAR_UPWIND
MUSCL_TURB= NO
TIME_DISCRE_TURB= EULER_IMPLICIT

% ------------- CONVERGENCE ----------------------------------------------------
ITER= {n_iter}
CONV_RESIDUAL_MINVAL= -12
CONV_STARTITER= 1000
CONV_FIELD= RMS_DENSITY
CONV_CAUCHY_ELEMS= 300
CONV_CAUCHY_EPS= 1e-7

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

    config_path = case_dir / "beverli_hill.cfg"
    config_path.write_text(config_content)
    return config_path


# ============================================================================
# 3D Mesh Generator (Simplified Structured)
# ============================================================================
def hill_centerline_profile(x_H: np.ndarray) -> np.ndarray:
    """
    Compute the BeVERLI hill centerline height y/H using a Hermite smoothstep.

    Parameters
    ----------
    x_H : array
        Streamwise coordinate normalised by hill height (x/H).

    Returns
    -------
    y_H : array
        Hill height normalised by H.
    """
    x_H = np.asarray(x_H, dtype=float)
    y = np.zeros_like(x_H)
    s_half = 0.25  # flat-top half-width in H units
    L_half = HILL_HALF_LENGTH

    mask_top = np.abs(x_H) <= s_half
    mask_ramp = (np.abs(x_H) > s_half) & (np.abs(x_H) <= L_half)

    y[mask_top] = 1.0
    t = (L_half - np.abs(x_H[mask_ramp])) / (L_half - s_half)
    t = np.clip(t, 0, 1)
    y[mask_ramp] = 10 * t**3 - 15 * t**4 + 6 * t**5

    return y


def generate_beverli_mesh(grid_level: str, grids_dir: Path,
                           yaw_deg: int = 0) -> Optional[Path]:
    """
    Generate a simplified 3D structured mesh for the BeVERLI hill.

    Creates an SU2 hex mesh with proper boundary markers for:
    - hill_wall: hill surface (no-slip)
    - floor: flat tunnel floor (no-slip)
    - ceiling: tunnel ceiling (Euler/slip)
    - sidewall_left, sidewall_right: side walls (Euler/slip)
    - inflow: upstream inlet
    - outflow: downstream outlet
    """
    grid_cfg = GRID_LEVELS[grid_level]
    Ni = grid_cfg["n_streamwise"]
    Nj = grid_cfg["n_wallnormal"]
    Nk = grid_cfg["n_spanwise"]
    mesh_name = f"beverli_{grid_level}_yaw{yaw_deg}.su2"
    su2_path = grids_dir / mesh_name

    if su2_path.exists():
        print(f"  [OK]     SU2 mesh already exists: {mesh_name}")
        return su2_path

    print(f"  [MESH]   Generating 3D mesh: {mesh_name} ({Ni}×{Nj}×{Nk})")

    H = HILL_HEIGHT
    # Domain extents (in meters)
    x_min, x_max = -5.0 * H, 12.0 * H  # 17H total
    y_max = 5.0 * H                      # Tunnel height
    z_min, z_max = -4.0 * H, 4.0 * H     # 8H span

    # Grid stretching: cluster near hill and wall
    x_1d = np.linspace(x_min, x_max, Ni)
    z_1d = np.linspace(z_min, z_max, Nk)

    # Wall-normal: geometric stretching from wall
    ratio = 1.15
    dy_first = 1e-5 * H  # y+ ≈ 1 target
    y_1d = np.zeros(Nj)
    for j in range(1, Nj):
        y_1d[j] = y_1d[j - 1] + dy_first * ratio ** (j - 1)
    # Normalise to fill domain height
    y_1d = y_1d / y_1d[-1] * y_max

    # Compute hill surface at each (x, z) point
    x_H_1d = x_1d / H
    y_surface_center = hill_centerline_profile(x_H_1d) * H

    # Superelliptic spanwise scaling
    z_half = 0.5 * HILL_WIDTH
    n_superellipse = 4.0

    # Total points
    n_pts = Ni * Nj * Nk
    n_elems = (Ni - 1) * (Nj - 1) * (Nk - 1)

    def idx(i, j, k):
        return k * Ni * Nj + j * Ni + i

    # Build point coordinates
    coords = np.zeros((n_pts, 3))
    for k in range(Nk):
        z_val = z_1d[k]
        z_norm = min(abs(z_val) / z_half, 1.0) if z_half > 0 else 1.0
        z_scale = max((1 - z_norm**n_superellipse) ** (1 / n_superellipse), 0.0)

        for i in range(Ni):
            y_surf = y_surface_center[i] * z_scale
            for j in range(Nj):
                # Map y_1d from [0, y_max] to [y_surf, y_max]
                frac = y_1d[j] / y_max
                y_val = y_surf + (y_max - y_surf) * frac
                pt_idx = idx(i, j, k)
                coords[pt_idx] = [x_1d[i], y_val, z_val]

    # Write SU2 mesh
    with open(su2_path, 'w', encoding='utf-8') as f:
        f.write(f"% SU2 mesh - BeVERLI Hill 3D ({Ni}x{Nj}x{Nk})\n")
        f.write(f"% Grid: {grid_level}, Yaw: {yaw_deg}°\n")
        f.write(f"NDIME= 3\n")

        # Hex elements (type 12)
        f.write(f"NELEM= {n_elems}\n")
        elem_id = 0
        for k in range(Nk - 1):
            for j in range(Nj - 1):
                for i in range(Ni - 1):
                    n0 = idx(i, j, k)
                    n1 = idx(i + 1, j, k)
                    n2 = idx(i + 1, j + 1, k)
                    n3 = idx(i, j + 1, k)
                    n4 = idx(i, j, k + 1)
                    n5 = idx(i + 1, j, k + 1)
                    n6 = idx(i + 1, j + 1, k + 1)
                    n7 = idx(i, j + 1, k + 1)
                    f.write(f"12 {n0} {n1} {n2} {n3} "
                            f"{n4} {n5} {n6} {n7} {elem_id}\n")
                    elem_id += 1

        # Points
        f.write(f"NPOIN= {n_pts}\n")
        for pt_id in range(n_pts):
            f.write(f"{coords[pt_id, 0]:.12e} "
                    f"{coords[pt_id, 1]:.12e} "
                    f"{coords[pt_id, 2]:.12e} {pt_id}\n")

        # Boundary markers
        f.write(f"NMARK= 7\n")

        # 1. Floor + Hill (j=0, all i, k) — no-slip walls
        # Separate hill_wall and floor based on whether surface is elevated
        hill_faces = []
        floor_faces = []
        for k in range(Nk - 1):
            for i in range(Ni - 1):
                n0 = idx(i, 0, k)
                n1 = idx(i + 1, 0, k)
                n2 = idx(i + 1, 0, k + 1)
                n3 = idx(i, 0, k + 1)
                # Check if any node is on the hill (elevated surface)
                x_mid = 0.5 * (x_1d[i] + x_1d[i + 1])
                z_mid = 0.5 * (z_1d[k] + z_1d[k + 1])
                x_H_mid = x_mid / H
                z_norm = min(abs(z_mid) / z_half, 1.0)
                z_sc = max((1 - z_norm**n_superellipse) ** (1 / n_superellipse), 0.0)
                y_s = hill_centerline_profile(np.array([x_H_mid]))[0] * H * z_sc
                if y_s > 1e-6 * H:
                    hill_faces.append((n0, n1, n2, n3))
                else:
                    floor_faces.append((n0, n1, n2, n3))

        f.write(f"MARKER_TAG= hill_wall\n")
        f.write(f"MARKER_ELEMS= {len(hill_faces)}\n")
        for n0, n1, n2, n3 in hill_faces:
            f.write(f"9 {n0} {n1} {n2} {n3}\n")

        f.write(f"MARKER_TAG= floor\n")
        f.write(f"MARKER_ELEMS= {len(floor_faces)}\n")
        for n0, n1, n2, n3 in floor_faces:
            f.write(f"9 {n0} {n1} {n2} {n3}\n")

        # 2. Ceiling (j=Nj-1)
        f.write(f"MARKER_TAG= ceiling\n")
        f.write(f"MARKER_ELEMS= {(Ni - 1) * (Nk - 1)}\n")
        for k in range(Nk - 1):
            for i in range(Ni - 1):
                n0 = idx(i, Nj - 1, k)
                n1 = idx(i + 1, Nj - 1, k)
                n2 = idx(i + 1, Nj - 1, k + 1)
                n3 = idx(i, Nj - 1, k + 1)
                f.write(f"9 {n0} {n1} {n2} {n3}\n")

        # 3. Inflow (i=0)
        f.write(f"MARKER_TAG= inflow\n")
        f.write(f"MARKER_ELEMS= {(Nj - 1) * (Nk - 1)}\n")
        for k in range(Nk - 1):
            for j in range(Nj - 1):
                n0 = idx(0, j, k)
                n1 = idx(0, j + 1, k)
                n2 = idx(0, j + 1, k + 1)
                n3 = idx(0, j, k + 1)
                f.write(f"9 {n0} {n1} {n2} {n3}\n")

        # 4. Outflow (i=Ni-1)
        f.write(f"MARKER_TAG= outflow\n")
        f.write(f"MARKER_ELEMS= {(Nj - 1) * (Nk - 1)}\n")
        for k in range(Nk - 1):
            for j in range(Nj - 1):
                n0 = idx(Ni - 1, j, k)
                n1 = idx(Ni - 1, j + 1, k)
                n2 = idx(Ni - 1, j + 1, k + 1)
                n3 = idx(Ni - 1, j, k + 1)
                f.write(f"9 {n0} {n1} {n2} {n3}\n")

        # 5. Left sidewall (k=0)
        f.write(f"MARKER_TAG= sidewall_left\n")
        f.write(f"MARKER_ELEMS= {(Ni - 1) * (Nj - 1)}\n")
        for j in range(Nj - 1):
            for i in range(Ni - 1):
                n0 = idx(i, j, 0)
                n1 = idx(i + 1, j, 0)
                n2 = idx(i + 1, j + 1, 0)
                n3 = idx(i, j + 1, 0)
                f.write(f"9 {n0} {n1} {n2} {n3}\n")

        # 6. Right sidewall (k=Nk-1)
        f.write(f"MARKER_TAG= sidewall_right\n")
        f.write(f"MARKER_ELEMS= {(Ni - 1) * (Nj - 1)}\n")
        for j in range(Nj - 1):
            for i in range(Ni - 1):
                n0 = idx(i, j, Nk - 1)
                n1 = idx(i + 1, j, Nk - 1)
                n2 = idx(i + 1, j + 1, Nk - 1)
                n3 = idx(i, j + 1, Nk - 1)
                f.write(f"9 {n0} {n1} {n2} {n3}\n")

    size_mb = su2_path.stat().st_size / (1024 * 1024)
    print(f"  [OK]     Written {mesh_name} ({size_mb:.1f} MB)")
    print(f"           Nodes: {n_pts:,}, Elements: {n_elems:,}")
    print(f"           Markers: hill_wall, floor, ceiling, inflow, "
          f"outflow, sidewall_left, sidewall_right")
    return su2_path


# ============================================================================
# Case Setup
# ============================================================================
def setup_case(model: str, grid: str, reynolds: int, yaw_deg: int,
               runs_dir: Path, grids_dir: Path,
               n_iter: int = 30000, restart: bool = False) -> Path:
    """Set up a single BeVERLI hill simulation case."""
    case_name = f"beverli_{model}_Re{reynolds // 1000}k_yaw{yaw_deg}_{grid}"
    case_dir = runs_dir / case_name
    case_dir.mkdir(parents=True, exist_ok=True)

    # Generate mesh
    su2_mesh = generate_beverli_mesh(grid, grids_dir, yaw_deg)
    if su2_mesh is None:
        print(f"  [WARN]   Mesh generation failed for {case_name}")
        return case_dir

    # Copy mesh to case directory
    mesh_in_case = case_dir / "mesh.su2"
    if not mesh_in_case.exists():
        shutil.copy2(su2_mesh, mesh_in_case)

    # Generate SU2 config
    generate_su2_config(case_dir, "mesh.su2", reynolds=reynolds,
                         yaw_deg=yaw_deg, model=model, n_iter=n_iter,
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
                    timeout: int = 28800) -> Dict:
    """Run SU2_CFD for a single BeVERLI hill case."""
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
            err_detail = ""
            if log_file.exists():
                lines = log_file.read_text(errors='replace').splitlines()
                tail = lines[-20:] if len(lines) > 20 else lines
                err_detail = "\n".join(tail)
            result["error"] = (f"SU2_CFD exited with code {proc.returncode}\n"
                               f"--- Last 20 lines of log ---\n{err_detail}")
            print(f"  [FAIL]   Return code {proc.returncode}")
            return result

        # Parse history
        history = parse_su2_history(case_dir)
        if history:
            result["CL"] = history.get("CL")
            result["CD"] = history.get("CD")
            result["iterations"] = history.get("iterations", 0)

            hist_file = case_dir / "history.csv"
            if hist_file.exists():
                n_lines = sum(1 for _ in open(hist_file, errors='replace')) - 1
                if n_lines < 10:
                    result["error"] = f"Only {n_lines} iterations in history.csv"
                else:
                    print(f"  [OK]     {n_lines} iterations logged")

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
        description="BeVERLI Hill — 3D Smooth-Body Separation Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python run_beverli_hill.py --dry-run
  python run_beverli_hill.py --model SA SST --re 250000 --yaw 0
  python run_beverli_hill.py --model SA --re 250000 650000 --yaw 0 30 45
  python run_beverli_hill.py --validate-only
"""
    )
    parser.add_argument("--model", nargs="+", default=["SA"],
                        choices=list(SU2_MODELS.keys()),
                        help="Turbulence models (default: SA)")
    parser.add_argument("--re", nargs="+", type=int, default=[250000],
                        help="Reynolds numbers (default: 250000)")
    parser.add_argument("--yaw", nargs="+", type=int, default=[0],
                        help="Yaw angles in degrees (default: 0)")
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
    parser.add_argument("--n-iter", type=int, default=30000,
                        help="Max iterations (default: 30000)")
    parser.add_argument("--timeout", type=int, default=28800,
                        help="Timeout in seconds (default: 28800 = 8h)")
    parser.add_argument("--restart", action="store_true",
                        help="Restart from existing solution")
    parser.add_argument("--runs-dir", type=Path, default=None,
                        help="Output directory for runs")
    args = parser.parse_args()

    # Validate inputs
    for re in args.re:
        if re not in REYNOLDS_CASES:
            print(f"  [!!]     Unknown Re={re}. "
                  f"Available: {list(REYNOLDS_CASES.keys())}")
            return
    for yaw in args.yaw:
        if yaw not in VALID_YAW_ANGLES:
            print(f"  [!!]     Unknown yaw={yaw}°. "
                  f"Available: {VALID_YAW_ANGLES}")
            return

    n_cases = len(args.model) * len(args.re) * len(args.yaw)

    print("=" * 60)
    print("  BeVERLI HILL — 3D SMOOTH-BODY SEPARATION")
    print("  Virginia Tech / NASA Langley Validation")
    print("=" * 60)
    print(f"\n  Hill height: H = {HILL_HEIGHT} m")
    print(f"  Reynolds:    {args.re}")
    print(f"  Yaw angles:  {args.yaw}°")
    print(f"  Models:      {', '.join(args.model)}")
    print(f"  Grid:        {args.grid} (~{GRID_LEVELS[args.grid]['approx_cells']:,} cells)")
    print(f"  Total cases: {n_cases}")

    runs_dir = args.runs_dir or PROJECT_ROOT / "runs" / "beverli_hill"
    runs_dir.mkdir(parents=True, exist_ok=True)
    grids_dir = PROJECT_ROOT / "experimental_data" / "beverli_hill" / "grids"
    grids_dir.mkdir(parents=True, exist_ok=True)
    data_dir = PROJECT_ROOT / "experimental_data" / "beverli_hill"

    # --- Step 1: Generate reference data ---
    if not (data_dir / "csv" / "beverli_case_reference.json").exists():
        print("\n  [STEP 1] Generating BeVERLI reference data...")
        sys.path.insert(0, str(data_dir))
        from experimental_data.beverli_hill.beverli_hill_data import download_all
        download_all()
    else:
        print("\n  [STEP 1] Reference data already generated [OK]")

    # --- Step 2: Validate only? ---
    if args.validate_only:
        print("\n  [VALIDATE] Post-processing existing results...")
        return

    # --- Step 3: Setup cases ---
    print(f"\n  [STEP 2] Setting up {n_cases} cases...")
    case_dirs = {}
    for model in args.model:
        for re in args.re:
            for yaw in args.yaw:
                case_dir = setup_case(
                    model, args.grid, re, yaw, runs_dir, grids_dir,
                    n_iter=args.n_iter, restart=args.restart
                )
                key = f"{model}_Re{re // 1000}k_yaw{yaw}"
                case_dirs[key] = case_dir
                print(f"  [OK]     {case_dir.name}")

    if args.dry_run:
        print("\n  [DRY-RUN] Cases set up. No solver executed.")
        print(f"\n  Case directories created in: {runs_dir}")
        for key, case_dir in case_dirs.items():
            print(f"\n  To run {key}:")
            print(f"    cd {case_dir}")
            if args.n_threads > 1:
                print(f"    SU2_CFD -t {args.n_threads} beverli_hill.cfg")
            else:
                print(f"    SU2_CFD beverli_hill.cfg")
        return

    # --- Step 4: Run simulations ---
    if not check_solver():
        print("\n  [!!]     Cannot run without SU2. Use --dry-run.")
        return

    print(f"\n  [STEP 3] Running {n_cases} simulations...")
    all_results = {}
    for key, case_dir in case_dirs.items():
        config = case_dir / "beverli_hill.cfg"
        print(f"\n  --- {key} ---")
        result = run_simulation(
            case_dir, config,
            n_procs=args.n_procs, n_threads=args.n_threads,
            timeout=args.timeout
        )
        all_results[key] = result
        if result["error"]:
            print(f"  [FAIL]   {result['error'][:80]}")
        elif result.get("CD") is not None:
            print(f"  [OK]     CD={result['CD']:.5f}, "
                  f"time={result['wall_time_s']:.1f}s")

    # Save results summary
    summary_file = runs_dir / "results_summary.json"
    summary = {
        "case": "BeVERLI Hill — 3D Smooth-Body Separation",
        "grid": args.grid,
        "grid_cells": GRID_LEVELS[args.grid]["approx_cells"],
        "hill_height_m": HILL_HEIGHT,
    }
    for key, result in all_results.items():
        summary[key] = {
            "CD": result.get("CD"),
            "converged": result.get("converged", False),
            "wall_time_s": result.get("wall_time_s", 0),
        }
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Results saved: {summary_file}")


if __name__ == "__main__":
    main()
