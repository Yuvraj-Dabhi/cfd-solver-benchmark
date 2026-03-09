#!/usr/bin/env python3
"""
NACA 0012 Simulation Runner (TMR Validation Case 2DN00)
========================================================
End-to-end runner for the NASA TMR NACA 0012 validation case.

Follows TMR specifications:
    M    = 0.15 (essentially incompressible)
    Re   = 6 × 10^6 per chord
    Flow = Fully turbulent
    TE   = Sharp (TMR scaled definition)

Workflow:
    1. Download TMR data & grids (if not present)
    2. Convert PLOT3D → SU2 mesh
    3. Generate SU2 configuration for each (alpha, model) pair
    4. Run SU2_CFD simulations
    5. Post-process and validate against TMR reference

Usage:
    python run_naca0012.py --dry-run              # Setup only, no solver
    python run_naca0012.py --alpha 0 10 15         # Run specific alphas
    python run_naca0012.py --model SA SST          # Run specific models
    python run_naca0012.py --validate-only         # Skip simulation, just validate
    python run_naca0012.py --grid medium           # Choose grid level

Requirements:
    SU2_CFD must be on PATH (see SOLVER_SETUP.md)
"""

import sys
import os
import json
import shutil
import subprocess
import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Project root
PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================================
# TMR Case Configuration
# ============================================================================
# From: https://turbmodels.larc.nasa.gov/naca0012_val.html
#       https://turbmodels.larc.nasa.gov/implementrans.sa.html
#
# Conditions:
#   M = 0.15, Re = 6 million (per chord), alpha = various
#   T_ref = 540 R (= 300 K) freestream static temperature
#   Fully turbulent boundary layers
#
# Fluid properties:
#   - Full compressible Navier-Stokes
#   - Pr = 0.72 (laminar), Prt = 0.9 (turbulent)
#   - gamma = 1.4
#   - Sutherland's Law for dynamic viscosity:
#       mu = mu_0 * (T/T_0)^(3/2) * (T_0 + S) / (T + S)
#       mu_0 = 1.716e-5 kg/(m*s), T_0 = 273.11 K (491.6 R), S = 110.33 K (198.6 R)
#   - T_ref = 300 K (540 R) for this case
#
# Boundary conditions:
#   - Farfield: Riemann invariant-based characteristic BC (~500c extent)
#     * Point vortex correction RECOMMENDED (Thomas & Salas, AIAA J. 24(7), 1986)
#     * DOI: https://doi.org/10.2514/3.9394
#   - Airfoil wall: Adiabatic solid wall (zero heat flux)
#
# Turbulence inflow (SA model, TMR note 5):
#   nu_tilde_inf = 3 * nu  (freestream eddy viscosity ratio mu_t/mu = 3)
#

# --- Sutherland's Law constants (from TMR / White, "Viscous Fluid Flow") ---
SUTHERLAND_MU0 = 1.716e-5      # Reference dynamic viscosity [kg/(m*s)]
SUTHERLAND_T0  = 273.11         # Reference temperature [K] (= 491.6 R)
SUTHERLAND_S   = 110.33         # Sutherland constant [K] (= 198.6 R)

CASE_CONFIG = {
    "name": "NACA 0012 TMR Validation (2DN00)",
    "mach": 0.15,
    "reynolds": 6e6,
    "chord": 1.0,
    # Thermodynamic properties
    "gamma": 1.4,                        # Heat capacity ratio (cp/cv)
    "gas_constant": 287.058,             # Specific gas constant [J/(kg*K)]
    "temperature_freestream": 300.0,     # K (= 540 R, TMR T_ref)
    "pressure_freestream": 101325.0,     # Pa
    # Prandtl numbers (TMR spec)
    "prandtl_lam": 0.72,                 # Laminar Prandtl number
    "prandtl_turb": 0.90,                # Turbulent Prandtl number
    # Viscosity via Sutherland's Law
    "viscosity_model": "SUTHERLAND",
    "mu_ref": SUTHERLAND_MU0,            # 1.716e-5 kg/(m*s)
    "mu_t_ref": SUTHERLAND_T0,           # 273.11 K (491.6 R)
    "sutherland_constant": SUTHERLAND_S, # 110.33 K (198.6 R)
    # Turbulence
    "fully_turbulent": True,
    "turbulence_intensity": 0.03,        # 3% freestream
    "mu_t_ratio": 3.0,                   # SA: nu_tilde_inf = 3*nu (TMR note 5)
}


def sutherlands_law(T: float,
                    mu_0: float = SUTHERLAND_MU0,
                    T_0: float = SUTHERLAND_T0,
                    S: float = SUTHERLAND_S) -> float:
    """
    Compute dynamic viscosity using Sutherland's Law.

    mu = mu_0 * (T / T_0)^(3/2) * (T_0 + S) / (T + S)

    From White, F. M., "Viscous Fluid Flow," McGraw Hill, 1974, p. 28.
    TMR constants: mu_0 = 1.716e-5 kg/(m*s), T_0 = 273.11 K (491.6 R),
                   S = 110.33 K (198.6 R).

    The nondimensional form (for codes using reference quantities):
      mu/mu_ref = (T/T_ref)^(3/2) * (T_ref + S) / (T + S)
    where T_ref = 300 K (540 R) and mu_ref = mu(T_ref).

    Parameters
    ----------
    T : float
        Local temperature [K].
    mu_0 : float
        Reference viscosity [kg/(m*s)].
    T_0 : float
        Reference temperature for mu_0 [K].
    S : float
        Sutherland constant [K].

    Returns
    -------
    float
        Dynamic viscosity [kg/(m*s)].
    """
    return mu_0 * (T / T_0) ** 1.5 * (T_0 + S) / (T + S)


def compute_turbulence_inflow(model: str = "SA") -> Dict:
    """
    Compute and report turbulence inflow conditions per TMR specifications.

    SA model:
      nu_tilde_inf = 3 * nu  (TMR note 5)

    SST model:
      k_inf   = 1.5 * (TI * U_inf)^2
      omega_inf = rho_inf * k_inf / (mu_t_ratio * mu_inf)

    Returns dict with computed values and documentation strings.
    """
    import math
    cfg = CASE_CONFIG
    T   = cfg['temperature_freestream']
    p   = cfg['pressure_freestream']
    R   = cfg['gas_constant']
    gam = cfg['gamma']
    TI  = cfg['turbulence_intensity']
    mu_t_ratio = cfg['mu_t_ratio']

    mu  = sutherlands_law(T)
    rho = p / (R * T)
    nu  = mu / rho
    a   = math.sqrt(gam * R * T)
    U   = cfg['mach'] * a

    result = {
        'mu_inf': mu,
        'rho_inf': rho,
        'nu_inf': nu,
        'U_inf': U,
        'a_inf': a,
    }

    if model == 'SA':
        nu_tilde = mu_t_ratio * nu
        result['nu_tilde_inf'] = nu_tilde
        result['report'] = (
            f"SA turbulence inflow:\n"
            f"  nu_inf        = {nu:.6e} m^2/s\n"
            f"  nu_tilde_inf  = {mu_t_ratio} * nu = {nu_tilde:.6e} m^2/s\n"
            f"  mu_t/mu ratio = {mu_t_ratio}"
        )
    elif model == 'SST':
        k_inf = 1.5 * (TI * U) ** 2
        omega_inf = rho * k_inf / (mu_t_ratio * mu)
        result['k_inf'] = k_inf
        result['omega_inf'] = omega_inf
        result['report'] = (
            f"SST turbulence inflow:\n"
            f"  TI             = {TI*100:.1f}%\n"
            f"  k_inf          = {k_inf:.6e} m^2/s^2\n"
            f"  omega_inf      = {omega_inf:.6e} 1/s\n"
            f"  mu_t/mu ratio  = {mu_t_ratio}"
        )

    return result


def point_vortex_correction(x_far: float, y_far: float, alpha_deg: float,
                            CL: float, chord: float = 1.0,
                            M_inf: float = 0.15, a_inf: float = 347.22,
                            rho_inf: float = 1.1766,
                            p_inf: float = 101325.0,
                            gamma: float = 1.4) -> Dict:
    """
    Compute farfield point vortex BC correction.

    Per Thomas & Salas (AIAA J. 24(7):1074-1080, 1986):
    The airfoil is modeled as a point vortex at the quarter-chord.
    The induced velocity at the farfield boundary corrects the
    freestream conditions to better represent an infinite domain.

    This correction is recommended by TMR even for the ~500c farfield
    extent, as its influence is noticeable at the detailed validation
    levels being investigated.

    Parameters
    ----------
    x_far, y_far : float
        Farfield point coordinates.
    alpha_deg : float
        Angle of attack [degrees].
    CL : float
        Lift coefficient (use initial estimate, iterate if needed).
    chord : float
        Chord length.
    M_inf, a_inf, rho_inf, p_inf, gamma : float
        Freestream conditions.

    Returns
    -------
    dict with corrected velocity components u_corr, v_corr and
    corrected pressure p_corr at the farfield point.
    """
    import math
    U_inf = M_inf * a_inf

    # Circulation from CL:  Gamma = 0.5 * CL * U_inf * chord
    Gamma = 0.5 * CL * U_inf * chord

    # Point vortex at quarter-chord (0.25, 0)
    dx = x_far - 0.25 * chord
    dy = y_far
    r2 = dx**2 + dy**2

    if r2 < 1e-10:
        return {'u_corr': 0.0, 'v_corr': 0.0, 'p_corr': 0.0}

    # Induced velocity from point vortex (incompressible approximation)
    # Compressibility correction: beta = sqrt(1 - M^2)
    beta = math.sqrt(1.0 - M_inf**2)
    u_vortex = Gamma / (2.0 * math.pi) * dy / r2
    v_vortex = -Gamma / (2.0 * math.pi) * dx / r2

    # Prandtl-Glauert compressibility correction
    u_vortex /= beta
    v_vortex /= beta

    # Corrected pressure (isentropic relation)
    alpha_rad = math.radians(alpha_deg)
    u_total = U_inf * math.cos(alpha_rad) + u_vortex
    v_total = U_inf * math.sin(alpha_rad) + v_vortex
    V2 = u_total**2 + v_total**2
    V_inf2 = U_inf**2

    # Bernoulli (low Mach): p_corr = p_inf + 0.5*rho*(V_inf^2 - V^2)
    p_corr = p_inf + 0.5 * rho_inf * (V_inf2 - V2)

    return {
        'u_corr': u_vortex,
        'v_corr': v_vortex,
        'p_corr': p_corr - p_inf,
        'Gamma': Gamma,
        'delta_V_over_V': math.sqrt(u_vortex**2 + v_vortex**2) / U_inf,
    }

# ===========================================================================
# TMR Grid Family Definitions
# ===========================================================================
#
# The TMR provides three families of structured C-grids for NACA 0012:
#
#   Family I   — Original grid sequence (7 separate grid files)
#   Family II  — Different node distribution / near-wall spacing (7 files)
#   Family III — Only finest level provided; coarser levels are extracted
#                by taking every other grid point in each direction
#
# All families have 7 levels with uniform refinement ratio r=2,
# enabling proper Richardson extrapolation for grid convergence studies.
#
# Level numbering:  1 = finest (7169×2049) → 7 = coarsest (113×33)
# The airfoil has (idim+1)/2 points on the surface (C-grid topology).
#
# From: https://turbmodels.larc.nasa.gov/naca0012_val.html
# ===========================================================================

FAMILY_I_DIR = "2-D version of the FAMILY I grids in PLOT3D format"
FAMILY_II_DIR = "2-D version of the FAMILY II grids in PLOT3D"
FAMILY_III_DIR = "2-D version of the finest FAMILY III grid in PLOT3D"

# Grid level definitions (shared across all families — same dimensions)
GRID_LEVEL_SPECS = {
    "coarse": {"dims": (113, 33),   "airfoil_pts": 65,   "family_level": 7},
    "medium": {"dims": (225, 65),   "airfoil_pts": 129,  "family_level": 6},
    "fine":   {"dims": (449, 129),  "airfoil_pts": 257,  "family_level": 5},
    "xfine":  {"dims": (897, 257),  "airfoil_pts": 513,  "family_level": 4},
    "ultra":  {"dims": (1793, 513), "airfoil_pts": 1025, "family_level": 3},
    "super":  {"dims": (3585, 1025),"airfoil_pts": 2049, "family_level": 2},
    "hyper":  {"dims": (7169, 2049),"airfoil_pts": 4097, "family_level": 1},
}

# Standalone fallback files (for the 4 original grids in grids/ directory)
_STANDALONE_ALT = {
    "coarse": "n0012_113-33.p2dfmt",
    "medium": "n0012_225-65.p2dfmt",
    "fine":   "n0012_449-129.p2dfmt",
    "xfine":  "n0012_897-257.p2dfmt",
}


def build_grid_levels(family: str = "I") -> dict:
    """Build complete GRID_LEVELS dict for the specified grid family.

    Parameters
    ----------
    family : str
        Grid family: 'I', 'II', or 'III'.

    Returns
    -------
    dict
        Grid level definitions with PLOT3D file paths.
    """
    family = family.upper()

    if family == "III":
        # Family III: only the finest grid is provided; coarser levels
        # are generated by taking every other point (n_coarsen steps)
        finest_file = f"{FAMILY_III_DIR}/n0012familyIII.1.p2dfmt.gz"
        levels = {}
        for name, spec in GRID_LEVEL_SPECS.items():
            coarsen_steps = spec["family_level"] - 1  # level 1 → 0 steps
            levels[name] = {
                "plot3d": finest_file,
                "dims": spec["dims"],
                "airfoil_pts": spec["airfoil_pts"],
                "family_level": spec["family_level"],
                "family": "III",
                "coarsen_steps": coarsen_steps,
            }
        return levels

    if family == "II":
        family_dir = FAMILY_II_DIR
        prefix = "n0012familyII"
    else:
        family_dir = FAMILY_I_DIR
        prefix = "n0012familyI"

    levels = {}
    for name, spec in GRID_LEVEL_SPECS.items():
        level = spec["family_level"]
        entry = {
            "plot3d": f"{family_dir}/{prefix}.{level}.p2dfmt.gz",
            "dims": spec["dims"],
            "airfoil_pts": spec["airfoil_pts"],
            "family_level": level,
            "family": family,
            "coarsen_steps": 0,
        }
        if name in _STANDALONE_ALT:
            entry["plot3d_alt"] = _STANDALONE_ALT[name]
        levels[name] = entry

    return levels


# Default: Family I (used for imports/tests; overridden in main() if --grid-family is set)
GRID_LEVELS = build_grid_levels("I")

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

# Default alpha sweep
DEFAULT_ALPHAS = [0.0, 5.0, 10.0, 13.0, 15.0]


# ============================================================================
# TMR NACA 0012 Geometry Definition
# ============================================================================
#
# From https://turbmodels.larc.nasa.gov/naca0012_val.html:
#
# The TMR definition uses a NACA 0012 that is slightly altered from the
# original so the airfoil closes at chord=1 with a sharp trailing edge.
#
# Original NACA 0012 formula:
#   y = +/- 0.6 * [0.2969*sqrt(x) - 0.1260*x - 0.3516*x^2
#                   + 0.2843*x^3 - 0.1015*x^4]
#
# This produces a sharp TE at x = 1.008930411365. The airfoil is then
# scaled down by that factor so that it closes at chord = 1 exactly.
#
# TMR scaled (revised) definition:
#   y = +/- 0.594689181 * [0.298222773*sqrt(x) - 0.127125232*x
#                          - 0.357907906*x^2 + 0.291984971*x^3
#                          - 0.105174606*x^4]
#
# Max thickness: ~11.894% relative to chord (vs. 12% for original blunted TE)

# TMR scaling factor
TMR_SCALE_FACTOR = 1.008930411365

# TMR scaled coefficients (for chord = 1, sharp TE)
TMR_COEFF = {
    "a0": 0.594689181,   # outer multiplier
    "a1": 0.298222773,   # sqrt(x) term
    "a2": -0.127125232,  # x term
    "a3": -0.357907906,  # x^2 term
    "a4": 0.291984971,   # x^3 term
    "a5": -0.105174606,  # x^4 term (ensures y=0 at x=1)
}

# Original NACA 0012 coefficients (blunted TE)
ORIGINAL_COEFF = {
    "a0": 0.6,
    "a1": 0.2969,
    "a2": -0.1260,
    "a3": -0.3516,
    "a4": 0.2843,
    "a5": -0.1015,
}


def naca0012_tmr_y(x):
    """
    Compute half-thickness y(x) for the TMR NACA 0012 (sharp TE, chord=1).

    Uses the revised TMR formula:
      y = 0.594689181 * [0.298222773*sqrt(x) - 0.127125232*x
                         - 0.357907906*x^2 + 0.291984971*x^3
                         - 0.105174606*x^4]

    Parameters
    ----------
    x : array-like
        Chordwise coordinates, 0 <= x <= 1.

    Returns
    -------
    ndarray
        Half-thickness (positive upper surface).
    """
    import numpy as np
    c = TMR_COEFF
    return c["a0"] * (c["a1"] * np.sqrt(x)
                      + c["a2"] * x
                      + c["a3"] * x**2
                      + c["a4"] * x**3
                      + c["a5"] * x**4)


def naca0012_tmr_surface(n_pts: int = 200):
    """
    Generate NACA 0012 surface coordinates using the TMR sharp-TE definition.

    Points are distributed using cosine clustering (finer at LE and TE).

    Parameters
    ----------
    n_pts : int
        Number of points on each surface (upper and lower).

    Returns
    -------
    x : ndarray, shape (2*n_pts - 1,)
        Chordwise coordinates (TE upper -> LE -> TE lower).
    y : ndarray, shape (2*n_pts - 1,)
        Surface y-coordinates.
    """
    import numpy as np

    # Cosine clustering: finer at LE and TE
    beta = np.linspace(0, np.pi, n_pts)
    x_upper = 0.5 * (1.0 - np.cos(beta))  # 0 -> 1

    y_half = naca0012_tmr_y(x_upper)

    # Upper surface (TE -> LE, positive y)
    x_u = x_upper[::-1]
    y_u = y_half[::-1]

    # Lower surface (LE -> TE, negative y), skip LE duplicate
    x_l = x_upper[1:]
    y_l = -y_half[1:]

    x = np.concatenate([x_u, x_l])
    y = np.concatenate([y_u, y_l])

    return x, y



# ============================================================================
# SU2 Configuration Generator
# ============================================================================

def generate_su2_config(case_dir: Path, mesh_file: str, alpha: float,
                         model: str = "SA", n_iter: int = 30000,
                         restart: bool = False) -> Path:
    """
    Generate SU2 configuration file for NACA 0012 at given alpha.

    Follows TMR recommendations:
    - Full compressible Navier-Stokes (RANS)
    - Sutherland's Law viscosity (mu_0=1.716e-5, T_0=273.11K, S=110.33K)
    - Pr=0.72, Prt=0.9, gamma=1.4
    - Freestream SA: nu_tilde = 3*nu (TMR note 5)
    - Farfield: Riemann characteristic BC
    - Wall: Adiabatic (zero heat flux)
    - Point vortex correction recommended (Thomas & Salas 1986)
    """
    cfg = CASE_CONFIG
    model_cfg = SU2_MODELS[model]

    import math

    # Compute mu_ref at freestream T for documentation
    mu_at_Tref = sutherlands_law(cfg['temperature_freestream'])

    # Compute turbulence inflow conditions for reporting
    turb_inflow = compute_turbulence_inflow(model)

    # Compute point vortex correction magnitude at farfield (~500c)
    # Use CL estimate for reporting (actual correction applied by solver if supported)
    cl_estimate = 2.0 * math.pi * math.radians(alpha)  # thin airfoil theory
    if abs(alpha) > 0.1:
        pv_corr = point_vortex_correction(
            x_far=500.0, y_far=0.0, alpha_deg=alpha, CL=cl_estimate,
            M_inf=cfg['mach'], a_inf=turb_inflow['a_inf'],
            rho_inf=turb_inflow['rho_inf'], p_inf=cfg['pressure_freestream'],
            gamma=cfg['gamma']
        )
        pv_note = f"Point vortex correction: delta_V/V = {pv_corr['delta_V_over_V']:.2e} at 500c"
    else:
        pv_note = "Point vortex correction: negligible at alpha=0"

    # Build turbulence inflow report for config header
    if model == 'SA':
        turb_report = f"nu_tilde_inf = {turb_inflow['nu_tilde_inf']:.6e} m^2/s (= {cfg['mu_t_ratio']}*nu)"
    else:
        turb_report = (f"k_inf = {turb_inflow.get('k_inf', 0):.4e}, "
                       f"omega_inf = {turb_inflow.get('omega_inf', 0):.4e}")

    # Alpha-dependent CFL strategy:
    #   Low alpha (attached flow): aggressive CFL for fast convergence
    #   High alpha (separated flow): conservative CFL for stability
    #   cfl_max_mg: reduced CFL max for multigrid stability
    if abs(alpha) <= 5.0:
        cfl_start = 5.0
        cfl_max = 50.0      # Reduced from 100 for safety
        cfl_max_mg = 25.0   # Reduced from 50
        venkat = 0.03
    elif abs(alpha) <= 12.0:
        cfl_start = 1.0     # Reduced from 3.0
        cfl_max = 15.0      # Reduced from 50.0
        cfl_max_mg = 10.0   # Reduced from 25.0
        venkat = 0.01       # More aggressive limiting for separated flow
    else:
        cfl_start = 0.5     # Reduced from 1.0
        cfl_max = 10.0      # Reduced from 25.0
        cfl_max_mg = 5.0    # Reduced from 15.0
        venkat = 0.005      # Very strong limiter for high AoA separation

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
% NACA 0012 — TMR Validation Case (2DN00)
% Turbulence model: {model} ({model_cfg['description']})
% Alpha = {alpha} deg, M = {cfg['mach']}, Re = {cfg['reynolds']:.0e}
% Generated by run_naca0012.py
%
% Fluid: Compressible N-S, Pr={cfg['prandtl_lam']}, Prt={cfg['prandtl_turb']}, gamma={cfg['gamma']}
% Viscosity: Sutherland's Law, mu_ref={mu_at_Tref:.6e} kg/(m*s) at T_ref={cfg['temperature_freestream']}K
% Turbulence inflow: {turb_report}
% {pv_note}
%
% BCs: Farfield = Riemann characteristic, Wall = adiabatic (q=0)
% Ref: Thomas & Salas, AIAA J. 24(7):1074-1080, 1986 (point vortex)
% =============================================================================

% ------------- DIRECT, ADJOINT, AND LINEARIZED PROBLEM DEFINITION -----------
SOLVER= RANS
KIND_TURB_MODEL= {model_cfg['KIND_TURB_MODEL']}
MATH_PROBLEM= DIRECT
RESTART_SOL= {'YES' if restart else 'NO'}

% ------------- COMPRESSIBLE FREE-STREAM DEFINITION --------------------------
% T_ref = {cfg['temperature_freestream']} K (= 540 R)
MACH_NUMBER= {cfg['mach']}
AOA= {alpha}
SIDESLIP_ANGLE= 0.0
FREESTREAM_OPTION= TEMPERATURE_FS
FREESTREAM_PRESSURE= {cfg['pressure_freestream']}
FREESTREAM_TEMPERATURE= {cfg['temperature_freestream']}
REYNOLDS_NUMBER= {cfg['reynolds']:.1f}
REYNOLDS_LENGTH= {cfg['chord']}

% ------------- FLUID MODEL (TMR spec) ---------------------------------------
% gamma = 1.4, gas constant = 287.058 J/(kg*K)
GAMMA_VALUE= {cfg['gamma']}
GAS_CONSTANT= {cfg['gas_constant']}

% ------------- VISCOSITY — Sutherland's Law (TMR / White 1974) ---------------
% mu = mu_0 * (T/T_0)^(3/2) * (T_0 + S) / (T + S)
% mu_0 = 1.716e-5 kg/(ms), T_0 = 273.11 K (491.6 R), S = 110.33 K (198.6 R)
VISCOSITY_MODEL= SUTHERLAND
MU_REF= {cfg['mu_ref']}
MU_T_REF= {cfg['mu_t_ref']}
SUTHERLAND_CONSTANT= {cfg['sutherland_constant']}

% ------------- THERMAL CONDUCTIVITY ------------------------------------------
% Pr = 0.72 (laminar), Prt = 0.9 (turbulent) — TMR specification
CONDUCTIVITY_MODEL= CONSTANT_PRANDTL
PRANDTL_LAM= {cfg['prandtl_lam']}
PRANDTL_TURB= {cfg['prandtl_turb']}

% ------------- TURBULENCE (TMR recommendations) -----------------------------
% Turbulence inflow: {turb_report}
{turb_inflow_block}

% ------------- REFERENCE VALUES ---------------------------------------------
REF_ORIGIN_MOMENT_X= 0.25
REF_ORIGIN_MOMENT_Y= 0.00
REF_ORIGIN_MOMENT_Z= 0.00
REF_LENGTH= {cfg['chord']}
REF_AREA= {cfg['chord']}
REF_DIMENSIONALIZATION= FREESTREAM_PRESS_EQ_ONE

% ------------- BOUNDARY CONDITIONS ------------------------------------------
% Wall: Adiabatic solid wall (zero heat flux, dT/dn = 0)
% Farfield: Riemann invariant-based characteristic BC (~500c extent)
% Note: Point vortex correction (Thomas & Salas 1986) is recommended.
%       {pv_note}
MARKER_HEATFLUX= ( airfoil, 0.0 )
MARKER_FAR= ( farfield )
MARKER_PLOTTING= ( airfoil )
MARKER_MONITORING= ( airfoil )

% ------------- LOW MACH NOTE ------------------------------------------------
% M=0.15 is low-Mach but SU2 tutorial does not use LOW_MACH_PREC.
% Non-dimensional formulation (FREESTREAM_PRESS_EQ_ONE) handles this well.

% ------------- MULTIGRID (accelerates convergence) -------------------------
MGLEVEL= 2
MGCYCLE= V_CYCLE
MG_PRE_SMOOTH= ( 1, 2, 3 )
MG_POST_SMOOTH= ( 0, 0, 0 )
MG_CORRECTION_SMOOTH= ( 0, 0, 0 )
MG_DAMP_RESTRICTION= 0.5
MG_DAMP_PROLONGATION= 0.5

% ------------- NUMERICAL METHOD ---------------------------------------------
NUM_METHOD_GRAD= WEIGHTED_LEAST_SQUARES
CFL_NUMBER= {cfl_start}
CFL_ADAPT= YES
CFL_ADAPT_PARAM= ( 0.5, 1.5, {cfl_start}, {cfl_max_mg} )

% ------------- LINEAR SOLVER ------------------------------------------------
LINEAR_SOLVER= FGMRES
LINEAR_SOLVER_PREC= LU_SGS
LINEAR_SOLVER_ERROR= 1e-10
LINEAR_SOLVER_ITER= 5

% ------------- FLOW NUMERICAL METHOD ----------------------------------------
CONV_NUM_METHOD_FLOW= ROE
MUSCL_FLOW= YES
SLOPE_LIMITER_FLOW= VENKATAKRISHNAN
VENKAT_LIMITER_COEFF= {venkat}
TIME_DISCRE_FLOW= EULER_IMPLICIT

% ------------- TURBULENCE NUMERICAL METHOD ----------------------------------
CONV_NUM_METHOD_TURB= SCALAR_UPWIND
MUSCL_TURB= NO
TIME_DISCRE_TURB= EULER_IMPLICIT

% ------------- CONVERGENCE --------------------------------------------------
ITER= {n_iter}
CONV_RESIDUAL_MINVAL= -14
CONV_STARTITER= 10
CONV_FIELD= DRAG
CONV_CAUCHY_ELEMS= 200
CONV_CAUCHY_EPS= 1e-8

% ------------- INPUT/OUTPUT -------------------------------------------------
MESH_FILENAME= {mesh_file}
MESH_FORMAT= SU2
SOLUTION_FILENAME= restart_flow.dat
RESTART_FILENAME= restart_flow.dat
CONV_FILENAME= history
VOLUME_FILENAME= flow
SURFACE_FILENAME= surface_flow
OUTPUT_WRT_FREQ= 500
OUTPUT_FILES= (RESTART, PARAVIEW, SURFACE_PARAVIEW, SURFACE_CSV)
SCREEN_OUTPUT= (INNER_ITER, RMS_DENSITY, RMS_ENERGY, LIFT, DRAG, MOMENT_Z, {turb_screen_vars})
HISTORY_OUTPUT= (ITER, RMS_RES, AERO_COEFF, CAUCHY)
"""

    config_path = case_dir / "naca0012.cfg"
    config_path.write_text(config_content)
    return config_path


# ============================================================================
# Simulation Runner
# ============================================================================

def check_solver():
    """Check if SU2_CFD is available and report capabilities."""
    result = shutil.which("SU2_CFD") or shutil.which("SU2_CFD.exe")
    if result:
        print(f"  [OK]     SU2_CFD found: {result}")
        # Check OpenMP support
        try:
            proc = subprocess.run(
                [result, "--help"], capture_output=True, text=True, timeout=5
            )
            has_omp = "--threads" in proc.stdout or "-t" in proc.stdout
            if has_omp:
                print(f"           OpenMP threading: supported (-t flag)")
        except Exception:
            pass
        return True
    else:
        print(f"  [!!]     SU2_CFD not found on PATH")
        print(f"           Install SU2 or add to PATH. See SOLVER_SETUP.md")
        return False


def run_simulation(case_dir: Path, config_file: Path,
                    n_procs: int = 1, n_threads: int = 1,
                    timeout: int = 14400) -> Dict:
    """
    Run SU2_CFD for a single case.

    Parallelization strategy (in priority order):
    1. OpenMP threads: SU2_CFD -t <N> config.cfg  (best for single-machine)
    2. MPI processes:  mpiexec -np <N> SU2_CFD config.cfg
    3. Serial:         SU2_CFD config.cfg

    Parameters
    ----------
    n_procs : int
        Number of MPI processes (default: 1 = no MPI).
    n_threads : int
        Number of OpenMP threads per process (default: 1 = no threading).
        Your 8-core CPU can use up to 7 threads effectively.
    """
    result = {
        "case_dir": str(case_dir),
        "config": str(config_file),
        "converged": False,
        "iterations": 0,
        "wall_time_s": 0.0,
        "CL": None,
        "CD": None,
        "CM": None,
        "error": None,
        "parallelization": "serial",
    }

    # Build command with parallelization
    su2_exe = shutil.which("SU2_CFD") or shutil.which("SU2_CFD.exe") or "SU2_CFD"
    cmd = [su2_exe]

    # Add OpenMP threads (SU2 v8+ supports -t flag)
    if n_threads > 1:
        cmd.extend(["-t", str(n_threads)])
        result["parallelization"] = f"OpenMP ({n_threads} threads)"

    cmd.append(str(config_file.name))

    # Wrap with MPI if requested
    if n_procs > 1:
        # Windows uses mpiexec, Linux uses mpirun
        mpi_cmd = shutil.which("mpiexec") or shutil.which("mpirun")
        if mpi_cmd:
            cmd = [mpi_cmd, "-np", str(n_procs)] + cmd
            result["parallelization"] = (
                f"MPI ({n_procs} procs"
                + (f" × {n_threads} threads)" if n_threads > 1 else ")")
            )
        else:
            print(f"  [WARN]   MPI not found, running serial")

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
            result["error"] = f"SU2_CFD exited with code {proc.returncode}"
            return result

        # Parse history for CL, CD, CM
        history = parse_su2_history(case_dir)
        if history:
            result["CL"] = history.get("CL")
            result["CD"] = history.get("CD")
            result["CM"] = history.get("CM")
            result["iterations"] = history.get("iterations", 0)
            result["converged"] = True

    except subprocess.TimeoutExpired:
        result["error"] = f"Timeout after {timeout}s"
    except FileNotFoundError:
        result["error"] = "SU2_CFD not found"
    except Exception as e:
        result["error"] = str(e)

    return result


def parse_su2_history(case_dir: Path) -> Optional[Dict]:
    """Parse SU2 history.csv for final CL, CD values."""
    history_file = case_dir / "history.csv"
    if not history_file.exists():
        # Try .dat extension
        history_file = case_dir / "history.dat"
    if not history_file.exists():
        return None

    try:
        import csv
        csv.field_size_limit(10 * 1024 * 1024)  # 10 MB limit
        with open(history_file, 'r') as f:
            reader = csv.reader(f)
            headers = None
            last_row = None
            for row in reader:
                if not row:
                    continue
                if row[0].strip().startswith('"') or 'Inner_Iter' in row[0] or 'Time_Iter' in row[0] or 'Iteration' in row[0]:
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
                    "CM": data.get("CMz", data.get("Moment_z",
                                   data.get("Moment", None))),
                    "iterations": int(data.get("Inner_Iter",
                                      data.get("Iteration", 0))),
                }
    except Exception as e:
        print(f"  [WARN]   Could not parse history: {e}")
    return None


# ============================================================================
# Dry-Run (Setup Only)
# ============================================================================

def setup_case(alpha: float, model: str, grid: str,
                runs_dir: Path, grids_dir: Path, n_iter: int = 30000,
                restart: bool = False) -> Path:
    """Set up a single simulation case directory."""
    case_name = f"alpha_{alpha:.1f}_{model}_{grid}"
    case_dir = runs_dir / case_name
    case_dir.mkdir(parents=True, exist_ok=True)

    grid_cfg = GRID_LEVELS[grid]
    dims = grid_cfg["dims"]
    family = grid_cfg.get("family", "I")

    # Output SU2 mesh — include family in name to avoid collisions
    su2_name = f"naca0012_fam{family}_{dims[0]}x{dims[1]}.su2"
    su2_mesh = grids_dir / su2_name
    if not su2_mesh.exists():
        # Locate PLOT3D source file
        data_dir = grids_dir.parent  # experimental_data/naca0012
        plot3d_file = data_dir / grid_cfg["plot3d"]

        # Fallback: try standalone grid file in grids_dir
        if not plot3d_file.exists() and "plot3d_alt" in grid_cfg:
            plot3d_file = grids_dir / grid_cfg["plot3d_alt"]

        if not plot3d_file.exists():
            print(f"  [WARN]   Grid file not found: {grid_cfg['plot3d']}")
            print(f"           Download grids from TMR website")
            return case_dir

        from scripts.preprocessing.plot3d_to_su2 import (
            read_plot3d_2d, identify_boundaries, convert_to_su2, coarsen_grid
        )
        x, y, idim, jdim = read_plot3d_2d(plot3d_file)

        # Family III: coarsen from finest level
        coarsen_steps = grid_cfg.get("coarsen_steps", 0)
        if coarsen_steps > 0:
            x, y, idim, jdim = coarsen_grid(x, y, coarsen_steps)

        boundaries = identify_boundaries(x, y, idim, jdim,
                                          grid_cfg["airfoil_pts"])
        convert_to_su2(x, y, idim, jdim, su2_mesh, boundaries)

    # Copy mesh to case directory
    mesh_in_case = case_dir / "mesh.su2"
    if not mesh_in_case.exists():
        shutil.copy2(su2_mesh, mesh_in_case)

    # Generate SU2 config
    generate_su2_config(case_dir, "mesh.su2", alpha, model, n_iter=n_iter,
                         restart=restart)

    return case_dir


# ============================================================================
# Validation (Post-Processing)
# ============================================================================

def validate_results(runs_dir: Path, alphas: List[float], model: str,
                      grid: str) -> Dict:
    """
    Validate simulation results against TMR reference data.

    Quantities of interest (QoI) from TMR:
    - CL at alpha = 0, 10, 15 deg
    - CD at alpha = 0, 10, 15 deg
    - CM at alpha = 0, 10, 15 deg (from SU2 output; not in TMR consensus)
    - Cp vs. x/c at alpha = 0, 10, 15 deg (plotted separately)
    - Cf vs. x/c at alpha = 0, 10, 15 deg (plotted separately)
    """
    data_dir = PROJECT_ROOT / "experimental_data" / "naca0012"
    csv_dir = data_dir / "csv"

    # Load TMR reference
    ref_file = csv_dir / "tmr_sa_reference.json"
    if ref_file.exists():
        with open(ref_file) as f:
            tmr_ref = json.load(f)
    else:
        tmr_ref = None

    # Load Ladson experimental data
    ladson_file = csv_dir / "ladson_forces.csv"
    ladson_data = None
    if ladson_file.exists():
        import csv as csv_mod
        with open(ladson_file) as f:
            reader = csv_mod.DictReader(f)
            ladson_data = {"alpha": [], "CL": [], "CD": []}
            for row in reader:
                ladson_data["alpha"].append(float(row["alpha"]))
                ladson_data["CL"].append(float(row["CL"]))
                if "CD" in row and row["CD"]:
                    ladson_data["CD"].append(float(row["CD"]))

    # Collect simulation results
    results = {"alphas": [], "CL": [], "CD": [], "CM": [], "errors": {}}

    for alpha in alphas:
        case_name = f"alpha_{alpha:.1f}_{model}_{grid}"
        case_dir = runs_dir / case_name
        history = parse_su2_history(case_dir)

        if history and history["CL"] is not None:
            results["alphas"].append(alpha)
            results["CL"].append(history["CL"])
            results["CD"].append(history["CD"])
            results["CM"].append(history.get("CM"))
        else:
            print(f"  [SKIP]   No results for alpha={alpha}")

    # Compare against TMR reference
    if tmr_ref and results["alphas"]:
        print("\n  CFL3D Reference Comparison (QoI: CL, CD, CM):")
        print("  " + "-" * 76)
        print(f"  {'alpha':>6s}  {'CL_sim':>8s}  {'CL_ref':>8s}  {'err%':>6s}  "
              f"{'CD_sim':>10s}  {'CD_ref':>10s}  {'err%':>6s}  "
              f"{'CM_sim':>8s}")
        print("  " + "-" * 76)

        for i, alpha in enumerate(results["alphas"]):
            ref_key = f"alpha_{int(alpha)}"
            if ref_key in tmr_ref:
                ref = tmr_ref[ref_key]
                cl_sim = results["CL"][i]
                cd_sim = results["CD"][i]
                cm_sim = results["CM"][i]
                cl_ref = ref["CL"]
                cd_ref = ref["CD"]

                cd_err = abs(cd_sim - cd_ref) / cd_ref * 100
                if abs(cl_ref) > 1e-6:
                    cl_err = abs(cl_sim - cl_ref) / abs(cl_ref) * 100
                    cl_err_str = f"{cl_err:>5.1f}%"
                else:
                    cl_err = abs(cl_sim - cl_ref)
                    cl_err_str = f" ~0 (delta={cl_err:.1e})"

                results["errors"][alpha] = {
                    "CL_error_pct": cl_err, "CD_error_pct": cd_err
                }

                cm_str = f"{cm_sim:>8.4f}" if cm_sim is not None else "    N/A "
                print(f"  {alpha:>6.1f}  {cl_sim:>8.4f}  {cl_ref:>8.4f}  "
                      f"{cl_err_str}  {cd_sim:>10.5f}  {cd_ref:>10.5f}  "
                      f"{cd_err:>5.1f}%  {cm_str}")

        # Note about QoI
        print(f"\n  Note: CL/CD ref from 7-code SA consensus on 897x257 grid (TMR).")

        # Show grid-specific reference if available (for grid convergence context)
        grid_spec = GRID_LEVEL_SPECS.get(grid, {})
        grid_dims = grid_spec.get("dims", (0, 0))
        grid_key = f"{grid_dims[0]}x{grid_dims[1]}"
        gc_key = "grid_convergence_alpha10_familyII_cfl3d"
        if gc_key in tmr_ref:
            gc_data = tmr_ref[gc_key]["data"]
            match = [g for g in gc_data if g["grid"] == grid_key]
            if match:
                g = match[0]
                print(f"        Grid-level ref (CFL3D, {grid_key}, α=10°): "
                      f"CL={g['CL']:.4f}, CD={g['CD']:.5f}")
                # Compare SU2 vs CFL3D on same grid
                for i, alpha in enumerate(results["alphas"]):
                    if abs(alpha - 10.0) < 0.5:
                        cd_grid_err = abs(results["CD"][i] - g["CD"]) / g["CD"] * 100
                        cl_grid_err = abs(results["CL"][i] - g["CL"]) / g["CL"] * 100
                        print(f"        SU2 vs CFL3D same-grid: "
                              f"CL err={cl_grid_err:.1f}%, CD err={cd_grid_err:.1f}%")

        print("        CM not in TMR consensus; shown for completeness.")
        print("        Cp and Cf distributions: use --validate-only with plots.")

    return results


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="NACA 0012 TMR Validation — End-to-End Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python run_naca0012.py --dry-run                    Setup only (no solver needed)
  python run_naca0012.py --alpha 0 10 15 --model SA   Run SA at 3 angles
  python run_naca0012.py -t 7 --alpha 0 10 15         Run with 7 OpenMP threads (FAST)
  python run_naca0012.py --validate-only              Post-process existing results
  python run_naca0012.py --grid fine --model SA SST   Run both models on fine grid
  python run_naca0012.py --grid-family III --grid fine Use Family III grids

Parallelization (SU2 v8+):
  -t N    Use N OpenMP threads (recommended: N = CPU_cores - 1)
  --n-procs N   Use N MPI processes (requires mpiexec)
"""
    )
    parser.add_argument("--alpha", nargs="+", type=float, default=DEFAULT_ALPHAS,
                        help="Angles of attack (default: 0 5 10 13 15; TMR QoI: 0 10 15)")
    parser.add_argument("--model", nargs="+", default=["SA"],
                        choices=list(SU2_MODELS.keys()),
                        help="Turbulence models (default: SA)")
    parser.add_argument("--grid", default="medium",
                        choices=list(GRID_LEVEL_SPECS.keys()),
                        help="Grid level (default: medium)")
    parser.add_argument("--grid-family", default="I", choices=["I", "II", "III"],
                        help="TMR grid family: I, II, or III (default: I)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Setup cases without running solver")
    parser.add_argument("--validate-only", action="store_true",
                        help="Only validate existing results")
    parser.add_argument("--n-procs", type=int, default=1,
                        help="Number of MPI processes (requires mpiexec)")
    parser.add_argument("--n-threads", "-t", type=int, default=1,
                        help="OpenMP threads per process (SU2 v8+, default: 1)")
    parser.add_argument("--n-iter", type=int, default=30000,
                        help="Max iterations per case (30K default for deep convergence)")
    parser.add_argument("--timeout", type=int, default=14400,
                        help="Timeout in seconds per case (default: 14400 = 4 hours)")
    parser.add_argument("--restart", action="store_true",
                        help="Restart from existing solution (RESTART_SOL=YES)")
    parser.add_argument("--runs-dir", type=Path, default=None,
                        help="Output directory for runs")
    args = parser.parse_args()

    # Rebuild GRID_LEVELS for the selected family
    global GRID_LEVELS
    GRID_LEVELS = build_grid_levels(args.grid_family)

    print("=" * 70)
    print("  NACA 0012 TMR VALIDATION CASE (2DN00)")
    print("  NASA Langley Turbulence Modeling Resource")
    print("=" * 70)
    print(f"\n  Mach = {CASE_CONFIG['mach']}, Re = {CASE_CONFIG['reynolds']:.0e}")
    print(f"  Grid: {args.grid} ({GRID_LEVELS[args.grid]['dims'][0]}x"
          f"{GRID_LEVELS[args.grid]['dims'][1]}) — Family {args.grid_family}")
    print(f"  Models: {', '.join(args.model)}")
    print(f"  Alphas: {', '.join(f'{a:.1f}' for a in args.alpha)}")

    runs_dir = args.runs_dir or PROJECT_ROOT / "runs" / "naca0012"
    runs_dir.mkdir(parents=True, exist_ok=True)
    grids_dir = PROJECT_ROOT / "experimental_data" / "naca0012" / "grids"

    # --- Step 1: Download data if needed ---
    data_dir = PROJECT_ROOT / "experimental_data" / "naca0012"
    if not (data_dir / "csv" / "tmr_sa_reference.json").exists():
        print("\n  [STEP 1] Downloading TMR data...")
        sys.path.insert(0, str(data_dir))
        from experimental_data.naca0012.naca0012_tmr_data import (
            download_all_data, export_all_csv
        )
        download_all_data(data_dir)
        export_all_csv(data_dir)
    else:
        print("\n  [STEP 1] TMR data already downloaded [OK]")

    # --- Step 2: Validate only? ---
    if args.validate_only:
        print("\n  [VALIDATE] Post-processing existing results...")
        for model in args.model:
            print(f"\n  --- Model: {model} ---")
            validate_results(runs_dir, args.alpha, model, args.grid)
        return

    # --- Step 3: Setup cases ---
    print(f"\n  [STEP 2] Setting up {len(args.alpha) * len(args.model)} cases...")

    case_dirs = {}
    for model in args.model:
        for alpha in args.alpha:
            key = (alpha, model)
            case_dir = setup_case(alpha, model, args.grid, runs_dir, grids_dir,
                                   n_iter=args.n_iter, restart=args.restart)
            case_dirs[key] = case_dir
            print(f"  [OK]     {case_dir.name}")

    if args.dry_run:
        print("\n  [DRY-RUN] Cases set up. No solver executed.")
        print(f"\n  Case directories created in: {runs_dir}")
        print(f"\n  To run manually:")
        for (alpha, model), case_dir in case_dirs.items():
            print(f"    cd {case_dir}")
            if args.n_threads > 1:
                print(f"    SU2_CFD -t {args.n_threads} naca0012.cfg")
            else:
                print(f"    SU2_CFD naca0012.cfg")
        print(f"\n  Recommended (use all {os.cpu_count() - 1} threads):")
        print(f"    python run_naca0012.py --alpha {' '.join(f'{a:.1f}' for a in args.alpha)} "
              f"--model {' '.join(args.model)} -t {os.cpu_count() - 1}")
        print(f"\n  After running, validate with:")
        print(f"    python run_naca0012.py --validate-only "
              f"--alpha {' '.join(f'{a:.1f}' for a in args.alpha)} "
              f"--model {' '.join(args.model)}")
        return

    # --- Step 4: Run simulations ---
    if not check_solver():
        print("\n  [!!]     Cannot run without SU2. Use --dry-run to set up cases.")
        print("           Then run SU2_CFD manually or install SU2.")
        return

    n_threads = args.n_threads
    n_procs = args.n_procs
    if n_threads > 1 or n_procs > 1:
        print(f"\n  Parallelization: ", end="")
        if n_threads > 1:
            print(f"OpenMP {n_threads} threads", end="")
        if n_procs > 1:
            print(f"{' + ' if n_threads > 1 else ''}MPI {n_procs} processes", end="")
        print()

    print(f"\n  [STEP 3] Running simulations...")
    all_results = {}
    for (alpha, model), case_dir in case_dirs.items():
        config = case_dir / "naca0012.cfg"
        print(f"\n  --- alpha={alpha}, model={model} ---")
        result = run_simulation(
            case_dir, config,
            n_procs=n_procs, n_threads=n_threads,
            timeout=args.timeout
        )
        all_results[(alpha, model)] = result

        if result["error"]:
            print(f"  [FAIL]   {result['error']}")
        elif result.get("CL") is not None and result.get("CD") is not None:
            cm_str = f", CM={result['CM']:.4f}" if result.get("CM") is not None else ""
            print(f"  [OK]     CL={result['CL']:.4f}, CD={result['CD']:.5f}"
                  f"{cm_str}, time={result['wall_time_s']:.1f}s "
                  f"({result['parallelization']})")
        else:
            print(f"  [WARN]   Completed but could not parse CL/CD")

    # --- Step 5: Validate ---
    print(f"\n  [STEP 4] Validation against CFL3D reference...")
    for model in args.model:
        print(f"\n  --- Model: {model} ---")
        validate_results(runs_dir, args.alpha, model, args.grid)

    # Save results summary
    summary_file = runs_dir / "results_summary.json"
    summary = {
        "case": "NACA 0012 TMR 2DN00",
        "grid": args.grid,
        "grid_family": args.grid_family,
        "grid_dims": GRID_LEVELS[args.grid]["dims"],
        "conditions": CASE_CONFIG,
        "parallelization": {
            "n_threads": n_threads,
            "n_procs": n_procs,
        },
    }
    for (alpha, model), result in all_results.items():
        key = f"alpha_{alpha:.1f}_{model}"
        summary[key] = {
            "CL": result["CL"],
            "CD": result["CD"],
            "CM": result.get("CM"),
            "converged": result["converged"],
            "wall_time_s": result["wall_time_s"],
            "parallelization": result.get("parallelization", "serial"),
        }
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Results saved: {summary_file}")


if __name__ == "__main__":
    main()

