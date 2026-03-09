#!/usr/bin/env python3
"""
ERCOFTAC T3 Flat Plate Transition Benchmark Runner
=====================================================
Validates the Langtry-Menter gamma-Re_theta transition model (coupled with
SST) against the ERCOFTAC T3A, T3B, and T3A- experiments.

These benchmark cases feature zero pressure gradient flat plates with varying
freestream turbulence intensity (Tu), triggering bypass transition where the
Tollmien-Schlichting instability mechanism is circumvented by high-energy
freestream disturbances.

The gamma-Re_theta model solves two additional transport equations:
  1. Intermittency (gamma): triggers transition locally, controls turbulent
     production term activation
  2. Transition onset momentum-thickness Reynolds number (Re_theta_t):
     captures non-local influence of freestream conditions on the BL

Physics
-------
  - T3A  (Tu=3.3%):  Moderate bypass transition, Re_x,tr ~ 130,000
  - T3B  (Tu=6.5%):  Strong bypass transition,   Re_x,tr ~  60,000
  - T3A- (Tu=0.87%): Natural/weak bypass,         Re_x,tr ~ 500,000

  All cases: M < 0.06 -> incompressible solver (INC_RANS)

References
----------
  - Langtry & Menter (2009), AIAA J. 47(12), pp. 2894-2906
  - Savill (1993), ERCOFTAC Transition SIG report
  - Abu-Ghannam & Shaw (1980), J. Mech. Eng. Sci. 22(5), pp. 213-228
  - Rolls-Royce Applied Science Division (ERCOFTAC T3 data)

Usage
-----
    python run_transition_t3.py --case T3A --model LM [--dry-run]
    python run_transition_t3.py --comparison       # Run all cases + models
"""

import argparse
import json
import math
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# ERCOFTAC T3 Case Definitions
# =============================================================================

@dataclass
class T3CaseConfig:
    """Configuration for an ERCOFTAC T3 flat plate case."""
    name: str
    description: str
    Tu_percent: float              # Freestream turbulence intensity (%)
    U_inf: float                   # Freestream velocity (m/s)
    rho: float                     # Density (kg/m^3)
    mu: float                      # Dynamic viscosity (Pa.s)
    L_plate: float                 # Plate length (m)
    transition_mechanism: str      # "bypass", "strong_bypass", "natural"
    Re_x_transition_exp: float     # Experimental transition onset Re_x
    # Digitized experimental Cf(Re_x) data [Savill 1993 compilation]
    exp_Rex: List[float] = field(default_factory=list)
    exp_Cf: List[float] = field(default_factory=list)

    @property
    def nu(self) -> float:
        """Kinematic viscosity."""
        return self.mu / self.rho

    @property
    def Re_per_meter(self) -> float:
        """Reynolds number per meter."""
        return self.U_inf * self.rho / self.mu

    @property
    def Re_L(self) -> float:
        """Reynolds number based on plate length."""
        return self.Re_per_meter * self.L_plate

    @property
    def Tu_fraction(self) -> float:
        """Turbulence intensity as fraction (not percent)."""
        return self.Tu_percent / 100.0

    @property
    def Mach(self) -> float:
        """Approximate Mach number (air at 300K)."""
        a = math.sqrt(1.4 * 287.0 * 300.0)
        return self.U_inf / a


# Digitized experimental data from Savill (1993) / ERCOFTAC database
# Rex values and corresponding Cf values
T3_CASES: Dict[str, T3CaseConfig] = {
    "T3A": T3CaseConfig(
        name="T3A",
        description="ERCOFTAC T3A: Moderate bypass transition (Tu=3.3%)",
        Tu_percent=3.3,
        U_inf=5.4,
        rho=1.2,
        mu=1.8e-5,
        L_plate=1.5,
        transition_mechanism="bypass",
        Re_x_transition_exp=130_000,
        exp_Rex=[
            2.0e4, 3.5e4, 5.0e4, 7.0e4, 8.5e4, 1.0e5,
            1.15e5, 1.3e5, 1.5e5, 1.7e5, 2.0e5, 2.5e5,
            3.0e5, 3.5e5, 4.0e5, 4.5e5, 5.0e5,
        ],
        exp_Cf=[
            4.60e-3, 3.50e-3, 3.00e-3, 2.50e-3, 2.30e-3, 2.10e-3,
            2.15e-3, 2.50e-3, 3.20e-3, 3.90e-3, 4.30e-3, 4.50e-3,
            4.30e-3, 4.10e-3, 3.95e-3, 3.80e-3, 3.65e-3,
        ],
    ),
    "T3B": T3CaseConfig(
        name="T3B",
        description="ERCOFTAC T3B: Strong bypass transition (Tu=6.5%)",
        Tu_percent=6.5,
        U_inf=9.4,
        rho=1.2,
        mu=1.8e-5,
        L_plate=1.5,
        transition_mechanism="strong_bypass",
        Re_x_transition_exp=60_000,
        exp_Rex=[
            1.0e4, 2.0e4, 3.0e4, 4.0e4, 5.0e4, 6.0e4,
            7.0e4, 8.0e4, 1.0e5, 1.2e5, 1.5e5, 2.0e5,
            2.5e5, 3.0e5, 4.0e5, 5.0e5, 6.0e5,
        ],
        exp_Cf=[
            6.50e-3, 4.60e-3, 3.80e-3, 3.40e-3, 3.10e-3, 3.00e-3,
            3.20e-3, 3.60e-3, 4.20e-3, 4.50e-3, 4.60e-3, 4.50e-3,
            4.30e-3, 4.10e-3, 3.85e-3, 3.65e-3, 3.50e-3,
        ],
    ),
    "T3A-": T3CaseConfig(
        name="T3A-",
        description="ERCOFTAC T3A-: Natural / weak bypass transition (Tu=0.87%)",
        Tu_percent=0.87,
        U_inf=19.8,
        rho=1.2,
        mu=1.8e-5,
        L_plate=2.0,
        transition_mechanism="natural",
        Re_x_transition_exp=500_000,
        exp_Rex=[
            5.0e4, 1.0e5, 1.5e5, 2.0e5, 2.5e5, 3.0e5,
            3.5e5, 4.0e5, 4.5e5, 5.0e5, 5.5e5, 6.0e5,
            7.0e5, 8.0e5, 1.0e6, 1.2e6, 1.5e6,
        ],
        exp_Cf=[
            2.95e-3, 2.10e-3, 1.72e-3, 1.49e-3, 1.33e-3, 1.22e-3,
            1.13e-3, 1.05e-3, 9.90e-4, 9.40e-4, 1.05e-3, 1.80e-3,
            3.50e-3, 3.90e-3, 3.80e-3, 3.60e-3, 3.40e-3,
        ],
    ),
}


# =============================================================================
# Analytical Correlations
# =============================================================================

def cf_blasius(Re_x: np.ndarray) -> np.ndarray:
    """
    Blasius laminar flat plate skin friction.
    Cf = 0.664 / sqrt(Re_x)
    """
    return 0.664 / np.sqrt(np.maximum(Re_x, 1.0))


def cf_turbulent_power(Re_x: np.ndarray) -> np.ndarray:
    """
    Turbulent flat plate Cf — 1/5th power law.
    Cf = 0.0592 / Re_x^(1/5)
    Valid for 5e5 < Re_x < 1e7.
    """
    return 0.0592 / np.power(np.maximum(Re_x, 1.0), 0.2)


def cf_turbulent_schlichting(Re_x: np.ndarray) -> np.ndarray:
    """
    Schlichting turbulent flat plate Cf.
    Cf = 0.370 / (log10(Re_x))^2.584
    """
    log_rex = np.log10(np.maximum(Re_x, 10.0))
    return 0.370 / np.power(log_rex, 2.584)


def re_theta_onset_abu_ghannam_shaw(Tu_percent: float) -> float:
    """
    Abu-Ghannam & Shaw (1980) correlation for transition onset.
    Re_theta,cr = 163 + exp(6.91 - Tu)  for Tu > 0.6%

    Parameters
    ----------
    Tu_percent : float
        Freestream turbulence intensity in percent.

    Returns
    -------
    Re_theta_cr : float
        Critical momentum-thickness Reynolds number.
    """
    Tu = Tu_percent
    if Tu > 0.6:
        return 163.0 + math.exp(6.91 - Tu)
    else:
        # Low-Tu limit (natural transition)
        return 163.0 + math.exp(6.91 - 0.6)


def re_x_from_re_theta(Re_theta: float) -> float:
    """
    Approximate Re_x from Re_theta using the Blasius relation:
    Re_theta = 0.664 * sqrt(Re_x)
    -> Re_x = (Re_theta / 0.664)^2
    """
    return (Re_theta / 0.664) ** 2


# =============================================================================
# SU2 Configuration Generator
# =============================================================================

def generate_su2_config_lm(
    case: T3CaseConfig,
    case_dir: Path,
    mesh_file: str = "grid.su2",
    n_iter: int = 15000,
) -> Path:
    """
    Generate SU2 config for Langtry-Menter transition model.

    Uses INC_RANS solver with SST + LM transition.
    """
    # Turbulent viscosity ratio (approximate for Tu > 1%)
    # mu_t/mu ~ (Tu/100)^2 * Re_L * 0.01 (rough estimate)
    # More precisely, for low-Tu freestream: mu_t/mu ~ 1-10
    turb2lam_ratio = max(1.0, min(100.0, (case.Tu_percent / 1.0) ** 2 * 10))

    config = f"""\
% ============================================================
% ERCOFTAC {case.name} — Flat Plate Bypass Transition
% Tu = {case.Tu_percent}%, U_inf = {case.U_inf} m/s
% Langtry-Menter gamma-Re_theta model + SST
% ============================================================

% --- Physical model ---
SOLVER = INC_RANS
KIND_TURB_MODEL = SST
KIND_TRANS_MODEL = LM
MATH_PROBLEM = DIRECT

% --- Incompressible freestream ---
INC_DENSITY_INIT = {case.rho}
INC_VELOCITY_INIT = ( {case.U_inf}, 0.0, 0.0 )
INC_DENSITY_MODEL = CONSTANT
INC_ENERGY_EQUATION = NO

% --- Viscosity ---
VISCOSITY_MODEL = CONSTANT_VISCOSITY
MU_CONSTANT = {case.mu}

% --- Freestream turbulence (CRITICAL for transition model) ---
FREESTREAM_TURBULENCEINTENSITY = {case.Tu_fraction}
FREESTREAM_TURB2LAMVISCRATIO = {turb2lam_ratio:.1f}

% --- Reference values ---
REF_AREA = {case.L_plate}
REF_LENGTH = {case.L_plate}

% --- Boundary conditions ---
MARKER_HEATFLUX = ( wall, 0.0 )
MARKER_SYM = ( symmetry )
INC_INLET_TYPE = VELOCITY_INLET
MARKER_INLET = ( inlet, 300.0, {case.U_inf}, 1.0, 0.0, 0.0 )
INC_OUTLET_TYPE = PRESSURE_OUTLET
MARKER_OUTLET = ( outlet, 0.0 )
MARKER_PLOTTING = ( wall )
MARKER_MONITORING = ( wall )

% --- Numerical methods ---
NUM_METHOD_GRAD = GREEN_GAUSS
CFL_NUMBER = 50.0
CFL_ADAPT = YES
CFL_ADAPT_PARAM = ( 0.5, 1.5, 1.0, 100.0 )

CONV_NUM_METHOD_FLOW = FDS
MUSCL_FLOW = YES

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
CONV_RESIDUAL_MINVAL = -12
CONV_STARTITER = 10
CONV_FIELD = RMS_PRESSURE

% --- Input/Output ---
MESH_FILENAME = {mesh_file}
MESH_FORMAT = SU2
SOLUTION_FILENAME = solution.dat
RESTART_FILENAME = restart.dat
CONV_FILENAME = history
VOLUME_FILENAME = flow
SURFACE_FILENAME = surface_flow

OUTPUT_FILES = (RESTART, PARAVIEW, SURFACE_PARAVIEW, SURFACE_CSV)
OUTPUT_WRT_FREQ = 1000
SCREEN_OUTPUT = (INNER_ITER, RMS_PRESSURE, RMS_VELOCITY-X, RMS_TKE, RMS_INTERMITTENCY, DRAG, LIFT)
HISTORY_OUTPUT = (ITER, RMS_RES, FLOW_COEFF)
"""

    case_dir.mkdir(parents=True, exist_ok=True)
    config_path = case_dir / f"{case.name}_LM.cfg"
    config_path.write_text(config)
    return config_path


def generate_su2_config_bcm(
    case: T3CaseConfig,
    case_dir: Path,
    mesh_file: str = "grid.su2",
    n_iter: int = 15000,
) -> Path:
    """
    Generate SU2 config for Bas-Cakmakcioglu (BCM) algebraic transition model.

    Uses INC_RANS solver with SA + BCM transition (no extra transport equations).
    """
    config = f"""\
% ============================================================
% ERCOFTAC {case.name} — Flat Plate Bypass Transition
% Tu = {case.Tu_percent}%, U_inf = {case.U_inf} m/s
% Bas-Cakmakcioglu algebraic transition model + SA
% ============================================================

% --- Physical model ---
SOLVER = INC_RANS
KIND_TURB_MODEL = SA
SA_OPTIONS = BCM
MATH_PROBLEM = DIRECT

% --- Incompressible freestream ---
INC_DENSITY_INIT = {case.rho}
INC_VELOCITY_INIT = ( {case.U_inf}, 0.0, 0.0 )
INC_DENSITY_MODEL = CONSTANT
INC_ENERGY_EQUATION = NO

% --- Viscosity ---
VISCOSITY_MODEL = CONSTANT_VISCOSITY
MU_CONSTANT = {case.mu}

% --- Freestream turbulence ---
FREESTREAM_TURBULENCEINTENSITY = {case.Tu_fraction}

% --- Reference values ---
REF_AREA = {case.L_plate}
REF_LENGTH = {case.L_plate}

% --- Boundary conditions ---
MARKER_HEATFLUX = ( wall, 0.0 )
MARKER_SYM = ( symmetry )
INC_INLET_TYPE = VELOCITY_INLET
MARKER_INLET = ( inlet, 300.0, {case.U_inf}, 1.0, 0.0, 0.0 )
INC_OUTLET_TYPE = PRESSURE_OUTLET
MARKER_OUTLET = ( outlet, 0.0 )
MARKER_PLOTTING = ( wall )
MARKER_MONITORING = ( wall )

% --- Numerical methods ---
NUM_METHOD_GRAD = GREEN_GAUSS
CFL_NUMBER = 50.0
CFL_ADAPT = YES
CFL_ADAPT_PARAM = ( 0.5, 1.5, 1.0, 100.0 )

CONV_NUM_METHOD_FLOW = FDS
MUSCL_FLOW = YES

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
CONV_RESIDUAL_MINVAL = -12
CONV_STARTITER = 10
CONV_FIELD = RMS_PRESSURE

% --- Input/Output ---
MESH_FILENAME = {mesh_file}
MESH_FORMAT = SU2
SOLUTION_FILENAME = solution.dat
RESTART_FILENAME = restart.dat
CONV_FILENAME = history
VOLUME_FILENAME = flow
SURFACE_FILENAME = surface_flow

OUTPUT_FILES = (RESTART, PARAVIEW, SURFACE_PARAVIEW, SURFACE_CSV)
OUTPUT_WRT_FREQ = 1000
SCREEN_OUTPUT = (INNER_ITER, RMS_PRESSURE, RMS_VELOCITY-X, RMS_NU_TILDE, DRAG, LIFT)
HISTORY_OUTPUT = (ITER, RMS_RES, FLOW_COEFF)
"""

    case_dir.mkdir(parents=True, exist_ok=True)
    config_path = case_dir / f"{case.name}_BCM.cfg"
    config_path.write_text(config)
    return config_path


def generate_su2_config_fully_turbulent(
    case: T3CaseConfig,
    case_dir: Path,
    mesh_file: str = "grid.su2",
    n_iter: int = 15000,
) -> Path:
    """
    Generate SU2 config for fully turbulent SST (no transition model).

    This serves as a comparison baseline to show the Cf overprediction
    in the laminar region when transition is ignored.
    """
    config = f"""\
% ============================================================
% ERCOFTAC {case.name} — Flat Plate (FULLY TURBULENT baseline)
% Tu = {case.Tu_percent}%, U_inf = {case.U_inf} m/s
% SST turbulence model — NO transition model
% ============================================================

% --- Physical model ---
SOLVER = INC_RANS
KIND_TURB_MODEL = SST
MATH_PROBLEM = DIRECT

% --- Incompressible freestream ---
INC_DENSITY_INIT = {case.rho}
INC_VELOCITY_INIT = ( {case.U_inf}, 0.0, 0.0 )
INC_DENSITY_MODEL = CONSTANT
INC_ENERGY_EQUATION = NO

% --- Viscosity ---
VISCOSITY_MODEL = CONSTANT_VISCOSITY
MU_CONSTANT = {case.mu}

% --- Freestream turbulence ---
FREESTREAM_TURBULENCEINTENSITY = {case.Tu_fraction}
FREESTREAM_TURB2LAMVISCRATIO = 10.0

% --- Reference values ---
REF_AREA = {case.L_plate}
REF_LENGTH = {case.L_plate}

% --- Boundary conditions ---
MARKER_HEATFLUX = ( wall, 0.0 )
MARKER_SYM = ( symmetry )
INC_INLET_TYPE = VELOCITY_INLET
MARKER_INLET = ( inlet, 300.0, {case.U_inf}, 1.0, 0.0, 0.0 )
INC_OUTLET_TYPE = PRESSURE_OUTLET
MARKER_OUTLET = ( outlet, 0.0 )
MARKER_PLOTTING = ( wall )
MARKER_MONITORING = ( wall )

% --- Numerical methods ---
NUM_METHOD_GRAD = GREEN_GAUSS
CFL_NUMBER = 50.0
CFL_ADAPT = YES
CFL_ADAPT_PARAM = ( 0.5, 1.5, 1.0, 100.0 )

CONV_NUM_METHOD_FLOW = FDS
MUSCL_FLOW = YES

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
CONV_RESIDUAL_MINVAL = -12
CONV_STARTITER = 10
CONV_FIELD = RMS_PRESSURE

% --- Input/Output ---
MESH_FILENAME = {mesh_file}
MESH_FORMAT = SU2
SOLUTION_FILENAME = solution.dat
RESTART_FILENAME = restart.dat
CONV_FILENAME = history
VOLUME_FILENAME = flow
SURFACE_FILENAME = surface_flow

OUTPUT_FILES = (RESTART, PARAVIEW, SURFACE_PARAVIEW, SURFACE_CSV)
OUTPUT_WRT_FREQ = 1000
SCREEN_OUTPUT = (INNER_ITER, RMS_PRESSURE, RMS_VELOCITY-X, RMS_TKE, DRAG, LIFT)
HISTORY_OUTPUT = (ITER, RMS_RES, FLOW_COEFF)
"""

    case_dir.mkdir(parents=True, exist_ok=True)
    config_path = case_dir / f"{case.name}_FullyTurbulent.cfg"
    config_path.write_text(config)
    return config_path


# =============================================================================
# Result Parsing
# =============================================================================

def parse_surface_cf(case_dir: Path) -> Optional[Dict[str, np.ndarray]]:
    """
    Parse Cf(x) from SU2 surface output.

    Returns dict with 'x', 'Cf', 'Rex' arrays or None if no output found.
    """
    csv_path = case_dir / "surface_flow.csv"
    if not csv_path.exists():
        return None

    try:
        import pandas as pd
        df = pd.read_csv(csv_path)

        # SU2 column names vary — try common variants
        x_col = None
        cf_col = None
        for col in df.columns:
            col_lower = col.strip().lower()
            if col_lower in ("x", "\"x\"", "points:0", "x_coord"):
                x_col = col
            if "skin_friction" in col_lower and ("x" in col_lower or "0" in col_lower):
                cf_col = col
            elif col_lower in ("cf", "cf_x", "skin_friction_coefficient"):
                cf_col = col

        if x_col is None or cf_col is None:
            # Fallback: try positional
            x_col = df.columns[0]
            for col in df.columns:
                if "friction" in col.lower():
                    cf_col = col
                    break
            if cf_col is None:
                return None

        x = df[x_col].values
        cf = np.abs(df[cf_col].values)  # Abs for consistent sign

        # Filter wall region (x > 0, Cf > 0)
        valid = (x > 0) & (cf > 0)
        x = x[valid]
        cf = cf[valid]

        # Sort by x
        sort_idx = np.argsort(x)
        x = x[sort_idx]
        cf = cf[sort_idx]

        return {"x": x, "Cf": cf}

    except Exception as e:
        print(f"  [WARN] Could not parse surface Cf: {e}")
        return None


def compute_rex(x: np.ndarray, U_inf: float, nu: float) -> np.ndarray:
    """Compute local Reynolds number Re_x = U_inf * x / nu."""
    return U_inf * x / nu


def find_transition_location(
    x: np.ndarray,
    Cf: np.ndarray,
    nu: float,
    U_inf: float,
) -> Optional[Dict[str, float]]:
    """
    Find the transition onset location from Cf(x) data.

    Transition onset is identified as the location where Cf reaches its
    minimum (laminar -> transitional), i.e. where the Cf curve departs
    upward from the Blasius correlation.

    Returns dict with 'x_tr', 'Rex_tr', 'Cf_min'.
    """
    if len(x) < 10:
        return None

    # Find Cf minimum (transition onset region)
    # Use smoothed Cf to avoid noise
    window = max(5, len(Cf) // 50)
    if window % 2 == 0:
        window += 1

    try:
        from scipy.signal import savgol_filter
        Cf_smooth = savgol_filter(Cf, window, 3)
    except ImportError:
        # Simple moving average
        kernel = np.ones(window) / window
        Cf_smooth = np.convolve(Cf, kernel, mode='same')

    # Find minimum in the Cf curve (transition onset)
    min_idx = np.argmin(Cf_smooth)
    x_tr = float(x[min_idx])
    Rex_tr = float(U_inf * x_tr / nu)
    Cf_min = float(Cf_smooth[min_idx])

    return {
        "x_tr": x_tr,
        "Rex_tr": Rex_tr,
        "Cf_min": Cf_min,
    }


# =============================================================================
# Validation Plotting
# =============================================================================

def plot_cf_vs_rex(
    case: T3CaseConfig,
    simulation_results: Dict[str, Dict],
    output_path: Path,
    show: bool = False,
):
    """
    Plot Cf vs Re_x for a single T3 case, comparing:
    - Experimental data (markers)
    - Simulation results (lines, one per model)
    - Blasius laminar correlation (dashed)
    - Turbulent correlation (dash-dot)
    - Abu-Ghannam & Shaw transition onset (vertical line)

    Parameters
    ----------
    case : T3CaseConfig
    simulation_results : dict mapping model name -> {'x': array, 'Cf': array}
    output_path : Path
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [SKIP] matplotlib not available, skipping plot")
        return

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Analytical correlations
    rex_range = np.logspace(3.5, 6.5, 500)
    ax.plot(rex_range, cf_blasius(rex_range), 'k--', linewidth=1.2,
            label='Blasius (laminar)', alpha=0.7)
    ax.plot(rex_range, cf_turbulent_power(rex_range), 'k-.', linewidth=1.2,
            label='1/5 power law (turb.)', alpha=0.7)

    # Abu-Ghannam & Shaw transition onset
    re_theta_cr = re_theta_onset_abu_ghannam_shaw(case.Tu_percent)
    rex_onset = re_x_from_re_theta(re_theta_cr)
    ax.axvline(rex_onset, color='gray', linestyle=':', linewidth=1.0,
               label=f'AG&S onset (Re_x={rex_onset:.0f})', alpha=0.6)

    # Experimental data
    if case.exp_Rex and case.exp_Cf:
        ax.scatter(case.exp_Rex, case.exp_Cf, marker='o', s=40,
                   facecolors='none', edgecolors='black', linewidth=1.2,
                   label=f'{case.name} exp. (Savill 1993)', zorder=5)

    # Simulation results
    colors = {'LM': '#E63946', 'BCM': '#457B9D', 'FullyTurbulent': '#2A9D8F'}
    labels = {'LM': 'SST + LM (gamma-Re_theta)',
              'BCM': 'SA + BCM (algebraic)',
              'FullyTurbulent': 'SST fully turbulent'}

    for model_name, data in simulation_results.items():
        if data is None:
            continue
        rex = compute_rex(data['x'], case.U_inf, case.nu)
        color = colors.get(model_name, '#666')
        label = labels.get(model_name, model_name)
        ax.plot(rex, data['Cf'], '-', color=color, linewidth=1.8,
                label=label, zorder=4)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$Re_x$', fontsize=13)
    ax.set_ylabel(r'$C_f$', fontsize=13)
    ax.set_title(f'ERCOFTAC {case.name} — Skin Friction Coefficient\n'
                 f'Tu = {case.Tu_percent}%, U = {case.U_inf} m/s, '
                 f'{case.transition_mechanism} transition',
                 fontsize=12)
    ax.legend(fontsize=9, loc='upper right', framealpha=0.9)
    ax.set_xlim(5e3, 2e6)
    ax.set_ylim(5e-4, 1.5e-2)
    ax.grid(True, which='both', alpha=0.3)
    ax.tick_params(labelsize=11)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=200, bbox_inches='tight')
    print(f"  Plot saved: {output_path}")

    if show:
        plt.show()
    plt.close(fig)


def plot_all_cases_comparison(
    results_by_case: Dict[str, Dict],
    model_name: str,
    output_path: Path,
):
    """
    Plot all T3 cases (T3A, T3B, T3A-) on a single figure for
    one model, showing how Tu affects transition location.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [SKIP] matplotlib not available")
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    case_order = ["T3B", "T3A", "T3A-"]
    colors = {'T3B': '#E63946', 'T3A': '#457B9D', 'T3A-': '#2A9D8F'}

    for idx, case_name in enumerate(case_order):
        ax = axes[idx]
        case = T3_CASES.get(case_name)
        if case is None:
            continue

        rex_range = np.logspace(3.5, 6.5, 500)
        ax.plot(rex_range, cf_blasius(rex_range), 'k--', linewidth=1.0,
                label='Blasius', alpha=0.6)
        ax.plot(rex_range, cf_turbulent_power(rex_range), 'k-.', linewidth=1.0,
                label='Turbulent', alpha=0.6)

        # Experimental
        if case.exp_Rex and case.exp_Cf:
            ax.scatter(case.exp_Rex, case.exp_Cf, marker='o', s=35,
                       facecolors='none', edgecolors='black', linewidth=1.0,
                       label='Experiment')

        # Simulation
        if case_name in results_by_case and results_by_case[case_name] is not None:
            data = results_by_case[case_name]
            rex = compute_rex(data['x'], case.U_inf, case.nu)
            ax.plot(rex, data['Cf'], '-', color=colors.get(case_name, 'blue'),
                    linewidth=1.8, label=model_name)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'$Re_x$', fontsize=12)
        if idx == 0:
            ax.set_ylabel(r'$C_f$', fontsize=12)
        ax.set_title(f'{case_name} (Tu={case.Tu_percent}%)', fontsize=11)
        ax.legend(fontsize=8)
        ax.set_xlim(5e3, 2e6)
        ax.set_ylim(5e-4, 1.5e-2)
        ax.grid(True, which='both', alpha=0.3)

    plt.suptitle(f'ERCOFTAC T3 Series — {model_name} Transition Model',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=200, bbox_inches='tight')
    print(f"  Multi-case plot saved: {output_path}")
    plt.close(fig)


# =============================================================================
# Main Runner
# =============================================================================

def run_single_case(
    case_name: str = "T3A",
    model: str = "LM",
    n_iter: int = 15000,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Run a single ERCOFTAC T3 transition case.

    Parameters
    ----------
    case_name : str
        "T3A", "T3B", or "T3A-"
    model : str
        "LM" (Langtry-Menter), "BCM" (Bas-Cakmakcioglu), or "FullyTurbulent"
    n_iter : int
        Number of iterations.
    dry_run : bool
        If True, only generate config.
    """
    case = T3_CASES.get(case_name)
    if case is None:
        print(f"  [ERROR] Unknown case: {case_name}")
        return {"error": f"Unknown case: {case_name}"}

    runs_dir = PROJECT_ROOT / "runs" / "transition_t3"
    case_dir = runs_dir / f"{case_name}_{model}"

    # Analytical predictions
    re_theta_cr = re_theta_onset_abu_ghannam_shaw(case.Tu_percent)
    rex_onset_pred = re_x_from_re_theta(re_theta_cr)

    print(f"\n{'='*65}")
    print(f"  ERCOFTAC {case.name}: {case.description}")
    print(f"  Model: {model} | Tu = {case.Tu_percent}% | U = {case.U_inf} m/s")
    print(f"  Re/m = {case.Re_per_meter:.0f} | Mach = {case.Mach:.4f}")
    print(f"  Predicted onset: Re_theta,cr = {re_theta_cr:.0f} "
          f"(Re_x ~ {rex_onset_pred:.0f})")
    print(f"  Experimental onset: Re_x ~ {case.Re_x_transition_exp:,.0f}")
    print(f"{'='*65}")

    # Generate config
    if model == "LM":
        config_path = generate_su2_config_lm(case, case_dir, n_iter=n_iter)
    elif model == "BCM":
        config_path = generate_su2_config_bcm(case, case_dir, n_iter=n_iter)
    elif model == "FullyTurbulent":
        config_path = generate_su2_config_fully_turbulent(case, case_dir, n_iter=n_iter)
    else:
        print(f"  [ERROR] Unknown model: {model}")
        return {"error": f"Unknown model: {model}"}

    print(f"  Config: {config_path}")

    result = {
        "case": case_name,
        "model": model,
        "Tu_percent": case.Tu_percent,
        "Re_per_meter": case.Re_per_meter,
        "Re_x_transition_exp": case.Re_x_transition_exp,
        "Re_x_transition_AGS": rex_onset_pred,
        "config_path": str(config_path),
        "converged": False,
    }

    if dry_run:
        print("  [DRY RUN] Config generated, simulation skipped")
        result["dry_run"] = True
        return result

    # Find SU2
    su2_exe = None
    for exe in ["SU2_CFD", "SU2_CFD.exe"]:
        try:
            subprocess.run([exe, "--version"], capture_output=True, timeout=5)
            su2_exe = exe
            break
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue

    if su2_exe is None:
        print("  [WARN] SU2_CFD not found. Config generated but simulation skipped.")
        result["error"] = "SU2 not found"
        return result

    # Run SU2
    print(f"  Running SU2 ({n_iter} iterations)...")
    try:
        proc = subprocess.run(
            [su2_exe, str(config_path)],
            cwd=str(case_dir),
            capture_output=True,
            text=True,
            timeout=7200,
        )
        if proc.returncode != 0:
            print(f"  [ERROR] SU2 returned {proc.returncode}")
            result["error"] = proc.stderr[-500:] if proc.stderr else "Unknown error"
            return result

    except subprocess.TimeoutExpired:
        result["error"] = "Timeout"
        return result

    result["converged"] = True

    # Parse results
    cf_data = parse_surface_cf(case_dir)
    if cf_data is not None:
        result["n_points"] = len(cf_data["x"])
        tr_info = find_transition_location(
            cf_data["x"], cf_data["Cf"], case.nu, case.U_inf
        )
        if tr_info:
            result["Re_x_transition_computed"] = tr_info["Rex_tr"]
            result["x_transition"] = tr_info["x_tr"]
            result["Cf_min"] = tr_info["Cf_min"]
            error_pct = (
                (tr_info["Rex_tr"] - case.Re_x_transition_exp) /
                case.Re_x_transition_exp * 100
            )
            result["transition_onset_error_pct"] = error_pct
            print(f"\n  Computed transition: Re_x = {tr_info['Rex_tr']:.0f}")
            print(f"  Experimental:       Re_x = {case.Re_x_transition_exp:,.0f}")
            print(f"  Error: {error_pct:+.1f}%")

        # Single-case plot
        plot_path = PROJECT_ROOT / "results" / "transition" / f"cf_{case_name}_{model}.png"
        plot_cf_vs_rex(case, {model: cf_data}, plot_path)
    else:
        print("  [WARN] No surface Cf data found")
        result["error"] = "No surface output"

    # Save results
    results_file = case_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"  Results saved: {results_file}")

    return result


def run_all_t3_cases(
    model: str = "LM",
    n_iter: int = 15000,
    dry_run: bool = False,
):
    """Run all ERCOFTAC T3 cases with a single model."""
    all_results = {}
    for case_name in ["T3A", "T3B", "T3A-"]:
        result = run_single_case(case_name, model, n_iter, dry_run)
        all_results[case_name] = result

    # Summary table
    print(f"\n{'='*70}")
    print(f"  ERCOFTAC T3 SERIES — {model} Transition Model Summary")
    print(f"{'='*70}")
    print(f"  {'Case':<6} {'Tu%':<6} {'Re_x,tr (exp)':<14} "
          f"{'Re_x,tr (CFD)':<14} {'Error%':<10}")
    print(f"  {'-'*50}")

    for case_name, result in all_results.items():
        rex_exp = result.get("Re_x_transition_exp", "—")
        rex_cfd = result.get("Re_x_transition_computed", "—")
        err = result.get("transition_onset_error_pct", "—")
        if isinstance(rex_exp, (int, float)):
            rex_exp = f"{rex_exp:,.0f}"
        if isinstance(rex_cfd, (int, float)):
            rex_cfd = f"{rex_cfd:,.0f}"
        if isinstance(err, (int, float)):
            err = f"{err:+.1f}%"
        print(f"  {case_name:<6} {T3_CASES[case_name].Tu_percent:<6} "
              f"{rex_exp:<14} {rex_cfd:<14} {err:<10}")

    # Save summary
    summary_path = PROJECT_ROOT / "results" / "transition" / f"t3_summary_{model}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Summary: {summary_path}")

    return all_results


def run_comparison(
    n_iter: int = 15000,
    dry_run: bool = False,
):
    """Run all T3 cases with LM, BCM, and fully turbulent for comparison."""
    models = ["LM", "BCM", "FullyTurbulent"]
    all_results = {}

    for model in models:
        print(f"\n{'#'*70}")
        print(f"  Running all T3 cases with {model}")
        print(f"{'#'*70}")
        all_results[model] = run_all_t3_cases(model, n_iter, dry_run)

    # Grand summary
    print(f"\n{'='*75}")
    print(f"  GRAND COMPARISON — All Models x All Cases")
    print(f"{'='*75}")
    for model in models:
        print(f"\n  {model}:")
        for case_name, result in all_results[model].items():
            err = result.get("transition_onset_error_pct", "N/A")
            if isinstance(err, (int, float)):
                err = f"{err:+.1f}%"
            print(f"    {case_name}: onset error = {err}")

    return all_results


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="ERCOFTAC T3 Flat Plate Transition Benchmark Runner"
    )
    parser.add_argument("--case", default="T3A", choices=["T3A", "T3B", "T3A-"],
                        help="ERCOFTAC T3 case (default: T3A)")
    parser.add_argument("--model", default="LM",
                        choices=["LM", "BCM", "FullyTurbulent"],
                        help="Transition model (default: LM)")
    parser.add_argument("--iter", type=int, default=15000,
                        help="Number of iterations (default: 15000)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Generate config only, don't run simulations")
    parser.add_argument("--all-cases", action="store_true",
                        help="Run all T3 cases with selected model")
    parser.add_argument("--comparison", action="store_true",
                        help="Run all cases x all models")
    args = parser.parse_args()

    if args.comparison:
        run_comparison(args.iter, args.dry_run)
    elif args.all_cases:
        run_all_t3_cases(args.model, args.iter, args.dry_run)
    else:
        run_single_case(args.case, args.model, args.iter, args.dry_run)


if __name__ == "__main__":
    main()
