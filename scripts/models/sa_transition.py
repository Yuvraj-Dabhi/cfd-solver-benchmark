#!/usr/bin/env python3
"""
SA-Based Transition Model Wrapper
===================================
Reusable SU2 configuration generators for SA-based transition models:
  - BCM (Bas-Cakmakcioglu algebraic transition)
  - SA-AFT (SA with Amplification Factor Transport)

Also provides a multi-model comparison orchestrator for side-by-side
evaluation of transition predictions (LM, BCM, fully-turbulent).

References
----------
  - Cakmakcioglu et al. (2018), AIAA J. 56(9), DOI:10.2514/1.J056467
  - Langtry & Menter (2009), AIAA J. 47(12), pp. 2894-2906
"""

import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class TransitionCaseConfig:
    """Configuration for a transition case."""
    name: str
    U_inf: float
    rho: float = 1.225
    mu: float = 1.7894e-5
    L_ref: float = 1.0
    Tu_percent: float = 3.0
    mach: float = 0.0
    alpha_deg: float = 0.0
    n_iter: int = 15000

    @property
    def nu(self) -> float:
        return self.mu / self.rho

    @property
    def Re_L(self) -> float:
        return self.U_inf * self.L_ref / self.nu


def generate_su2_config_bcm(
    case: TransitionCaseConfig,
    case_dir: Path,
    mesh_file: str = "grid.su2",
    n_iter: Optional[int] = None,
) -> Path:
    """
    Generate SU2 config for BCM algebraic transition model.

    Uses INC_RANS solver with SA + BCM transition trigger.
    No extra transport equations needed — uses local vorticity
    and strain rate to detect transition.

    Parameters
    ----------
    case : TransitionCaseConfig
        Case parameters.
    case_dir : Path
        Output directory for config.
    mesh_file : str
        Mesh filename.
    n_iter : int, optional
        Override iteration count.

    Returns
    -------
    Path to generated config file.
    """
    case_dir = Path(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)

    iters = n_iter or case.n_iter
    config_path = case_dir / "transition_bcm.cfg"

    config = f"""\
% ============================================================
% SU2 Configuration: SA-BCM Algebraic Transition
% Case: {case.name}
% ============================================================

% --- Problem Definition ---
SOLVER= INC_RANS
KIND_TURB_MODEL= SA
SA_OPTIONS= BCM

MATH_PROBLEM= DIRECT
RESTART_SOL= NO

% --- Freestream ---
INC_DENSITY_MODEL= CONSTANT
INC_DENSITY_INIT= {case.rho}
INC_VELOCITY_INIT= ({case.U_inf * np.cos(np.radians(case.alpha_deg)):.6f}, {case.U_inf * np.sin(np.radians(case.alpha_deg)):.6f}, 0.0)
INC_TEMPERATURE_INIT= 300.0
VISCOSITY_MODEL= CONSTANT
MU_CONSTANT= {case.mu:.6e}
FREESTREAM_TURBULENCEINTENSITY= {case.Tu_percent / 100.0:.4f}
FREESTREAM_NU_FACTOR= 3.0

% --- Reference ---
REF_LENGTH= {case.L_ref}
REYNOLDS_NUMBER= {case.Re_L:.0f}

% --- Boundary Conditions ---
MARKER_HEATFLUX= ( wall, 0.0 )
MARKER_FAR= ( farfield )
MARKER_INLET= ( inlet, 300.0, {case.U_inf:.4f}, 1.0, 0.0, 0.0 )
MARKER_OUTLET= ( outlet, 0.0 )

% --- Numerics ---
NUM_METHOD_GRAD= GREEN_GAUSS
CFL_NUMBER= 10.0
CFL_ADAPT= YES
CFL_ADAPT_PARAM= ( 0.1, 2.0, 10.0, 1e10 )

CONV_NUM_METHOD_FLOW= FDS
MUSCL_FLOW= YES
SLOPE_LIMITER_FLOW= VENKATAKRISHNAN
VENKAT_LIMITER_COEFF= 0.05

CONV_NUM_METHOD_TURB= SCALAR_UPWIND
MUSCL_TURB= NO

% --- Convergence ---
ITER= {iters}
CONV_RESIDUAL_MINVAL= -12
CONV_FIELD= RMS_PRESSURE, RMS_NU_TILDE
CONV_STARTITER= 100

% --- Output ---
MESH_FILENAME= {mesh_file}
OUTPUT_FILES= RESTART, PARAVIEW_MULTIBLOCK, SURFACE_PARAVIEW
VOLUME_OUTPUT= RESIDUAL, PRIMITIVE, TURBULENT
SURFACE_OUTPUT= SKIN_FRICTION, PRESSURE_COEFFICIENT
OUTPUT_WRT_FREQ= 500
CONV_FILENAME= history
RESTART_FILENAME= restart_flow.dat
SOLUTION_FILENAME= solution_flow.dat
"""

    config_path.write_text(config)
    return config_path


def generate_su2_config_sa_aft(
    case: TransitionCaseConfig,
    case_dir: Path,
    mesh_file: str = "grid.su2",
    n_iter: Optional[int] = None,
) -> Path:
    """
    Generate SU2 config for SA with Amplification Factor Transport.

    Uses SA base model with AFT transition model which solves one
    additional transport equation for the amplification factor.

    Parameters
    ----------
    case : TransitionCaseConfig
        Case parameters.
    case_dir : Path
        Output directory for config.
    mesh_file : str
        Mesh filename.
    n_iter : int, optional
        Override iteration count.

    Returns
    -------
    Path to generated config file.
    """
    case_dir = Path(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)

    iters = n_iter or case.n_iter
    config_path = case_dir / "transition_sa_aft.cfg"

    config = f"""\
% ============================================================
% SU2 Configuration: SA-AFT (Amplification Factor Transport)
% Case: {case.name}
% ============================================================

% --- Problem Definition ---
SOLVER= INC_RANS
KIND_TURB_MODEL= SA
KIND_TRANS_MODEL= LM
LM_OPTIONS= CROSSFLOW

MATH_PROBLEM= DIRECT
RESTART_SOL= NO

% --- Freestream ---
INC_DENSITY_MODEL= CONSTANT
INC_DENSITY_INIT= {case.rho}
INC_VELOCITY_INIT= ({case.U_inf * np.cos(np.radians(case.alpha_deg)):.6f}, {case.U_inf * np.sin(np.radians(case.alpha_deg)):.6f}, 0.0)
INC_TEMPERATURE_INIT= 300.0
VISCOSITY_MODEL= CONSTANT
MU_CONSTANT= {case.mu:.6e}
FREESTREAM_TURBULENCEINTENSITY= {case.Tu_percent / 100.0:.4f}
FREESTREAM_NU_FACTOR= 3.0

% --- Reference ---
REF_LENGTH= {case.L_ref}
REYNOLDS_NUMBER= {case.Re_L:.0f}

% --- Boundary Conditions ---
MARKER_HEATFLUX= ( wall, 0.0 )
MARKER_FAR= ( farfield )
MARKER_INLET= ( inlet, 300.0, {case.U_inf:.4f}, 1.0, 0.0, 0.0 )
MARKER_OUTLET= ( outlet, 0.0 )

% --- Numerics ---
NUM_METHOD_GRAD= GREEN_GAUSS
CFL_NUMBER= 5.0
CFL_ADAPT= YES
CFL_ADAPT_PARAM= ( 0.1, 2.0, 5.0, 1e10 )

CONV_NUM_METHOD_FLOW= FDS
MUSCL_FLOW= YES
SLOPE_LIMITER_FLOW= VENKATAKRISHNAN
VENKAT_LIMITER_COEFF= 0.05

CONV_NUM_METHOD_TURB= SCALAR_UPWIND
MUSCL_TURB= NO

% --- Convergence ---
ITER= {iters}
CONV_RESIDUAL_MINVAL= -12
CONV_FIELD= RMS_PRESSURE, RMS_NU_TILDE
CONV_STARTITER= 100

% --- Output ---
MESH_FILENAME= {mesh_file}
OUTPUT_FILES= RESTART, PARAVIEW_MULTIBLOCK, SURFACE_PARAVIEW
VOLUME_OUTPUT= RESIDUAL, PRIMITIVE, TURBULENT
SURFACE_OUTPUT= SKIN_FRICTION, PRESSURE_COEFFICIENT
OUTPUT_WRT_FREQ= 500
CONV_FILENAME= history
RESTART_FILENAME= restart_flow.dat
SOLUTION_FILENAME= solution_flow.dat
"""

    config_path.write_text(config)
    return config_path


def generate_su2_config_fully_turbulent(
    case: TransitionCaseConfig,
    case_dir: Path,
    mesh_file: str = "grid.su2",
    n_iter: Optional[int] = None,
) -> Path:
    """
    Generate SU2 config for fully turbulent SA (no transition model).

    Serves as baseline to quantify the Cf improvement from transition modeling.
    """
    case_dir = Path(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)

    iters = n_iter or case.n_iter
    config_path = case_dir / "fully_turbulent.cfg"

    config = f"""\
% ============================================================
% SU2 Configuration: SA Fully Turbulent (no transition)
% Case: {case.name}
% ============================================================

SOLVER= INC_RANS
KIND_TURB_MODEL= SA
MATH_PROBLEM= DIRECT
RESTART_SOL= NO

INC_DENSITY_MODEL= CONSTANT
INC_DENSITY_INIT= {case.rho}
INC_VELOCITY_INIT= ({case.U_inf * np.cos(np.radians(case.alpha_deg)):.6f}, {case.U_inf * np.sin(np.radians(case.alpha_deg)):.6f}, 0.0)
INC_TEMPERATURE_INIT= 300.0
VISCOSITY_MODEL= CONSTANT
MU_CONSTANT= {case.mu:.6e}
FREESTREAM_NU_FACTOR= 3.0

REF_LENGTH= {case.L_ref}
REYNOLDS_NUMBER= {case.Re_L:.0f}

MARKER_HEATFLUX= ( wall, 0.0 )
MARKER_FAR= ( farfield )
MARKER_INLET= ( inlet, 300.0, {case.U_inf:.4f}, 1.0, 0.0, 0.0 )
MARKER_OUTLET= ( outlet, 0.0 )

NUM_METHOD_GRAD= GREEN_GAUSS
CFL_NUMBER= 15.0
CFL_ADAPT= YES
CFL_ADAPT_PARAM= ( 0.1, 2.0, 15.0, 1e10 )

CONV_NUM_METHOD_FLOW= FDS
MUSCL_FLOW= YES
SLOPE_LIMITER_FLOW= VENKATAKRISHNAN
VENKAT_LIMITER_COEFF= 0.05

CONV_NUM_METHOD_TURB= SCALAR_UPWIND
MUSCL_TURB= NO

ITER= {iters}
CONV_RESIDUAL_MINVAL= -12
CONV_FIELD= RMS_PRESSURE, RMS_NU_TILDE
CONV_STARTITER= 100

MESH_FILENAME= {mesh_file}
OUTPUT_FILES= RESTART, PARAVIEW_MULTIBLOCK, SURFACE_PARAVIEW
VOLUME_OUTPUT= RESIDUAL, PRIMITIVE, TURBULENT
SURFACE_OUTPUT= SKIN_FRICTION, PRESSURE_COEFFICIENT
OUTPUT_WRT_FREQ= 500
CONV_FILENAME= history
RESTART_FILENAME= restart_flow.dat
SOLUTION_FILENAME= solution_flow.dat
"""

    config_path.write_text(config)
    return config_path


def compare_transition_models(
    case: TransitionCaseConfig,
    output_dir: Path,
    models: List[str] = None,
    mesh_file: str = "grid.su2",
) -> Dict[str, Path]:
    """
    Generate SU2 configs for multiple transition models.

    Parameters
    ----------
    case : TransitionCaseConfig
        Case parameters.
    output_dir : Path
        Base output directory.
    models : list of str
        Model names: 'BCM', 'SA_AFT', 'fully_turbulent'.
    mesh_file : str
        Mesh filename.

    Returns
    -------
    dict mapping model name → config file path.
    """
    if models is None:
        models = ["BCM", "SA_AFT", "fully_turbulent"]

    output_dir = Path(output_dir)
    generators = {
        "BCM": generate_su2_config_bcm,
        "SA_AFT": generate_su2_config_sa_aft,
        "fully_turbulent": generate_su2_config_fully_turbulent,
    }

    configs = {}
    for model_name in models:
        if model_name not in generators:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(generators.keys())}")
        case_dir = output_dir / model_name
        config_path = generators[model_name](case, case_dir, mesh_file)
        configs[model_name] = config_path
        print(f"  Generated config for {model_name}: {config_path}")

    return configs


if __name__ == "__main__":
    # Demo: generate configs for NACA 0012 at alpha=0 transition case
    case = TransitionCaseConfig(
        name="NACA_0012_alpha0_transition",
        U_inf=51.4,        # ~M=0.15 at 300K
        rho=1.225,
        mu=1.7894e-5,
        L_ref=1.0,
        Tu_percent=0.1,    # Low-Tu wind tunnel
        alpha_deg=0.0,
    )

    out = PROJECT_ROOT / "runs" / "naca0012_transition"
    configs = compare_transition_models(case, out)
    print(f"\nGenerated {len(configs)} configs in {out}")
