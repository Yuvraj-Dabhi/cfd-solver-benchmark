#!/usr/bin/env python3
"""
Hybrid RANS-LES Setup Helpers
================================
Configuration generators and resolution assessment tools for hybrid methods:
  - DDES (Delayed Detached Eddy Simulation)
  - SAS (Scale-Adaptive Simulation)

Also provides grid resolution assessment to verify LES region adequacy.

References
----------
  - Spalart (2009), Ann. Rev. Fluid Mech. 41, pp.181-202
  - Menter & Egorov (2010), Flow Turb. Combust. 85, pp.113-138
"""

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class HybridConfig:
    """Configuration for hybrid RANS-LES simulation."""
    method: str = "DDES"            # DDES or SAS
    base_model: str = "SA"          # SA or SST
    delta_type: str = "cubeRootVol" # LES length scale: cubeRootVol, maxDeltaxyz, etc.
    time_scheme: str = "backward"   # backward, CrankNicolson
    cn_blend: float = 0.9           # Crank-Nicolson blending factor
    cfl_target: float = 1.0         # Target CFL for LES region
    write_interval: int = 100       # Time steps between writes
    end_time: float = 10.0          # Physical end time or flow-through times
    dt: float = 1e-4                # Time step
    n_startup_steady: int = 5000    # Steady RANS iterations before switching


@dataclass
class LESResolution:
    """Assessment of LES grid resolution."""
    delta_over_eta: float = 0.0       # Grid spacing / Kolmogorov scale
    delta_plus_x: float = 0.0        # x+ wall units
    delta_plus_y: float = 0.0        # y+ at first cell
    delta_plus_z: float = 0.0        # z+ (spanwise)
    cfl_estimate: float = 0.0        # Estimated CFL
    adequate: bool = False            # Overall assessment
    recommendations: list = field(default_factory=list)


def generate_ddes_config(
    case_name: str,
    base_model: str = "SA",
    output_dir: Optional[Path] = None,
    config: Optional[HybridConfig] = None,
) -> Dict[str, str]:
    """
    Generate OpenFOAM DDES configuration files.

    Parameters
    ----------
    case_name : str
        Benchmark case name.
    base_model : str
        Base RANS model ('SA' or 'SST').
    output_dir : Path, optional
        Output directory.
    config : HybridConfig, optional
        Override defaults.

    Returns
    -------
    dict with paths to generated files.
    """
    cfg = config or HybridConfig(method="DDES", base_model=base_model)
    cfg.method = "DDES"
    cfg.base_model = base_model

    if output_dir is None:
        output_dir = PROJECT_ROOT / "runs" / case_name / f"DDES_{base_model}"
    output_dir = Path(output_dir)
    constant_dir = output_dir / "constant"
    system_dir = output_dir / "system"
    constant_dir.mkdir(parents=True, exist_ok=True)
    system_dir.mkdir(parents=True, exist_ok=True)

    files = {}

    # momentumTransport
    if base_model == "SA":
        les_model = "SpalartAllmarasDDES"
    else:
        les_model = "kOmegaSSTDDES"

    transport = f"""\
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "constant";
    object      momentumTransport;
}}

simulationType  LES;

LES
{{
    LESModel    {les_model};
    turbulence  on;
    printCoeffs on;

    delta       {cfg.delta_type};

    {cfg.delta_type}Coeffs
    {{
        deltaCoeff  1;
    }}
}}
"""
    transport_path = constant_dir / "momentumTransport"
    transport_path.write_text(transport)
    files["momentumTransport"] = str(transport_path)

    # fvSchemes (time-accurate)
    if cfg.time_scheme == "CrankNicolson":
        ddt_scheme = f"CrankNicolson {cfg.cn_blend}"
    else:
        ddt_scheme = "backward"

    schemes = f"""\
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "system";
    object      fvSchemes;
}}

ddtSchemes
{{
    default         {ddt_scheme};
}}

gradSchemes
{{
    default         Gauss linear;
    grad(U)         cellLimited Gauss linear 1;
}}

divSchemes
{{
    default         none;
    div(phi,U)      Gauss LUST grad(U);
    div(phi,nuTilda) Gauss limitedLinear 1;
    div(phi,k)      Gauss limitedLinear 1;
    div(phi,omega)  Gauss limitedLinear 1;
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}}

laplacianSchemes
{{
    default         Gauss linear corrected;
}}

interpolationSchemes
{{
    default         linear;
}}

snGradSchemes
{{
    default         corrected;
}}
"""
    schemes_path = system_dir / "fvSchemes"
    schemes_path.write_text(schemes)
    files["fvSchemes"] = str(schemes_path)

    # controlDict (time-accurate)
    controldict = f"""\
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "system";
    object      controlDict;
}}

application     pimpleFoam;

startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         {cfg.end_time};
deltaT          {cfg.dt};

writeControl    adjustableRunTime;
writeInterval   {cfg.write_interval * cfg.dt};
purgeWrite      5;

writeFormat     binary;
writePrecision  8;
writeCompression off;

timeFormat      general;
timePrecision   8;

adjustTimeStep  yes;
maxCo           {cfg.cfl_target};
maxDeltaT       {cfg.dt * 10};

runTimeModifiable true;

functions
{{
    fieldAverage
    {{
        type            fieldAverage;
        libs            ("libfieldFunctionObjects.so");
        writeControl    writeTime;
        timeStart       {cfg.end_time * 0.5};
        fields
        (
            U
            {{
                mean        on;
                prime2Mean  on;
                base        time;
            }}
            p
            {{
                mean        on;
                prime2Mean  on;
                base        time;
            }}
        );
    }}
    wallShearStress
    {{
        type            wallShearStress;
        libs            ("libfieldFunctionObjects.so");
        writeControl    writeTime;
        patches         (wall);
    }}
}}
"""
    control_path = system_dir / "controlDict"
    control_path.write_text(controldict)
    files["controlDict"] = str(control_path)

    # fvSolution (PIMPLE)
    solution = f"""\
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "system";
    object      fvSolution;
}}

solvers
{{
    p
    {{
        solver          GAMG;
        smoother        DICGaussSeidel;
        tolerance       1e-6;
        relTol          0.01;
    }}

    pFinal
    {{
        $p;
        relTol          0;
    }}

    "(U|nuTilda|k|omega)"
    {{
        solver          PBiCGStab;
        preconditioner  DILU;
        tolerance       1e-8;
        relTol          0.01;
    }}

    "(U|nuTilda|k|omega)Final"
    {{
        $U;
        relTol          0;
    }}
}}

PIMPLE
{{
    nOuterCorrectors    2;
    nCorrectors         1;
    nNonOrthogonalCorrectors 0;

    residualControl
    {{
        U
        {{
            tolerance   1e-5;
            relTol      0;
        }}
        p
        {{
            tolerance   1e-4;
            relTol      0;
        }}
    }}
}}
"""
    solution_path = system_dir / "fvSolution"
    solution_path.write_text(solution)
    files["fvSolution"] = str(solution_path)

    print(f"DDES ({base_model}) config generated for {case_name}")
    print(f"  Output dir: {output_dir}")
    return files


def generate_sas_config(
    case_name: str,
    output_dir: Optional[Path] = None,
    config: Optional[HybridConfig] = None,
) -> Dict[str, str]:
    """
    Generate OpenFOAM SAS (Scale-Adaptive Simulation) configuration.

    SAS uses the kOmegaSSTSAS model which automatically adjusts
    the turbulent length scale based on the von Kármán length scale.

    Parameters
    ----------
    case_name : str
        Benchmark case name.
    output_dir : Path, optional
        Output directory.
    config : HybridConfig, optional
        Override defaults.

    Returns
    -------
    dict with paths to generated files.
    """
    cfg = config or HybridConfig(method="SAS", base_model="SST")

    if output_dir is None:
        output_dir = PROJECT_ROOT / "runs" / case_name / "SAS"
    output_dir = Path(output_dir)
    constant_dir = output_dir / "constant"
    system_dir = output_dir / "system"
    constant_dir.mkdir(parents=True, exist_ok=True)
    system_dir.mkdir(parents=True, exist_ok=True)

    files = {}

    transport = f"""\
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "constant";
    object      momentumTransport;
}}

simulationType  RAS;

RAS
{{
    RASModel    kOmegaSSTSAS;
    turbulence  on;
    printCoeffs on;

    kOmegaSSTSASCoeffs
    {{
        // SAS-specific: activate von Karman length scale
        Cs          0.11;       // SAS coefficient
        // Standard SST coefficients inherited
    }}
}}
"""
    transport_path = constant_dir / "momentumTransport"
    transport_path.write_text(transport)
    files["momentumTransport"] = str(transport_path)

    print(f"SAS config generated for {case_name}")
    print(f"  Output dir: {output_dir}")
    return files


def estimate_les_requirements(
    Re: float,
    L_ref: float,
    U_ref: float,
    nu: float = 1.5e-5,
) -> Dict[str, float]:
    """
    Estimate grid and timestep requirements for LES/hybrid RANS-LES.

    Based on Pope (2000) and Piomelli (1999) guidelines.

    Parameters
    ----------
    Re : float
        Reynolds number.
    L_ref : float
        Reference length.
    U_ref : float
        Reference velocity.
    nu : float
        Kinematic viscosity.

    Returns
    -------
    dict with estimated requirements.
    """
    Re_tau = 0.09 * Re ** 0.88  # Empirical estimate

    # Kolmogorov length scale
    eta = L_ref * Re ** (-3.0 / 4.0)

    # Wall-unit-based requirements (target: Δx+ ≈ 50, Δy+ ≈ 1, Δz+ ≈ 25)
    u_tau = U_ref * math.sqrt(0.026 / (Re ** (1.0 / 7.0)))
    delta_nu = nu / u_tau

    dx_target = 50 * delta_nu    # Streamwise
    dy_target = 1.0 * delta_nu   # Wall-normal first cell
    dz_target = 25 * delta_nu    # Spanwise

    # Timestep for CFL ≈ 1
    dt_target = dx_target / U_ref

    # Total cell count estimate (rough)
    aspect = L_ref  # approximate domain size
    nx = int(aspect / dx_target)
    ny = 100  # typical wall-normal count
    nz = int(0.2 * aspect / dz_target)  # 20% span
    n_total = nx * ny * nz

    # Flow-through times for statistics
    t_ft = L_ref / U_ref
    n_flow_throughs = 10  # Typical for converged statistics
    total_steps = int(n_flow_throughs * t_ft / dt_target)

    return {
        "Re_tau_estimate": Re_tau,
        "kolmogorov_eta": eta,
        "u_tau_estimate": u_tau,
        "delta_nu": delta_nu,
        "dx_plus_50": dx_target,
        "dy_plus_1": dy_target,
        "dz_plus_25": dz_target,
        "dt_cfl1": dt_target,
        "n_cells_estimate": n_total,
        "n_flow_through_times": n_flow_throughs,
        "total_timesteps": total_steps,
        "wall_time_hours_estimate": total_steps * n_total / 1e9,  # Very rough
    }


def check_les_resolution(
    dx: float,
    dy_wall: float,
    dz: float,
    Re: float,
    U_ref: float,
    L_ref: float,
    nu: float = 1.5e-5,
    dt: float = 1e-4,
) -> LESResolution:
    """
    Check if mesh resolution is adequate for LES/hybrid RANS-LES.

    Parameters
    ----------
    dx, dy_wall, dz : float
        Grid spacings (streamwise, wall-normal first cell, spanwise).
    Re : float
        Reynolds number.
    U_ref : float
        Reference velocity.
    L_ref : float
        Reference length.
    nu : float
        Kinematic viscosity.
    dt : float
        Time step.

    Returns
    -------
    LESResolution
        Assessment result.
    """
    u_tau = U_ref * math.sqrt(0.026 / (Re ** (1.0 / 7.0)))
    delta_nu = nu / u_tau

    dx_plus = dx / delta_nu
    dy_plus = dy_wall / delta_nu
    dz_plus = dz / delta_nu
    cfl = U_ref * dt / dx

    eta = L_ref * Re ** (-3.0 / 4.0)
    delta = (dx * dy_wall * dz) ** (1.0 / 3.0)
    delta_over_eta = delta / eta

    recs = []
    adequate = True

    if dy_plus > 2.0:
        recs.append(f"y+ = {dy_plus:.1f} > 2.0: reduce wall-normal first cell height")
        adequate = False
    if dx_plus > 100:
        recs.append(f"Δx+ = {dx_plus:.0f} > 100: refine streamwise grid")
        adequate = False
    if dz_plus > 50:
        recs.append(f"Δz+ = {dz_plus:.0f} > 50: refine spanwise grid")
        adequate = False
    if cfl > 2.0:
        recs.append(f"CFL = {cfl:.2f} > 2.0: reduce time step")
        adequate = False

    if adequate:
        recs.append("Grid resolution meets LES requirements")

    return LESResolution(
        delta_over_eta=delta_over_eta,
        delta_plus_x=dx_plus,
        delta_plus_y=dy_plus,
        delta_plus_z=dz_plus,
        cfl_estimate=cfl,
        adequate=adequate,
        recommendations=recs,
    )


if __name__ == "__main__":
    # Demo: estimate requirements for wall hump
    print("=== LES Requirements for NASA Wall Hump ===")
    reqs = estimate_les_requirements(Re=936_000, L_ref=0.420, U_ref=34.6)
    for k, v in reqs.items():
        print(f"  {k}: {v:.4g}")

    print("\n=== LES Resolution Check (example mesh) ===")
    res = check_les_resolution(
        dx=0.005, dy_wall=5e-6, dz=0.002,
        Re=936_000, U_ref=34.6, L_ref=0.420,
    )
    print(f"  Adequate: {res.adequate}")
    for r in res.recommendations:
        print(f"  - {r}")
