#!/usr/bin/env python3
"""
Reynolds Stress Model (RSM) Runner
====================================
OpenFOAM configuration generator for Reynolds Stress Models (LRR, SSG).

Supports wall hump, backward-facing step, and periodic hill cases.

References
----------
  - Launder, Reece & Rodi (1975), J. Fluid Mech. 68(3), pp.537-566
  - Speziale, Sarkar & Gatski (1991), J. Fluid Mech. 227, pp.245-272
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# Supported cases with default mesh levels
SUPPORTED_CASES = {
    "nasa_hump": {
        "description": "NASA Wall-Mounted Hump",
        "mesh_levels": {"coarse": 150_000, "medium": 350_000, "fine": 700_000},
        "Re": 936_000,
    },
    "backward_facing_step": {
        "description": "2D Backward-Facing Step",
        "mesh_levels": {"coarse": 40_000, "medium": 90_000, "fine": 200_000},
        "Re": 36_000,
    },
    "periodic_hill": {
        "description": "Periodic Hill (ERCOFTAC)",
        "mesh_levels": {"coarse": 50_000, "medium": 150_000, "fine": 400_000},
        "Re": 10_595,
    },
}


@dataclass
class RSMConfig:
    """Configuration for RSM simulation."""
    variant: str = "LRR"            # LRR or SSG
    wall_reflection: bool = True     # Wall reflection terms
    pressure_strain: str = "default" # Pressure-strain correlation
    diffusion: str = "Daly-Harlow"   # Diffusion model
    max_iterations: int = 10000
    write_interval: int = 500
    convergence_residual: float = 1e-6


def generate_openfoam_rsm(
    case_name: str,
    mesh_level: str = "medium",
    rsm_variant: str = "LRR",
    output_dir: Optional[Path] = None,
    config: Optional[RSMConfig] = None,
) -> Dict[str, str]:
    """
    Generate OpenFOAM constant/momentumTransport and system/ files for RSM.

    Parameters
    ----------
    case_name : str
        Benchmark case name (must be in SUPPORTED_CASES).
    mesh_level : str
        Mesh resolution level.
    rsm_variant : str
        RSM variant: 'LRR' or 'SSG'.
    output_dir : Path, optional
        Output directory. If None, uses runs/{case_name}/RSM_{variant}_{mesh_level}/.
    config : RSMConfig, optional
        Override default RSM settings.

    Returns
    -------
    dict with paths to generated files.
    """
    if case_name not in SUPPORTED_CASES:
        raise ValueError(
            f"Unknown case: {case_name}. Available: {list(SUPPORTED_CASES.keys())}"
        )

    if rsm_variant not in ("LRR", "SSG"):
        raise ValueError(f"Unknown RSM variant: {rsm_variant}. Use 'LRR' or 'SSG'.")

    cfg = config or RSMConfig(variant=rsm_variant)
    cfg.variant = rsm_variant

    if output_dir is None:
        output_dir = PROJECT_ROOT / "runs" / case_name / f"RSM_{rsm_variant}_{mesh_level}"

    output_dir = Path(output_dir)
    constant_dir = output_dir / "constant"
    system_dir = output_dir / "system"
    constant_dir.mkdir(parents=True, exist_ok=True)
    system_dir.mkdir(parents=True, exist_ok=True)

    files = {}

    # --- momentumTransport ---
    if rsm_variant == "LRR":
        transport_content = _generate_lrr_transport(cfg)
    else:
        transport_content = _generate_ssg_transport(cfg)

    transport_path = constant_dir / "momentumTransport"
    transport_path.write_text(transport_content)
    files["momentumTransport"] = str(transport_path)

    # --- fvSchemes ---
    schemes_content = _generate_rsm_fvschemes()
    schemes_path = system_dir / "fvSchemes"
    schemes_path.write_text(schemes_content)
    files["fvSchemes"] = str(schemes_path)

    # --- fvSolution ---
    solution_content = _generate_rsm_fvsolution(cfg)
    solution_path = system_dir / "fvSolution"
    solution_path.write_text(solution_content)
    files["fvSolution"] = str(solution_path)

    # --- controlDict ---
    control_content = _generate_rsm_controldict(cfg)
    control_path = system_dir / "controlDict"
    control_path.write_text(control_content)
    files["controlDict"] = str(control_path)

    print(f"RSM ({rsm_variant}) config generated for {case_name}/{mesh_level}")
    print(f"  Output dir: {output_dir}")
    return files


def _generate_lrr_transport(cfg: RSMConfig) -> str:
    """Generate momentumTransport dict for LRR RSM."""
    return f"""\
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
    RASModel    LRR;

    turbulence  on;
    printCoeffs on;

    LRRCoeffs
    {{
        Cmu         0.09;
        C1          1.8;
        C2          0.6;
        Ceps1       1.44;
        Ceps2       1.92;
        Cs          0.22;       // Daly-Harlow diffusion coefficient
        Ceps        0.15;
        wallReflection  {"yes" if cfg.wall_reflection else "no"};

        // Wall reflection (Gibson & Launder)
        Cref1       0.5;
        Cref2       0.3;
    }}
}}
"""


def _generate_ssg_transport(cfg: RSMConfig) -> str:
    """Generate momentumTransport dict for SSG (Speziale-Sarkar-Gatski) RSM."""
    return f"""\
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
    RASModel    SSG;

    turbulence  on;
    printCoeffs on;

    SSGCoeffs
    {{
        Cmu         0.09;

        // Slow pressure-strain
        C1          3.4;
        C1s         1.8;

        // Rapid pressure-strain
        C2          4.2;
        C3          0.8;
        C3s         1.3;
        C4          1.25;
        C5          0.40;

        // Dissipation equation
        Ceps1       1.44;
        Ceps2       1.83;
        Cs          0.22;
        Ceps        0.15;

        wallReflection  {"yes" if cfg.wall_reflection else "no"};
    }}
}}
"""


def _generate_rsm_fvschemes() -> str:
    """Generate fvSchemes for RSM with appropriate settings."""
    return """\
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "system";
    object      fvSchemes;
}

ddtSchemes
{
    default         steadyState;
}

gradSchemes
{
    default         cellLimited Gauss linear 1;
    grad(U)         cellLimited Gauss linear 1;
}

divSchemes
{
    default         none;
    div(phi,U)      bounded Gauss linearUpwind grad(U);
    div(phi,R)      bounded Gauss upwind;
    div(phi,epsilon) bounded Gauss upwind;
    div(phi,k)      bounded Gauss upwind;
    div(R)          Gauss linear;
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}

laplacianSchemes
{
    default         Gauss linear corrected;
}

interpolationSchemes
{
    default         linear;
}

snGradSchemes
{
    default         corrected;
}
"""


def _generate_rsm_fvsolution(cfg: RSMConfig) -> str:
    """Generate fvSolution for RSM."""
    return f"""\
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
        smoother        GaussSeidel;
        tolerance       1e-6;
        relTol          0.1;
    }}

    "(U|R|epsilon|k)"
    {{
        solver          smoothSolver;
        smoother        GaussSeidel;
        tolerance       1e-8;
        relTol          0.1;
    }}
}}

SIMPLE
{{
    nNonOrthogonalCorrectors 0;
    consistent      yes;

    residualControl
    {{
        p               {cfg.convergence_residual};
        U               {cfg.convergence_residual};
        R               {cfg.convergence_residual};
        epsilon         {cfg.convergence_residual};
    }}
}}

relaxationFactors
{{
    fields
    {{
        p               0.3;
    }}
    equations
    {{
        U               0.5;
        R               0.3;    // RSM needs lower relaxation for stability
        epsilon         0.5;
    }}
}}
"""


def _generate_rsm_controldict(cfg: RSMConfig) -> str:
    """Generate controlDict for RSM."""
    return f"""\
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "system";
    object      controlDict;
}}

application     simpleFoam;

startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         {cfg.max_iterations};
deltaT          1;

writeControl    timeStep;
writeInterval   {cfg.write_interval};
purgeWrite      3;

writeFormat     ascii;
writePrecision  8;
writeCompression off;

timeFormat      general;
timePrecision   6;

runTimeModifiable true;

functions
{{
    wallShearStress
    {{
        type            wallShearStress;
        libs            ("libfieldFunctionObjects.so");
        writeControl    writeTime;
        patches         (wall);
    }}
    yPlus
    {{
        type            yPlus;
        libs            ("libfieldFunctionObjects.so");
        writeControl    writeTime;
    }}
}}
"""


if __name__ == "__main__":
    # Demo: generate RSM configs for wall hump
    for variant in ["LRR", "SSG"]:
        files = generate_openfoam_rsm("nasa_hump", "medium", variant)
        print(f"  Generated {len(files)} files")
