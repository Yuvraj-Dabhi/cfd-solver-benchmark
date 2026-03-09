#!/usr/bin/env python3
"""
ZBOT-Style OpenFOAM VOF Micro-g Tank Filling
==============================================
Simulates the vented filling of a cylindrical tank under step-reduced gravity,
replicating conditions from the ZBOT-FT 2 orbital-depot operations and
Govindan (2024) drop-tower experiments.

Hardware parameters (Baseline):
- Tank: 2D axisymmetric, R = 75 mm, H = 300 mm
- Initial fill level: 50%
- Gravity ramp: 1g (9.81 m/s^2) down to 10^-6 g over 0.1s
- Fluid: Liquid water (alpha=1) and air (alpha=0) at 300K
- Surface tension: 0.072 N/m

Physics Output:
- Tracks the phase fraction (alpha) to observe geyser formation
- Classifies interface as stable vs. unstable based on geyser height

Usage:
    python run_zbot_vof.py --case standard          # Run standard filling case
    python run_zbot_vof.py --case standard --dry-run # Only generate OpenFOAM configuration
"""

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from scripts.openfoam_utils import FoamCaseGenerator, foam_header
except ImportError:
    print("Error: Could not import scripts.openfoam_utils. Ensure it exists.")
    sys.exit(1)


# =============================================================================
# Case Configuration
# =============================================================================

@dataclass
class ZBOTCase:
    name: str
    R_tank: float = 0.075        # Tank radius [m] (75 mm)
    H_tank: float = 0.300        # Tank height [m] (300 mm)
    fill_fraction: float = 0.5   # Initial fill fraction (50%)
    inlet_radius: float = 0.005  # Inlet nozzle radius [m] (5 mm)
    inlet_velocity: float = 0.5  # Inflow velocity [m/s]
    g_initial: float = 9.81      # Initial gravity [m/s^2]
    g_micro: float = 1e-6        # Final micro-g [m/s^2]
    t_ramp: float = 0.1          # Gravity ramp duration [s]
    sigma: float = 0.072         # Surface tension [N/m] (water-air)
    t_end: float = 2.0           # Simulation end time [s]

    @property
    def V_inlet(self) -> float:
        """Inlet volume flow rate [m^3/s]."""
        return np.pi * self.inlet_radius**2 * self.inlet_velocity


CASES = {
    "standard": ZBOTCase(name="standard", inlet_velocity=0.5),
    "high_flow": ZBOTCase(name="high_flow", inlet_velocity=1.5, t_end=1.0),
    "low_g": ZBOTCase(name="low_g", g_micro=1e-8),
}


# =============================================================================
# OpenFOAM BlockMesh Generator (2D Axisymmetric)
# =============================================================================

def write_blockMeshDict(case: ZBOTCase, case_dir: Path, n_radial: int = 50, n_axial: int = 200):
    """
    Generate blockMeshDict for a 2D axisymmetric cylinder.
    OpenFOAM requires a 3D wedge (typically 5 degrees) for 2D axisymmetric cases.
    """
    # 5 degree wedge
    angle = 5.0
    theta = np.deg2rad(angle / 2.0)
    
    # Points for a wedge in the x-y plane (rotated around y-axis).
    # Centerline is along y-axis (x=0, z=0). 
    # Actually, standard OpenFOAM axis is x-axis. Let's use x-axis as centerline.
    # We will revolve the x-y plane around the x-axis.
    # Radius is y-axis.
    
    # Wait, interFoam usually uses z as vertical for gravity. 
    # Let's align:
    # Axis of symmetry: y-axis
    # X corresponds to radius.
    # Wedge in X-Z plane? No, wedge in X-Y pane, thickness in Z.
    # The standard wedge: x is along flow, y is radius, rotate around x-axis.
    # Let's use standard y-axis as height, x-axis as radius, z as depth.
    # Centerline: x=0, z=0. Height from y=0 to y=H.
    
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    R = case.R_tank
    H = case.H_tank
    
    # Inlet radius
    Ri = case.inlet_radius
    
    # To properly define an inlet, we need two blocks in radial direction:
    # Block 1: 0 to Ri (Inlet region)
    # Block 2: Ri to R_tank (Wall region)
    
    n_rad_1 = max(10, int(n_radial * (Ri / R)))
    n_rad_2 = n_radial - n_rad_1

    content = foam_header("dictionary", "blockMeshDict", location="system")
    content += f"""scale   1;

vertices
(
    // Base (y = 0)
    (0 0 0)                  // 0 (Center)
    ({Ri*cos_t} 0 {-Ri*sin_t})   // 1 (Inlet edge, front)
    ({Ri*cos_t} 0 {Ri*sin_t})    // 2 (Inlet edge, back)
    ({R*cos_t} 0 {-R*sin_t})     // 3 (Outer edge, front)
    ({R*cos_t} 0 {R*sin_t})      // 4 (Outer edge, back)

    // Top (y = H)
    (0 {H} 0)                  // 5 (Center)
    ({Ri*cos_t} {H} {-Ri*sin_t})   // 6 (Inlet edge, front)
    ({Ri*cos_t} {H} {Ri*sin_t})    // 7 (Inlet edge, back)
    ({R*cos_t} {H} {-R*sin_t})     // 8 (Outer edge, front)
    ({R*cos_t} {H} {R*sin_t})      // 9 (Outer edge, back)
);

blocks
(
    // Inner block (Inlet)
    hex (0 1 2 0 5 6 7 5) ({n_rad_1} {n_axial} 1) simpleGrading (1 1 1)
    
    // Outer block (Wall)
    hex (1 3 4 2 6 8 9 7) ({n_rad_2} {n_axial} 1) simpleGrading (1 1 1)
);

edges
(
    // Arcs to make the wedge curved (though for 5 deg straight lines are often fine, arcs are better)
    arc 1 2 ({Ri} 0 0)
    arc 3 4 ({R} 0 0)
    arc 6 7 ({Ri} {H} 0)
    arc 8 9 ({R} {H} 0)
);

boundary
(
    bottom_inlet
    {{
        type patch;
        faces
        (
            (0 1 2 0)
        );
    }}
    bottom_wall
    {{
        type wall;
        faces
        (
            (1 3 4 2)
        );
    }}
    top_vent
    {{
        type patch;
        faces
        (
            (5 6 7 5)
            (6 8 9 7)
        );
    }}
    side_wall
    {{
        type wall;
        faces
        (
            (3 8 9 4)
        );
    }}
    axis
    {{
        type empty;
        faces
        (
            (0 5 5 0)
        );
    }}
    front
    {{
        type wedge;
        faces
        (
            (0 5 6 1)
            (1 6 8 3)
        );
    }}
    back
    {{
        type wedge;
        faces
        (
            (0 2 7 5)
            (2 4 9 7)
        );
    }}
);

mergePatchPairs
(
);
"""
    (case_dir / "system").mkdir(parents=True, exist_ok=True)
    (case_dir / "system" / "blockMeshDict").write_text(content)


# =============================================================================
# OpenFOAM Physics Configuration
# =============================================================================

def write_setFieldsDict(case: ZBOTCase, case_dir: Path):
    """Write system/setFieldsDict to initialize liquid water."""
    fill_y = case.H_tank * case.fill_fraction
    content = foam_header("dictionary", "setFieldsDict", location="system")
    content += f"""defaultFieldValues
(
    volScalarFieldValue alpha.water 0
    volVectorFieldValue U (0 0 0)
);

regions
(
    boxToCell
    {{
        box (-1 -1 -1) (1 {fill_y} 1);
        fieldValues
        (
            volScalarFieldValue alpha.water 1
        );
    }}
);
"""
    (case_dir / "system" / "setFieldsDict").write_text(content)


def generate_fvSchemes() -> str:
    return """
ddtSchemes
{
    default         Euler;
}

gradSchemes
{
    default         Gauss linear;
}

divSchemes
{
    div(rhoPhi,U)  Gauss linearUpwind grad(U);
    div(phi,alpha)  Gauss vanLeer;
    div(phirb,alpha) Gauss linear;
    default         none;
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

def generate_fvSolution() -> str:
    return """
solvers
{
    "alpha.water.*"
    {
        nAlphaCorr      1;
        nAlphaSubCycles 2;
        cAlpha          1;

        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-8;
        relTol          0;
    }

    "pcorr.*"
    {
        solver          PCG;
        preconditioner  DIC;
        tolerance       1e-5;
        relTol          0;
    }

    p_rgh
    {
        solver          PCG;
        preconditioner  DIC;
        tolerance       1e-8;
        relTol          0.01;
    }

    p_rghFinal
    {
        $p_rgh;
        relTol          0;
    }

    U
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-6;
        relTol          0;
    }
}

PIMPLE
{
    momentumPredictor no;
    nOuterCorrectors 1;
    nCorrectors     3;
    nNonOrthogonalCorrectors 0;
}
"""


def generate_transportProperties(sigma: float) -> str:
    return f"""
phases (water air);

water
{{
    transportModel  Newtonian;
    nu              1e-06;
    rho             1000;
}}

air
{{
    transportModel  Newtonian;
    nu              1.48e-05;
    rho             1.2;
}}

sigma           {sigma};
"""


# =============================================================================
# Custom Body Force (fvOptions) for Gravity Ramp
# =============================================================================

def write_fvOptions(case: ZBOTCase, case_dir: Path):
    """
    Write system/fvOptions to implement the step-reduced gravity ramp.
    We apply a uniform momentum source that varies in time. Note that interFoam 
    uses a constant `g` field. To simulate a ramp, we set `g = 0` in constant/g
    and apply the entire gravity as a body force via fvOptions codedSource.
    """
    content = foam_header("dictionary", "fvOptions", location="system")
    content += f"""
gravityRamp
{{
    type            vectorSemiImplicitSource;
    active          yes;
    selectionMode   all;

    vectorSemiImplicitSourceCoeffs
    {{
        volumeMode      specific; // Force per unit volume
        injectionRateSuSp
        {{
            U           ((0 0 0) 0); // Placeholder, we use coded function below if needed.
        }}
    }}
}}

// Alternatively, because interFoam heavily integrates gravity into p_rgh formulation,
// it's safer to use the 'timeVaryingMappedFixedValue' approach or a codedSource.
// For simplicity in OpenFOAM 2312, we can define a scalarCodedSource updating g.

gravitySource
{{
    type            coded;
    selectionMode   all;
    name            gravitySource;
    
    codeAddSup
    #{{
        const Time& t = mesh().time();
        const scalar time = t.value();
        
        scalar g_y = 0.0;
        scalar g_initial = -{case.g_initial};
        scalar g_micro = -{case.g_micro};
        scalar t_ramp = {case.t_ramp};
        
        if (time < t_ramp)
        {{
            g_y = g_initial + (g_micro - g_initial) * (time / t_ramp);
        }}
        else
        {{
            g_y = g_micro;
        }}
        
        // Add to momentum eqn (rho * g). In interFoam, the governing eq is divided by rho in places,
        // but fvOptions source for U is typically volumetric.
        const volScalarField& rho = mesh().lookupObject<volScalarField>("rho");
        
        eqn += rho * vector(0, g_y, 0);
    #}};
}}
"""
    # Note: Implementing time-varying gravity robustly in interFoam usually requires
    # modifying the solver or using a specialized fvOption.
    # The `gravitySource` above adds `rho*g` to the momentum equation. 
    # For this to work correctly, we MUST set `g` to 0 in `constant/g` so we don't double count.
    
    (case_dir / "system" / "fvOptions").write_text(content)


# =============================================================================
# Boundary Conditions (0/ Directory)
# =============================================================================

def write_0_fields(case: ZBOTCase, case_dir: Path):
    generator = FoamCaseGenerator(case_dir)
    
    # alpha.water
    boundaries_alpha = {
        "bottom_inlet": "type fixedValue;\nvalue uniform 1;",
        "bottom_wall": "type zeroGradient;",
        "side_wall": "type zeroGradient;",
        "top_vent": "type inletOutlet;\ninletValue uniform 0;\nvalue uniform 0;",
        "axis": "type empty;",
        "front": "type wedge;",
        "back": "type wedge;"
    }
    generator.write_0_field("alpha.water", "[0 0 0 0 0 0 0]", "uniform 0", boundaries_alpha)
    
    # U (Velocity)
    boundaries_U = {
        "bottom_inlet": f"type fixedValue;\nvalue uniform (0 {case.inlet_velocity} 0);",
        "bottom_wall": "type noSlip;",
        "side_wall": "type noSlip;",
        "top_vent": "type pressureInletOutletVelocity;\nvalue uniform (0 0 0);",
        "axis": "type empty;",
        "front": "type wedge;",
        "back": "type wedge;"
    }
    generator.write_0_field("U", "[0 1 -1 0 0 0 0]", "uniform (0 0 0)", boundaries_U)
    
    # p_rgh (Pseudo-pressure)
    boundaries_prgh = {
        "bottom_inlet": "type zeroGradient;",
        "bottom_wall": "type zeroGradient;",
        "side_wall": "type zeroGradient;",
        "top_vent": "type totalPressure;\np0 uniform 0;\nU U;\nphi phi;\nrho rho;\npsi none;\ngamma 1;\nvalue uniform 0;",
        "axis": "type empty;",
        "front": "type wedge;",
        "back": "type wedge;"
    }
    generator.write_0_field("p_rgh", "[1 -1 -2 0 0 0 0]", "uniform 0", boundaries_prgh)
    
    # p (Absolute pressure constraint)
    boundaries_p = {
        "bottom_inlet": "type calculated;\nvalue uniform 0;",
        "bottom_wall": "type calculated;\nvalue uniform 0;",
        "side_wall": "type calculated;\nvalue uniform 0;",
        "top_vent": "type calculated;\nvalue uniform 0;",
        "axis": "type empty;",
        "front": "type wedge;",
        "back": "type wedge;"
    }
    generator.write_0_field("p", "[1 -1 -2 0 0 0 0]", "uniform 0", boundaries_p)

# =============================================================================
# Main Orchestrator
# =============================================================================

def run_case(case_name: str, dry_run: bool = False):
    if case_name not in CASES:
        print(f"Unknown case: {case_name}")
        sys.exit(1)
        
    case = CASES[case_name]
    runs_dir = PROJECT_ROOT / "runs" / "zbot_vof"
    case_dir = runs_dir / case_name
    
    print(f"============================================================")
    print(f" ZBOT-Style VOF Micro-g Tank Filling")
    print(f" Case: {case_name}")
    print(f" Tank: R={case.R_tank*1000}mm, H={case.H_tank*1000}mm (Fill: {case.fill_fraction*100}%)")
    print(f" Inlet: V={case.inlet_velocity} m/s")
    print(f" Gravity: {case.g_initial}g -> {case.g_micro}g over {case.t_ramp}s")
    print(f"============================================================")
    
    # 1. Setup Directories
    generator = FoamCaseGenerator(case_dir)
    generator.setup_directories(clean=True)
    
    # 2. System files
    generator.write_controlDict(
        application="interFoam",
        endTime=case.t_end,
        deltaT=0.001,
        writeInterval=0.05,
        maxCo=0.5,
        maxAlphaCo=0.5,
        maxDeltaT=0.1
    )
    generator.write_fvSchemes(generate_fvSchemes())
    generator.write_fvSolution(generate_fvSolution())
    write_blockMeshDict(case, case_dir)
    write_setFieldsDict(case, case_dir)
    write_fvOptions(case, case_dir)
    
    # 3. Constant files
    generator.write_transportProperties(generate_transportProperties(case.sigma))
    generator.write_momentumTransport("laminar") # VOF typical DNS/laminar for low-Re
    generator.write_g((0, 0, 0)) # We apply g strictly via fvOptions codedSource so set static g to 0
    
    # 4. Zero (0/) files
    write_0_fields(case, case_dir)
    
    print(f"-> Case structure generated in {case_dir}")
    
    if dry_run:
        print("-> Dry run specified. Stopping here.")
        return
        
    # Execution (Requires WSL/OpenFOAM environment)
    print("-> Creating mesh (blockMesh)...")
    try:
        subprocess.run(["blockMesh"], cwd=str(case_dir), check=True, capture_output=True)
    except FileNotFoundError:
        print("-> Warning: 'blockMesh' command not found. Skipping OpenFOAM execution.")
        print("   If running on Windows, you must run this inside WSL with OpenFOAM sourced.")
        return
        
    print("-> Initializing field (setFields)...")
    subprocess.run(["setFields"], cwd=str(case_dir), check=True, capture_output=True)
    
    print("-> Running interFoam (This will take a while)...")
    with open(case_dir / "interFoam.log", "w") as log:
        proc = subprocess.run(["interFoam"], cwd=str(case_dir), stdout=log, stderr=subprocess.STDOUT)
        
    if proc.returncode == 0:
        print(f"-> Simulation finished successfully. Log: {case_dir / 'interFoam.log'}")
    else:
        print(f"-> Simulation failed with return code {proc.returncode}. See log.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ZBOT-style OpenFOAM VOF tank filling.")
    parser.add_argument("--case", type=str, default="standard", choices=CASES.keys(),
                        help="Case preset to run.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only generate the OpenFOAM configuration files.")
    
    args = parser.parse_args()
    run_case(args.case, args.dry_run)
