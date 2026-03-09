"""
OpenFOAM Solver Runner
======================
Manages OpenFOAM case setup, execution, and convergence monitoring.
Supports all 14 turbulence models from config.py and scheme switching.

Usage:
    runner = OpenFOAMRunner(case_dir, model="SST", mesh_level="fine")
    runner.setup_case()
    runner.run()
    runner.check_convergence()
"""

import json
import logging
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Lazy import config to avoid circular deps
def _load_config():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from config import (
        BENCHMARK_CASES, TURBULENCE_MODELS, SOLVER_DEFAULTS,
        NUMERICAL_SCHEMES, SCHEME_SENSITIVITY_MATRIX,
    )
    return BENCHMARK_CASES, TURBULENCE_MODELS, SOLVER_DEFAULTS, NUMERICAL_SCHEMES, SCHEME_SENSITIVITY_MATRIX


logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================
@dataclass
class ResidualHistory:
    """Tracked residual convergence."""
    iteration: List[int] = field(default_factory=list)
    Ux: List[float] = field(default_factory=list)
    Uy: List[float] = field(default_factory=list)
    p: List[float] = field(default_factory=list)
    turbulence: List[float] = field(default_factory=list)
    continuity: List[float] = field(default_factory=list)

    @property
    def converged(self) -> bool:
        if not self.p:
            return False
        return self.p[-1] < 1e-5 and self.Ux[-1] < 1e-5


@dataclass
class RunResult:
    """Result from an OpenFOAM simulation."""
    case_dir: str
    model: str
    mesh_level: str
    converged: bool = False
    iterations: int = 0
    wall_time_s: float = 0.0
    final_residuals: Dict[str, float] = field(default_factory=dict)
    error: Optional[str] = None
    log_file: str = ""


# =============================================================================
# OpenFOAM Runner
# =============================================================================
class OpenFOAMRunner:
    """
    Manages OpenFOAM case lifecycle: setup → run → monitor → post-process.

    Parameters
    ----------
    case_dir : Path
        Root directory for the OpenFOAM case.
    case_name : str
        Benchmark case key from config.BENCHMARK_CASES.
    model : str
        Turbulence model key from config.TURBULENCE_MODELS.
    mesh_level : str
        Grid level: 'coarse', 'medium', 'fine', 'xfine'.
    scheme_id : int
        Index into SCHEME_SENSITIVITY_MATRIX (0-3).
    n_procs : int
        Number of MPI processes for parallel runs.
    """

    def __init__(
        self,
        case_dir: Path,
        case_name: str = "backward_facing_step",
        model: str = "SST",
        mesh_level: str = "medium",
        scheme_id: int = 1,
        n_procs: int = 1,
    ):
        self.case_dir = Path(case_dir)
        self.case_name = case_name
        self.model_key = model
        self.mesh_level = mesh_level
        self.scheme_id = scheme_id
        self.n_procs = n_procs

        # Load config
        cases, models, defaults, schemes, scheme_matrix = _load_config()
        self.case_config = cases.get(case_name)
        self.model_config = models.get(model)
        self.solver_defaults = defaults
        self.numerical_schemes = schemes
        self.scheme_matrix = scheme_matrix

        self.residuals = ResidualHistory()
        self._log_file = self.case_dir / "log.simpleFoam"

    # -------------------------------------------------------------------------
    # Case Setup
    # -------------------------------------------------------------------------
    def setup_case(self) -> None:
        """Create full OpenFOAM case directory structure."""
        logger.info(f"Setting up case: {self.case_name} / {self.model_key} / {self.mesh_level}")

        # Create directories
        for d in ["0", "constant", "system"]:
            (self.case_dir / d).mkdir(parents=True, exist_ok=True)

        # Write all dictionary files
        self._write_control_dict()
        self._write_fv_schemes()
        self._write_fv_solution()
        self._write_turbulence_properties()
        self._write_transport_properties()
        self._write_boundary_conditions()

        logger.info(f"Case setup complete at {self.case_dir}")

    def _write_control_dict(self) -> None:
        """Write system/controlDict."""
        is_les = self.model_config and self.model_config.model_type.value in [
            "Hybrid RANS-LES", "Wall-Modeled LES"
        ]

        content = f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      controlDict;
}}

application     {"pimpleFoam" if is_les else "simpleFoam"};
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         {5000 if not is_les else 0.5};
deltaT          {1 if not is_les else 1e-4};
writeControl    {"timeStep" if not is_les else "adjustableRunTime"};
writeInterval   {500 if not is_les else 0.01};
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
    }}

    yPlus
    {{
        type            yPlus;
        libs            ("libfieldFunctionObjects.so");
        writeControl    writeTime;
    }}

    fieldAverage
    {{
        type            fieldAverage;
        libs            ("libfieldFunctionObjects.so");
        writeControl    writeTime;
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

    residuals
    {{
        type            residuals;
        libs            ("libutilityFunctionObjects.so");
        writeControl    timeStep;
        writeInterval   1;
        fields          (U p {"nuTilda" if self.model_key.startswith("SA") else "k"});
    }}
}}
"""
        (self.case_dir / "system" / "controlDict").write_text(content)

    def _write_fv_schemes(self) -> None:
        """Write system/fvSchemes based on scheme_id."""
        scheme = self.scheme_matrix[self.scheme_id] if self.scheme_id < len(self.scheme_matrix) else self.scheme_matrix[1]

        grad_scheme = self.numerical_schemes["gradient"][scheme["gradient"]]
        conv_scheme = self.numerical_schemes["convection"][scheme["convection"]]

        is_les = self.model_config and self.model_config.model_type.value in [
            "Hybrid RANS-LES", "Wall-Modeled LES"
        ]
        time_scheme = self.numerical_schemes["time"]["backward" if is_les else "steady"]

        content = f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fvSchemes;
}}

// Scheme: {scheme.get("label", "default")}

ddtSchemes
{{
    default         {time_scheme};
}}

gradSchemes
{{
    default         {grad_scheme};
}}

divSchemes
{{
    default         none;
    div(phi,U)      {conv_scheme};
    div(phi,k)      bounded Gauss upwind;
    div(phi,omega)  bounded Gauss upwind;
    div(phi,epsilon) bounded Gauss upwind;
    div(phi,nuTilda) bounded Gauss upwind;
    div(phi,R)      bounded Gauss upwind;
    div(R)          Gauss linear;
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
        (self.case_dir / "system" / "fvSchemes").write_text(content)

    def _write_fv_solution(self) -> None:
        """Write system/fvSolution."""
        sd = self.solver_defaults
        is_les = self.model_config and self.model_config.model_type.value in [
            "Hybrid RANS-LES", "Wall-Modeled LES"
        ]

        content = f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fvSolution;
}}

solvers
{{
    p
    {{
        solver          {sd["linear_solvers"]["p"]["solver"]};
        smoother        {sd["linear_solvers"]["p"]["smoother"]};
        tolerance       {sd["linear_solvers"]["p"]["tolerance"]};
        relTol          {sd["linear_solvers"]["p"]["relTol"]};
    }}

    "(U|k|omega|epsilon|nuTilda|R)"
    {{
        solver          {sd["linear_solvers"]["U"]["solver"]};
        smoother        {sd["linear_solvers"]["U"]["smoother"]};
        tolerance       {sd["linear_solvers"]["U"]["tolerance"]};
        relTol          {sd["linear_solvers"]["U"]["relTol"]};
    }}
}}

{"PIMPLE" if is_les else "SIMPLE"}
{{
    {"nOuterCorrectors 2;" if is_les else ""}
    nCorrectors     {3 if is_les else 1};
    nNonOrthogonalCorrectors {sd["nNonOrthogonalCorrectors"]};
    {"pRefCell        0;" if not is_les else ""}
    {"pRefValue       0;" if not is_les else ""}

    residualControl
    {{
        U               {sd["convergence_residual"]};
        p               {sd["convergence_residual"]};
    }}
}}

relaxationFactors
{{
    {"equations" if is_les else "fields"}
    {{
        {"" if is_les else f"p               {sd['relaxation']['p']};"}
    }}
    {"" if is_les else "equations"}
    {{
        {"" if is_les else f'''U               {sd["relaxation"]["U"]};
        k               {sd["relaxation"]["k"]};
        omega           {sd["relaxation"]["omega"]};
        epsilon         {sd["relaxation"]["epsilon"]};
        nuTilda         {sd["relaxation"]["nuTilda"]};'''}
    }}
}}
"""
        (self.case_dir / "system" / "fvSolution").write_text(content)

    def _write_turbulence_properties(self) -> None:
        """Write constant/turbulenceProperties."""
        if not self.model_config:
            return

        settings = self.model_config.openfoam_settings
        sim_type = settings.get("simulationType", "RAS")

        if sim_type == "RAS":
            ras_model = settings.get("RASModel", self.model_config.openfoam_name)
            content = f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      turbulenceProperties;
}}

simulationType  RAS;

RAS
{{
    RASModel        {ras_model};
    turbulence      on;
    printCoeffs     on;
}}
"""
        else:  # LES
            les_model = settings.get("LESModel", self.model_config.openfoam_name)
            delta_type = settings.get("delta", "cubeRootVol")
            content = f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      turbulenceProperties;
}}

simulationType  LES;

LES
{{
    LESModel        {les_model};
    turbulence      on;
    printCoeffs     on;
    delta           {delta_type};

    cubeRootVolCoeffs
    {{
        deltaCoeff      1;
    }}
}}
"""
        (self.case_dir / "constant" / "turbulenceProperties").write_text(content)

    def _write_transport_properties(self) -> None:
        """Write constant/transportProperties."""
        Re = self.case_config.reynolds_number if self.case_config else 36000
        U_ref = self.case_config.reference_velocity if self.case_config and self.case_config.reference_velocity > 0 else 1.0
        L_ref = self.case_config.reference_length if self.case_config else 1.0
        nu = U_ref * L_ref / Re

        content = f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      transportProperties;
}}

transportModel  Newtonian;
nu              [0 2 -1 0 0 0 0] {nu:.6e};
"""
        (self.case_dir / "constant" / "transportProperties").write_text(content)

    def _write_boundary_conditions(self) -> None:
        """Write 0/ boundary condition files (U, p, turbulence)."""
        U_ref = 44.2  # Default
        if self.case_config and self.case_config.reference_velocity > 0:
            U_ref = self.case_config.reference_velocity

        # --- U ---
        u_content = f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       volVectorField;
    object      U;
}}

dimensions      [0 1 -1 0 0 0 0];
internalField   uniform ({U_ref} 0 0);

boundaryField
{{
    inlet
    {{
        type            fixedValue;
        value           uniform ({U_ref} 0 0);
    }}
    outlet
    {{
        type            zeroGradient;
    }}
    walls
    {{
        type            noSlip;
    }}
    "(top|bottom)"
    {{
        type            noSlip;
    }}
    "(front|back)"
    {{
        type            empty;
    }}
}}
"""
        (self.case_dir / "0" / "U").write_text(u_content)

        # --- p ---
        p_content = """FoamFile
{
    version     2.0;
    format      ascii;
    class       volScalarField;
    object      p;
}

dimensions      [0 2 -2 0 0 0 0];
internalField   uniform 0;

boundaryField
{
    inlet
    {
        type            zeroGradient;
    }
    outlet
    {
        type            fixedValue;
        value           uniform 0;
    }
    walls
    {
        type            zeroGradient;
    }
    "(top|bottom)"
    {
        type            zeroGradient;
    }
    "(front|back)"
    {
        type            empty;
    }
}
"""
        (self.case_dir / "0" / "p").write_text(p_content)

        # --- Turbulence fields ---
        self._write_turbulence_bcs(U_ref)

    def _write_turbulence_bcs(self, U_ref: float) -> None:
        """Write turbulence-specific boundary conditions."""
        TI = 0.05  # 5% turbulence intensity
        L_t = 0.01  # turbulent length scale

        k_inlet = 1.5 * (U_ref * TI) ** 2
        omega_inlet = k_inlet ** 0.5 / (0.09 ** 0.25 * L_t)
        epsilon_inlet = 0.09 * k_inlet ** 1.5 / L_t
        nut_inlet = k_inlet / omega_inlet

        if self.model_key.startswith("SA"):
            # nuTilda
            content = f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       volScalarField;
    object      nuTilda;
}}

dimensions      [0 2 -1 0 0 0 0];
internalField   uniform {nut_inlet * 3:.6e};

boundaryField
{{
    inlet
    {{
        type            fixedValue;
        value           uniform {nut_inlet * 3:.6e};
    }}
    outlet
    {{
        type            zeroGradient;
    }}
    "(walls|top|bottom)"
    {{
        type            fixedValue;
        value           uniform 0;
    }}
    "(front|back)"
    {{
        type            empty;
    }}
}}
"""
            (self.case_dir / "0" / "nuTilda").write_text(content)
        else:
            # k
            k_content = f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       volScalarField;
    object      k;
}}

dimensions      [0 2 -2 0 0 0 0];
internalField   uniform {k_inlet:.6e};

boundaryField
{{
    inlet
    {{
        type            fixedValue;
        value           uniform {k_inlet:.6e};
    }}
    outlet
    {{
        type            zeroGradient;
    }}
    "(walls|top|bottom)"
    {{
        type            kqRWallFunction;
        value           uniform {k_inlet * 1e-4:.6e};
    }}
    "(front|back)"
    {{
        type            empty;
    }}
}}
"""
            (self.case_dir / "0" / "k").write_text(k_content)

            # omega
            omega_content = f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       volScalarField;
    object      omega;
}}

dimensions      [0 0 -1 0 0 0 0];
internalField   uniform {omega_inlet:.6e};

boundaryField
{{
    inlet
    {{
        type            fixedValue;
        value           uniform {omega_inlet:.6e};
    }}
    outlet
    {{
        type            zeroGradient;
    }}
    "(walls|top|bottom)"
    {{
        type            omegaWallFunction;
        value           uniform {omega_inlet:.6e};
    }}
    "(front|back)"
    {{
        type            empty;
    }}
}}
"""
            (self.case_dir / "0" / "omega").write_text(omega_content)

            # nut
            nut_content = f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       volScalarField;
    object      nut;
}}

dimensions      [0 2 -1 0 0 0 0];
internalField   uniform 0;

boundaryField
{{
    inlet
    {{
        type            calculated;
        value           uniform 0;
    }}
    outlet
    {{
        type            calculated;
        value           uniform 0;
    }}
    "(walls|top|bottom)"
    {{
        type            nutkWallFunction;
        value           uniform 0;
    }}
    "(front|back)"
    {{
        type            empty;
    }}
}}
"""
            (self.case_dir / "0" / "nut").write_text(nut_content)

    # -------------------------------------------------------------------------
    # Execution
    # -------------------------------------------------------------------------
    def run(self, timeout: int = 7200) -> RunResult:
        """
        Execute the OpenFOAM solver.

        Parameters
        ----------
        timeout : int
            Maximum wall time in seconds.

        Returns
        -------
        RunResult
        """
        result = RunResult(
            case_dir=str(self.case_dir),
            model=self.model_key,
            mesh_level=self.mesh_level,
        )

        is_les = self.model_config and self.model_config.model_type.value in [
            "Hybrid RANS-LES", "Wall-Modeled LES"
        ]
        solver = "pimpleFoam" if is_les else "simpleFoam"

        # Decompose if parallel
        if self.n_procs > 1:
            self._write_decompose_par_dict()
            self._run_cmd(["decomposePar"], self.case_dir)

        # Build command
        if self.n_procs > 1:
            cmd = ["mpirun", "-np", str(self.n_procs), solver, "-parallel"]
        else:
            cmd = [solver]

        logger.info(f"Running: {' '.join(cmd)} in {self.case_dir}")
        t0 = time.time()

        try:
            with open(self._log_file, "w") as log:
                proc = subprocess.run(
                    cmd, cwd=self.case_dir,
                    stdout=log, stderr=subprocess.STDOUT,
                    timeout=timeout,
                )
            result.wall_time_s = time.time() - t0
            result.log_file = str(self._log_file)

            # Parse residuals
            self._parse_log()
            result.converged = self.residuals.converged
            result.iterations = len(self.residuals.iteration)
            result.final_residuals = self._get_final_residuals()

        except subprocess.TimeoutExpired:
            result.error = f"Timeout after {timeout}s"
            result.wall_time_s = timeout
            logger.warning(result.error)
        except FileNotFoundError:
            result.error = f"Solver '{solver}' not found. Is OpenFOAM installed?"
            logger.error(result.error)
        except Exception as e:
            result.error = str(e)
            logger.error(f"Solver failed: {e}")

        # Reconstruct if parallel
        if self.n_procs > 1:
            self._run_cmd(["reconstructPar", "-latestTime"], self.case_dir)

        return result

    def _write_decompose_par_dict(self) -> None:
        """Write system/decomposeParDict for parallel runs."""
        # Simple scotch for most cases
        content = f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      decomposeParDict;
}}

numberOfSubdomains  {self.n_procs};
method              scotch;
"""
        (self.case_dir / "system" / "decomposeParDict").write_text(content)

    def _run_cmd(self, cmd: List[str], cwd: Path) -> subprocess.CompletedProcess:
        """Run a shell command and log output."""
        logger.info(f"  Running: {' '.join(cmd)}")
        try:
            return subprocess.run(
                cmd, cwd=cwd,
                capture_output=True, text=True, timeout=300,
            )
        except FileNotFoundError:
            logger.warning(f"Command not found: {cmd[0]}")
            return subprocess.CompletedProcess(cmd, 1, "", f"{cmd[0]} not found")

    # -------------------------------------------------------------------------
    # Log Parsing & Convergence
    # -------------------------------------------------------------------------
    def _parse_log(self) -> None:
        """Parse OpenFOAM log file for residual history."""
        if not self._log_file.exists():
            return

        self.residuals = ResidualHistory()
        iteration = 0

        # Regex patterns for SIMPLE residual output
        re_residual = re.compile(
            r"Solving for (\w+), Initial residual = ([\d.eE+-]+)"
        )
        re_time = re.compile(r"^Time = (\d+)")

        with open(self._log_file) as f:
            for line in f:
                time_match = re_time.match(line)
                if time_match:
                    iteration = int(time_match.group(1))
                    self.residuals.iteration.append(iteration)

                res_match = re_residual.search(line)
                if res_match:
                    var = res_match.group(1)
                    val = float(res_match.group(2))
                    if var == "Ux":
                        self.residuals.Ux.append(val)
                    elif var == "Uy":
                        self.residuals.Uy.append(val)
                    elif var == "p":
                        self.residuals.p.append(val)
                    elif var in ("k", "omega", "epsilon", "nuTilda"):
                        self.residuals.turbulence.append(val)

    def _get_final_residuals(self) -> Dict[str, float]:
        """Return the last residual value for each tracked field."""
        finals = {}
        for field_name in ["Ux", "Uy", "p", "turbulence"]:
            vals = getattr(self.residuals, field_name, [])
            if vals:
                finals[field_name] = vals[-1]
        return finals

    def check_convergence(self, target: float = 1e-5) -> Dict[str, Any]:
        """
        Check convergence status against target residual.

        Returns
        -------
        dict with keys: converged, iterations, final_residuals, rate
        """
        self._parse_log()
        result = {
            "converged": self.residuals.converged,
            "iterations": len(self.residuals.iteration),
            "final_residuals": self._get_final_residuals(),
        }

        # Convergence rate (last 100 iterations)
        if len(self.residuals.p) > 100:
            recent = self.residuals.p[-100:]
            if recent[0] > 0 and recent[-1] > 0:
                result["convergence_rate"] = np.log(recent[-1] / recent[0]) / 100

        return result

    # -------------------------------------------------------------------------
    # Post-processing hooks
    # -------------------------------------------------------------------------
    def extract_wall_data(self) -> Optional[Dict]:
        """Run postProcess to extract wall data."""
        try:
            self._run_cmd(
                ["simpleFoam", "-postProcess", "-func", "wallShearStress", "-latestTime"],
                self.case_dir,
            )
            self._run_cmd(
                ["postProcess", "-func", "yPlus", "-latestTime"],
                self.case_dir,
            )
            return {"status": "extracted", "case_dir": str(self.case_dir)}
        except Exception as e:
            logger.error(f"Post-processing failed: {e}")
            return None

    def get_latest_time(self) -> Optional[str]:
        """Find the latest time directory."""
        time_dirs = []
        for d in self.case_dir.iterdir():
            if d.is_dir():
                try:
                    t = float(d.name)
                    time_dirs.append((t, d.name))
                except ValueError:
                    continue
        if time_dirs:
            return sorted(time_dirs)[-1][1]
        return None


# =============================================================================
# Convenience Functions
# =============================================================================
def run_openfoam_case(
    case_dir: Path,
    case_name: str,
    model: str,
    mesh_level: str = "medium",
    scheme_id: int = 1,
    n_procs: int = 1,
) -> RunResult:
    """One-shot function to setup and run an OpenFOAM case."""
    runner = OpenFOAMRunner(case_dir, case_name, model, mesh_level, scheme_id, n_procs)
    runner.setup_case()
    return runner.run()


def setup_scheme_sensitivity_study(
    base_dir: Path,
    case_name: str,
    model: str,
    mesh_level: str = "fine",
) -> List[OpenFOAMRunner]:
    """
    Create cases for all 4 numerical scheme variants.

    Returns list of configured runners (not yet executed).
    """
    _, _, _, _, scheme_matrix = _load_config()
    runners = []
    for i, scheme in enumerate(scheme_matrix):
        case_dir = base_dir / f"{case_name}_{model}_{mesh_level}_scheme{i}"
        runner = OpenFOAMRunner(case_dir, case_name, model, mesh_level, i)
        runner.setup_case()
        runners.append(runner)
        logger.info(f"  Scheme {i}: {scheme['label']}")
    return runners


# =============================================================================
# CLI
# =============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="OpenFOAM Solver Runner")
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--case-name", default="backward_facing_step")
    parser.add_argument("--model", default="SST")
    parser.add_argument("--mesh-level", default="medium")
    parser.add_argument("--scheme-id", type=int, default=1)
    parser.add_argument("--n-procs", type=int, default=1)
    parser.add_argument("--setup-only", action="store_true",
                       help="Only set up the case, do not run")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    runner = OpenFOAMRunner(
        args.case_dir, args.case_name, args.model,
        args.mesh_level, args.scheme_id, args.n_procs,
    )
    runner.setup_case()

    if not args.setup_only:
        result = runner.run()
        print(f"\n{'='*50}")
        print(f"  Model:      {result.model}")
        print(f"  Mesh:       {result.mesh_level}")
        print(f"  Converged:  {result.converged}")
        print(f"  Iterations: {result.iterations}")
        print(f"  Wall time:  {result.wall_time_s:.1f}s")
        if result.error:
            print(f"  Error:      {result.error}")
        print(f"{'='*50}")
    else:
        print(f"Case set up at: {args.case_dir}")
