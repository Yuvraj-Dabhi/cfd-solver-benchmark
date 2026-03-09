"""
SU2 Solver Runner
=================
Manages SU2 case setup, execution, and convergence monitoring.
Supports SA and SST models via SU2's native configuration.

Usage:
    runner = SU2Runner(case_dir, case_name="backward_facing_step", model="SA")
    runner.setup_case()
    runner.run()
"""

import logging
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Import custom exceptions for typed error handling
try:
    from scripts.utils.exceptions import (
        SolverNotFoundError, SolverCrashError, ConvergenceError,
        MeshNotFoundError, DataFormatError,
    )
except ImportError:
    # Graceful fallback if exceptions module not yet available
    SolverNotFoundError = FileNotFoundError
    SolverCrashError = RuntimeError
    ConvergenceError = RuntimeError
    MeshNotFoundError = FileNotFoundError
    DataFormatError = ValueError


def _load_config():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from config import BENCHMARK_CASES, TURBULENCE_MODELS, SOLVER_DEFAULTS
    return BENCHMARK_CASES, TURBULENCE_MODELS, SOLVER_DEFAULTS


@dataclass
class SU2RunResult:
    """Result from a SU2 simulation."""
    case_dir: str
    model: str
    mesh_level: str
    converged: bool = False
    iterations: int = 0
    wall_time_s: float = 0.0
    final_residuals: Dict[str, float] = field(default_factory=dict)
    cl: float = 0.0
    cd: float = 0.0
    error: Optional[str] = None
    log_file: str = ""


# SU2 turbulence model mapping
SU2_TURB_MODELS = {
    "SA":            "SA",
    "SA-QCR":        "SA",  # QCR handled via SA_OPTIONS
    "SA-RC":         "SA",  # Rotation/curvature correction
    "SA-neg":        "SA",  # SA-negative for improved convergence
    "SST":           "SST",
    "kEpsilon":      "KE",
    "gammaReTheta":  "SST",  # SST base + TRANS_MODEL= LM
}


class SU2Runner:
    """
    Manages SU2 case lifecycle: config generation → run → convergence check.

    Parameters
    ----------
    case_dir : Path
        Root directory for the SU2 case.
    case_name : str
        Benchmark case key from config.
    model : str
        Turbulence model key (SA, SST, or kEpsilon).
    mesh_level : str
        Grid level.
    mesh_file : str
        SU2 mesh filename (must exist or be generated separately).
    n_procs : int
        Number of MPI processes.
    """

    def __init__(
        self,
        case_dir: Path,
        case_name: str = "backward_facing_step",
        model: str = "SA",
        mesh_level: str = "medium",
        mesh_file: str = "mesh.su2",
        n_procs: int = 1,
        n_threads: int = 1,
        conv_residual_minval: int = -10,
        enable_dual_time: bool = False,
    ):
        self.case_dir = Path(case_dir)
        self.case_name = case_name
        self.model_key = model
        self.mesh_level = mesh_level
        self.mesh_file = mesh_file
        self.n_procs = n_procs
        self.n_threads = n_threads
        self.conv_residual_minval = conv_residual_minval
        self.enable_dual_time = enable_dual_time

        cases, models, defaults = _load_config()
        self.case_config = cases.get(case_name)
        self.model_config = models.get(model)
        self.solver_defaults = defaults

        self._config_file = self.case_dir / "config.cfg"
        self._log_file = self.case_dir / "log.SU2_CFD"

    def setup_case(self) -> None:
        """Generate SU2 configuration file."""
        self.case_dir.mkdir(parents=True, exist_ok=True)

        # Flow conditions
        Re = self.case_config.reynolds_number if self.case_config else 36000
        M = self.case_config.mach_number if self.case_config else 0.1
        U_ref = self.case_config.reference_velocity if self.case_config and self.case_config.reference_velocity > 0 else 1.0
        L_ref = self.case_config.reference_length if self.case_config else 1.0

        # Determine if compressible
        is_compressible = M > 0.3

        # SU2 model name
        su2_model = SU2_TURB_MODELS.get(self.model_key, "SA")

        # SA variant options (rotation/curvature correction, QCR, negative)
        sa_options = ""
        if self.model_key == "SA-RC":
            sa_options = "SA_OPTIONS= RC"
        elif self.model_key == "SA-QCR":
            sa_options = "SA_OPTIONS= QCR2000"
        elif self.model_key == "SA-neg":
            sa_options = "SA_OPTIONS= NEGATIVE"

        config_lines = [
            "% ==========================================",
            f"% CFD Solver Benchmark: {self.case_name}",
            f"% Model: {self.model_key}, Mesh: {self.mesh_level}",
            "% ==========================================",
            "",
            "% ---------- PROBLEM DEFINITION ----------",
            f"SOLVER= {'RANS' if is_compressible else 'INC_RANS'}",
            f"KIND_TURB_MODEL= {su2_model}",
        ]
        if sa_options:
            config_lines.append(sa_options)
        config_lines += [
            "MATH_PROBLEM= DIRECT",
            "RESTART_SOL= NO",
            "",
            "% ---------- FLOW CONDITIONS ----------",
            f"MACH_NUMBER= {M}",
            f"AOA= 0.0",
            f"SIDESLIP_ANGLE= 0.0",
            f"REYNOLDS_NUMBER= {Re:.0f}",
            f"REYNOLDS_LENGTH= {L_ref}",
            f"FREESTREAM_TEMPERATURE= {getattr(self.case_config, 'temperature', getattr(self.case_config, 'temperature_freestream', 300.0)) if self.case_config else 300.0}",
        ]

        if not is_compressible:
            config_lines += [
                f"INC_VELOCITY_INIT= ({U_ref}, 0.0, 0.0)",
                "INC_DENSITY_INIT= 1.225",
                "INC_TEMPERATURE_INIT= 300.0",
            ]

        config_lines += [
            "",
            "% ---------- TURBULENCE ----------",
            "FREESTREAM_TURBULENCEINTENSITY= 0.05",
            f"FREESTREAM_NU_FACTOR= 3.0" if su2_model == "SA" else f"FREESTREAM_TURB2LAMVISCRATIO= 5.0",
        ]

        # gamma-ReTheta SST transition model
        if self.model_key == "gammaReTheta":
            config_lines += [
                "",
                "% ---------- TRANSITION MODEL ----------",
                "% Menter gamma-ReTheta coupled with SST (Langtry-Menter)",
                "% Adds 2 extra transport equations: intermittency (gamma)",
                "% and transition momentum thickness Reynolds number (ReTheta_t).",
                "TRANS_MODEL= LM",
                "FREESTREAM_INTERMITTENCY= 1.0",
                "FREESTREAM_TURBULENCEINTENSITY= 0.01",
            ]

        config_lines += [
            "% ---------- REFERENCE VALUES ----------",
            f"REF_LENGTH= {L_ref}",
            f"REF_AREA= {L_ref}",
            f"REF_ORIGIN_MOMENT_X= 0.0",
            f"REF_ORIGIN_MOMENT_Y= 0.0",
            f"REF_ORIGIN_MOMENT_Z= 0.0",
            "",
            "% ---------- BOUNDARY CONDITIONS ----------",
            "MARKER_HEATFLUX= (walls, 0.0)",
            f"MARKER_FAR= (inlet, outlet)" if not is_compressible else
            f"MARKER_INLET= (inlet, {getattr(self.case_config, 'temperature', getattr(self.case_config, 'temperature_freestream', 300.0)) if self.case_config else 300.0}, {U_ref}, 1.0, 0.0, 0.0)",
            "MARKER_OUTLET= (outlet, 0.0)",
            "MARKER_SYM= (front, back)",
            f"MARKER_MONITORING= (walls)",
            f"MARKER_PLOTTING= (walls)",
            "",
            "% ---------- NUMERICAL METHOD ----------",
            f"NUM_METHOD_GRAD= GREEN_GAUSS",
            f"CFL_NUMBER= 10.0",
            f"CFL_ADAPT= YES",
            f"CFL_ADAPT_PARAM= (0.5, 1.5, 1.0, 100.0)",
            f"ITER= {self.solver_defaults['max_iterations']}",
            "",
            "% ---------- SPATIAL DISCRETIZATION ----------",
            "% Second-order everywhere: NASA TMR states first-order is",
            "% 'inadequate for verification' (see TMR guidelines).",
            f"CONV_NUM_METHOD_FLOW= ROE",
            f"MUSCL_FLOW= YES",
            f"SLOPE_LIMITER_FLOW= VENKATAKRISHNAN",
            f"CONV_NUM_METHOD_TURB= SCALAR_UPWIND",
            f"MUSCL_TURB= YES",
            f"SLOPE_LIMITER_TURB= VENKATAKRISHNAN",
        ]

        # Dual-time stepping for unsteady / hypersonic cases
        if self.enable_dual_time:
            config_lines += [
                "",
                "% ---------- DUAL TIME STEPPING ----------",
                "% Used for unsteady separated flows and stiff hypersonic problems.",
                "% Reference: NASA Wind-US line-implicit solver strategy.",
                "TIME_MARCHING= DUAL_TIME_STEPPING-2ND_ORDER",
                "TIME_STEP= 1e-5",
                "MAX_TIME= 1.0",
                "TIME_ITER= 1000",
                "INNER_ITER= 50",
                "UNST_CFL_NUMBER= 5.0",
            ]

        config_lines += [
            "",
            "% ---------- CONVERGENCE ----------",
            f"CONV_RESIDUAL_MINVAL= {self.conv_residual_minval}",
            f"CONV_FIELD= RMS_DENSITY" if is_compressible else f"CONV_FIELD= RMS_PRESSURE",
            f"CONV_STARTITER= 10",
            "",
            "% ---------- INPUT / OUTPUT ----------",
            f"MESH_FILENAME= {self.mesh_file}",
            f"MESH_FORMAT= SU2",
            f"SOLUTION_FILENAME= restart_flow.dat",
            f"OUTPUT_FILES= RESTART, PARAVIEW",
            f"OUTPUT_WRT_FREQ= 100",
            f"SCREEN_OUTPUT= INNER_ITER, RMS_DENSITY, RMS_ENERGY, LIFT, DRAG" if is_compressible
            else f"SCREEN_OUTPUT= INNER_ITER, RMS_PRESSURE, RMS_VELOCITY-X, LIFT, DRAG",
            f"HISTORY_OUTPUT= ITER, RMS_RES, AERO_COEFF",
            f"CONV_FILENAME= history",
            f"VOLUME_FILENAME= flow",
            f"SURFACE_FILENAME= surface_flow",
        ]

        self._config_file.write_text("\n".join(config_lines) + "\n")
        logger.info(f"SU2 config written to {self._config_file}")

    def run(self, timeout: int = 7200) -> SU2RunResult:
        """Execute SU2_CFD."""
        result = SU2RunResult(
            case_dir=str(self.case_dir),
            model=self.model_key,
            mesh_level=self.mesh_level,
        )

        cmd = ["SU2_CFD"]
        if self.n_threads > 1:
            cmd.extend(["-t", str(self.n_threads)])
        cmd.append(str(self._config_file))

        if self.n_procs > 1:
            mpi_cmd = shutil.which("mpiexec") or shutil.which("mpirun") or "mpirun"
            cmd = [mpi_cmd, "-np", str(self.n_procs)] + cmd

        logger.info(f"Running SU2: {' '.join(cmd)}")
        t0 = time.time()

        try:
            with open(self._log_file, "w") as log:
                subprocess.run(
                    cmd, cwd=self.case_dir,
                    stdout=log, stderr=subprocess.STDOUT,
                    timeout=timeout,
                )
            result.wall_time_s = time.time() - t0
            result.log_file = str(self._log_file)

            # Parse history file
            self._parse_history(result)

        except subprocess.TimeoutExpired:
            result.error = f"Timeout after {timeout}s"
            result.wall_time_s = timeout
            logger.warning(f"SU2 timed out after {timeout}s in {self.case_dir}")
        except FileNotFoundError:
            result.error = "SU2_CFD not found. Is SU2 installed and in PATH?"
            logger.error(result.error)
        except subprocess.CalledProcessError as e:
            result.error = f"SU2 exited with code {e.returncode}"
            logger.error(f"SU2 crashed: exit code {e.returncode} in {self.case_dir}")
        except OSError as e:
            result.error = f"OS error running SU2: {e}"
            logger.error(result.error)
        except Exception as e:
            result.error = f"Unexpected error: {type(e).__name__}: {e}"
            logger.error(f"SU2 failed unexpectedly: {e}", exc_info=True)

        return result

    def _parse_history(self, result: SU2RunResult) -> None:
        """Parse SU2 history.csv for convergence data."""
        history_file = self.case_dir / "history.csv"
        if not history_file.exists():
            return

        try:
            data = np.genfromtxt(history_file, delimiter=",", names=True, skip_header=0)
            if len(data) > 0:
                result.iterations = len(data)
                # Try to extract CL/CD
                for cl_name in ["CL", "\"CL\"", "Cl"]:
                    if cl_name in data.dtype.names:
                        result.cl = float(data[cl_name][-1])
                        break
                for cd_name in ["CD", "\"CD\"", "Cd"]:
                    if cd_name in data.dtype.names:
                        result.cd = float(data[cd_name][-1])
                        break

                # Check convergence from residual drop
                for rms_field in data.dtype.names:
                    if "rms" in rms_field.lower() or "RMS" in rms_field:
                        vals = data[rms_field]
                        result.final_residuals[rms_field] = float(vals[-1])
                        if vals[-1] < -7:  # RMS residual < 1e-7
                            result.converged = True
        except (ValueError, IndexError, KeyError) as e:
            logger.warning(f"Could not parse SU2 history ({type(e).__name__}): {e}")
        except Exception as e:
            logger.warning(f"Unexpected error parsing SU2 history: {e}")


# =============================================================================
# Convenience
# =============================================================================
def run_su2_case(
    case_dir: Path,
    case_name: str,
    model: str = "SA",
    mesh_level: str = "medium",
    mesh_file: str = "mesh.su2",
    n_procs: int = 1,
    n_threads: int = 1,
) -> SU2RunResult:
    """One-shot function to setup and run a SU2 case."""
    runner = SU2Runner(case_dir, case_name, model, mesh_level, mesh_file, n_procs, n_threads)
    runner.setup_case()
    return runner.run()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SU2 Solver Runner")
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--case-name", default="backward_facing_step")
    parser.add_argument("--model", default="SA")
    parser.add_argument("--mesh-level", default="medium")
    parser.add_argument("--mesh-file", default="mesh.su2")
    parser.add_argument("--n-procs", type=int, default=1)
    parser.add_argument("--setup-only", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    runner = SU2Runner(
        args.case_dir, args.case_name, args.model,
        args.mesh_level, args.mesh_file, args.n_procs,
    )
    runner.setup_case()

    if not args.setup_only:
        result = runner.run()
        print(f"\nModel: {result.model} | Converged: {result.converged} | "
              f"Iterations: {result.iterations} | Time: {result.wall_time_s:.1f}s")
        if result.error:
            print(f"Error: {result.error}")
    else:
        print(f"SU2 case set up at: {args.case_dir}")
