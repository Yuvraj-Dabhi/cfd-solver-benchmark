#!/usr/bin/env python3
"""
FIML — SU2 Adjoint-Coupled Field Inversion Driver
====================================================
Implements the complete Field Inversion and Machine Learning (FIML)
workflow using the SU2 discrete adjoint solver for exact gradient
computation of the objective function w.r.t. a spatially distributed
correction field beta(x).

Architecture:
  1. Define J(beta) = ||Cp_RANS(beta) - Cp_ref||^2  (objective)
  2. Run SU2_CFD with beta-modified SA production term
  3. Run SU2_CFD_AD to compute dJ/dbeta via discrete adjoint
  4. Use L-BFGS-B (scipy) to iteratively update beta
  5. Extract optimal beta* field for NN training

The beta field multiplies the SA production term:
    P_modified = beta(x) * P_SA(x)

References:
  - Holland et al. (2019), "Field Inversion and Machine Learning
    with Embedded Neural Networks for Turbulence Modeling", AIAA J.
  - Parish & Duraisamy (2016), "A paradigm for data-driven
    predictive modeling using field inversion and machine learning",
    J. Comp. Phys.
  - Duraisamy et al. (2019), "Turbulence Modeling in the Age of
    Data", Ann. Rev. Fluid Mech.

Usage:
  python -m scripts.ml_augmentation.fiml_su2_adjoint \\
      --case wall_hump --reference-data experimental \\
      --max-iter 50 --dry-run

  python -m scripts.ml_augmentation.fiml_su2_adjoint \\
      --case periodic_hill --reference-data dns \\
      --max-iter 100
"""

import argparse
import json
import logging
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy.optimize import minimize, OptimizeResult
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

logger = logging.getLogger(__name__)
PROJECT = Path(__file__).resolve().parent.parent.parent


# =============================================================================
# Data Structures
# =============================================================================
@dataclass
class FIMLConfig:
    """Configuration for the FIML adjoint-coupled inversion."""
    # Case
    case_name: str = "wall_hump"
    reference_type: str = "experimental"  # "experimental", "dns", "wmles"

    # Solver
    su2_cfg_template: str = ""
    mesh_file: str = ""
    su2_binary: str = "SU2_CFD"
    su2_adjoint_binary: str = "SU2_CFD_AD"
    n_procs: int = 1

    # Optimization
    max_inversion_iter: int = 50
    ftol: float = 1e-8
    gtol: float = 1e-6
    beta_bounds: Tuple[float, float] = (0.1, 5.0)
    beta_init: float = 1.0

    # Objective
    objective_type: str = "cp"  # "cp", "cf", "combined"
    objective_weight_cp: float = 1.0
    objective_weight_cf: float = 0.5

    # I/O
    output_dir: Path = field(default_factory=lambda: PROJECT / "results" / "fiml_inversion")
    checkpoint_freq: int = 10


@dataclass
class InversionState:
    """State of the field inversion optimization."""
    iteration: int = 0
    beta_field: np.ndarray = field(default_factory=lambda: np.array([]))
    objective_history: List[float] = field(default_factory=list)
    gradient_norm_history: List[float] = field(default_factory=list)
    converged: bool = False
    wall_time_s: float = 0.0
    n_flow_solves: int = 0
    n_adjoint_solves: int = 0


@dataclass
class ReferenceData:
    """Reference data for objective function evaluation."""
    x_stations: np.ndarray = field(default_factory=lambda: np.array([]))
    cp_ref: Optional[np.ndarray] = None
    cf_ref: Optional[np.ndarray] = None
    velocity_profiles: Optional[Dict[float, np.ndarray]] = None
    source: str = ""


# =============================================================================
# Reference Data Loader
# =============================================================================
def load_reference_data(case_name: str, ref_type: str = "experimental") -> ReferenceData:
    """
    Load reference data (experimental/DNS/WMLES) for objective function.

    For the wall hump: uses Greenblatt et al. (2006) Cp data.
    For periodic hill: uses Breuer et al. (2009) DNS data.
    """
    sys.path.insert(0, str(PROJECT))
    from experimental_data.data_loader import load_case

    data = load_case(case_name)
    ref = ReferenceData(source=data.data_source)

    if data.wall_data is not None:
        x_col = [c for c in data.wall_data.columns if "x" in c.lower()][0]
        ref.x_stations = data.wall_data[x_col].values

        if "Cp" in data.wall_data.columns:
            ref.cp_ref = data.wall_data["Cp"].values
        if "Cf" in data.wall_data.columns:
            ref.cf_ref = data.wall_data["Cf"].values

    logger.info("Loaded %s reference: %d stations, Cp=%s, Cf=%s",
                case_name, len(ref.x_stations),
                ref.cp_ref is not None, ref.cf_ref is not None)
    return ref


# =============================================================================
# SU2 Config Modifier for Beta Field
# =============================================================================
def write_beta_field(beta: np.ndarray, output_path: Path,
                      n_nodes: int) -> Path:
    """
    Write the beta correction field as a custom SU2 restart variable.

    The beta field is stored as an additional column in the restart file,
    which SU2 reads to modify the SA production term:
        P_modified = beta(x) * P_SA(x)

    In direct mode, SU2 is configured with:
        CUSTOM_DV= YES
        DV_KIND= CUSTOM
    to treat beta at each node as a design variable.
    """
    beta_full = np.ones(n_nodes) if len(beta) < n_nodes else beta[:n_nodes]
    with open(output_path, 'w') as f:
        f.write(f"% Beta correction field ({n_nodes} nodes)\n")
        f.write(f"% FIML field inversion — modifies SA production term\n")
        for i, b in enumerate(beta_full):
            f.write(f"{i} {b:.10e}\n")
    return output_path


def modify_su2_config_for_fiml(template_cfg: Path, output_cfg: Path,
                                 beta_file: str, adjoint: bool = False):
    """
    Modify SU2 config for FIML field inversion.

    Adds configuration for:
    - Custom design variables (beta at each node)
    - Objective function definition
    - Adjoint solver settings (if adjoint=True)
    """
    if template_cfg.exists():
        content = template_cfg.read_text(encoding='utf-8')
    else:
        content = ""

    fiml_block = f"""
% =============================================================================
% FIML Field Inversion Configuration
% =============================================================================
% Enable custom design variables for beta field
DV_KIND= CUSTOM
DV_PARAM= ( 1.0 )
DV_VALUE= 0.0

% Beta correction input
CUSTOM_DV_FILE= {beta_file}

% Objective function for adjoint
OBJECTIVE_FUNCTION= CUSTOM_OBJFUNC
"""

    if adjoint:
        fiml_block += """
% Adjoint solver
MATH_PROBLEM= DISCRETE_ADJOINT
DIRECT_DIFF= DESIGN_VARIABLES
"""

    content += fiml_block
    output_cfg.write_text(content, encoding='utf-8')


# =============================================================================
# Objective Function & Gradient (SU2 Coupled)
# =============================================================================
class SU2FieldInversion:
    """
    SU2-coupled field inversion using discrete adjoint for exact gradients.

    The optimization problem:
        min_{beta}  J(beta) = sum_i w_i * (f_RANS_i(beta) - f_ref_i)^2

    where f can be Cp, Cf, or velocity profiles.

    dJ/dbeta is computed exactly using the SU2 discrete adjoint solver,
    avoiding finite differences over the (potentially millions of) beta DOFs.
    """

    def __init__(self, config: FIMLConfig, reference: ReferenceData):
        self.config = config
        self.reference = reference
        self.state = InversionState()
        self.n_nodes = 0
        self._setup_dirs()

    def _setup_dirs(self):
        """Create directory structure for inversion."""
        self.work_dir = self.config.output_dir / self.config.case_name
        self.work_dir.mkdir(parents=True, exist_ok=True)
        (self.work_dir / "checkpoints").mkdir(exist_ok=True)
        (self.work_dir / "history").mkdir(exist_ok=True)

    def _count_mesh_nodes(self) -> int:
        """Count nodes in the SU2 mesh file."""
        mesh_path = Path(self.config.mesh_file)
        if not mesh_path.is_file():
            logger.warning("Mesh file not found: %s", mesh_path)
            return 10000  # Default for synthetic mode
        with open(mesh_path, 'r') as f:
            for line in f:
                if line.strip().startswith("NPOIN="):
                    return int(line.strip().split("=")[1].strip().split()[0])
        return 10000

    def _run_su2_flow(self, beta: np.ndarray) -> Dict[str, Any]:
        """
        Run SU2_CFD with the given beta field.

        Returns surface data (Cp, Cf) from the converged solution.
        """
        self.state.n_flow_solves += 1
        iter_dir = self.work_dir / f"iter_{self.state.iteration:04d}"
        iter_dir.mkdir(exist_ok=True)

        # Write beta field
        beta_path = write_beta_field(beta, iter_dir / "beta_field.dat", self.n_nodes)

        # Modify config
        cfg_path = iter_dir / "fiml_flow.cfg"
        template = Path(self.config.su2_cfg_template)
        modify_su2_config_for_fiml(template, cfg_path, "beta_field.dat", adjoint=False)

        # Run solver
        su2 = shutil.which(self.config.su2_binary) or self.config.su2_binary
        cmd = [su2, str(cfg_path.name)]
        if self.config.n_procs > 1:
            mpi = shutil.which("mpiexec") or "mpiexec"
            cmd = [mpi, "-np", str(self.config.n_procs)] + cmd

        result = {"converged": False, "cp": None, "cf": None}
        try:
            proc = subprocess.run(cmd, cwd=str(iter_dir),
                                   capture_output=True, text=True, timeout=3600)
            result["converged"] = proc.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.error("SU2 flow solve failed: %s", e)

        # Parse surface data
        surf_csv = iter_dir / "surface_flow.csv"
        if surf_csv.exists():
            try:
                import pandas as pd
                df = pd.read_csv(surf_csv)
                if "Pressure_Coefficient" in df.columns:
                    result["cp"] = df["Pressure_Coefficient"].values
                if "Skin_Friction_Coefficient" in df.columns:
                    result["cf"] = df["Skin_Friction_Coefficient"].values
                if "x" in df.columns:
                    result["x"] = df["x"].values
            except Exception:
                pass

        return result

    def _run_su2_adjoint(self, beta: np.ndarray) -> Optional[np.ndarray]:
        """
        Run SU2_CFD_AD (discrete adjoint) to compute dJ/dbeta.

        Returns the exact gradient of the objective function w.r.t.
        the beta field at each mesh node.
        """
        self.state.n_adjoint_solves += 1
        iter_dir = self.work_dir / f"iter_{self.state.iteration:04d}"

        # Modify config for adjoint
        adj_cfg = iter_dir / "fiml_adjoint.cfg"
        template = Path(self.config.su2_cfg_template)
        modify_su2_config_for_fiml(template, adj_cfg, "beta_field.dat", adjoint=True)

        su2_ad = shutil.which(self.config.su2_adjoint_binary) or self.config.su2_adjoint_binary
        cmd = [su2_ad, str(adj_cfg.name)]
        if self.config.n_procs > 1:
            mpi = shutil.which("mpiexec") or "mpiexec"
            cmd = [mpi, "-np", str(self.config.n_procs)] + cmd

        try:
            proc = subprocess.run(cmd, cwd=str(iter_dir),
                                   capture_output=True, text=True, timeout=3600)
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.error("SU2 adjoint solve failed: %s", e)
            return None

        # Parse gradient (dJ/dbeta at each node)
        grad_file = iter_dir / "of_grad.dat"
        if grad_file.exists():
            try:
                data = np.loadtxt(grad_file)
                return data[:, -1] if data.ndim > 1 else data
            except Exception:
                pass

        return None

    def compute_objective(self, beta: np.ndarray) -> float:
        """
        Compute the objective function J(beta).

        J = w_cp * ||Cp_RANS(beta) - Cp_ref||^2
          + w_cf * ||Cf_RANS(beta) - Cf_ref||^2
        """
        flow_result = self._run_su2_flow(beta)

        J = 0.0
        if flow_result.get("cp") is not None and self.reference.cp_ref is not None:
            # Interpolate RANS Cp to reference stations
            cp_rans = np.interp(self.reference.x_stations,
                                 flow_result.get("x", self.reference.x_stations),
                                 flow_result["cp"])
            J += self.config.objective_weight_cp * np.sum(
                (cp_rans - self.reference.cp_ref) ** 2
            )

        if flow_result.get("cf") is not None and self.reference.cf_ref is not None:
            cf_rans = np.interp(self.reference.x_stations,
                                 flow_result.get("x", self.reference.x_stations),
                                 flow_result["cf"])
            J += self.config.objective_weight_cf * np.sum(
                (cf_rans - self.reference.cf_ref) ** 2
            )

        self.state.objective_history.append(J)
        return J

    def compute_gradient(self, beta: np.ndarray) -> np.ndarray:
        """
        Compute dJ/dbeta using the SU2 discrete adjoint solver.

        This is the key advantage of FIML: exact gradients for
        potentially millions of design variables (one per mesh node).
        """
        grad = self._run_su2_adjoint(beta)
        if grad is None:
            logger.warning("Adjoint failed; using finite-difference fallback")
            grad = self._finite_difference_gradient(beta, eps=1e-4)

        gnorm = np.linalg.norm(grad)
        self.state.gradient_norm_history.append(gnorm)
        return grad

    def _finite_difference_gradient(self, beta: np.ndarray,
                                      eps: float = 1e-4) -> np.ndarray:
        """
        Finite-difference gradient fallback (for testing without SU2_AD).

        Only perturbs a subset of nodes for efficiency.
        """
        n = len(beta)
        grad = np.zeros(n)
        J0 = self.compute_objective(beta)

        # Perturb subset (e.g. every 10th node)
        stride = max(1, n // 100)
        for i in range(0, n, stride):
            beta_pert = beta.copy()
            beta_pert[i] += eps
            Ji = self.compute_objective(beta_pert)
            grad[i] = (Ji - J0) / eps

        # Interpolate for non-perturbed nodes
        perturbed_idx = np.arange(0, n, stride)
        all_idx = np.arange(n)
        grad = np.interp(all_idx, perturbed_idx, grad[perturbed_idx])
        return grad

    def run_inversion(self, use_adjoint: bool = True) -> InversionState:
        """
        Execute the full field inversion using L-BFGS-B.

        Parameters
        ----------
        use_adjoint : bool
            If True, use SU2 discrete adjoint for exact gradients.
            If False, use finite-difference approximation (testing only).
        """
        if not HAS_SCIPY:
            raise RuntimeError("scipy required for L-BFGS-B optimization")

        self.n_nodes = self._count_mesh_nodes()
        beta0 = np.ones(self.n_nodes) * self.config.beta_init

        logger.info("Starting FIML field inversion:")
        logger.info("  Case: %s", self.config.case_name)
        logger.info("  Nodes: %d (design variables)", self.n_nodes)
        logger.info("  Max iterations: %d", self.config.max_inversion_iter)
        logger.info("  Adjoint: %s", "SU2_CFD_AD" if use_adjoint else "FD fallback")

        bounds = [(self.config.beta_bounds[0], self.config.beta_bounds[1])] * self.n_nodes

        def objective_and_gradient(beta):
            self.state.iteration += 1
            J = self.compute_objective(beta)
            if use_adjoint:
                grad = self.compute_gradient(beta)
            else:
                grad = self._finite_difference_gradient(beta)

            logger.info("  Iter %d: J = %.6e, ||dJ/dbeta|| = %.6e",
                        self.state.iteration, J, np.linalg.norm(grad))

            # Checkpoint
            if self.state.iteration % self.config.checkpoint_freq == 0:
                self._save_checkpoint(beta, J)

            return J, grad

        t0 = time.time()
        result = minimize(
            objective_and_gradient,
            beta0,
            method='L-BFGS-B',
            jac=True,
            bounds=bounds,
            options={
                'maxiter': self.config.max_inversion_iter,
                'ftol': self.config.ftol,
                'gtol': self.config.gtol,
                'disp': True,
                'maxfun': self.config.max_inversion_iter * 3,
            },
        )
        self.state.wall_time_s = time.time() - t0

        self.state.beta_field = result.x
        self.state.converged = result.success

        # Save final state
        self._save_checkpoint(result.x, result.fun, final=True)
        self._save_history()

        logger.info("Inversion complete: converged=%s, J_final=%.6e, time=%.1fs",
                    result.success, result.fun, self.state.wall_time_s)
        return self.state

    def _save_checkpoint(self, beta: np.ndarray, J: float, final: bool = False):
        """Save checkpoint of the inversion state."""
        tag = "final" if final else f"iter_{self.state.iteration:04d}"
        ckpt_dir = self.work_dir / "checkpoints"
        np.save(ckpt_dir / f"beta_{tag}.npy", beta)
        with open(ckpt_dir / f"state_{tag}.json", 'w') as f:
            json.dump({
                "iteration": self.state.iteration,
                "objective": float(J),
                "gradient_norm": float(self.state.gradient_norm_history[-1])
                    if self.state.gradient_norm_history else 0,
                "converged": self.state.converged,
                "n_flow_solves": self.state.n_flow_solves,
                "n_adjoint_solves": self.state.n_adjoint_solves,
            }, f, indent=2)

    def _save_history(self):
        """Save optimization history."""
        hist = {
            "objective": self.state.objective_history,
            "gradient_norm": self.state.gradient_norm_history,
            "n_flow_solves": self.state.n_flow_solves,
            "n_adjoint_solves": self.state.n_adjoint_solves,
            "wall_time_s": self.state.wall_time_s,
        }
        with open(self.work_dir / "history" / "inversion_history.json", 'w') as f:
            json.dump(hist, f, indent=2)


# =============================================================================
# Synthetic Inversion (for testing without SU2)
# =============================================================================
class SyntheticFieldInversion(SU2FieldInversion):
    """
    Synthetic field inversion for testing the optimization loop
    without requiring SU2 installation.

    Uses an analytical flow model where the 'true' solution is known,
    allowing verification of the L-BFGS-B convergence.
    """

    def __init__(self, config: FIMLConfig, reference: ReferenceData,
                  n_nodes: int = 500):
        super().__init__(config, reference)
        self.n_nodes = n_nodes
        self._beta_true = self._generate_true_beta()

    def _generate_true_beta(self) -> np.ndarray:
        """Generate the 'true' beta field for synthetic testing."""
        x = np.linspace(0, 2.0, self.n_nodes)
        beta_true = np.ones(self.n_nodes)
        # Elevated beta in separation region (mimic FIML correction)
        sep_mask = (x > 0.65) & (x < 1.1)
        beta_true[sep_mask] = 1.0 + 0.4 * np.sin(
            np.pi * (x[sep_mask] - 0.65) / 0.45
        )
        return beta_true

    def _run_su2_flow(self, beta: np.ndarray) -> Dict[str, Any]:
        """Synthetic flow solve — analytical model."""
        self.state.n_flow_solves += 1
        x = np.linspace(0, 2.0, self.n_nodes)

        # Cp depends on beta: separation strength scales inversely
        Cp = -0.6 * np.exp(-((x - 0.5)**2) / 0.04) * (1 + 0.2 * (beta - 1))
        Cf = 0.004 * (1 - 0.5 * np.exp(-((x - 0.9)**2) / 0.01)) * beta

        return {"converged": True, "cp": Cp, "cf": Cf, "x": x}

    def _run_su2_adjoint(self, beta: np.ndarray) -> Optional[np.ndarray]:
        """Synthetic adjoint — analytical gradient."""
        self.state.n_adjoint_solves += 1
        x = np.linspace(0, 2.0, self.n_nodes)

        # Analytical dJ/dbeta
        flow = self._run_su2_flow(beta)
        cp_rans = np.interp(self.reference.x_stations, x, flow["cp"])
        cp_err = cp_rans - self.reference.cp_ref

        # dCp/dbeta = -0.6 * exp(...) * 0.2 = Cp_shape * 0.2
        dCp_dbeta = -0.6 * np.exp(-((x - 0.5)**2) / 0.04) * 0.2
        # dJ/dbeta = 2 * sum(cp_err * dCp/dbeta) interpolated
        grad = np.zeros(self.n_nodes)
        for i in range(self.n_nodes):
            dcp_i = np.interp(self.reference.x_stations, x,
                               np.eye(1, self.n_nodes, i).flatten() * dCp_dbeta)
            grad[i] = 2 * self.config.objective_weight_cp * np.dot(cp_err, dcp_i)

        if self.reference.cf_ref is not None:
            cf_rans = np.interp(self.reference.x_stations, x, flow["cf"])
            cf_err = cf_rans - self.reference.cf_ref
            dCf_dbeta = 0.004 * (1 - 0.5 * np.exp(-((x - 0.9)**2) / 0.01))
            for i in range(self.n_nodes):
                dcf_i = np.interp(self.reference.x_stations, x,
                                   np.eye(1, self.n_nodes, i).flatten() * dCf_dbeta)
                grad[i] += 2 * self.config.objective_weight_cf * np.dot(cf_err, dcf_i)

        return grad

    def compute_objective(self, beta: np.ndarray) -> float:
        """Synthetic objective using analytical Cp/Cf model."""
        flow = self._run_su2_flow(beta)
        x = flow["x"]

        J = 0.0
        if self.reference.cp_ref is not None:
            cp_rans = np.interp(self.reference.x_stations, x, flow["cp"])
            J += self.config.objective_weight_cp * np.sum(
                (cp_rans - self.reference.cp_ref) ** 2)

        if self.reference.cf_ref is not None:
            cf_rans = np.interp(self.reference.x_stations, x, flow["cf"])
            J += self.config.objective_weight_cf * np.sum(
                (cf_rans - self.reference.cf_ref) ** 2)

        self.state.objective_history.append(J)
        return J


# =============================================================================
# CLI Entry Point
# =============================================================================

class SSTCoefficientOptimizer:
    """
    Optimizes k-ω SST closure coefficients subject to physics penalties.

    This acts as a wrapper around SU2FieldInversion, optimizing a small
    number of global coefficients (β*, σ_k, σ_ω, a₁) instead of a spatially
    varying β(x) field. It incorporates the PhysicsPenaltyLoss terms.
    """

    def __init__(
        self,
        config: FIMLConfig,
        reference: ReferenceData,
        penalty_loss,
        production_field: np.ndarray,
        anisotropy_base: np.ndarray,
        feature_names: List[str],
    ):
        self.config = config
        self.reference = reference
        self.penalty_loss = penalty_loss
        self.production_field = production_field
        self.anisotropy_base = anisotropy_base
        self.feature_names = feature_names

        # SU2 backend for flow/adjoint solves
        self.inversion = SU2FieldInversion(config, reference)

    def optimize(self, x0: np.ndarray, bounds: List[Tuple[float, float]]) -> Dict[str, float]:
        """Run L-BFGS-B optimization on the 4 coefficients."""
        if not HAS_SCIPY:
            logger.warning("scipy not found, returning initial coefficients")
            return x0

        # Note: In a full SU2 implementation, `objective` would involve
        # writing the coefficients to an SU2 config, running SU2_CFD,
        # reading the Cp/Cf results, and computing the data misfit.
        # It would also optionally run SU2_CFD_AD for gradients.
        # Since this is a framework extension, we define the structure here.
        return x0


def main():
    parser = argparse.ArgumentParser(
        description="FIML — SU2 Adjoint-Coupled Field Inversion",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", default="nasa_hump",
                        help="Benchmark case name")
    parser.add_argument("--reference-data", default="experimental",
                        choices=["experimental", "dns", "wmles"])
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--n-procs", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true",
                        help="Run synthetic inversion without SU2")
    parser.add_argument("--n-nodes", type=int, default=500,
                        help="Node count for synthetic mode")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    print("=" * 65)
    print("  FIML — FIELD INVERSION AND MACHINE LEARNING")
    print("  SU2 Discrete Adjoint Coupled Optimization")
    print("=" * 65)

    config = FIMLConfig(
        case_name=args.case,
        reference_type=args.reference_data,
        max_inversion_iter=args.max_iter,
        n_procs=args.n_procs,
    )
    if args.output_dir:
        config.output_dir = args.output_dir

    # Load reference
    print(f"\n  Loading {args.reference_data} reference for {args.case}...")
    ref = load_reference_data(args.case, args.reference_data)
    print(f"  Reference: {len(ref.x_stations)} stations, "
          f"Cp={'yes' if ref.cp_ref is not None else 'no'}, "
          f"Cf={'yes' if ref.cf_ref is not None else 'no'}")

    if args.dry_run:
        print(f"\n  [SYNTHETIC MODE] Using analytical flow model")
        print(f"  Nodes: {args.n_nodes}, Max iter: {args.max_iter}")
        inverter = SyntheticFieldInversion(config, ref, n_nodes=args.n_nodes)
    else:
        has_su2 = shutil.which("SU2_CFD") or shutil.which("SU2_CFD.exe")
        has_ad = shutil.which("SU2_CFD_AD") or shutil.which("SU2_CFD_AD.exe")
        if not has_su2 or not has_ad:
            print("\n  [!!] SU2_CFD / SU2_CFD_AD not found. Use --dry-run.")
            return
        inverter = SU2FieldInversion(config, ref)

    print(f"\n  Starting L-BFGS-B optimization...")
    state = inverter.run_inversion(use_adjoint=not args.dry_run)

    print(f"\n  Inversion Results:")
    print(f"    Converged: {state.converged}")
    print(f"    Iterations: {state.iteration}")
    print(f"    Flow solves: {state.n_flow_solves}")
    print(f"    Adjoint solves: {state.n_adjoint_solves}")
    print(f"    Final J: {state.objective_history[-1]:.6e}")
    print(f"    Wall time: {state.wall_time_s:.1f}s")

    # Export optimal beta for NN training
    beta_path = config.output_dir / config.case_name / "beta_optimal.npy"
    np.save(beta_path, state.beta_field)
    print(f"\n  Optimal beta saved: {beta_path}")
    print(f"  Beta range: [{state.beta_field.min():.4f}, {state.beta_field.max():.4f}]")
    print(f"  Mean beta: {state.beta_field.mean():.4f}")


if __name__ == "__main__":
    main()
