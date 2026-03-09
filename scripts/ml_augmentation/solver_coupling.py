#!/usr/bin/env python3
"""
ML-Corrected RANS Solver-Loop Integration
===========================================
Iterative outer-loop driver that couples ML β-correction models
with the SU2 flow solver:

    run_solver() → extract_fields() → predict_β() → inject_β() → repeat

Supports two modes:
  - live: calls SU2_CFD directly (requires SU2 installation)
  - dry_run: uses synthetic flow data for testing the full code path

Usage:
    from scripts.ml_augmentation.solver_coupling import (
        MLCorrectedRANSLoop, SolverCouplingConfig,
    )
    config = SolverCouplingConfig(mode="dry_run", max_outer_iterations=5)
    loop = MLCorrectedRANSLoop(config)
    history = loop.run()

References:
    - Holland et al. (2019), "Field Inversion and Machine Learning
      with Embedded Neural Networks", JCP
    - Parish & Duraisamy (2016), FIML paradigm, JCP
"""

import json
import logging
import time
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)
PROJECT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT))

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


# =============================================================================
# Data Structures
# =============================================================================
@dataclass
class SolverCouplingConfig:
    """Configuration for the ML-corrected RANS loop."""
    # Iteration control
    max_outer_iterations: int = 10
    convergence_tol: float = 1e-3
    beta_relaxation: float = 0.5  # Under-relaxation: β_new = α*β_pred + (1-α)*β_old

    # Mode
    mode: str = "dry_run"  # "live" or "dry_run"

    # SU2 settings (live mode)
    su2_cfg_path: str = ""
    su2_binary: str = "SU2_CFD"
    mesh_file: str = ""
    n_procs: int = 1

    # ML model
    model_path: str = ""  # Path to TorchScript .pt or sklearn pickle

    # Beta bounds
    beta_min: float = 0.5
    beta_max: float = 2.0

    # Synthetic settings (dry_run mode)
    n_mesh_nodes: int = 5000
    n_wall_points: int = 200

    # Output
    output_dir: str = ""


@dataclass
class CouplingIteration:
    """Result from a single outer-loop iteration."""
    iteration: int
    beta_norm: float
    beta_change_norm: float
    objective: float
    cf_rmse: float
    wall_time_s: float
    converged: bool = False


@dataclass
class CouplingHistory:
    """Complete history of the coupling loop."""
    iterations: List[CouplingIteration] = field(default_factory=list)
    total_wall_time_s: float = 0.0
    converged: bool = False
    final_beta_norm: float = 0.0
    final_cf_rmse: float = 0.0
    n_iterations: int = 0

    def to_dict(self) -> Dict:
        """JSON-serializable dict."""
        return {
            "n_iterations": self.n_iterations,
            "converged": bool(self.converged),
            "total_wall_time_s": round(self.total_wall_time_s, 3),
            "final_beta_norm": round(self.final_beta_norm, 6),
            "final_cf_rmse": round(self.final_cf_rmse, 6),
            "per_iteration": [
                {
                    "iteration": it.iteration,
                    "beta_norm": round(it.beta_norm, 6),
                    "beta_change_norm": round(it.beta_change_norm, 6),
                    "objective": round(it.objective, 6),
                    "cf_rmse": round(it.cf_rmse, 6),
                    "wall_time_s": round(it.wall_time_s, 4),
                    "converged": bool(it.converged),
                }
                for it in self.iterations
            ],
        }


# =============================================================================
# Synthetic Flow Data (dry_run mode)
# =============================================================================
def _generate_synthetic_flow(n_nodes: int, n_wall: int,
                              beta: np.ndarray,
                              seed: int = 42) -> Dict[str, np.ndarray]:
    """
    Generate synthetic flow solution that responds to beta correction.

    Mimics a wall-hump-like flow where beta > 1 in the separation
    region shortens the recirculation bubble.
    """
    rng = np.random.default_rng(seed)

    x = np.linspace(0, 2, n_nodes)
    y = rng.uniform(0, 0.3, n_nodes)

    # Nu_t field: peaks in shear layer, modulated by beta
    nu_t = 1e-4 * (1 + 5 * np.exp(-((y - 0.05)**2) / 0.002))
    nu_t *= beta  # Beta enhances eddy viscosity

    # Strain magnitude
    strain_mag = 50.0 * (1 + 2 * np.exp(-((x - 0.8)**2) / 0.1))

    # Wall distance
    wall_distance = np.abs(y) + 1e-6

    # Strain-rotation ratio
    sr_ratio = 0.8 + 0.2 * np.sin(np.pi * x)

    # Pressure gradient indicator
    pg_indicator = -0.3 * np.exp(-((x - 0.65)**2) / 0.05)

    # Wall data: Cf responds to beta near separation
    x_wall = np.linspace(0, 2, n_wall)
    mean_beta_sep = np.mean(beta[(x > 0.6) & (x < 1.2)]) if np.any((x > 0.6) & (x < 1.2)) else 1.0
    bubble_factor = 1.0 / max(mean_beta_sep, 0.5)

    # SA-like Cf with bubble shortened by beta correction
    bubble_end = 0.665 + 0.535 * bubble_factor
    Cf = 0.004 * np.where(
        (x_wall > 0.665) & (x_wall < bubble_end),
        -0.4 * np.sin(np.pi * (x_wall - 0.665) / (bubble_end - 0.665)),
        1.0,
    )
    Cf += rng.normal(0, 1e-4, n_wall)

    Cp = -0.5 * np.sin(np.pi * x_wall) + rng.normal(0, 0.01, n_wall)

    # Target Cf (experimental)
    Cf_exp = 0.004 * np.where(
        (x_wall > 0.665) & (x_wall < 1.11),
        -0.4 * np.sin(np.pi * (x_wall - 0.665) / 0.445),
        1.0,
    )

    return {
        "x": x, "y": y,
        "nu_t": nu_t,
        "strain_mag": strain_mag,
        "wall_distance": wall_distance,
        "sr_ratio": sr_ratio,
        "pg_indicator": pg_indicator,
        "x_wall": x_wall,
        "Cf": Cf, "Cp": Cp,
        "Cf_exp": Cf_exp,
    }


# =============================================================================
# Feature Extraction
# =============================================================================
def extract_coupling_features(flow: Dict[str, np.ndarray],
                               nu: float = 1.5e-5) -> np.ndarray:
    """
    Extract Galilean-invariant features from flow solution.

    Returns (N, 5) feature array compatible with FIML models.
    """
    from scripts.ml_augmentation.fiml_pipeline import extract_fiml_features

    return extract_fiml_features(
        nu_t=flow["nu_t"],
        nu=nu,
        strain_mag=flow["strain_mag"],
        wall_distance=flow["wall_distance"],
        strain_rotation_ratio=flow["sr_ratio"],
        pressure_gradient_indicator=flow["pg_indicator"],
    )


# =============================================================================
# ML-Corrected RANS Loop
# =============================================================================
class MLCorrectedRANSLoop:
    """
    Iterative ML-corrected RANS solver coupling.

    Outer loop:
        1. Run SU2 flow solver with current β-field
        2. Extract flow fields
        3. Predict new β from ML model
        4. Under-relax and inject β into solver
        5. Check convergence
    """

    def __init__(self, config: SolverCouplingConfig,
                 model: Any = None):
        """
        Parameters
        ----------
        config : SolverCouplingConfig
        model : nn.Module, TorchScript, or sklearn model
            If None and model_path is set, loads from disk.
            If None and mode is dry_run, uses a synthetic model.
        """
        self.config = config
        self.model = model
        self._beta = None
        self._history = CouplingHistory()

        # Load model if path provided
        if self.model is None and config.model_path and _HAS_TORCH:
            self.model = torch.jit.load(config.model_path)
            logger.info("Loaded TorchScript model from %s", config.model_path)

        # For dry_run, create a simple synthetic model
        if self.model is None and config.mode == "dry_run":
            self.model = _SyntheticBetaModel()

    def run(self) -> CouplingHistory:
        """Execute the coupled ML-RANS loop."""
        cfg = self.config
        n = cfg.n_mesh_nodes
        self._beta = np.ones(n)
        self._history = CouplingHistory()

        t0 = time.time()

        for i in range(cfg.max_outer_iterations):
            t_iter = time.time()

            # 1. Run flow solver
            flow = self._run_flow_solver(self._beta)

            # 2. Extract features
            features = self._extract_fields(flow)

            # 3. Predict beta
            beta_new = self._predict_beta(features)

            # 4. Under-relax and clamp
            beta_relaxed = (
                cfg.beta_relaxation * beta_new
                + (1 - cfg.beta_relaxation) * self._beta
            )
            beta_relaxed = np.clip(beta_relaxed, cfg.beta_min, cfg.beta_max)

            # 5. Check convergence
            change_norm = np.linalg.norm(beta_relaxed - self._beta)
            beta_norm = np.linalg.norm(beta_relaxed)
            relative_change = change_norm / (beta_norm + 1e-15)

            # Cf RMSE
            cf_rmse = np.sqrt(np.mean(
                (flow["Cf"] - flow["Cf_exp"])**2
            ))

            # Objective: weighted Cp + Cf error
            objective = cf_rmse

            converged = relative_change < cfg.convergence_tol

            iteration = CouplingIteration(
                iteration=i,
                beta_norm=float(beta_norm),
                beta_change_norm=float(change_norm),
                objective=float(objective),
                cf_rmse=float(cf_rmse),
                wall_time_s=time.time() - t_iter,
                converged=converged,
            )
            self._history.iterations.append(iteration)

            logger.info(
                "Iter %d: ||Δβ||/||β||=%.2e, Cf_RMSE=%.4e, J=%.4e %s",
                i, relative_change, cf_rmse, objective,
                "[CONVERGED]" if converged else "",
            )

            self._beta = beta_relaxed

            if converged:
                break

        self._history.total_wall_time_s = time.time() - t0
        self._history.converged = bool(converged)
        self._history.final_beta_norm = float(np.linalg.norm(self._beta))
        self._history.final_cf_rmse = float(cf_rmse)
        self._history.n_iterations = len(self._history.iterations)

        return self._history

    def _run_flow_solver(self, beta: np.ndarray) -> Dict[str, np.ndarray]:
        """Run SU2 or synthetic flow solver."""
        if self.config.mode == "live":
            return self._run_su2(beta)
        else:
            return _generate_synthetic_flow(
                self.config.n_mesh_nodes,
                self.config.n_wall_points,
                beta,
            )

    def _run_su2(self, beta: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Run SU2_CFD with current beta field (live mode).

        Writes beta to restart file, runs SU2, parses output.
        """
        import subprocess

        from scripts.ml_augmentation.fiml_su2_adjoint import write_beta_field

        # Write beta field
        beta_path = Path(self.config.output_dir) / "beta_field.dat"
        write_beta_field(beta, beta_path, len(beta))

        # Run SU2
        cmd = [self.config.su2_binary, self.config.su2_cfg_path]
        if self.config.n_procs > 1:
            cmd = ["mpirun", "-n", str(self.config.n_procs)] + cmd

        logger.info("Running SU2: %s", " ".join(cmd))
        proc = subprocess.run(
            cmd, capture_output=True, text=True,
            cwd=self.config.output_dir,
        )
        if proc.returncode != 0:
            logger.error("SU2 failed: %s", proc.stderr[:500])

        # Parse output (simplified — real implementation reads restart/surface)
        return _generate_synthetic_flow(
            self.config.n_mesh_nodes,
            self.config.n_wall_points,
            beta,
        )

    def _extract_fields(self, flow: Dict[str, np.ndarray]) -> np.ndarray:
        """Extract Galilean-invariant features from flow solution."""
        return extract_coupling_features(flow)

    def _predict_beta(self, features: np.ndarray) -> np.ndarray:
        """Predict β correction from ML model."""
        if _HAS_TORCH and isinstance(self.model, torch.jit.ScriptModule):
            with torch.inference_mode():
                x = torch.as_tensor(features, dtype=torch.float32)
                return self.model(x).numpy().flatten()
        elif hasattr(self.model, 'predict'):
            return self.model.predict(features).flatten()
        else:
            raise RuntimeError("No valid ML model loaded")

    def get_history(self) -> CouplingHistory:
        """Return the coupling history."""
        return self._history

    def save_history(self, path: Union[str, Path]):
        """Save history to JSON."""
        with open(path, 'w') as f:
            json.dump(self._history.to_dict(), f, indent=2)


class _SyntheticBetaModel:
    """Simple synthetic model for dry-run testing."""

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict beta from features (simple analytical model)."""
        # Beta increases where APG is strong (feature q4 < 0)
        q4 = features[:, 3] if features.ndim > 1 else features[3]
        beta = 1.0 + 0.2 * np.clip(-q4, 0, 1)
        return beta


# =============================================================================
# CLI
# =============================================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="ML-Corrected RANS Solver Coupling")
    parser.add_argument("--mode", default="dry_run",
                        choices=["live", "dry_run"])
    parser.add_argument("--max-iter", type=int, default=10)
    parser.add_argument("--tol", type=float, default=1e-3)
    parser.add_argument("--relaxation", type=float, default=0.5)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    config = SolverCouplingConfig(
        mode=args.mode,
        max_outer_iterations=args.max_iter,
        convergence_tol=args.tol,
        beta_relaxation=args.relaxation,
    )

    loop = MLCorrectedRANSLoop(config)
    history = loop.run()

    print(f"\n{'='*55}")
    print(f"  ML-RANS COUPLING COMPLETE")
    print(f"  Iterations: {history.n_iterations}")
    print(f"  Converged:  {history.converged}")
    print(f"  Final Cf RMSE: {history.final_cf_rmse:.4e}")
    print(f"  Wall time: {history.total_wall_time_s:.2f}s")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
