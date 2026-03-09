#!/usr/bin/env python3
"""
Constrained RANS Recalibration — Physics-Penalty Optimization of SST Coefficients
=====================================================================================
Implements the approach of Bin et al. (2024), "Constrained Recalibration of
Two-Equation RANS Models with Physics Penalties", TAML 14, 100503.

Key idea:  Instead of free-form ML correction, recalibrate the existing k-ω SST
closure constants under physics-derived penalty constraints that guarantee:
  1. Log-layer consistency (von Kármán constant κ ≈ 0.41, B ≈ 5.0)
  2. Realizability (Lumley triangle bounds on Reynolds-stress eigenvalues)
  3. Free-stream turbulence decay rate consistency

The module provides:
  - SSTCoefficientSpace:        Parameterized SST closure constant set
  - PhysicsPenaltyLoss:         Physics-derived penalty functions
  - SyntheticSSTEvaluator:      Evaluate coefficients against benchmark flow data
  - ConstrainedRecalibrator:    scipy-based constrained optimizer
  - RecalibrationReport:        Results comparison and reporting
  - run_recalibration():        End-to-end CLI entry point

Usage:
    recalibrator = ConstrainedRecalibrator(
        penalty_weights={"log_layer": 10.0, "realizability": 5.0, "decay": 2.0},
    )
    result = recalibrator.optimize(target_cases=["wall_hump", "periodic_hill"])
    print(result.report())

References
----------
[31] Bin, Y. et al. (2024). "Constrained Recalibration of Two-Equation RANS
     Models for Improved Adverse-Pressure-Gradient Predictions," Theoretical
     and Applied Mechanics Letters, 14, 100503.
"""

import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy import optimize

logger = logging.getLogger(__name__)
PROJECT = Path(__file__).resolve().parent.parent.parent


# =============================================================================
# SST Coefficient Space
# =============================================================================

@dataclass
class SSTCoefficientSpace:
    """
    Parameterized k-ω SST closure constants (Menter 1994/2003).

    The 9 tunable constants and their default (published) values:
        σ_k1   = 0.85   (inner k diffusion)
        σ_k2   = 1.0    (outer k diffusion)
        σ_ω1   = 0.5    (inner ω diffusion)
        σ_ω2   = 0.856  (outer ω diffusion)
        β_1    = 0.075  (inner ω destruction)
        β_2    = 0.0828 (outer ω destruction)
        β_star = 0.09   (k destruction / Cμ)
        a1     = 0.31   (Bradshaw structural parameter)
        kappa  = 0.41   (von Kármán constant)

    Each constant has physically motivated lower/upper bounds.
    """
    sigma_k1: float = 0.85
    sigma_k2: float = 1.0
    sigma_omega1: float = 0.5
    sigma_omega2: float = 0.856
    beta_1: float = 0.075
    beta_2: float = 0.0828
    beta_star: float = 0.09
    a1: float = 0.31
    kappa: float = 0.41

    # Physical bounds — derived from theoretical constraints and literature
    BOUNDS = {
        "sigma_k1":     (0.4, 1.5),
        "sigma_k2":     (0.5, 2.0),
        "sigma_omega1": (0.2, 1.0),
        "sigma_omega2": (0.4, 1.5),
        "beta_1":       (0.04, 0.15),
        "beta_2":       (0.04, 0.15),
        "beta_star":    (0.06, 0.12),
        "a1":           (0.15, 0.50),
        "kappa":        (0.38, 0.44),
    }

    NAMES = [
        "sigma_k1", "sigma_k2", "sigma_omega1", "sigma_omega2",
        "beta_1", "beta_2", "beta_star", "a1", "kappa",
    ]

    def to_vector(self) -> np.ndarray:
        """Convert to flat parameter vector."""
        return np.array([getattr(self, n) for n in self.NAMES])

    @classmethod
    def from_vector(cls, vec: np.ndarray) -> "SSTCoefficientSpace":
        """Create from flat parameter vector."""
        return cls(**{n: float(v) for n, v in zip(cls.NAMES, vec)})

    def get_bounds(self) -> List[Tuple[float, float]]:
        """Return (lower, upper) bounds list for scipy optimizer."""
        return [self.BOUNDS[n] for n in self.NAMES]

    def perturb(self, scale: float = 0.1, seed: int = 42) -> "SSTCoefficientSpace":
        """Return a randomly perturbed copy (within bounds)."""
        rng = np.random.default_rng(seed)
        vec = self.to_vector()
        bounds = self.get_bounds()
        perturbed = vec * (1.0 + scale * rng.standard_normal(len(vec)))
        clipped = np.array([
            np.clip(p, lo, hi) for p, (lo, hi) in zip(perturbed, bounds)
        ])
        return self.from_vector(clipped)

    def to_dict(self) -> Dict[str, float]:
        """Export as dictionary."""
        return {n: getattr(self, n) for n in self.NAMES}

    def diff(self, other: "SSTCoefficientSpace") -> Dict[str, Dict[str, float]]:
        """Compute differences with another coefficient set."""
        result = {}
        for n in self.NAMES:
            v0, v1 = getattr(self, n), getattr(other, n)
            result[n] = {
                "default": v0,
                "optimized": v1,
                "delta": v1 - v0,
                "delta_pct": 100.0 * (v1 - v0) / (abs(v0) + 1e-15),
            }
        return result

    @classmethod
    def default(cls) -> "SSTCoefficientSpace":
        """Return the standard Menter (1994) SST constants."""
        return cls()


# =============================================================================
# Physics Penalty Loss Functions
# =============================================================================

class PhysicsPenaltyLoss:
    """
    Physics-derived penalty functions for constrained SST recalibration.

    Following Bin et al. (2024), three constraints are enforced:
      1. Log-layer: U+ = (1/κ) ln(y+) + B must hold in the overlap region
      2. Realizability: Reynolds stress anisotropy within Lumley triangle
      3. Free-stream decay: turbulence intensity decay matches theory

    Each penalty returns a non-negative scalar; zero indicates perfect
    satisfaction.
    """

    def __init__(
        self,
        kappa_target: float = 0.41,
        B_target: float = 5.0,
        y_plus_range: Tuple[float, float] = (30.0, 300.0),
        n_log_points: int = 50,
    ):
        self.kappa_target = kappa_target
        self.B_target = B_target
        self.y_plus_range = y_plus_range
        self.n_log_points = n_log_points

    def log_layer_penalty(self, coeffs: SSTCoefficientSpace) -> float:
        """
        Penalty for deviation from the logarithmic law of the wall.

        The SST model must reproduce U+ = (1/κ) ln(y+) + B in the
        log layer.  We check the implied κ and B from the coefficients.

        The von Kármán constant κ is explicit in the SST model.
        β* and a1 indirectly affect the log-layer slope through the
        eddy viscosity formulation.

        Returns non-negative penalty scalar.
        """
        # Direct κ deviation
        kappa_penalty = (coeffs.kappa - self.kappa_target) ** 2

        # β* consistency: in the log layer, β* = Cμ ≈ 0.09 connects to
        # κ via the relation κ² / (σ_ω · √β*) ≈ consistent.
        # Menter's consistency relation: κ² / √β* = σ_ω2 * (β_2 - β_2_implied)
        sqrt_beta_star = np.sqrt(coeffs.beta_star)
        implied_sigma_check = coeffs.kappa ** 2 / (
            sqrt_beta_star * coeffs.sigma_omega2
        )
        # Should be close to (β_2 / β_star - 1) → roughly 0.44 for default
        target_ratio = (coeffs.beta_2 / coeffs.beta_star) - 1.0
        consistency_penalty = (implied_sigma_check - target_ratio) ** 2

        # Generate log-layer profile for y+ range validation
        y_plus = np.linspace(
            self.y_plus_range[0], self.y_plus_range[1], self.n_log_points
        )
        u_plus_log = (1.0 / coeffs.kappa) * np.log(y_plus) + self.B_target
        u_plus_target = (1.0 / self.kappa_target) * np.log(y_plus) + self.B_target
        profile_mse = np.mean((u_plus_log - u_plus_target) ** 2)

        return float(kappa_penalty + 0.5 * consistency_penalty + 0.1 * profile_mse)

    def realizability_penalty(
        self,
        coeffs: SSTCoefficientSpace,
        anisotropy_samples: Optional[np.ndarray] = None,
    ) -> float:
        """
        Realizability penalty — checks the Lumley triangle bounds.

        If anisotropy_samples (N, 3, 3) are provided, checks eigenvalue
        bounds.  Otherwise, checks implied structural constraints from
        the coefficients (a1 = √Cμ / 2 in the Bradshaw hypothesis relates
        to realizability through -b12 ≤ a1).

        Returns non-negative penalty scalar.
        """
        penalty = 0.0

        # Structural constraint: a1 must bound the ratio -uv/k
        # For attached TBLs, -b12_max ≈ 0.15, so a1 must exceed this
        if coeffs.a1 < 0.15:
            penalty += (0.15 - coeffs.a1) ** 2

        # β* = Cμ must satisfy 0 < Cμ ≤ 0.11 for physical turbulence
        if coeffs.beta_star > 0.11:
            penalty += (coeffs.beta_star - 0.11) ** 2

        # If explicit anisotropy samples are given, check Lumley bounds
        if anisotropy_samples is not None:
            for i in range(min(len(anisotropy_samples), 200)):
                b = anisotropy_samples[i]
                b_sym = 0.5 * (b + b.T)
                eigvals = np.linalg.eigvalsh(b_sym)
                violations = np.sum(np.maximum(-1.0/3.0 - eigvals, 0.0) ** 2)
                violations += np.sum(np.maximum(eigvals - 2.0/3.0, 0.0) ** 2)
                penalty += violations

        return float(penalty)

    def free_stream_decay_penalty(self, coeffs: SSTCoefficientSpace) -> float:
        """
        Free-stream turbulence decay rate penalty.

        In the free stream (no production), k decays as k ~ t^(-n) where
        the exponent n = 1 / (β* - 1) for homogeneous isotropic turbulence.
        Experiments give n ≈ 1.25–1.30 (Townsend, Batchelor).

        The SST relationship implies n = 1 / (β_star * σ_ω2 / β_2 - 1)
        in the k-ω branch.

        Returns non-negative penalty scalar.
        """
        # Target decay exponent from experiments
        n_target = 1.27

        # SST-implied decay: in free stream, ω equation gives
        # dω/dt = -β_2 * ω², dk/dt = -β* k ω
        # Combined: k ~ t^(-β*/(β_2 - σ_ω2*β*)), approximately
        ratio = coeffs.beta_star / (coeffs.beta_2 + 1e-15)
        # The free-stream decay exponent for k-ε type:
        # n = 1 / (C_ε2 - 1), where C_ε2 ≈ β_2 / β* + σ_ω2 * κ² / √β*
        c_eps2_approx = (
            coeffs.beta_2 / coeffs.beta_star
            + coeffs.sigma_omega2 * coeffs.kappa ** 2 / np.sqrt(coeffs.beta_star)
        )
        n_implied = 1.0 / max(c_eps2_approx - 1.0, 0.01)

        return float((n_implied - n_target) ** 2)

    def total_penalty(
        self,
        coeffs: SSTCoefficientSpace,
        anisotropy_samples: Optional[np.ndarray] = None,
        weights: Optional[Dict[str, float]] = None,
    ) -> float:
        """
        Weighted sum of all physics penalties.

        Parameters
        ----------
        coeffs : SSTCoefficientSpace
        anisotropy_samples : optional (N, 3, 3)
        weights : dict with keys 'log_layer', 'realizability', 'decay'

        Returns
        -------
        Non-negative total penalty.
        """
        w = weights or {"log_layer": 10.0, "realizability": 5.0, "decay": 2.0}
        p_log = self.log_layer_penalty(coeffs)
        p_real = self.realizability_penalty(coeffs, anisotropy_samples)
        p_decay = self.free_stream_decay_penalty(coeffs)
        return (
            w.get("log_layer", 10.0) * p_log
            + w.get("realizability", 5.0) * p_real
            + w.get("decay", 2.0) * p_decay
        )


# =============================================================================
# Synthetic SST Evaluator
# =============================================================================

class SyntheticSSTEvaluator:
    """
    Evaluate SST coefficients against benchmark flow cases using
    simplified analytical RANS models.

    Follows the project's proxy-model pattern (like the DRL flow control
    environments): fast analytical evaluation rather than live CFD.

    Supported cases:
      - flat_plate: Zero-pressure-gradient turbulent BL
      - wall_hump: APG-driven separation (Greenblatt et al.)
      - periodic_hill: Periodic hill flow (Breuer et al.)
    """

    CASES = {
        "flat_plate": {
            "description": "ZPG flat-plate TBL at Re_x = 5e6",
            "target_Cf": 0.00285,         # Schlichting (1979)
            "target_H": 1.40,              # Shape factor
            "target_Re_theta": 5000.0,
        },
        "wall_hump": {
            "description": "NASA wall-mounted hump, separation at x/c ≈ 0.66",
            "target_Cf_min": -0.002,
            "target_x_sep": 0.66,
            "target_x_reat": 1.11,
            "target_bubble_length": 0.45,
        },
        "periodic_hill": {
            "description": "Periodic hill, Re_H = 10595",
            "target_x_sep": 0.22,
            "target_x_reat": 4.72,
            "target_Cf_min": -0.003,
        },
    }

    def evaluate(
        self,
        coeffs: SSTCoefficientSpace,
        cases: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate SST coefficients against target metrics for each case.

        Returns dict of {case_name: {metric: error, ...}}.
        """
        cases = cases or list(self.CASES.keys())
        results = {}
        for case_name in cases:
            if case_name not in self.CASES:
                logger.warning(f"Unknown case: {case_name}, skipping")
                continue
            if case_name == "flat_plate":
                results[case_name] = self._eval_flat_plate(coeffs)
            elif case_name == "wall_hump":
                results[case_name] = self._eval_wall_hump(coeffs)
            elif case_name == "periodic_hill":
                results[case_name] = self._eval_periodic_hill(coeffs)
        return results

    def total_data_loss(
        self,
        coeffs: SSTCoefficientSpace,
        cases: Optional[List[str]] = None,
    ) -> float:
        """Compute aggregate data-fit loss across all cases."""
        results = self.evaluate(coeffs, cases)
        total = 0.0
        for case_metrics in results.values():
            for v in case_metrics.values():
                total += v ** 2
        return total

    def _eval_flat_plate(self, coeffs: SSTCoefficientSpace) -> Dict[str, float]:
        """
        Simplified flat-plate prediction.

        Uses the log-law Cf formula:
            Cf = 2 / [1/κ * ln(Re_θ) + C(κ, β*)]²
        """
        target = self.CASES["flat_plate"]
        Re_theta = target["target_Re_theta"]

        # SST-based Cf estimate using log-law
        C_const = 2.0 * self.B_from_coeffs(coeffs)
        u_plus_edge = (1.0 / coeffs.kappa) * np.log(Re_theta) + C_const / 6.0
        cf_pred = 2.0 / (u_plus_edge ** 2 + 1e-10)

        # Shape factor — weakly depends on β*
        H_pred = 1.40 + 0.5 * (coeffs.beta_star - 0.09)

        return {
            "Cf_error": cf_pred - target["target_Cf"],
            "H_error": H_pred - target["target_H"],
        }

    def _eval_wall_hump(self, coeffs: SSTCoefficientSpace) -> Dict[str, float]:
        """
        Simplified wall-hump separation prediction.

        Separation location depends on SST's ability to predict APG effects:
        a1 (Bradshaw parameter) strongly controls separation onset.
        """
        target = self.CASES["wall_hump"]

        # a1 controls shear-stress transport limiter:
        # smaller a1 → earlier separation (more conservative)
        # larger a1 → delayed separation (more aggressive)
        x_sep_pred = target["target_x_sep"] + 0.8 * (coeffs.a1 - 0.31)

        # β_1/β_2 ratio affects ω destruction → bubble length
        beta_ratio = coeffs.beta_1 / (coeffs.beta_2 + 1e-15)
        bubble_adj = 0.3 * (beta_ratio - 0.075 / 0.0828)
        bubble_pred = target["target_bubble_length"] + bubble_adj

        # Reattachment
        x_reat_pred = x_sep_pred + bubble_pred

        return {
            "x_sep_error": x_sep_pred - target["target_x_sep"],
            "bubble_length_error": bubble_pred - target["target_bubble_length"],
            "x_reat_error": x_reat_pred - target["target_x_reat"],
        }

    def _eval_periodic_hill(self, coeffs: SSTCoefficientSpace) -> Dict[str, float]:
        """
        Simplified periodic hill separation prediction.

        Similar sensitivities to wall hump; separation & reattachment
        are the key metrics.
        """
        target = self.CASES["periodic_hill"]

        x_sep_pred = target["target_x_sep"] + 0.6 * (coeffs.a1 - 0.31)
        x_reat_pred = target["target_x_reat"] + 1.5 * (
            coeffs.beta_1 / (coeffs.beta_2 + 1e-15) - 0.075 / 0.0828
        )

        return {
            "x_sep_error": x_sep_pred - target["target_x_sep"],
            "x_reat_error": x_reat_pred - target["target_x_reat"],
        }

    @staticmethod
    def B_from_coeffs(coeffs: SSTCoefficientSpace) -> float:
        """Estimate log-law intercept B from SST constants."""
        # B depends weakly on the near-wall model; approximate
        return 5.0 + 2.0 * (coeffs.kappa - 0.41)


# =============================================================================
# Constrained Recalibrator
# =============================================================================

class ConstrainedRecalibrator:
    """
    Physics-constrained optimization of SST closure coefficients.

    Minimizes:
        L_total = L_data(metrics) + λ₁·L_loglaw + λ₂·L_realiz + λ₃·L_decay

    subject to coefficient bounds using scipy.optimize.minimize.

    Parameters
    ----------
    penalty_weights : dict
        Weights for {log_layer, realizability, decay} penalty terms.
    method : str
        Scipy optimizer method — 'trust-constr' or 'SLSQP'.
    max_iter : int
        Maximum optimization iterations.
    """

    def __init__(
        self,
        penalty_weights: Optional[Dict[str, float]] = None,
        method: str = "SLSQP",
        max_iter: int = 200,
        seed: int = 42,
    ):
        self.penalty_weights = penalty_weights or {
            "log_layer": 10.0,
            "realizability": 5.0,
            "decay": 2.0,
        }
        self.method = method
        self.max_iter = max_iter
        self.seed = seed

        self.penalty_fn = PhysicsPenaltyLoss()
        self.evaluator = SyntheticSSTEvaluator()

        # Optimization history
        self.history: List[Dict[str, float]] = []

    def objective(
        self,
        x: np.ndarray,
        target_cases: List[str],
        anisotropy_samples: Optional[np.ndarray] = None,
    ) -> float:
        """
        Combined data + physics penalty objective.

        Parameters
        ----------
        x : parameter vector (9,)
        target_cases : list of case names
        anisotropy_samples : optional (N, 3, 3)

        Returns
        -------
        Total loss scalar.
        """
        coeffs = SSTCoefficientSpace.from_vector(x)

        # Data loss
        data_loss = self.evaluator.total_data_loss(coeffs, target_cases)

        # Physics penalty
        physics_penalty = self.penalty_fn.total_penalty(
            coeffs, anisotropy_samples, self.penalty_weights
        )

        total = data_loss + physics_penalty

        # Log to history
        self.history.append({
            "data_loss": float(data_loss),
            "physics_penalty": float(physics_penalty),
            "total_loss": float(total),
        })

        return total

    def optimize(
        self,
        target_cases: Optional[List[str]] = None,
        initial_coeffs: Optional[SSTCoefficientSpace] = None,
        anisotropy_samples: Optional[np.ndarray] = None,
    ) -> "RecalibrationResult":
        """
        Run constrained optimization.

        Parameters
        ----------
        target_cases : list of case names (default: all available)
        initial_coeffs : starting point (default: Menter 1994)
        anisotropy_samples : optional DNS/LES anisotropy data

        Returns
        -------
        RecalibrationResult with optimized coefficients and diagnostics.
        """
        target_cases = target_cases or ["flat_plate", "wall_hump", "periodic_hill"]
        initial = initial_coeffs or SSTCoefficientSpace.default()
        x0 = initial.to_vector()
        bounds = initial.get_bounds()

        self.history.clear()
        t0 = time.time()

        logger.info(
            f"Starting constrained recalibration: method={self.method}, "
            f"cases={target_cases}, penalty_weights={self.penalty_weights}"
        )

        result = optimize.minimize(
            self.objective,
            x0,
            args=(target_cases, anisotropy_samples),
            method=self.method,
            bounds=bounds,
            options={"maxiter": self.max_iter, "disp": False},
        )

        elapsed = time.time() - t0
        optimized = SSTCoefficientSpace.from_vector(result.x)

        logger.info(
            f"Optimization completed: success={result.success}, "
            f"iterations={result.nit}, elapsed={elapsed:.2f}s"
        )

        # Evaluate both default and optimized
        default_coeffs = SSTCoefficientSpace.default()
        default_metrics = self.evaluator.evaluate(default_coeffs, target_cases)
        optimized_metrics = self.evaluator.evaluate(optimized, target_cases)

        # Physics constraint satisfaction
        default_penalty = self.penalty_fn.total_penalty(
            default_coeffs, anisotropy_samples, self.penalty_weights
        )
        optimized_penalty = self.penalty_fn.total_penalty(
            optimized, anisotropy_samples, self.penalty_weights
        )

        return RecalibrationResult(
            default_coeffs=default_coeffs,
            optimized_coeffs=optimized,
            default_metrics=default_metrics,
            optimized_metrics=optimized_metrics,
            default_penalty=default_penalty,
            optimized_penalty=optimized_penalty,
            optimizer_result=result,
            elapsed_seconds=elapsed,
            history=list(self.history),
            target_cases=target_cases,
        )


# =============================================================================
# Recalibration Result & Report
# =============================================================================

@dataclass
class RecalibrationResult:
    """Container for constrained recalibration results."""
    default_coeffs: SSTCoefficientSpace
    optimized_coeffs: SSTCoefficientSpace
    default_metrics: Dict[str, Dict[str, float]]
    optimized_metrics: Dict[str, Dict[str, float]]
    default_penalty: float
    optimized_penalty: float
    optimizer_result: Any
    elapsed_seconds: float
    history: List[Dict[str, float]]
    target_cases: List[str]

    def coefficient_changes(self) -> Dict[str, Dict[str, float]]:
        """Return coefficient change summary."""
        return self.default_coeffs.diff(self.optimized_coeffs)

    def improvement_summary(self) -> Dict[str, float]:
        """Compute aggregate improvement metrics."""
        # Default total error
        default_total = sum(
            sum(v ** 2 for v in case.values())
            for case in self.default_metrics.values()
        )
        # Optimized total error
        optimized_total = sum(
            sum(v ** 2 for v in case.values())
            for case in self.optimized_metrics.values()
        )
        return {
            "default_data_error": default_total,
            "optimized_data_error": optimized_total,
            "data_error_reduction_pct": (
                100.0 * (1.0 - optimized_total / (default_total + 1e-15))
            ),
            "default_physics_penalty": self.default_penalty,
            "optimized_physics_penalty": self.optimized_penalty,
            "optimizer_success": bool(self.optimizer_result.success),
            "optimizer_iterations": int(self.optimizer_result.nit),
            "elapsed_seconds": self.elapsed_seconds,
        }

    def report(self) -> str:
        """Generate human-readable recalibration report."""
        lines = [
            "=" * 72,
            "  CONSTRAINED RANS RECALIBRATION REPORT (Bin et al. 2024)",
            "=" * 72,
            "",
        ]

        # Coefficient changes
        lines.append("COEFFICIENT CHANGES:")
        lines.append("-" * 60)
        changes = self.coefficient_changes()
        lines.append(f"  {'Parameter':<14} {'Default':>10} {'Optimized':>10} {'Δ%':>8}")
        lines.append(f"  {'-'*14} {'-'*10} {'-'*10} {'-'*8}")
        for name, vals in changes.items():
            lines.append(
                f"  {name:<14} {vals['default']:>10.5f} "
                f"{vals['optimized']:>10.5f} {vals['delta_pct']:>+7.2f}%"
            )

        # Per-case metrics
        lines.append("")
        lines.append("PER-CASE METRIC ERRORS:")
        lines.append("-" * 60)
        for case in self.target_cases:
            lines.append(f"\n  {case}:")
            default_m = self.default_metrics.get(case, {})
            optimized_m = self.optimized_metrics.get(case, {})
            for metric in default_m:
                d_val = default_m[metric]
                o_val = optimized_m.get(metric, float("nan"))
                marker = "✓" if abs(o_val) < abs(d_val) else "✗"
                lines.append(
                    f"    {metric:<25} default={d_val:>+.6f}  "
                    f"optimized={o_val:>+.6f}  {marker}"
                )

        # Summary
        lines.append("")
        lines.append("SUMMARY:")
        lines.append("-" * 60)
        summary = self.improvement_summary()
        for k, v in summary.items():
            if isinstance(v, float):
                lines.append(f"  {k}: {v:.4f}")
            else:
                lines.append(f"  {k}: {v}")

        lines.append("")
        lines.append("=" * 72)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Serializable result dictionary."""
        return {
            "default_coefficients": self.default_coeffs.to_dict(),
            "optimized_coefficients": self.optimized_coeffs.to_dict(),
            "coefficient_changes": self.coefficient_changes(),
            "default_metrics": self.default_metrics,
            "optimized_metrics": self.optimized_metrics,
            "improvement_summary": self.improvement_summary(),
            "target_cases": self.target_cases,
        }


# =============================================================================
# End-to-End Recalibration Runner
# =============================================================================

def run_recalibration(
    target_cases: Optional[List[str]] = None,
    penalty_weights: Optional[Dict[str, float]] = None,
    method: str = "SLSQP",
    max_iter: int = 200,
    output_dir: Optional[str] = None,
) -> RecalibrationResult:
    """
    Run end-to-end constrained RANS recalibration.

    Parameters
    ----------
    target_cases : list of case names
    penalty_weights : physics penalty weights
    method : scipy optimizer method
    max_iter : max iterations
    output_dir : optional directory to save results

    Returns
    -------
    RecalibrationResult
    """
    recalibrator = ConstrainedRecalibrator(
        penalty_weights=penalty_weights,
        method=method,
        max_iter=max_iter,
    )
    result = recalibrator.optimize(target_cases=target_cases)

    # Print report
    print(result.report())

    # Save if output directory specified
    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # Save JSON results
        with open(out_path / "recalibration_results.json", "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, default=str)

        # Save report
        with open(out_path / "recalibration_report.txt", "w", encoding="utf-8") as f:
            f.write(result.report())

        # Save optimization history
        with open(out_path / "optimization_history.json", "w", encoding="utf-8") as f:
            json.dump(result.history, f, indent=2)

        logger.info(f"Results saved to {out_path}")

    return result


# =============================================================================
# CLI
# =============================================================================

def main():
    """CLI entry point for constrained RANS recalibration."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Constrained RANS Recalibration (Bin et al. 2024)"
    )
    parser.add_argument(
        "--cases", nargs="+",
        default=["flat_plate", "wall_hump", "periodic_hill"],
        help="Target benchmark cases",
    )
    parser.add_argument(
        "--method", default="SLSQP",
        choices=["SLSQP", "trust-constr"],
        help="Scipy optimizer method",
    )
    parser.add_argument(
        "--max-iter", type=int, default=200,
        help="Maximum optimization iterations",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory for results",
    )
    parser.add_argument(
        "--log-weight", type=float, default=10.0,
        help="Log-layer penalty weight",
    )
    parser.add_argument(
        "--real-weight", type=float, default=5.0,
        help="Realizability penalty weight",
    )
    parser.add_argument(
        "--decay-weight", type=float, default=2.0,
        help="Free-stream decay penalty weight",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    penalty_weights = {
        "log_layer": args.log_weight,
        "realizability": args.real_weight,
        "decay": args.decay_weight,
    }

    run_recalibration(
        target_cases=args.cases,
        penalty_weights=penalty_weights,
        method=args.method,
        max_iter=args.max_iter,
        output_dir=args.output_dir or str(PROJECT / "results" / "constrained_recalibration"),
    )


if __name__ == "__main__":
    main()
