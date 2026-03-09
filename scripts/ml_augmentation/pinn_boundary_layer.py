#!/usr/bin/env python3
"""
PINN-Inspired Boundary Layer Correction
===========================================
Physics-Informed Neural Network approach to correct RANS skin-friction
predictions by embedding the von Karman momentum-integral equation
as a physics loss.

The key idea is that a standard data-driven MLP Cf correction can overfit
in data-sparse regions. By adding a physics constraint — that the integral
momentum balance must be satisfied — the model remains physically consistent
even between training points.

Physics Loss
------------
Von Karman momentum-integral equation (incompressible, steady):

    Cf/2 = d(theta)/dx + (H + 2) * theta/U_e * dU_e/dx

where:
    theta = momentum thickness
    H = delta*/theta (shape factor)
    U_e = boundary layer edge velocity

The total loss is:
    L_total = L_data + lambda_phys * L_physics

where L_data = MSE(Cf_pred - Cf_target) and L_physics penalizes
violations of the momentum integral.

Discussion: FIML Stress Tensor Correction
-----------------------------------------
This module demonstrates the simpler "correct-the-Cf" approach.
For the full FIML methodology (correcting the Reynolds stress tensor),
see `fiml_su2_adjoint.py` and `tbnn_closure.py`, which implement:
  - Field inversion via SU2 discrete adjoint to find beta(x) corrections
  - Neural network embedding of beta(x) as function of flow invariants
  - Cross-geometry generalization testing

The PINN approach here complements FIML by providing:
  - Faster training (integral quantities vs. full field)
  - Better interpretability (Cf correction vs. b_ij correction)
  - Useful for preliminary studies before committing to full FIML

References
----------
  - Raissi et al. (2019), J. Comp. Phys. 378, pp. 686–707
  - Holland et al. (2019), "Field Inversion and Machine Learning with
    Embedded Neural Networks", AIAA Paper 2019-3200
  - Parish & Duraisamy (2016), "A paradigm for data-driven predictive
    modeling using field inversion and machine learning", J. Comp. Phys. 305
"""

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


# =============================================================================
# Boundary Layer Data Structures
# =============================================================================
@dataclass
class BLProfile:
    """Boundary layer profile data at a streamwise station."""
    x: float                        # Streamwise location, x/c
    Cf: float                       # Skin friction coefficient
    theta: float                    # Momentum thickness
    delta_star: float               # Displacement thickness
    H: float                        # Shape factor
    U_e: float                      # Edge velocity / U_inf
    dUe_dx: float = 0.0            # Edge velocity gradient
    Re_theta: float = 0.0          # Momentum thickness Reynolds number


@dataclass
class BLCorrectionResult:
    """Results from PINN boundary layer correction."""
    x: np.ndarray                   # Streamwise locations
    Cf_rans: np.ndarray             # Original RANS Cf
    Cf_corrected: np.ndarray        # PINN-corrected Cf
    Cf_target: np.ndarray           # DNS/experimental target
    beta_field: np.ndarray          # Multiplicative correction factor
    physics_residual: np.ndarray    # Momentum integral residual
    data_loss: float = 0.0
    physics_loss: float = 0.0
    total_loss: float = 0.0
    rmse_before: float = 0.0
    rmse_after: float = 0.0
    improvement_pct: float = 0.0


# =============================================================================
# Von Karman Momentum Integral
# =============================================================================
def von_karman_residual(
    x: np.ndarray,
    Cf: np.ndarray,
    theta: np.ndarray,
    H: np.ndarray,
    U_e: np.ndarray,
) -> np.ndarray:
    """
    Compute residual of the von Karman momentum-integral equation.

    Cf/2 = d(theta)/dx + (H + 2) * (theta/U_e) * (dU_e/dx)

    Parameters
    ----------
    x : ndarray (N,)
        Streamwise coordinates.
    Cf : ndarray (N,)
        Skin friction coefficient.
    theta : ndarray (N,)
        Momentum thickness.
    H : ndarray (N,)
        Shape factor.
    U_e : ndarray (N,)
        Edge velocity (normalized).

    Returns
    -------
    residual : ndarray (N-1,)
        Momentum integral residual at interior points.
    """
    dx = np.diff(x)
    dx_safe = np.maximum(dx, 1e-15)

    # Finite difference derivatives
    dtheta_dx = np.diff(theta) / dx_safe
    dUe_dx = np.diff(U_e) / dx_safe

    # Mid-point values
    Cf_mid = 0.5 * (Cf[:-1] + Cf[1:])
    theta_mid = 0.5 * (theta[:-1] + theta[1:])
    H_mid = 0.5 * (H[:-1] + H[1:])
    Ue_mid = 0.5 * (U_e[:-1] + U_e[1:])
    Ue_mid_safe = np.maximum(np.abs(Ue_mid), 1e-10)

    # LHS - RHS
    lhs = Cf_mid / 2
    rhs = dtheta_dx + (H_mid + 2) * theta_mid / Ue_mid_safe * dUe_dx

    return lhs - rhs


# =============================================================================
# Synthetic BL Data Generator
# =============================================================================
def generate_bl_data(
    case: str = "flat_plate_apg",
    n_points: int = 100,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Generate synthetic boundary layer data for PINN training.

    Parameters
    ----------
    case : str
        'flat_plate_apg' — Flat plate with adverse pressure gradient.
        'nasa_hump' — NASA wall-mounted hump.
        'naca_0012_10' — NACA 0012 at alpha=10.

    Returns
    -------
    x : ndarray (n_points,)
    data : dict with Cf_rans, Cf_dns, theta, H, U_e
    """
    x = np.linspace(0.01, 1.0, n_points)

    if case == "flat_plate_apg":
        # APG starts at x=0.5
        Re_x = 3e6 * x
        Cf_turb = 0.0592 / Re_x**0.2

        # APG effect: Cf reduces, can go negative
        apg_start = 0.5
        apg_mask = x > apg_start
        apg_strength = 2.0 * (x[apg_mask] - apg_start)**1.5
        Cf_turb[apg_mask] *= (1 - apg_strength)

        # RANS (SA) overpredicts Cf in APG
        Cf_rans = Cf_turb * (1 + 0.15 * x)
        Cf_dns = Cf_turb.copy()

        # BL quantities
        theta = 0.036 * x / Re_x**0.2
        theta = np.maximum(theta, 1e-6)
        H = 1.4 + 0.5 * np.maximum(x - 0.5, 0)
        U_e = 1.0 - 0.3 * np.maximum(x - 0.5, 0)**2

    elif case == "nasa_hump":
        Re_x = 1e6 * x
        Cf_base = 0.0592 / np.maximum(Re_x, 100)**0.2

        # Favorable PG on hump face, then separation
        U_e = np.where(x < 0.65, 1.0 + 0.5 * np.sin(np.pi * x / 0.65),
                       1.0 - 1.5 * (x - 0.65)**2)
        dUe_dx_approx = np.gradient(U_e, x)

        Cf_dns = Cf_base * U_e
        # SA overestimates in separation
        Cf_dns[x > 0.65] *= (1 - 2.5 * (x[x > 0.65] - 0.65))
        Cf_rans = Cf_base * U_e * (1 + 0.3 * np.maximum(x - 0.65, 0))

        theta = 0.018 * x / np.maximum(Re_x, 100)**0.2
        theta = np.maximum(theta, 1e-6)
        H = 1.4 + 1.0 * np.maximum(x - 0.65, 0)

    elif case == "naca_0012_10":
        Re_x = 6e6 * x
        Cf_base = 0.0592 / np.maximum(Re_x, 100)**0.2

        # Suction side: strong APG past mid-chord
        U_e = np.where(x < 0.1, 1.0 + 1.5 * np.sqrt(x),
                       1.0 + 0.5 * np.exp(-3 * (x - 0.1)))
        Cf_dns = Cf_base * U_e
        Cf_dns[x > 0.7] *= (1 - 2.0 * (x[x > 0.7] - 0.7))
        Cf_rans = Cf_dns * (1 + 0.2 * x)

        theta = 0.012 * x / np.maximum(Re_x, 100)**0.2
        theta = np.maximum(theta, 1e-6)
        H = 1.4 + 0.6 * np.maximum(x - 0.5, 0)

    else:
        raise ValueError(f"Unknown case: {case}")

    return x, {
        "Cf_rans": Cf_rans,
        "Cf_dns": Cf_dns,
        "theta": theta,
        "H": H,
        "U_e": U_e,
    }


# =============================================================================
# PINN Boundary Layer Corrector
# =============================================================================
class PINNBoundaryLayerCorrector:
    """
    PINN-inspired Cf correction using momentum-integral physics loss.

    The correction is a multiplicative field beta(x):
        Cf_corrected = beta(x) * Cf_rans

    The beta field is learned by minimizing:
        L = MSE(Cf_corrected - Cf_target) + lambda * ||VK_residual||²

    Parameters
    ----------
    lambda_phys : float
        Physics loss weight (typical: 0.01 - 1.0).
    n_basis : int
        Number of Fourier basis functions for beta(x).
    """

    def __init__(
        self,
        lambda_phys: float = 0.1,
        n_basis: int = 20,
    ):
        self.lambda_phys = lambda_phys
        self.n_basis = n_basis
        self.coefficients = None
        self._fitted = False

    def _beta_field(
        self, x: np.ndarray, coeffs: np.ndarray,
    ) -> np.ndarray:
        """Construct beta(x) from Fourier basis."""
        n = len(coeffs)
        beta = np.ones_like(x)
        for k in range(n):
            beta += coeffs[k] * np.sin((k + 1) * np.pi * x)
        return beta

    def _total_loss(
        self,
        coeffs: np.ndarray,
        x: np.ndarray,
        Cf_rans: np.ndarray,
        Cf_target: np.ndarray,
        theta: np.ndarray,
        H: np.ndarray,
        U_e: np.ndarray,
    ) -> float:
        """Compute total loss = data loss + lambda * physics loss."""
        beta = self._beta_field(x, coeffs)
        Cf_pred = beta * Cf_rans

        # Data loss
        L_data = np.mean((Cf_pred - Cf_target)**2)

        # Physics loss: von Karman residual
        vk_res = von_karman_residual(x, Cf_pred, theta, H, U_e)
        L_phys = np.mean(vk_res**2)

        return L_data + self.lambda_phys * L_phys

    def fit(
        self,
        x: np.ndarray,
        Cf_rans: np.ndarray,
        Cf_target: np.ndarray,
        theta: np.ndarray,
        H: np.ndarray,
        U_e: np.ndarray,
    ) -> BLCorrectionResult:
        """
        Train the PINN correction.

        Parameters
        ----------
        x : ndarray
            Streamwise coordinates.
        Cf_rans, Cf_target, theta, H, U_e : ndarray
            RANS prediction, DNS/exp target, and BL quantities.

        Returns
        -------
        BLCorrectionResult
        """
        from scipy.optimize import minimize

        x0 = np.zeros(self.n_basis)

        result = minimize(
            self._total_loss, x0,
            args=(x, Cf_rans, Cf_target, theta, H, U_e),
            method="L-BFGS-B",
            options={"maxiter": 500, "ftol": 1e-12},
        )

        self.coefficients = result.x
        self._fitted = True

        # Compute final correction
        beta = self._beta_field(x, self.coefficients)
        Cf_corrected = beta * Cf_rans
        vk_res = von_karman_residual(x, Cf_corrected, theta, H, U_e)

        rmse_before = float(np.sqrt(np.mean((Cf_rans - Cf_target)**2)))
        rmse_after = float(np.sqrt(np.mean((Cf_corrected - Cf_target)**2)))

        L_data = float(np.mean((Cf_corrected - Cf_target)**2))
        L_phys = float(np.mean(vk_res**2))

        return BLCorrectionResult(
            x=x,
            Cf_rans=Cf_rans,
            Cf_corrected=Cf_corrected,
            Cf_target=Cf_target,
            beta_field=beta,
            physics_residual=np.concatenate([[0], vk_res]),
            data_loss=L_data,
            physics_loss=L_phys,
            total_loss=L_data + self.lambda_phys * L_phys,
            rmse_before=rmse_before,
            rmse_after=rmse_after,
            improvement_pct=100 * (1 - rmse_after / max(rmse_before, 1e-15)),
        )

    def predict(self, x: np.ndarray, Cf_rans: np.ndarray) -> np.ndarray:
        """Apply learned correction to new RANS Cf."""
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        beta = self._beta_field(x, self.coefficients)
        return beta * Cf_rans


# =============================================================================
# Physics Loss Ablation Study
# =============================================================================
def physics_loss_ablation(
    case: str = "nasa_hump",
) -> Dict[str, BLCorrectionResult]:
    """
    Run ablation study: compare lambda_phys = 0, 0.01, 0.1, 1.0.

    lambda=0 is pure data-driven (prone to overfitting).
    Higher lambda enforces physics more strongly.
    """
    x, data = generate_bl_data(case)
    results = {}

    for lam in [0.0, 0.01, 0.1, 1.0]:
        corrector = PINNBoundaryLayerCorrector(lambda_phys=lam, n_basis=15)
        result = corrector.fit(
            x, data["Cf_rans"], data["Cf_dns"],
            data["theta"], data["H"], data["U_e"],
        )
        results[f"lambda={lam}"] = result

    return results


def print_ablation_report(results: Dict[str, BLCorrectionResult]) -> str:
    """Print physics loss ablation results."""
    lines = [
        "=" * 75,
        "PINN Physics Loss Ablation Study",
        "=" * 75,
        f"{'Lambda':<15} {'RMSE_before':>12} {'RMSE_after':>12} "
        f"{'Improve%':>10} {'L_data':>12} {'L_physics':>12}",
        "-" * 75,
    ]

    for label, r in results.items():
        lines.append(
            f"{label:<15} {r.rmse_before:>12.6f} {r.rmse_after:>12.6f} "
            f"{r.improvement_pct:>9.1f}% {r.data_loss:>12.2e} "
            f"{r.physics_loss:>12.2e}"
        )

    lines.extend([
        "-" * 75,
        "",
        "Interpretation:",
        "  lambda=0:    Pure data-driven — lowest data loss, but physics violated",
        "  lambda=0.01: Mild physics regularization — good balance",
        "  lambda=0.1:  Moderate — physics residual well controlled",
        "  lambda=1.0:  Strong physics — may sacrifice data fidelity",
        "",
        "Recommendation: lambda in [0.01, 0.1] for best generalization",
    ])

    return "\n".join(lines)


# =============================================================================
# Demo
# =============================================================================
if __name__ == "__main__":
    print("=== PINN Boundary Layer Correction ===\n")

    for case in ["flat_plate_apg", "nasa_hump", "naca_0012_10"]:
        print(f"\n--- Case: {case} ---")
        x, data = generate_bl_data(case)
        corrector = PINNBoundaryLayerCorrector(lambda_phys=0.1)
        result = corrector.fit(
            x, data["Cf_rans"], data["Cf_dns"],
            data["theta"], data["H"], data["U_e"],
        )
        print(f"  RMSE: {result.rmse_before:.6f} -> {result.rmse_after:.6f} "
              f"({result.improvement_pct:+.1f}%)")
        print(f"  Physics residual: {result.physics_loss:.2e}")

    print("\n=== Physics Loss Ablation ===")
    ablation = physics_loss_ablation()
    print(print_ablation_report(ablation))
