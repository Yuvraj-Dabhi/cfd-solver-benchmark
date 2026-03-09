"""
Flat Plate Verification
=======================
Code verification against analytical solutions for the
turbulent flat plate boundary layer.

References:
- Blasius solution (laminar)
- Prandtl-Schlichting Cf correlation (turbulent)
- Law of the wall (viscous sublayer + log layer)
- Boundary layer integral quantities (δ₉₉, δ*, θ, H)

This is MRR Level 0 verification (see vv_framework.py).
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class VerificationResult:
    """Result from flat plate verification."""
    quantity: str
    analytical: float
    computed: float
    error_percent: float
    tolerance: float
    passed: bool
    details: str = ""


def blasius_cf_laminar(Re_x: float) -> float:
    """Laminar flat plate Cf (Blasius solution): Cf = 0.664 / √Re_x."""
    return 0.664 / np.sqrt(Re_x)


def turbulent_cf(Re_x: float) -> float:
    """Turbulent flat plate Cf: Cf = 0.059 / Re_x^0.2."""
    return 0.059 / Re_x ** 0.2


def prandtl_schlichting_cf(Re_L: float) -> float:
    """Prandtl-Schlichting Cf correlation (Re > 10⁶): Cf = 0.455/(log₁₀Re)^2.58."""
    return 0.455 / (np.log10(Re_L)) ** 2.58


def turbulent_delta99(x: float, Re_x: float) -> float:
    """Turbulent BL thickness: δ₉₉ = 0.37·x / Re_x^0.2."""
    return 0.37 * x / Re_x ** 0.2


def displacement_thickness(x: float, Re_x: float) -> float:
    """Displacement thickness estimate: δ* ≈ δ₉₉/8."""
    return turbulent_delta99(x, Re_x) / 8.0


def momentum_thickness(x: float, Re_x: float) -> float:
    """Momentum thickness estimate: θ ≈ (7/72)·δ₉₉."""
    return 7.0 / 72.0 * turbulent_delta99(x, Re_x)


def law_of_wall(yplus: np.ndarray, kappa: float = 0.41, B: float = 5.0) -> np.ndarray:
    """
    Compute U⁺ from the composite law of the wall.

    - Viscous sublayer (y⁺ < 5): U⁺ = y⁺
    - Buffer layer (5 < y⁺ < 30): blended
    - Log layer (y⁺ > 30): U⁺ = (1/κ)·ln(y⁺) + B
    """
    uplus = np.zeros_like(yplus)

    # Viscous sublayer
    visc = yplus < 5
    uplus[visc] = yplus[visc]

    # Log layer
    log_layer = yplus >= 30
    uplus[log_layer] = (1 / kappa) * np.log(yplus[log_layer]) + B

    # Buffer (blended)
    buffer = (~visc) & (~log_layer)
    if np.any(buffer):
        # Spalding's implicit formula approximation
        yp = yplus[buffer]
        uplus[buffer] = (1 / kappa) * np.log(yp) + B

    return uplus


def spalding_law(yplus: np.ndarray, kappa: float = 0.41, B: float = 5.0) -> np.ndarray:
    """
    Spalding's single-equation law (valid for all y⁺).

    y⁺ = U⁺ + e^{-κB} · [e^{κU⁺} - 1 - κU⁺ - (κU⁺)²/2 - (κU⁺)³/6]
    Solved iteratively for U⁺.
    """
    from scipy.optimize import brentq

    uplus = np.zeros_like(yplus)

    def spalding_residual(up, yp):
        ku = kappa * up
        return up + np.exp(-kappa * B) * (np.exp(ku) - 1 - ku - ku**2/2 - ku**3/6) - yp

    for i, yp in enumerate(yplus):
        if yp < 0.1:
            uplus[i] = yp
        else:
            try:
                uplus[i] = brentq(spalding_residual, 0, yp, args=(yp,))
            except (ValueError, RuntimeError):
                uplus[i] = (1 / kappa) * np.log(max(yp, 1)) + B

    return uplus


def verify_cf(
    x_cfd: np.ndarray,
    cf_cfd: np.ndarray,
    U_inf: float,
    nu: float,
    tolerance: float = 0.02,
) -> List[VerificationResult]:
    """
    Verify computed Cf against flat plate correlation.

    Parameters
    ----------
    x_cfd : ndarray
        Streamwise stations.
    cf_cfd : ndarray
        Computed Cf values.
    U_inf : float
        Free-stream velocity [m/s].
    nu : float
        Kinematic viscosity [m²/s].
    tolerance : float
        Acceptable relative error (default 2%).

    Returns
    -------
    List of VerificationResult for each station.
    """
    results = []
    for x, cf in zip(x_cfd, cf_cfd):
        Re_x = U_inf * x / nu
        if Re_x < 5e5:
            cf_analytical = blasius_cf_laminar(Re_x)
            flow_type = "laminar (Blasius)"
        else:
            cf_analytical = turbulent_cf(Re_x)
            flow_type = "turbulent"

        error = abs(cf - cf_analytical) / (cf_analytical + 1e-15) * 100
        results.append(VerificationResult(
            quantity=f"Cf at x={x:.3f} ({flow_type})",
            analytical=cf_analytical,
            computed=cf,
            error_percent=error,
            tolerance=tolerance * 100,
            passed=error < tolerance * 100,
            details=f"Re_x={Re_x:.2e}",
        ))
    return results


def verify_law_of_wall(
    yplus_cfd: np.ndarray,
    uplus_cfd: np.ndarray,
    kappa: float = 0.41,
    B: float = 5.0,
    tolerance: float = 0.05,
) -> List[VerificationResult]:
    """
    Verify U⁺ profile against analytical law of the wall.

    Only checks in the viscous sublayer (y⁺ < 5) and log layer (y⁺ > 30).
    """
    results = []

    # Viscous sublayer check
    visc = yplus_cfd < 5
    if np.any(visc):
        uplus_analytical = yplus_cfd[visc]  # U⁺ = y⁺
        error = np.mean(np.abs(uplus_cfd[visc] - uplus_analytical) / (uplus_analytical + 1e-10)) * 100
        results.append(VerificationResult(
            quantity="Viscous sublayer (y⁺ < 5)",
            analytical=np.mean(uplus_analytical),
            computed=np.mean(uplus_cfd[visc]),
            error_percent=error,
            tolerance=tolerance * 100,
            passed=error < tolerance * 100,
        ))

    # Log layer check
    log = yplus_cfd > 30
    if np.any(log):
        uplus_analytical = (1 / kappa) * np.log(yplus_cfd[log]) + B
        error = np.mean(np.abs(uplus_cfd[log] - uplus_analytical) / (uplus_analytical + 1e-10)) * 100
        results.append(VerificationResult(
            quantity=f"Log layer (y⁺ > 30), κ={kappa}, B={B}",
            analytical=np.mean(uplus_analytical),
            computed=np.mean(uplus_cfd[log]),
            error_percent=error,
            tolerance=tolerance * 100,
            passed=error < tolerance * 100,
        ))

    return results


def verify_boundary_layer(
    x_station: float,
    delta99_cfd: float,
    delta_star_cfd: float,
    theta_cfd: float,
    U_inf: float,
    nu: float,
    tolerance: float = 0.10,
) -> List[VerificationResult]:
    """
    Verify BL integral quantities against correlations.
    """
    Re_x = U_inf * x_station / nu
    results = []

    # δ₉₉
    delta99_analytical = turbulent_delta99(x_station, Re_x)
    error = abs(delta99_cfd - delta99_analytical) / delta99_analytical * 100
    results.append(VerificationResult(
        quantity="δ₉₉", analytical=delta99_analytical,
        computed=delta99_cfd, error_percent=error,
        tolerance=tolerance * 100,
        passed=error < tolerance * 100,
    ))

    # δ*
    dstar_analytical = displacement_thickness(x_station, Re_x)
    error = abs(delta_star_cfd - dstar_analytical) / dstar_analytical * 100
    results.append(VerificationResult(
        quantity="δ*", analytical=dstar_analytical,
        computed=delta_star_cfd, error_percent=error,
        tolerance=tolerance * 100,
        passed=error < tolerance * 100,
    ))

    # θ
    theta_analytical = momentum_thickness(x_station, Re_x)
    error = abs(theta_cfd - theta_analytical) / theta_analytical * 100
    results.append(VerificationResult(
        quantity="θ", analytical=theta_analytical,
        computed=theta_cfd, error_percent=error,
        tolerance=tolerance * 100,
        passed=error < tolerance * 100,
    ))

    # Shape factor
    H_cfd = delta_star_cfd / (theta_cfd + 1e-15)
    H_analytical = dstar_analytical / theta_analytical
    error = abs(H_cfd - H_analytical) / H_analytical * 100
    results.append(VerificationResult(
        quantity="H (shape factor)", analytical=H_analytical,
        computed=H_cfd, error_percent=error,
        tolerance=tolerance * 100,
        passed=error < tolerance * 100,
    ))

    return results


def run_full_verification(
    x_cfd: np.ndarray = None,
    cf_cfd: np.ndarray = None,
    yplus_cfd: np.ndarray = None,
    uplus_cfd: np.ndarray = None,
    U_inf: float = 10.0,
    nu: float = 1.5e-5,
) -> Dict[str, List[VerificationResult]]:
    """
    Run all flat plate verification checks.

    Returns dict of {category: [VerificationResult]}.
    """
    results = {}

    if x_cfd is not None and cf_cfd is not None:
        results["skin_friction"] = verify_cf(x_cfd, cf_cfd, U_inf, nu)

    if yplus_cfd is not None and uplus_cfd is not None:
        results["law_of_wall"] = verify_law_of_wall(yplus_cfd, uplus_cfd)

    return results


def print_verification_report(results: Dict[str, List[VerificationResult]]) -> None:
    """Print formatted verification report."""
    print(f"\n{'='*65}")
    print(f"  Flat Plate Verification Report")
    print(f"{'='*65}")

    all_passed = True
    for category, checks in results.items():
        print(f"\n  [{category}]")
        for v in checks:
            status = "✓" if v.passed else "✗"
            all_passed = all_passed and v.passed
            print(f"    {status} {v.quantity}")
            print(f"      Analytical: {v.analytical:.6f}")
            print(f"      Computed:   {v.computed:.6f}")
            print(f"      Error:      {v.error_percent:.2f}% (tol: {v.tolerance:.1f}%)")

    print(f"\n  Overall: {'ALL PASSED ✓' if all_passed else 'SOME FAILED ✗'}")
    print(f"{'='*65}")
