"""
y⁺ Estimator
=============
Standalone utility for estimating and computing y⁺-related quantities
for CFD grid design. Supports flat plate correlations, turbulent BL
estimates, and first cell height calculations.

Usage:
    python yplus_estimator.py --Re 5e6 --L 1.0 --U 50.0 --yplus 1.0
"""

import argparse
import numpy as np
from typing import Dict, Tuple


def skin_friction_flat_plate(Re_x: float) -> float:
    """
    Estimate Cf for a turbulent flat plate using Schlichting correlation.

    Cf = 0.059 / Re_x^0.2  (valid for 5×10⁵ < Re_x < 10⁷)
    """
    return 0.059 / Re_x ** 0.2


def skin_friction_prandtl(Re_L: float) -> float:
    """
    Prandtl-Schlichting formula (higher Re range, Re > 10⁶).

    Cf = 0.455 / (log10(Re_L))^2.58
    """
    return 0.455 / (np.log10(Re_L)) ** 2.58


def wall_shear_stress(Cf: float, rho: float, U_inf: float) -> float:
    """Compute wall shear stress τ_w = 0.5 * Cf * ρ * U²."""
    return 0.5 * Cf * rho * U_inf ** 2


def friction_velocity(tau_w: float, rho: float) -> float:
    """Compute friction velocity u_τ = √(τ_w / ρ)."""
    return np.sqrt(tau_w / rho)


def required_first_cell_height(
    Re: float,
    L: float,
    U: float,
    y_plus_target: float = 1.0,
    rho: float = 1.225,
    mu: float = 1.789e-5,
) -> float:
    """
    Compute required first cell height for target y⁺.

    Parameters
    ----------
    Re : float
        Reynolds number based on reference length.
    L : float
        Reference length [m].
    U : float
        Free-stream velocity [m/s].
    y_plus_target : float
        Target y⁺ value.
    rho, mu : float
        Fluid properties.

    Returns
    -------
    float
        Required first cell height Δy₁ [m].
    """
    nu = mu / rho
    Cf = skin_friction_flat_plate(Re)
    tau_w = wall_shear_stress(Cf, rho, U)
    u_tau = friction_velocity(tau_w, rho)
    return y_plus_target * nu / u_tau


def estimate_yplus(
    Re: float,
    L: float,
    U: float,
    y1: float,
    rho: float = 1.225,
    mu: float = 1.789e-5,
) -> float:
    """
    Estimate y⁺ for a given first cell height.

    Parameters
    ----------
    Re : float
        Reynolds number.
    L : float
        Reference length [m].
    U : float
        Free-stream velocity [m/s].
    y1 : float
        First cell height [m].
    rho, mu : float
        Fluid properties.

    Returns
    -------
    float
        Estimated y⁺ value.
    """
    nu = mu / rho
    Cf = skin_friction_flat_plate(Re)
    tau_w = wall_shear_stress(Cf, rho, U)
    u_tau = friction_velocity(tau_w, rho)
    return u_tau * y1 / nu


def boundary_layer_thickness(x: float, Re_x: float) -> Dict[str, float]:
    """
    Estimate boundary layer properties at station x.

    Returns
    -------
    dict with keys: delta_99, delta_star, theta, shape_factor
    """
    # Turbulent BL (1/7 power law)
    delta_99 = 0.37 * x / Re_x ** 0.2
    delta_star = delta_99 / 8.0  # displacement thickness
    theta = delta_99 * 7.0 / 72.0  # momentum thickness
    H = delta_star / theta  # shape factor

    return {
        "delta_99": delta_99,
        "delta_star": delta_star,
        "theta": theta,
        "shape_factor": H,
    }


def geometric_grading(
    y1: float,
    total_height: float,
    n_cells: int,
) -> Tuple[float, float]:
    """
    Compute geometric grading ratio for boundary-layer meshing.

    Parameters
    ----------
    y1 : float
        First cell height.
    total_height : float
        Total BL region height.
    n_cells : int
        Number of cells in BL region.

    Returns
    -------
    tuple
        (expansion_ratio, last_cell_height)
    """
    # Iteratively solve: y1 * (r^n - 1) / (r - 1) = H
    from scipy.optimize import brentq

    def residual(r):
        if abs(r - 1.0) < 1e-12:
            return y1 * n_cells - total_height
        return y1 * (r ** n_cells - 1) / (r - 1) - total_height

    try:
        r = brentq(residual, 1.001, 5.0)
    except ValueError:
        r = 1.2  # Fallback
    last_cell = y1 * r ** (n_cells - 1)
    return r, last_cell


def yplus_table(
    Re_range: np.ndarray,
    L: float = 1.0,
    U: float = 10.0,
    yplus_targets: list = None,
) -> Dict:
    """
    Generate a table of required first cell heights for various Re and y⁺ targets.

    Returns dict of {yplus_target: {Re: y1}}.
    """
    if yplus_targets is None:
        yplus_targets = [0.5, 1.0, 5.0, 30.0, 50.0]

    table = {}
    for yp in yplus_targets:
        table[yp] = {}
        for Re in Re_range:
            actual_U = U * Re / (Re_range[0])  # scale velocity
            y1 = required_first_cell_height(Re, L, actual_U, yp)
            table[yp][Re] = y1
    return table


def print_yplus_report(
    Re: float, L: float, U: float, y1: float = None, yplus_target: float = 1.0,
) -> None:
    """Print a formatted y⁺ analysis report."""
    rho = 1.225
    mu = 1.789e-5
    nu = mu / rho

    Cf = skin_friction_flat_plate(Re)
    tau_w = wall_shear_stress(Cf, rho, U)
    u_tau = friction_velocity(tau_w, rho)

    if y1 is None:
        y1 = required_first_cell_height(Re, L, U, yplus_target)
        yplus = yplus_target
    else:
        yplus = estimate_yplus(Re, L, U, y1)

    # BL thickness at trailing edge
    bl = boundary_layer_thickness(L, Re)

    print(f"\n{'='*55}")
    print(f"  y⁺ Estimator Report")
    print(f"{'='*55}")
    print(f"  Flow Conditions:")
    print(f"    Re          = {Re:.2e}")
    print(f"    L_ref       = {L:.4f} m")
    print(f"    U_ref       = {U:.2f} m/s")
    print(f"    ν           = {nu:.4e} m²/s")
    print(f"")
    print(f"  Wall Properties:")
    print(f"    Cf          = {Cf:.6f}")
    print(f"    τ_w         = {tau_w:.4f} Pa")
    print(f"    u_τ         = {u_tau:.4f} m/s")
    print(f"")
    print(f"  Grid Requirements:")
    print(f"    y⁺          = {yplus:.2f}")
    print(f"    Δy₁         = {y1:.6e} m")
    print(f"    Δy₁/L       = {y1/L:.6e}")
    print(f"")
    print(f"  BL at x = L:")
    print(f"    δ₉₉         = {bl['delta_99']:.6f} m")
    print(f"    δ*           = {bl['delta_star']:.6f} m")
    print(f"    θ            = {bl['theta']:.6f} m")
    print(f"    H            = {bl['shape_factor']:.3f}")
    print(f"{'='*55}")

    # Table for different y⁺ targets
    print(f"\n  Required Δy₁ for different y⁺ targets:")
    print(f"  {'y⁺':>6s}  {'Δy₁ (m)':>12s}  {'Treatment':>20s}")
    print(f"  {'-'*42}")
    for yp in [0.5, 1.0, 5.0, 30.0, 100.0]:
        dy = required_first_cell_height(Re, L, U, yp)
        treatment = "Wall-resolved" if yp <= 5 else "Wall function"
        print(f"  {yp:6.1f}  {dy:12.4e}  {treatment:>20s}")


# =============================================================================
# CLI
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="y⁺ Estimator for CFD Grid Design",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--Re", type=float, required=True, help="Reynolds number")
    parser.add_argument("--L", type=float, default=1.0, help="Reference length [m]")
    parser.add_argument("--U", type=float, required=True, help="Free-stream velocity [m/s]")
    parser.add_argument("--yplus", type=float, default=1.0, help="Target y⁺")
    parser.add_argument("--y1", type=float, default=None, help="First cell height [m] (estimate y⁺)")

    args = parser.parse_args()
    print_yplus_report(args.Re, args.L, args.U, args.y1, args.yplus)
