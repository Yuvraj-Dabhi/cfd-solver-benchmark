"""
Grid Convergence Study
======================
Implements Richardson extrapolation, Grid Convergence Index (GCI),
observed order of accuracy, and asymptotic range checking.

Based on:
  - Roache, P.J. (1994) "Perspective: A Method for Uniform Reporting
    of Grid Refinement Studies"
  - AIAA G-077-1998 "Guide for the Verification and Validation of
    Computational Fluid Dynamics Simulations"
  - Celik et al. (2008) "Procedure for Estimation and Reporting of
    Uncertainty Due to Discretization in CFD Applications"
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class GCIResult:
    """Result of a Grid Convergence Index analysis."""
    # Grid info
    n_grids: int = 0
    grid_sizes: List[int] = field(default_factory=list)
    refinement_ratios: List[float] = field(default_factory=list)

    # Solution values
    phi_values: List[float] = field(default_factory=list)
    phi_extrapolated: float = 0.0  # Richardson extrapolation

    # GCI
    observed_order: float = 0.0
    gci_fine: float = 0.0       # GCI on finest grid (%)
    gci_coarse: float = 0.0     # GCI on coarser grid (%)

    # Validation
    asymptotic_ratio: float = 0.0
    in_asymptotic_range: bool = False
    monotonic_convergence: bool = True

    # Status
    status: str = "UNKNOWN"
    notes: List[str] = field(default_factory=list)


def richardson_extrapolation(
    phi_fine: float,
    phi_medium: float,
    phi_coarse: float,
    r_21: float = 2.0,
    r_32: float = 2.0,
    Fs: float = 1.25,
) -> GCIResult:
    """
    Perform Richardson extrapolation and compute GCI for three grids.

    Parameters
    ----------
    phi_fine : float
        Solution on the finest grid.
    phi_medium : float
        Solution on the medium grid.
    phi_coarse : float
        Solution on the coarsest grid.
    r_21 : float
        Refinement ratio (medium / fine grid spacing).
    r_32 : float
        Refinement ratio (coarse / medium grid spacing).
    Fs : float
        Safety factor (1.25 for 3+ grids, 3.0 for 2 grids).

    Returns
    -------
    GCIResult
        Complete grid convergence analysis results.
    """
    result = GCIResult(n_grids=3)
    result.phi_values = [phi_fine, phi_medium, phi_coarse]
    result.refinement_ratios = [r_21, r_32]

    eps_21 = phi_medium - phi_fine
    eps_32 = phi_coarse - phi_medium

    # Check for monotonic convergence
    if abs(eps_21) < 1e-15 and abs(eps_32) < 1e-15:
        result.status = "CONVERGED (differences < machine precision)"
        result.phi_extrapolated = phi_fine
        result.gci_fine = 0.0
        result.in_asymptotic_range = True
        return result

    # Check convergence type
    # Convention: R = eps_21 / eps_32 (Celik et al. 2008)
    #   0 < R < 1  → monotonic convergence (errors shrink with refinement)
    #   R < 0      → oscillatory convergence
    #   R > 1      → divergence (errors grow with refinement)
    # Here ratio = eps_32 / eps_21 = 1/R, so:
    #   ratio > 1  → monotonic convergence (R < 1)
    #   ratio < 0  → oscillatory
    #   0 < ratio < 1 → divergence (R > 1)
    ratio = eps_32 / eps_21 if abs(eps_21) > 1e-15 else float("inf")

    if ratio < 0:
        # Oscillatory convergence
        result.monotonic_convergence = False
        result.notes.append("Oscillatory convergence detected")
        eps_21_abs = abs(eps_21)
        eps_32_abs = abs(eps_32)
    elif 0 < ratio < 1:
        # Divergent: fine→medium change is larger than medium→coarse
        result.status = "DIVERGENT (solution not converging with refinement)"
        result.notes.append("Anti-convergence: solutions diverge with refinement")
        return result
    else:
        # Monotonic convergence (ratio >= 1)
        result.monotonic_convergence = True
        eps_21_abs = abs(eps_21)
        eps_32_abs = abs(eps_32)

    # Observed order of accuracy (iterative for non-uniform refinement)
    if abs(eps_32_abs) > 1e-15 and abs(eps_21_abs) > 1e-15:
        if r_21 == r_32:
            # Uniform refinement
            p = np.log(eps_32_abs / eps_21_abs) / np.log(r_21)
        else:
            # Non-uniform: iterative method (Celik et al. 2008)
            p = _iterative_order(eps_21, eps_32, r_21, r_32)
    else:
        p = 2.0  # Default to second order
        result.notes.append("Cannot determine order; defaulting to p=2.0")

    p = max(0.5, min(p, 10.0))  # Bound to physical range
    result.observed_order = p

    # Richardson extrapolation: phi_ext = phi_1 + (phi_1 - phi_2) / (r^p - 1)
    # Since eps_21 = phi_medium - phi_fine = -(phi_1 - phi_2), we negate it
    result.phi_extrapolated = phi_fine - eps_21 / (r_21**p - 1)

    # GCI
    e_21 = abs(eps_21 / phi_fine) if abs(phi_fine) > 1e-15 else abs(eps_21)
    e_32 = abs(eps_32 / phi_medium) if abs(phi_medium) > 1e-15 else abs(eps_32)

    result.gci_fine = Fs * e_21 / (r_21**p - 1) * 100  # As percentage
    result.gci_coarse = Fs * e_32 / (r_32**p - 1) * 100

    # Asymptotic range check
    if result.gci_fine > 1e-15:
        result.asymptotic_ratio = result.gci_coarse / (r_21**p * result.gci_fine)
    else:
        result.asymptotic_ratio = 1.0
    result.in_asymptotic_range = abs(result.asymptotic_ratio - 1.0) < 0.1

    # Status
    if result.gci_fine < 5.0:
        result.status = "GRID INDEPENDENT (GCI < 5%)"
    elif result.gci_fine < 10.0:
        result.status = "ACCEPTABLE (5% < GCI < 10%)"
    else:
        result.status = f"NEEDS REFINEMENT (GCI = {result.gci_fine:.1f}%)"

    return result


def _iterative_order(
    eps_21: float, eps_32: float, r_21: float, r_32: float, tol: float = 1e-6
) -> float:
    """Iteratively solve for observed order with non-uniform refinement (Celik 2008)."""
    p = 1.5  # Initial guess
    for _ in range(100):
        s = np.sign(eps_32 / eps_21) if abs(eps_21) > 1e-15 else 1.0
        q = np.log((r_21**p - s) / (r_32**p - s))
        if abs(q) < 1e-15:
            break
        p_new = abs(np.log(abs(eps_32 / eps_21))) / np.log(r_21) + q / np.log(r_21)
        p_new = abs(p_new)
        if abs(p_new - p) < tol:
            return p_new
        p = p_new
    return p


# =============================================================================
# Multi-Quantity GCI
# =============================================================================
def multi_quantity_gci(
    quantities: Dict[str, Tuple[float, float, float]],
    r_21: float = 2.0,
    r_32: float = 2.0,
) -> Dict[str, GCIResult]:
    """
    Run GCI analysis on multiple output quantities simultaneously.

    Parameters
    ----------
    quantities : dict
        {name: (phi_fine, phi_medium, phi_coarse)} for each quantity.
    r_21, r_32 : float
        Refinement ratios.

    Returns
    -------
    dict of GCIResult for each quantity.
    """
    results = {}
    for name, (f, m, c) in quantities.items():
        results[name] = richardson_extrapolation(f, m, c, r_21, r_32)
    return results


# =============================================================================
# Reporting
# =============================================================================
def print_gci_report(result: GCIResult, label: str = "") -> None:
    """Print a formatted GCI analysis report."""
    print(f"\n{'='*60}")
    print(f"  Grid Convergence Study{f': {label}' if label else ''}")
    print(f"{'='*60}")
    print(f"  Grids:                {result.n_grids}")
    print(f"  φ values:             {result.phi_values}")
    print(f"  Refinement ratios:    {result.refinement_ratios}")
    print(f"  Observed order p:     {result.observed_order:.3f}")
    print(f"  Richardson extrap.:   {result.phi_extrapolated:.6f}")
    print(f"  GCI (fine):           {result.gci_fine:.2f}%")
    print(f"  GCI (coarse):         {result.gci_coarse:.2f}%")
    print(f"  Asymptotic ratio:     {result.asymptotic_ratio:.4f}")
    print(f"  In asymptotic range:  {result.in_asymptotic_range}")
    print(f"  Monotonic:            {result.monotonic_convergence}")
    print(f"  Status:               {result.status}")
    if result.notes:
        for note in result.notes:
            print(f"  Note: {note}")
    print(f"{'='*60}")
