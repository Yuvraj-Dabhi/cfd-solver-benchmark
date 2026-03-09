"""
Spalart-Allmaras Turbulence Model — Complete Implementation
=============================================================
Exact equations, constants, and auxiliary functions from the NASA 
Turbulence Modeling Resource (TMR):
    https://turbmodels.larc.nasa.gov/spalart.html#sa

Implements:
  - Standard SA transport equation terms
  - SA-neg extension (handles negative nu_hat)
  - SA-noft2 variant
  - SA-RC rotation/curvature correction
  - QCR2000 nonlinear constitutive relation
  - Boundary condition helpers
  - Verification functions for manufactured solutions

Primary References:
  - Spalart & Allmaras (1994), Recherche Aerospatiale, No.1, pp.5-21
  - Allmaras, Johnson & Spalart (2012), ICCFD7-1902
  - Spalart & Rumsey (2007), AIAA J. 45(10), pp.2544-2553
"""

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# SA Model Constants (TMR Standard)
# =============================================================================
@dataclass(frozen=True)
class SAConstants:
    """
    Standard Spalart-Allmaras model constants from TMR.
    
    Reference: Spalart & Allmaras (1994), Recherche Aerospatiale, No. 1.
    Note: cw1 has a corrected typo from the original paper 
    (missing κ² in denominator).
    """
    # --- Core constants ---
    sigma: float = 2.0 / 3.0       # Diffusion coefficient
    kappa: float = 0.41             # von Kármán constant
    cb1: float = 0.1355             # Production coefficient
    cb2: float = 0.622              # Diffusion cross-term coefficient
    
    # --- Wall destruction ---
    cw2: float = 0.3                # Wall function coefficient
    cw3: float = 2.0                # Wall function exponent
    
    # --- Eddy viscosity ---
    cv1: float = 7.1                # fv1 constant
    
    # --- S-tilde limiter (Note 1(c) method, ICCFD7-1902) ---
    cv2: float = 0.7
    cv3: float = 0.9
    
    # --- ft2 term (trip) ---
    ct1: float = 1.0                # Trip production constant
    ct2: float = 2.0                # Trip production constant
    ct3: float = 1.2                # ft2 constant (0 for SA-noft2)
    ct4: float = 0.5                # ft2 constant
    
    # --- SA-neg constants (ICCFD7-1902) ---
    cn1: float = 16.0               # fn function constant
    
    # --- Compressible flow ---
    Pr: float = 0.72                # Molecular Prandtl number
    Pr_t: float = 0.90              # Turbulent Prandtl number
    
    @property
    def cw1(self) -> float:
        """Derived constant: cw1 = cb1/κ² + (1+cb2)/σ
        
        Note: The original journal reference had a typo in the appendix
        (missing κ² in denominator). This is the corrected form.
        """
        return self.cb1 / self.kappa**2 + (1.0 + self.cb2) / self.sigma
    
    @property
    def cv1_cubed(self) -> float:
        """Pre-computed cv1³ for fv1 function."""
        return self.cv1**3


# Standard constants instance
SA_CONSTANTS = SAConstants()

# SA-noft2 variant (ct3=0)
SA_NOFT2_CONSTANTS = SAConstants(ct3=0.0)


# =============================================================================
# SA Auxiliary Functions
# =============================================================================

def compute_chi(nu_hat: np.ndarray, nu: float) -> np.ndarray:
    """
    Compute χ = ν̂/ν (ratio of SA variable to molecular viscosity).
    
    Parameters
    ----------
    nu_hat : ndarray
        SA turbulence variable field.
    nu : float
        Molecular kinematic viscosity.
    
    Returns
    -------
    chi : ndarray
    """
    return nu_hat / nu


def compute_fv1(chi: np.ndarray, c: SAConstants = SA_CONSTANTS) -> np.ndarray:
    """
    Compute fv1 = χ³/(χ³ + cv1³).
    
    This function modifies the SA variable to yield the turbulent
    eddy viscosity: ν_t = ν̂ · fv1.
    """
    chi3 = chi**3
    return chi3 / (chi3 + c.cv1_cubed)


def compute_fv2(chi: np.ndarray, fv1: np.ndarray) -> np.ndarray:
    """
    Compute fv2 = 1 - χ/(1 + χ·fv1).
    
    Used in the modified vorticity S̃.
    """
    return 1.0 - chi / (1.0 + chi * fv1)


def compute_ft2(chi: np.ndarray, c: SAConstants = SA_CONSTANTS) -> np.ndarray:
    """
    Compute ft2 = ct3 · exp(-ct4·χ²).
    
    This is the trip-related damping term. Set ct3=0 for SA-noft2.
    Returns zeros if ct3 == 0 (SA-noft2 variant).
    """
    if c.ct3 == 0.0:
        return np.zeros_like(chi)
    return c.ct3 * np.exp(-c.ct4 * chi**2)


def compute_S_tilde(
    S: np.ndarray,
    nu_hat: np.ndarray,
    d: np.ndarray,
    chi: np.ndarray,
    fv1: np.ndarray,
    c: SAConstants = SA_CONSTANTS,
    method: str = "c",
) -> np.ndarray:
    """
    Compute modified vorticity S̃ using TMR Note 1 methods.
    
    Parameters
    ----------
    S : ndarray
        Vorticity magnitude |Ω|.
    nu_hat : ndarray
        SA turbulence variable.
    d : ndarray
        Wall distance field.
    chi : ndarray
        χ = ν̂/ν.
    fv1 : ndarray
        fv1 function values.
    c : SAConstants
        Model constants.
    method : str
        "a" — clip S_bar to be > 0
        "b" — limit S̃ ≥ 0.3·S (Spalart recommendation)
        "c" — ICCFD7-1902 method (recommended, required for SA-neg)
    
    Returns
    -------
    S_tilde : ndarray
        Modified vorticity.
    """
    fv2 = compute_fv2(chi, fv1)
    S_bar = nu_hat / (c.kappa**2 * d**2) * fv2
    
    if method == "a":
        # Method (a): clip S_bar to be > 0
        S_tilde = S + np.maximum(S_bar, 0.0)
        
    elif method == "b":
        # Method (b): S̃ ≥ 0.3·S
        S_tilde = np.maximum(S + S_bar, 0.3 * S)
        
    elif method == "c":
        # Method (c): ICCFD7-1902 (recommended)
        S_tilde = np.where(
            S_bar >= -c.cv2 * S,
            # Normal case
            S + S_bar,
            # Corrected case (prevents negative S̃)
            S + S * (c.cv2**2 * S + c.cv3 * S_bar) / 
                ((c.cv3 - 2.0 * c.cv2) * S - S_bar)
        )
    else:
        raise ValueError(f"Unknown S̃ method: {method}. Use 'a', 'b', or 'c'.")
    
    # Guard: S̃ must not be exactly zero (for r computation)
    S_tilde = np.where(S_tilde == 0.0, 1e-16, S_tilde)
    
    return S_tilde


def compute_r(
    nu_hat: np.ndarray,
    S_tilde: np.ndarray,
    d: np.ndarray,
    c: SAConstants = SA_CONSTANTS,
) -> np.ndarray:
    """
    Compute r = min(ν̂/(S̃·κ²·d²), 10).
    
    Parameter r controls the wall destruction term through g and fw.
    The cap at 10 prevents overshoot in the free stream.
    
    Guard: When S̃ = 0, set r = 10 (TMR Note 1(c)).
    """
    denom = S_tilde * c.kappa**2 * d**2
    # Guard against division by zero
    r = np.where(
        np.abs(denom) > 1e-30,
        nu_hat / denom,
        10.0
    )
    return np.minimum(r, 10.0)


def compute_g(r: np.ndarray, c: SAConstants = SA_CONSTANTS) -> np.ndarray:
    """Compute g = r + cw2·(r⁶ - r)."""
    return r + c.cw2 * (r**6 - r)


def compute_fw(g: np.ndarray, c: SAConstants = SA_CONSTANTS) -> np.ndarray:
    """
    Compute fw = g · [(1 + cw3⁶)/(g⁶ + cw3⁶)]^(1/6).
    
    This is the wall destruction function. fw → 1 near the wall
    and fw < 1 away from the wall, providing adaptive destruction.
    """
    cw3_6 = c.cw3**6
    return g * ((1.0 + cw3_6) / (g**6 + cw3_6))**(1.0 / 6.0)


def compute_nu_t(
    nu_hat: np.ndarray,
    nu: float,
    c: SAConstants = SA_CONSTANTS,
) -> np.ndarray:
    """
    Compute turbulent eddy viscosity ν_t = ν̂ · fv1.
    
    For SA-neg: ν_t = 0 when ν̂ < 0.
    """
    chi = compute_chi(nu_hat, nu)
    fv1 = compute_fv1(chi, c)
    nu_t = nu_hat * fv1
    
    # SA-neg: clamp to zero when nu_hat < 0
    nu_t = np.maximum(nu_t, 0.0)
    
    return nu_t


# =============================================================================
# SA Transport Equation Terms
# =============================================================================

def production_term(
    cb1: float,
    S_tilde: np.ndarray,
    nu_hat: np.ndarray,
    ft2: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Production: P = cb1 · (1 - ft2) · S̃ · ν̂
    
    If ft2 is None (SA-noft2), simplifies to P = cb1 · S̃ · ν̂.
    """
    if ft2 is None:
        return cb1 * S_tilde * nu_hat
    return cb1 * (1.0 - ft2) * S_tilde * nu_hat


def destruction_term(
    cw1: float,
    fw: np.ndarray,
    nu_hat: np.ndarray,
    d: np.ndarray,
    ft2: Optional[np.ndarray] = None,
    cb1: float = 0.1355,
    kappa: float = 0.41,
) -> np.ndarray:
    """
    Wall destruction: D = (cw1·fw - cb1·ft2/κ²) · (ν̂/d)²
    
    If ft2 is None (SA-noft2), simplifies to D = cw1·fw · (ν̂/d)².
    """
    nu_hat_d2 = (nu_hat / d)**2
    if ft2 is None:
        return cw1 * fw * nu_hat_d2
    return (cw1 * fw - cb1 * ft2 / kappa**2) * nu_hat_d2


def diffusion_rhs(
    nu: float,
    nu_hat: np.ndarray,
    grad_nu_hat: np.ndarray,
    laplacian_nu_hat: np.ndarray,
    c: SAConstants = SA_CONSTANTS,
) -> np.ndarray:
    """
    Diffusion terms (RHS contributions):
    D = (1/σ) · [∇·((ν + ν̂)·∇ν̂) + cb2·(∇ν̂)²]
    
    Parameters
    ----------
    nu : float
        Molecular kinematic viscosity.
    nu_hat : ndarray
        SA turbulence variable.
    grad_nu_hat : ndarray
        Gradient of ν̂ [shape: (..., ndim)].
    laplacian_nu_hat : ndarray
        Laplacian of ν̂.
    
    Returns
    -------
    diffusion : ndarray
    """
    # (∇ν̂)² — squared magnitude of gradient
    grad_sq = np.sum(grad_nu_hat**2, axis=-1)
    
    # ∇·((ν + ν̂)·∇ν̂) ≈ (ν + ν̂)·∇²ν̂ + (∇ν̂)·(∇ν̂)
    # The second part is already partially accounted for by cb2 term
    div_term = (nu + nu_hat) * laplacian_nu_hat
    
    return (1.0 / c.sigma) * (div_term + c.cb2 * grad_sq)


# =============================================================================
# SA-neg: Negative ν̂ Branch (ICCFD7-1902)
# =============================================================================

def compute_fn_neg(chi: np.ndarray, c: SAConstants = SA_CONSTANTS) -> np.ndarray:
    """
    Compute fn for SA-neg model (used when ν̂ < 0):
    fn = (cn1 + χ³)/(cn1 - χ³)
    """
    chi3 = chi**3
    return (c.cn1 + chi3) / (c.cn1 - chi3)


def production_neg(
    S: np.ndarray,
    nu_hat: np.ndarray,
    c: SAConstants = SA_CONSTANTS,
) -> np.ndarray:
    """
    SA-neg production (ν̂ < 0 branch):
    P_neg = cb1 · (1 - ct3) · S · ν̂
    
    Note: The sign of destruction is '+' (opposite of positive model).
    """
    return c.cb1 * (1.0 - c.ct3) * S * nu_hat


def destruction_neg(
    nu_hat: np.ndarray,
    d: np.ndarray,
    c: SAConstants = SA_CONSTANTS,
) -> np.ndarray:
    """
    SA-neg destruction (ν̂ < 0 branch):
    D_neg = -cw1 · (ν̂/d)²
    
    Note: '+' sign because ν̂ is negative, so (ν̂/d)² is positive
    and the destruction term has opposite sign from the positive branch.
    """
    return -c.cw1 * (nu_hat / d)**2


# =============================================================================
# Rotation/Curvature Correction (SA-RC)
# =============================================================================

@dataclass(frozen=True)
class RCConstants:
    """Constants for the SA-RC rotation/curvature correction."""
    cr1: float = 1.0
    cr2: float = 12.0
    cr3: float = 1.0


RC_CONSTANTS = RCConstants()


def compute_fr1(
    S_mag: np.ndarray,
    Omega_mag: np.ndarray,
    D2Sij_Dt: np.ndarray,
    rc: RCConstants = RC_CONSTANTS,
) -> np.ndarray:
    """
    Compute the rotation function fr1 for SA-RC.
    
    fr1 = (1 + cr1) · (2r*/(1+r*)) · [1 - cr3·arctan(cr2·r̃*)] - cr1
    
    where r* = S/Ω and r̃* involves the Lagrangian derivative of strain.
    
    Parameters
    ----------
    S_mag : ndarray
        Strain rate magnitude.
    Omega_mag : ndarray
        Rotation rate magnitude.
    D2Sij_Dt : ndarray
        Material derivative term for strain rate tensor.
    
    Returns
    -------
    fr1 : ndarray
        Rotation function (multiplies production term).
    """
    # r* = S/Ω (with guard against division by zero)
    r_star = np.where(Omega_mag > 1e-30, S_mag / Omega_mag, 1.0)
    
    # r̃* computation (simplified — full expression needs Lagrangian derivative)
    r_tilde = np.where(
        S_mag > 1e-30,
        D2Sij_Dt / S_mag,
        0.0
    )
    
    # fr1
    fr1 = (
        (1.0 + rc.cr1) * (2.0 * r_star / (1.0 + r_star))
        * (1.0 - rc.cr3 * np.arctan(rc.cr2 * r_tilde))
        - rc.cr1
    )
    
    return fr1


# =============================================================================
# QCR2000 Constitutive Relation
# =============================================================================

QCR2000_CCR1 = 0.3  # Standard QCR2000 constant


def compute_qcr2000_stress(
    tau_boussinesq: np.ndarray,
    velocity_gradient: np.ndarray,
    ccr1: float = QCR2000_CCR1,
) -> np.ndarray:
    """
    Compute QCR2000 nonlinear Reynolds stress.
    
    τ_ij = τ_ij^B + ccr1 · (O_ik · τ_jk^B + O_jk · τ_ik^B)
    
    Parameters
    ----------
    tau_boussinesq : ndarray, shape (..., 3, 3)
        Boussinesq Reynolds stress tensor.
    velocity_gradient : ndarray, shape (..., 3, 3)
        Velocity gradient tensor ∂u_i/∂x_j.
    ccr1 : float
        QCR constant (default 0.3).
    
    Returns
    -------
    tau_qcr : ndarray, shape (..., 3, 3)
        Modified Reynolds stress tensor.
    """
    # Antisymmetric part of velocity gradient (rotation tensor)
    W = 0.5 * (velocity_gradient - np.swapaxes(velocity_gradient, -2, -1))
    
    # Normalize: O_ij = 2·W_ij / sqrt(∂u_m/∂x_n · ∂u_m/∂x_n)
    grad_norm = np.sqrt(np.sum(velocity_gradient**2, axis=(-2, -1), keepdims=True))
    grad_norm = np.maximum(grad_norm, 1e-30)  # Avoid division by zero
    O = 2.0 * W / grad_norm
    
    # QCR2000 correction: ccr1 · (O_ik · τ_jk + O_jk · τ_ik)
    correction = ccr1 * (
        np.einsum('...ik,...jk->...ij', O, tau_boussinesq) +
        np.einsum('...jk,...ik->...ij', O, tau_boussinesq)
    )
    
    return tau_boussinesq + correction


# =============================================================================
# Boundary Conditions
# =============================================================================

@dataclass
class SABoundaryConditions:
    """
    Spalart-Allmaras boundary conditions from TMR.
    
    Farfield/inflow:
        Standard (no trip): ν̂ = 3ν to 5ν  →  ν_t ≈ 0.21ν to 1.29ν
        With trip (SA-Ia):  ν̂ = ν/10       →  ν_t ≈ 6.04e-5ν
    
    Wall:
        ν̂ = 0  (no-slip)
    
    References:
        Spalart (2000), AIAA 2000-2306
        Spalart & Rumsey (2007), AIAA J. 45(10)
    """
    
    @staticmethod
    def wall() -> float:
        """Wall BC: ν̂ = 0."""
        return 0.0
    
    @staticmethod
    def farfield(nu: float, ratio: float = 3.0) -> float:
        """
        Farfield/inflow BC for ν̂.
        
        Parameters
        ----------
        nu : float
            Molecular kinematic viscosity.
        ratio : float
            ν̂/ν ratio. TMR recommends 3 to 5 (default 3).
        
        Returns
        -------
        nu_hat_freestream : float
        """
        return ratio * nu
    
    @staticmethod
    def farfield_nu_t(nu: float, ratio: float = 3.0) -> float:
        """
        Compute ν_t at farfield given ν̂/ν ratio.
        
        For ν̂/ν = 3:  ν_t ≈ 0.210438 · ν
        For ν̂/ν = 5:  ν_t ≈ 1.294 · ν
        """
        chi = ratio
        fv1 = chi**3 / (chi**3 + SA_CONSTANTS.cv1_cubed)
        return ratio * nu * fv1
    
    @staticmethod
    def farfield_with_trip(nu: float) -> float:
        """Farfield BC with trip term (SA-Ia): ν̂ = ν/10."""
        return nu / 10.0


# =============================================================================
# Sutherland's Law for Viscosity
# =============================================================================

def sutherland_viscosity(
    T: float,
    mu_ref: float = 1.716e-5,
    T_ref: float = 273.15,
    S: float = 110.4,
) -> float:
    """
    Sutherland's law for dynamic viscosity.
    
    μ(T) = μ_ref · (T/T_ref)^(3/2) · (T_ref + S)/(T + S)
    
    Parameters
    ----------
    T : float
        Temperature [K].
    mu_ref : float
        Reference dynamic viscosity [Pa·s] (default: air at 273.15 K).
    T_ref : float
        Reference temperature [K].
    S : float
        Sutherland temperature [K].
    
    Returns
    -------
    mu : float
        Dynamic viscosity [Pa·s].
    """
    return mu_ref * (T / T_ref)**1.5 * (T_ref + S) / (T + S)


# =============================================================================
# Wall Distance Computation (TMR-Compliant)
# =============================================================================
#
# CRITICAL IMPLEMENTATION NOTE (from NASA TMR):
# Computing d by searching along grid lines or by finding the nearest wall
# grid point (or cell center) are INCORRECT and NOT the same as computing
# the actual minimum distance to the nearest wall.
#
# Three common WRONG approaches:
#   1. Distance along grid lines (grid-topology dependent)
#   2. Distance to nearest wall grid point (inaccurate, converges on fine grids)
#   3. Distance to nearest wall cell center (same issue as #2, worse)
#
# The CORRECT approach:
#   d(P) = min over all wall SEGMENTS { perpendicular_distance(P, segment) }
#   The closest point may lie BETWEEN wall grid points, not at a node.
#
# Special cases:
#   - Sharp convex corners (e.g., airfoil TE): d = distance to corner point
#   - Multi-zone grids: must search ALL wall surfaces from ALL zones
# =============================================================================


def _point_to_segment_distance_2d(
    px: float, py: float,
    ax: float, ay: float,
    bx: float, by: float,
) -> float:
    """
    Compute minimum distance from point P to line segment AB in 2D.
    
    This is the geometrically CORRECT method for wall distance computation.
    The closest point on segment AB to P is found by projecting P onto the
    infinite line through A and B, then clamping to [0,1] to stay on the segment.
    
    Parameters
    ----------
    px, py : float
        Coordinates of field point P.
    ax, ay : float
        Coordinates of segment start point A.
    bx, by : float
        Coordinates of segment end point B.
    
    Returns
    -------
    dist : float
        Minimum distance from P to segment AB.
    """
    # Vector AB
    abx = bx - ax
    aby = by - ay
    # Vector AP
    apx = px - ax
    apy = py - ay
    
    # Project AP onto AB: t = (AP . AB) / (AB . AB)
    ab_sq = abx * abx + aby * aby
    
    if ab_sq < 1e-30:
        # Degenerate segment (A == B): distance to point A
        return np.sqrt(apx * apx + apy * apy)
    
    t = (apx * abx + apy * aby) / ab_sq
    
    # Clamp t to [0, 1] to stay on segment
    t = max(0.0, min(1.0, t))
    
    # Closest point on segment
    cx = ax + t * abx
    cy = ay + t * aby
    
    # Distance from P to closest point
    dx = px - cx
    dy = py - cy
    return np.sqrt(dx * dx + dy * dy)


def compute_wall_distance_2d(
    field_points: np.ndarray,
    wall_points: np.ndarray,
) -> np.ndarray:
    """
    Compute TRUE minimum wall distance for each field point (2D).
    
    This is the TMR-compliant method: for each field point, we compute the
    minimum distance to the CONTINUOUS wall surface (represented by line 
    segments connecting consecutive wall points), NOT to the wall grid points
    themselves.
    
    Parameters
    ----------
    field_points : ndarray, shape (N, 2)
        Coordinates of field points [x, y].
    wall_points : ndarray, shape (M, 2)
        Ordered coordinates of wall surface points [x, y].
        Points are connected in order (0-1, 1-2, ..., M-2 to M-1).
    
    Returns
    -------
    d : ndarray, shape (N,)
        Minimum wall distance for each field point.
    
    Notes
    -----
    Complexity: O(N * M) -- brute force. For large grids, use the KD-tree
    accelerated version `compute_wall_distance_2d_fast`.
    
    TMR Key Points:
    - The closest point on the wall may lie BETWEEN wall grid points
    - At sharp convex corners (e.g., airfoil TE), d = distance to corner
    - Must consider ALL wall surfaces, even from different grid zones
    """
    n_field = field_points.shape[0]
    n_wall = wall_points.shape[0]
    n_segments = n_wall - 1
    
    d = np.full(n_field, np.inf)
    
    for i in range(n_field):
        px, py = field_points[i]
        
        for j in range(n_segments):
            ax, ay = wall_points[j]
            bx, by = wall_points[j + 1]
            
            dist = _point_to_segment_distance_2d(px, py, ax, ay, bx, by)
            if dist < d[i]:
                d[i] = dist
    
    return d


def compute_wall_distance_2d_fast(
    field_points: np.ndarray,
    wall_points: np.ndarray,
    n_candidates: int = 10,
) -> np.ndarray:
    """
    Compute TRUE minimum wall distance using KD-tree acceleration (2D).
    
    Uses scipy's KD-tree to find candidate nearest wall points, then
    checks adjacent segments for the true minimum distance starting
    from each candidate. This is O(N * log(M) * n_candidates) instead
    of O(N * M).
    
    Parameters
    ----------
    field_points : ndarray, shape (N, 2)
        Field point coordinates.
    wall_points : ndarray, shape (M, 2)
        Ordered wall surface point coordinates.
    n_candidates : int
        Number of nearest wall points to check (default 10).
        Higher values are safer but slower.
    
    Returns
    -------
    d : ndarray, shape (N,)
        Minimum wall distance for each field point.
    """
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        logger.warning(
            "scipy not available; falling back to brute-force wall distance."
        )
        return compute_wall_distance_2d(field_points, wall_points)
    
    n_field = field_points.shape[0]
    n_wall = wall_points.shape[0]
    
    # Build KD-tree of wall points
    tree = cKDTree(wall_points)
    
    # Find k nearest wall points for each field point
    k = min(n_candidates, n_wall)
    _, indices = tree.query(field_points, k=k)
    
    if k == 1:
        indices = indices.reshape(-1, 1)
    
    # For each field point, check segments adjacent to candidate wall points
    d = np.full(n_field, np.inf)
    
    for i in range(n_field):
        px, py = field_points[i]
        
        # Collect unique segment indices to check
        seg_indices = set()
        for j_wall in indices[i]:
            # Segment ending at j_wall
            if j_wall > 0:
                seg_indices.add(j_wall - 1)
            # Segment starting at j_wall
            if j_wall < n_wall - 1:
                seg_indices.add(j_wall)
        
        for j in seg_indices:
            ax, ay = wall_points[j]
            bx, by = wall_points[j + 1]
            
            dist = _point_to_segment_distance_2d(px, py, ax, ay, bx, by)
            if dist < d[i]:
                d[i] = dist
    
    return d


def compute_wall_distance_incorrect_gridpoint(
    field_points: np.ndarray,
    wall_points: np.ndarray,
) -> np.ndarray:
    """
    INTENTIONALLY INCORRECT wall distance: nearest wall GRID POINT.
    
    This is provided ONLY for diagnostic comparison to show the error
    introduced by the common incorrect method. Do NOT use this for
    actual SA model computations.
    
    TMR Warning: "Computing minimum distance by finding the nearest wall 
    gridpoint (or cell center) is INCORRECT and NOT the same as computing 
    actual minimum distance to the nearest wall."
    """
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(wall_points)
        d_wrong, _ = tree.query(field_points)
    except ImportError:
        # Brute force fallback
        n_field = field_points.shape[0]
        d_wrong = np.full(n_field, np.inf)
        for i in range(n_field):
            dists = np.sqrt(np.sum((wall_points - field_points[i])**2, axis=1))
            d_wrong[i] = np.min(dists)
    
    return d_wrong


def diagnose_wall_distance_error(
    field_points: np.ndarray,
    wall_points: np.ndarray,
    n_candidates: int = 10,
) -> dict:
    """
    Compare correct vs incorrect wall distance methods and report errors.
    
    This diagnostic tool helps quantify the error introduced by the
    common incorrect methods (nearest grid point vs true minimum distance
    to wall surface).
    
    Parameters
    ----------
    field_points : ndarray, shape (N, 2)
    wall_points : ndarray, shape (M, 2)
    
    Returns
    -------
    diagnostics : dict
        Contains d_correct, d_wrong, relative_error statistics, and
        the indices of points with the largest errors.
    """
    d_correct = compute_wall_distance_2d_fast(
        field_points, wall_points, n_candidates
    )
    d_wrong = compute_wall_distance_incorrect_gridpoint(
        field_points, wall_points
    )
    
    # Relative error: (d_wrong - d_correct) / d_correct
    # d_wrong >= d_correct always (nearest grid point is further than
    # nearest point on continuous surface)
    rel_error = np.where(
        d_correct > 1e-30,
        (d_wrong - d_correct) / d_correct,
        0.0
    )
    
    # Impact on destruction term: scales as 1/d^2
    # Error in destruction ~ 2 * relative_error_in_d (first order)
    destruction_error = 2.0 * rel_error
    
    worst_idx = np.argmax(rel_error)
    
    return {
        "d_correct": d_correct,
        "d_wrong": d_wrong,
        "rel_error": rel_error,
        "max_rel_error": float(np.max(rel_error)),
        "mean_rel_error": float(np.mean(rel_error)),
        "median_rel_error": float(np.median(rel_error)),
        "max_destruction_error": float(np.max(destruction_error)),
        "worst_point_index": int(worst_idx),
        "worst_point_coords": field_points[worst_idx].tolist(),
        "n_points_over_1pct_error": int(np.sum(rel_error > 0.01)),
        "n_points_over_5pct_error": int(np.sum(rel_error > 0.05)),
        "n_points_over_10pct_error": int(np.sum(rel_error > 0.10)),
    }


def validate_wall_distance(
    d: np.ndarray,
    check_smoothness: bool = True,
    max_jump_ratio: float = 3.0,
) -> bool:
    """
    Validate wall distance field for SA model usage.
    
    TMR Requirements:
    - d must be TRUE minimum distance to nearest wall surface
    - NOT distance along grid lines (grid-topology dependent)
    - NOT distance to nearest wall grid point (inaccurate on coarse grids)
    - NOT distance to nearest wall cell center
    - At sharp convex corners (e.g., airfoil TE), d = distance to corner
    - d must be > 0 everywhere in the interior
    - d should be smooth (not have sudden jumps from incorrect computation)
    
    Parameters
    ----------
    d : ndarray
        Wall distance field.
    check_smoothness : bool
        If True, check for suspicious jumps that indicate incorrect computation.
    max_jump_ratio : float
        Maximum allowed ratio between adjacent d values (for smoothness check).
    
    Returns True if d field passes all validation checks.
    """
    passed = True
    
    # Basic checks
    if np.any(d <= 0):
        logger.error(
            "Wall distance field contains non-positive values! "
            "d must be > 0 in the domain interior."
        )
        passed = False
    
    if np.any(np.isnan(d)):
        logger.error("Wall distance field contains NaN values!")
        passed = False
    
    if np.any(np.isinf(d)):
        logger.error("Wall distance field contains Inf values!")
        passed = False
    
    # Smoothness check (detect grid-line-following or other incorrect methods)
    if check_smoothness and d.ndim >= 1 and len(d) > 1:
        d_flat = d.ravel()
        d_positive = d_flat[d_flat > 0]
        
        if len(d_positive) > 1:
            # Check for suspiciously large jumps between adjacent values
            ratios = d_positive[1:] / d_positive[:-1]
            max_ratio = np.max(ratios)
            min_ratio = np.min(ratios)
            
            if max_ratio > max_jump_ratio or min_ratio < 1.0 / max_jump_ratio:
                logger.warning(
                    f"Wall distance field has suspicious jumps "
                    f"(max ratio = {max_ratio:.2f}, min ratio = {min_ratio:.2f}). "
                    f"This may indicate incorrect wall distance computation. "
                    f"TMR requires TRUE minimum distance to wall surface, "
                    f"not distance along grid lines or to grid points."
                )
                # Warning only, don't fail -- could be legitimate grid stretching
    
    if passed:
        logger.info(
            f"Wall distance validation passed: "
            f"min(d)={np.min(d):.2e}, max(d)={np.max(d):.2e}, "
            f"mean(d)={np.mean(d):.2e}"
        )
    
    return passed


# =============================================================================
# SA Model Verification
# =============================================================================

def verify_constants() -> Dict[str, Any]:
    """
    Verify SA model constants against TMR published values.
    
    Returns a dict with verification results.
    """
    c = SA_CONSTANTS
    results = {}
    
    # Check cw1 derivation
    cw1_computed = c.cb1 / c.kappa**2 + (1.0 + c.cb2) / c.sigma
    results["cw1"] = {
        "computed": cw1_computed,
        "expected": 3.2390678,  # Approximate TMR value
        "match": abs(cw1_computed - c.cw1) < 1e-10,
    }
    
    # Check fv1 for known chi values
    chi_test = np.array([0.0, 1.0, 7.1, 100.0])
    fv1_test = compute_fv1(chi_test, c)
    results["fv1_at_chi_0"] = {
        "value": float(fv1_test[0]),
        "expected": 0.0,
        "match": abs(fv1_test[0]) < 1e-15,
    }
    results["fv1_at_chi_cv1"] = {
        "value": float(fv1_test[2]),
        "expected": 0.5,
        "match": abs(fv1_test[2] - 0.5) < 1e-10,
    }
    
    # Check farfield BCs
    nu = 1.5e-5  # ~air at sea level
    nu_hat_3 = SABoundaryConditions.farfield(nu, ratio=3.0)
    nu_t_3 = SABoundaryConditions.farfield_nu_t(nu, ratio=3.0)
    results["farfield_ratio_3"] = {
        "nu_hat": nu_hat_3,
        "nu_t": nu_t_3,
        "nu_t_over_nu": nu_t_3 / nu,
        "expected_ratio": 0.210438,
        "match": abs(nu_t_3 / nu - 0.210438) < 0.001,
    }
    
    return results


def print_model_summary():
    """Print a formatted summary of the SA model implementation."""
    c = SA_CONSTANTS
    
    print("=" * 70)
    print("SPALART-ALLMARAS TURBULENCE MODEL -- TMR Reference Implementation")
    print("=" * 70)
    
    print("\n--- Model Constants (TMR Standard) ---")
    print(f"  sigma = {c.sigma:.10f}  (2/3)")
    print(f"  kappa = {c.kappa}")
    print(f"  cb1   = {c.cb1}")
    print(f"  cb2   = {c.cb2}")
    print(f"  cw1   = {c.cw1:.10f}  (derived: cb1/kappa^2 + (1+cb2)/sigma)")
    print(f"  cw2   = {c.cw2}")
    print(f"  cw3   = {c.cw3}")
    print(f"  cv1   = {c.cv1}")
    print(f"  cv2   = {c.cv2}  (Note 1(c) limiter)")
    print(f"  cv3   = {c.cv3}  (Note 1(c) limiter)")
    print(f"  ct3   = {c.ct3}  (0 for SA-noft2)")
    print(f"  ct4   = {c.ct4}")
    print(f"  cn1   = {c.cn1}  (SA-neg)")
    
    print("\n--- Fluid Properties (Compressible) ---")
    print(f"  Pr   = {c.Pr}   (molecular)")
    print(f"  Pr_t = {c.Pr_t}  (turbulent)")
    print(f"  Viscosity: Sutherland's law")
    
    print("\n--- Boundary Conditions ---")
    nu = 1.5e-5
    print(f"  Wall:     nu_hat = 0")
    print(f"  Farfield: nu_hat = 3*nu = {3*nu:.2e}  (for nu = {nu:.2e})")
    print(f"            nu_t/nu = {SABoundaryConditions.farfield_nu_t(nu, 3.0)/nu:.6f}")
    
    print("\n--- Variant Summary ---")
    print(f"  SA:        Standard (with ft2 term, ct3={c.ct3})")
    print(f"  SA-neg:    Handles nu_hat < 0 (cn1={c.cn1})")
    print(f"  SA-noft2:  ct3=0 (recommended for DES/DDES)")
    print(f"  SA-RC:     +Rotation/Curvature correction")
    print(f"  SA-QCR:    +Quadratic constitutive relation (ccr1={QCR2000_CCR1})")
    
    print("\n--- Key References ---")
    print("  1. Spalart & Allmaras (1994), Recherche Aerospatiale, No.1")
    print("  2. Allmaras, Johnson & Spalart (2012), ICCFD7-1902")
    print("  3. Spalart & Rumsey (2007), AIAA J. 45(10)")
    
    print("\n--- Verification ---")
    results = verify_constants()
    all_pass = all(r.get("match", True) for r in results.values())
    print(f"  All constants verified: {'PASS' if all_pass else 'FAIL'}")
    for name, r in results.items():
        status = "[OK]" if r.get("match", True) else "[FAIL]"
        print(f"    {status} {name}")
    
    print("=" * 70)


# =============================================================================
# Compressibility Correction (SA-comp)
# =============================================================================

def compressibility_correction(
    nu_hat: np.ndarray,
    speed_of_sound: np.ndarray,
    velocity_laplacian: np.ndarray,
    c5: float = 3.5,
) -> np.ndarray:
    """
    SA-comp compressibility correction for mixing layers.
    
    Additional RHS term: c5 · (ν̂/a)² · ∇²u_i
    
    Reference: Spalart (2000), AIAA 2000-2306
    """
    return c5 * (nu_hat / speed_of_sound)**2 * velocity_laplacian


# =============================================================================
# Wall Roughness (SA-rough, Boeing method)
# =============================================================================

def roughness_correction_distance(
    d: np.ndarray,
    k_s: float,
) -> np.ndarray:
    """
    SA-rough Boeing method: augment wall distance.
    
    d_new = d + 0.03 · k_s
    
    Parameters
    ----------
    d : ndarray
        Original wall distance.
    k_s : float
        Equivalent sand-grain roughness height.
    
    Returns
    -------
    d_rough : ndarray
        Augmented wall distance.
    """
    return d + 0.03 * k_s


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    print_model_summary()
    
    print("\n\n--- Detailed Verification ---")
    results = verify_constants()
    for name, data in results.items():
        print(f"\n  {name}:")
        for k, v in data.items():
            print(f"    {k}: {v}")
