"""
Physics Diagnostics
===================
Advanced physics-based diagnostics for turbulence model assessment.
Identifies fundamental model-form errors in RANS closures.

Diagnostics implemented:
1. Boussinesq validity map
2. Production-to-dissipation ratio (P/ε)
3. Lumley triangle (anisotropy invariants)
4. Curvature Richardson number
5. Secondary flow strength
6. WMLES resolved stress budget
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class DiagnosticResult:
    """Container for physics diagnostic output."""
    name: str
    description: str
    values: np.ndarray = field(default_factory=lambda: np.array([]))
    regions_of_concern: List[str] = field(default_factory=list)
    summary: str = ""


# =============================================================================
# 1. Boussinesq Hypothesis Validity
# =============================================================================
def boussinesq_validity(
    Sij: np.ndarray,
    tau_ij: np.ndarray,
    k: np.ndarray,
) -> DiagnosticResult:
    """
    Assess validity of Boussinesq hypothesis: τ_ij = -2νt·S_ij + (2/3)k·δ_ij.

    The Boussinesq assumption fails when:
    - |a_ij - a_ij^Boussinesq| / |a_ij| > 0.5 (significant misalignment)
    - Strong curvature, 3D separation, or adverse pressure gradient

    Parameters
    ----------
    Sij : ndarray (N, 3, 3)
        Mean strain rate tensor.
    tau_ij : ndarray (N, 3, 3)
        Reynolds stress tensor (from DNS/LES or experiment).
    k : ndarray (N,)
        Turbulent kinetic energy.

    Returns
    -------
    DiagnosticResult with validity index (0=invalid, 1=valid)
    """
    N = len(k)
    I3 = np.eye(3)[np.newaxis, :, :]  # (1, 3, 3) for broadcasting

    # Normalized anisotropy from actual stresses: a_ij = tau/(2k) - delta/3
    k_safe = np.maximum(k, 1e-15)[:, np.newaxis, np.newaxis]  # (N, 1, 1)
    a_ij = tau_ij / (2 * k_safe) - I3 / 3  # (N, 3, 3)

    # Boussinesq prediction: a_ij_B ∝ -S_ij (normalized)
    S_norm = np.linalg.norm(Sij, axis=(-2, -1), keepdims=True)  # (N, 1, 1)
    S_norm = np.maximum(S_norm, 1e-15)
    a_ij_B = -Sij / S_norm  # (N, 3, 3)

    # Frobenius norms via einsum
    a_norm = np.sqrt(np.einsum('nij,nij->n', a_ij, a_ij))  # (N,)
    a_B_norm = np.sqrt(np.einsum('nij,nij->n', a_ij_B, a_ij_B))  # (N,)

    # Cosine similarity via einsum Frobenius inner product
    inner = np.einsum('nij,nij->n', a_ij, a_ij_B)  # (N,)
    denom = a_norm * a_B_norm + 1e-15
    cos_sim = inner / denom

    # Valid where turbulence exists and strain is non-trivial
    validity = np.maximum(cos_sim, 0.0)
    trivial = (k < 1e-15) | (S_norm.squeeze() < 1e-15) | (a_norm < 1e-15)
    validity = np.where(trivial, 1.0, validity)

    result = DiagnosticResult(
        name="Boussinesq Validity",
        description="Alignment between actual and Boussinesq-predicted anisotropy",
        values=validity,
    )

    invalid_frac = np.mean(validity < 0.5) * 100
    result.summary = f"{invalid_frac:.1f}% of points show Boussinesq breakdown"
    if invalid_frac > 20:
        result.regions_of_concern.append(
            "Significant Boussinesq breakdown detected - consider RSM or EARSM"
        )

    return result


# =============================================================================
# 2. Production-to-Dissipation Ratio
# =============================================================================
def production_dissipation_ratio(
    Sij: np.ndarray,
    tau_ij: np.ndarray,
    epsilon: np.ndarray,
) -> DiagnosticResult:
    """
    Compute P/ε ratio at each point.

    P/ε ≈ 1 in equilibrium (attached BL).
    P/ε >> 1 → strong production (APG)
    P/ε << 1 → transport-dominated (recirculation)

    Parameters
    ----------
    Sij : ndarray (N, 3, 3)
        Mean strain rate tensor.
    tau_ij : ndarray (N, 3, 3)
        Reynolds stress tensor.
    epsilon : ndarray (N,)
        Turbulent dissipation rate.
    """
    # Vectorized tensor contraction: P = -τ_ij · S_ij via einsum
    P = -np.einsum('nij,nij->n', tau_ij, Sij)  # (N,)

    P_over_eps = np.where(epsilon > 1e-15, P / epsilon, 0.0)

    result = DiagnosticResult(
        name="P/ε Ratio",
        description="Production-to-dissipation ratio (equilibrium = 1.0)",
        values=P_over_eps,
    )

    # Identify non-equilibrium regions
    non_eq = np.mean(np.abs(P_over_eps - 1.0) > 0.3) * 100
    result.summary = f"P/ε range: [{np.min(P_over_eps):.2f}, {np.max(P_over_eps):.2f}]; " \
                     f"{non_eq:.1f}% non-equilibrium"

    if np.max(P_over_eps) > 3.0:
        result.regions_of_concern.append("Strong excess production (APG/separation zone)")
    if np.min(P_over_eps) < 0.1:
        result.regions_of_concern.append("Low P/ε in recirculation (transport-dominated)")

    return result


# =============================================================================
# 3. Lumley Triangle (Anisotropy Invariants)
# =============================================================================
def lumley_triangle_invariants(
    uu: np.ndarray, vv: np.ndarray, ww: np.ndarray,
    uv: np.ndarray = None, uw: np.ndarray = None, vw: np.ndarray = None,
) -> DiagnosticResult:
    """
    Compute Lumley triangle invariants (ξ, η) for turbulence anisotropy.

    Maps turbulent state on the AIM (Anisotropy Invariant Map):
    - Origin: isotropic
    - Right vertex: 1-component
    - Top curve: 2-component axisymmetric
    - Bottom curve: disk-like axisymmetric

    Parameters
    ----------
    uu, vv, ww : ndarray
        Normal Reynolds stresses.
    uv, uw, vw : ndarray, optional
        Shear Reynolds stresses (default: 0).
    """
    N = len(uu)
    if uv is None: uv = np.zeros(N)
    if uw is None: uw = np.zeros(N)
    if vw is None: vw = np.zeros(N)

    k = 0.5 * (uu + vv + ww)  # (N,)
    k_safe = np.maximum(k, 1e-15)
    inv_2k = 1.0 / (2 * k_safe)  # (N,)

    # Build batch anisotropy tensor b_ij: shape (N, 3, 3)
    b = np.zeros((N, 3, 3))
    b[:, 0, 0] = uu * inv_2k - 1.0 / 3
    b[:, 1, 1] = vv * inv_2k - 1.0 / 3
    b[:, 2, 2] = ww * inv_2k - 1.0 / 3
    b[:, 0, 1] = b[:, 1, 0] = uv * inv_2k
    b[:, 0, 2] = b[:, 2, 0] = uw * inv_2k
    b[:, 1, 2] = b[:, 2, 1] = vw * inv_2k

    # II_b = -0.5 * tr(b @ b) = -0.5 * b_ij * b_ji via einsum
    II_b = -0.5 * np.einsum('nij,nji->n', b, b)  # (N,)

    # III_b = det(b) — batch determinant
    III_b = np.linalg.det(b)  # (N,)

    # eta = sqrt(|II_b| / 3), xi = cbrt(III_b / 2)
    eta = np.sqrt(np.abs(II_b) / 3.0)
    xi = np.sign(III_b) * np.cbrt(np.abs(III_b / 2.0))

    # Zero out points with negligible turbulence
    no_turb = k < 1e-15
    eta[no_turb] = 0.0
    xi[no_turb] = 0.0

    result = DiagnosticResult(
        name="Lumley Triangle",
        description="Anisotropy invariants (xi, eta) on the AIM",
        values=np.column_stack([xi, eta]),
    )

    # Classify dominant anisotropy state
    near_1c = np.sum(eta > 0.25) / N * 100
    near_iso = np.sum(eta < 0.05) / N * 100
    result.summary = f"Near-isotropic: {near_iso:.1f}%, Near-1-component: {near_1c:.1f}%"

    if near_1c > 30:
        result.regions_of_concern.append(
            "High 1-component anisotropy - Boussinesq models will fail"
        )

    return result


# =============================================================================
# 4. Curvature Richardson Number
# =============================================================================
def curvature_richardson_number(
    U: np.ndarray,
    y: np.ndarray,
    R: np.ndarray,
) -> DiagnosticResult:
    """
    Compute curvature Richardson number Ri_c = (U/R) / (dU/dy).

    Ri_c > 0 → stabilizing curvature (suppresses turbulence)
    Ri_c < 0 → destabilizing curvature (enhances turbulence)
    |Ri_c| > 0.01 → linear models unreliable

    Parameters
    ----------
    U : ndarray
        Streamwise velocity profile.
    y : ndarray
        Wall-normal coordinate.
    R : ndarray
        Local radius of curvature.
    """
    dU_dy = np.gradient(U, y)
    dU_dy = np.where(np.abs(dU_dy) > 1e-15, dU_dy, 1e-15)

    Ri_c = np.where(np.abs(R) > 1e-10, (U / R) / dU_dy, 0.0)

    result = DiagnosticResult(
        name="Curvature Richardson Number",
        description="Ri_c: stability parameter for curved flows",
        values=Ri_c,
    )

    destabilizing = np.mean(Ri_c < -0.01) * 100
    stabilizing = np.mean(Ri_c > 0.01) * 100
    result.summary = f"Destabilizing: {destabilizing:.1f}%, Stabilizing: {stabilizing:.1f}%"

    if destabilizing > 10 or stabilizing > 10:
        result.regions_of_concern.append(
            "Significant curvature effects — use SA-RC or rotation-corrected model"
        )

    return result


# =============================================================================
# 5. Secondary Flow Strength
# =============================================================================
def secondary_flow_strength(
    U: np.ndarray,
    V: np.ndarray,
    W: np.ndarray,
) -> DiagnosticResult:
    """
    Quantify secondary flow intensity as |V² + W²| / |U²|.

    Strong secondary flows (> 5%) indicate 3D effects that
    Boussinesq models typically miss (corner flows, horseshoe vortex).

    Parameters
    ----------
    U, V, W : ndarray
        Velocity components.
    """
    U_mag2 = U ** 2
    secondary_mag2 = V ** 2 + W ** 2
    total_mag2 = U_mag2 + secondary_mag2 + 1e-15

    intensity = np.sqrt(secondary_mag2 / total_mag2) * 100  # Percentage

    result = DiagnosticResult(
        name="Secondary Flow Intensity",
        description="Cross-flow intensity as % of total velocity",
        values=intensity,
    )

    strong = np.mean(intensity > 5.0) * 100
    result.summary = f"Mean intensity: {np.mean(intensity):.2f}%; >{5}%: {strong:.1f}% of field"

    if strong > 10:
        result.regions_of_concern.append(
            "Strong secondary flows — use QCR or RSM for improved accuracy"
        )

    return result


# =============================================================================
# 6. WMLES Resolved Stress Budget
# =============================================================================
def wmles_resolved_fraction(
    k_resolved: np.ndarray,
    k_modeled: np.ndarray,
) -> DiagnosticResult:
    """
    Compute fraction of TKE resolved in WMLES.

    Target: ≥80% resolved for quality LES.

    Parameters
    ----------
    k_resolved : ndarray
        Resolved TKE (from velocity fluctuations).
    k_modeled : ndarray
        Subgrid-scale TKE (from SGS model).
    """
    k_total = k_resolved + k_modeled + 1e-15
    fraction = k_resolved / k_total * 100

    result = DiagnosticResult(
        name="WMLES Resolved TKE Fraction",
        description="Percentage of TKE resolved (target ≥80%)",
        values=fraction,
    )

    well_resolved = np.mean(fraction > 80) * 100
    result.summary = f"Mean resolved: {np.mean(fraction):.1f}%; " \
                     f">{80}% resolved: {well_resolved:.1f}% of field"

    if well_resolved < 50:
        result.regions_of_concern.append(
            "Insufficient resolution — refine grid or increase SGS model"
        )

    return result


# =============================================================================
# All-in-one Summary
# =============================================================================
def run_all_diagnostics(
    Sij: np.ndarray = None,
    tau_ij: np.ndarray = None,
    k: np.ndarray = None,
    epsilon: np.ndarray = None,
    uu: np.ndarray = None,
    vv: np.ndarray = None,
    ww: np.ndarray = None,
    U: np.ndarray = None,
    V: np.ndarray = None,
    W: np.ndarray = None,
    y: np.ndarray = None,
    R: np.ndarray = None,
) -> Dict[str, DiagnosticResult]:
    """Run all applicable diagnostics and return results dict."""
    results = {}

    if Sij is not None and tau_ij is not None and k is not None:
        results["boussinesq"] = boussinesq_validity(Sij, tau_ij, k)

    if Sij is not None and tau_ij is not None and epsilon is not None:
        results["p_over_eps"] = production_dissipation_ratio(Sij, tau_ij, epsilon)

    if uu is not None and vv is not None and ww is not None:
        results["lumley"] = lumley_triangle_invariants(uu, vv, ww)

    if U is not None and y is not None and R is not None:
        results["ri_curvature"] = curvature_richardson_number(U, y, R)

    if U is not None and V is not None and W is not None:
        results["secondary_flow"] = secondary_flow_strength(U, V, W)

    return results


def print_diagnostics_report(results: Dict[str, DiagnosticResult]) -> None:
    """Print formatted summary of all diagnostics."""
    print(f"\n{'='*60}")
    print(f"  Physics Diagnostics Report")
    print(f"{'='*60}")
    for name, result in results.items():
        print(f"\n  [{result.name}]")
        print(f"    {result.summary}")
        for concern in result.regions_of_concern:
            print(f"    ⚠  {concern}")
    print(f"\n{'='*60}")
