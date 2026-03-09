"""
Profile Extraction and Physics Diagnostics
==========================================
Extract velocity/pressure profiles from CFD results and compute
physics diagnostics for turbulence model assessment.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# =============================================================================
# Profile Extraction
# =============================================================================
def extract_wall_data(
    x: np.ndarray,
    tau_w: np.ndarray,
    p: np.ndarray,
    rho: float,
    U_ref: float,
) -> pd.DataFrame:
    """
    Compute wall quantities from raw field data.

    Parameters
    ----------
    x : array
        Streamwise coordinates at wall.
    tau_w : array
        Wall shear stress [Pa].
    p : array
        Wall pressure [Pa].
    rho : float
        Reference density [kg/m³].
    U_ref : float
        Reference velocity [m/s].

    Returns
    -------
    DataFrame with x, Cf, Cp, u_tau columns.
    """
    q = 0.5 * rho * U_ref ** 2
    Cf = tau_w / q
    Cp = (p - np.mean(p[:5])) / q  # Reference to inlet pressure
    u_tau = np.sqrt(np.abs(tau_w) / rho) * np.sign(tau_w)

    return pd.DataFrame({"x": x, "Cf": Cf, "Cp": Cp, "u_tau": u_tau})


def extract_velocity_profile(
    y: np.ndarray,
    U: np.ndarray,
    V: Optional[np.ndarray] = None,
    uu: Optional[np.ndarray] = None,
    vv: Optional[np.ndarray] = None,
    uv: Optional[np.ndarray] = None,
    u_tau: float = 1.0,
    nu: float = 1.5e-5,
) -> pd.DataFrame:
    """
    Build a profile DataFrame with inner and outer scaling.

    Parameters
    ----------
    y : array
        Wall-normal coordinate [m].
    U : array
        Streamwise velocity [m/s].
    V, uu, vv, uv : array, optional
        Additional quantities.
    u_tau : float
        Friction velocity [m/s].
    nu : float
        Kinematic viscosity [m²/s].

    Returns
    -------
    DataFrame with y, U, y_plus, U_plus, and optional columns.
    """
    y_plus = y * u_tau / nu
    U_plus = U / u_tau

    data = {"y": y, "U": U, "y_plus": y_plus, "U_plus": U_plus}
    if V is not None:
        data["V"] = V
    if uu is not None:
        data["uu"] = uu
        data["uu_plus"] = uu / u_tau**2
    if vv is not None:
        data["vv"] = vv
    if uv is not None:
        data["uv"] = uv
        data["uv_plus"] = uv / u_tau**2

    return pd.DataFrame(data)


def find_separation_point(x: np.ndarray, Cf: np.ndarray) -> Optional[float]:
    """Find separation point where Cf crosses zero (positive -> negative)."""
    # Vectorized sign-change detection
    sign_change = np.diff(np.sign(Cf))
    # Positive-to-negative crossings have sign_change < 0
    crossings = np.where(sign_change < 0)[0]
    if len(crossings) == 0:
        return None
    i = crossings[0]
    # Linear interpolation for exact crossing location
    return float(x[i] - Cf[i] * (x[i + 1] - x[i]) / (Cf[i + 1] - Cf[i]))


def find_reattachment_point(x: np.ndarray, Cf: np.ndarray) -> Optional[float]:
    """Find reattachment point where Cf crosses zero (negative -> positive).

    Reattachment is defined as the first neg→pos crossing that occurs
    *after* a separation (pos→neg crossing), so that boundary artefacts
    where Cf starts at exactly zero are not mis-identified.
    """
    sign_change = np.diff(np.sign(Cf))

    # Find first separation (pos -> neg)
    sep_crossings = np.where(sign_change < 0)[0]
    if len(sep_crossings) == 0:
        return None            # no separation → no reattachment
    sep_idx = sep_crossings[0]

    # Find neg -> pos crossings that come *after* the separation
    reat_crossings = np.where(sign_change > 0)[0]
    reat_crossings = reat_crossings[reat_crossings > sep_idx]
    if len(reat_crossings) == 0:
        return None
    i = reat_crossings[0]
    # Linear interpolation for exact crossing location
    return float(x[i] - Cf[i] * (x[i + 1] - x[i]) / (Cf[i + 1] - Cf[i]))


# =============================================================================
# Physics Diagnostics
# =============================================================================
def boussinesq_validity(
    Sij: np.ndarray, aij: np.ndarray
) -> float:
    """
    Check Boussinesq hypothesis validity.

    Parameters
    ----------
    Sij : array, shape (..., 3, 3)
        Mean strain rate tensor.
    aij : array, shape (..., 3, 3)
        Reynolds stress anisotropy tensor.

    Returns
    -------
    float
        Alignment factor (1 = perfect Boussinesq, 0 = complete failure).
    """
    num = np.sum(aij * Sij, axis=(-2, -1))
    denom = np.sqrt(np.sum(aij ** 2, axis=(-2, -1)) * np.sum(Sij ** 2, axis=(-2, -1)))
    denom = np.maximum(denom, 1e-15)
    alignment = -num / denom  # Negative because a_ij ∝ -S_ij
    return float(np.mean(alignment))


def lumley_triangle(
    uu: np.ndarray, vv: np.ndarray, ww: np.ndarray,
    uv: np.ndarray = None, uw: np.ndarray = None, vw: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Lumley triangle invariants (ξ, η).

    Parameters
    ----------
    uu, vv, ww : array
        Normal Reynolds stresses.
    uv, uw, vw : array, optional
        Shear Reynolds stresses (default 0).

    Returns
    -------
    xi, eta : arrays
        Lumley triangle coordinates.
    """
    k = 0.5 * (uu + vv + ww)
    k = np.maximum(k, 1e-15)

    # Anisotropy tensor
    a11 = uu / (2 * k) - 1 / 3
    a22 = vv / (2 * k) - 1 / 3
    a33 = ww / (2 * k) - 1 / 3
    a12 = (uv if uv is not None else 0) / (2 * k)
    a13 = (uw if uw is not None else 0) / (2 * k)
    a23 = (vw if vw is not None else 0) / (2 * k)

    # Second invariant: II = a_ij * a_ji
    II = (a11**2 + a22**2 + a33**2
          + 2 * a12**2 + 2 * a13**2 + 2 * a23**2)

    # Third invariant: III = a_ij * a_jk * a_ki
    III = (a11**3 + a22**3 + a33**3
           + 3 * a11 * a12**2 + 3 * a22 * a12**2
           + 3 * a11 * a13**2 + 3 * a33 * a13**2
           + 3 * a22 * a23**2 + 3 * a33 * a23**2
           + 6 * a12 * a13 * a23)

    eta = np.sqrt(np.maximum(II / 6, 0))
    xi = np.cbrt(np.clip(III / 6, -1e10, 1e10))

    return xi, eta


def production_dissipation_ratio(
    P_k: np.ndarray, epsilon: np.ndarray
) -> np.ndarray:
    """
    Compute P/ε ratio.

    In equilibrium: P/ε ≈ 1.
    In separation: P/ε can be >> 1 or << 1.
    """
    return P_k / np.maximum(epsilon, 1e-15)


def curvature_richardson(
    U: np.ndarray, y: np.ndarray, kappa: float
) -> np.ndarray:
    """
    Compute curvature Richardson number Ri = (U/R) * (dU/dy) / (dU/dy)².

    Parameters
    ----------
    U : array
        Velocity profile.
    y : array
        Wall-normal coordinate.
    kappa : float
        Streamline curvature (1/R).

    Returns
    -------
    array
        Richardson number profile.
    """
    dUdy = np.gradient(U, y)
    dUdy = np.maximum(np.abs(dUdy), 1e-15)
    Ri = 2 * (U * kappa / dUdy) * (1 + U * kappa / dUdy)
    return Ri


def classify_separation_topology(
    Cf_x: np.ndarray, Cf_z: Optional[np.ndarray] = None
) -> str:
    """
    Classify separation topology from wall shear stress patterns.

    Returns one of:
      - "2D_SEPARATION" : Clean 2D Cf_x zero crossing
      - "3D_CLOSED" : 3D closed separation bubble
      - "3D_OPEN" : Open 3D separation
      - "NO_SEPARATION" : No Cf_x < 0
    """
    has_reversal = np.any(Cf_x < 0)
    if not has_reversal:
        return "NO_SEPARATION"

    if Cf_z is None:
        return "2D_SEPARATION"

    # Check for spanwise variation
    Cf_z_var = np.std(Cf_z) / (np.mean(np.abs(Cf_z)) + 1e-15)
    if Cf_z_var < 0.1:
        return "2D_SEPARATION"

    # Check if reversal region is closed
    reversal_mask = Cf_x < 0
    n_contiguous = _max_contiguous(reversal_mask)
    if n_contiguous / len(reversal_mask) < 0.3:
        return "3D_CLOSED"
    else:
        return "3D_OPEN"


def _max_contiguous(mask: np.ndarray) -> int:
    """Find maximum contiguous True region using vectorized run-length encoding."""
    if len(mask) == 0 or not np.any(mask):
        return 0
    # Pad with False at boundaries for clean diff
    padded = np.concatenate([[False], mask.astype(bool), [False]])
    diffs = np.diff(padded.astype(int))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    if len(starts) == 0:
        return 0
    return int(np.max(ends - starts))
