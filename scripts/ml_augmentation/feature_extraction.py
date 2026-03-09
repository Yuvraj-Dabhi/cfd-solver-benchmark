"""
Tensor Invariant Feature Extraction
====================================
Extracts physically-motivated features from RANS flow fields
for ML-augmented turbulence modeling.

Features include:
- Strain rate invariants (I1_S, I2_S)
- Vorticity invariants (I1_Ω)
- Pressure gradient indicators
- Wall distance features
- Turbulence field features (k, ε, ν_t)
- Galilean-invariant tensor basis (Pope 1975)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class FlowFeatures:
    """Container for extracted flow features."""
    names: List[str] = field(default_factory=list)
    values: np.ndarray = field(default_factory=lambda: np.array([]))
    n_points: int = 0
    n_features: int = 0

    def to_dict(self) -> Dict[str, np.ndarray]:
        return {name: self.values[:, i] for i, name in enumerate(self.names)}


def compute_strain_rate(dudx: np.ndarray) -> np.ndarray:
    """
    Compute symmetric strain rate tensor S_ij = 0.5(∂u_i/∂x_j + ∂u_j/∂x_i).

    Parameters
    ----------
    dudx : ndarray (N, 3, 3)
        Velocity gradient tensor.

    Returns
    -------
    ndarray (N, 3, 3) — strain rate tensor.
    """
    return 0.5 * (dudx + np.swapaxes(dudx, -2, -1))


def compute_rotation_rate(dudx: np.ndarray) -> np.ndarray:
    """
    Compute antisymmetric rotation rate tensor Ω_ij = 0.5(∂u_i/∂x_j - ∂u_j/∂x_i).
    """
    return 0.5 * (dudx - np.swapaxes(dudx, -2, -1))


def strain_invariants(S: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute strain rate invariants I1_S and I2_S.

    I1_S = tr(S) = S_kk (should be ~0 for incompressible)
    I2_S = -0.5 * (tr(S)² - tr(S²)) = S_ij·S_ij
    """
    N = S.shape[0]
    I1 = np.trace(S, axis1=-2, axis2=-1)

    # I2 = S_ij * S_ij (double contraction)
    I2 = np.einsum("nij,nij->n", S, S)

    return I1, I2


def vorticity_invariants(Omega: np.ndarray) -> np.ndarray:
    """
    Compute vorticity invariant I1_Ω = Ω_ij·Ω_ij.
    """
    return np.einsum("nij,nij->n", Omega, Omega)


def q_criterion(S: np.ndarray, Omega: np.ndarray) -> np.ndarray:
    """
    Compute Q-criterion: Q = 0.5(|Ω|² - |S|²).
    Q > 0 → rotation-dominated (vortex core).
    """
    S_norm2 = np.einsum("nij,nij->n", S, S)
    O_norm2 = np.einsum("nij,nij->n", Omega, Omega)
    return 0.5 * (O_norm2 - S_norm2)


def extract_invariant_features(
    dudx: np.ndarray,
    k: np.ndarray,
    epsilon: np.ndarray,
    wall_dist: np.ndarray,
    p_gradient: np.ndarray = None,
    nu: float = 1.5e-5,
) -> FlowFeatures:
    """
    Extract full Galilean-invariant feature set for ML turbulence modeling.

    Parameters
    ----------
    dudx : ndarray (N, 3, 3)
        Velocity gradient tensor.
    k : ndarray (N,)
        Turbulent kinetic energy.
    epsilon : ndarray (N,)
        Turbulent dissipation rate.
    wall_dist : ndarray (N,)
        Wall distance.
    p_gradient : ndarray (N, 3), optional
        Pressure gradient vector.
    nu : float
        Kinematic viscosity.

    Returns
    -------
    FlowFeatures with invariant features.
    """
    N = len(k)
    S = compute_strain_rate(dudx)
    Omega = compute_rotation_rate(dudx)

    # Safe denominators
    eps_safe = np.maximum(epsilon, 1e-10)
    k_safe = np.maximum(k, 1e-10)

    # Turbulent time scale
    tau = k_safe / eps_safe

    # Non-dimensionalized tensors
    S_hat = S * tau[:, None, None]
    O_hat = Omega * tau[:, None, None]

    # ---- Core invariants ----
    I1_S, I2_S = strain_invariants(S)
    I1_Omega = vorticity_invariants(Omega)

    # Non-dimensional strain & rotation magnitude
    S_norm = np.sqrt(np.einsum("nij,nij->n", S_hat, S_hat))
    O_norm = np.sqrt(np.einsum("nij,nij->n", O_hat, O_hat))

    # ---- Tensor basis invariants (Pope 1975, Ling et al. 2016) ----
    # λ1 = S²_kk
    S2 = np.einsum("nij,njk->nik", S_hat, S_hat)
    lambda1 = np.trace(S2, axis1=-2, axis2=-1)

    # λ2 = Ω²_kk
    O2 = np.einsum("nij,njk->nik", O_hat, O_hat)
    lambda2 = np.trace(O2, axis1=-2, axis2=-1)

    # λ3 = S³_kk
    S3 = np.einsum("nij,njk->nik", S2, S_hat)
    lambda3 = np.trace(S3, axis1=-2, axis2=-1)

    # λ4 = (Ω²·S)_kk
    O2S = np.einsum("nij,njk->nik", O2, S_hat)
    lambda4 = np.trace(O2S, axis1=-2, axis2=-1)

    # λ5 = (Ω²·S²)_kk
    O2S2 = np.einsum("nij,njk->nik", O2, S2)
    lambda5 = np.trace(O2S2, axis1=-2, axis2=-1)

    # ---- Wall-distance features ----
    Re_d = np.sqrt(k_safe) * wall_dist / nu  # Wall-distance Reynolds number
    tau_wall = k_safe / (eps_safe * wall_dist ** 2 + 1e-15)  # Non-dim wall distance

    # ---- Q-criterion ----
    Q = q_criterion(S, Omega)

    # ---- Turbulence ratio ----
    nut_over_nu = k_safe ** 2 / (eps_safe * nu + 1e-15)  # ≈ ν_t/ν

    # ---- Assemble feature vector ----
    feature_names = [
        "S_norm", "O_norm", "Q_criterion",
        "lambda1_S2", "lambda2_O2", "lambda3_S3",
        "lambda4_O2S", "lambda5_O2S2",
        "Re_d", "tau_wall", "nut_over_nu",
        "k", "epsilon", "k_over_eps",
    ]

    features = np.column_stack([
        S_norm, O_norm, Q,
        lambda1, lambda2, lambda3, lambda4, lambda5,
        Re_d, tau_wall, nut_over_nu,
        k, epsilon, tau,
    ])

    # ---- Optional pressure gradient features ----
    if p_gradient is not None:
        dp_mag = np.linalg.norm(p_gradient, axis=-1)
        # Non-dimensional APG indicator
        apg_indicator = dp_mag * k_safe / (eps_safe ** 2 + 1e-15)
        features = np.column_stack([features, dp_mag, apg_indicator])
        feature_names.extend(["dp_mag", "APG_indicator"])

    return FlowFeatures(
        names=feature_names,
        values=features,
        n_points=N,
        n_features=len(feature_names),
    )


def normalize_features(
    features: FlowFeatures,
    method: str = "standard",
) -> Tuple[FlowFeatures, Dict]:
    """
    Normalize features for ML training.

    Parameters
    ----------
    method : str
        'standard' (zero mean, unit var), 'minmax' (0-1), or 'robust' (median/IQR).

    Returns
    -------
    (normalized_features, scaler_params)
    """
    X = features.values.copy()
    params = {}

    if method == "standard":
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0) + 1e-10
        X = (X - mean) / std
        params = {"mean": mean, "std": std}
    elif method == "minmax":
        xmin = np.min(X, axis=0)
        xmax = np.max(X, axis=0)
        X = (X - xmin) / (xmax - xmin + 1e-10)
        params = {"min": xmin, "max": xmax}
    elif method == "robust":
        median = np.median(X, axis=0)
        q75 = np.percentile(X, 75, axis=0)
        q25 = np.percentile(X, 25, axis=0)
        iqr = q75 - q25 + 1e-10
        X = (X - median) / iqr
        params = {"median": median, "iqr": iqr}

    normalized = FlowFeatures(
        names=features.names,
        values=X,
        n_points=features.n_points,
        n_features=features.n_features,
    )
    return normalized, params
