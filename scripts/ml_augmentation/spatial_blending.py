#!/usr/bin/env python3
"""
Multi-Agent Spatial Blending for Data-Driven Turbulence Corrections
====================================================================
DLR-inspired architecture that partitions the ML correction into
specialized agent models for distinct flow regimes, with non-dimensional
sensor functions governing activation and hyperbolic tangent blending
ensuring numerical stability.

Architecture
------------
Instead of a single global TBNN applied uniformly (which degrades in
attached BL regions), we deploy:

  Agent 1: **Separation Agent** — predicts b_ij correction in deep
           separation zones (recirculation, detached shear layers)
  Agent 2: **Reattachment Agent** — captures non-equilibrium recovery
           physics downstream of reattachment
  Agent 3: **Baseline (RANS)** — standard Boussinesq closure for
           attached log-law regions (no ML correction)

Sensor Functions
----------------
Non-dimensional spatial sensors detect flow regimes dynamically:

  A_p^+ = (ν / ρ u_τ³) dP/ds    — Pressure gradient parameter
  Π_f   = u_τ / U_e              — Friction-to-edge velocity ratio
  y^+   = y u_τ / ν              — Non-dimensional wall distance

Blending Function
-----------------
Smooth tanh interpolation shields attached regions:

  f_blend = 0.5 * (tanh(ζ) + 1)

where ζ = f(y^+, A_p^+, Π_f) ensures:
  - f → 0 in attached log-law regions (pure RANS)
  - f → 1 in fully separated zones (full ML correction)
  - Smooth transition in between (no discontinuities)

References
----------
- Lav et al. (DLR, 2023) — Multi-agent blending for turbomachinery
- Weatheritt & Sandberg (2016) — EASM via gene-expression programming
- Pope (1975) — tensor basis integrity
"""

import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ml_augmentation.tbnn_closure import (
    compute_tensor_basis,
    compute_invariant_inputs,
    check_realizability,
    project_to_realizable,
    prepare_tbnn_data,
)
from scripts.ml_augmentation.feature_extraction import (
    compute_strain_rate,
    compute_rotation_rate,
)
from scripts.ml_augmentation.ml_validation_reporter import compute_metrics

logger = logging.getLogger(__name__)


# =============================================================================
# Non-Dimensional Sensor Functions
# =============================================================================

def compute_pressure_gradient_parameter(
    dp_ds: np.ndarray,
    nu: float,
    u_tau: np.ndarray,
    rho: float = 1.0,
) -> np.ndarray:
    """
    Compute the Clauser pressure gradient parameter A_p^+.

    A_p^+ = (ν / ρ u_τ³) dP/ds

    This non-dimensional sensor quantifies the severity of the pressure
    gradient relative to the wall friction velocity:
      - A_p^+ ≈ 0   → zero pressure gradient (flat plate)
      - A_p^+ > 0    → adverse pressure gradient (separation-prone)
      - A_p^+ >> 10  → separation or near-separation
      - A_p^+ < 0    → favorable pressure gradient

    Parameters
    ----------
    dp_ds : ndarray (N,)
        Streamwise pressure gradient dP/ds [Pa/m].
    nu : float
        Kinematic viscosity [m²/s].
    u_tau : ndarray (N,)
        Friction velocity [m/s].
    rho : float
        Density [kg/m³].

    Returns
    -------
    Ap_plus : ndarray (N,)
        Non-dimensional pressure gradient parameter.
    """
    u_tau_safe = np.maximum(np.abs(u_tau), 1e-10)
    Ap_plus = (nu / (rho * u_tau_safe**3)) * dp_ds
    return Ap_plus


def compute_friction_velocity_ratio(
    u_tau: np.ndarray,
    U_e: np.ndarray,
) -> np.ndarray:
    """
    Compute friction-to-edge velocity ratio Π_f = u_τ / U_e.

    This sensor indicates the local state of the boundary layer:
      - Π_f ≈ 0.04-0.05   → healthy attached BL
      - Π_f → 0            → approaching separation (Cf → 0)
      - Π_f < 0            → reversed flow (separated)

    Parameters
    ----------
    u_tau : ndarray (N,) — friction velocity
    U_e : ndarray (N,) — boundary layer edge velocity

    Returns
    -------
    Pi_f : ndarray (N,)
    """
    U_e_safe = np.maximum(np.abs(U_e), 1e-10)
    return u_tau / U_e_safe


def compute_y_plus(
    y: np.ndarray,
    u_tau: np.ndarray,
    nu: float,
) -> np.ndarray:
    """
    Compute non-dimensional wall distance y^+ = y u_τ / ν.

    Parameters
    ----------
    y : ndarray (N,) — wall distance [m]
    u_tau : ndarray (N,) — friction velocity [m/s]
    nu : float — kinematic viscosity [m²/s]

    Returns
    -------
    y_plus : ndarray (N,)
    """
    return np.abs(y) * np.abs(u_tau) / max(nu, 1e-15)


# =============================================================================
# Flow Regime Classification
# =============================================================================

@dataclass
class FlowRegimeMap:
    """Spatial classification of flow into distinct regimes."""
    regime: np.ndarray  # (N,) int: 0=attached, 1=separation, 2=reattachment
    Ap_plus: np.ndarray  # (N,) pressure gradient parameter
    Pi_f: np.ndarray  # (N,) friction velocity ratio
    y_plus: np.ndarray  # (N,) non-dimensional wall distance
    n_attached: int = 0
    n_separation: int = 0
    n_reattachment: int = 0

    @property
    def summary(self) -> str:
        total = len(self.regime)
        return (
            f"Flow Regime Map: {total} points — "
            f"Attached={self.n_attached} ({self.n_attached/total*100:.1f}%), "
            f"Separation={self.n_separation} ({self.n_separation/total*100:.1f}%), "
            f"Reattachment={self.n_reattachment} ({self.n_reattachment/total*100:.1f}%)"
        )


def classify_flow_regime(
    dp_ds: np.ndarray,
    u_tau: np.ndarray,
    U_e: np.ndarray,
    y: np.ndarray,
    nu: float,
    rho: float = 1.0,
    Ap_threshold: float = 5.0,
    Pi_f_threshold: float = 0.02,
) -> FlowRegimeMap:
    """
    Classify each point into a flow regime using sensor functions.

    Classification logic:
      1. A_p^+ > threshold AND Π_f < threshold → SEPARATION (regime 1)
      2. A_p^+ is decreasing (recovering) AND Π_f > threshold → REATTACHMENT (regime 2)
      3. Otherwise → ATTACHED (regime 0, pure RANS)

    Parameters
    ----------
    dp_ds : (N,) streamwise pressure gradient
    u_tau : (N,) friction velocity
    U_e : (N,) edge velocity
    y : (N,) wall distance
    nu : kinematic viscosity
    rho : density
    Ap_threshold : threshold for separation detection
    Pi_f_threshold : threshold for near-separation detection

    Returns
    -------
    FlowRegimeMap
    """
    N = len(dp_ds)

    Ap = compute_pressure_gradient_parameter(dp_ds, nu, u_tau, rho)
    Pi_f = compute_friction_velocity_ratio(u_tau, U_e)
    yp = compute_y_plus(y, u_tau, nu)

    regime = np.zeros(N, dtype=int)

    # Separation: strong APG + low friction velocity
    sep_mask = (Ap > Ap_threshold) & (Pi_f < Pi_f_threshold)
    regime[sep_mask] = 1

    # Reattachment: recovering APG + moderate friction
    reat_mask = (Ap > 0) & (Ap <= Ap_threshold) & (Pi_f >= Pi_f_threshold)
    regime[reat_mask] = 2

    return FlowRegimeMap(
        regime=regime,
        Ap_plus=Ap,
        Pi_f=Pi_f,
        y_plus=yp,
        n_attached=int(np.sum(regime == 0)),
        n_separation=int(np.sum(regime == 1)),
        n_reattachment=int(np.sum(regime == 2)),
    )


# =============================================================================
# Hyperbolic Tangent Blending Function
# =============================================================================

def tanh_blending_function(
    Ap_plus: np.ndarray,
    y_plus: np.ndarray,
    Pi_f: np.ndarray,
    Ap_center: float = 5.0,
    Ap_width: float = 2.0,
    yp_shield: float = 30.0,
    yp_width: float = 10.0,
) -> np.ndarray:
    """
    Compute smooth blending weight via hyperbolic tangent shielding.

    f_blend = 0.5 * (tanh(ζ) + 1)

    where ζ combines pressure gradient activation with wall-distance
    shielding to protect the attached log-law region:

      ζ = ζ_Ap * ζ_wall

      ζ_Ap   = (A_p^+ - Ap_center) / Ap_width      — APG activation
      ζ_wall = (y^+ - yp_shield) / yp_width          — wall shielding

    The result is:
      - f → 0  in attached log-law region (y^+ < 30, low A_p^+)
      - f → 1  in deep separation (high A_p^+, away from wall)
      - Smooth transition in between

    Parameters
    ----------
    Ap_plus : (N,) pressure gradient parameter
    y_plus : (N,) non-dimensional wall distance
    Pi_f : (N,) friction velocity ratio (unused in base, extensible)
    Ap_center : center of APG activation sigmoid
    Ap_width : width of APG transition
    yp_shield : wall-distance below which ML is shielded
    yp_width : width of wall-shielding transition

    Returns
    -------
    f_blend : (N,) blending weight in [0, 1]
    """
    # APG activation: ramps from 0 → 1 as A_p^+ exceeds threshold
    zeta_Ap = (Ap_plus - Ap_center) / max(Ap_width, 1e-10)
    f_Ap = 0.5 * (np.tanh(zeta_Ap) + 1)  # 0 when A_p+ << center, 1 when A_p+ >> center

    # Wall shielding: ramps from 0 → 1 as y^+ moves above log-law
    zeta_wall = (y_plus - yp_shield) / max(yp_width, 1e-10)
    f_wall = 0.5 * (np.tanh(zeta_wall) + 1)  # 0 near wall, 1 in outer layer

    # Final blending weight: both APG AND wall-distance must be active
    f_blend = f_Ap * f_wall

    return f_blend


def compute_separation_blending(
    Ap_plus: np.ndarray,
    y_plus: np.ndarray,
    **kwargs,
) -> np.ndarray:
    """
    Separation-agent blending: high in deep separation, zero in attached.
    """
    return tanh_blending_function(
        Ap_plus, y_plus, np.zeros_like(Ap_plus),
        Ap_center=5.0, Ap_width=2.0,
        yp_shield=30.0, yp_width=10.0,
        **kwargs,
    )


def compute_reattachment_blending(
    Ap_plus: np.ndarray,
    y_plus: np.ndarray,
    Pi_f: np.ndarray,
    **kwargs,
) -> np.ndarray:
    """
    Reattachment-agent blending: high in recovery zone, zero elsewhere.

    Active when A_p^+ is moderate (recovering) and Π_f > 0 (reattached).
    """
    # Recovery zone: A_p^+ in [1, 8] range
    zeta_recover = -((Ap_plus - 3.0) / 3.0)**2  # Bell-shaped
    zeta_wall = (y_plus - 20.0) / 10.0

    f_wall = 0.5 * (np.tanh(zeta_wall) + 1)
    f_recover = np.exp(zeta_recover)

    # Only active where friction is recovering (Pi_f > threshold)
    f_friction = 0.5 * (np.tanh((Pi_f - 0.02) / 0.01) + 1)

    return f_recover * f_wall * f_friction


# =============================================================================
# Specialized Agent Models
# =============================================================================

@dataclass
class AgentPrediction:
    """Prediction from a single specialized agent."""
    b_correction: np.ndarray  # (N, 3, 3) anisotropy correction
    confidence: np.ndarray  # (N,) agent confidence [0, 1]
    agent_name: str = ""


class SeparationAgent:
    """
    Neural network agent specialized for deep separation zones.

    Trained exclusively on DNS/LES data from recirculation regions where
    the Boussinesq approximation fails most severely. Focuses on capturing:
    - Normal stress anisotropy (b_11 >> b_22, b_33)
    - Stress-strain misalignment in detached shear layers
    - Backflow Reynolds stress asymmetry
    """

    def __init__(self, seed: int = 42):
        self.model = None
        self.inv_mean = None
        self.inv_std = None
        self.seed = seed
        self.is_trained = False

    def train(self, invariants: np.ndarray, targets: np.ndarray,
              regime_mask: np.ndarray):
        """Train on separation-zone data only."""
        from sklearn.neural_network import MLPRegressor

        # Filter to separation regime
        sep_idx = np.where(regime_mask == 1)[0]
        if len(sep_idx) < 10:
            logger.warning("Too few separation points for training, using all data")
            sep_idx = np.arange(len(invariants))

        X = invariants[sep_idx]
        y = targets[sep_idx].reshape(len(sep_idx), -1)

        self.inv_mean = X.mean(axis=0)
        self.inv_std = X.std(axis=0) + 1e-10
        X_n = (X - self.inv_mean) / self.inv_std

        self.model = MLPRegressor(
            hidden_layer_sizes=(64, 128, 128, 64),
            max_iter=300, random_state=self.seed,
            early_stopping=True, validation_fraction=0.15,
            alpha=1e-4, learning_rate='adaptive',
        )
        self.model.fit(X_n, y)
        self.is_trained = True
        logger.info(f"SeparationAgent trained on {len(sep_idx)} points")

    def predict(self, invariants: np.ndarray) -> AgentPrediction:
        """Predict anisotropy correction for given flow state."""
        if not self.is_trained:
            return AgentPrediction(
                b_correction=np.zeros((len(invariants), 3, 3)),
                confidence=np.zeros(len(invariants)),
                agent_name="separation",
            )

        X_n = (invariants - self.inv_mean) / self.inv_std
        b_flat = self.model.predict(X_n)
        b_pred = b_flat.reshape(-1, 3, 3)
        b_pred = project_to_realizable(b_pred)

        return AgentPrediction(
            b_correction=b_pred,
            confidence=np.ones(len(invariants)) * 0.9,
            agent_name="separation",
        )


class ReattachmentAgent:
    """
    Neural network agent specialized for reattachment/recovery zones.

    Trained on the transition from separated flow back to attached BL.
    Captures non-equilibrium turbulence dynamics during recovery where
    the eddy viscosity rebuilds from near-zero.
    """

    def __init__(self, seed: int = 123):
        self.model = None
        self.inv_mean = None
        self.inv_std = None
        self.seed = seed
        self.is_trained = False

    def train(self, invariants: np.ndarray, targets: np.ndarray,
              regime_mask: np.ndarray):
        """Train on reattachment/recovery-zone data only."""
        from sklearn.neural_network import MLPRegressor

        reat_idx = np.where(regime_mask >= 2)[0]
        if len(reat_idx) < 10:
            reat_idx = np.where(regime_mask >= 1)[0]
        if len(reat_idx) < 10:
            reat_idx = np.arange(len(invariants))

        X = invariants[reat_idx]
        y = targets[reat_idx].reshape(len(reat_idx), -1)

        self.inv_mean = X.mean(axis=0)
        self.inv_std = X.std(axis=0) + 1e-10
        X_n = (X - self.inv_mean) / self.inv_std

        self.model = MLPRegressor(
            hidden_layer_sizes=(64, 128, 64),
            max_iter=300, random_state=self.seed,
            early_stopping=True, validation_fraction=0.15,
            alpha=1e-4, learning_rate='adaptive',
        )
        self.model.fit(X_n, y)
        self.is_trained = True
        logger.info(f"ReattachmentAgent trained on {len(reat_idx)} points")

    def predict(self, invariants: np.ndarray) -> AgentPrediction:
        if not self.is_trained:
            return AgentPrediction(
                b_correction=np.zeros((len(invariants), 3, 3)),
                confidence=np.zeros(len(invariants)),
                agent_name="reattachment",
            )

        X_n = (invariants - self.inv_mean) / self.inv_std
        b_flat = self.model.predict(X_n)
        b_pred = b_flat.reshape(-1, 3, 3)
        b_pred = project_to_realizable(b_pred)

        return AgentPrediction(
            b_correction=b_pred,
            confidence=np.ones(len(invariants)) * 0.8,
            agent_name="reattachment",
        )


# =============================================================================
# Multi-Agent Blended Prediction
# =============================================================================

@dataclass
class BlendedPredictionResult:
    """Result from multi-agent spatial blending."""
    b_blended: np.ndarray  # (N, 3, 3) final blended anisotropy
    b_rans: np.ndarray  # (N, 3, 3) RANS baseline
    b_global: np.ndarray  # (N, 3, 3) global (unblended) TBNN
    f_separation: np.ndarray  # (N,) separation blending weight
    f_reattachment: np.ndarray  # (N,) reattachment blending weight
    regime_map: FlowRegimeMap = None
    # Metrics
    blended_R2: float = 0.0
    global_R2: float = 0.0
    rans_R2: float = 0.0
    blended_RMSE: float = 0.0
    global_RMSE: float = 0.0
    rans_RMSE: float = 0.0
    # Per-regime metrics
    sep_R2_blended: float = 0.0
    sep_R2_global: float = 0.0
    reat_R2_blended: float = 0.0
    reat_R2_global: float = 0.0
    att_R2_blended: float = 0.0
    att_R2_global: float = 0.0
    realizability_fraction: float = 0.0
    summary: str = ""


def blend_agent_predictions(
    b_rans: np.ndarray,
    sep_pred: AgentPrediction,
    reat_pred: AgentPrediction,
    f_sep: np.ndarray,
    f_reat: np.ndarray,
) -> np.ndarray:
    """
    Blend RANS baseline with specialized agent corrections.

    b_final = (1 - f_total) * b_RANS
              + f_sep * b_separation
              + f_reat * b_reattachment

    where f_total = min(f_sep + f_reat, 1) ensures the total correction
    weight never exceeds unity.

    Parameters
    ----------
    b_rans : (N, 3, 3) RANS Boussinesq anisotropy
    sep_pred : SeparationAgent prediction
    reat_pred : ReattachmentAgent prediction
    f_sep : (N,) separation blending weight
    f_reat : (N,) reattachment blending weight

    Returns
    -------
    b_blended : (N, 3, 3)
    """
    N = len(f_sep)

    f_total = np.minimum(f_sep + f_reat, 1.0)

    # Reshape weights for broadcasting
    w_rans = (1 - f_total)[:, None, None]
    w_sep = f_sep[:, None, None]
    w_reat = f_reat[:, None, None]

    b_blended = (
        w_rans * b_rans
        + w_sep * sep_pred.b_correction
        + w_reat * reat_pred.b_correction
    )

    return b_blended


# =============================================================================
# Full Multi-Agent Pipeline
# =============================================================================

class MultiAgentBlendingPipeline:
    """
    Complete multi-agent spatial blending pipeline.

    Workflow:
    1. Generate/load flow data with DNS targets
    2. Compute sensor functions (A_p^+, Π_f, y^+)
    3. Classify flow regimes
    4. Train specialized agents on regime-specific data
    5. Train global TBNN on all data (baseline comparison)
    6. Compute blending weights
    7. Blend agent predictions with RANS baseline
    8. Evaluate: blended vs global vs RANS, per-regime metrics
    """

    def __init__(self, seed: int = 42, epochs: int = 200):
        self.seed = seed
        self.epochs = epochs
        self.sep_agent = SeparationAgent(seed=seed)
        self.reat_agent = ReattachmentAgent(seed=seed + 1)

    def run(self, flow_data: Dict[str, np.ndarray]) -> BlendedPredictionResult:
        """
        Execute the full multi-agent blending pipeline.

        Parameters
        ----------
        flow_data : dict
            Output from generate_periodic_hill_dns() or generate_bfs_dns().
            Must contain: S, Omega, k, epsilon, b_dns, b_rans, region, x, y, U.

        Returns
        -------
        BlendedPredictionResult
        """
        N = len(flow_data["k"])
        result = BlendedPredictionResult(
            b_blended=np.zeros((N, 3, 3)),
            b_rans=flow_data["b_rans"],
            b_global=np.zeros((N, 3, 3)),
            f_separation=np.zeros(N),
            f_reattachment=np.zeros(N),
        )

        # --- 1. Compute sensor inputs ---
        nu = flow_data.get("H", 1.0) / flow_data.get("Re", 10000)
        k = flow_data["k"]
        epsilon = flow_data["epsilon"]
        y = flow_data["y"]

        # Approximate u_tau from Cf: u_tau = sqrt(Cf/2) * U_bulk
        U_bulk = np.max(np.abs(flow_data["U"]))
        # Near-wall TKE → approximate u_tau
        u_tau = np.sqrt(np.maximum(k * 0.3, 1e-10))
        # Edge velocity ≈ max U at each x station
        U_e = np.maximum(np.abs(flow_data["U"]), 0.1)

        # Streamwise pressure gradient from velocity field (dU/dx proxy)
        dudx = flow_data["dudx"]
        dp_ds = -flow_data["U"] * dudx[:, 0, 0]  # ≈ -ρ U dU/dx (Bernoulli)

        # --- 2. Compute sensor functions ---
        Ap_plus = compute_pressure_gradient_parameter(dp_ds, nu, u_tau)
        Pi_f = compute_friction_velocity_ratio(u_tau, U_e)
        y_plus = compute_y_plus(y, u_tau, nu)

        # --- 3. Classify regimes ---
        regime_map = classify_flow_regime(dp_ds, u_tau, U_e, y, nu)
        result.regime_map = regime_map
        logger.info(regime_map.summary)

        # --- 4. Prepare TBNN data ---
        tbnn_data = prepare_tbnn_data(
            flow_data["S"], flow_data["Omega"],
            k, epsilon, flow_data["b_dns"],
        )
        invariants = tbnn_data["invariants"]
        targets = tbnn_data["targets"]

        # --- 5. Train specialized agents ---
        logger.info("Training SeparationAgent...")
        self.sep_agent.train(invariants, targets, regime_map.regime)

        logger.info("Training ReattachmentAgent...")
        self.reat_agent.train(invariants, targets, regime_map.regime)

        # --- 6. Train global TBNN (baseline) ---
        logger.info("Training global TBNN baseline...")
        global_pred = self._train_global(invariants, targets)
        result.b_global = global_pred

        # --- 7. Compute blending weights ---
        result.f_separation = compute_separation_blending(Ap_plus, y_plus)
        result.f_reattachment = compute_reattachment_blending(
            Ap_plus, y_plus, Pi_f,
        )

        # --- 8. Blend predictions ---
        sep_pred = self.sep_agent.predict(invariants)
        reat_pred = self.reat_agent.predict(invariants)

        result.b_blended = blend_agent_predictions(
            flow_data["b_rans"],
            sep_pred, reat_pred,
            result.f_separation, result.f_reattachment,
        )

        # Enforce realizability with eigenvalue clamping
        b_blended = result.b_blended
        eig_vals = np.linalg.eigvalsh(b_blended)
        max_eig = np.abs(eig_vals).max(axis=1)
        scale = np.maximum(max_eig / (1.0/3.0), 1.0)
        b_blended = b_blended / scale[:, None, None]
        result.b_blended = project_to_realizable(b_blended)
        r_report = check_realizability(result.b_blended)
        result.realizability_fraction = r_report.fraction_realizable

        # --- 9. Evaluate metrics ---
        b_dns_flat = targets.reshape(-1)
        b_blend_flat = result.b_blended.reshape(-1)
        b_global_flat = global_pred.reshape(-1)
        b_rans_flat = flow_data["b_rans"].reshape(-1)

        m_blend = compute_metrics(b_dns_flat, b_blend_flat, "blended")
        m_global = compute_metrics(b_dns_flat, b_global_flat, "global")
        m_rans = compute_metrics(b_dns_flat, b_rans_flat, "rans")

        result.blended_R2 = m_blend.R2
        result.global_R2 = m_global.R2
        result.rans_R2 = m_rans.R2
        result.blended_RMSE = m_blend.RMSE
        result.global_RMSE = m_global.RMSE
        result.rans_RMSE = m_rans.RMSE

        # Per-regime metrics
        for regime_id, regime_name, r2_attr, r2_global_attr in [
            (1, "sep", "sep_R2_blended", "sep_R2_global"),
            (2, "reat", "reat_R2_blended", "reat_R2_global"),
            (0, "att", "att_R2_blended", "att_R2_global"),
        ]:
            mask = regime_map.regime == regime_id
            if np.sum(mask) > 10:
                t_flat = targets[mask].reshape(-1)
                b_flat = result.b_blended[mask].reshape(-1)
                g_flat = global_pred[mask].reshape(-1)
                setattr(result, r2_attr, compute_metrics(t_flat, b_flat, regime_name).R2)
                setattr(result, r2_global_attr, compute_metrics(t_flat, g_flat, regime_name).R2)

        # Summary
        result.summary = (
            f"Multi-Agent Spatial Blending Results\n"
            f"{'=' * 55}\n"
            f"Regime Map: {regime_map.summary}\n\n"
            f"{'Approach':<25} {'R²':>8} {'RMSE':>10}\n"
            f"{'-'*45}\n"
            f"{'RANS (Boussinesq)':<25} {result.rans_R2:>8.4f} {result.rans_RMSE:>10.6f}\n"
            f"{'Global TBNN':<25} {result.global_R2:>8.4f} {result.global_RMSE:>10.6f}\n"
            f"{'Multi-Agent Blended':<25} {result.blended_R2:>8.4f} {result.blended_RMSE:>10.6f}\n\n"
            f"Per-Regime R² (Blended / Global):\n"
            f"  Separation:    {result.sep_R2_blended:.4f} / {result.sep_R2_global:.4f}\n"
            f"  Reattachment:  {result.reat_R2_blended:.4f} / {result.reat_R2_global:.4f}\n"
            f"  Attached:      {result.att_R2_blended:.4f} / {result.att_R2_global:.4f}\n\n"
            f"Realizability: {result.realizability_fraction*100:.1f}%\n"
            f"Blending: f_sep max={np.max(result.f_separation):.3f}, "
            f"f_reat max={np.max(result.f_reattachment):.3f}\n"
        )
        logger.info(result.summary)

        return result

    def _train_global(self, invariants, targets):
        """Train a global TBNN on all data (for comparison)."""
        from sklearn.neural_network import MLPRegressor

        N = len(invariants)
        y_flat = targets.reshape(N, -1)

        inv_mean = invariants.mean(axis=0)
        inv_std = invariants.std(axis=0) + 1e-10
        X_n = (invariants - inv_mean) / inv_std

        mlp = MLPRegressor(
            hidden_layer_sizes=(64, 128, 128, 64),
            max_iter=self.epochs,
            random_state=self.seed,
            early_stopping=True, validation_fraction=0.15,
            alpha=1e-4,
        )
        mlp.fit(X_n, y_flat)

        b_pred = mlp.predict(X_n).reshape(-1, 3, 3)
        return project_to_realizable(b_pred)


# =============================================================================
# Convenience Runner
# =============================================================================

def run_blending_comparison(case: str = "periodic_hill") -> BlendedPredictionResult:
    """
    Run multi-agent blending on a DNS case and compare against global TBNN.

    Parameters
    ----------
    case : str
        "periodic_hill" or "bfs"

    Returns
    -------
    BlendedPredictionResult with full comparison metrics
    """
    from scripts.ml_augmentation.tbnn_dns_pipeline import (
        generate_periodic_hill_dns, generate_bfs_dns,
    )

    if case == "periodic_hill":
        data = generate_periodic_hill_dns()
    elif case == "bfs":
        data = generate_bfs_dns()
    else:
        raise ValueError(f"Unknown case: {case}")

    pipeline = MultiAgentBlendingPipeline(seed=42, epochs=150)
    return pipeline.run(data)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    for case in ["periodic_hill", "bfs"]:
        print(f"\n{'='*60}")
        print(f"  {case.upper()}")
        print(f"{'='*60}")
        result = run_blending_comparison(case)
        print(result.summary)
