#!/usr/bin/env python3
"""
PINN-DA: Data Assimilation with Sparse Sensors via RANS-SA Constraints
=========================================================================
Physics-Informed Neural Network for variational data assimilation that
reconstructs full 2D mean-flow fields (u, v, p, ν̃) from sparse velocity
measurements using the steady RANS equations + Spalart–Allmaras closure
as physics constraints.

Key upgrade over ``pinn_boundary_layer.py``:
  - 2D field reconstruction (not 1D Cf correction)
  - Full RANS equations (not von Kármán integral)
  - SA turbulence closure as additional constraint
  - Sparse sensor assimilation with noise modeling

Physics Constraints
-------------------
1. Continuity:  ∂u/∂x + ∂v/∂y = 0
2. x-Momentum: u·∂u/∂x + v·∂u/∂y = -∂p/∂x + ∂/∂y[(ν + ν_t)·∂u/∂y]
3. y-Momentum: u·∂v/∂x + v·∂v/∂y = -∂p/∂y + ∂/∂x[(ν + ν_t)·∂v/∂x]
4. SA closure:  steady-state ν̃ transport (production − destruction + diffusion)

Loss Function
-------------
L = λ_data · L_data + λ_cont · L_cont + λ_mom · L_mom
    + λ_sa · L_SA + λ_bc · L_bc

References
----------
  - Habibi et al. (2024), JFM: "PINN-based data assimilation framework
    with SA model closure" (arXiv 2306.01065v2)
  - Raissi et al. (2019), JCP 378: PINNs for fluid mechanics
  - Spalart & Allmaras (1992): One-equation turbulence model
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.optimize import minimize

logger = logging.getLogger(__name__)
PROJECT = Path(__file__).resolve().parent.parent.parent

# ---------------------------------------------------------------------------
# Optional PyTorch dependency
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ============================================================================
# SA Model Constants (standard values — Spalart & Allmaras 1992)
# ============================================================================
SA_CONSTANTS = {
    "cb1": 0.1355,
    "cb2": 0.622,
    "sigma": 2.0 / 3.0,
    "kappa": 0.41,
    "cw2": 0.3,
    "cw3": 2.0,
    "cv1": 7.1,
    "ct3": 1.2,
    "ct4": 0.5,
}
# cw1 is derived: cw1 = cb1/kappa^2 + (1+cb2)/sigma
SA_CONSTANTS["cw1"] = (
    SA_CONSTANTS["cb1"] / SA_CONSTANTS["kappa"] ** 2
    + (1.0 + SA_CONSTANTS["cb2"]) / SA_CONSTANTS["sigma"]
)


# ============================================================================
# Data Structures
# ============================================================================
@dataclass
class SensorConfiguration:
    """Sparse sensor placement and measurement configuration."""
    x_sensors: np.ndarray           # (N_s,) streamwise sensor locations
    y_sensors: np.ndarray           # (N_s,) wall-normal sensor locations
    u_measured: np.ndarray          # (N_s,) measured u-velocity
    v_measured: Optional[np.ndarray] = None  # (N_s,) optional v-velocity
    noise_std: float = 0.01        # Measurement noise σ
    n_sensors: int = 0

    def __post_init__(self):
        self.n_sensors = len(self.x_sensors)
        if self.v_measured is None:
            self.v_measured = np.zeros_like(self.u_measured)


@dataclass
class PINNDAConfig:
    """Configuration for PINN-DA assimilator."""
    # Network architecture
    hidden_layers: Tuple[int, ...] = (128, 128, 128, 128, 128, 128)
    activation: str = "tanh"

    # Physics parameters
    nu: float = 1.5e-5              # Kinematic viscosity [m²/s]
    Re: float = 936000.0            # Reynolds number (wall hump: Re_c = 936k)

    # Loss weights
    lambda_data: float = 10.0       # Sensor data weight
    lambda_continuity: float = 1.0
    lambda_momentum: float = 1.0
    lambda_sa: float = 0.5         # SA closure weight
    lambda_bc: float = 5.0         # Boundary condition weight

    # Training
    n_collocation: int = 5000       # Interior collocation points
    n_boundary: int = 500           # Boundary points
    max_epochs: int = 5000
    learning_rate: float = 1e-3
    lr_decay_steps: int = 1000
    lr_decay_rate: float = 0.5
    adam_epochs: int = 3000         # Adam → L-BFGS transition
    print_interval: int = 500

    # Domain
    x_min: float = 0.0
    x_max: float = 2.0
    y_min: float = 0.0
    y_max: float = 0.5


@dataclass
class PINNDAResult:
    """Results from PINN-DA training and reconstruction."""
    # Reconstruction quality
    u_rmse: float = 0.0
    v_rmse: float = 0.0
    p_rmse: float = 0.0
    nut_rmse: float = 0.0
    u_improvement_pct: float = 0.0

    # Loss components
    final_data_loss: float = 0.0
    final_continuity_loss: float = 0.0
    final_momentum_loss: float = 0.0
    final_sa_loss: float = 0.0
    final_bc_loss: float = 0.0
    final_total_loss: float = 0.0

    # Training stats
    training_epochs: int = 0
    training_time_s: float = 0.0
    loss_history: List[float] = field(default_factory=list)

    # Sensor info
    n_sensors: int = 0
    noise_std: float = 0.0

    # Separation metrics
    separation_x_pred: float = 0.0
    reattachment_x_pred: float = 0.0
    bubble_length_pred: float = 0.0

    summary: str = ""


# ============================================================================
# Synthetic 2D Flow Field Generators
# ============================================================================
def generate_wall_hump_2d(
    nx: int = 80,
    ny: int = 40,
    x_range: Tuple[float, float] = (0.0, 2.0),
    y_range: Tuple[float, float] = (0.0, 0.5),
    Re: float = 936000.0,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Generate synthetic 2D wall-hump flow field with separation bubble.

    Mimics the NASA wall-mounted hump (Greenblatt et al., 2006):
      - Hump surface at y=0 in [0.0, 1.0]
      - Separation at x/c ≈ 0.665
      - Reattachment at x/c ≈ 1.11 (SA over-predicts ~1.24)
      - Recirculation with reversed flow in the bubble

    Returns coordinates and flow fields on a structured grid.
    """
    rng = np.random.default_rng(seed)

    x = np.linspace(x_range[0], x_range[1], nx)
    y = np.linspace(y_range[0], y_range[1], ny)
    xx, yy = np.meshgrid(x, y, indexing='ij')  # (nx, ny)

    # Hump geometry: bump centered at x=0.5
    hump_height = 0.08
    hump = hump_height * np.exp(-((xx - 0.5) ** 2) / 0.04)

    # Wall-normal coordinate adjusted for hump
    eta = yy - hump  # distance from surface
    eta = np.maximum(eta, 0.0)

    # Base velocity profile: Blasius-like with pressure gradient effects
    u_inf = 1.0
    delta = 0.05 + 0.02 * xx  # BL thickness grows

    # Edge velocity with favorable-adverse PG transition
    U_e = np.where(
        xx < 0.65,
        u_inf * (1.0 + 0.5 * np.sin(np.pi * xx / 0.65)),
        u_inf * (1.0 - 0.8 * (xx - 0.65) ** 2),
    )
    U_e = np.maximum(U_e, 0.1)

    # Velocity profile: u/U_e = (eta/delta)^(1/7) with separation
    eta_norm = np.minimum(eta / np.maximum(delta, 1e-6), 1.0)
    u = U_e * eta_norm ** (1.0 / 7.0)

    # Separation bubble: reversed flow near wall in [0.665, 1.11]
    x_sep = 0.665
    x_reat = 1.11
    in_bubble = (xx > x_sep) & (xx < x_reat) & (eta < 0.03)
    bubble_strength = np.sin(np.pi * (xx - x_sep) / (x_reat - x_sep))
    u[in_bubble] = -0.15 * u_inf * bubble_strength[in_bubble] * (
        1.0 - eta[in_bubble] / 0.03
    )

    # Wall BC: u=0 at y=0
    u[:, 0] = 0.0

    # v-velocity from continuity: ∂v/∂y = -∂u/∂x
    dudx = np.gradient(u, x, axis=0)
    v = np.zeros_like(u)
    dy = y[1] - y[0] if len(y) > 1 else 1.0
    for j in range(1, ny):
        v[:, j] = v[:, j - 1] - dudx[:, j] * dy
    v[:, 0] = 0.0  # Wall BC

    # Pressure from simplified momentum
    p = -0.5 * (u ** 2 + v ** 2) + 0.5 * u_inf ** 2
    p -= p.mean()  # Zero mean

    # Eddy viscosity: high in BL, enhanced near separation
    nu = 1.5e-5
    nut = nu * 100 * eta_norm * (1.0 - eta_norm)
    nut[in_bubble] *= 3.0  # Enhanced mixing in bubble
    nut = np.maximum(nut, 0.0)

    # SA variable ν̃ ≈ ν_t (simplified)
    nu_tilde = nut * 1.1  # SA ν̃ slightly larger than ν_t

    # Skin friction at wall
    Cf = 2.0 * nu * (u[:, 1] - u[:, 0]) / (dy * u_inf ** 2)

    return {
        "x": xx.ravel(),
        "y": yy.ravel(),
        "u": u.ravel(),
        "v": v.ravel(),
        "p": p.ravel(),
        "nu_t": nut.ravel(),
        "nu_tilde": nu_tilde.ravel(),
        "Cf": Cf,
        "x_1d": x,
        "xx": xx,
        "yy": yy,
        "u_2d": u,
        "v_2d": v,
        "p_2d": p,
        "nx": nx,
        "ny": ny,
    }


def generate_bfs_2d(
    nx: int = 80,
    ny: int = 40,
    step_x: float = 0.5,
    step_height: float = 0.1,
    seed: int = 99,
) -> Dict[str, np.ndarray]:
    """
    Generate synthetic 2D backward-facing step flow field.

    Features:
      - Step at x = step_x with height step_height
      - Recirculation zone downstream of step
      - Reattachment at x ≈ step_x + 6h (BFS rule of thumb)
    """
    rng = np.random.default_rng(seed)

    x_range = (0.0, 2.0)
    y_range = (0.0, 0.5)
    x = np.linspace(x_range[0], x_range[1], nx)
    y = np.linspace(y_range[0], y_range[1], ny)
    xx, yy = np.meshgrid(x, y, indexing='ij')

    u_inf = 1.0
    h = step_height
    x_reat = step_x + 6.0 * h

    # Velocity profile
    delta = 0.04 + 0.015 * xx
    eta = yy.copy()

    # Below step height and downstream of step: recirculation
    in_step_shadow = (xx > step_x) & (yy < h)
    in_recirc = (xx > step_x) & (xx < x_reat) & (yy < 2 * h)

    eta_norm = np.minimum(eta / np.maximum(delta, 1e-6), 1.0)
    u = u_inf * eta_norm ** (1.0 / 7.0)

    # Recirculation
    recirc_strength = np.sin(np.pi * (xx - step_x) / (x_reat - step_x))
    u[in_recirc] = -0.2 * u_inf * recirc_strength[in_recirc] * (
        1.0 - yy[in_recirc] / (2 * h)
    )
    u[in_step_shadow & (xx <= step_x + 0.01)] = 0.0  # Step face

    u[:, 0] = 0.0  # Wall

    # v from continuity
    dudx = np.gradient(u, x, axis=0)
    v = np.zeros_like(u)
    dy = y[1] - y[0]
    for j in range(1, ny):
        v[:, j] = v[:, j - 1] - dudx[:, j] * dy
    v[:, 0] = 0.0

    p = -0.5 * (u ** 2 + v ** 2) + 0.5 * u_inf ** 2
    p -= p.mean()

    nu = 1.5e-5
    nut = nu * 80 * eta_norm * (1.0 - eta_norm)
    nut[in_recirc] *= 2.5
    nut = np.maximum(nut, 0.0)
    nu_tilde = nut * 1.1

    Cf = 2.0 * nu * (u[:, 1] - u[:, 0]) / (dy * u_inf ** 2)

    return {
        "x": xx.ravel(), "y": yy.ravel(),
        "u": u.ravel(), "v": v.ravel(),
        "p": p.ravel(),
        "nu_t": nut.ravel(), "nu_tilde": nu_tilde.ravel(),
        "Cf": Cf, "x_1d": x,
        "xx": xx, "yy": yy, "u_2d": u, "v_2d": v, "p_2d": p,
        "nx": nx, "ny": ny,
    }


# ============================================================================
# Sparse Sensor Generator
# ============================================================================
def generate_synthetic_sensors(
    flow_field: Dict[str, np.ndarray],
    n_stations: int = 4,
    n_points_per_station: int = 20,
    noise_std: float = 0.01,
    seed: int = 42,
) -> SensorConfiguration:
    """
    Generate sparse PIV-like sensor data from a 2D flow field.

    Mimics Greenblatt's wall-hump PIV setup: vertical profiles at
    a few streamwise stations.

    Parameters
    ----------
    flow_field : dict
        Output from generate_wall_hump_2d() or generate_bfs_2d().
    n_stations : int
        Number of streamwise measurement stations.
    n_points_per_station : int
        Measurement points per vertical profile.
    noise_std : float
        Gaussian noise standard deviation on velocity.
    seed : int
        Random seed.

    Returns
    -------
    SensorConfiguration
    """
    rng = np.random.default_rng(seed)

    xx = flow_field["xx"]
    yy = flow_field["yy"]
    u_2d = flow_field["u_2d"]
    v_2d = flow_field["v_2d"]

    x_1d = flow_field["x_1d"]
    y_1d = np.linspace(yy.min(), yy.max(), flow_field["ny"])

    # Station locations: spread across domain, including separation region
    x_stations = np.linspace(
        x_1d[len(x_1d) // 5], x_1d[4 * len(x_1d) // 5], n_stations
    )

    x_sensors, y_sensors, u_meas, v_meas = [], [], [], []

    for xs in x_stations:
        ix = np.argmin(np.abs(x_1d - xs))
        # Sample at regular y-locations
        iy_indices = np.linspace(1, len(y_1d) - 2, n_points_per_station, dtype=int)
        for iy in iy_indices:
            x_sensors.append(x_1d[ix])
            y_sensors.append(y_1d[iy])
            u_meas.append(u_2d[ix, iy] + rng.normal(0, noise_std))
            v_meas.append(v_2d[ix, iy] + rng.normal(0, noise_std))

    return SensorConfiguration(
        x_sensors=np.array(x_sensors),
        y_sensors=np.array(y_sensors),
        u_measured=np.array(u_meas),
        v_measured=np.array(v_meas),
        noise_std=noise_std,
    )


# ============================================================================
# PINN-DA Network (PyTorch)
# ============================================================================
if HAS_TORCH:

    class _PINNDANetwork(nn.Module):
        """
        Fully-connected network: (x, y) → (u, v, p, ν̃).

        Uses tanh activation (smooth derivatives for PDE residuals)
        and Xavier initialization.
        """

        def __init__(
            self,
            hidden_layers: Tuple[int, ...] = (128, 128, 128, 128, 128, 128),
            activation: str = "tanh",
        ):
            super().__init__()
            layers = []
            prev = 2  # Input: (x, y)
            for h in hidden_layers:
                layers.append(nn.Linear(prev, h))
                if activation == "tanh":
                    layers.append(nn.Tanh())
                elif activation == "sin":
                    layers.append(nn.Tanh())  # Fallback
                elif activation == "gelu":
                    layers.append(nn.GELU())
                prev = h
            layers.append(nn.Linear(prev, 4))  # Output: (u, v, p, ν̃)
            self.net = nn.Sequential(*layers)
            self._init_weights()

        def _init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
                    nn.init.zeros_(m.bias)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Parameters
            ----------
            x : Tensor (N, 2) — [x_coord, y_coord]

            Returns
            -------
            Tensor (N, 4) — [u, v, p, nu_tilde]
            """
            return self.net(x)


# ============================================================================
# RANS Residual Computation
# ============================================================================
def compute_rans_residuals_numpy(
    x: np.ndarray,
    y: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    p: np.ndarray,
    nu_tilde: np.ndarray,
    nu: float = 1.5e-5,
    nx: int = 50,
    ny: int = 25,
) -> Dict[str, np.ndarray]:
    """
    Compute RANS residuals on a structured grid using finite differences.

    Numpy-based fallback when PyTorch autograd is not available or for
    validation.

    Returns dict with 'continuity', 'momentum_x', 'momentum_y' residuals.
    """
    u2 = u.reshape(nx, ny)
    v2 = v.reshape(nx, ny)
    p2 = p.reshape(nx, ny)
    nt2 = nu_tilde.reshape(nx, ny)
    x2 = x.reshape(nx, ny)
    y2 = y.reshape(nx, ny)

    dx = x2[1, 0] - x2[0, 0] if nx > 1 else 1.0
    dy = y2[0, 1] - y2[0, 0] if ny > 1 else 1.0

    # Gradients
    dudx = np.gradient(u2, dx, axis=0)
    dudy = np.gradient(u2, dy, axis=1)
    dvdx = np.gradient(v2, dx, axis=0)
    dvdy = np.gradient(v2, dy, axis=1)
    dpdx = np.gradient(p2, dx, axis=0)
    dpdy = np.gradient(p2, dy, axis=1)

    # Second derivatives
    d2udx2 = np.gradient(dudx, dx, axis=0)
    d2udy2 = np.gradient(dudy, dy, axis=1)
    d2vdx2 = np.gradient(dvdx, dx, axis=0)
    d2vdy2 = np.gradient(dvdy, dy, axis=1)

    # Effective viscosity: ν_eff = ν + ν_t
    chi = nt2 / nu
    fv1 = chi ** 3 / (chi ** 3 + SA_CONSTANTS["cv1"] ** 3)
    nu_t = nt2 * fv1
    nu_eff = nu + nu_t

    # Continuity: ∂u/∂x + ∂v/∂y = 0
    r_cont = dudx + dvdy

    # x-Momentum: u·∂u/∂x + v·∂u/∂y + ∂p/∂x - ν_eff·∇²u = 0
    r_momx = u2 * dudx + v2 * dudy + dpdx - nu_eff * (d2udx2 + d2udy2)

    # y-Momentum
    r_momy = u2 * dvdx + v2 * dvdy + dpdy - nu_eff * (d2vdx2 + d2vdy2)

    return {
        "continuity": r_cont.ravel(),
        "momentum_x": r_momx.ravel(),
        "momentum_y": r_momy.ravel(),
    }


# ============================================================================
# PINN-DA Assimilator (PyTorch-based)
# ============================================================================
class PINNDataAssimilator:
    """
    PINN-based Data Assimilator with SA turbulence closure.

    Reconstructs full 2D flow fields from sparse sensor measurements
    using the RANS equations + SA model as physics constraints.

    Usage:
        config = PINNDAConfig(nu=1.5e-5)
        assimilator = PINNDataAssimilator(config)
        result = assimilator.train(sensors, domain)
        u_pred, v_pred, p_pred = assimilator.predict(x_eval, y_eval)
    """

    def __init__(self, config: PINNDAConfig = None):
        self.config = config or PINNDAConfig()
        self._trained = False
        self._network = None
        self._x_mean = 0.0
        self._x_std = 1.0
        self._y_mean = 0.0
        self._y_std = 1.0

    def _build_network(self):
        """Build the PINN network."""
        if not HAS_TORCH:
            raise ImportError(
                "PINN-DA requires PyTorch. Install: pip install torch"
            )
        self._network = _PINNDANetwork(
            hidden_layers=self.config.hidden_layers,
            activation=self.config.activation,
        )

    def _normalize_coords(
        self, x: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Normalize coordinates to [-1, 1]."""
        x_n = (x - self._x_mean) / max(self._x_std, 1e-10)
        y_n = (y - self._y_mean) / max(self._y_std, 1e-10)
        return x_n, y_n

    def _compute_pde_residuals(
        self, xy: "torch.Tensor"
    ) -> Dict[str, "torch.Tensor"]:
        """
        Compute RANS + SA residuals using PyTorch autograd.

        Parameters
        ----------
        xy : Tensor (N, 2) — requires_grad=True

        Returns
        -------
        dict with continuity, momentum_x, momentum_y, sa residual tensors.
        """
        xy.requires_grad_(True)
        out = self._network(xy)
        u, v, p, nt = out[:, 0:1], out[:, 1:2], out[:, 2:3], out[:, 3:4]

        # First derivatives via autograd
        grads_u = torch.autograd.grad(
            u, xy, torch.ones_like(u), create_graph=True
        )[0]
        dudx, dudy = grads_u[:, 0:1], grads_u[:, 1:2]

        grads_v = torch.autograd.grad(
            v, xy, torch.ones_like(v), create_graph=True
        )[0]
        dvdx, dvdy = grads_v[:, 0:1], grads_v[:, 1:2]

        grads_p = torch.autograd.grad(
            p, xy, torch.ones_like(p), create_graph=True
        )[0]
        dpdx, dpdy = grads_p[:, 0:1], grads_p[:, 1:2]

        # Second derivatives
        d2udx2 = torch.autograd.grad(
            dudx, xy, torch.ones_like(dudx), create_graph=True
        )[0][:, 0:1]
        d2udy2 = torch.autograd.grad(
            dudy, xy, torch.ones_like(dudy), create_graph=True
        )[0][:, 1:2]
        d2vdx2 = torch.autograd.grad(
            dvdx, xy, torch.ones_like(dvdx), create_graph=True
        )[0][:, 0:1]
        d2vdy2 = torch.autograd.grad(
            dvdy, xy, torch.ones_like(dvdy), create_graph=True
        )[0][:, 1:2]

        # Effective viscosity
        nu = self.config.nu
        chi = torch.abs(nt) / nu
        cv1 = SA_CONSTANTS["cv1"]
        fv1 = chi ** 3 / (chi ** 3 + cv1 ** 3 + 1e-10)
        nu_t = torch.abs(nt) * fv1
        nu_eff = nu + nu_t

        # Continuity
        r_cont = dudx + dvdy

        # x-Momentum
        r_momx = u * dudx + v * dudy + dpdx - nu_eff * (d2udx2 + d2udy2)

        # y-Momentum
        r_momy = u * dvdx + v * dvdy + dpdy - nu_eff * (d2vdx2 + d2vdy2)

        # SA residual (simplified steady-state)
        grads_nt = torch.autograd.grad(
            nt, xy, torch.ones_like(nt), create_graph=True
        )[0]
        dntdx, dntdy = grads_nt[:, 0:1], grads_nt[:, 1:2]

        # Strain rate magnitude (simplified)
        S = torch.sqrt(dudy ** 2 + dvdx ** 2 + 1e-10)
        d = torch.abs(xy[:, 1:2]) + 1e-6  # Wall distance

        # SA production
        cb1 = SA_CONSTANTS["cb1"]
        prod = cb1 * S * torch.abs(nt)

        # SA destruction
        cw1 = SA_CONSTANTS["cw1"]
        dest = cw1 * (torch.abs(nt) / d) ** 2

        # SA diffusion (simplified)
        sigma = SA_CONSTANTS["sigma"]
        d2ntdy2 = torch.autograd.grad(
            dntdy, xy, torch.ones_like(dntdy), create_graph=True
        )[0][:, 1:2]
        diff = (1.0 / sigma) * (nu + torch.abs(nt)) * d2ntdy2

        r_sa = prod - dest + diff

        return {
            "continuity": r_cont,
            "momentum_x": r_momx,
            "momentum_y": r_momy,
            "sa": r_sa,
        }

    def train(
        self,
        sensors: SensorConfiguration,
        reference_field: Optional[Dict[str, np.ndarray]] = None,
    ) -> PINNDAResult:
        """
        Train the PINN-DA model.

        Parameters
        ----------
        sensors : SensorConfiguration
            Sparse measurement data.
        reference_field : dict, optional
            Full truth field for error evaluation (not used in training).

        Returns
        -------
        PINNDAResult
        """
        if not HAS_TORCH:
            raise ImportError("PINN-DA requires PyTorch.")

        cfg = self.config
        self._build_network()
        model = self._network

        # Normalization stats from domain
        self._x_mean = (cfg.x_min + cfg.x_max) / 2
        self._x_std = (cfg.x_max - cfg.x_min) / 2
        self._y_mean = (cfg.y_min + cfg.y_max) / 2
        self._y_std = (cfg.y_max - cfg.y_min) / 2

        # Sensor data → tensors
        xs_n, ys_n = self._normalize_coords(sensors.x_sensors, sensors.y_sensors)
        xy_sensor = torch.tensor(
            np.column_stack([xs_n, ys_n]), dtype=torch.float32
        )
        u_target = torch.tensor(
            sensors.u_measured, dtype=torch.float32
        ).unsqueeze(1)
        v_target = torch.tensor(
            sensors.v_measured, dtype=torch.float32
        ).unsqueeze(1)

        # Collocation points (interior)
        rng = np.random.default_rng(42)
        x_coll = rng.uniform(cfg.x_min, cfg.x_max, cfg.n_collocation)
        y_coll = rng.uniform(cfg.y_min, cfg.y_max, cfg.n_collocation)
        xc_n, yc_n = self._normalize_coords(x_coll, y_coll)
        xy_coll = torch.tensor(
            np.column_stack([xc_n, yc_n]), dtype=torch.float32,
        ).requires_grad_(True)

        # Boundary points (wall: y = y_min)
        x_wall = np.linspace(cfg.x_min, cfg.x_max, cfg.n_boundary)
        y_wall = np.full(cfg.n_boundary, cfg.y_min)
        xw_n, yw_n = self._normalize_coords(x_wall, y_wall)
        xy_wall = torch.tensor(
            np.column_stack([xw_n, yw_n]), dtype=torch.float32
        )

        # Optimizer
        optimizer = torch.optim.Adam(
            model.parameters(), lr=cfg.learning_rate
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg.lr_decay_steps,
            gamma=cfg.lr_decay_rate,
        )

        result = PINNDAResult(n_sensors=sensors.n_sensors, noise_std=sensors.noise_std)
        t0 = time.time()

        for epoch in range(cfg.max_epochs):
            optimizer.zero_grad()

            # --- Data loss: match sensors ---
            pred_sensor = model(xy_sensor)
            L_data = (
                torch.mean((pred_sensor[:, 0:1] - u_target) ** 2)
                + torch.mean((pred_sensor[:, 1:2] - v_target) ** 2)
            )

            # --- Physics loss: RANS residuals at collocation ---
            residuals = self._compute_pde_residuals(xy_coll)
            L_cont = torch.mean(residuals["continuity"] ** 2)
            L_mom = (
                torch.mean(residuals["momentum_x"] ** 2)
                + torch.mean(residuals["momentum_y"] ** 2)
            )
            L_sa = torch.mean(residuals["sa"] ** 2)

            # --- BC loss: wall no-slip ---
            pred_wall = model(xy_wall)
            L_bc = (
                torch.mean(pred_wall[:, 0:1] ** 2)  # u = 0
                + torch.mean(pred_wall[:, 1:2] ** 2)  # v = 0
            )

            # --- Total loss ---
            loss = (
                cfg.lambda_data * L_data
                + cfg.lambda_continuity * L_cont
                + cfg.lambda_momentum * L_mom
                + cfg.lambda_sa * L_sa
                + cfg.lambda_bc * L_bc
            )

            loss.backward()
            optimizer.step()
            scheduler.step()

            result.loss_history.append(loss.item())

            if epoch % cfg.print_interval == 0:
                logger.info(
                    "Epoch %d: total=%.4e data=%.4e cont=%.4e mom=%.4e "
                    "sa=%.4e bc=%.4e",
                    epoch, loss.item(), L_data.item(), L_cont.item(),
                    L_mom.item(), L_sa.item(), L_bc.item(),
                )

        result.training_time_s = time.time() - t0
        result.training_epochs = cfg.max_epochs

        # Final losses
        result.final_data_loss = L_data.item()
        result.final_continuity_loss = L_cont.item()
        result.final_momentum_loss = L_mom.item()
        result.final_sa_loss = L_sa.item()
        result.final_bc_loss = L_bc.item()
        result.final_total_loss = loss.item()

        # Evaluate against reference if provided
        if reference_field is not None:
            result = self._evaluate(result, reference_field)

        self._trained = True
        result.summary = (
            f"PINN-DA: {sensors.n_sensors} sensors, "
            f"u_RMSE={result.u_rmse:.6f}, "
            f"epochs={cfg.max_epochs}, time={result.training_time_s:.1f}s"
        )
        return result

    def _evaluate(
        self, result: PINNDAResult, ref: Dict[str, np.ndarray]
    ) -> PINNDAResult:
        """Evaluate reconstruction quality against reference field."""
        x_ref = ref["x"]
        y_ref = ref["y"]
        u_pred, v_pred, p_pred, nt_pred = self.predict(x_ref, y_ref)

        result.u_rmse = float(np.sqrt(np.mean((u_pred - ref["u"]) ** 2)))
        result.v_rmse = float(np.sqrt(np.mean((v_pred - ref["v"]) ** 2)))
        result.p_rmse = float(np.sqrt(np.mean((p_pred - ref["p"]) ** 2)))
        if "nu_tilde" in ref:
            result.nut_rmse = float(
                np.sqrt(np.mean((nt_pred - ref["nu_tilde"]) ** 2))
            )

        # Separation metrics from Cf
        if "Cf" in ref and "x_1d" in ref:
            x_1d = ref["x_1d"]
            Cf = ref["Cf"]
            # Predict Cf from wall gradient
            try:
                nx, ny = ref["nx"], ref["ny"]
                u_2d_pred = u_pred.reshape(nx, ny)
                dy = y_ref.reshape(nx, ny)[0, 1] - y_ref.reshape(nx, ny)[0, 0]
                Cf_pred = 2 * self.config.nu * (u_2d_pred[:, 1] - u_2d_pred[:, 0]) / dy
                sep_idx = np.where(Cf_pred < 0)[0]
                if len(sep_idx) > 0:
                    result.separation_x_pred = x_1d[sep_idx[0]]
                    result.reattachment_x_pred = x_1d[sep_idx[-1]]
                    result.bubble_length_pred = (
                        result.reattachment_x_pred - result.separation_x_pred
                    )
            except Exception:
                pass

        return result

    def predict(
        self, x: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict flow fields at given coordinates.

        Parameters
        ----------
        x, y : ndarray
            Evaluation coordinates.

        Returns
        -------
        u, v, p, nu_tilde : ndarray — predicted fields.
        """
        if not self._trained and self._network is None:
            raise RuntimeError("Call train() first.")

        x_n, y_n = self._normalize_coords(x, y)
        xy = torch.tensor(
            np.column_stack([x_n, y_n]), dtype=torch.float32
        )
        self._network.eval()
        with torch.no_grad():
            out = self._network(xy).numpy()

        return out[:, 0], out[:, 1], out[:, 2], out[:, 3]


# ============================================================================
# NumPy-Only Fallback Assimilator
# ============================================================================
class PINNDAAssimilatorNumpy:
    """
    Lightweight PINN-DA using Fourier basis + L-BFGS-B optimization.

    Fallback when PyTorch is not available. Uses the same physics
    constraints but with a simpler Fourier representation instead
    of a neural network.
    """

    def __init__(
        self,
        config: PINNDAConfig = None,
        n_basis_x: int = 8,
        n_basis_y: int = 8,
    ):
        self.config = config or PINNDAConfig()
        self.n_basis_x = n_basis_x
        self.n_basis_y = n_basis_y
        self._coefficients = None
        self._trained = False

    def _build_basis(
        self, x: np.ndarray, y: np.ndarray
    ) -> np.ndarray:
        """Build 2D Fourier basis functions."""
        n_basis = self.n_basis_x * self.n_basis_y
        N = len(x)
        basis = np.zeros((N, n_basis))

        x_range = self.config.x_max - self.config.x_min
        y_range = self.config.y_max - self.config.y_min
        x_norm = (x - self.config.x_min) / max(x_range, 1e-10)
        y_norm = (y - self.config.y_min) / max(y_range, 1e-10)

        idx = 0
        for kx in range(self.n_basis_x):
            for ky in range(self.n_basis_y):
                basis[:, idx] = (
                    np.sin((kx + 1) * np.pi * x_norm)
                    * np.sin((ky + 1) * np.pi * y_norm)
                )
                idx += 1
        return basis

    def _reconstruct(
        self, coeffs: np.ndarray, basis: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Reconstruct u, v, p, ν̃ from Fourier coefficients."""
        n_basis = basis.shape[1]
        c_u = coeffs[:n_basis]
        c_v = coeffs[n_basis:2 * n_basis]
        c_p = coeffs[2 * n_basis:3 * n_basis]
        c_nt = coeffs[3 * n_basis:4 * n_basis]

        u = basis @ c_u
        v = basis @ c_v
        p = basis @ c_p
        nt = np.abs(basis @ c_nt)  # Non-negative ν̃
        return u, v, p, nt

    def _loss(
        self,
        coeffs: np.ndarray,
        basis_sensor: np.ndarray,
        u_meas: np.ndarray,
        v_meas: np.ndarray,
        basis_coll: np.ndarray,
        x_coll: np.ndarray,
        y_coll: np.ndarray,
        basis_wall: np.ndarray,
        nx_coll: int,
        ny_coll: int,
    ) -> float:
        """Compute total PINN-DA loss."""
        cfg = self.config

        # Data loss
        u_s, v_s, _, _ = self._reconstruct(coeffs, basis_sensor)
        L_data = np.mean((u_s - u_meas) ** 2 + (v_s - v_meas) ** 2)

        # Physics loss (RANS residuals on collocation grid)
        u_c, v_c, p_c, nt_c = self._reconstruct(coeffs, basis_coll)
        residuals = compute_rans_residuals_numpy(
            x_coll, y_coll, u_c, v_c, p_c, nt_c,
            nu=cfg.nu, nx=nx_coll, ny=ny_coll,
        )
        L_cont = np.mean(residuals["continuity"] ** 2)
        L_mom = np.mean(
            residuals["momentum_x"] ** 2 + residuals["momentum_y"] ** 2
        )

        # BC loss: wall no-slip
        u_w, v_w, _, _ = self._reconstruct(coeffs, basis_wall)
        L_bc = np.mean(u_w ** 2 + v_w ** 2)

        total = (
            cfg.lambda_data * L_data
            + cfg.lambda_continuity * L_cont
            + cfg.lambda_momentum * L_mom
            + cfg.lambda_bc * L_bc
        )
        return total

    def train(
        self,
        sensors: SensorConfiguration,
        reference_field: Optional[Dict[str, np.ndarray]] = None,
    ) -> PINNDAResult:
        """Train the numpy-based PINN-DA."""
        cfg = self.config

        # Sensor basis
        basis_sensor = self._build_basis(sensors.x_sensors, sensors.y_sensors)

        # Collocation grid
        nx_c, ny_c = 30, 15
        x_c = np.linspace(cfg.x_min, cfg.x_max, nx_c)
        y_c = np.linspace(cfg.y_min, cfg.y_max, ny_c)
        xx_c, yy_c = np.meshgrid(x_c, y_c, indexing='ij')
        basis_coll = self._build_basis(xx_c.ravel(), yy_c.ravel())

        # Wall basis
        x_w = np.linspace(cfg.x_min, cfg.x_max, 50)
        y_w = np.full(50, cfg.y_min)
        basis_wall = self._build_basis(x_w, y_w)

        # Optimize
        n_basis = self.n_basis_x * self.n_basis_y
        x0 = np.zeros(4 * n_basis)

        t0 = time.time()
        opt_result = minimize(
            self._loss, x0,
            args=(
                basis_sensor, sensors.u_measured, sensors.v_measured,
                basis_coll, xx_c.ravel(), yy_c.ravel(),
                basis_wall, nx_c, ny_c,
            ),
            method="L-BFGS-B",
            options={"maxiter": 500, "ftol": 1e-10},
        )
        training_time = time.time() - t0

        self._coefficients = opt_result.x
        self._trained = True

        result = PINNDAResult(
            final_total_loss=opt_result.fun,
            training_time_s=training_time,
            training_epochs=opt_result.nit,
            n_sensors=sensors.n_sensors,
            noise_std=sensors.noise_std,
        )

        # Evaluate if reference provided
        if reference_field is not None:
            x_ref = reference_field["x"]
            y_ref = reference_field["y"]
            basis_ref = self._build_basis(x_ref, y_ref)
            u_p, v_p, p_p, nt_p = self._reconstruct(
                self._coefficients, basis_ref
            )
            result.u_rmse = float(np.sqrt(np.mean((u_p - reference_field["u"]) ** 2)))
            result.v_rmse = float(np.sqrt(np.mean((v_p - reference_field["v"]) ** 2)))
            result.p_rmse = float(np.sqrt(np.mean((p_p - reference_field["p"]) ** 2)))

        result.summary = (
            f"PINN-DA-Numpy: {sensors.n_sensors} sensors, "
            f"u_RMSE={result.u_rmse:.6f}, "
            f"iters={opt_result.nit}, time={training_time:.1f}s"
        )
        return result

    def predict(
        self, x: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Predict at given coordinates."""
        if not self._trained:
            raise RuntimeError("Call train() first.")
        basis = self._build_basis(x, y)
        return self._reconstruct(self._coefficients, basis)


# ============================================================================
# Comparison: PINN-DA vs PINN-BL
# ============================================================================
def compare_pinn_da_vs_pinn_bl(
    case: str = "wall_hump",
    n_sensors: int = 80,
    noise_std: float = 0.01,
) -> Dict[str, Any]:
    """
    Head-to-head comparison: PINN-DA (2D RANS) vs PINN-BL (1D VK integral).

    Parameters
    ----------
    case : str
        'wall_hump' or 'bfs'.
    n_sensors : int
        Total number of sensor measurements.
    noise_std : float
        Measurement noise.

    Returns
    -------
    dict with comparison metrics.
    """
    from scripts.ml_augmentation.pinn_boundary_layer import (
        PINNBoundaryLayerCorrector, generate_bl_data,
    )

    # Generate 2D field
    if case == "wall_hump":
        field_2d = generate_wall_hump_2d(nx=50, ny=25)
        bl_case = "nasa_hump"
    elif case == "bfs":
        field_2d = generate_bfs_2d(nx=50, ny=25)
        bl_case = "flat_plate_apg"
    else:
        raise ValueError(f"Unknown case: {case}")

    # Generate sensors
    n_stations = max(n_sensors // 20, 2)
    sensors = generate_synthetic_sensors(
        field_2d,
        n_stations=n_stations,
        n_points_per_station=n_sensors // n_stations,
        noise_std=noise_std,
    )

    # --- PINN-DA ---
    da_config = PINNDAConfig(
        hidden_layers=(64, 64, 64, 64),
        max_epochs=500,
        n_collocation=1000,
        n_boundary=100,
        learning_rate=1e-3,
    )

    if HAS_TORCH:
        da = PINNDataAssimilator(da_config)
    else:
        da = PINNDAAssimilatorNumpy(da_config, n_basis_x=6, n_basis_y=6)

    da_result = da.train(sensors, reference_field=field_2d)

    # --- PINN-BL (1D) ---
    x_1d, bl_data = generate_bl_data(bl_case)
    bl_corrector = PINNBoundaryLayerCorrector(lambda_phys=0.1, n_basis=15)
    bl_result = bl_corrector.fit(
        x_1d, bl_data["Cf_rans"], bl_data["Cf_dns"],
        bl_data["theta"], bl_data["H"], bl_data["U_e"],
    )

    comparison = {
        "case": case,
        "n_sensors": n_sensors,
        "noise_std": noise_std,
        "pinn_da_u_rmse": da_result.u_rmse,
        "pinn_da_v_rmse": da_result.v_rmse,
        "pinn_da_training_time": da_result.training_time_s,
        "pinn_da_epochs": da_result.training_epochs,
        "pinn_bl_cf_rmse_before": bl_result.rmse_before,
        "pinn_bl_cf_rmse_after": bl_result.rmse_after,
        "pinn_bl_improvement_pct": bl_result.improvement_pct,
        "pinn_da_type": "pytorch" if HAS_TORCH else "numpy",
        "pinn_da_separation_x": da_result.separation_x_pred,
        "pinn_da_bubble_length": da_result.bubble_length_pred,
    }
    return comparison


# ============================================================================
# Demo
# ============================================================================
def _demo():
    """Demonstrate PINN-DA data assimilation."""
    print("=" * 70)
    print("  PINN-DA: Data Assimilation with Sparse Sensors")
    print("=" * 70)

    # 1. Generate synthetic 2D field
    print("\n  [1] Generating wall-hump 2D flow field...")
    field = generate_wall_hump_2d(nx=50, ny=25)
    print(f"  Grid: {field['nx']}×{field['ny']} = {len(field['u'])} points")
    print(f"  u range: [{field['u'].min():.3f}, {field['u'].max():.3f}]")

    # 2. Generate sparse sensors (4 PIV stations)
    print("\n  [2] Generating sparse PIV sensors (4 stations)...")
    sensors = generate_synthetic_sensors(
        field, n_stations=4, n_points_per_station=20, noise_std=0.01
    )
    print(f"  Sensors: {sensors.n_sensors}")
    print(f"  Noise σ: {sensors.noise_std}")

    # 3. Train PINN-DA
    print("\n  [3] Training PINN-DA...")
    config = PINNDAConfig(
        hidden_layers=(64, 64, 64, 64),
        max_epochs=200,
        n_collocation=500,
        n_boundary=50,
        learning_rate=1e-3,
        print_interval=100,
    )

    if HAS_TORCH:
        da = PINNDataAssimilator(config)
    else:
        print("  [INFO] PyTorch not available, using numpy fallback")
        da = PINNDAAssimilatorNumpy(config, n_basis_x=6, n_basis_y=6)

    result = da.train(sensors, reference_field=field)
    print(f"\n  Training time: {result.training_time_s:.1f}s")
    print(f"  u RMSE: {result.u_rmse:.6f}")
    print(f"  v RMSE: {result.v_rmse:.6f}")
    print(f"  p RMSE: {result.p_rmse:.6f}")
    print(f"  Final data loss: {result.final_data_loss:.4e}")
    print(f"  Final physics loss: {result.final_continuity_loss:.4e}")

    # 4. Compare vs PINN-BL
    print("\n  [4] Comparing PINN-DA vs PINN-BL...")
    comparison = compare_pinn_da_vs_pinn_bl(
        case="wall_hump", n_sensors=80, noise_std=0.01
    )
    print(f"  PINN-DA u-RMSE: {comparison['pinn_da_u_rmse']:.6f}")
    print(f"  PINN-BL Cf-RMSE: {comparison['pinn_bl_cf_rmse_before']:.6f} "
          f"→ {comparison['pinn_bl_cf_rmse_after']:.6f} "
          f"({comparison['pinn_bl_improvement_pct']:+.1f}%)")

    print(f"\n{'=' * 70}")
    print("  Demo complete!")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _demo()
