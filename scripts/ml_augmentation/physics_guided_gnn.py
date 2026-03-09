#!/usr/bin/env python3
"""
Physics-Guided Graph Neural Network (PG-GNN) for RANS Augmentation
====================================================================
Extends MeshGraphNet with physics-informed loss terms that enforce
governing equations during training, enabling better generalization
for turbulent separated flows.

Physics Losses:
  - Continuity residual: ∇·u ≈ 0 via graph-based divergence
  - Momentum residual: simplified RANS equation residual
  - Realizability: β > 0, bounded eddy-viscosity ratio
  - Smoothness: graph Laplacian regularization on predicted β

This follows the PG-GNN framework from ASME J. Turbomachinery (2025)
and the Generalized Field Inversion strategy from Srivastava et al.
(AIAA SciTech 2024).

References:
  - PG-GNN (ASME J. Turbomachinery, Dec. 2025): physics-guided GNNs
    with Euler/RANS constraints for turbine vane flows
  - Srivastava et al. (2024), AIAA SciTech: generalized field
    inversion for improved RANS separation prediction
  - Pfaff et al. (2021), ICML: MeshGraphNet foundation
"""

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)
PROJECT = Path(__file__).resolve().parent.parent.parent

# ---------------------------------------------------------------------------
# Dependency guards
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    class _ModuleStub:
        Module = object
    nn = _ModuleStub()
    class _FStub:
        @staticmethod
        def mse_loss(*a, **k): raise ImportError("torch required")
        @staticmethod
        def relu(*a, **k): raise ImportError("torch required")
    F = _FStub()

try:
    import torch_geometric
    from torch_geometric.data import Data
    HAS_TORCH_GEOMETRIC = True
except ImportError:
    HAS_TORCH_GEOMETRIC = False


def _require_pyg():
    if not HAS_TORCH or not HAS_TORCH_GEOMETRIC:
        raise ImportError(
            "Physics-Guided GNN requires PyTorch and PyTorch Geometric. "
            "Install with: pip install torch torch-geometric"
        )


# ============================================================================
# Configuration
# ============================================================================
@dataclass
class PhysicsGNNConfig:
    """Configuration for Physics-Guided GNN."""

    # Physics loss weights
    lambda_data: float = 1.0        # Data fitting loss weight
    lambda_continuity: float = 0.1  # Continuity residual weight
    lambda_momentum: float = 0.05   # Momentum residual weight
    lambda_realizability: float = 0.5  # Realizability constraint weight
    lambda_smoothness: float = 0.01    # Graph Laplacian smoothness

    # Realizability bounds
    beta_min: float = 0.1   # Minimum β (production can't be negative)
    beta_max: float = 5.0   # Maximum β (physical upper bound)
    nut_ratio_max: float = 1e4  # Max ν_t/ν ratio

    # Training schedule for physics losses
    warmup_epochs: int = 20    # Epochs before physics losses kick in
    ramp_epochs: int = 30      # Epochs to ramp physics loss to full weight


# ============================================================================
# Graph-Based Differential Operators
# ============================================================================
def compute_graph_gradient(
    field_values: Any,
    pos: Any,
    edge_index: Any,
) -> Any:
    """
    Approximate spatial gradient of a scalar field on an unstructured
    mesh using weighted least-squares on neighbor stencils.

    Parameters
    ----------
    field_values : Tensor (N,) or (N, 1)
    pos : Tensor (N, ndim)
    edge_index : Tensor (2, E)

    Returns
    -------
    Tensor (N, ndim) — estimated gradient at each node.
    """
    _require_pyg()
    from torch_geometric.utils import scatter

    ndim = pos.shape[1]
    N = pos.shape[0]
    f = field_values.view(-1, 1) if field_values.dim() == 1 else field_values

    src, dst = edge_index[0], edge_index[1]
    dx = pos[dst] - pos[src]  # (E, ndim)
    df = f[dst] - f[src]      # (E, 1)

    # Inverse-distance weighting
    dist = torch.norm(dx, dim=1, keepdim=True).clamp(min=1e-10)
    w = 1.0 / dist  # (E, 1)

    w_dx = w * dx  # (E, ndim)
    w_df = w * df   # (E, 1)

    # Accumulate pseudo-normal equations per node
    ATA_flat = torch.zeros(N, ndim * ndim, device=pos.device)
    ATb = torch.zeros(N, ndim, device=pos.device)

    for d1 in range(ndim):
        for d2 in range(ndim):
            val = w_dx[:, d1] * w_dx[:, d2]
            ATA_flat[:, d1 * ndim + d2] = scatter(
                val, src, dim=0, dim_size=N, reduce='sum'
            )
        b_val = w_dx[:, d1] * w_df[:, 0]
        ATb[:, d1] = scatter(
            b_val, src, dim=0, dim_size=N, reduce='sum'
        )

    ATA = ATA_flat.view(N, ndim, ndim)
    ATA = ATA + 1e-8 * torch.eye(ndim, device=pos.device).unsqueeze(0)

    try:
        grad = torch.linalg.solve(ATA, ATb.unsqueeze(-1)).squeeze(-1)
    except RuntimeError:
        grad = torch.bmm(
            torch.linalg.pinv(ATA), ATb.unsqueeze(-1)
        ).squeeze(-1)

    return grad


def compute_graph_divergence(
    vector_field: Any,
    pos: Any,
    edge_index: Any,
) -> Any:
    """
    Approximate divergence ∇·u using graph-based gradient.

    Parameters
    ----------
    vector_field : Tensor (N, ndim)
    pos : Tensor (N, ndim)
    edge_index : Tensor (2, E)

    Returns
    -------
    Tensor (N,) — divergence at each node.
    """
    _require_pyg()
    ndim = vector_field.shape[1]
    div = torch.zeros(pos.shape[0], device=pos.device)

    for d in range(ndim):
        grad_d = compute_graph_gradient(
            vector_field[:, d], pos, edge_index
        )
        div = div + grad_d[:, d]

    return div


def graph_laplacian_smoothness(
    field_values: Any,
    edge_index: Any,
) -> Any:
    """
    Compute graph Laplacian smoothness penalty.

    L_smooth = (1/E) Σ_{(i,j)} (f_i - f_j)^2
    """
    _require_pyg()
    f = field_values.view(-1)
    src, dst = edge_index[0], edge_index[1]
    diff = f[src] - f[dst]
    return (diff ** 2).mean()


# ============================================================================
# Physics Loss Functions
# ============================================================================
class ContinuityLoss(nn.Module):
    """
    Continuity equation residual: ∇·u = 0.
    Computed via graph-based divergence on the velocity field.
    """

    def forward(self, velocity, pos, edge_index):
        _require_pyg()
        div = compute_graph_divergence(velocity, pos, edge_index)
        return (div ** 2).mean()


class RealizabilityLoss(nn.Module):
    """
    Realizability constraints on the predicted correction field.
    Enforces β > beta_min, β < beta_max, and smooth β.
    """

    def __init__(self, beta_min: float = 0.1, beta_max: float = 5.0):
        if HAS_TORCH:
            super().__init__()
        self.beta_min = beta_min
        self.beta_max = beta_max

    def forward(self, beta_pred, edge_index):
        _require_pyg()
        beta = beta_pred.view(-1)

        violation_low = F.relu(self.beta_min - beta)
        loss_low = (violation_low ** 2).mean()

        violation_high = F.relu(beta - self.beta_max)
        loss_high = (violation_high ** 2).mean()

        loss_smooth = graph_laplacian_smoothness(beta, edge_index)

        return loss_low + loss_high + 0.1 * loss_smooth


class MomentumResidualLoss(nn.Module):
    """
    Simplified RANS momentum residual using predicted β and known flow field.
    """

    def forward(self, beta_pred, velocity, pressure, nu_t, nu, pos, edge_index):
        _require_pyg()
        beta = beta_pred.view(-1)

        du_dx = compute_graph_gradient(velocity[:, 0], pos, edge_index)
        dp_dx = compute_graph_gradient(pressure, pos, edge_index)

        nu_eff = nu + beta * nu_t.view(-1)

        advection = velocity[:, 0] * du_dx[:, 0]
        if velocity.shape[1] > 1:
            advection = advection + velocity[:, 1] * du_dx[:, 1]

        pressure_grad = dp_dx[:, 0]

        du_dy_grad = compute_graph_gradient(du_dx[:, -1], pos, edge_index)
        diffusion = nu_eff * du_dy_grad[:, -1]

        residual = advection + pressure_grad - diffusion
        return (residual ** 2).mean()


# ============================================================================
# Physics-Guided GNN Wrapper
# ============================================================================
class PhysicsGuidedGNN(nn.Module):
    """
    Physics-Guided GNN that wraps MeshGraphNet with physics losses.

    Combines data-driven learning (MSE on β target) with physics
    constraints from the governing RANS equations.

    Usage:
        from scripts.ml_augmentation.mesh_graphnet import MeshGraphNet
        base = MeshGraphNet(config)
        pg_gnn = PhysicsGuidedGNN(base, PhysicsGNNConfig())
        loss = pg_gnn.compute_loss(data, epoch=50)
    """

    def __init__(
        self,
        base_model: Any,
        physics_config: Optional[PhysicsGNNConfig] = None,
    ):
        _require_pyg()
        super().__init__()
        self.base_model = base_model
        self.physics_config = physics_config or PhysicsGNNConfig()

        self.continuity_loss = ContinuityLoss()
        self.realizability_loss = RealizabilityLoss(
            beta_min=self.physics_config.beta_min,
            beta_max=self.physics_config.beta_max,
        )
        self.momentum_loss = MomentumResidualLoss()

    def forward(self, x, edge_index, edge_attr, batch=None):
        """Forward pass through the base model."""
        return self.base_model(x, edge_index, edge_attr, batch)

    def forward_data(self, data):
        """Forward from PyG Data object."""
        return self.base_model.forward_data(data)

    def _get_physics_weight(self, epoch: int) -> float:
        """
        Compute physics loss weight with warmup + ramp schedule.
        Returns 0 during warmup, then ramps linearly to 1.0.
        """
        cfg = self.physics_config
        if epoch < cfg.warmup_epochs:
            return 0.0
        ramp_progress = min(
            1.0,
            (epoch - cfg.warmup_epochs) / max(cfg.ramp_epochs, 1)
        )
        return ramp_progress

    def compute_loss(
        self,
        data: Any,
        epoch: int = 0,
        velocity: Optional[Any] = None,
        pressure: Optional[Any] = None,
        nu_t: Optional[Any] = None,
        nu: float = 1.5e-5,
    ) -> Dict[str, Any]:
        """
        Compute combined data + physics loss.

        Parameters
        ----------
        data : Data
            Graph with .x, .edge_index, .edge_attr, .y (target β).
        epoch : int
            Current training epoch (for warmup scheduling).
        velocity, pressure, nu_t : Tensor, optional
            Flow fields for physics losses.
        nu : float
            Molecular viscosity.

        Returns
        -------
        dict with 'total', 'data', 'continuity', 'momentum',
        'realizability', 'smoothness' loss components.
        """
        cfg = self.physics_config
        phi = self._get_physics_weight(epoch)

        beta_pred = self.forward_data(data)
        loss_data = F.mse_loss(beta_pred, data.y)

        losses = {
            "data": loss_data,
            "continuity": torch.tensor(0.0),
            "momentum": torch.tensor(0.0),
            "realizability": torch.tensor(0.0),
            "smoothness": torch.tensor(0.0),
        }

        losses["realizability"] = self.realizability_loss(
            beta_pred, data.edge_index
        )
        losses["smoothness"] = graph_laplacian_smoothness(
            beta_pred.view(-1), data.edge_index
        )

        if velocity is not None and phi > 0:
            losses["continuity"] = self.continuity_loss(
                velocity, data.pos, data.edge_index
            )

        if all(v is not None for v in [velocity, pressure, nu_t]) and phi > 0:
            losses["momentum"] = self.momentum_loss(
                beta_pred, velocity, pressure, nu_t,
                nu, data.pos, data.edge_index
            )

        total = (
            cfg.lambda_data * losses["data"]
            + phi * cfg.lambda_realizability * losses["realizability"]
            + phi * cfg.lambda_smoothness * losses["smoothness"]
            + phi * cfg.lambda_continuity * losses["continuity"]
            + phi * cfg.lambda_momentum * losses["momentum"]
        )
        losses["total"] = total

        return losses

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# Training with Physics Losses
# ============================================================================
@dataclass
class PGGNNTrainingResult:
    """Result from Physics-Guided GNN training."""
    train_loss_history: List[float] = field(default_factory=list)
    val_loss_history: List[float] = field(default_factory=list)
    physics_loss_history: Dict[str, List[float]] = field(
        default_factory=lambda: {
            "data": [], "continuity": [], "momentum": [],
            "realizability": [], "smoothness": [],
        }
    )
    best_val_loss: float = float("inf")
    best_epoch: int = 0
    total_epochs: int = 0
    n_parameters: int = 0


def train_physics_guided_gnn(
    model: Any,
    train_graphs: List[Any],
    val_graphs: Optional[List[Any]] = None,
    physics_config: Optional[PhysicsGNNConfig] = None,
    max_epochs: int = 300,
    learning_rate: float = 1e-4,
    early_stopping_patience: int = 50,
) -> PGGNNTrainingResult:
    """
    Train PG-GNN with combined data + physics losses.

    Parameters
    ----------
    model : PhysicsGuidedGNN
    train_graphs : list of Data
    val_graphs : list of Data, optional
    max_epochs, learning_rate, early_stopping_patience : training params

    Returns
    -------
    PGGNNTrainingResult
    """
    _require_pyg()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=20, factor=0.5
    )

    result = PGGNNTrainingResult(n_parameters=model.count_parameters())
    best_state = None
    patience = 0

    for epoch in range(max_epochs):
        model.train()
        epoch_losses = {k: 0.0 for k in result.physics_loss_history}
        total_train = 0.0
        n_graphs = 0

        for data in train_graphs:
            optimizer.zero_grad()
            losses = model.compute_loss(data, epoch=epoch)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_train += losses["total"].item()
            for k in epoch_losses:
                if k in losses:
                    epoch_losses[k] += losses[k].item()
            n_graphs += 1

        avg_train = total_train / max(n_graphs, 1)
        result.train_loss_history.append(avg_train)
        for k in result.physics_loss_history:
            result.physics_loss_history[k].append(
                epoch_losses.get(k, 0.0) / max(n_graphs, 1)
            )

        # Validation
        val_loss = 0.0
        if val_graphs:
            model.eval()
            with torch.no_grad():
                for data in val_graphs:
                    losses = model.compute_loss(data, epoch=epoch)
                    val_loss += losses["data"].item()
                val_loss /= max(len(val_graphs), 1)
        result.val_loss_history.append(val_loss)

        scheduler.step(val_loss if val_graphs else avg_train)

        # Early stopping
        monitor = val_loss if val_graphs else avg_train
        if monitor < result.best_val_loss - 1e-7:
            result.best_val_loss = monitor
            result.best_epoch = epoch
            patience = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience += 1

        if patience >= early_stopping_patience:
            break

        if epoch % 50 == 0:
            logger.info(
                "PG-GNN Epoch %d: total=%.4e data=%.4e realiz=%.4e",
                epoch, avg_train,
                epoch_losses.get("data", 0) / max(n_graphs, 1),
                epoch_losses.get("realizability", 0) / max(n_graphs, 1),
            )

    if best_state:
        model.load_state_dict(best_state)

    result.total_epochs = epoch + 1
    return result


# ============================================================================
# Demo
# ============================================================================
def _demo():
    """Demonstrate Physics-Guided GNN."""
    print("=" * 65)
    print("  Physics-Guided GNN (PG-GNN) Demo")
    print("=" * 65)

    _require_pyg()

    from scripts.ml_augmentation.mesh_graph_utils import (
        generate_synthetic_mesh, build_graph_from_mesh, graph_to_pyg,
    )
    from scripts.ml_augmentation.mesh_graphnet import (
        MeshGraphNet, MeshGraphNetConfig,
    )

    # Generate graphs
    print("\n  Generating synthetic data...")
    rng = np.random.default_rng(42)
    graphs = []
    for i in range(5):
        mesh = generate_synthetic_mesh(nx=12, ny=6, seed=i)
        graph = build_graph_from_mesh(mesh)
        x = graph.node_coords[:, 0]
        beta = np.ones(graph.n_nodes, dtype=np.float32)
        sep = (x > 0.8) & (x < 1.4)
        beta[sep] = 1.2
        pyg_data = graph_to_pyg(graph, target=beta)
        graphs.append(pyg_data)

    sample = graphs[0]
    print(f"  {len(graphs)} graphs, {sample.num_nodes} nodes each")

    # Build models
    mgn_config = MeshGraphNetConfig(
        node_in_dim=sample.x.shape[1],
        edge_in_dim=sample.edge_attr.shape[1],
        latent_dim=32, n_message_passing=3,
        output_dim=1,
        encoder_hidden=(32,), decoder_hidden=(32,),
    )
    base = MeshGraphNet(mgn_config)
    pg_config = PhysicsGNNConfig(
        lambda_realizability=0.5,
        lambda_smoothness=0.01,
        warmup_epochs=5,
        ramp_epochs=10,
    )
    pg_gnn = PhysicsGuidedGNN(base, pg_config)
    print(f"  Parameters: {pg_gnn.count_parameters():,}")

    # Train
    print("\n  Training PG-GNN...")
    result = train_physics_guided_gnn(
        pg_gnn, graphs[:4], graphs[4:],
        max_epochs=30, early_stopping_patience=15,
    )
    print(f"  Epochs: {result.total_epochs}")
    print(f"  Best val loss: {result.best_val_loss:.6e}")

    # Predict
    pg_gnn.eval()
    with torch.no_grad():
        pred = pg_gnn.forward_data(graphs[4])
    print(f"  β prediction: [{pred.min():.3f}, {pred.max():.3f}]")

    print(f"\n{'=' * 65}")
    print("  Demo complete!")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _demo()
