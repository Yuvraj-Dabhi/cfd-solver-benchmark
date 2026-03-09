#!/usr/bin/env python3
"""
MeshGraphNet for CFD Flow Prediction on Unstructured Meshes
=============================================================
Implements the encoder–processor–decoder Graph Neural Network
architecture from Pfaff et al. (2021) for learning mesh-based
simulations directly on unstructured finite-volume grids.

Architecture:
  Encoder:  node/edge MLPs → latent space (128-d default)
  Processor: K message-passing layers with residual + LayerNorm
  Decoder:  MLP → output field (β, Cp, Cf, or full state)

Multi-scale option (AMGNET-inspired):
  Graph coarsening via edge-contraction pooling for large meshes
  (millions of nodes), with hierarchical message passing.

References:
  - Pfaff et al. (2021), "Learning Mesh-Based Simulation with Graph
    Networks", ICML 2021 (MeshGraphNet, Google DeepMind)
  - Chen et al. (2021), "Graph Neural Networks for Laminar Flow
    Prediction around Random 2D Shapes", PoF
  - AMGNET: Multi-scale GNN validated on airfoil/cylinder flow
"""

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

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
    # Provide a stub so class definitions below don't fail at parse time
    class _ModuleStub:
        Module = object
    nn = _ModuleStub()

try:
    import torch_geometric
    from torch_geometric.nn import MessagePassing as _MessagePassing
    from torch_geometric.data import Data, Batch
    HAS_TORCH_GEOMETRIC = True
except ImportError:
    HAS_TORCH_GEOMETRIC = False
    _MessagePassing = object  # Stub for class inheritance


def _require_pyg():
    if not HAS_TORCH or not HAS_TORCH_GEOMETRIC:
        raise ImportError(
            "MeshGraphNet requires PyTorch and PyTorch Geometric. "
            "Install with: pip install torch torch-geometric"
        )


# ============================================================================
# Configuration
# ============================================================================
@dataclass
class MeshGraphNetConfig:
    """Configuration for MeshGraphNet architecture."""

    # Input dimensions
    node_in_dim: int = 4          # (x, y, wall_dist, boundary_flag) or custom
    edge_in_dim: int = 5          # (dx, dy, ||dx||, dir_x, dir_y) for 2D

    # Latent dimensions
    latent_dim: int = 128
    n_message_passing: int = 15   # Number of processor blocks

    # Output
    output_dim: int = 1           # e.g., β field

    # Architecture details
    encoder_hidden: Tuple[int, ...] = (128,)
    decoder_hidden: Tuple[int, ...] = (128,)
    use_layer_norm: bool = True
    dropout: float = 0.0

    # Training
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    max_epochs: int = 500
    batch_size: int = 1           # Graphs per batch
    scheduler_patience: int = 20
    scheduler_factor: float = 0.5
    early_stopping_patience: int = 50
    gradient_clip: float = 1.0

    # Multi-scale (AMGNET-inspired)
    use_multiscale: bool = False
    coarsen_ratio: float = 0.5    # Fraction of nodes to keep


# ============================================================================
# Building Blocks
# ============================================================================
class MLPBlock(nn.Module):
    """Multi-layer perceptron with optional LayerNorm and dropout."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dims: Tuple[int, ...] = (),
        activation: str = "relu",
        use_layer_norm: bool = True,
        dropout: float = 0.0,
    ):
        _require_pyg()
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "gelu":
                layers.append(nn.GELU())
            elif activation == "silu":
                layers.append(nn.SiLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        if use_layer_norm:
            layers.append(nn.LayerNorm(out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class GraphNetworkBlock(_MessagePassing):
    """
    Single message-passing block of MeshGraphNet.

    Edge update:  e'_ij = MLP([e_ij || h_i || h_j])
    Node update:  h'_i  = MLP([h_i || Σ_j e'_ij])       (residual)
    """

    def __init__(
        self,
        latent_dim: int,
        edge_latent_dim: int,
        use_layer_norm: bool = True,
        dropout: float = 0.0,
    ):
        _require_pyg()
        super().__init__(aggr="sum")
        self.latent_dim = latent_dim

        # Edge update MLP: (e_ij, h_i, h_j) → e'_ij
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_latent_dim + 2 * latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, edge_latent_dim),
        )
        if use_layer_norm:
            self.edge_norm = nn.LayerNorm(edge_latent_dim)
        else:
            self.edge_norm = nn.Identity()

        # Node update MLP: (h_i, agg) → h'_i
        self.node_mlp = nn.Sequential(
            nn.Linear(latent_dim + edge_latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
        )
        if use_layer_norm:
            self.node_norm = nn.LayerNorm(latent_dim)
        else:
            self.node_norm = nn.Identity()

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x, edge_index, edge_attr):
        """
        Parameters
        ----------
        x : Tensor (N, latent_dim)
        edge_index : Tensor (2, E)
        edge_attr : Tensor (E, edge_latent_dim)

        Returns
        -------
        x_new : Tensor (N, latent_dim)
        edge_attr_new : Tensor (E, edge_latent_dim)
        """
        # Edge update
        row, col = edge_index
        edge_input = torch.cat([edge_attr, x[row], x[col]], dim=-1)
        edge_attr_new = self.edge_norm(
            edge_attr + self.dropout(self.edge_mlp(edge_input))
        )

        # Node update via message passing
        agg = self.propagate(edge_index, edge_attr=edge_attr_new)
        node_input = torch.cat([x, agg], dim=-1)
        x_new = self.node_norm(
            x + self.dropout(self.node_mlp(node_input))
        )

        return x_new, edge_attr_new

    def message(self, edge_attr):
        return edge_attr

    def update(self, aggr_out):
        return aggr_out


# ============================================================================
# MeshGraphNet Model
# ============================================================================
class MeshGraphNet(nn.Module):
    """
    MeshGraphNet: Encoder–Processor–Decoder GNN for mesh-based
    simulation.

    Operates directly on unstructured CFD meshes:
      - Encoder: maps raw node/edge features to latent space
      - Processor: K message-passing blocks with residual connections
      - Decoder: maps latent node features to output field

    Parameters
    ----------
    config : MeshGraphNetConfig
        Architecture and training configuration.
    """

    def __init__(self, config: MeshGraphNetConfig = None):
        _require_pyg()
        super().__init__()
        self.config = config or MeshGraphNetConfig()
        c = self.config

        # --- Encoder ---
        self.node_encoder = MLPBlock(
            c.node_in_dim, c.latent_dim,
            hidden_dims=c.encoder_hidden,
            use_layer_norm=c.use_layer_norm,
            dropout=c.dropout,
        )
        self.edge_encoder = MLPBlock(
            c.edge_in_dim, c.latent_dim,
            hidden_dims=c.encoder_hidden,
            use_layer_norm=c.use_layer_norm,
            dropout=c.dropout,
        )

        # --- Processor (K message-passing blocks) ---
        self.processor = nn.ModuleList([
            GraphNetworkBlock(
                c.latent_dim, c.latent_dim,
                use_layer_norm=c.use_layer_norm,
                dropout=c.dropout,
            )
            for _ in range(c.n_message_passing)
        ])

        # --- Decoder ---
        self.decoder = MLPBlock(
            c.latent_dim, c.output_dim,
            hidden_dims=c.decoder_hidden,
            use_layer_norm=False,  # No norm on final output
            dropout=0.0,
        )

    def forward(
        self,
        x: Any,
        edge_index: Any,
        edge_attr: Any,
        batch: Optional[Any] = None,
    ) -> Any:
        """
        Forward pass.

        Parameters
        ----------
        x : Tensor (N, node_in_dim)
            Raw node features.
        edge_index : Tensor (2, E)
            Edge connectivity (COO format).
        edge_attr : Tensor (E, edge_in_dim)
            Raw edge features.
        batch : Tensor (N,), optional
            Batch assignment for batched graphs.

        Returns
        -------
        Tensor (N, output_dim) — predicted field at each node.
        """
        # Encode
        h = self.node_encoder(x)
        e = self.edge_encoder(edge_attr)

        # Process (message passing)
        for block in self.processor:
            h, e = block(h, edge_index, e)

        # Decode
        out = self.decoder(h)
        return out

    def forward_data(self, data: Any) -> Any:
        """
        Forward pass from a PyG ``Data`` object.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Graph data with ``x``, ``edge_index``, ``edge_attr``.
        """
        return self.forward(
            data.x, data.edge_index, data.edge_attr,
            batch=getattr(data, "batch", None),
        )

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# Training Utilities
# ============================================================================
@dataclass
class TrainingResult:
    """Result from MeshGraphNet training."""
    train_loss_history: List[float] = field(default_factory=list)
    val_loss_history: List[float] = field(default_factory=list)
    best_val_loss: float = float("inf")
    best_epoch: int = 0
    total_epochs: int = 0
    n_parameters: int = 0
    final_train_rmse: float = 0.0
    final_val_rmse: float = 0.0


def train_meshgraphnet(
    model: Any,
    train_graphs: List[Any],
    val_graphs: Optional[List[Any]] = None,
    config: Optional[MeshGraphNetConfig] = None,
) -> TrainingResult:
    """
    Train a MeshGraphNet model on a list of PyG ``Data`` graphs.

    Parameters
    ----------
    model : MeshGraphNet
        The model to train.
    train_graphs : list of Data
        Training graphs with ``.y`` target fields.
    val_graphs : list of Data, optional
        Validation graphs.
    config : MeshGraphNetConfig, optional
        Training config (uses model.config if not provided).

    Returns
    -------
    TrainingResult
    """
    _require_pyg()
    c = config or model.config

    optimizer = torch.optim.Adam(
        model.parameters(), lr=c.learning_rate, weight_decay=c.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=c.scheduler_patience, factor=c.scheduler_factor
    )
    criterion = nn.MSELoss()
    result = TrainingResult(n_parameters=model.count_parameters())

    best_state = None
    patience_counter = 0

    for epoch in range(c.max_epochs):
        # --- Train ---
        model.train()
        total_loss = 0.0
        n_nodes_total = 0

        for data in train_graphs:
            optimizer.zero_grad()
            pred = model.forward_data(data)
            loss = criterion(pred, data.y)
            loss.backward()
            if c.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), c.gradient_clip
                )
            optimizer.step()
            total_loss += loss.item() * data.num_nodes
            n_nodes_total += data.num_nodes

        train_loss = total_loss / max(n_nodes_total, 1)
        result.train_loss_history.append(train_loss)

        # --- Validate ---
        val_loss = 0.0
        if val_graphs:
            model.eval()
            with torch.no_grad():
                val_total = 0.0
                val_nodes = 0
                for data in val_graphs:
                    pred = model.forward_data(data)
                    vloss = criterion(pred, data.y)
                    val_total += vloss.item() * data.num_nodes
                    val_nodes += data.num_nodes
                val_loss = val_total / max(val_nodes, 1)
        result.val_loss_history.append(val_loss)

        scheduler.step(val_loss if val_graphs else train_loss)

        # --- Early stopping ---
        monitor = val_loss if val_graphs else train_loss
        if monitor < result.best_val_loss - 1e-7:
            result.best_val_loss = monitor
            result.best_epoch = epoch
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if patience_counter >= c.early_stopping_patience:
            logger.info("Early stopping at epoch %d", epoch)
            break

        if epoch % 50 == 0:
            logger.info(
                "Epoch %d: train_loss=%.6e, val_loss=%.6e",
                epoch, train_loss, val_loss,
            )

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    result.total_epochs = epoch + 1
    result.final_train_rmse = np.sqrt(
        result.train_loss_history[-1]
    ) if result.train_loss_history else 0.0
    result.final_val_rmse = np.sqrt(
        result.val_loss_history[-1]
    ) if result.val_loss_history else 0.0

    logger.info(
        "Training complete: %d epochs, best_val=%.6e (epoch %d), "
        "%d parameters",
        result.total_epochs, result.best_val_loss, result.best_epoch,
        result.n_parameters,
    )
    return result


# ============================================================================
# Demo / Main
# ============================================================================
def _demo():
    """Demonstrate MeshGraphNet on synthetic mesh data."""
    print("=" * 65)
    print("  MeshGraphNet — Encoder-Processor-Decoder GNN Demo")
    print("=" * 65)

    _require_pyg()

    from scripts.ml_augmentation.mesh_graph_utils import (
        generate_synthetic_mesh, build_graph_from_mesh, graph_to_pyg,
    )

    # 1. Generate synthetic graphs
    print("\n  Generating synthetic mesh graphs...")
    graphs = []
    rng = np.random.default_rng(42)
    for i in range(5):
        mesh = generate_synthetic_mesh(nx=15, ny=8, seed=i)
        graph = build_graph_from_mesh(mesh)

        # Synthetic β target: elevated in "separation" region
        x = graph.node_coords[:, 0]
        beta = np.ones(graph.n_nodes, dtype=np.float32)
        sep = (x > 0.8) & (x < 1.4)
        beta[sep] = 1.0 + 0.3 * np.sin(
            np.pi * (x[sep] - 0.8) / 0.6
        )
        beta += rng.normal(0, 0.02, graph.n_nodes).astype(np.float32)

        pyg_data = graph_to_pyg(graph, target=beta)
        graphs.append(pyg_data)

    sample = graphs[0]
    print(f"  Graphs: {len(graphs)}")
    print(f"  Sample: {sample.num_nodes} nodes, "
          f"{sample.edge_index.shape[1]} edges")
    print(f"  Node features: {sample.x.shape[1]}")
    print(f"  Edge features: {sample.edge_attr.shape[1]}")

    # 2. Build model
    config = MeshGraphNetConfig(
        node_in_dim=sample.x.shape[1],
        edge_in_dim=sample.edge_attr.shape[1],
        latent_dim=64,
        n_message_passing=5,
        output_dim=1,
        encoder_hidden=(64,),
        decoder_hidden=(64,),
        max_epochs=50,
        early_stopping_patience=20,
        learning_rate=1e-3,
    )
    model = MeshGraphNet(config)
    print(f"\n  Model parameters: {model.count_parameters():,}")

    # 3. Train
    print("\n  Training...")
    train_graphs = graphs[:4]
    val_graphs = graphs[4:]
    result = train_meshgraphnet(model, train_graphs, val_graphs, config)
    print(f"  Epochs: {result.total_epochs}")
    print(f"  Best val loss: {result.best_val_loss:.6e}")
    print(f"  Final train RMSE: {result.final_train_rmse:.6f}")
    print(f"  Final val RMSE: {result.final_val_rmse:.6f}")

    # 4. Predict
    model.eval()
    with torch.no_grad():
        pred = model.forward_data(val_graphs[0])
    print(f"\n  Prediction shape: {pred.shape}")
    print(f"  β range: [{pred.min().item():.4f}, {pred.max().item():.4f}]")

    print(f"\n{'=' * 65}")
    print("  Demo complete!")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _demo()
