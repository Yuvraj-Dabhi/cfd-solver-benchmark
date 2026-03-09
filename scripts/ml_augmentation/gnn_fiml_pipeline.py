#!/usr/bin/env python3
"""
GNN-Based FIML Pipeline for β-Correction Prediction
======================================================
Drop-in replacement for ``FIMLPipeline`` (fiml_pipeline.py) that uses
Graph Neural Networks instead of MLPs, operating directly on SU2
unstructured meshes as graphs.

Key Advantages over MLP Pipeline:
  - Preserves mesh connectivity / topology information
  - Generalizes across mesh refinements via message passing
  - Captures non-local flow interactions through multi-hop edges
  - Enables the Generalized Field Inversion strategy (Srivastava
    et al., AIAA SciTech 2024)

Usage:
    pipeline = GNNFIMLPipeline()
    pipeline.add_case("wall_hump", mesh_path, beta_target, solution)
    result = pipeline.train(test_case="wall_hump")
    beta_pred = pipeline.predict(test_graph)

References:
  - Srivastava et al. (2024), AIAA SciTech 2024: generalized field
    inversion for RANS separation predictions
  - Pfaff et al. (2021), ICML: MeshGraphNet
  - Parish & Duraisamy (2016), JCP: FIML framework
"""

import json
import logging
import time
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
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import torch_geometric
    from torch_geometric.data import Data
    HAS_TORCH_GEOMETRIC = True
except ImportError:
    HAS_TORCH_GEOMETRIC = False

# Local imports (deferred to avoid circular imports at module level)


def _require_deps():
    if not HAS_TORCH or not HAS_TORCH_GEOMETRIC:
        raise ImportError(
            "GNN FIML Pipeline requires PyTorch and PyTorch Geometric. "
            "Install: pip install torch torch-geometric"
        )


# ============================================================================
# Data Structures
# ============================================================================
@dataclass
class GNNCaseData:
    """Data for a single case in the GNN FIML pipeline."""
    name: str
    graph: Any                       # GraphData or PyG Data
    beta_target: np.ndarray          # (N,) correction factor
    n_nodes: int = 0
    n_edges: int = 0
    cf_baseline: Optional[np.ndarray] = None
    cf_experimental: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GNNFIMLResult:
    """Result from GNN FIML training/evaluation."""
    model_type: str = "meshgraphnet"
    train_cases: List[str] = field(default_factory=list)
    test_cases: List[str] = field(default_factory=list)
    train_rmse: float = 0.0
    test_rmse: float = 0.0
    train_r2: float = 0.0
    test_r2: float = 0.0
    n_parameters: int = 0
    n_message_passing: int = 0
    training_epochs: int = 0
    training_time_s: float = 0.0
    cf_improvement: Dict[str, float] = field(default_factory=dict)
    summary: str = ""


# ============================================================================
# GNN FIML Pipeline
# ============================================================================
class GNNFIMLPipeline:
    """
    End-to-end GNN-based FIML training pipeline.

    Mirrors the ``FIMLPipeline`` API but uses MeshGraphNet to predict
    β(x) at each mesh node, using graph structure instead of fixed-grid
    MLP evaluation.

    Usage:
        pipeline = GNNFIMLPipeline()
        pipeline.add_case_from_mesh("hump", mesh_path, beta_target)
        result = pipeline.train(test_case="hump")
        beta_pred = pipeline.predict(test_graph)
    """

    def __init__(
        self,
        latent_dim: int = 128,
        n_message_passing: int = 10,
        encoder_hidden: Tuple[int, ...] = (128,),
        decoder_hidden: Tuple[int, ...] = (128,),
        max_epochs: int = 300,
        learning_rate: float = 1e-4,
        use_physics_loss: bool = False,
        physics_lambda_realiz: float = 0.5,
        physics_lambda_smooth: float = 0.01,
    ):
        self.latent_dim = latent_dim
        self.n_message_passing = n_message_passing
        self.encoder_hidden = encoder_hidden
        self.decoder_hidden = decoder_hidden
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.use_physics_loss = use_physics_loss
        self.physics_lambda_realiz = physics_lambda_realiz
        self.physics_lambda_smooth = physics_lambda_smooth

        self.cases: Dict[str, GNNCaseData] = {}
        self.model = None
        self._trained = False
        self._node_in_dim = None
        self._edge_in_dim = None

    def add_case_from_mesh(
        self,
        name: str,
        mesh_path: Union[str, Path],
        beta_target: np.ndarray,
        solution_fields: Optional[Dict[str, np.ndarray]] = None,
        cf_baseline: Optional[np.ndarray] = None,
        cf_experimental: Optional[np.ndarray] = None,
    ):
        """
        Add a case by loading an SU2 mesh file.

        Parameters
        ----------
        name : str
            Case name (e.g., "wall_hump", "periodic_hill").
        mesh_path : str or Path
            Path to ``.su2`` mesh file.
        beta_target : ndarray (N,)
            Optimal β correction from field inversion.
        solution_fields : dict, optional
            Extra node fields (Cp, nu_t, velocity, etc.).
        cf_baseline : ndarray, optional
            Baseline RANS Cf at wall.
        cf_experimental : ndarray, optional
            Experimental/DNS Cf at wall.
        """
        from scripts.ml_augmentation.mesh_graph_utils import (
            load_su2_mesh_as_graph, augment_graph_with_solution, graph_to_pyg,
        )

        graph = load_su2_mesh_as_graph(mesh_path)
        if solution_fields:
            graph = augment_graph_with_solution(graph, solution_fields)

        pyg_data = graph_to_pyg(graph, target=beta_target)

        self.cases[name] = GNNCaseData(
            name=name,
            graph=pyg_data,
            beta_target=beta_target,
            n_nodes=graph.n_nodes,
            n_edges=graph.n_edges,
            cf_baseline=cf_baseline,
            cf_experimental=cf_experimental,
            metadata=graph.metadata,
        )
        self._node_in_dim = pyg_data.x.shape[1]
        self._edge_in_dim = pyg_data.edge_attr.shape[1]

        logger.info(
            "Added GNN case '%s': %d nodes, %d edges",
            name, graph.n_nodes, graph.n_edges,
        )

    def add_case_from_graph(
        self,
        name: str,
        pyg_data: Any,
        beta_target: np.ndarray,
        cf_baseline: Optional[np.ndarray] = None,
        cf_experimental: Optional[np.ndarray] = None,
    ):
        """
        Add a case from a pre-built PyG Data object.

        Parameters
        ----------
        name : str
            Case name.
        pyg_data : Data
            PyG Data with x, edge_index, edge_attr, y.
        beta_target : ndarray
            Target β field.
        """
        _require_deps()

        self.cases[name] = GNNCaseData(
            name=name,
            graph=pyg_data,
            beta_target=beta_target,
            n_nodes=pyg_data.num_nodes,
            n_edges=pyg_data.edge_index.shape[1],
            cf_baseline=cf_baseline,
            cf_experimental=cf_experimental,
        )
        self._node_in_dim = pyg_data.x.shape[1]
        self._edge_in_dim = pyg_data.edge_attr.shape[1]

    def _build_model(self):
        """Build MeshGraphNet with current configuration."""
        from scripts.ml_augmentation.mesh_graphnet import (
            MeshGraphNet, MeshGraphNetConfig,
        )

        config = MeshGraphNetConfig(
            node_in_dim=self._node_in_dim,
            edge_in_dim=self._edge_in_dim,
            latent_dim=self.latent_dim,
            n_message_passing=self.n_message_passing,
            output_dim=1,
            encoder_hidden=self.encoder_hidden,
            decoder_hidden=self.decoder_hidden,
            max_epochs=self.max_epochs,
            learning_rate=self.learning_rate,
        )
        model = MeshGraphNet(config)

        if self.use_physics_loss:
            from scripts.ml_augmentation.physics_guided_gnn import (
                PhysicsGuidedGNN, PhysicsGNNConfig,
            )
            pg_config = PhysicsGNNConfig(
                lambda_realizability=self.physics_lambda_realiz,
                lambda_smoothness=self.physics_lambda_smooth,
            )
            model = PhysicsGuidedGNN(model, pg_config)

        return model

    def train(
        self,
        test_case: Optional[str] = None,
        train_cases: Optional[List[str]] = None,
    ) -> GNNFIMLResult:
        """
        Train the GNN FIML correction model.

        Parameters
        ----------
        test_case : str, optional
            Case to exclude from training (leave-one-case-out).
        train_cases : list of str, optional
            Explicit training cases.

        Returns
        -------
        GNNFIMLResult
        """
        _require_deps()

        all_cases = list(self.cases.keys())
        if not all_cases:
            raise ValueError("No cases added. Use add_case_from_*() first.")

        if train_cases is None:
            if test_case:
                train_cases = [c for c in all_cases if c != test_case]
            else:
                train_cases = all_cases
        test_cases = [test_case] if test_case else []

        # Build model
        self.model = self._build_model()

        # Assemble graphs
        train_graphs = [self.cases[n].graph for n in train_cases]
        val_graphs = [self.cases[n].graph for n in test_cases] if test_cases else None

        # Train
        t0 = time.time()

        if self.use_physics_loss:
            from scripts.ml_augmentation.physics_guided_gnn import (
                train_physics_guided_gnn,
            )
            train_result = train_physics_guided_gnn(
                self.model, train_graphs, val_graphs,
                max_epochs=self.max_epochs,
                learning_rate=self.learning_rate,
            )
        else:
            from scripts.ml_augmentation.mesh_graphnet import (
                train_meshgraphnet,
            )
            train_result = train_meshgraphnet(
                self.model, train_graphs, val_graphs,
            )

        training_time = time.time() - t0
        self._trained = True

        # Evaluate
        result = GNNFIMLResult(
            model_type="pg_gnn" if self.use_physics_loss else "meshgraphnet",
            train_cases=train_cases,
            test_cases=test_cases,
            n_parameters=self.model.count_parameters()
                if hasattr(self.model, 'count_parameters')
                else sum(p.numel() for p in self.model.parameters()),
            n_message_passing=self.n_message_passing,
            training_epochs=train_result.total_epochs,
            training_time_s=training_time,
        )

        # Train metrics
        result.train_rmse = self._compute_rmse(train_cases)
        result.train_r2 = self._compute_r2(train_cases)

        # Test metrics
        if test_cases:
            result.test_rmse = self._compute_rmse(test_cases)
            result.test_r2 = self._compute_r2(test_cases)
            for tc in test_cases:
                imp = self._compute_cf_improvement(tc)
                if imp is not None:
                    result.cf_improvement[tc] = imp

        result.summary = (
            f"GNN-FIML ({result.model_type}): "
            f"R²_train={result.train_r2:.4f}, R²_test={result.test_r2:.4f}, "
            f"{result.n_parameters:,} params, {result.training_epochs} epochs"
        )
        logger.info(result.summary)
        return result

    def predict(self, graph_data: Any) -> np.ndarray:
        """
        Predict β correction field for a new mesh graph.

        Parameters
        ----------
        graph_data : Data
            PyG Data with x, edge_index, edge_attr.

        Returns
        -------
        ndarray (N,) — predicted β at each mesh node.
        """
        if not self._trained:
            raise RuntimeError("Pipeline not trained. Call train() first.")

        self.model.eval()
        with torch.no_grad():
            if hasattr(self.model, 'forward_data'):
                pred = self.model.forward_data(graph_data)
            else:
                pred = self.model(
                    graph_data.x, graph_data.edge_index, graph_data.edge_attr
                )
        return pred.cpu().numpy().flatten()

    def _compute_rmse(self, case_names: List[str]) -> float:
        """Compute RMSE over specified cases."""
        errors = []
        self.model.eval()
        with torch.no_grad():
            for name in case_names:
                case = self.cases[name]
                pred = self.predict(case.graph)
                errors.append(
                    np.sqrt(np.mean((pred - case.beta_target) ** 2))
                )
        return float(np.mean(errors))

    def _compute_r2(self, case_names: List[str]) -> float:
        """Compute R² over specified cases."""
        all_true, all_pred = [], []
        self.model.eval()
        with torch.no_grad():
            for name in case_names:
                case = self.cases[name]
                pred = self.predict(case.graph)
                all_true.append(case.beta_target)
                all_pred.append(pred)
        y_true = np.concatenate(all_true)
        y_pred = np.concatenate(all_pred)

        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - y_true.mean()) ** 2)
        return float(1 - ss_res / max(ss_tot, 1e-15))

    def _compute_cf_improvement(self, case_name: str) -> Optional[float]:
        """Compute fractional Cf improvement."""
        case = self.cases.get(case_name)
        if case is None or case.cf_baseline is None or case.cf_experimental is None:
            return None

        n_wall = min(len(case.cf_baseline), len(case.cf_experimental))
        cf_base = case.cf_baseline[:n_wall]
        cf_exp = case.cf_experimental[:n_wall]
        err_base = np.sqrt(np.mean((cf_base - cf_exp) ** 2))

        pred = self.predict(case.graph)
        beta_wall = pred[:n_wall] if len(pred) >= n_wall else pred
        if len(beta_wall) < n_wall:
            return None
        cf_corrected = cf_base * beta_wall
        err_corrected = np.sqrt(np.mean((cf_corrected - cf_exp) ** 2))

        if err_base > 1e-10:
            return float((err_base - err_corrected) / err_base)
        return 0.0

    def cross_validate(self) -> Dict[str, GNNFIMLResult]:
        """
        Leave-one-case-out cross-validation.

        Returns dict mapping test case name → GNNFIMLResult.
        """
        results = {}
        for test_case in self.cases:
            logger.info("GNN CV: holding out '%s'", test_case)
            result = self.train(test_case=test_case)
            results[test_case] = result
        return results

    def save(self, output_dir: Path):
        """Save trained model and metadata."""
        _require_deps()
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if self._trained and self.model is not None:
            torch.save(
                self.model.state_dict(),
                output_dir / "gnn_fiml_model.pth",
            )
        meta = {
            "model_type": "meshgraphnet",
            "latent_dim": self.latent_dim,
            "n_message_passing": self.n_message_passing,
            "use_physics_loss": self.use_physics_loss,
            "cases": list(self.cases.keys()),
            "node_in_dim": self._node_in_dim,
            "edge_in_dim": self._edge_in_dim,
        }
        with open(output_dir / "gnn_fiml_meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        logger.info("GNN FIML pipeline saved to %s", output_dir)


# ============================================================================
# Comparison: GNN vs MLP
# ============================================================================
def compare_gnn_vs_mlp(
    gnn_result: GNNFIMLResult,
    mlp_result: Any,
) -> Dict[str, Any]:
    """
    Compare GNN-FIML vs MLP-FIML results.

    Parameters
    ----------
    gnn_result : GNNFIMLResult
    mlp_result : FIMLResult (from fiml_pipeline.py)

    Returns
    -------
    dict with comparison metrics.
    """
    comparison = {
        "gnn_test_r2": gnn_result.test_r2,
        "mlp_test_r2": getattr(mlp_result, "test_r2", 0.0),
        "gnn_test_rmse": gnn_result.test_rmse,
        "mlp_test_rmse": getattr(mlp_result, "test_rmse", 0.0),
        "gnn_params": gnn_result.n_parameters,
        "mlp_params": 0,  # MLP doesn't track this
        "r2_improvement": gnn_result.test_r2 - getattr(mlp_result, "test_r2", 0.0),
        "rmse_improvement_pct": 0.0,
    }
    mlp_rmse = getattr(mlp_result, "test_rmse", 0.0)
    if mlp_rmse > 1e-10:
        comparison["rmse_improvement_pct"] = (
            (mlp_rmse - gnn_result.test_rmse) / mlp_rmse * 100
        )
    return comparison


# ============================================================================
# Demo with Synthetic Data
# ============================================================================
def _demo():
    """Demonstrate the GNN FIML pipeline with synthetic data."""
    print("=" * 65)
    print("  GNN-Based FIML Pipeline Demo")
    print("=" * 65)

    _require_deps()

    from scripts.ml_augmentation.mesh_graph_utils import (
        generate_synthetic_mesh, build_graph_from_mesh, graph_to_pyg,
    )

    # Generate synthetic cases
    print("\n  Generating synthetic mesh cases...")
    rng = np.random.default_rng(42)
    pipeline = GNNFIMLPipeline(
        latent_dim=64,
        n_message_passing=5,
        encoder_hidden=(64,),
        decoder_hidden=(64,),
        max_epochs=50,
        learning_rate=1e-3,
    )

    case_names = ["hump_like", "bump_like", "bfs_like"]
    for i, name in enumerate(case_names):
        mesh = generate_synthetic_mesh(nx=15, ny=8, seed=i * 10)
        graph = build_graph_from_mesh(mesh)

        # Synthetic β
        x = graph.node_coords[:, 0]
        beta = np.ones(graph.n_nodes, dtype=np.float32)
        sep = (x > 0.8) & (x < 1.4)
        beta[sep] = 1.0 + 0.3 * np.sin(np.pi * (x[sep] - 0.8) / 0.6)
        beta += rng.normal(0, 0.02, graph.n_nodes).astype(np.float32)

        pyg_data = graph_to_pyg(graph, target=beta)
        pipeline.add_case_from_graph(name, pyg_data, beta)
        print(f"  Case '{name}': {graph.n_nodes} nodes, {graph.n_edges} edges")

    # Train with leave-one-out
    print("\n  Training (hold out 'bfs_like')...")
    result = pipeline.train(test_case="bfs_like")
    print(f"  Model: {result.model_type}")
    print(f"  Parameters: {result.n_parameters:,}")
    print(f"  Epochs: {result.training_epochs}")
    print(f"  Train R²: {result.train_r2:.4f}")
    print(f"  Test R²:  {result.test_r2:.4f}")
    print(f"  Train RMSE: {result.train_rmse:.6f}")
    print(f"  Test RMSE:  {result.test_rmse:.6f}")
    print(f"  Time: {result.training_time_s:.1f}s")

    # Predict
    test_graph = pipeline.cases["bfs_like"].graph
    beta_pred = pipeline.predict(test_graph)
    print(f"\n  Prediction: β ∈ [{beta_pred.min():.4f}, {beta_pred.max():.4f}]")

    print(f"\n{'=' * 65}")
    print("  Demo complete!")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _demo()
