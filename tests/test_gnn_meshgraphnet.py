#!/usr/bin/env python3
"""
Tests for GNN-Based Mesh Prediction Modules
=============================================
Comprehensive tests for the Graph Neural Network extension:
  - SU2 mesh → graph conversion (mesh_graph_utils)
  - MeshGraphNet model architecture (mesh_graphnet)
  - Physics-Guided GNN losses (physics_guided_gnn)
  - GNN FIML pipeline (gnn_fiml_pipeline)

All tests use synthetic meshes (small 2D triangular grids, ~100–200
nodes) so they run in seconds without real SU2 data or GPU.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =========================================================================
# Skip marker if torch-geometric not installed
# =========================================================================
try:
    import torch
    import torch_geometric
    from torch_geometric.data import Data
    HAS_PYG = True
except ImportError:
    HAS_PYG = False

requires_pyg = pytest.mark.skipif(
    not HAS_PYG,
    reason="torch-geometric not installed",
)


# =========================================================================
# TestMeshGraphUtils — SU2→Graph conversion
# =========================================================================
class TestMeshGraphUtils:
    """Tests for SU2 mesh → graph conversion utilities."""

    def test_generate_synthetic_mesh(self):
        """Synthetic mesh should have correct dimensions."""
        from scripts.ml_augmentation.mesh_graph_utils import (
            generate_synthetic_mesh,
        )
        mesh = generate_synthetic_mesh(nx=10, ny=5)
        assert mesh.n_points == 50
        assert mesh.coords.shape == (50, 2)
        assert mesh.n_dim == 2
        assert len(mesh.elements) > 0
        assert "wall" in mesh.boundary_markers

    def test_mesh_elements_are_triangles(self):
        """All elements should be triangles (type 5)."""
        from scripts.ml_augmentation.mesh_graph_utils import (
            generate_synthetic_mesh,
        )
        mesh = generate_synthetic_mesh(nx=8, ny=4)
        for elem in mesh.elements:
            assert elem[0] == 5  # Triangle VTK type
            assert len(elem) == 4  # (type, n0, n1, n2)

    def test_boundary_node_identification(self):
        """Wall boundary nodes should be identified."""
        from scripts.ml_augmentation.mesh_graph_utils import (
            generate_synthetic_mesh,
        )
        mesh = generate_synthetic_mesh(nx=10, ny=5)
        assert len(mesh.boundary_node_ids) > 0
        # Wall nodes are at y=0 (indices 0, 5, 10, ...)
        for nid in mesh.boundary_node_ids:
            assert nid < mesh.n_points

    def test_build_graph_edges(self):
        """Graph should have edges from element connectivity."""
        from scripts.ml_augmentation.mesh_graph_utils import (
            generate_synthetic_mesh, build_graph_from_mesh,
        )
        mesh = generate_synthetic_mesh(nx=8, ny=4)
        graph = build_graph_from_mesh(mesh)
        assert graph.n_nodes == 32
        assert graph.n_edges > 0
        # Each edge should be bidirectional
        assert graph.n_edges % 2 == 0
        # Edge index shape
        assert graph.edge_index.shape[0] == 2
        assert graph.edge_index.shape[1] == graph.n_edges

    def test_edge_features_shape(self):
        """Edge features should include displacement + distance + direction."""
        from scripts.ml_augmentation.mesh_graph_utils import (
            generate_synthetic_mesh, build_graph_from_mesh,
        )
        mesh = generate_synthetic_mesh(nx=8, ny=4)
        graph = build_graph_from_mesh(mesh)
        # For 2D: dx(2) + dist(1) + direction(2) = 5
        assert graph.edge_features.shape == (graph.n_edges, 5)
        # Distances should be positive
        dists = graph.edge_features[:, 2]
        assert np.all(dists > 0)

    def test_node_features_shape(self):
        """Node features should include position + wall_dist + boundary."""
        from scripts.ml_augmentation.mesh_graph_utils import (
            generate_synthetic_mesh, build_graph_from_mesh,
        )
        mesh = generate_synthetic_mesh(nx=8, ny=4)
        graph = build_graph_from_mesh(mesh)
        # position(2) + wall_dist(1) + boundary(1) = 4
        assert graph.node_features.shape == (32, 4)

    def test_wall_distance_computation(self):
        """Wall distance should be 0 at wall and positive elsewhere."""
        from scripts.ml_augmentation.mesh_graph_utils import (
            generate_synthetic_mesh, build_graph_from_mesh,
        )
        mesh = generate_synthetic_mesh(nx=10, ny=5)
        graph = build_graph_from_mesh(mesh)
        # Wall distance should be >= 0
        assert np.all(graph.wall_distance >= 0)
        # Wall nodes should have distance ~= 0
        for nid in mesh.boundary_node_ids:
            assert graph.wall_distance[nid] < 1e-10

    def test_augment_with_solution(self):
        """Augmenting with solution should add extra node features."""
        from scripts.ml_augmentation.mesh_graph_utils import (
            generate_synthetic_mesh, build_graph_from_mesh,
            augment_graph_with_solution,
        )
        mesh = generate_synthetic_mesh(nx=8, ny=4)
        graph = build_graph_from_mesh(mesh)
        original_cols = graph.node_features.shape[1]

        solution = {
            "Cp": np.random.randn(32),
            "nu_t": np.abs(np.random.randn(32)) * 1e-5,
        }
        graph = augment_graph_with_solution(graph, solution)
        assert graph.node_features.shape[1] == original_cols + 2

    def test_augment_wrong_size_skipped(self):
        """Solution fields with wrong size should be skipped."""
        from scripts.ml_augmentation.mesh_graph_utils import (
            generate_synthetic_mesh, build_graph_from_mesh,
            augment_graph_with_solution,
        )
        mesh = generate_synthetic_mesh(nx=8, ny=4)
        graph = build_graph_from_mesh(mesh)
        original_cols = graph.node_features.shape[1]

        solution = {"wrong": np.random.randn(99)}  # Wrong size
        graph = augment_graph_with_solution(graph, solution)
        assert graph.node_features.shape[1] == original_cols

    @requires_pyg
    def test_graph_to_pyg(self):
        """Conversion to PyG Data should preserve all fields."""
        from scripts.ml_augmentation.mesh_graph_utils import (
            generate_synthetic_mesh, build_graph_from_mesh, graph_to_pyg,
        )
        mesh = generate_synthetic_mesh(nx=8, ny=4)
        graph = build_graph_from_mesh(mesh)
        beta = np.ones(32, dtype=np.float32)

        pyg_data = graph_to_pyg(graph, target=beta)
        assert pyg_data.x.shape == (32, 4)
        assert pyg_data.edge_index.shape[0] == 2
        assert pyg_data.edge_attr.shape[1] == 5
        assert pyg_data.y.shape == (32, 1)
        assert hasattr(pyg_data, "wall_distance")
        assert hasattr(pyg_data, "boundary_mask")

    def test_load_su2_convenience(self):
        """load_su2_mesh_as_graph should produce a valid graph."""
        # Write a minimal SU2 mesh file
        import tempfile
        mesh_content = """%
% Minimal SU2 mesh for testing
%
NDIME= 2
NELEM= 2
5  0 1 3  0
5  1 2 3  1
NPOIN= 4
0.0  0.0  0
1.0  0.0  1
1.0  1.0  2
0.0  1.0  3
NMARK= 1
MARKER_TAG= wall
MARKER_ELEMS= 1
3  0 1
"""
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.su2', delete=False
        ) as f:
            f.write(mesh_content)
            tmp_path = f.name

        from scripts.ml_augmentation.mesh_graph_utils import (
            load_su2_mesh_as_graph,
        )
        graph = load_su2_mesh_as_graph(tmp_path)
        assert graph.n_nodes == 4
        assert graph.n_edges > 0
        Path(tmp_path).unlink()


# =========================================================================
# TestMeshGraphNet — Model Architecture
# =========================================================================
@requires_pyg
class TestMeshGraphNet:
    """Tests for MeshGraphNet encoder-processor-decoder."""

    def _make_sample_data(self):
        """Create a small sample PyG Data."""
        from scripts.ml_augmentation.mesh_graph_utils import (
            generate_synthetic_mesh, build_graph_from_mesh, graph_to_pyg,
        )
        mesh = generate_synthetic_mesh(nx=8, ny=4)
        graph = build_graph_from_mesh(mesh)
        beta = np.ones(32, dtype=np.float32) * 1.1
        return graph_to_pyg(graph, target=beta), graph

    def test_forward_pass(self):
        """Model should produce output of correct shape."""
        from scripts.ml_augmentation.mesh_graphnet import (
            MeshGraphNet, MeshGraphNetConfig,
        )
        data, graph = self._make_sample_data()
        config = MeshGraphNetConfig(
            node_in_dim=data.x.shape[1],
            edge_in_dim=data.edge_attr.shape[1],
            latent_dim=32, n_message_passing=3,
            output_dim=1,
            encoder_hidden=(32,), decoder_hidden=(32,),
        )
        model = MeshGraphNet(config)
        out = model.forward_data(data)
        assert out.shape == (32, 1)

    def test_output_dim(self):
        """Multi-output should respect output_dim."""
        from scripts.ml_augmentation.mesh_graphnet import (
            MeshGraphNet, MeshGraphNetConfig,
        )
        data, _ = self._make_sample_data()
        config = MeshGraphNetConfig(
            node_in_dim=data.x.shape[1],
            edge_in_dim=data.edge_attr.shape[1],
            latent_dim=16, n_message_passing=2,
            output_dim=3,  # Multi-output
            encoder_hidden=(16,), decoder_hidden=(16,),
        )
        model = MeshGraphNet(config)
        out = model.forward_data(data)
        assert out.shape == (32, 3)

    def test_gradient_flow(self):
        """Gradients should flow through all parameters."""
        from scripts.ml_augmentation.mesh_graphnet import (
            MeshGraphNet, MeshGraphNetConfig,
        )
        data, _ = self._make_sample_data()
        config = MeshGraphNetConfig(
            node_in_dim=data.x.shape[1],
            edge_in_dim=data.edge_attr.shape[1],
            latent_dim=16, n_message_passing=2,
            output_dim=1,
            encoder_hidden=(16,), decoder_hidden=(16,),
        )
        model = MeshGraphNet(config)
        out = model.forward_data(data)
        loss = out.sum()
        loss.backward()
        # All parameters should have gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_count_parameters(self):
        """Parameter count should be positive."""
        from scripts.ml_augmentation.mesh_graphnet import (
            MeshGraphNet, MeshGraphNetConfig,
        )
        data, _ = self._make_sample_data()
        config = MeshGraphNetConfig(
            node_in_dim=data.x.shape[1],
            edge_in_dim=data.edge_attr.shape[1],
            latent_dim=32, n_message_passing=3,
            output_dim=1,
        )
        model = MeshGraphNet(config)
        n_params = model.count_parameters()
        assert n_params > 0

    def test_train_reduces_loss(self):
        """Training should reduce the loss."""
        from scripts.ml_augmentation.mesh_graphnet import (
            MeshGraphNet, MeshGraphNetConfig, train_meshgraphnet,
        )
        data, _ = self._make_sample_data()
        config = MeshGraphNetConfig(
            node_in_dim=data.x.shape[1],
            edge_in_dim=data.edge_attr.shape[1],
            latent_dim=16, n_message_passing=2,
            output_dim=1,
            encoder_hidden=(16,), decoder_hidden=(16,),
            max_epochs=30, learning_rate=1e-3,
            early_stopping_patience=100,
        )
        model = MeshGraphNet(config)
        result = train_meshgraphnet(model, [data], config=config)
        # Loss should decrease
        assert result.train_loss_history[-1] <= result.train_loss_history[0]


# =========================================================================
# TestPhysicsGuidedGNN — Physics Loss Computation
# =========================================================================
@requires_pyg
class TestPhysicsGuidedGNN:
    """Tests for Physics-Guided GNN loss functions."""

    def _make_sample(self):
        from scripts.ml_augmentation.mesh_graph_utils import (
            generate_synthetic_mesh, build_graph_from_mesh, graph_to_pyg,
        )
        mesh = generate_synthetic_mesh(nx=8, ny=4)
        graph = build_graph_from_mesh(mesh)
        beta = np.ones(32, dtype=np.float32) * 1.1
        return graph_to_pyg(graph, target=beta), graph

    def test_realizability_loss_bounded_beta(self):
        """Realizability loss should be ~0 for valid β."""
        from scripts.ml_augmentation.physics_guided_gnn import (
            RealizabilityLoss,
        )
        data, _ = self._make_sample()
        loss_fn = RealizabilityLoss(beta_min=0.1, beta_max=5.0)
        # β = 1.0 (valid)
        beta_valid = torch.ones(32)
        loss = loss_fn(beta_valid, data.edge_index)
        # Should be very small (only smoothness component)
        assert loss.item() < 0.1

    def test_realizability_loss_penalizes_violations(self):
        """Should penalize β below min or above max."""
        from scripts.ml_augmentation.physics_guided_gnn import (
            RealizabilityLoss,
        )
        data, _ = self._make_sample()
        loss_fn = RealizabilityLoss(beta_min=0.1, beta_max=5.0)
        # β with violations
        beta_bad = torch.cat([
            torch.zeros(16) - 1.0,   # Below min
            torch.ones(16) * 10.0,   # Above max
        ])
        loss = loss_fn(beta_bad, data.edge_index)
        assert loss.item() > 1.0  # Significant penalty

    def test_smoothness_loss(self):
        """Smooth fields should have low Laplacian loss."""
        from scripts.ml_augmentation.physics_guided_gnn import (
            graph_laplacian_smoothness,
        )
        data, _ = self._make_sample()
        # Constant field = perfectly smooth
        f_smooth = torch.ones(32)
        loss_smooth = graph_laplacian_smoothness(f_smooth, data.edge_index)
        assert loss_smooth.item() < 1e-10

        # Random field = not smooth
        f_noisy = torch.randn(32)
        loss_noisy = graph_laplacian_smoothness(f_noisy, data.edge_index)
        assert loss_noisy.item() > loss_smooth.item()

    def test_pg_gnn_compute_loss(self):
        """PhysicsGuidedGNN should compute all loss components."""
        from scripts.ml_augmentation.mesh_graphnet import (
            MeshGraphNet, MeshGraphNetConfig,
        )
        from scripts.ml_augmentation.physics_guided_gnn import (
            PhysicsGuidedGNN, PhysicsGNNConfig,
        )
        data, _ = self._make_sample()
        config = MeshGraphNetConfig(
            node_in_dim=data.x.shape[1],
            edge_in_dim=data.edge_attr.shape[1],
            latent_dim=16, n_message_passing=2,
            output_dim=1,
            encoder_hidden=(16,), decoder_hidden=(16,),
        )
        base = MeshGraphNet(config)
        pg_config = PhysicsGNNConfig(warmup_epochs=0, ramp_epochs=1)
        pg_gnn = PhysicsGuidedGNN(base, pg_config)

        losses = pg_gnn.compute_loss(data, epoch=5)
        assert "total" in losses
        assert "data" in losses
        assert "realizability" in losses
        assert "smoothness" in losses
        assert losses["total"].item() > 0

    def test_physics_warmup(self):
        """Physics losses should be zero during warmup."""
        from scripts.ml_augmentation.physics_guided_gnn import (
            PhysicsGuidedGNN, PhysicsGNNConfig,
        )
        from scripts.ml_augmentation.mesh_graphnet import (
            MeshGraphNet, MeshGraphNetConfig,
        )
        data, _ = self._make_sample()
        config = MeshGraphNetConfig(
            node_in_dim=data.x.shape[1],
            edge_in_dim=data.edge_attr.shape[1],
            latent_dim=16, n_message_passing=2,
            output_dim=1,
        )
        base = MeshGraphNet(config)
        pg_config = PhysicsGNNConfig(warmup_epochs=100, ramp_epochs=50)
        pg_gnn = PhysicsGuidedGNN(base, pg_config)

        w = pg_gnn._get_physics_weight(epoch=5)
        assert w == 0.0  # During warmup

    def test_pg_gnn_training(self):
        """PG-GNN training should complete without errors."""
        from scripts.ml_augmentation.mesh_graphnet import (
            MeshGraphNet, MeshGraphNetConfig,
        )
        from scripts.ml_augmentation.physics_guided_gnn import (
            PhysicsGuidedGNN, PhysicsGNNConfig, train_physics_guided_gnn,
        )
        data, _ = self._make_sample()
        config = MeshGraphNetConfig(
            node_in_dim=data.x.shape[1],
            edge_in_dim=data.edge_attr.shape[1],
            latent_dim=16, n_message_passing=2,
            output_dim=1,
            encoder_hidden=(16,), decoder_hidden=(16,),
        )
        base = MeshGraphNet(config)
        pg_config = PhysicsGNNConfig(warmup_epochs=5, ramp_epochs=5)
        pg_gnn = PhysicsGuidedGNN(base, pg_config)

        result = train_physics_guided_gnn(
            pg_gnn, [data], max_epochs=15,
            early_stopping_patience=100,
        )
        assert result.total_epochs > 0
        assert len(result.train_loss_history) > 0


# =========================================================================
# TestGNNFIMLPipeline — End-to-End Pipeline
# =========================================================================
@requires_pyg
class TestGNNFIMLPipeline:
    """Tests for the GNN FIML pipeline."""

    def _make_cases(self):
        """Generate synthetic cases for pipeline testing."""
        from scripts.ml_augmentation.mesh_graph_utils import (
            generate_synthetic_mesh, build_graph_from_mesh, graph_to_pyg,
        )
        rng = np.random.default_rng(42)
        cases = []
        for i, name in enumerate(["case_a", "case_b", "case_c"]):
            mesh = generate_synthetic_mesh(nx=10, ny=5, seed=i * 10)
            graph = build_graph_from_mesh(mesh)
            x = graph.node_coords[:, 0]
            beta = np.ones(graph.n_nodes, dtype=np.float32)
            sep = (x > 0.8) & (x < 1.4)
            beta[sep] = 1.2
            pyg_data = graph_to_pyg(graph, target=beta)
            cases.append((name, pyg_data, beta))
        return cases

    def test_add_case(self):
        """Should accept and register cases."""
        from scripts.ml_augmentation.gnn_fiml_pipeline import GNNFIMLPipeline
        cases = self._make_cases()
        pipeline = GNNFIMLPipeline(latent_dim=16, n_message_passing=2)
        for name, data, beta in cases:
            pipeline.add_case_from_graph(name, data, beta)
        assert len(pipeline.cases) == 3

    def test_train_synthetic(self):
        """Training should produce valid metrics."""
        from scripts.ml_augmentation.gnn_fiml_pipeline import GNNFIMLPipeline
        cases = self._make_cases()
        pipeline = GNNFIMLPipeline(
            latent_dim=16, n_message_passing=2,
            encoder_hidden=(16,), decoder_hidden=(16,),
            max_epochs=20, learning_rate=1e-3,
        )
        for name, data, beta in cases:
            pipeline.add_case_from_graph(name, data, beta)

        result = pipeline.train(test_case="case_c")
        assert result.train_cases == ["case_a", "case_b"]
        assert result.test_cases == ["case_c"]
        assert result.n_parameters > 0
        assert result.training_epochs > 0
        assert isinstance(result.train_rmse, float)
        assert isinstance(result.test_rmse, float)

    def test_predict(self):
        """Prediction should return correct shape."""
        from scripts.ml_augmentation.gnn_fiml_pipeline import GNNFIMLPipeline
        cases = self._make_cases()
        pipeline = GNNFIMLPipeline(
            latent_dim=16, n_message_passing=2,
            encoder_hidden=(16,), decoder_hidden=(16,),
            max_epochs=10, learning_rate=1e-3,
        )
        for name, data, beta in cases:
            pipeline.add_case_from_graph(name, data, beta)
        pipeline.train()

        pred = pipeline.predict(cases[0][1])
        assert len(pred) == cases[0][1].num_nodes
        assert np.all(np.isfinite(pred))

    def test_cross_validate(self):
        """Leave-one-out CV should produce results for each case."""
        from scripts.ml_augmentation.gnn_fiml_pipeline import GNNFIMLPipeline
        cases = self._make_cases()[:2]  # Use 2 cases for speed
        pipeline = GNNFIMLPipeline(
            latent_dim=16, n_message_passing=2,
            encoder_hidden=(16,), decoder_hidden=(16,),
            max_epochs=10, learning_rate=1e-3,
        )
        for name, data, beta in cases:
            pipeline.add_case_from_graph(name, data, beta)

        cv_results = pipeline.cross_validate()
        assert len(cv_results) == 2
        for name, result in cv_results.items():
            assert result.training_epochs > 0

    def test_compare_gnn_vs_mlp(self):
        """Comparison function should produce valid metrics."""
        from scripts.ml_augmentation.gnn_fiml_pipeline import (
            GNNFIMLResult, compare_gnn_vs_mlp,
        )
        gnn_result = GNNFIMLResult(
            test_r2=0.85, test_rmse=0.05, n_parameters=10000,
        )
        # Fake MLP result
        class FakeMLP:
            test_r2 = 0.75
            test_rmse = 0.08
        comp = compare_gnn_vs_mlp(gnn_result, FakeMLP())
        assert comp["r2_improvement"] == pytest.approx(0.10)
        assert comp["rmse_improvement_pct"] > 0

    def test_save_model(self, tmp_path):
        """Should save model weights and metadata."""
        from scripts.ml_augmentation.gnn_fiml_pipeline import GNNFIMLPipeline
        cases = self._make_cases()[:2]
        pipeline = GNNFIMLPipeline(
            latent_dim=16, n_message_passing=2,
            encoder_hidden=(16,), decoder_hidden=(16,),
            max_epochs=5,
        )
        for name, data, beta in cases:
            pipeline.add_case_from_graph(name, data, beta)
        pipeline.train()
        pipeline.save(tmp_path / "gnn_model")

        assert (tmp_path / "gnn_model" / "gnn_fiml_model.pth").exists()
        assert (tmp_path / "gnn_model" / "gnn_fiml_meta.json").exists()
