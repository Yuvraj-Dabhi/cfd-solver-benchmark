#!/usr/bin/env python3
"""
Tests for DeepONet Surrogate
================================
Validates branch/trunk architectures, forward pass, training,
physics-informed losses, and synthetic data generation.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ml_augmentation.deeponet_surrogate import (
    BranchNetwork,
    TrunkNetwork,
    DeepONet,
    DeepONetConfig,
    DeepONetSurrogate,
    DeepONetTrainer,
    PhysicsInformedDeepONetLoss,
    generate_swbli_data,
    generate_transonic_airfoil_data,
)


# =========================================================================
# TestBranchNetwork
# =========================================================================
class TestBranchNetwork:
    """Tests for the branch (input function encoder) network."""

    def test_output_shape(self):
        net = BranchNetwork(input_dim=50, hidden_dims=(64, 64), output_dim=32)
        u = np.random.randn(8, 50)
        out = net.forward(u)
        assert out.shape == (8, 32)

    def test_different_batch_sizes(self):
        net = BranchNetwork(input_dim=30, output_dim=16)
        for batch in [1, 5, 20]:
            out = net.forward(np.random.randn(batch, 30))
            assert out.shape == (batch, 16)

    def test_param_count_positive(self):
        net = BranchNetwork(input_dim=50, hidden_dims=(128, 128), output_dim=64)
        assert net.count_params() > 0

    def test_get_set_params_roundtrip(self):
        net = BranchNetwork(input_dim=20, hidden_dims=(32,), output_dim=16)
        params = net.get_params()
        u = np.random.randn(4, 20)
        out1 = net.forward(u)
        # Perturb
        for key in params:
            params[key] += 0.1
        net.set_params(params)
        out2 = net.forward(u)
        # Output should change after perturbation
        assert not np.allclose(out1, out2)

    def test_different_inputs_different_outputs(self):
        """Different input functions should produce different coefficients."""
        net = BranchNetwork(input_dim=50, output_dim=32, seed=123)
        u1 = np.ones((1, 50))
        u2 = -np.ones((1, 50))
        out1 = net.forward(u1)
        out2 = net.forward(u2)
        assert not np.allclose(out1, out2)


# =========================================================================
# TestTrunkNetwork
# =========================================================================
class TestTrunkNetwork:
    """Tests for the trunk (spatial coordinate encoder) network."""

    def test_output_shape(self):
        net = TrunkNetwork(input_dim=2, hidden_dims=(64, 64), output_dim=32)
        y = np.random.randn(100, 2)
        out = net.forward(y)
        assert out.shape == (100, 32)

    def test_1d_coordinates(self):
        net = TrunkNetwork(input_dim=1, output_dim=16)
        y = np.linspace(0, 1, 50).reshape(-1, 1)
        out = net.forward(y)
        assert out.shape == (50, 16)

    def test_3d_coordinates(self):
        """Trunk should handle 3D spatial coordinates."""
        net = TrunkNetwork(input_dim=3, output_dim=32)
        y = np.random.randn(200, 3)
        out = net.forward(y)
        assert out.shape == (200, 32)

    def test_param_count_positive(self):
        net = TrunkNetwork(input_dim=2, hidden_dims=(128, 128), output_dim=64)
        assert net.count_params() > 0


# =========================================================================
# TestDeepONet
# =========================================================================
class TestDeepONet:
    """Tests for the full DeepONet architecture."""

    def test_forward_shape(self):
        model = DeepONet(branch_input_dim=50, trunk_input_dim=1,
                         basis_dim=32, n_outputs=1)
        u = np.random.randn(8, 50)
        y = np.linspace(0, 1, 40).reshape(-1, 1)
        out = model.forward(u, y)
        assert out.shape == (8, 1, 40)

    def test_multi_output(self):
        """Multi-output (e.g. Cp + Cf) should produce correct shape."""
        model = DeepONet(branch_input_dim=50, trunk_input_dim=1,
                         basis_dim=32, n_outputs=3)
        u = np.random.randn(5, 50)
        y = np.linspace(0, 1, 60).reshape(-1, 1)
        out = model.forward(u, y)
        assert out.shape == (5, 3, 60)

    def test_2d_trunk_input(self):
        """2D spatial coordinates (x, y)."""
        model = DeepONet(branch_input_dim=30, trunk_input_dim=2,
                         basis_dim=16, n_outputs=2)
        u = np.random.randn(4, 30)
        y = np.random.randn(80, 2)
        out = model.forward(u, y)
        assert out.shape == (4, 2, 80)

    def test_different_inputs_change_output(self):
        model = DeepONet(branch_input_dim=20, trunk_input_dim=1, basis_dim=16)
        y = np.linspace(0, 1, 30).reshape(-1, 1)
        u1 = np.ones((1, 20))
        u2 = -np.ones((1, 20))
        out1 = model.forward(u1, y)
        out2 = model.forward(u2, y)
        assert not np.allclose(out1, out2)

    def test_different_query_points_change_output(self):
        model = DeepONet(branch_input_dim=20, trunk_input_dim=1, basis_dim=16)
        u = np.random.randn(2, 20)
        y1 = np.array([[0.0], [0.5]])
        y2 = np.array([[0.3], [0.8]])
        out1 = model.forward(u, y1)
        out2 = model.forward(u, y2)
        assert not np.allclose(out1, out2)

    def test_param_count(self):
        model = DeepONet(branch_input_dim=50, trunk_input_dim=2,
                         hidden_dims=(128, 128), basis_dim=64, n_outputs=2)
        n = model.count_params()
        assert n > 0
        # Should include branch + trunk + bias
        assert n > 128 * 128  # At least one hidden layer

    def test_single_sample(self):
        """Should work with batch_size=1."""
        model = DeepONet(branch_input_dim=10, trunk_input_dim=1, basis_dim=8)
        u = np.random.randn(1, 10)
        y = np.linspace(0, 1, 20).reshape(-1, 1)
        out = model.forward(u, y)
        assert out.shape == (1, 1, 20)
        assert np.all(np.isfinite(out))


# =========================================================================
# TestPhysicsInformedLoss
# =========================================================================
class TestPhysicsInformedLoss:
    """Tests for physics-informed loss computation."""

    def test_data_loss_zero_for_perfect(self):
        loss_fn = PhysicsInformedDeepONetLoss()
        pred = np.random.randn(4, 2, 50)
        assert loss_fn.data_loss(pred, pred) < 1e-10

    def test_data_loss_positive_for_mismatch(self):
        loss_fn = PhysicsInformedDeepONetLoss()
        pred = np.random.randn(4, 2, 50)
        target = pred + 0.5
        assert loss_fn.data_loss(pred, target) > 0

    def test_continuity_residual_zero_for_constant(self):
        """Constant velocity fields should have zero divergence."""
        loss_fn = PhysicsInformedDeepONetLoss()
        u = np.ones((3, 40))
        v = np.ones((3, 40))
        res = loss_fn.continuity_residual(u, v, dx=0.01)
        assert res < 1e-10

    def test_momentum_residual_nonnegative(self):
        loss_fn = PhysicsInformedDeepONetLoss()
        u = np.random.randn(2, 30)
        p = np.random.randn(2, 30)
        res = loss_fn.momentum_residual(u, p, dx=0.01)
        assert res >= 0

    def test_total_loss_structure(self):
        loss_fn = PhysicsInformedDeepONetLoss(physics_weight=0.5)
        pred = np.random.randn(4, 2, 50)
        target = pred + np.random.randn(4, 2, 50) * 0.1
        result = loss_fn.total_loss(pred, target)
        assert "data" in result
        assert "physics" in result
        assert "total" in result
        assert result["total"] >= result["data"]


# =========================================================================
# TestSyntheticDataGeneration
# =========================================================================
class TestSyntheticDataGeneration:
    """Tests for synthetic training data generators."""

    def test_swbli_data_shapes(self):
        data = generate_swbli_data(n_samples=20, n_sensors=30, n_query_points=50)
        assert data["input_functions"].shape == (20, 30)
        assert data["query_coords"].shape == (50, 1)
        assert data["target_fields"].shape == (20, 2, 50)
        assert data["mach_numbers"].shape == (20,)

    def test_swbli_mach_range(self):
        data = generate_swbli_data(n_samples=50, mach_range=(3.0, 5.0))
        assert np.all(data["mach_numbers"] >= 3.0)
        assert np.all(data["mach_numbers"] <= 5.0)

    def test_swbli_data_finite(self):
        data = generate_swbli_data(n_samples=10)
        assert np.all(np.isfinite(data["input_functions"]))
        assert np.all(np.isfinite(data["target_fields"]))

    def test_transonic_data_shapes(self):
        data = generate_transonic_airfoil_data(
            n_samples=15, n_sensors=40, n_query_points=60)
        assert data["input_functions"].shape == (15, 40)
        assert data["query_coords"].shape == (60, 1)
        assert data["target_fields"].shape == (15, 1, 60)
        assert data["flow_params"].shape == (15, 2)

    def test_transonic_data_finite(self):
        data = generate_transonic_airfoil_data(n_samples=10)
        assert np.all(np.isfinite(data["input_functions"]))
        assert np.all(np.isfinite(data["target_fields"]))

    def test_seed_reproducibility(self):
        d1 = generate_swbli_data(n_samples=5, seed=99)
        d2 = generate_swbli_data(n_samples=5, seed=99)
        np.testing.assert_array_equal(d1["input_functions"],
                                      d2["input_functions"])


# =========================================================================
# TestDeepONetTrainer
# =========================================================================
class TestDeepONetTrainer:
    """Tests for training pipeline."""

    def test_training_runs(self):
        data = generate_swbli_data(n_samples=30, n_sensors=20, n_query_points=25)
        model = DeepONet(branch_input_dim=20, trunk_input_dim=1,
                         basis_dim=16, n_outputs=2, seed=42)
        trainer = DeepONetTrainer(model, n_epochs=10, batch_size=8)
        history = trainer.train(
            data["input_functions"], data["query_coords"],
            data["target_fields"],
        )
        assert len(history["train_loss"]) > 0
        assert len(history["val_loss"]) > 0
        assert all(np.isfinite(history["train_loss"]))

    def test_early_stopping(self):
        """With patience=2, training should stop early."""
        data = generate_swbli_data(n_samples=20, n_sensors=10, n_query_points=15)
        model = DeepONet(branch_input_dim=10, trunk_input_dim=1,
                         basis_dim=8, n_outputs=2)
        trainer = DeepONetTrainer(model, n_epochs=200, patience=2, batch_size=8)
        history = trainer.train(
            data["input_functions"], data["query_coords"],
            data["target_fields"],
        )
        # Should have stopped well before 200 epochs
        assert len(history["train_loss"]) < 200


# =========================================================================
# TestDeepONetSurrogate
# =========================================================================
class TestDeepONetSurrogate:
    """Tests for high-level surrogate wrapper."""

    def test_fit_and_predict(self):
        data = generate_swbli_data(n_samples=30, n_sensors=20, n_query_points=25)
        config = DeepONetConfig(
            branch_input_dim=20, trunk_input_dim=1,
            basis_dim=16, n_outputs=2, n_epochs=5,
        )
        surrogate = DeepONetSurrogate(config=config)
        history = surrogate.fit(
            data["input_functions"], data["query_coords"],
            data["target_fields"],
        )
        assert surrogate._fitted
        pred = surrogate.predict(data["input_functions"][:5],
                                 data["query_coords"])
        assert pred.shape == (5, 2, 25)

    def test_evaluate_metrics(self):
        data = generate_swbli_data(n_samples=20, n_sensors=15, n_query_points=20)
        surrogate = DeepONetSurrogate(
            branch_input_dim=15, trunk_input_dim=1,
            basis_dim=8, n_outputs=2, n_epochs=3,
        )
        surrogate.fit(data["input_functions"], data["query_coords"],
                      data["target_fields"])
        metrics = surrogate.evaluate(
            data["input_functions"][:5], data["query_coords"],
            data["target_fields"][:5],
        )
        assert "output_0_rmse" in metrics
        assert "output_1_relative_l2" in metrics
        assert all(v >= 0 for v in metrics.values())

    def test_to_dict(self):
        surrogate = DeepONetSurrogate(branch_input_dim=10, trunk_input_dim=1)
        info = surrogate.to_dict()
        assert "config" in info
        assert "n_params" in info
        assert info["n_params"] > 0

    def test_summary_string(self):
        surrogate = DeepONetSurrogate(branch_input_dim=10, trunk_input_dim=1)
        s = surrogate.summary()
        assert "DeepONet" in s
        assert "Branch" in s


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
