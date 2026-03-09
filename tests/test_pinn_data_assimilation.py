#!/usr/bin/env python3
"""
Tests for PINN-DA: Data Assimilation with Sparse Sensors
==========================================================
Tests for the PINN-DA-SA module including:
  - RANS residual computation
  - Sensor configuration and generation
  - PINN-DA training and reconstruction
  - Comparison with PINN-BL (1D)

All tests use synthetic 2D fields on small grids for speed.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =========================================================================
# PyTorch availability
# =========================================================================
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

requires_torch = pytest.mark.skipif(
    not HAS_TORCH, reason="PyTorch not installed"
)


# =========================================================================
# TestSyntheticFields — 2D flow field generators
# =========================================================================
class TestSyntheticFields:
    """Tests for synthetic 2D flow field generation."""

    def test_wall_hump_shapes(self):
        """Wall hump field should have correct shapes."""
        from scripts.ml_augmentation.pinn_data_assimilation import (
            generate_wall_hump_2d,
        )
        field = generate_wall_hump_2d(nx=30, ny=15)
        N = 30 * 15
        assert len(field["u"]) == N
        assert len(field["v"]) == N
        assert len(field["p"]) == N
        assert field["xx"].shape == (30, 15)

    def test_wall_hump_wall_bc(self):
        """Wall BC should be enforced: u=0, v=0 at y=0."""
        from scripts.ml_augmentation.pinn_data_assimilation import (
            generate_wall_hump_2d,
        )
        field = generate_wall_hump_2d(nx=30, ny=15)
        u_2d = field["u_2d"]
        v_2d = field["v_2d"]
        assert np.allclose(u_2d[:, 0], 0.0)
        assert np.allclose(v_2d[:, 0], 0.0)

    def test_wall_hump_separation(self):
        """Should have reversed flow in separation region."""
        from scripts.ml_augmentation.pinn_data_assimilation import (
            generate_wall_hump_2d,
        )
        field = generate_wall_hump_2d(nx=80, ny=40)
        Cf = field["Cf"]
        x_1d = field["x_1d"]
        # There should be some negative Cf (separation)
        assert np.any(Cf < 0)
        # Separation should start near x/c ~ 0.665
        sep_idx = np.where(Cf < 0)[0]
        if len(sep_idx) > 0:
            x_sep = x_1d[sep_idx[0]]
            assert 0.5 < x_sep < 0.8

    def test_bfs_shapes(self):
        """BFS field should have correct shapes."""
        from scripts.ml_augmentation.pinn_data_assimilation import (
            generate_bfs_2d,
        )
        field = generate_bfs_2d(nx=30, ny=15)
        assert len(field["u"]) == 30 * 15
        assert field["xx"].shape == (30, 15)

    def test_bfs_recirculation(self):
        """BFS should have recirculation zone."""
        from scripts.ml_augmentation.pinn_data_assimilation import (
            generate_bfs_2d,
        )
        field = generate_bfs_2d(nx=80, ny=40)
        u_2d = field["u_2d"]
        # Should have some negative u (recirculation)
        assert np.any(u_2d < 0)

    def test_nu_tilde_nonnegative(self):
        """SA variable ν̃ should be non-negative."""
        from scripts.ml_augmentation.pinn_data_assimilation import (
            generate_wall_hump_2d,
        )
        field = generate_wall_hump_2d(nx=40, ny=20)
        assert np.all(field["nu_tilde"] >= 0)


# =========================================================================
# TestRANSResiduals — PDE residual computation
# =========================================================================
class TestRANSResiduals:
    """Tests for RANS residual computation."""

    def test_continuity_uniform_flow(self):
        """Uniform flow should satisfy continuity exactly."""
        from scripts.ml_augmentation.pinn_data_assimilation import (
            compute_rans_residuals_numpy,
        )
        nx, ny = 20, 10
        x = np.linspace(0, 1, nx)
        y = np.linspace(0, 0.5, ny)
        xx, yy = np.meshgrid(x, y, indexing='ij')

        u = np.ones(nx * ny)  # Uniform u = 1
        v = np.zeros(nx * ny)  # v = 0
        p = np.zeros(nx * ny)
        nt = np.ones(nx * ny) * 1e-4

        res = compute_rans_residuals_numpy(
            xx.ravel(), yy.ravel(), u, v, p, nt,
            nu=1e-5, nx=nx, ny=ny,
        )
        # Continuity of uniform flow should be ~0
        assert np.max(np.abs(res["continuity"])) < 0.1

    def test_residuals_are_finite(self):
        """All residuals should be finite for a physical field."""
        from scripts.ml_augmentation.pinn_data_assimilation import (
            generate_wall_hump_2d, compute_rans_residuals_numpy,
        )
        field = generate_wall_hump_2d(nx=30, ny=15)
        res = compute_rans_residuals_numpy(
            field["x"], field["y"],
            field["u"], field["v"], field["p"], field["nu_tilde"],
            nu=1.5e-5, nx=30, ny=15,
        )
        assert np.all(np.isfinite(res["continuity"]))
        assert np.all(np.isfinite(res["momentum_x"]))
        assert np.all(np.isfinite(res["momentum_y"]))

    def test_residual_keys(self):
        """Should return all expected residual components."""
        from scripts.ml_augmentation.pinn_data_assimilation import (
            compute_rans_residuals_numpy,
        )
        nx, ny = 10, 5
        res = compute_rans_residuals_numpy(
            np.zeros(50), np.zeros(50),
            np.ones(50), np.zeros(50), np.zeros(50), np.ones(50) * 1e-4,
            nx=nx, ny=ny,
        )
        assert "continuity" in res
        assert "momentum_x" in res
        assert "momentum_y" in res


# =========================================================================
# TestSensorConfig — Sparse sensor generation
# =========================================================================
class TestSensorConfig:
    """Tests for sparse sensor data generation."""

    def test_sensor_count(self):
        """Should generate correct number of sensors."""
        from scripts.ml_augmentation.pinn_data_assimilation import (
            generate_wall_hump_2d, generate_synthetic_sensors,
        )
        field = generate_wall_hump_2d(nx=40, ny=20)
        sensors = generate_synthetic_sensors(
            field, n_stations=4, n_points_per_station=10
        )
        assert sensors.n_sensors == 40

    def test_sensor_locations_in_domain(self):
        """Sensor locations should be within domain."""
        from scripts.ml_augmentation.pinn_data_assimilation import (
            generate_wall_hump_2d, generate_synthetic_sensors,
        )
        field = generate_wall_hump_2d(nx=40, ny=20)
        sensors = generate_synthetic_sensors(field, n_stations=3)
        assert np.all(sensors.x_sensors >= 0)
        assert np.all(sensors.x_sensors <= 2.0)
        assert np.all(sensors.y_sensors >= 0)
        assert np.all(sensors.y_sensors <= 0.5)

    def test_noise_affects_measurements(self):
        """Noisy sensors should differ from clean flow field."""
        from scripts.ml_augmentation.pinn_data_assimilation import (
            generate_wall_hump_2d, generate_synthetic_sensors,
        )
        field = generate_wall_hump_2d(nx=40, ny=20)
        clean = generate_synthetic_sensors(field, noise_std=0.0, seed=42)
        noisy = generate_synthetic_sensors(field, noise_std=0.1, seed=42)
        # Same locations, different values
        np.testing.assert_array_equal(clean.x_sensors, noisy.x_sensors)
        assert not np.allclose(clean.u_measured, noisy.u_measured, atol=0.01)

    def test_v_measured_default(self):
        """v_measured should default to zeros if not provided."""
        from scripts.ml_augmentation.pinn_data_assimilation import (
            SensorConfiguration,
        )
        sc = SensorConfiguration(
            x_sensors=np.array([0.5, 1.0]),
            y_sensors=np.array([0.1, 0.2]),
            u_measured=np.array([0.5, 0.8]),
        )
        assert len(sc.v_measured) == 2
        assert np.allclose(sc.v_measured, 0)


# =========================================================================
# TestSAConstants — SA model constants
# =========================================================================
class TestSAConstants:
    """Tests for Spalart-Allmaras model constants."""

    def test_sa_constants_exist(self):
        """All standard SA constants should be defined."""
        from scripts.ml_augmentation.pinn_data_assimilation import (
            SA_CONSTANTS,
        )
        expected = ["cb1", "cb2", "sigma", "kappa", "cw1", "cw2", "cw3", "cv1"]
        for key in expected:
            assert key in SA_CONSTANTS

    def test_cw1_derived(self):
        """cw1 should be correctly derived from other constants."""
        from scripts.ml_augmentation.pinn_data_assimilation import (
            SA_CONSTANTS,
        )
        cw1_expected = (
            SA_CONSTANTS["cb1"] / SA_CONSTANTS["kappa"] ** 2
            + (1 + SA_CONSTANTS["cb2"]) / SA_CONSTANTS["sigma"]
        )
        assert SA_CONSTANTS["cw1"] == pytest.approx(cw1_expected)


# =========================================================================
# TestPINNDANumpy — Numpy-based fallback
# =========================================================================
class TestPINNDANumpy:
    """Tests for the numpy L-BFGS-B PINN-DA."""

    def test_basis_shape(self):
        """Fourier basis should have correct shape."""
        from scripts.ml_augmentation.pinn_data_assimilation import (
            PINNDAAssimilatorNumpy, PINNDAConfig,
        )
        da = PINNDAAssimilatorNumpy(PINNDAConfig(), n_basis_x=4, n_basis_y=4)
        x = np.array([0.5, 1.0, 1.5])
        y = np.array([0.1, 0.2, 0.3])
        basis = da._build_basis(x, y)
        assert basis.shape == (3, 16)  # 4 * 4 = 16 basis functions

    def test_training_reduces_loss(self):
        """Training should reduce the overall loss."""
        from scripts.ml_augmentation.pinn_data_assimilation import (
            PINNDAAssimilatorNumpy, PINNDAConfig,
            generate_wall_hump_2d, generate_synthetic_sensors,
        )
        field = generate_wall_hump_2d(nx=20, ny=10)
        sensors = generate_synthetic_sensors(
            field, n_stations=3, n_points_per_station=5, noise_std=0.01
        )
        config = PINNDAConfig(
            lambda_data=10.0, lambda_continuity=0.1,
            lambda_momentum=0.1, lambda_bc=1.0,
        )
        da = PINNDAAssimilatorNumpy(config, n_basis_x=4, n_basis_y=4)
        result = da.train(sensors, reference_field=field)
        assert result.final_total_loss < 100  # Should converge somewhat
        assert result.training_epochs > 0

    def test_predict_shape(self):
        """Predict should return 4 fields."""
        from scripts.ml_augmentation.pinn_data_assimilation import (
            PINNDAAssimilatorNumpy, PINNDAConfig,
            generate_wall_hump_2d, generate_synthetic_sensors,
        )
        field = generate_wall_hump_2d(nx=20, ny=10)
        sensors = generate_synthetic_sensors(
            field, n_stations=2, n_points_per_station=5
        )
        da = PINNDAAssimilatorNumpy(PINNDAConfig(), n_basis_x=3, n_basis_y=3)
        da.train(sensors)

        x_eval = np.array([0.5, 1.0])
        y_eval = np.array([0.1, 0.2])
        u, v, p, nt = da.predict(x_eval, y_eval)
        assert len(u) == 2
        assert len(v) == 2
        assert len(p) == 2
        assert len(nt) == 2


# =========================================================================
# TestPINNDATorch — PyTorch-based PINN-DA
# =========================================================================
@requires_torch
class TestPINNDATorch:
    """Tests for the PyTorch PINN-DA."""

    def test_network_forward(self):
        """Network should output 4 fields."""
        from scripts.ml_augmentation.pinn_data_assimilation import (
            _PINNDANetwork,
        )
        net = _PINNDANetwork(hidden_layers=(32, 32))
        xy = torch.randn(10, 2)
        out = net(xy)
        assert out.shape == (10, 4)

    def test_network_gradients(self):
        """Autograd should flow through the network."""
        from scripts.ml_augmentation.pinn_data_assimilation import (
            _PINNDANetwork,
        )
        net = _PINNDANetwork(hidden_layers=(32, 32))
        xy = torch.randn(5, 2, requires_grad=True)
        out = net(xy)
        loss = out.sum()
        loss.backward()
        assert xy.grad is not None

    def test_pde_residuals_torch(self):
        """PDE residuals should be computable via autograd."""
        from scripts.ml_augmentation.pinn_data_assimilation import (
            PINNDataAssimilator, PINNDAConfig,
        )
        config = PINNDAConfig(hidden_layers=(32, 32))
        da = PINNDataAssimilator(config)
        da._build_network()
        xy = torch.randn(20, 2, requires_grad=True)
        res = da._compute_pde_residuals(xy)
        assert "continuity" in res
        assert "momentum_x" in res
        assert "momentum_y" in res
        assert "sa" in res
        for key, val in res.items():
            assert val.shape[0] == 20
            assert torch.all(torch.isfinite(val))

    def test_training_completes(self):
        """Training should complete without error."""
        from scripts.ml_augmentation.pinn_data_assimilation import (
            PINNDataAssimilator, PINNDAConfig,
            generate_wall_hump_2d, generate_synthetic_sensors,
        )
        field = generate_wall_hump_2d(nx=20, ny=10)
        sensors = generate_synthetic_sensors(
            field, n_stations=2, n_points_per_station=5, noise_std=0.01
        )
        config = PINNDAConfig(
            hidden_layers=(32, 32),
            max_epochs=20,
            n_collocation=50,
            n_boundary=10,
        )
        da = PINNDataAssimilator(config)
        result = da.train(sensors, reference_field=field)
        assert result.training_epochs == 20
        assert result.training_time_s > 0
        assert len(result.loss_history) == 20

    def test_predict_after_training(self):
        """Should be able to predict after training."""
        from scripts.ml_augmentation.pinn_data_assimilation import (
            PINNDataAssimilator, PINNDAConfig,
            generate_wall_hump_2d, generate_synthetic_sensors,
        )
        field = generate_wall_hump_2d(nx=20, ny=10)
        sensors = generate_synthetic_sensors(
            field, n_stations=2, n_points_per_station=5
        )
        config = PINNDAConfig(
            hidden_layers=(32, 32), max_epochs=10,
            n_collocation=30, n_boundary=10,
        )
        da = PINNDataAssimilator(config)
        da.train(sensors)
        u, v, p, nt = da.predict(
            np.array([0.5, 1.0]), np.array([0.1, 0.2])
        )
        assert len(u) == 2
        assert np.all(np.isfinite(u))


# =========================================================================
# TestComparison — PINN-DA vs PINN-BL
# =========================================================================
class TestComparison:
    """Tests for PINN-DA vs PINN-BL comparison."""

    def test_comparison_returns_metrics(self):
        """Comparison should return expected metric keys."""
        from scripts.ml_augmentation.pinn_data_assimilation import (
            compare_pinn_da_vs_pinn_bl,
        )
        result = compare_pinn_da_vs_pinn_bl(
            case="wall_hump", n_sensors=20, noise_std=0.01
        )
        expected_keys = [
            "case", "n_sensors", "pinn_da_u_rmse",
            "pinn_bl_cf_rmse_before", "pinn_bl_cf_rmse_after",
            "pinn_bl_improvement_pct",
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

    def test_pinn_bl_improves_cf(self):
        """PINN-BL should show Cf improvement."""
        from scripts.ml_augmentation.pinn_data_assimilation import (
            compare_pinn_da_vs_pinn_bl,
        )
        result = compare_pinn_da_vs_pinn_bl(
            case="wall_hump", n_sensors=20
        )
        assert result["pinn_bl_cf_rmse_after"] <= result["pinn_bl_cf_rmse_before"]
