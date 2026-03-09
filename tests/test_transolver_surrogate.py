"""
Tests for Transformer-Based Physics Surrogate (Transolver / AB-UPT)
=====================================================================
Validates all sub-components (geometry encoder, slice attention, blocks,
FiLM conditioning, branches, divergence-free projection) and the
top-level TransolverSurrogate API.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ml_augmentation.transolver_surrogate import (
    GeometryEncoder,
    PhysicsSliceAttention,
    TransolverBlock,
    FiLMLayer,
    SurfaceBranch,
    VolumeBranch,
    DivergenceFreeProjection,
    TransolverSurrogate,
    generate_transolver_training_data,
    N_CHANNELS,
)


# =========================================================================
# Geometry Encoder
# =========================================================================
class TestGeometryEncoder:
    """Test AB-UPT geometry encoding."""

    def test_output_shape(self):
        enc = GeometryEncoder(n_geo_in=2, d_geo=64)
        coords = np.random.randn(4, 80, 2)
        out = enc.encode(coords)
        assert out.shape == (4, 80, 64)

    def test_finite_values(self):
        enc = GeometryEncoder(n_geo_in=2, d_geo=32)
        coords = np.random.randn(2, 40, 2)
        out = enc.encode(coords)
        assert np.all(np.isfinite(out))

    def test_with_normals(self):
        enc = GeometryEncoder(n_geo_in=4, d_geo=32)
        coords = np.random.randn(3, 40, 4)
        out = enc.encode(coords)
        assert out.shape == (3, 40, 32)

    def test_different_coords_differ(self):
        enc = GeometryEncoder(n_geo_in=2, d_geo=32)
        c1 = np.zeros((1, 10, 2))
        c2 = np.ones((1, 10, 2))
        o1 = enc.encode(c1)
        o2 = enc.encode(c2)
        assert not np.allclose(o1, o2)


# =========================================================================
# Physics Slice Attention
# =========================================================================
class TestPhysicsSliceAttention:
    """Test Transolver slice-attention mechanism."""

    def test_output_shape(self):
        attn = PhysicsSliceAttention(d_model=32, n_slices=4, n_heads=4)
        x = np.random.randn(2, 40, 32)
        out = attn.forward(x)
        assert out.shape == (2, 40, 32)

    def test_finite_values(self):
        attn = PhysicsSliceAttention(d_model=16, n_slices=2, n_heads=2)
        x = np.random.randn(3, 20, 16)
        out = attn.forward(x)
        assert np.all(np.isfinite(out))

    def test_different_n_slices(self):
        """Should work with different slice counts."""
        for n_slices in [1, 2, 4, 8]:
            attn = PhysicsSliceAttention(d_model=16, n_slices=n_slices, n_heads=2)
            x = np.random.randn(2, 20, 16)
            out = attn.forward(x)
            assert out.shape == (2, 20, 16), f"Failed at n_slices={n_slices}"

    def test_slice_assignment_sums_to_one(self):
        """Soft slice weights should sum to 1 across slices."""
        attn = PhysicsSliceAttention(d_model=16, n_slices=4, n_heads=2)
        x = np.random.randn(2, 20, 16)
        logits = x @ attn.W_slice + attn.b_slice
        weights = np.exp(logits - logits.max(axis=-1, keepdims=True))
        weights = weights / weights.sum(axis=-1, keepdims=True)
        np.testing.assert_allclose(weights.sum(axis=-1), 1.0, atol=1e-6)


# =========================================================================
# Transolver Block
# =========================================================================
class TestTransolverBlock:
    """Test full transformer block."""

    def test_output_shape(self):
        blk = TransolverBlock(d_model=32, n_slices=4, n_heads=4)
        x = np.random.randn(2, 40, 32)
        out = blk.forward(x)
        assert out.shape == (2, 40, 32)

    def test_residual_connection(self):
        """Output should not be zero (residual present)."""
        blk = TransolverBlock(d_model=16, n_slices=2, n_heads=2)
        x = np.ones((1, 10, 16)) * 0.5
        out = blk.forward(x)
        assert not np.allclose(out, np.zeros_like(out))

    def test_finite_output(self):
        blk = TransolverBlock(d_model=16, n_slices=2, n_heads=2)
        x = np.random.randn(3, 30, 16)
        out = blk.forward(x)
        assert np.all(np.isfinite(out))


# =========================================================================
# FiLM Conditioning
# =========================================================================
class TestFiLMLayer:
    """Test FiLM conditioning layer."""

    def test_modulation_shape(self):
        film = FiLMLayer(cond_dim=3, d_model=32)
        features = np.random.randn(4, 40, 32)
        cond = np.random.randn(4, 3)
        out = film.modulate(features, cond)
        assert out.shape == (4, 40, 32)

    def test_different_conditions_differ(self):
        film = FiLMLayer(cond_dim=3, d_model=16)
        features = np.random.randn(1, 10, 16)
        c1 = np.array([[0.1, 0.2, 0.3]])
        c2 = np.array([[0.9, 0.8, 0.7]])
        o1 = film.modulate(features, c1)
        o2 = film.modulate(features, c2)
        assert not np.allclose(o1, o2)

    def test_condition_encoding(self):
        film = FiLMLayer(cond_dim=3, d_model=16)
        cond = film.encode_conditions(
            np.array([5.0, 10.0]),
            np.array([6e6, 6e6]),
            np.array([0.15, 0.15]),
        )
        assert cond.shape == (2, 3)
        assert np.all(np.isfinite(cond))


# =========================================================================
# Surface Branch
# =========================================================================
class TestSurfaceBranch:
    """Test Cp/Cf surface output branch."""

    def test_output_shape(self):
        branch = SurfaceBranch(d_model=32, n_channels=2)
        h = np.random.randn(4, 80, 32)
        out = branch.forward(h)
        assert out.shape == (4, 80, 2)

    def test_finite_output(self):
        branch = SurfaceBranch(d_model=16, n_channels=2)
        h = np.random.randn(2, 40, 16)
        out = branch.forward(h)
        assert np.all(np.isfinite(out))


# =========================================================================
# Volume Branch
# =========================================================================
class TestVolumeBranch:
    """Test velocity/β-field volume output branch."""

    def test_output_shape_2d(self):
        branch = VolumeBranch(d_model=32, n_vol_channels=2)
        h = np.random.randn(3, 40, 32)
        out = branch.forward(h)
        assert out.shape == (3, 40, 2)

    def test_output_shape_scalar(self):
        branch = VolumeBranch(d_model=32, n_vol_channels=1)
        h = np.random.randn(2, 40, 32)
        out = branch.forward(h)
        assert out.shape == (2, 40, 1)


# =========================================================================
# Divergence-Free Projection
# =========================================================================
class TestDivergenceFreeProjection:
    """Test Helmholtz divergence-free projection."""

    def test_output_shape(self):
        proj = DivergenceFreeProjection(n_modes=8)
        v = np.random.randn(2, 40, 2)
        out = proj.project(v)
        assert out.shape == (2, 40, 2)

    def test_reduces_divergence(self):
        """Projected field should have lower divergence than raw."""
        proj = DivergenceFreeProjection(n_modes=8)
        v = np.random.randn(4, 80, 2)
        v_proj = proj.project(v)
        div_raw = proj.compute_divergence(v)
        div_proj = proj.compute_divergence(v_proj)
        assert np.mean(div_proj) <= np.mean(div_raw) + 1e-10

    def test_removes_dc_component(self):
        """DC (mean) should be removed by projection."""
        proj = DivergenceFreeProjection(n_modes=8)
        v = np.ones((1, 40, 2)) * 5.0  # constant field
        v_proj = proj.project(v)
        np.testing.assert_allclose(v_proj.mean(), 0.0, atol=1e-10)

    def test_divergence_computation(self):
        proj = DivergenceFreeProjection()
        v = np.random.randn(2, 40, 2)
        div = proj.compute_divergence(v)
        assert div.shape == (2, 40)
        assert np.all(div >= 0)  # absolute value


# =========================================================================
# Transolver Surrogate (Full Pipeline)
# =========================================================================
class TestTransolverSurrogate:
    """Test full training and prediction pipeline."""

    @pytest.fixture
    def small_model(self):
        return TransolverSurrogate(
            spatial_res=40,
            d_model=16,
            n_layers=2,
            n_slices=2,
            n_heads=2,
            ffn_mult=2,
            n_vol_channels=2,
        )

    @pytest.fixture
    def small_data(self):
        rng = np.random.RandomState(42)
        n = 30
        X = np.stack([
            rng.uniform(-5, 18, n),
            10 ** rng.uniform(5.7, 7, n),
            rng.uniform(0.1, 0.3, n),
        ], axis=-1)
        Y_Cp = rng.randn(n, 40)
        Y_Cf = rng.randn(n, 40) * 0.01
        return X, Y_Cp, Y_Cf

    def test_training_runs(self, small_model, small_data):
        X, Y_Cp, Y_Cf = small_data
        history = small_model.fit(X, Y_Cp, Y_Cf, n_epochs=3, batch_size=8)
        assert "train_loss" in history
        assert len(history["train_loss"]) == 3
        assert all(np.isfinite(l) for l in history["train_loss"])

    def test_model_becomes_fitted(self, small_model, small_data):
        X, Y_Cp, Y_Cf = small_data
        assert not small_model._fitted
        small_model.fit(X, Y_Cp, Y_Cf, n_epochs=2)
        assert small_model._fitted

    def test_predict_before_fit_raises(self, small_model):
        with pytest.raises(RuntimeError, match="not fitted"):
            small_model.predict(np.array([5.0]), np.array([6e6]), np.array([0.15]))

    def test_info_dict(self, small_model):
        info = small_model.get_info()
        assert info["model_type"] == "TransolverSurrogate"
        assert info["architecture"] == "Transolver + AB-UPT"
        assert info["divergence_free"] is True
        assert info["n_params"] > 0

    def test_summary_string(self, small_model):
        s = small_model.summary()
        assert "Transolver" in s
        assert "AB-UPT" in s


# =========================================================================
# Surface Prediction
# =========================================================================
class TestSurfacePrediction:
    """Test surface-specific predictions."""

    @pytest.fixture
    def trained_model(self):
        model = TransolverSurrogate(
            spatial_res=40, d_model=16, n_layers=2,
            n_slices=2, n_heads=2, ffn_mult=2,
        )
        rng = np.random.RandomState(42)
        n = 20
        X = np.stack([
            rng.uniform(-5, 18, n),
            10 ** rng.uniform(5.7, 7, n),
            rng.uniform(0.1, 0.3, n),
        ], axis=-1)
        Y_Cp = rng.randn(n, 40)
        Y_Cf = rng.randn(n, 40) * 0.01
        model.fit(X, Y_Cp, Y_Cf, n_epochs=2)
        return model

    def test_predict_surface_shapes(self, trained_model):
        Cp, Cf = trained_model.predict_surface(
            np.array([5.0, 10.0]),
            np.array([6e6, 6e6]),
            np.array([0.15, 0.15]),
        )
        assert Cp.shape == (2, 40)
        assert Cf.shape == (2, 40)

    def test_predict_volume_shapes(self, trained_model):
        vol = trained_model.predict_volume(
            np.array([5.0]),
            np.array([6e6]),
            np.array([0.15]),
        )
        assert vol.shape == (1, 40, 2)

    def test_predict_finite(self, trained_model):
        Cp, Cf = trained_model.predict_surface(
            np.array([0.0, 5.0, 10.0, 15.0]),
            np.array([3e6, 6e6, 9e6, 6e6]),
            np.array([0.15, 0.2, 0.25, 0.15]),
        )
        assert np.all(np.isfinite(Cp))
        assert np.all(np.isfinite(Cf))


# =========================================================================
# FNO Comparison Metrics
# =========================================================================
class TestFNOComparison:
    """Test comparison output format."""

    @pytest.fixture
    def trained_model_and_data(self):
        model = TransolverSurrogate(
            spatial_res=40, d_model=16, n_layers=2,
            n_slices=2, n_heads=2, ffn_mult=2,
        )
        rng = np.random.RandomState(42)
        n = 20
        X = np.stack([
            rng.uniform(-5, 18, n),
            10 ** rng.uniform(5.7, 7, n),
            rng.uniform(0.1, 0.3, n),
        ], axis=-1)
        Y_Cp = rng.randn(n, 40)
        Y_Cf = rng.randn(n, 40) * 0.01
        model.fit(X, Y_Cp, Y_Cf, n_epochs=2)
        return model, X, Y_Cp, Y_Cf

    def test_comparison_returns_expected_keys(self, trained_model_and_data):
        model, X, Y_Cp, Y_Cf = trained_model_and_data
        metrics = model.compare_with_fno(X[:5], Y_Cp[:5], Y_Cf[:5])

        assert metrics["model_name"] == "TransolverSurrogate"
        assert metrics["architecture"] == "Transolver + AB-UPT"
        assert "Cp_RMSE" in metrics
        assert "Cf_RMSE" in metrics
        assert "Cp_R2" in metrics
        assert "Cf_R2" in metrics
        assert "Cp_MAE" in metrics
        assert "Cf_MAE" in metrics
        assert "n_params" in metrics
        assert np.isfinite(metrics["Cp_RMSE"])


# =========================================================================
# Data Generation
# =========================================================================
class TestDataGeneration:
    """Test synthetic training data generator."""

    def test_shapes(self):
        X, Y_Cp, Y_Cf = generate_transolver_training_data(
            n_samples=20, spatial_res=40
        )
        assert X.shape == (20, 3)
        assert Y_Cp.shape == (20, 40)
        assert Y_Cf.shape == (20, 40)

    def test_finite_values(self):
        X, Y_Cp, Y_Cf = generate_transolver_training_data(n_samples=10)
        assert np.all(np.isfinite(X))
        assert np.all(np.isfinite(Y_Cp))
        assert np.all(np.isfinite(Y_Cf))

    def test_condition_ranges(self):
        X, _, _ = generate_transolver_training_data(n_samples=50)
        assert np.all(X[:, 0] >= -5.0)   # AoA
        assert np.all(X[:, 0] <= 18.0)
        assert np.all(X[:, 1] >= 5e5)    # Re
        assert np.all(X[:, 1] <= 1e7)
        assert np.all(X[:, 2] >= 0.1)    # Mach
        assert np.all(X[:, 2] <= 0.3)
