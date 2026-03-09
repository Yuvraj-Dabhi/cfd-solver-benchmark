"""
Tests for Multi-Fidelity Hierarchical Learning Framework
==========================================================
Validates all sub-components (FidelityLevel, MultiFidelityDataset,
ResidualCorrectionNet, ConditionalInvertibleBlock, ConditionalINN,
CoKrigingSurrogate) and the top-level MultiFidelityFramework API.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ml_augmentation.multifidelity_framework import (
    FidelityLevel,
    MultiFidelityDataset,
    ResidualCorrectionNet,
    ConditionalInvertibleBlock,
    ConditionalINN,
    CoKrigingSurrogate,
    MultiFidelityFramework,
    generate_multifidelity_data,
)


# =========================================================================
# FidelityLevel
# =========================================================================
class TestFidelityLevel:
    """Test fidelity metadata dataclass."""

    def test_creation(self):
        lev = FidelityLevel("coarse_RANS", 0, cost=1.0, resolution=40)
        assert lev.name == "coarse_RANS"
        assert lev.level == 0
        assert lev.cost == 1.0

    def test_ordering(self):
        levels = [
            FidelityLevel("DNS", 2),
            FidelityLevel("coarse", 0),
            FidelityLevel("fine", 1),
        ]
        ordered = sorted(levels, key=lambda l: l.level)
        assert [l.name for l in ordered] == ["coarse", "fine", "DNS"]


# =========================================================================
# MultiFidelityDataset
# =========================================================================
class TestMultiFidelityDataset:
    """Test multi-fidelity data management."""

    @pytest.fixture
    def dataset(self):
        ds = generate_multifidelity_data(
            n_low=50, n_mid=20, n_high=10, seed=42
        )
        return ds

    def test_levels_populated(self, dataset):
        summ = dataset.summary
        assert summ["n_levels"] == 3
        names = [l["name"] for l in summ["levels"]]
        assert "coarse_RANS" in names
        assert "DNS_LES" in names

    def test_get_level_data(self, dataset):
        X, Y = dataset.get_level_data("coarse_RANS")
        assert X.shape[0] == 50
        assert Y.shape[0] == 50

    def test_invalid_level_raises(self, dataset):
        with pytest.raises(ValueError, match="Unknown level"):
            dataset.add_level_data("invalid", np.zeros((5, 3)), np.zeros((5, 1)))

    def test_get_unknown_level_raises(self, dataset):
        with pytest.raises(KeyError):
            dataset.get_level_data("nonexistent")

    def test_aligned_pairs(self, dataset):
        X, Y_lo, Y_hi = dataset.get_aligned_pairs("coarse_RANS", "DNS_LES")
        assert X.shape[0] == 10  # same as high-fidelity count
        assert Y_lo.shape[0] == 10
        assert Y_hi.shape[0] == 10

    def test_compute_residuals(self, dataset):
        X, delta = dataset.compute_residuals("coarse_RANS", "DNS_LES")
        assert X.shape[0] == 10
        assert delta.shape[0] == 10
        assert np.all(np.isfinite(delta))


# =========================================================================
# ResidualCorrectionNet
# =========================================================================
class TestResidualCorrectionNet:
    """Test the Δ(high − low) correction MLP."""

    @pytest.fixture
    def trained_net(self):
        rng = np.random.RandomState(42)
        n = 30
        X = rng.randn(n, 3)
        Y_lo = rng.randn(n, 1)
        Y_hi = Y_lo + 0.5 * X[:, :1] + rng.randn(n, 1) * 0.1
        net = ResidualCorrectionNet(n_features=3, n_low_out=1, n_out=1)
        net.fit(X, Y_lo, Y_hi, n_epochs=5)
        return net

    def test_training_runs(self):
        rng = np.random.RandomState(42)
        net = ResidualCorrectionNet(n_features=3, n_low_out=1, n_out=1)
        X = rng.randn(30, 3)
        Y_lo = rng.randn(30, 1)
        Y_hi = Y_lo + 0.1
        hist = net.fit(X, Y_lo, Y_hi, n_epochs=5)
        assert "train_loss" in hist
        assert len(hist["train_loss"]) == 5

    def test_predict_before_fit_raises(self):
        net = ResidualCorrectionNet()
        with pytest.raises(RuntimeError, match="Not fitted"):
            net.predict_correction(np.zeros((1, 3)), np.zeros((1, 1)))

    def test_predict_correction_shape(self, trained_net):
        delta = trained_net.predict_correction(np.zeros((5, 3)), np.zeros((5, 1)))
        assert delta.shape == (5, 1)

    def test_predict_corrected_output(self, trained_net):
        Y_lo = np.ones((3, 1)) * 0.5
        Y_corr = trained_net.predict(np.zeros((3, 3)), Y_lo)
        assert Y_corr.shape == (3, 1)
        assert np.all(np.isfinite(Y_corr))

    def test_1d_input_auto_reshape(self, trained_net):
        """1D Y_low should be auto-reshaped."""
        Y_corr = trained_net.predict(np.zeros((3, 3)), np.zeros(3))
        assert Y_corr.shape == (3, 1)


# =========================================================================
# ConditionalInvertibleBlock
# =========================================================================
class TestConditionalInvertibleBlock:
    """Test single cINN coupling layer."""

    def test_forward_shape(self):
        block = ConditionalInvertibleBlock(dim=4, cond_dim=3)
        x = np.random.randn(5, 4)
        c = np.random.randn(5, 3)
        z, log_det = block.forward(x, c)
        assert z.shape == (5, 4)
        assert log_det.shape == (5,)

    def test_invertibility(self):
        """Inverse of forward should recover input."""
        block = ConditionalInvertibleBlock(dim=4, cond_dim=3, seed=99)
        x = np.random.randn(8, 4)
        c = np.random.randn(8, 3)
        z, _ = block.forward(x, c)
        x_rec = block.inverse(z, c)
        np.testing.assert_allclose(x_rec, x, atol=1e-10)

    def test_different_conditions_differ(self):
        block = ConditionalInvertibleBlock(dim=4, cond_dim=3)
        x = np.random.randn(2, 4)
        c1 = np.zeros((2, 3))
        c2 = np.ones((2, 3))
        z1, _ = block.forward(x, c1)
        z2, _ = block.forward(x, c2)
        assert not np.allclose(z1, z2)


# =========================================================================
# ConditionalINN
# =========================================================================
class TestConditionalINN:
    """Test stacked cINN."""

    def test_forward_inverse_roundtrip(self):
        inn = ConditionalINN(dim=4, cond_dim=3, n_blocks=3)
        x = np.random.randn(10, 4)
        c = np.random.randn(10, 3)
        z, _ = inn.forward(x, c)
        x_rec = inn.inverse(z, c)
        np.testing.assert_allclose(x_rec, x, atol=1e-8)

    def test_training_runs(self):
        inn = ConditionalINN(dim=4, cond_dim=3, n_blocks=2, hidden_dim=16)
        X_cond = np.random.randn(50, 3)
        Y_data = np.random.randn(50, 4)
        hist = inn.fit(X_cond, Y_data, n_epochs=3)
        assert "train_loss" in hist
        assert len(hist["train_loss"]) == 3

    def test_sample_before_fit_raises(self):
        inn = ConditionalINN(dim=4, cond_dim=3)
        with pytest.raises(RuntimeError, match="Not fitted"):
            inn.sample(np.zeros((1, 3)))

    def test_sample_shapes(self):
        inn = ConditionalINN(dim=4, cond_dim=3, n_blocks=2, hidden_dim=16)
        inn.fit(np.random.randn(30, 3), np.random.randn(30, 4), n_epochs=2)
        samples = inn.sample(np.random.randn(3, 3), n_samples=10)
        assert samples.shape == (3, 10, 4)

    def test_predict_mean_std(self):
        inn = ConditionalINN(dim=4, cond_dim=3, n_blocks=2, hidden_dim=16)
        inn.fit(np.random.randn(30, 3), np.random.randn(30, 4), n_epochs=2)
        mean, std = inn.predict_mean_std(np.random.randn(5, 3), n_samples=20)
        assert mean.shape == (5, 4)
        assert std.shape == (5, 4)
        assert np.all(std >= 0)


# =========================================================================
# CoKrigingSurrogate
# =========================================================================
class TestCoKrigingSurrogate:
    """Test multi-fidelity Gaussian Process."""

    @pytest.fixture
    def fitted_cok(self):
        rng = np.random.RandomState(42)
        X_lo = rng.randn(30, 3)
        Y_lo = np.sin(X_lo[:, 0]) + rng.randn(30) * 0.3
        X_hi = rng.randn(10, 3)
        Y_hi = np.sin(X_hi[:, 0]) + rng.randn(10) * 0.01
        cok = CoKrigingSurrogate(kernel_length=1.0)
        cok.fit(X_lo, Y_lo, X_hi, Y_hi)
        return cok

    def test_fit_returns_rho(self):
        cok = CoKrigingSurrogate()
        rng = np.random.RandomState(42)
        info = cok.fit(rng.randn(20, 2), rng.randn(20),
                       rng.randn(5, 2), rng.randn(5))
        assert "rho" in info
        assert np.isfinite(info["rho"])

    def test_predict_before_fit_raises(self):
        cok = CoKrigingSurrogate()
        with pytest.raises(RuntimeError, match="Not fitted"):
            cok.predict(np.zeros((1, 3)))

    def test_predict_shapes(self, fitted_cok):
        mean, std = fitted_cok.predict(np.random.randn(5, 3))
        assert mean.shape == (5,)
        assert std.shape == (5,)
        assert np.all(std >= 0)

    def test_predict_with_low_fidelity(self, fitted_cok):
        mean, std = fitted_cok.predict(np.random.randn(5, 3), np.random.randn(5))
        assert mean.shape == (5,)
        assert np.all(np.isfinite(mean))


# =========================================================================
# MultiFidelityFramework (Full Pipeline)
# =========================================================================
class TestMultiFidelityFramework:
    """Test the unified multi-fidelity API."""

    @pytest.fixture
    def framework_and_data(self):
        ds = generate_multifidelity_data(
            n_low=50, n_mid=20, n_high=15, seed=42
        )
        fw = MultiFidelityFramework(
            n_features=3, n_outputs=1, strategy="all"
        )
        return fw, ds

    def test_training_runs(self, framework_and_data):
        fw, ds = framework_and_data
        results = fw.fit(ds, n_epochs=3)
        assert "residual" in results
        assert "cinn" in results
        assert "cokriging" in results

    def test_becomes_fitted(self, framework_and_data):
        fw, ds = framework_and_data
        assert not fw._fitted
        fw.fit(ds, n_epochs=2)
        assert fw._fitted

    def test_predict_before_fit_raises(self, framework_and_data):
        fw, _ = framework_and_data
        with pytest.raises(RuntimeError, match="Not fitted"):
            fw.predict(np.zeros((1, 3)), np.zeros((1, 1)))

    def test_predict_residual(self, framework_and_data):
        fw, ds = framework_and_data
        fw.fit(ds, n_epochs=2)
        Y = fw.predict(np.zeros((3, 3)), np.zeros((3, 1)), method="residual")
        assert Y.shape == (3, 1)

    def test_predict_cokriging(self, framework_and_data):
        fw, ds = framework_and_data
        fw.fit(ds, n_epochs=2)
        Y = fw.predict(np.zeros((3, 3)), np.zeros((3, 1)), method="cokriging")
        assert Y.shape == (3, 1)

    def test_predict_with_uncertainty(self, framework_and_data):
        fw, ds = framework_and_data
        fw.fit(ds, n_epochs=2)
        mean, std = fw.predict_with_uncertainty(
            np.zeros((3, 3)), method="cinn", n_samples=10
        )
        assert mean.shape == (3, 1)
        assert std.shape == (3, 1)

    def test_compare_fidelities(self, framework_and_data):
        fw, ds = framework_and_data
        fw.fit(ds, n_epochs=2)
        metrics = fw.compare_fidelities(ds)
        assert "low_fidelity_baseline" in metrics
        assert "residual_correction" in metrics
        assert "RMSE" in metrics["low_fidelity_baseline"]

    def test_info_dict(self, framework_and_data):
        fw, _ = framework_and_data
        info = fw.get_info()
        assert info["model_type"] == "MultiFidelityFramework"
        assert info["strategy"] == "all"

    def test_summary_string(self, framework_and_data):
        fw, _ = framework_and_data
        s = fw.summary()
        assert "Multi-Fidelity" in s
        assert "CoKriging" in s


# =========================================================================
# Data Generation
# =========================================================================
class TestDataGeneration:
    """Test synthetic multi-fidelity data generator."""

    def test_default_generation(self):
        ds = generate_multifidelity_data(n_low=30, n_mid=15, n_high=5)
        X_lo, Y_lo = ds.get_level_data("coarse_RANS")
        assert X_lo.shape[0] == 30
        X_hi, Y_hi = ds.get_level_data("DNS_LES")
        assert X_hi.shape[0] == 5

    def test_finite_values(self):
        ds = generate_multifidelity_data(n_low=20, n_mid=10, n_high=5)
        for name in ["coarse_RANS", "fine_RANS", "DNS_LES"]:
            X, Y = ds.get_level_data(name)
            assert np.all(np.isfinite(X))
            assert np.all(np.isfinite(Y))

    def test_multi_output(self):
        ds = generate_multifidelity_data(
            n_low=20, n_mid=10, n_high=5, n_outputs=3
        )
        _, Y_lo = ds.get_level_data("coarse_RANS")
        assert Y_lo.shape[1] == 3
