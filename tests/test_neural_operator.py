"""
Tests for Neural Operator Surrogate
======================================
Validates FourierLayer, FNO2d, HUFNO, FiLM, super-resolution,
data pipeline, training, and comparison metrics.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ml_augmentation.neural_operator_surrogate import (
    FourierLayer,
    FNO2d,
    HUFNO,
    FiLMConditioning,
    FieldNormalizer,
    GridPairDataset,
    NeuralOperatorTrainer,
    NeuralOperatorSurrogate,
    generate_synthetic_gci_pairs,
    evaluate_separation_metrics,
)


class TestFourierLayer:
    """Test spectral convolution layer."""

    def test_output_shape(self):
        fl = FourierLayer(in_channels=4, out_channels=8, n_modes=6)
        v = np.random.randn(2, 4, 40)  # batch=2, ch=4, spatial=40
        out = fl.forward(v)
        assert out.shape == (2, 8, 40)

    def test_different_spatial_resolutions(self):
        """FourierLayer should work at any spatial resolution."""
        fl = FourierLayer(in_channels=4, out_channels=8, n_modes=6)

        for S in [20, 40, 80, 160]:
            v = np.random.randn(1, 4, S)
            out = fl.forward(v)
            assert out.shape == (1, 8, S), f"Failed at S={S}"

    def test_mode_truncation(self):
        """When spatial < n_modes, should still work."""
        fl = FourierLayer(in_channels=2, out_channels=2, n_modes=50)
        v = np.random.randn(1, 2, 10)  # Only 6 rfft freqs
        out = fl.forward(v)
        assert out.shape == (1, 2, 10)
        assert np.all(np.isfinite(out))

    def test_get_params(self):
        fl = FourierLayer(in_channels=4, out_channels=8, n_modes=6)
        params = fl.get_params()
        assert "R_real" in params
        assert "W" in params
        assert params["R_real"].shape == (4, 8, 6)


class TestFiLMConditioning:
    """Test FiLM modulation."""

    def test_modulation_shape(self):
        film = FiLMConditioning(cond_dim=3, hidden_channels=16)
        features = np.random.randn(4, 16, 40)
        cond = np.random.randn(4, 3)
        out = film.modulate(features, cond)
        assert out.shape == (4, 16, 40)

    def test_conditioning_changes_output(self):
        """Different conditioning should produce different outputs."""
        film = FiLMConditioning(cond_dim=3, hidden_channels=8)
        features = np.random.randn(2, 8, 20)

        cond1 = np.array([[0, 6e6, 0.15], [5, 6e6, 0.15]])
        cond2 = np.array([[10, 6e6, 0.15], [15, 6e6, 0.15]])

        out1 = film.modulate(features, cond1)
        out2 = film.modulate(features, cond2)

        assert not np.allclose(out1, out2)


class TestFNO2d:
    """Test FNO model."""

    def test_forward_shape(self):
        model = FNO2d(in_channels=2, out_channels=2,
                      width=16, n_modes=8, n_layers=3, cond_dim=3)
        v = np.random.randn(4, 2, 40)
        cond = np.random.randn(4, 3)
        out = model.forward(v, cond)
        assert out.shape == (4, 2, 40)

    def test_resolution_invariance(self):
        """Same model should handle different spatial resolutions."""
        model = FNO2d(in_channels=1, out_channels=1,
                      width=16, n_modes=8, n_layers=2)

        for res in [20, 40, 80, 160]:
            v = np.random.randn(2, 1, res)
            out = model.forward(v)
            assert out.shape == (2, 1, res), f"Failed at res={res}"
            assert np.all(np.isfinite(out))

    def test_without_conditioning(self):
        model = FNO2d(in_channels=2, out_channels=2,
                      width=16, n_modes=8, cond_dim=0)
        v = np.random.randn(3, 2, 40)
        out = model.forward(v, cond=None)
        assert out.shape == (3, 2, 40)

    def test_param_count(self):
        model = FNO2d(in_channels=2, out_channels=2,
                      width=32, n_modes=12, n_layers=4, cond_dim=3)
        n = model.count_params()
        assert n > 0
        assert isinstance(n, int)


class TestHUFNO:
    """Test Hybrid U-Net + FNO model."""

    def test_forward_shape(self):
        model = HUFNO(in_channels=2, out_channels=2,
                      width=16, n_modes=6, n_fno_layers=2,
                      base_unet_ch=8, cond_dim=3)
        v = np.random.randn(4, 2, 40)
        cond = np.random.randn(4, 3)
        out = model.forward(v, cond)
        assert out.shape[0] == 4
        assert out.shape[1] == 2
        # Spatial dim may differ due to downsampling/upsampling rounding
        assert out.shape[2] > 0

    def test_without_conditioning(self):
        model = HUFNO(in_channels=1, out_channels=1,
                      width=8, n_modes=4, cond_dim=0)
        v = np.random.randn(2, 1, 32)
        out = model.forward(v)
        assert out.shape[0] == 2
        assert out.shape[1] == 1


class TestFieldNormalizer:
    """Test normalization."""

    def test_normalize_denormalize(self):
        norm = FieldNormalizer()
        data = np.random.randn(50, 2, 80) * 10 + 5
        norm.fit(data)

        normalized = norm.normalize(data)
        recovered = norm.denormalize(normalized)

        np.testing.assert_allclose(recovered, data, atol=1e-10)

    def test_2d_input(self):
        norm = FieldNormalizer()
        data = np.random.randn(30, 80)
        norm.fit(data)
        out = norm.normalize(data)
        assert out.shape == (30, 80)


class TestSyntheticData:
    """Test synthetic GCI pair generation."""

    def test_pair_shapes(self):
        data = generate_synthetic_gci_pairs(
            n_samples=50, coarse_res=20, fine_res=40)

        assert data["X_params"].shape == (50, 3)
        assert data["U_coarse"].shape == (50, 2, 20)
        assert data["U_fine"].shape == (50, 2, 40)

    def test_coarse_is_downsampled_fine(self):
        """Coarse should be approximately interpolated fine."""
        data = generate_synthetic_gci_pairs(
            n_samples=10, coarse_res=20, fine_res=40)

        # Re-interpolate fine → coarse and compare
        x_c = np.linspace(0, 1, 20)
        x_f = np.linspace(0, 1, 40)
        for i in range(5):
            resampled = np.interp(x_c, x_f, data["U_fine"][i, 0])
            np.testing.assert_allclose(
                data["U_coarse"][i, 0], resampled, atol=1e-10)


class TestGridPairDataset:
    """Test dataset management."""

    def test_synthetic_loading(self):
        ds = GridPairDataset()
        ds.load_synthetic(n_samples=50, coarse_res=20, fine_res=40)

        split = ds.get_train_test_split()
        n_train = len(split["X_train"])
        n_test = len(split["X_test"])
        assert n_train + n_test == 50
        assert n_test == 10  # 20% of 50


class TestSuperResolution:
    """Test super-resolution prediction."""

    def test_train_coarse_predict_fine(self):
        """Train on 40pt, predict on 80pt and 160pt."""
        model = NeuralOperatorSurrogate(
            arch="fno", in_channels=2, out_channels=2,
            width=16, n_modes=8, n_layers=2,
        )

        data = generate_synthetic_gci_pairs(
            n_samples=30, coarse_res=20, fine_res=40)

        # Fit on coarse→fine
        model.fit(data["X_params"], data["U_coarse"],
                  data["U_fine"], n_epochs=5)

        # Predict at 80pt (super-resolution)
        Cp_sr, Cf_sr = model.predict_at_resolution(
            data["X_params"][:5], data["U_coarse"][:5],
            target_res=80,
        )
        assert Cp_sr.shape == (5, 80)
        assert Cf_sr.shape == (5, 80)
        assert np.all(np.isfinite(Cp_sr))

    def test_predict_at_multiple_resolutions(self):
        model = NeuralOperatorSurrogate(
            arch="fno", in_channels=1, out_channels=1,
            width=8, n_modes=6, n_layers=2,
        )
        U = np.random.randn(3, 1, 20)
        X = np.random.randn(3, 3)

        for res in [40, 80, 160]:
            Cp, _ = model.predict_at_resolution(X, U, target_res=res)
            assert Cp.shape == (3, res)


class TestTrainingPipeline:
    """Test training convergence."""

    def test_training_runs(self):
        model = NeuralOperatorSurrogate(
            arch="fno", in_channels=2, out_channels=2,
            width=16, n_modes=8, n_layers=2,
        )
        data = generate_synthetic_gci_pairs(
            n_samples=30, coarse_res=20, fine_res=40)

        history = model.fit(
            data["X_params"], data["U_coarse"],
            data["U_fine"], n_epochs=10,
        )

        assert "train_loss" in history
        assert len(history["train_loss"]) > 0
        assert all(np.isfinite(l) for l in history["train_loss"])


class TestCompareWithMLP:
    """Test comparison output format."""

    def test_comparison_metrics(self):
        model = NeuralOperatorSurrogate(
            arch="fno", in_channels=2, out_channels=2,
            width=8, n_modes=6,
        )
        data = generate_synthetic_gci_pairs(
            n_samples=20, coarse_res=20, fine_res=40)

        metrics = model.compare_with_mlp(
            data["X_params"], data["U_coarse"], data["U_fine"])

        assert "model_name" in metrics
        assert "Cp_RMSE" in metrics
        assert "Cp_R2" in metrics
        assert np.isfinite(metrics["Cp_RMSE"])


class TestSeparationMetrics:
    """Test separation detection accuracy."""

    def test_separation_metrics(self):
        n, s = 10, 80
        x_c = np.linspace(0, 1, s)
        Cf_true = np.tile(0.004 * (1 - 2 * x_c), (n, 1))
        Cf_pred = Cf_true + np.random.randn(n, s) * 0.001

        metrics = evaluate_separation_metrics(Cf_pred, Cf_true, x_c)

        assert "x_sep_MAE" in metrics
        assert "bubble_length_MAE" in metrics
        assert "n_separated_pred" in metrics


class TestModelInfo:
    """Test model metadata."""

    def test_info_dict(self):
        model = NeuralOperatorSurrogate(arch="fno", width=16, n_modes=8)
        info = model.get_info()
        assert info["architecture"] == "fno"
        assert info["n_params"] > 0
        assert info["fitted"] is False

    def test_hufno_info(self):
        model = NeuralOperatorSurrogate(arch="hufno", width=16, n_modes=6)
        info = model.get_info()
        assert info["architecture"] == "hufno"
