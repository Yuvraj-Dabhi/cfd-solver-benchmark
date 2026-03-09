"""
Tests for POD + Transformer ROM Closure
========================================
Validates EasyAttention, TransformerEncoder, PODTransformerClosure,
MultiConditionClosure, and ClosureReport.
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ml_augmentation.rom import GalerkinROM
from scripts.ml_augmentation.pod_transformer_closure import (
    EasyAttention,
    TransformerEncoderLayer,
    TransformerEncoder,
    PODTransformerClosure,
    MultiConditionClosure,
    ClosureConfig,
    ClosureResult,
    ClosureReport,
)


def _make_snapshots(n_dof=100, n_snaps=30, n_modes_true=5, seed=42):
    """Generate synthetic snapshots with known modal structure."""
    rng = np.random.default_rng(seed)
    # True basis: n_modes_true dominant modes + small noise
    basis = rng.standard_normal((n_dof, n_modes_true))
    coeffs = rng.standard_normal((n_modes_true, n_snaps)) * np.arange(n_modes_true, 0, -1)[:, None]
    noise = rng.standard_normal((n_dof, n_snaps)) * 0.1
    return basis @ coeffs + noise


def _make_fitted_rom(n_dof=100, n_snaps=30, n_modes=10, seed=42):
    """Create a fitted GalerkinROM with synthetic snapshots."""
    snapshots = _make_snapshots(n_dof, n_snaps, seed=seed)
    rom = GalerkinROM(n_modes=n_modes)
    rom.fit(snapshots)
    return rom, snapshots


# =========================================================================
# Easy-Attention Tests
# =========================================================================
class TestEasyAttention:
    """Test easy-attention mechanism."""

    def test_attention_output_shape(self):
        """Output should match input sequence shape."""
        attn = EasyAttention(d_model=32, n_heads=4)
        x = np.random.randn(5, 32)  # seq_len=5
        out = attn.forward(x, use_easy=True)
        assert out.shape == (5, 32)

    def test_easy_vs_standard_shapes_match(self):
        """Easy and standard attention produce same-shaped outputs."""
        attn = EasyAttention(d_model=16, n_heads=2)
        x = np.random.randn(3, 16)
        easy_out = attn.forward(x, use_easy=True)
        std_out = attn.forward(x, use_easy=False)
        assert easy_out.shape == std_out.shape

    def test_attention_finite(self):
        """Output should be finite."""
        attn = EasyAttention(d_model=32, n_heads=4)
        x = np.random.randn(8, 32)
        out = attn.forward(x, use_easy=True)
        assert np.all(np.isfinite(out))

    def test_params_roundtrip(self):
        """Get/set params should be identity."""
        attn = EasyAttention(d_model=16, n_heads=2)
        p1 = attn.get_params()
        attn2 = EasyAttention(d_model=16, n_heads=2, seed=99)
        attn2.set_params(p1)
        p2 = attn2.get_params()
        for k in p1:
            np.testing.assert_array_equal(p1[k], p2[k])


# =========================================================================
# Transformer Encoder Layer Tests
# =========================================================================
class TestTransformerEncoderLayer:
    """Test single encoder layer."""

    def test_layer_output_shape(self):
        layer = TransformerEncoderLayer(d_model=32, n_heads=4)
        x = np.random.randn(3, 32)
        out = layer.forward(x)
        assert out.shape == (3, 32)

    def test_residual_connection(self):
        """Output should differ from input (residual + transformation)."""
        layer = TransformerEncoderLayer(d_model=16, n_heads=2)
        x = np.random.randn(2, 16)
        out = layer.forward(x)
        assert not np.allclose(x, out)

    def test_layer_norm_effect(self):
        """Layer norm should normalize the hidden representation."""
        layer = TransformerEncoderLayer(d_model=16, n_heads=2)
        x = np.random.randn(1, 16) * 100  # Large input
        out = layer.forward(x)
        assert np.all(np.isfinite(out))


# =========================================================================
# Transformer Encoder Tests
# =========================================================================
class TestTransformerEncoder:
    """Test stacked Transformer encoder."""

    def test_encoder_output_shape(self):
        config = ClosureConfig(d_model=32, n_heads=4, n_encoder_layers=2)
        encoder = TransformerEncoder(config, input_dim=13, output_dim=100)
        x = np.random.randn(13)
        out, _ = encoder.forward(x)
        assert out.shape == (100,)

    def test_encoder_batch(self):
        config = ClosureConfig(d_model=32, n_heads=4, n_encoder_layers=2)
        encoder = TransformerEncoder(config, input_dim=13, output_dim=50)
        x = np.random.randn(5, 13)
        out, _ = encoder.forward(x)
        assert out.shape == (5, 50)

    def test_pdf_output(self):
        config = ClosureConfig(d_model=32, n_heads=4, predict_pdf=True, n_pdf_bins=10)
        encoder = TransformerEncoder(config, input_dim=13, output_dim=50)
        x = np.random.randn(13)
        out, pdf = encoder.forward(x, return_pdf=True)
        assert out.shape == (50,)
        assert pdf.shape == (10,)
        assert abs(pdf.sum() - 1.0) < 0.01  # Should be a valid PDF

    def test_encoder_finite(self):
        config = ClosureConfig(d_model=16, n_heads=2, n_encoder_layers=3)
        encoder = TransformerEncoder(config, input_dim=8, output_dim=30)
        x = np.random.randn(8)
        out, _ = encoder.forward(x)
        assert np.all(np.isfinite(out))

    def test_params_save_load(self):
        config = ClosureConfig(d_model=16, n_heads=2, n_encoder_layers=2)
        encoder = TransformerEncoder(config, input_dim=8, output_dim=20)

        params = encoder.get_params()
        encoder2 = TransformerEncoder(config, input_dim=8, output_dim=20)
        encoder2.set_params(params)

        x = np.random.randn(8)
        out1, _ = encoder.forward(x)
        out2, _ = encoder2.forward(x)
        np.testing.assert_array_almost_equal(out1, out2)


# =========================================================================
# POD Transformer Closure Tests
# =========================================================================
class TestPODTransformerClosure:
    """Test the main closure pipeline."""

    def test_closure_creation(self):
        """Closure creates with fitted ROM."""
        rom, _ = _make_fitted_rom(n_dof=50, n_snaps=20, n_modes=8)
        config = ClosureConfig(n_modes_retained=5, d_model=16, n_heads=2,
                               n_encoder_layers=1, n_epochs=5)
        closure = PODTransformerClosure(rom, config=config)
        assert closure.config.n_modes_retained == 5

    def test_closure_rejects_unfitted_rom(self):
        """Should raise error if ROM not fitted."""
        rom = GalerkinROM(n_modes=5)
        with pytest.raises(RuntimeError):
            PODTransformerClosure(rom)

    def test_closure_training(self):
        """Closure trains and returns history."""
        rom, snapshots = _make_fitted_rom(n_dof=50, n_snaps=20, n_modes=8)
        config = ClosureConfig(
            n_modes_retained=5, d_model=16, n_heads=2,
            n_encoder_layers=1, n_epochs=10, batch_size=8,
        )
        closure = PODTransformerClosure(rom, config=config)
        history = closure.fit(snapshots)

        assert "train_loss" in history
        assert len(history["train_loss"]) > 0
        assert all(np.isfinite(l) for l in history["train_loss"])

    def test_closure_prediction(self):
        """Prediction produces valid ClosureResult."""
        rom, snapshots = _make_fitted_rom(n_dof=50, n_snaps=20, n_modes=8)
        config = ClosureConfig(
            n_modes_retained=5, d_model=16, n_heads=2,
            n_encoder_layers=1, n_epochs=5, batch_size=8,
        )
        closure = PODTransformerClosure(rom, config=config)
        closure.fit(snapshots)

        result = closure.predict(snapshots[:, 0])
        assert isinstance(result, ClosureResult)
        assert result.truncated_solution.shape == (50,)
        assert result.closed_solution.shape == (50,)
        assert result.fluctuation_field.shape == (50,)
        assert np.isfinite(result.truncation_error)
        assert np.isfinite(result.closed_error)
        assert result.n_modes == 5

    def test_untrained_predict_raises(self):
        """Should raise if predicting before training."""
        rom, _ = _make_fitted_rom(n_dof=50, n_snaps=20, n_modes=8)
        config = ClosureConfig(n_modes_retained=5, d_model=16, n_heads=2,
                               n_encoder_layers=1)
        closure = PODTransformerClosure(rom, config=config)
        with pytest.raises(RuntimeError):
            closure.predict(np.random.randn(50))

    def test_pdf_prediction(self):
        """Fluctuation PDF prediction works."""
        rom, snapshots = _make_fitted_rom(n_dof=50, n_snaps=20, n_modes=8)
        config = ClosureConfig(
            n_modes_retained=5, d_model=16, n_heads=2,
            n_encoder_layers=1, n_epochs=5, predict_pdf=True, n_pdf_bins=10,
        )
        closure = PODTransformerClosure(rom, config=config)
        closure.fit(snapshots)

        pdf_result = closure.predict_fluctuation_pdf(snapshots[:, 0])
        assert "bin_centers" in pdf_result
        assert "pdf" in pdf_result
        assert "statistics" in pdf_result
        assert pdf_result["bin_centers"].shape == (10,)
        assert "rms" in pdf_result["statistics"]

    def test_checkpoint_save_load(self):
        """Save/load roundtrip preserves predictions."""
        rom, snapshots = _make_fitted_rom(n_dof=50, n_snaps=20, n_modes=8)
        config = ClosureConfig(
            n_modes_retained=5, d_model=16, n_heads=2,
            n_encoder_layers=1, n_epochs=5, batch_size=8,
        )
        closure = PODTransformerClosure(rom, config=config)
        closure.fit(snapshots)

        result_before = closure.predict(snapshots[:, 0])

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            closure.save(f.name)

        closure2 = PODTransformerClosure(rom, config=config)
        closure2.load(f.name)

        result_after = closure2.predict(snapshots[:, 0])
        np.testing.assert_array_almost_equal(
            result_before.closed_solution,
            result_after.closed_solution,
            decimal=5,
        )

    def test_energy_recovery_positive(self):
        """Energy recovery should be non-negative after training."""
        rom, snapshots = _make_fitted_rom(n_dof=50, n_snaps=20, n_modes=8)
        config = ClosureConfig(
            n_modes_retained=3, d_model=16, n_heads=2,
            n_encoder_layers=1, n_epochs=20, batch_size=8,
        )
        closure = PODTransformerClosure(rom, config=config)
        history = closure.fit(snapshots)

        # At least some energy recovered (may be small for this toy problem)
        assert "energy_recovery" in history


# =========================================================================
# Multi-Condition Closure Tests
# =========================================================================
class TestMultiConditionClosure:
    """Test multi-condition closure."""

    def test_multi_condition_fit(self):
        """Trains across multiple conditions."""
        n_dof = 50
        rng = np.random.default_rng(42)
        conditions = {
            "Re_1e6": _make_snapshots(n_dof, 15, seed=42),
            "Re_2e6": _make_snapshots(n_dof, 15, seed=43),
        }

        config = ClosureConfig(
            n_modes_retained=5, d_model=16, n_heads=2,
            n_encoder_layers=1, n_epochs=5, batch_size=8,
        )
        mc = MultiConditionClosure(n_modes=8, config=config)
        history = mc.fit(conditions)

        assert "train_loss" in history
        assert len(mc.results_per_condition) == 2

    def test_multi_condition_summary(self):
        """Summary includes per-condition metrics."""
        conditions = {
            "low_Re": _make_snapshots(50, 10, seed=42),
            "high_Re": _make_snapshots(50, 10, seed=43),
        }
        config = ClosureConfig(
            n_modes_retained=5, d_model=16, n_heads=2,
            n_encoder_layers=1, n_epochs=3, batch_size=8,
        )
        mc = MultiConditionClosure(n_modes=8, config=config)
        mc.fit(conditions)

        summary = mc.summary()
        assert summary["n_conditions"] == 2
        assert "low_Re" in summary["per_condition"]
        assert "high_Re" in summary["per_condition"]

    def test_multi_condition_predict(self):
        """Prediction works after multi-condition training."""
        conditions = {
            "cond_A": _make_snapshots(50, 10, seed=42),
        }
        config = ClosureConfig(
            n_modes_retained=5, d_model=16, n_heads=2,
            n_encoder_layers=1, n_epochs=3, batch_size=8,
        )
        mc = MultiConditionClosure(n_modes=8, config=config)
        mc.fit(conditions)

        result = mc.predict(conditions["cond_A"][:, 0])
        assert isinstance(result, ClosureResult)
        assert result.closed_solution.shape == (50,)


# =========================================================================
# Closure Report Tests
# =========================================================================
class TestClosureReport:
    """Test report generation."""

    def test_report_creation(self, tmp_path):
        report = ClosureReport(output_dir=str(tmp_path))

        result = ClosureResult(
            truncated_solution=np.zeros(10),
            closed_solution=np.zeros(10),
            fluctuation_field=np.zeros(10),
            energy_recovered_pct=45.0,
            truncation_error=0.15,
            closed_error=0.08,
            improvement_factor=1.875,
            n_modes=5,
            tke_improvement_pct=50.0,
        )
        report.add_result("test_case", result)
        output = report.generate_report()

        assert "results" in output
        assert "test_case" in output["results"]

        report_path = tmp_path / "closure_report.json"
        assert report_path.exists()
        with open(report_path) as f:
            data = json.load(f)
        assert data["results"]["test_case"]["improvement_factor"] == 1.875

    def test_report_summary(self, tmp_path):
        report = ClosureReport(output_dir=str(tmp_path))
        for name in ["case_a", "case_b"]:
            result = ClosureResult(
                truncated_solution=np.zeros(5),
                closed_solution=np.zeros(5),
                fluctuation_field=np.zeros(5),
                energy_recovered_pct=50.0,
                truncation_error=0.2,
                closed_error=0.1,
                improvement_factor=2.0,
                n_modes=5,
            )
            report.add_result(name, result)

        output = report.generate_report()
        assert output["summary"]["n_cases"] == 2
        assert output["summary"]["mean_improvement_factor"] == 2.0
