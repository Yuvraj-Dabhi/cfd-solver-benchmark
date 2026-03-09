"""
Tests for Non-Intrusive ML Separation Correction
==================================================
Validates feature extraction, model training, correction prediction,
and the error-reduction evaluation pipeline.

Run: pytest tests/test_separation_correction.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =========================================================================
# Feature Extractor Tests
# =========================================================================
class TestSeparationFeatureExtractor:
    """Test physics-informed feature extraction."""

    @pytest.fixture
    def extractor(self):
        from scripts.ml_augmentation.separation_correction import (
            SeparationFeatureExtractor,
        )
        return SeparationFeatureExtractor()

    @pytest.fixture
    def sample_data(self):
        """Generate sample Cp/Cf surface data."""
        x = np.linspace(-0.5, 2.0, 300)
        Cp = np.zeros_like(x)
        hump = (x >= 0) & (x <= 0.65)
        Cp[hump] = -0.8 * np.sin(np.pi * x[hump] / 0.65)
        recov = x > 0.65
        Cp[recov] = -0.8 * np.exp(-2.5 * (x[recov] - 0.65))

        Cf = np.ones_like(x) * 0.003
        sep = (x >= 0.665) & (x <= 1.1)
        t = (x[sep] - 0.665) / 0.435
        Cf[sep] = -0.002 * np.sin(np.pi * t)
        return x, Cp, Cf

    def test_extract_returns_8_features(self, extractor, sample_data):
        x, Cp, Cf = sample_data
        features = extractor.extract(x, Cp, Cf, "SA")
        assert features.shape == (8,)

    def test_feature_values_finite(self, extractor, sample_data):
        x, Cp, Cf = sample_data
        features = extractor.extract(x, Cp, Cf, "SA")
        assert np.all(np.isfinite(features))

    def test_cp_slope_nonzero_at_separation(self, extractor, sample_data):
        x, Cp, Cf = sample_data
        features = extractor.extract(x, Cp, Cf, "SA")
        # f1: Cp slope at separation captures APG signal (non-zero)
        assert abs(features[0]) > 0.1  # Non-trivial pressure gradient

    def test_cf_min_negative(self, extractor, sample_data):
        x, Cp, Cf = sample_data
        features = extractor.extract(x, Cp, Cf, "SA")
        assert features[1] < 0  # Cf_min is negative in separation

    def test_model_encoding(self, extractor, sample_data):
        x, Cp, Cf = sample_data
        f_sa = extractor.extract(x, Cp, Cf, "SA")
        f_sst = extractor.extract(x, Cp, Cf, "SST")
        assert f_sa[7] == 0.0
        assert f_sst[7] == 1.0

    def test_extract_batch(self, extractor, sample_data):
        x, Cp, Cf = sample_data
        samples = [{"x": x, "Cp": Cp, "Cf": Cf, "model": "SA"}] * 5
        batch = extractor.extract_batch(samples)
        assert batch.shape == (5, 8)

    def test_bubble_length_estimation(self, extractor, sample_data):
        x, Cp, Cf = sample_data
        L = extractor._estimate_bubble_length(x, Cf)
        assert L is not None
        assert 0.3 < L < 0.6

    def test_peak_suction_captured(self, extractor, sample_data):
        x, Cp, Cf = sample_data
        features = extractor.extract(x, Cp, Cf, "SA")
        # f6: peak suction should be negative
        assert features[5] < -0.5


# =========================================================================
# Correction Model Tests
# =========================================================================
class TestSeparationCorrectionModel:
    """Test ensemble MLP correction model."""

    @pytest.fixture
    def trained_model(self):
        from scripts.ml_augmentation.separation_correction import (
            SeparationCorrectionModel, CorrectionTrainer,
        )
        trainer = CorrectionTrainer(n_samples_per_model=50)
        features, corrections, _ = trainer.generate_training_data()
        model = SeparationCorrectionModel(n_ensemble=3)
        model.train(features, corrections)
        return model, features

    def test_model_trains(self, trained_model):
        model, _ = trained_model
        assert model._trained
        assert len(model.models) == 3

    def test_predict_shape_single(self, trained_model):
        model, features = trained_model
        pred = model.predict(features[0])
        assert pred.shape == (4,)

    def test_predict_shape_batch(self, trained_model):
        model, features = trained_model
        pred = model.predict(features[:5])
        assert pred.shape == (5, 4)

    def test_predict_with_uncertainty(self, trained_model):
        model, features = trained_model
        mean, std = model.predict_with_uncertainty(features[0])
        assert mean.shape == (4,)
        assert std.shape == (4,)
        assert np.all(std >= 0)

    def test_corrections_reasonable_magnitude(self, trained_model):
        model, features = trained_model
        pred = model.predict(features[0])
        # Corrections should be small (< 0.1 for x_sep, x_reatt)
        assert abs(pred[0]) < 0.2  # Δx_sep
        assert abs(pred[1]) < 0.2  # Δx_reatt

    def test_untrained_raises(self):
        from scripts.ml_augmentation.separation_correction import (
            SeparationCorrectionModel,
        )
        model = SeparationCorrectionModel()
        with pytest.raises(RuntimeError, match="not trained"):
            model.predict(np.zeros(8))


# =========================================================================
# Training Data Tests
# =========================================================================
class TestCorrectionTrainer:
    """Test training data generation."""

    def test_generate_data_shapes(self):
        from scripts.ml_augmentation.separation_correction import CorrectionTrainer
        trainer = CorrectionTrainer(n_samples_per_model=10)
        features, corrections, raw = trainer.generate_training_data()
        assert features.shape == (30, 8)  # 3 models × 10
        assert corrections.shape == (30, 4)
        assert raw.shape == (30, 4)

    def test_corrections_centered_near_zero(self):
        from scripts.ml_augmentation.separation_correction import CorrectionTrainer
        trainer = CorrectionTrainer(n_samples_per_model=50)
        _, corrections, _ = trainer.generate_training_data()
        # Mean corrections should be small (centered on bias)
        assert abs(np.mean(corrections[:, 0])) < 0.05  # Δx_sep

    def test_features_finite(self):
        from scripts.ml_augmentation.separation_correction import CorrectionTrainer
        trainer = CorrectionTrainer(n_samples_per_model=20)
        features, _, _ = trainer.generate_training_data()
        assert np.all(np.isfinite(features))


# =========================================================================
# Training Result Tests
# =========================================================================
class TestTrainingResult:
    """Test training result metrics."""

    def test_training_r2_positive(self):
        from scripts.ml_augmentation.separation_correction import (
            SeparationCorrectionModel, CorrectionTrainer,
        )
        trainer = CorrectionTrainer(n_samples_per_model=80)
        features, corrections, _ = trainer.generate_training_data()
        model = SeparationCorrectionModel(n_ensemble=3)
        result = model.train(features, corrections)
        assert result.train_r2 > 0  # Should learn something

    def test_cv_r2_computed(self):
        from scripts.ml_augmentation.separation_correction import (
            SeparationCorrectionModel, CorrectionTrainer,
        )
        trainer = CorrectionTrainer(n_samples_per_model=50)
        features, corrections, _ = trainer.generate_training_data()
        model = SeparationCorrectionModel(n_ensemble=2)
        result = model.train(features, corrections)
        assert not np.isnan(result.cv_r2_mean)
        assert result.cv_r2_std >= 0


# =========================================================================
# Correction Evaluator Tests
# =========================================================================
class TestCorrectionEvaluator:
    """Test error-reduction evaluation."""

    @pytest.fixture
    def evaluator(self):
        from scripts.ml_augmentation.separation_correction import (
            SeparationCorrectionModel, SeparationFeatureExtractor,
            CorrectionTrainer, CorrectionEvaluator,
        )
        trainer = CorrectionTrainer(n_samples_per_model=80)
        features, corrections, _ = trainer.generate_training_data()
        model = SeparationCorrectionModel(n_ensemble=3)
        model.train(features, corrections)
        extractor = SeparationFeatureExtractor()
        return CorrectionEvaluator(model, extractor)

    def test_evaluate_all_models(self, evaluator):
        results = evaluator.evaluate_all_models()
        assert "SA" in results
        assert "SST" in results
        assert "kEpsilon" in results

    def test_corrected_metrics_finite(self, evaluator):
        results = evaluator.evaluate_all_models()
        for model, result in results.items():
            assert np.isfinite(result.corrected_x_sep)
            assert np.isfinite(result.corrected_x_reatt)

    def test_format_table(self, evaluator):
        results = evaluator.evaluate_all_models()
        table = evaluator.format_error_reduction_table(results)
        assert "AVERAGE" in table
        assert "SA" in table
        assert "%" in table


# =========================================================================
# CorrectionResult Tests
# =========================================================================
class TestCorrectionResult:
    """Test CorrectionResult computations."""

    def test_error_reduction_calculated(self):
        from scripts.ml_augmentation.separation_correction import (
            CorrectionResult, SeparationMetrics,
        )
        raw = SeparationMetrics(x_sep=0.670, x_reatt=1.15, Cf_min=-0.002)
        result = CorrectionResult(
            raw_metrics=raw,
            corrected_x_sep=0.666,
            corrected_x_reatt=1.11,
            exp_x_sep=0.665,
            exp_x_reatt=1.10,
        )
        assert result.raw_sep_error == pytest.approx(0.005, abs=1e-6)
        assert result.corrected_sep_error == pytest.approx(0.001, abs=1e-6)
        assert result.sep_error_reduction_pct == pytest.approx(80.0, abs=1e-6)

    def test_zero_raw_error_no_crash(self):
        from scripts.ml_augmentation.separation_correction import (
            CorrectionResult, SeparationMetrics,
        )
        raw = SeparationMetrics(x_sep=0.665, x_reatt=1.10)
        result = CorrectionResult(
            raw_metrics=raw,
            corrected_x_sep=0.665, corrected_x_reatt=1.10,
        )
        assert result.sep_error_reduction_pct == 0.0


# =========================================================================
# Integration: Full Pipeline
# =========================================================================
class TestFullPipeline:
    """Test the complete separation correction pipeline."""

    def test_pipeline_runs(self):
        from scripts.ml_augmentation.separation_correction import (
            run_separation_correction_pipeline,
        )
        result = run_separation_correction_pipeline(n_samples=30, n_ensemble=2)
        assert "train_result" in result
        assert "results" in result
        assert len(result["results"]) == 3
