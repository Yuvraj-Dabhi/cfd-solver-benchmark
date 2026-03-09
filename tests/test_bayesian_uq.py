#!/usr/bin/env python3
"""
Tests for Bayesian / Probabilistic UQ Upgrades
=================================================
Covers:
  - Bayesian Model Averaging
  - Active Subspace detection
  - Aleatoric/Epistemic decomposition
  - OAT vs Sobol comparison
  - Probabilistic metrics (CRPS, ELPD, calibration)

All tests use numpy/scipy only — no external APIs or SALib required.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =========================================================================
# TestBMA — Bayesian Model Averaging
# =========================================================================
class TestBMA:
    """Tests for Bayesian Model Averaging."""

    def test_single_model(self):
        """Single model should get weight 1.0."""
        from scripts.analysis.bayesian_uq import BayesianModelAveraging
        bma = BayesianModelAveraging()
        bma.add_model("SA", prediction=0.5, error=0.05)
        result = bma.compute()
        assert result.weights[0] == pytest.approx(1.0)
        assert result.weighted_mean == pytest.approx(0.5)

    def test_best_model_highest_weight(self):
        """Model with smallest error should get highest weight."""
        from scripts.analysis.bayesian_uq import BayesianModelAveraging
        bma = BayesianModelAveraging(sigma=0.05)
        bma.add_model("SA", prediction=0.65, error=0.10)
        bma.add_model("SST", prediction=0.70, error=0.02)
        bma.add_model("k-eps", prediction=0.55, error=0.15)
        result = bma.compute()
        # SST has smallest error → highest weight
        sst_idx = result.model_names.index("SST")
        assert result.weights[sst_idx] == max(result.weights)

    def test_weights_sum_to_one(self):
        """Weights must sum to 1."""
        from scripts.analysis.bayesian_uq import BayesianModelAveraging
        bma = BayesianModelAveraging()
        bma.add_model("A", prediction=0.5, error=0.05)
        bma.add_model("B", prediction=0.6, error=0.1)
        bma.add_model("C", prediction=0.4, error=0.08)
        result = bma.compute()
        assert result.weights.sum() == pytest.approx(1.0)

    def test_credible_interval_contains_mean(self):
        """95% CI should contain the weighted mean."""
        from scripts.analysis.bayesian_uq import BayesianModelAveraging
        bma = BayesianModelAveraging()
        bma.add_model("SA", prediction=0.6, error=0.05, uncertainty=0.03)
        bma.add_model("SST", prediction=0.7, error=0.03, uncertainty=0.02)
        result = bma.compute()
        assert result.credible_interval_95[0] <= result.weighted_mean
        assert result.credible_interval_95[1] >= result.weighted_mean

    def test_distribution_sampling(self):
        """BMA distribution samples should be reasonable."""
        from scripts.analysis.bayesian_uq import BayesianModelAveraging
        bma = BayesianModelAveraging(sigma=0.05)
        bma.add_model("SA", prediction=0.6, error=0.05, uncertainty=0.03)
        bma.add_model("SST", prediction=0.7, error=0.02, uncertainty=0.02)
        samples = bma.compute_distribution(n_samples=1000)
        assert len(samples) == 1000
        assert 0.3 < np.mean(samples) < 1.0

    def test_empty_raises(self):
        """Should raise when no models added."""
        from scripts.analysis.bayesian_uq import BayesianModelAveraging
        bma = BayesianModelAveraging()
        with pytest.raises(ValueError):
            bma.compute()


# =========================================================================
# TestActiveSubspace
# =========================================================================
class TestActiveSubspace:
    """Tests for Active Subspace detection."""

    def test_linear_function_1d_subspace(self):
        """Purely linear function should have 1D active subspace."""
        from scripts.analysis.bayesian_uq import ActiveSubspace
        def f(x): return 3.0 * x[0] + 0.001 * x[1]
        asub = ActiveSubspace(
            parameter_names=["dominant", "weak"],
            parameter_bounds=np.array([[0, 1], [0, 1]]),
        )
        result = asub.compute(f, n_samples=200)
        assert result.subspace_dim >= 1
        # First eigenvalue should dominate
        assert result.eigenvalues[0] > 10 * result.eigenvalues[1]

    def test_activity_scores_sum_to_one(self):
        """Activity scores should sum to ~1."""
        from scripts.analysis.bayesian_uq import ActiveSubspace
        def f(x): return x[0] ** 2 + 0.5 * x[1]
        asub = ActiveSubspace(parameter_bounds=np.array([[0, 1]] * 3))
        result = asub.compute(f, n_samples=100, n_params=3)
        assert result.activity_scores.sum() == pytest.approx(1.0, abs=0.01)

    def test_explained_variance_ratio(self):
        """EVR should sum to ~1."""
        from scripts.analysis.bayesian_uq import ActiveSubspace
        def f(x): return x[0] + x[1]
        asub = ActiveSubspace(parameter_bounds=np.array([[0, 1]] * 2))
        result = asub.compute(f, n_samples=100)
        assert result.explained_variance_ratio.sum() == pytest.approx(1.0, abs=0.01)


# =========================================================================
# TestUncertaintyDecomposer
# =========================================================================
class TestDecomposition:
    """Tests for aleatoric/epistemic decomposition."""

    def test_pure_epistemic(self):
        """Disagreeing members with zero noise → pure epistemic."""
        from scripts.analysis.bayesian_uq import UncertaintyDecomposer
        preds = np.array([
            [0.5, 0.6, 0.7],
            [0.8, 0.9, 1.0],
        ])
        result = UncertaintyDecomposer.decompose_ensemble(preds, noise_floor=0.0)
        assert result.epistemic_fraction == pytest.approx(1.0)
        assert result.aleatoric_fraction == pytest.approx(0.0)

    def test_pure_aleatoric(self):
        """Identical members with noise → pure aleatoric."""
        from scripts.analysis.bayesian_uq import UncertaintyDecomposer
        preds = np.array([
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5],
        ])
        individual_vars = np.array([
            [0.01, 0.01, 0.01],
            [0.01, 0.01, 0.01],
        ])
        result = UncertaintyDecomposer.decompose_ensemble(
            preds, individual_variances=individual_vars
        )
        assert result.aleatoric_fraction > 0.99

    def test_fractions_sum_to_one(self):
        """Epistemic + aleatoric fractions should sum to 1."""
        from scripts.analysis.bayesian_uq import UncertaintyDecomposer
        rng = np.random.default_rng(42)
        preds = rng.normal(0.5, 0.1, size=(5, 20))
        result = UncertaintyDecomposer.decompose_ensemble(preds, noise_floor=0.01)
        total = result.epistemic_fraction + result.aleatoric_fraction
        assert total == pytest.approx(1.0, abs=0.01)

    def test_noise_floor_estimation(self):
        """Noise floor should be positive for noisy data."""
        from scripts.analysis.bayesian_uq import UncertaintyDecomposer
        rng = np.random.default_rng(42)
        y_true = np.sin(np.linspace(0, 2 * np.pi, 50))
        y_pred = y_true + rng.normal(0, 0.1, 50)
        noise = UncertaintyDecomposer.estimate_noise_floor(y_true, y_pred)
        assert noise > 0


# =========================================================================
# TestSensitivityComparator
# =========================================================================
class TestSensitivityComparator:
    """Tests for OAT vs Sobol comparison."""

    def test_oat_detects_dominant_parameter(self):
        """OAT should identify the dominant parameter."""
        from scripts.analysis.bayesian_uq import SensitivityComparator
        def f(x): return 10.0 * x[0] + 0.1 * x[1] + 0.01 * x[2]
        baseline = np.array([1.0, 1.0, 1.0])
        sens, names = SensitivityComparator.oat_sensitivity(
            f, baseline, parameter_names=["Re", "TI", "Mach"]
        )
        assert names[np.argmax(sens)] == "Re"

    def test_comparison_rank_correlation(self):
        """OAT and Sobol should agree for additive models."""
        from scripts.analysis.bayesian_uq import SensitivityComparator
        def f(x): return 5.0 * x[0] + 2.0 * x[1] + 0.5 * x[2]
        baseline = np.array([0.5, 0.5, 0.5])
        comp = SensitivityComparator.compare(
            f, baseline, ["A", "B", "C"]
        )
        # For additive models, rankings should correlate well
        assert comp.rank_correlation > 0.5

    def test_interaction_index_low_for_additive(self):
        """Additive model should have low interaction index."""
        from scripts.analysis.bayesian_uq import SensitivityComparator
        def f(x): return 3.0 * x[0] + 1.0 * x[1]
        baseline = np.array([0.5, 0.5])
        comp = SensitivityComparator.compare(f, baseline, ["A", "B"])
        # Interaction index should be small for additive models
        assert comp.interaction_index < 0.5


# =========================================================================
# TestProbabilisticMetrics
# =========================================================================
class TestProbabilisticMetrics:
    """Tests for CRPS, ELPD, calibration, sharpness."""

    def test_crps_perfect_prediction(self):
        """CRPS should be small for accurate predictions."""
        from scripts.analysis.bayesian_uq import ProbabilisticMetrics
        y_true = np.array([1.0, 2.0, 3.0])
        y_mean = np.array([1.0, 2.0, 3.0])
        y_std = np.array([0.01, 0.01, 0.01])
        crps = ProbabilisticMetrics.crps_gaussian(y_true, y_mean, y_std)
        assert crps < 0.01

    def test_crps_nonnegative(self):
        """CRPS should be non-negative."""
        from scripts.analysis.bayesian_uq import ProbabilisticMetrics
        rng = np.random.default_rng(42)
        y_true = rng.normal(0, 1, 50)
        y_mean = rng.normal(0, 1, 50)
        y_std = np.abs(rng.normal(0.5, 0.2, 50))
        crps = ProbabilisticMetrics.crps_gaussian(y_true, y_mean, y_std)
        assert crps >= 0

    def test_elpd_higher_for_better_predictions(self):
        """Better predictions should have higher ELPD."""
        from scripts.analysis.bayesian_uq import ProbabilisticMetrics
        y_true = np.array([1.0, 2.0, 3.0])
        good_mean = np.array([1.01, 2.01, 3.01])
        bad_mean = np.array([5.0, 0.0, 8.0])
        y_std = np.array([0.5, 0.5, 0.5])
        elpd_good = ProbabilisticMetrics.elpd(y_true, good_mean, y_std)
        elpd_bad = ProbabilisticMetrics.elpd(y_true, bad_mean, y_std)
        assert elpd_good > elpd_bad

    def test_coverage_well_calibrated(self):
        """95% coverage should be ~0.95 for well-calibrated predictions."""
        from scripts.analysis.bayesian_uq import ProbabilisticMetrics
        rng = np.random.default_rng(42)
        n = 5000
        y_mean = np.zeros(n)
        y_std = np.ones(n)
        y_true = rng.normal(0, 1, n)  # Perfectly calibrated
        cov = ProbabilisticMetrics.coverage(y_true, y_mean, y_std, 0.95)
        assert 0.93 < cov < 0.97

    def test_sharpness_narrower_is_better(self):
        """Smaller std → smaller sharpness."""
        from scripts.analysis.bayesian_uq import ProbabilisticMetrics
        sharp = ProbabilisticMetrics.sharpness(np.array([0.01, 0.01]))
        wide = ProbabilisticMetrics.sharpness(np.array([1.0, 1.0]))
        assert sharp < wide

    def test_compute_all(self):
        """compute_all should return all scores."""
        from scripts.analysis.bayesian_uq import ProbabilisticMetrics
        rng = np.random.default_rng(42)
        y_true = rng.normal(0, 1, 100)
        y_mean = y_true + rng.normal(0, 0.1, 100)
        y_std = np.full(100, 0.5)
        scores = ProbabilisticMetrics.compute_all(y_true, y_mean, y_std)
        assert hasattr(scores, 'crps')
        assert hasattr(scores, 'elpd')
        assert hasattr(scores, 'calibration_error')
        assert hasattr(scores, 'sharpness')
        assert hasattr(scores, 'coverage_90')
        assert hasattr(scores, 'coverage_95')
