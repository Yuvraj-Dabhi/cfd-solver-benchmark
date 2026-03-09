#!/usr/bin/env python3
"""
Tests for Calibrated Stochastic ML Closures
=============================================
Validates coverage calibration, space-dependent aggregation,
extended error budgets, and report generation.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ml_augmentation.calibrated_stochastic_closures import (
    AggregationResult,
    BNNClosureWrapper,
    CalibrationCase,
    CalibrationDataGenerator,
    CoverageCalibrator,
    CoverageResult,
    DiffusionClosureWrapper,
    EnsembleClosureWrapper,
    ExtendedBudgetEntry,
    ExtendedErrorBudget,
    SpaceDependentAggregator,
    StochasticClosureExperiment,
    StochasticClosureReport,
    create_default_stochastic_wrappers,
    run_calibration_study,
)


# =========================================================================
# TestCalibrationCase
# =========================================================================
class TestCalibrationCase:
    """Tests for case definition and data generation."""

    def test_default_cases(self):
        gen = CalibrationDataGenerator()
        assert len(gen.cases) == 3

    def test_data_shapes(self):
        cases = [CalibrationCase("test", "T", Re=1e5, n_points=50)]
        gen = CalibrationDataGenerator(cases=cases, seed=42)
        data = gen.generate_all()
        assert "test" in data
        d = data["test"]
        assert d["x_coord"].shape == (50,)
        assert d["X_features"].shape == (50, 5)
        assert d["Cf_dns"].shape == (50,)
        assert d["Cf_rans"].shape == (50,)
        assert d["Cp_dns"].shape == (50,)

    def test_all_three_generated(self):
        gen = CalibrationDataGenerator(seed=42)
        data = gen.generate_all()
        assert set(data.keys()) == {"wall_hump", "bfs", "periodic_hill"}


# =========================================================================
# TestStochasticClosureWrappers
# =========================================================================
class TestStochasticClosureWrappers:
    """Tests for each stochastic closure wrapper."""

    @pytest.fixture
    def data(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((80, 5))
        Y = np.sin(X[:, 0]) + 0.1 * rng.standard_normal(80)
        return X, Y

    def test_bnn_wrapper(self, data):
        X, Y = data
        w = BNNClosureWrapper(n_ensemble=2, seed=42)
        w.fit(X, Y)
        mean, lo, hi = w.predict_intervals(X, confidence=0.95)
        assert mean.shape[0] == 80
        assert np.all(lo <= hi)

    def test_ensemble_wrapper(self, data):
        X, Y = data
        w = EnsembleClosureWrapper(n_models=3, seed=42)
        w.fit(X, Y)
        mean, lo, hi = w.predict_intervals(X, confidence=0.95)
        assert np.all(lo <= hi)

    def test_diffusion_wrapper(self, data):
        X, Y = data
        w = DiffusionClosureWrapper(n_samples=4, seed=42)
        w.fit(X, Y)
        mean, lo, hi = w.predict_intervals(X, confidence=0.90)
        assert np.all(lo <= hi)

    def test_create_defaults(self):
        wrappers = create_default_stochastic_wrappers()
        assert len(wrappers) == 3
        names = {w.name for w in wrappers}
        assert names == {"BNN", "DeepEnsemble", "DiffusionSurrogate"}


# =========================================================================
# TestCoverageCalibrator
# =========================================================================
class TestCoverageCalibrator:
    """Tests for coverage and calibration metrics."""

    def test_perfect_coverage(self):
        n = 100
        y = np.linspace(0, 1, n)
        mean = y.copy()
        lo = y - 10.0  # very wide
        hi = y + 10.0
        cal = CoverageCalibrator(confidence=0.95)
        cr = cal.evaluate(y, mean, lo, hi, "test_model", "test_case", "Cf")
        assert cr.coverage_pct == 100.0

    def test_zero_coverage(self):
        n = 100
        y = np.ones(n)
        mean = np.zeros(n)
        lo = mean - 0.001
        hi = mean + 0.001
        cal = CoverageCalibrator(confidence=0.95)
        cr = cal.evaluate(y, mean, lo, hi, "test_model", "test_case", "Cf")
        assert cr.coverage_pct < 5.0

    def test_calibrated_flag(self):
        n = 200
        rng = np.random.default_rng(42)
        y = rng.standard_normal(n)
        mean = y + rng.standard_normal(n) * 0.1
        lo = mean - 3.0
        hi = mean + 3.0
        cal = CoverageCalibrator(confidence=0.95)
        cr = cal.evaluate(y, mean, lo, hi)
        # Wide bounds should give ~100% coverage — within 10% of 95%
        assert cr.is_well_calibrated

    def test_ece_finite(self):
        n = 100
        rng = np.random.default_rng(42)
        y = rng.standard_normal(n)
        mean = y + 0.1
        lo = mean - 1.0
        hi = mean + 1.0
        cal = CoverageCalibrator(confidence=0.95)
        cr = cal.evaluate(y, mean, lo, hi)
        assert np.isfinite(cr.ece)


# =========================================================================
# TestSpaceDependentAggregator
# =========================================================================
class TestSpaceDependentAggregator:
    """Tests for space-dependent multi-model aggregation."""

    def _make_predictions(self, n=100):
        rng = np.random.default_rng(42)
        preds = []
        for i in range(3):
            mean = rng.standard_normal(n)
            width = 0.5 + rng.uniform(0, 1, n)
            lo = mean - width
            hi = mean + width
            preds.append((mean, lo, hi))
        return preds

    def test_weights_sum_to_one(self):
        x = np.linspace(0, 1, 100)
        preds = self._make_predictions(100)
        agg = SpaceDependentAggregator()
        result = agg.aggregate(x, preds, ["A", "B", "C"])
        weight_sums = result.weights.sum(axis=0)
        np.testing.assert_allclose(weight_sums, 1.0, atol=1e-10)

    def test_combined_shapes(self):
        x = np.linspace(0, 1, 50)
        preds = self._make_predictions(50)
        agg = SpaceDependentAggregator()
        result = agg.aggregate(x, preds, ["A", "B", "C"])
        assert result.combined_mean.shape == (50,)
        assert result.combined_std.shape == (50,)
        assert result.weights.shape == (3, 50)
        assert result.regime_labels.shape == (50,)

    def test_zone_weights_populated(self):
        x = np.linspace(0, 1, 100)
        preds = self._make_predictions(100)
        agg = SpaceDependentAggregator()
        result = agg.aggregate(x, preds, ["A", "B", "C"])
        assert len(result.per_zone_weights) > 0
        for zone, weights in result.per_zone_weights.items():
            total = sum(weights.values())
            assert abs(total - 1.0) < 0.05  # weights approximately sum to 1

    def test_regime_classification(self):
        x = np.linspace(0, 1, 100)
        preds = self._make_predictions(100)
        agg = SpaceDependentAggregator(regime_boundaries=(0.3, 0.7))
        result = agg.aggregate(x, preds, ["A", "B", "C"])
        assert 0 in result.regime_labels  # attached
        assert 1 in result.regime_labels  # separated
        assert 2 in result.regime_labels  # reattaching


# =========================================================================
# TestExtendedErrorBudget
# =========================================================================
class TestExtendedErrorBudget:
    """Tests for extended RSS error budget."""

    def test_rss_includes_all(self):
        crs = [
            CoverageResult("BNN", "wall_hump", "Cf", mean_interval_width=0.01),
            CoverageResult("DeepEnsemble", "wall_hump", "Cf", mean_interval_width=0.02),
        ]
        budget = ExtendedErrorBudget()
        entry = budget.build_entry("wall_hump", "Cf", crs, dns_rmse=0.005, dns_scale=0.01)
        assert entry.total_rss_pct > 0
        # RSS >= max component
        components = [entry.gci_pct, entry.rans_model_pct, entry.aggregated_ml_pct]
        assert entry.total_rss_pct >= max(components) - 0.01

    def test_unknown_case_defaults(self):
        budget = ExtendedErrorBudget()
        entry = budget.build_entry("unknown_case", "CL", [], dns_rmse=0, dns_scale=1)
        assert entry.gci_pct == 2.0  # default
        assert entry.total_rss_pct > 0

    def test_per_model_epistemic(self):
        crs = [
            CoverageResult("BNN", "bfs", "Cf", mean_interval_width=0.05),
            CoverageResult("DeepEnsemble", "bfs", "Cf", mean_interval_width=0.03),
            CoverageResult("DiffusionSurrogate", "bfs", "Cf", mean_interval_width=0.04),
        ]
        budget = ExtendedErrorBudget()
        entry = budget.build_entry("bfs", "Cf", crs, dns_scale=0.01)
        assert entry.bnn_epistemic_pct > 0
        assert entry.ensemble_epistemic_pct > 0
        assert entry.diffusion_epistemic_pct > 0


# =========================================================================
# TestStochasticClosureExperiment
# =========================================================================
class TestStochasticClosureExperiment:
    """Tests for the experiment runner."""

    def _make_mini_experiment(self):
        cases = [CalibrationCase("test_a", "A", Re=1e5, n_points=40)]
        wrappers = [
            BNNClosureWrapper(n_ensemble=2, seed=42),
            EnsembleClosureWrapper(n_models=2, seed=42),
        ]
        return StochasticClosureExperiment(
            cases=cases, wrappers=wrappers, seed=42,
        )

    def test_run_completes(self):
        exp = self._make_mini_experiment()
        exp.run()
        assert len(exp.coverage_results) > 0
        assert len(exp.budget_entries) > 0

    def test_coverage_results_populated(self):
        exp = self._make_mini_experiment()
        exp.run()
        # 1 case × 2 quantities × 2 models = 4
        assert len(exp.coverage_results) == 4
        for cr in exp.coverage_results:
            assert np.isfinite(cr.coverage_pct)

    def test_aggregation_populated(self):
        exp = self._make_mini_experiment()
        exp.run()
        assert len(exp.aggregation_results) > 0

    def test_budget_entries(self):
        exp = self._make_mini_experiment()
        exp.run()
        # 1 case × 2 quantities = 2 entries
        assert len(exp.budget_entries) == 2
        for entry in exp.budget_entries:
            assert np.isfinite(entry.total_rss_pct)


# =========================================================================
# TestStochasticClosureReport
# =========================================================================
class TestStochasticClosureReport:
    """Tests for report generation."""

    @pytest.fixture
    def report(self):
        cases = [CalibrationCase("test_a", "A", Re=1e5, n_points=40)]
        wrappers = [
            BNNClosureWrapper(n_ensemble=2, seed=42),
            EnsembleClosureWrapper(n_models=2, seed=42),
        ]
        exp = StochasticClosureExperiment(cases=cases, wrappers=wrappers, seed=42)
        exp.run()
        return StochasticClosureReport(exp)

    def test_coverage_table(self, report):
        ct = report.coverage_table()
        assert len(ct) > 0

    def test_aggregation_summary(self, report):
        aws = report.aggregation_weight_summary()
        assert len(aws) > 0

    def test_markdown(self, report):
        md = report.generate_markdown()
        assert "Calibrated Stochastic ML Closures" in md
        assert "Coverage Calibration" in md
        assert "Space-Dependent Aggregation" in md
        assert "Extended RSS Error Budget" in md

    def test_json_serializable(self, report):
        j = report.to_json()
        data = json.loads(j)
        assert "coverage" in data
        assert "budget" in data

    def test_summary(self, report):
        s = report.summary()
        assert "Stochastic Closure Calibration" in s


# =========================================================================
# TestRunCalibrationStudy
# =========================================================================
class TestRunCalibrationStudy:
    """Test the convenience runner."""

    def test_mini_study(self):
        report = run_calibration_study(confidence=0.95, seed=42)
        assert "Stochastic Closure Calibration" in report.summary()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
