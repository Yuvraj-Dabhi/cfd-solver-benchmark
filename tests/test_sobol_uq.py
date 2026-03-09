"""
Tests for Sobol UQ + Bayesian Model Averaging
================================================
Validates Sobol indices, BMA weights, UQ report generation,
and synthetic test data.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ml_augmentation.sobol_uq_bayesian import (
    SobolProblem,
    SobolResult,
    SobolAnalyzer,
    BayesianModelAverager,
    UQReportGenerator,
    generate_sobol_test_problem,
    generate_bma_test_data,
)


class TestSobolAnalyzer:
    """Test Sobol sensitivity analysis."""

    def test_ishigami_shapes(self):
        problem, func = generate_sobol_test_problem()
        analyzer = SobolAnalyzer(problem)
        result = analyzer.analyze(func, n_samples=256)

        assert result.S1.shape == (3,)
        assert result.ST.shape == (3,)
        assert result.S1_conf.shape == (3,)
        assert result.ST_conf.shape == (3,)

    def test_ishigami_x3_first_order_near_zero(self):
        """x3 has no first-order effect in Ishigami."""
        problem, func = generate_sobol_test_problem()
        analyzer = SobolAnalyzer(problem)
        result = analyzer.analyze(func, n_samples=512)

        # S1 for x3 should be close to 0
        assert result.S1[2] < 0.1, f"S1[x3]={result.S1[2]:.3f}, expected ~0"

    def test_ishigami_x2_no_interaction(self):
        """x2 has no interactions, so S1 ≈ ST."""
        problem, func = generate_sobol_test_problem()
        analyzer = SobolAnalyzer(problem)
        result = analyzer.analyze(func, n_samples=512)

        # For x2: S1 ≈ ST (no interactions)
        assert abs(result.S1[1] - result.ST[1]) < 0.15

    def test_indices_bounded(self):
        """All indices should be in [0, 1]."""
        problem, func = generate_sobol_test_problem()
        analyzer = SobolAnalyzer(problem)
        result = analyzer.analyze(func, n_samples=256)

        assert np.all(result.S1 >= 0)
        assert np.all(result.S1 <= 1)
        assert np.all(result.ST >= 0)
        assert np.all(result.ST <= 1)

    def test_to_dict(self):
        problem, func = generate_sobol_test_problem()
        analyzer = SobolAnalyzer(problem)
        result = analyzer.analyze(func, n_samples=128)

        d = result.to_dict()
        assert "first_order" in d
        assert "x1" in d["first_order"]
        assert "S1" in d["first_order"]["x1"]

    def test_to_markdown_table(self):
        problem, func = generate_sobol_test_problem()
        analyzer = SobolAnalyzer(problem)
        result = analyzer.analyze(func, n_samples=128)

        md = result.to_markdown_table()
        assert "S₁" in md
        assert "x1" in md


class TestBMA:
    """Test Bayesian Model Averaging."""

    def test_weights_sum_to_one(self):
        bma = BayesianModelAverager()
        bma.add_model("SA", np.array([1, 2, 3.0]), -10.0)
        bma.add_model("SST", np.array([1, 2, 3.0]), -8.0)
        bma.add_model("RSM", np.array([1, 2, 3.0]), -12.0)

        weights = bma.compute_weights()
        assert abs(sum(weights.values()) - 1.0) < 1e-10

    def test_better_model_higher_weight(self):
        """Higher log-likelihood → higher weight."""
        bma = BayesianModelAverager()
        bma.add_model("poor", np.array([1.0]), -100.0)
        bma.add_model("good", np.array([1.0]), -10.0)

        weights = bma.compute_weights()
        assert weights["good"] > weights["poor"]

    def test_average_shape(self):
        bma = BayesianModelAverager()
        bma.add_model("SA", np.random.randn(50), -10.0)
        bma.add_model("SST", np.random.randn(50), -8.0)

        result = bma.average()
        assert result["mean"].shape == (50,)
        assert result["std"].shape == (50,)
        assert result["credible_lower"].shape == (50,)
        assert result["credible_upper"].shape == (50,)

    def test_credible_interval_ordering(self):
        bma = BayesianModelAverager()
        bma.add_model("SA", np.ones(20), -10.0)
        bma.add_model("SST", np.ones(20) * 2, -8.0)

        result = bma.average()
        assert np.all(result["credible_lower"] <= result["mean"])
        assert np.all(result["credible_upper"] >= result["mean"])

    def test_from_residuals(self):
        data = generate_bma_test_data()
        bma = BayesianModelAverager()
        for name, preds in data["predictions"].items():
            bma.add_model_from_residuals(name, preds, data["observations"])

        weights = bma.compute_weights()
        assert len(weights) == 3
        assert abs(sum(weights.values()) - 1.0) < 1e-10

    def test_bic_prior(self):
        data = generate_bma_test_data()
        bma = BayesianModelAverager(prior="bic")
        for name, preds in data["predictions"].items():
            bma.add_model_from_residuals(
                name, preds, data["observations"], n_params=2)

        weights = bma.compute_weights()
        assert abs(sum(weights.values()) - 1.0) < 1e-10

    def test_to_dict(self):
        bma = BayesianModelAverager()
        bma.add_model("SA", np.array([1.0]), -10.0)
        bma.add_model("SST", np.array([1.0]), -8.0)

        d = bma.to_dict()
        assert d["n_models"] == 2
        assert "weights" in d


class TestUQReport:
    """Test report generation."""

    def test_sobol_section(self):
        gen = UQReportGenerator()
        result = SobolResult(
            S1=np.array([0.3, 0.4, 0.0]),
            ST=np.array([0.5, 0.4, 0.2]),
            S1_conf=np.array([0.05, 0.04, 0.01]),
            ST_conf=np.array([0.06, 0.04, 0.03]),
            var_names=["Re", "Mach", "alpha"],
        )
        gen.add_sobol_results("wall_hump", result)
        md = gen.generate_report()
        assert "Sobol Sensitivity" in md
        assert "wall_hump" in md
        assert "Re" in md

    def test_bma_section(self):
        gen = UQReportGenerator()
        bma = BayesianModelAverager()
        bma.add_model("SA", np.array([1.0]), -10.0)
        bma.add_model("SST", np.array([1.0]), -8.0)
        gen.add_bma_results("flat_plate", bma)
        md = gen.generate_report()
        assert "BMA Model Weights" in md
        assert "SA" in md

    def test_error_budget_section(self):
        gen = UQReportGenerator()
        budget = {"U_num": 0.02, "U_input": 0.05, "U_model": 0.15, "U_total": 0.16}
        gen.add_error_budget("swbli", budget)
        md = gen.generate_report()
        assert "Error Budget" in md
        assert "swbli" in md

    def test_combined_report(self):
        gen = UQReportGenerator()
        result = SobolResult(
            S1=np.array([0.3]), ST=np.array([0.5]),
            S1_conf=np.array([0.05]), ST_conf=np.array([0.06]),
            var_names=["Re"],
        )
        gen.add_sobol_results("case1", result)
        bma = BayesianModelAverager()
        bma.add_model("SA", np.array([1.0]), -10.0)
        gen.add_bma_results("case1", bma)
        gen.add_error_budget("case1", {"U_total": 0.10})

        md = gen.generate_report()
        assert md.count("case1") >= 3  # Appears in all 3 sections
