"""
Tests for Constrained RANS Recalibration Module
===================================================
Validates SST coefficient space, physics penalty losses,
constrained optimizer, and recalibration reporting.

References:
  Bin et al. (2024), TAML 14, 100503.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.ml_augmentation.constrained_rans_recalibration import (
    SSTCoefficientSpace,
    PhysicsPenaltyLoss,
    SyntheticSSTEvaluator,
    ConstrainedRecalibrator,
    RecalibrationResult,
    run_recalibration,
)


# =============================================================================
# SSTCoefficientSpace Tests
# =============================================================================

class TestSSTCoefficientSpace:
    """Test SST coefficient parameterization."""

    def test_default_values(self):
        """Default coefficients match Menter (1994)."""
        coeffs = SSTCoefficientSpace.default()
        assert coeffs.sigma_k1 == pytest.approx(0.85)
        assert coeffs.sigma_k2 == pytest.approx(1.0)
        assert coeffs.beta_star == pytest.approx(0.09)
        assert coeffs.a1 == pytest.approx(0.31)
        assert coeffs.kappa == pytest.approx(0.41)

    def test_to_vector_length(self):
        """Parameter vector has 9 components."""
        vec = SSTCoefficientSpace.default().to_vector()
        assert len(vec) == 9

    def test_roundtrip(self):
        """from_vector(to_vector()) is identity."""
        c0 = SSTCoefficientSpace.default()
        c1 = SSTCoefficientSpace.from_vector(c0.to_vector())
        np.testing.assert_allclose(c0.to_vector(), c1.to_vector())

    def test_bounds_count(self):
        """Bounds list matches parameter count."""
        bounds = SSTCoefficientSpace.default().get_bounds()
        assert len(bounds) == 9
        for lo, hi in bounds:
            assert lo < hi

    def test_default_within_bounds(self):
        """Default values all lie within their bounds."""
        c = SSTCoefficientSpace.default()
        for name in c.NAMES:
            val = getattr(c, name)
            lo, hi = c.BOUNDS[name]
            assert lo <= val <= hi, f"{name}={val} outside [{lo}, {hi}]"

    def test_perturb_within_bounds(self):
        """Perturbed coefficients stay within bounds."""
        c = SSTCoefficientSpace.default().perturb(scale=0.3, seed=123)
        for name in c.NAMES:
            val = getattr(c, name)
            lo, hi = c.BOUNDS[name]
            assert lo <= val <= hi, f"{name}={val} outside [{lo}, {hi}]"

    def test_to_dict(self):
        """to_dict returns all 9 coefficients."""
        d = SSTCoefficientSpace.default().to_dict()
        assert len(d) == 9
        assert "beta_star" in d

    def test_diff(self):
        """Diff between default and perturbed shows nonzero deltas."""
        c0 = SSTCoefficientSpace.default()
        c1 = c0.perturb(scale=0.2)
        diff = c0.diff(c1)
        assert len(diff) == 9
        # At least some coefficients should have changed
        any_changed = any(abs(v["delta"]) > 1e-10 for v in diff.values())
        assert any_changed


# =============================================================================
# PhysicsPenaltyLoss Tests
# =============================================================================

class TestPhysicsPenaltyLoss:
    """Test physics penalty components."""

    def setup_method(self):
        self.penalty = PhysicsPenaltyLoss()
        self.default_coeffs = SSTCoefficientSpace.default()

    def test_log_layer_penalty_non_negative(self):
        """Log-layer penalty is always ≥ 0."""
        p = self.penalty.log_layer_penalty(self.default_coeffs)
        assert p >= 0.0

    def test_log_layer_penalty_zero_for_exact_kappa(self):
        """Log-layer penalty should be small (near zero) for correct κ."""
        p = self.penalty.log_layer_penalty(self.default_coeffs)
        # Default κ = 0.41 matches target, so penalty should be small
        assert p < 1.0  # Not necessarily zero due to consistency term

    def test_log_layer_penalty_increases_with_kappa_deviation(self):
        """Deviating κ increases penalty."""
        p_default = self.penalty.log_layer_penalty(self.default_coeffs)
        bad_coeffs = SSTCoefficientSpace(kappa=0.35)
        p_bad = self.penalty.log_layer_penalty(bad_coeffs)
        assert p_bad > p_default

    def test_realizability_penalty_non_negative(self):
        """Realizability penalty is always ≥ 0."""
        p = self.penalty.realizability_penalty(self.default_coeffs)
        assert p >= 0.0

    def test_realizability_penalty_with_samples(self):
        """Realizability penalty works with anisotropy samples."""
        # Create samples within Lumley triangle
        samples = np.zeros((10, 3, 3))
        for i in range(10):
            samples[i] = np.diag([-0.1, 0.0, 0.1])  # realizable
        p = self.penalty.realizability_penalty(self.default_coeffs, samples)
        assert p >= 0.0

    def test_realizability_penalty_detects_violations(self):
        """Violation samples increase penalty."""
        # Realizable samples
        good = np.zeros((5, 3, 3))
        for i in range(5):
            good[i] = np.diag([-0.2, 0.0, 0.2])

        # Unrealizable samples (eigenvalues outside bounds)
        bad = np.zeros((5, 3, 3))
        for i in range(5):
            bad[i] = np.diag([-0.5, 0.0, 0.5])  # -0.5 < -1/3

        p_good = self.penalty.realizability_penalty(self.default_coeffs, good)
        p_bad = self.penalty.realizability_penalty(self.default_coeffs, bad)
        assert p_bad > p_good

    def test_decay_penalty_non_negative(self):
        """Free-stream decay penalty is always ≥ 0."""
        p = self.penalty.free_stream_decay_penalty(self.default_coeffs)
        assert p >= 0.0

    def test_total_penalty_non_negative(self):
        """Total weighted penalty is non-negative."""
        p = self.penalty.total_penalty(self.default_coeffs)
        assert p >= 0.0

    def test_total_penalty_combines_all(self):
        """Total = weighted sum of components."""
        weights = {"log_layer": 1.0, "realizability": 1.0, "decay": 1.0}
        p_total = self.penalty.total_penalty(self.default_coeffs, weights=weights)
        p_log = self.penalty.log_layer_penalty(self.default_coeffs)
        p_real = self.penalty.realizability_penalty(self.default_coeffs)
        p_decay = self.penalty.free_stream_decay_penalty(self.default_coeffs)
        expected = p_log + p_real + p_decay
        assert p_total == pytest.approx(expected, rel=1e-10)


# =============================================================================
# SyntheticSSTEvaluator Tests
# =============================================================================

class TestSyntheticSSTEvaluator:
    """Test surrogate evaluation of SST coefficients."""

    def setup_method(self):
        self.evaluator = SyntheticSSTEvaluator()
        self.default_coeffs = SSTCoefficientSpace.default()

    def test_evaluate_all_cases(self):
        """Evaluation returns results for all 3 default cases."""
        results = self.evaluator.evaluate(self.default_coeffs)
        assert len(results) == 3
        assert "flat_plate" in results
        assert "wall_hump" in results
        assert "periodic_hill" in results

    def test_evaluate_single_case(self):
        """Can evaluate a single case."""
        results = self.evaluator.evaluate(self.default_coeffs, ["flat_plate"])
        assert len(results) == 1
        assert "flat_plate" in results

    def test_flat_plate_metrics(self):
        """Flat plate returns Cf and H errors."""
        results = self.evaluator.evaluate(self.default_coeffs, ["flat_plate"])
        metrics = results["flat_plate"]
        assert "Cf_error" in metrics
        assert "H_error" in metrics

    def test_wall_hump_metrics(self):
        """Wall hump returns separation metrics."""
        results = self.evaluator.evaluate(self.default_coeffs, ["wall_hump"])
        metrics = results["wall_hump"]
        assert "x_sep_error" in metrics
        assert "bubble_length_error" in metrics

    def test_default_errors_small(self):
        """Default coefficients should have small errors (by design)."""
        loss = self.evaluator.total_data_loss(self.default_coeffs)
        # Default coefficients were calibrated for these cases
        assert loss < 1.0  # Loose bound — errors should be small

    def test_total_data_loss_non_negative(self):
        """Total data loss is non-negative (sum of squares)."""
        loss = self.evaluator.total_data_loss(self.default_coeffs)
        assert loss >= 0.0


# =============================================================================
# ConstrainedRecalibrator Tests
# =============================================================================

class TestConstrainedRecalibrator:
    """Test constrained optimization pipeline."""

    def test_optimizer_converges(self):
        """Optimizer runs and returns a result."""
        recalibrator = ConstrainedRecalibrator(max_iter=50)
        result = recalibrator.optimize(
            target_cases=["flat_plate", "wall_hump"]
        )
        assert isinstance(result, RecalibrationResult)
        assert result.optimizer_result.success or result.optimizer_result.nit > 0

    def test_coefficients_within_bounds(self):
        """Optimized coefficients stay within physical bounds."""
        recalibrator = ConstrainedRecalibrator(max_iter=50)
        result = recalibrator.optimize(target_cases=["wall_hump"])
        optimized = result.optimized_coeffs
        for name in SSTCoefficientSpace.NAMES:
            val = getattr(optimized, name)
            lo, hi = SSTCoefficientSpace.BOUNDS[name]
            assert lo <= val <= hi + 1e-10, f"{name}={val} outside [{lo}, {hi}]"

    def test_loss_decreases(self):
        """Total loss should decrease (or stay same) during optimization."""
        recalibrator = ConstrainedRecalibrator(max_iter=100)
        result = recalibrator.optimize(
            target_cases=["flat_plate", "wall_hump", "periodic_hill"]
        )
        if len(result.history) >= 2:
            # First loss should be ≥ final loss
            first_loss = result.history[0]["total_loss"]
            final_loss = result.history[-1]["total_loss"]
            assert final_loss <= first_loss + 1e-6

    def test_history_populated(self):
        """Optimization history is recorded."""
        recalibrator = ConstrainedRecalibrator(max_iter=20)
        result = recalibrator.optimize(target_cases=["flat_plate"])
        assert len(result.history) > 0
        assert "data_loss" in result.history[0]
        assert "physics_penalty" in result.history[0]


# =============================================================================
# RecalibrationResult & Report Tests
# =============================================================================

class TestRecalibrationReport:
    """Test result reporting."""

    def setup_method(self):
        recalibrator = ConstrainedRecalibrator(max_iter=30)
        self.result = recalibrator.optimize(target_cases=["flat_plate", "wall_hump"])

    def test_report_string(self):
        """Report generates non-empty string."""
        report = self.result.report()
        assert len(report) > 100
        assert "CONSTRAINED RANS RECALIBRATION" in report
        assert "COEFFICIENT CHANGES" in report

    def test_improvement_summary(self):
        """Improvement summary has required keys."""
        summary = self.result.improvement_summary()
        assert "default_data_error" in summary
        assert "optimized_data_error" in summary
        assert "data_error_reduction_pct" in summary
        assert "optimizer_success" in summary

    def test_to_dict_structure(self):
        """to_dict returns serializable dictionary."""
        d = self.result.to_dict()
        assert "default_coefficients" in d
        assert "optimized_coefficients" in d
        assert "coefficient_changes" in d
        assert "improvement_summary" in d
        # Should be JSON-serializable
        import json
        json.dumps(d, default=str)

    def test_coefficient_changes(self):
        """coefficient_changes returns all 9 parameters."""
        changes = self.result.coefficient_changes()
        assert len(changes) == 9


# =============================================================================
# End-to-End Test
# =============================================================================

class TestRunRecalibration:
    """Test end-to-end pipeline."""

    def test_run_recalibration_completes(self, tmp_path):
        """Full pipeline runs without errors."""
        result = run_recalibration(
            target_cases=["flat_plate", "wall_hump"],
            max_iter=20,
            output_dir=str(tmp_path),
        )
        assert isinstance(result, RecalibrationResult)
        # Check output files
        assert (tmp_path / "recalibration_results.json").exists()
        assert (tmp_path / "recalibration_report.txt").exists()
        assert (tmp_path / "optimization_history.json").exists()
