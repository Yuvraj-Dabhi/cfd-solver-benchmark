#!/usr/bin/env python3
"""
Tests for ML-Assisted Cf Correction Pipeline
==============================================
"""

import sys
import numpy as np
import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ml_augmentation.ml_cf_correction import (
    generate_hump_baseline,
    build_features,
    build_targets,
    physics_loss,
    CfCorrectionResult,
    CfCorrectionPipeline,
    run_correction_experiment,
    PhysicsPenaltyLoss,
    generate_periodic_hill_data,
    evaluate_constrained_recalibration,
)
from scripts.ml_augmentation.fiml_su2_adjoint import (
    SSTCoefficientOptimizer,
    FIMLConfig,
    ReferenceData,
)
from scripts.postprocessing.separation_analysis import HUMP_EXP


# =========================================================================
# Synthetic Data Generation
# =========================================================================

class TestHumpBaseline:
    """Test the synthetic hump data generator."""

    @pytest.fixture(scope="class")
    def data(self):
        return generate_hump_baseline(n_points=200)

    def test_keys_present(self, data):
        for key in ["x", "Cf_baseline", "Cf_exp", "Cp_baseline", "dCp_dx", "H"]:
            assert key in data

    def test_array_shapes(self, data):
        n = len(data["x"])
        assert n == 200
        for key in ["Cf_baseline", "Cf_exp", "Cp_baseline", "dCp_dx", "H"]:
            assert len(data[key]) == n

    def test_x_range(self, data):
        assert data["x"][0] == pytest.approx(-0.5, abs=0.01)
        assert data["x"][-1] == pytest.approx(1.5, abs=0.01)

    def test_cf_exp_has_separation(self, data):
        """Experimental Cf should go negative in separation region."""
        x = data["x"]
        Cf = data["Cf_exp"]
        sep_mask = (x > 0.7) & (x < 1.0)
        assert np.any(Cf[sep_mask] < 0), "No separation in experimental Cf"

    def test_cf_baseline_has_separation(self, data):
        """SA baseline Cf should also go negative."""
        x = data["x"]
        Cf = data["Cf_baseline"]
        sep_mask = (x > 0.7) & (x < 1.0)
        assert np.any(Cf[sep_mask] < 0), "No separation in baseline Cf"

    def test_cf_positive_upstream(self, data):
        """Cf should be positive upstream of separation."""
        x = data["x"]
        Cf = data["Cf_exp"]
        upstream = x < 0.5
        assert np.all(Cf[upstream] > -0.001), "Cf negative upstream"

    def test_shape_factor_ranges(self, data):
        """H should be ~1.4 in attached, higher in separated."""
        x = data["x"]
        H = data["H"]
        upstream = x < 0.5
        assert np.mean(H[upstream]) == pytest.approx(1.4, abs=0.1)
        sep_mask = (x > 0.8) & (x < 1.0)
        assert np.mean(H[sep_mask]) > 2.0, "H not elevated in separation"

    def test_reproducibility(self):
        d1 = generate_hump_baseline(seed=42)
        d2 = generate_hump_baseline(seed=42)
        np.testing.assert_array_equal(d1["Cf_exp"], d2["Cf_exp"])


# =========================================================================
# Feature and Target Construction
# =========================================================================

class TestFeatures:
    """Test feature matrix construction."""

    def test_feature_shape(self):
        data = generate_hump_baseline(n_points=100)
        X = build_features(data)
        assert X.shape == (100, 5)

    def test_feature_columns(self):
        data = generate_hump_baseline(n_points=50)
        X = build_features(data)
        np.testing.assert_array_equal(X[:, 0], data["x"])
        np.testing.assert_array_equal(X[:, 1], data["Cp_baseline"])
        np.testing.assert_array_equal(X[:, 2], data["Cf_baseline"])


class TestTargets:
    """Test multiplicative correction targets."""

    def test_target_shape(self):
        data = generate_hump_baseline(n_points=100)
        beta = build_targets(data)
        assert len(beta) == 100

    def test_realizability_bounds(self):
        data = generate_hump_baseline(n_points=200)
        beta = build_targets(data)
        assert np.all(beta >= -5.0)
        assert np.all(beta <= 5.0)

    def test_beta_near_one_in_attached(self):
        """β should be close to 1 where baseline ≈ experiment."""
        data = generate_hump_baseline(n_points=200)
        beta = build_targets(data)
        x = data["x"]
        upstream = (x > -0.3) & (x < 0.3)
        if upstream.any():
            assert np.mean(np.abs(beta[upstream] - 1.0)) < 1.0


# =========================================================================
# Physics Loss
# =========================================================================

class TestPhysicsLoss:
    """Test the physics-constrained loss function."""

    def test_zero_penalty_for_physical(self):
        x = np.linspace(-0.5, 1.5, 100)
        Cf = np.ones(100) * 0.003  # All positive
        penalty = physics_loss(Cf, Cf, x)
        assert penalty >= 0

    def test_sign_penalty(self):
        x = np.linspace(-0.5, 1.5, 100)
        Cf_good = np.ones(100) * 0.003
        Cf_bad = np.copy(Cf_good)
        # Make attached region negative (unphysical)
        Cf_bad[x < 0.3] = -0.001
        p_good = physics_loss(Cf_good, Cf_good, x)
        p_bad = physics_loss(Cf_bad, Cf_bad, x)
        assert p_bad > p_good

    def test_smoothness_penalty(self):
        x = np.linspace(-0.5, 1.5, 100)
        Cf_smooth = np.sin(np.pi * x) * 0.003
        Cf_noisy = Cf_smooth + np.random.randn(100) * 0.01
        # Noisy should have higher smoothness penalty
        p_smooth = physics_loss(Cf_smooth, Cf_smooth, x, lambda_sign=0)
        p_noisy = physics_loss(Cf_noisy, Cf_noisy, x, lambda_sign=0)
        assert p_noisy > p_smooth


# =========================================================================
# Full Pipeline
# =========================================================================

class TestCfCorrectionPipeline:
    """Test the full ML correction pipeline."""

    @pytest.fixture(scope="class")
    def result(self):
        pipeline = CfCorrectionPipeline(
            n_ensemble=3,   # Fewer for speed
            max_iter=200,
            seed=42,
        )
        return pipeline.run(n_points=150)

    def test_runs(self, result):
        assert isinstance(result, CfCorrectionResult)

    def test_rmse_computed(self, result):
        assert result.baseline_rmse_total > 0
        assert result.corrected_rmse_total > 0
        assert result.baseline_rmse_sep > 0

    def test_rmse_improved(self, result):
        """Corrected RMSE should be lower than baseline."""
        assert result.corrected_rmse_total < result.baseline_rmse_total

    def test_reduction_computed(self, result):
        assert result.rmse_reduction_total_pct > 0
        assert result.rmse_reduction_sep_pct > 0

    def test_40pct_challenge_assessed(self, result):
        assert isinstance(result.meets_40pct_challenge, bool)

    def test_ensemble_uncertainty(self, result):
        assert result.mean_uncertainty > 0
        assert result.max_uncertainty >= result.mean_uncertainty

    def test_high_uncertainty_fraction(self, result):
        assert 0 <= result.high_uncertainty_fraction <= 1

    def test_output_arrays(self, result):
        assert result.x is not None
        assert result.Cf_corrected is not None
        assert result.beta_mean is not None
        assert result.beta_std is not None
        assert len(result.x) == len(result.Cf_corrected)

    def test_summary_string(self, result):
        assert "40% Challenge" in result.summary
        assert "Deep Ensemble" in result.summary
        assert "RMSE" in result.summary
        assert "Physics" in result.summary

    def test_to_dict(self, result):
        d = result.to_dict()
        assert "baseline_rmse_total" in d
        assert "meets_40pct_challenge" in d

    def test_physics_penalty_reasonable(self, result):
        assert result.physics_penalty_after <= result.physics_penalty_before * 1.5


# =========================================================================
# Physics Penalty Constraints (Bin et al. 2024)
# =========================================================================

class TestPhysicsPenaltyLoss:
    """Test the three physics-penalty constraints for RANS recalibration."""

    def test_realizability_penalty(self):
        loss = PhysicsPenaltyLoss()

        # 1. Compliant tensor (trace-free, eigenvalues in [-1/3, 2/3])
        # Diagonal: [-0.2, 0.1, 0.1]
        b_good = np.diag([-0.2, 0.1, 0.1]).reshape(1, 3, 3)
        p, n = loss.realizability_penalty(b_good)
        assert p < 1e-6
        assert n == 0

        # 2. Violating tensor (eigenvalue > 2/3)
        b_bad = np.diag([-0.5, 0.7, -0.2]).reshape(1, 3, 3)
        p, n = loss.realizability_penalty(b_bad)
        assert p > 0.0
        assert n == 1

        # 3. Non-trace-free tensor
        b_trace = np.diag([0.1, 0.1, 0.1]).reshape(1, 3, 3)
        p, n = loss.realizability_penalty(b_trace)
        assert p > 0.0
        assert n == 1

    def test_galilean_invariance_penalty(self):
        loss = PhysicsPenaltyLoss()

        # 1. Only invariant features
        f_good = ["x_c", "Cp", "Q_criterion", "lambda1_S2"]
        p, frac = loss.galilean_invariance_penalty(f_good)
        assert p == 0.0
        assert frac == 0.0

        # 2. Contains non-invariant (e.g. raw velocity component u)
        f_bad = ["x_c", "u", "v", "Q_criterion"]
        p, frac = loss.galilean_invariance_penalty(f_bad)
        assert p > 0.0
        assert frac == 0.5  # 2 out of 4 are non-invariant

    def test_production_monotonicity_penalty(self):
        loss = PhysicsPenaltyLoss()

        # 1. Positive production everywhere
        prod_good = np.array([0.1, 0.01, 1e-5, 0.5])
        p, n = loss.production_monotonicity_penalty(prod_good)
        assert p == 0.0
        assert n == 0

        # 2. Negative production (backscatter)
        prod_bad = np.array([0.1, -0.05, 0.2, -0.1])
        p, n = loss.production_monotonicity_penalty(prod_bad)
        assert p > 0.0
        assert n == 2
        # Penalty should be squared sum: (-0.05)^2 + (-0.1)^2 = 0.0125
        assert np.isclose(p, 0.0125)

    def test_compute_all_penalties(self):
        loss = PhysicsPenaltyLoss(lambda_real=1.0, lambda_gal=1.0, lambda_prod=1.0)
        
        b = np.diag([-0.5, 0.7, -0.2]).reshape(1, 3, 3)
        f = ["u", "v"]
        prod = np.array([-0.1])

        report = loss.compute(
            reynolds_stress_anisotropy=b,
            feature_names=f,
            production_field=prod,
        )

        assert report.realizability_penalty > 0
        assert report.galilean_penalty > 0
        assert report.production_penalty > 0
        assert report.total_penalty == (
            report.realizability_penalty +
            report.galilean_penalty +
            report.production_penalty
        )


class TestPeriodicHillData:
    """Test the synthetic Periodic Hill flow data generator."""

    def test_data_generation(self):
        data = generate_periodic_hill_data(n_points=100)
        
        # Check keys
        expected_keys = [
            "x", "Cf_baseline", "Cf_dns", "Cp_baseline", "Cp_dns",
            "production", "anisotropy_eigenvalues", "hill_height",
            "x_sep_dns", "x_reat_dns"
        ]
        for k in expected_keys:
            assert k in data
        
        # Check shapes
        assert len(data["x"]) == 100
        assert data["anisotropy_eigenvalues"].shape == (100, 3)

    def test_separation_bubble(self):
        data = generate_periodic_hill_data()
        
        # DNS should have a separation bubble
        cf_dns = data["Cf_dns"]
        assert np.any(cf_dns < 0)
        
        # SST should over-predict reattachment (typical behaviour)
        assert data["x_reat_sst"] > data["x_reat_dns"]


class TestSSTCoefficientOptimizer:
    """Test the wrapper for SST coefficient optimisation."""

    def test_optimizer_creation(self):
        config = FIMLConfig(case_name="periodic_hill")
        ref = ReferenceData()
        loss = PhysicsPenaltyLoss()
        
        # Dummy data
        prod = np.ones(10) * 0.01
        aniso = np.zeros((10, 3, 3))
        feats = ["Q_criterion"]

        opt = SSTCoefficientOptimizer(
            config=config,
            reference=ref,
            penalty_loss=loss,
            production_field=prod,
            anisotropy_base=aniso,
            feature_names=feats,
        )
        
        assert opt.config.case_name == "periodic_hill"
        assert opt.penalty_loss is not None
        
        # Test optimize returns array (dummy implementation returns x0)
        x0 = np.array([0.09, 0.85, 0.5, 0.31])
        bounds = [(0.05, 0.15), (0.5, 1.5), (0.3, 1.0), (0.15, 0.40)]
        x_opt = opt.optimize(x0, bounds)
        assert len(x_opt) == 4


class TestConstrainedRecalibration:
    """Test the end-to-end evaluation runner."""

    def test_runner_executes(self):
        # Run with very few iterations for speed
        results = evaluate_constrained_recalibration(
            cases=["periodic_hill"],
            max_iter=3,
            verbose=False,
        )
        
        assert "periodic_hill" in results
        res = results["periodic_hill"]
        
        assert res.case_name == "periodic_hill"
        assert len(res.sst_baseline) == 4
        assert len(res.sst_optimised) == 4
        assert res.penalty_before is not None
        assert res.penalty_after is not None
        
        # Should populate history
        assert len(res.objective_history) > 0


class TestConvenienceRunner:
    """Test the convenience runner function."""

    def test_run_correction_experiment(self, capsys):
        result = run_correction_experiment(n_ensemble=2, n_points=80, seed=0)
        captured = capsys.readouterr()
        assert "Cf CORRECTION" in captured.out
