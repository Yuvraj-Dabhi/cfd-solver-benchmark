#!/usr/bin/env python3
"""
Tests for ML Physics-Informed Enhancements
=============================================
Tests multi-output distribution surrogate, PINN boundary layer
correction, and ML validation reporter.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =========================================================================
# Distribution Surrogate Tests
# =========================================================================
class TestDistributionSurrogate:
    """Tests for multi-output Cp/Cf distribution predictor."""

    def test_generate_training_data(self):
        """Training data should have correct shapes."""
        from scripts.ml_augmentation.distribution_surrogate import (
            generate_training_data, N_SURFACE_POINTS,
        )
        X, Y_Cp, Y_Cf = generate_training_data(n_samples=50)
        assert X.shape == (50, 6)
        assert Y_Cp.shape == (50, N_SURFACE_POINTS)
        assert Y_Cf.shape == (50, N_SURFACE_POINTS)

    def test_physics_features(self):
        """Physics features should be physically reasonable."""
        from scripts.ml_augmentation.distribution_surrogate import compute_bl_features
        feats = compute_bl_features(aoa_deg=10.0, Re=3e6, Mach=0.15)
        assert feats.shape_factor_H > 1.0  # Turbulent BL: H > 1
        assert feats.Re_theta > 0
        assert feats.dpCp_dx_max > 0
        assert len(feats.to_array()) == 6

    def test_fit_and_predict(self):
        """Model should train and predict Cp/Cf distributions."""
        from scripts.ml_augmentation.distribution_surrogate import (
            DistributionSurrogate, generate_training_data, N_SURFACE_POINTS,
        )
        X, Y_Cp, Y_Cf = generate_training_data(n_samples=100)
        model = DistributionSurrogate(hidden_layers=[32, 64, 32])
        metrics = model.fit(X, Y_Cp, Y_Cf)

        assert "Cp_R2" in metrics
        assert "Cf_R2" in metrics
        assert "Cp_MAPE" in metrics
        assert metrics["n_test"] > 0

        # Predict
        Cp, Cf = model.predict(X[:5])
        assert Cp.shape == (5, N_SURFACE_POINTS)
        assert Cf.shape == (5, N_SURFACE_POINTS)

    def test_separation_detection(self):
        """Should detect separation from Cf distribution."""
        from scripts.ml_augmentation.distribution_surrogate import (
            DistributionSurrogate, generate_training_data, compute_bl_features,
        )
        X, Y_Cp, Y_Cf = generate_training_data(n_samples=100)
        model = DistributionSurrogate(hidden_layers=[32, 64, 32])
        model.fit(X, Y_Cp, Y_Cf)

        # High AoA should have separation
        feats = compute_bl_features(15.0, 3e6)
        _, Cf = model.predict(feats.to_array().reshape(1, -1))
        sep = model.detect_separation(Cf)
        assert isinstance(sep, list)
        assert len(sep) == 1

    def test_cp_distribution_realistic(self):
        """Generated Cp should have suction peak near leading edge."""
        from scripts.ml_augmentation.distribution_surrogate import (
            generate_cp_distribution,
        )
        x_c = np.linspace(0.001, 1.0, 80)
        Cp = generate_cp_distribution(x_c, aoa_deg=5.0, Re=3e6)
        assert np.min(Cp) < 0  # Suction side negative
        assert np.argmin(Cp) < 30  # Peak near front


# =========================================================================
# PINN Boundary Layer Tests
# =========================================================================
class TestPINNBoundaryLayer:
    """Tests for PINN-inspired BL correction."""

    def test_von_karman_residual(self):
        """VK residual should be small for consistent BL data."""
        from scripts.ml_augmentation.pinn_boundary_layer import (
            von_karman_residual,
        )
        n = 50
        x = np.linspace(0.01, 1.0, n)
        Re_x = 3e6 * x
        Cf = 0.0592 / Re_x**0.2
        theta = 0.036 * x / Re_x**0.2
        H = np.full(n, 1.4)
        U_e = np.ones(n)

        res = von_karman_residual(x, Cf, theta, H, U_e)
        assert len(res) == n - 1
        # Residual should be finite
        assert np.all(np.isfinite(res))

    def test_pinn_correction_improves_cf(self):
        """PINN correction should reduce RMSE vs DNS."""
        from scripts.ml_augmentation.pinn_boundary_layer import (
            PINNBoundaryLayerCorrector, generate_bl_data,
        )
        x, data = generate_bl_data("flat_plate_apg")
        corrector = PINNBoundaryLayerCorrector(lambda_phys=0.1, n_basis=10)
        result = corrector.fit(
            x, data["Cf_rans"], data["Cf_dns"],
            data["theta"], data["H"], data["U_e"],
        )
        assert result.rmse_after <= result.rmse_before
        assert result.improvement_pct >= 0

    def test_pinn_nasa_hump_case(self):
        """PINN should work for NASA hump case."""
        from scripts.ml_augmentation.pinn_boundary_layer import (
            PINNBoundaryLayerCorrector, generate_bl_data,
        )
        x, data = generate_bl_data("nasa_hump")
        corrector = PINNBoundaryLayerCorrector(lambda_phys=0.05, n_basis=15)
        result = corrector.fit(
            x, data["Cf_rans"], data["Cf_dns"],
            data["theta"], data["H"], data["U_e"],
        )
        assert result.improvement_pct >= 0

    def test_physics_loss_ablation(self):
        """Ablation should produce results for all lambda values."""
        from scripts.ml_augmentation.pinn_boundary_layer import (
            physics_loss_ablation,
        )
        results = physics_loss_ablation("flat_plate_apg")
        assert len(results) == 4
        assert "lambda=0.0" in results
        assert "lambda=1.0" in results

    def test_beta_field_near_unity(self):
        """Beta field should be close to 1.0 (small corrections)."""
        from scripts.ml_augmentation.pinn_boundary_layer import (
            PINNBoundaryLayerCorrector, generate_bl_data,
        )
        x, data = generate_bl_data("flat_plate_apg")
        corrector = PINNBoundaryLayerCorrector(lambda_phys=0.1, n_basis=10)
        result = corrector.fit(
            x, data["Cf_rans"], data["Cf_dns"],
            data["theta"], data["H"], data["U_e"],
        )
        # Beta should be O(1), not wildly divergent
        assert np.all(np.abs(result.beta_field) < 5.0)

    def test_predict_applies_correction(self):
        """predict() should apply learned correction."""
        from scripts.ml_augmentation.pinn_boundary_layer import (
            PINNBoundaryLayerCorrector, generate_bl_data,
        )
        x, data = generate_bl_data("flat_plate_apg")
        corrector = PINNBoundaryLayerCorrector(lambda_phys=0.1, n_basis=10)
        corrector.fit(
            x, data["Cf_rans"], data["Cf_dns"],
            data["theta"], data["H"], data["U_e"],
        )
        Cf_pred = corrector.predict(x, data["Cf_rans"])
        assert Cf_pred.shape == data["Cf_rans"].shape


# =========================================================================
# ML Validation Reporter Tests
# =========================================================================
class TestMLValidationReporter:
    """Tests for the ML validation reporter."""

    def _make_data(self, n=100):
        """Generate simple regression data."""
        rng = np.random.RandomState(42)
        X = rng.uniform(-5, 18, (n, 2))
        y = 0.1 * X[:, 0] + 0.01 * X[:, 1] + rng.normal(0, 0.05, n)
        return X, y

    def test_compute_metrics(self):
        """Metrics should be computed correctly."""
        from scripts.ml_augmentation.ml_validation_reporter import compute_metrics
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 2.0, 2.9, 4.1, 5.0])
        m = compute_metrics(y_true, y_pred, "test")
        assert m.R2 > 0.99
        assert m.RMSE < 0.1
        assert m.output_name == "test"

    def test_overfitting_analysis(self):
        """Overfitting analysis should run without error."""
        from sklearn.linear_model import LinearRegression
        from scripts.ml_augmentation.ml_validation_reporter import analyze_overfitting

        X, y = self._make_data(80)
        lr = LinearRegression()
        result = analyze_overfitting(
            lambda X_t, y_t: lr.fit(X_t, y_t),
            lambda X_p: lr.predict(X_p),
            X, y, n_sizes=5,
        )
        assert not result.is_overfitting  # Linear model shouldn't overfit
        assert len(result.learning_curve_train) > 0

    def test_cross_validation(self):
        """5-fold CV should produce 5 fold results."""
        from sklearn.linear_model import LinearRegression
        from scripts.ml_augmentation.ml_validation_reporter import cross_validate

        X, y = self._make_data(100)
        lr = LinearRegression()
        cv = cross_validate(
            lambda X_t, y_t: lr.fit(X_t, y_t),
            lambda X_p: lr.predict(X_p).reshape(-1, 1),
            X, y.reshape(-1, 1), k=5,
        )
        assert cv.k == 5
        assert len(cv.fold_R2s) == 5
        assert cv.R2_mean > 0.5

    def test_report_generation(self):
        """Report should be a non-empty string."""
        from scripts.ml_augmentation.ml_validation_reporter import (
            compute_metrics, generate_validation_report,
        )
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 2.0, 2.9, 4.1, 5.0])
        metrics = [compute_metrics(y_true, y_pred, "CL")]
        report = generate_validation_report(metrics)
        assert "R2" in report
        assert "CL" in report

    def test_latex_table(self):
        """LaTeX table should contain valid markup."""
        from scripts.ml_augmentation.ml_validation_reporter import (
            compute_metrics, generate_latex_metrics_table,
        )
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 2.0, 2.9, 4.1, 5.0])
        metrics = [compute_metrics(y_true, y_pred, "CL")]
        latex = generate_latex_metrics_table(metrics)
        assert r"\begin{table}" in latex
        assert r"R^2" in latex

    def test_model_comparison(self):
        """Model comparison should rank models by test R2."""
        from sklearn.linear_model import LinearRegression, Ridge
        from scripts.ml_augmentation.ml_validation_reporter import compare_models

        X, y = self._make_data(100)
        y = y.reshape(-1, 1)

        lr, ridge = LinearRegression(), Ridge(alpha=1.0)
        models = {
            "LinearReg": (
                lambda X_t, y_t, m=lr: m.fit(X_t, y_t),
                lambda X_p, m=lr: m.predict(X_p),
                "linear",
            ),
            "Ridge": (
                lambda X_t, y_t, m=ridge: m.fit(X_t, y_t),
                lambda X_p, m=ridge: m.predict(X_p),
                "linear",
            ),
        }
        results = compare_models(models, X, y)
        assert len(results) == 2
        # First should have highest test R2
        assert results[0].test_R2 >= results[1].test_R2
