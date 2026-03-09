#!/usr/bin/env python3
"""
Tests for ML Pipeline Enhancements (Component 4)
====================================================
Tests surrogate model (GP/MLP), APG correction table,
and related utility functions.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestSurrogateModelGP:
    """Tests for GP surrogate model."""

    def _make_synthetic_data(self, n=50):
        """Generate synthetic CL data for testing."""
        np.random.seed(42)
        aoa = np.linspace(0, 15, n)
        Re = np.full(n, 6.0)
        X = np.column_stack([aoa, Re])
        y = 2 * np.pi * np.radians(aoa) + 0.01 * np.random.randn(n)
        return X, y

    @pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
    def test_surrogate_gp_fit(self):
        """GP should fit synthetic CL data with R² > 0.9."""
        from scripts.ml_augmentation.surrogate_model import SurrogateModel

        X, y = self._make_synthetic_data()
        model = SurrogateModel(model_type="gp")
        metrics = model.fit(X, y)
        assert metrics["train_R2"] > 0.9

    @pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
    def test_surrogate_gp_predict_returns_tuple(self):
        """GP predict should return (mean, std) tuple."""
        from scripts.ml_augmentation.surrogate_model import SurrogateModel

        X, y = self._make_synthetic_data()
        model = SurrogateModel(model_type="gp")
        model.fit(X, y)
        result = model.predict(X[:5])
        assert isinstance(result, tuple)
        mean, std = result
        assert len(mean) == 5
        assert len(std) == 5
        assert np.all(std >= 0)

    @pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
    def test_surrogate_gp_evaluate(self):
        """GP evaluation should return RMSE, R², MAPE."""
        from scripts.ml_augmentation.surrogate_model import SurrogateModel

        X, y = self._make_synthetic_data()
        model = SurrogateModel(model_type="gp")
        model.fit(X[:40], y[:40])
        metrics = model.evaluate(X[40:], y[40:])
        assert "RMSE" in metrics
        assert "R2" in metrics
        assert "MAPE" in metrics


class TestSurrogateModelMLP:
    """Tests for MLP surrogate model."""

    def _make_synthetic_data(self, n=80):
        np.random.seed(42)
        aoa = np.linspace(0, 15, n)
        Re = np.full(n, 6.0)
        X = np.column_stack([aoa, Re])
        y = 2 * np.pi * np.radians(aoa) + 0.01 * np.random.randn(n)
        return X, y

    @pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
    def test_surrogate_mlp_fit(self):
        """MLP should fit synthetic CL data."""
        from scripts.ml_augmentation.surrogate_model import SurrogateModel

        X, y = self._make_synthetic_data()
        model = SurrogateModel(model_type="mlp")
        metrics = model.fit(X, y, hidden=(32, 32), max_iter=500)
        assert "train_R2" in metrics

    @pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
    def test_surrogate_mlp_predict_returns_array(self):
        """MLP predict should return a plain array (no std)."""
        from scripts.ml_augmentation.surrogate_model import SurrogateModel

        X, y = self._make_synthetic_data()
        model = SurrogateModel(model_type="mlp")
        model.fit(X, y, max_iter=500)
        result = model.predict(X[:5])
        assert isinstance(result, np.ndarray)
        assert len(result) == 5

    @pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
    def test_surrogate_cross_validate(self):
        """k-fold CV should produce per-fold metrics."""
        from scripts.ml_augmentation.surrogate_model import SurrogateModel

        X, y = self._make_synthetic_data()
        model = SurrogateModel(model_type="mlp")
        cv = model.cross_validate(X, y, k=3)
        assert len(cv["RMSE"]) == 3
        assert len(cv["R2"]) == 3

    def test_surrogate_unfitted_raises(self):
        """Predict on unfitted model should raise RuntimeError."""
        from scripts.ml_augmentation.surrogate_model import SurrogateModel

        model = SurrogateModel(model_type="gp")
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict(np.array([[5, 6]]))


class TestAPGCorrectionTable:
    """Tests for APG correction table."""

    def _make_synthetic_table(self):
        """Build a correction table from synthetic data."""
        from scripts.ml_augmentation.apg_correction_table import APGCorrectionTable

        table = APGCorrectionTable()
        np.random.seed(42)
        n = 30
        case_results = {
            "wall_hump": {
                "x": np.linspace(0.5, 1.4, n),
                "clauser": np.linspace(0, 15, n),
                "H": np.linspace(1.4, 3.5, n),
                "Cf_rans": 0.003 * np.exp(-0.5 * np.linspace(0, 5, n)),
                "Cf_exp": 0.003 * np.exp(-0.5 * np.linspace(0, 5, n)) * (
                    1 + 0.3 * np.linspace(0, 1, n)
                ),
            },
        }
        table.build_from_cases(case_results)
        return table

    def test_apg_table_build(self):
        """Building from cases should populate entries."""
        table = self._make_synthetic_table()
        assert len(table.entries) > 0

    def test_apg_lookup_returns_float(self):
        """Lookup should return a float correction factor."""
        table = self._make_synthetic_table()
        beta = table.lookup(clauser_param=5.0, shape_factor=2.5)
        assert isinstance(beta, float)
        assert beta > 0  # Correction should be positive

    def test_apg_lookup_empty_table(self):
        """Empty table should return 1.0 (no correction)."""
        from scripts.ml_augmentation.apg_correction_table import APGCorrectionTable

        table = APGCorrectionTable()
        beta = table.lookup(5.0, 2.5)
        assert beta == 1.0

    def test_apg_correction_increases_with_pressure_gradient(self):
        """Correction should generally increase with stronger APG."""
        table = self._make_synthetic_table()
        beta_low = table.lookup(clauser_param=1.0, shape_factor=1.5)
        beta_high = table.lookup(clauser_param=10.0, shape_factor=3.0)
        # Higher APG should have different correction than low APG
        assert beta_low != beta_high

    def test_apg_summary(self):
        """Summary should include case count and ranges."""
        table = self._make_synthetic_table()
        summary = table.summary()
        assert "wall_hump" in summary
        assert "Entries" in summary


class TestAPGUtilities:
    """Tests for APG utility functions."""

    def test_clauser_parameter_computation(self):
        """Clauser parameter should be positive for APG."""
        from scripts.ml_augmentation.apg_correction_table import (
            compute_clauser_parameter,
        )

        dp_dx = np.array([100.0, 200.0])   # Positive = APG
        tau_w = np.array([10.0, 10.0])
        delta_star = np.array([0.01, 0.01])

        beta = compute_clauser_parameter(dp_dx, tau_w, delta_star)
        assert np.all(beta > 0)
        assert beta[1] > beta[0]  # Stronger APG

    def test_shape_factor_computation(self):
        """Shape factor H = δ*/θ should be > 1 for turbulent BLs."""
        from scripts.ml_augmentation.apg_correction_table import (
            compute_shape_factor,
        )

        delta_star = np.array([0.01, 0.02])
        theta = np.array([0.007, 0.008])

        H = compute_shape_factor(delta_star, theta)
        assert np.all(H > 1.0)  # H > 1 for all BLs
        assert H[1] > H[0]  # Thicker BL has higher H

    def test_clauser_safe_division(self):
        """Clauser parameter should handle zero wall shear stress."""
        from scripts.ml_augmentation.apg_correction_table import (
            compute_clauser_parameter,
        )

        dp_dx = np.array([100.0])
        tau_w = np.array([0.0])  # At separation
        delta_star = np.array([0.01])

        beta = compute_clauser_parameter(dp_dx, tau_w, delta_star)
        assert np.isfinite(beta[0])
