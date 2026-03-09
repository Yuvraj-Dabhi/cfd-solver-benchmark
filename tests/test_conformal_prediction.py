"""Tests for Conformal Prediction UQ module."""
import numpy as np
import pytest
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))


class TestNonconformityScores:
    def test_absolute_residual(self):
        from scripts.ml_augmentation.conformal_prediction import AbsoluteResidualScore
        score = AbsoluteResidualScore()
        y_pred = np.array([1.0, 2.0, 3.0])
        y_true = np.array([1.1, 1.8, 3.5])
        scores = score(y_pred, y_true)
        np.testing.assert_allclose(scores, [0.1, 0.2, 0.5], atol=1e-10)

    def test_normalized_residual(self):
        from scripts.ml_augmentation.conformal_prediction import NormalizedResidualScore
        score = NormalizedResidualScore()
        y_pred = np.array([10.0, 20.0])
        y_true = np.array([11.0, 22.0])
        scores = score(y_pred, y_true)
        assert scores[0] < scores[1]  # 1/10 < 2/20 is false but 1/10.000001 vs 2/20.000001

    def test_quantile_score(self):
        from scripts.ml_augmentation.conformal_prediction import QuantileScore
        score = QuantileScore()
        score.set_quantiles(np.array([0.5, 1.5]), np.array([1.5, 2.5]))
        y_true = np.array([1.0, 3.0])
        scores = score(np.zeros(2), y_true)
        # max(0.5 - 1.0, 1.0 - 1.5) = max(-0.5, -0.5) = -0.5
        # max(1.5 - 3.0, 3.0 - 2.5) = max(-1.5, 0.5) = 0.5
        np.testing.assert_allclose(scores, [-0.5, 0.5], atol=1e-10)


class TestSplitConformal:
    def test_calibrate_and_predict(self):
        from scripts.ml_augmentation.conformal_prediction import SplitConformalPredictor
        rng = np.random.RandomState(42)
        y_true = rng.randn(100)
        y_pred = y_true + rng.randn(100) * 0.1

        cp = SplitConformalPredictor(alpha=0.1)
        cp.calibrate(y_pred, y_true)
        assert cp.q_hat is not None
        assert cp.q_hat > 0

        intervals = cp.predict_interval(y_pred)
        assert intervals.method == "split"
        assert len(intervals.lower) == 100
        assert np.all(intervals.width > 0)

    def test_coverage_guarantee(self):
        from scripts.ml_augmentation.conformal_prediction import SplitConformalPredictor
        rng = np.random.RandomState(123)
        n_cal, n_test = 200, 500

        y_cal_true = rng.randn(n_cal) * 2
        y_cal_pred = y_cal_true + rng.randn(n_cal) * 0.3

        y_test_true = rng.randn(n_test) * 2
        y_test_pred = y_test_true + rng.randn(n_test) * 0.3

        cp = SplitConformalPredictor(alpha=0.1)
        cp.calibrate(y_cal_pred, y_cal_true)
        intervals = cp.predict_interval(y_test_pred)

        coverage = intervals.coverage(y_test_true)
        # Coverage should be ≥ 1 - α = 0.90 (with high probability)
        assert coverage >= 0.85, f"Coverage {coverage:.3f} too low"

    def test_alpha_validation(self):
        from scripts.ml_augmentation.conformal_prediction import SplitConformalPredictor
        with pytest.raises(ValueError, match="alpha must be in"):
            SplitConformalPredictor(alpha=0.0)
        with pytest.raises(ValueError, match="alpha must be in"):
            SplitConformalPredictor(alpha=1.0)

    def test_uncalibrated_error(self):
        from scripts.ml_augmentation.conformal_prediction import SplitConformalPredictor
        cp = SplitConformalPredictor()
        with pytest.raises(RuntimeError, match="calibrate"):
            cp.predict_interval(np.array([1.0]))

    def test_summary(self):
        from scripts.ml_augmentation.conformal_prediction import SplitConformalPredictor
        cp = SplitConformalPredictor(alpha=0.05)
        cp.calibrate(np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
                     np.array([1.1, 2.2, 2.8, 4.1, 5.3]))
        intervals = cp.predict_interval(np.array([1.5, 2.5, 3.5]))
        s = intervals.summary()
        assert s["method"] == "split"
        assert s["target_coverage"] == 0.95
        assert s["n_predictions"] == 3


class TestCQR:
    def test_fit_calibrate_predict(self):
        from scripts.ml_augmentation.conformal_prediction import (
            ConformalizedQuantileRegression
        )
        rng = np.random.RandomState(42)
        X_train = rng.randn(100, 3)
        y_train = X_train @ np.array([1.0, -0.5, 0.3]) + rng.randn(100) * 0.2
        X_cal = rng.randn(50, 3)
        y_cal = X_cal @ np.array([1.0, -0.5, 0.3]) + rng.randn(50) * 0.2

        cqr = ConformalizedQuantileRegression(
            alpha=0.1, input_dim=3, hidden_dim=32, n_epochs=50
        )
        cqr.fit(X_train, y_train)
        cqr.calibrate(X_cal, y_cal)

        X_test = rng.randn(30, 3)
        intervals = cqr.predict_interval(X_test)
        assert intervals.method == "cqr"
        assert len(intervals.lower) == 30
        assert np.all(intervals.width > 0)


class TestJackknifeplus:
    def test_calibrate_and_predict(self):
        from scripts.ml_augmentation.conformal_prediction import ConformalJackknifeplus
        rng = np.random.RandomState(42)
        n = 20  # Small dataset (like Greenblatt PIV)
        X = rng.randn(n, 2)
        y = X @ np.array([1.5, -0.7]) + rng.randn(n) * 0.1

        jk = ConformalJackknifeplus(alpha=0.1)
        jk.calibrate(X, y)

        X_test = rng.randn(10, 2)
        intervals = jk.predict_interval(X_test)
        assert intervals.method == "jackknife+"
        assert len(intervals.lower) == 10
        assert np.all(intervals.width > 0)


class TestOODDetector:
    def test_ood_detection(self):
        from scripts.ml_augmentation.conformal_prediction import (
            SplitConformalPredictor, OODFlowDetector
        )
        rng = np.random.RandomState(42)

        # Calibrate on well-behaved data
        y_cal_true = rng.randn(200)
        y_cal_pred = y_cal_true + rng.randn(200) * 0.1
        cp = SplitConformalPredictor(alpha=0.05)
        cp.calibrate(y_cal_pred, y_cal_true)

        # Test predictions: mostly normal but some extreme
        y_test = np.concatenate([
            rng.randn(90) * 0.1,  # Normal
            rng.randn(10) * 10.0  # OOD (much wider)
        ])

        detector = OODFlowDetector(cp, threshold_factor=2.0)
        report = detector.detect(y_test)
        assert isinstance(report.ood_fraction, float)
        assert len(report.ood_flags) == 100


class TestConformalPCEWrapper:
    def test_wrap_pce(self):
        from scripts.ml_augmentation.conformal_prediction import ConformalPCEWrapper

        class MockPCE:
            def predict(self, X):
                return X[:, 0] * 2 + X[:, 1]

        pce = MockPCE()
        wrapper = ConformalPCEWrapper(pce, alpha=0.1)

        rng = np.random.RandomState(42)
        X_cal = rng.randn(50, 2)
        y_cal = X_cal[:, 0] * 2 + X_cal[:, 1] + rng.randn(50) * 0.1
        wrapper.calibrate(X_cal, y_cal)

        X_test = rng.randn(20, 2)
        intervals = wrapper.predict_interval(X_test)
        assert intervals.method == "split"
        assert len(intervals.lower) == 20


class TestConformalizeSurrogate:
    def test_one_liner_api(self):
        from scripts.ml_augmentation.conformal_prediction import conformalize_surrogate
        rng = np.random.RandomState(42)

        def mock_predict(X):
            return X[:, 0] + X[:, 1]

        X_cal = rng.randn(100, 2)
        y_cal = X_cal[:, 0] + X_cal[:, 1] + rng.randn(100) * 0.05
        cp = conformalize_surrogate(mock_predict, X_cal, y_cal, alpha=0.05)
        assert cp._calibrated
