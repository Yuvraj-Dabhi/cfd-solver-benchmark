#!/usr/bin/env python3
"""
Conformal Prediction for Distribution-Free Uncertainty Quantification
=====================================================================
Provides exact, non-asymptotic, distribution-free coverage guarantees
for all ML surrogate predictions in the CFD benchmark pipeline.

Implements three complementary conformal prediction strategies:

1. **Split Conformal Prediction** — partitions data into train + calibration;
   computes empirical quantile of absolute residuals to produce guaranteed
   prediction intervals on CL/CD surrogate outputs.

2. **Conformalized Quantile Regression (CQR)** — wraps dual-quantile base
   models with conformal calibration for spatially heteroscedastic Cp/Cf
   uncertainty bounds.

3. **Conformal Jackknife+** — leave-one-out conformal for small datasets
   (e.g., Greenblatt PIV hump data) where withholding a calibration split
   is infeasible.

Additionally provides an **OOD Flow Detector** that leverages expanding
prediction sets to flag spatial regions where the Boussinesq approximation
breaks down.

Mathematical Guarantees
-----------------------
For miscoverage rate α ∈ (0,1) and calibration set of size n:

    q̂ = ⌈(1 − α)(n + 1)⌉-th smallest nonconformity score

    P(Y_{n+1} ∈ C(X_{n+1})) ≥ 1 − α   (finite-sample, distribution-free)

References
----------
- Vovk, Gammerman & Shafer (2005) "Algorithmic Learning in a Random World"
- Romano, Patterson & Candès (2019) "Conformalized Quantile Regression"
- Barber et al. (2021) "Predictive Inference with the Jackknife+"

Usage
-----
    from scripts.ml_augmentation.conformal_prediction import (
        SplitConformalPredictor,
        ConformalizedQuantileRegression,
        ConformalJackknifeplus,
        OODFlowDetector,
    )

    # Wrap CL/CD MLP surrogate with split conformal
    cp = SplitConformalPredictor(alpha=0.05)
    cp.calibrate(cal_predictions, cal_true_values)
    intervals = cp.predict_interval(new_predictions)
    # intervals.lower, intervals.upper guaranteed to cover ≥95% of truths
"""

import logging
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)
PROJECT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT))


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class PredictionInterval:
    """Conformal prediction interval with guaranteed coverage.

    Attributes
    ----------
    lower : ndarray
        Lower bound of the prediction interval.
    upper : ndarray
        Upper bound of the prediction interval.
    point : ndarray
        Point prediction from the base model.
    width : ndarray
        Interval width (upper − lower).
    alpha : float
        Miscoverage rate (e.g., 0.05 for 95% coverage).
    method : str
        CP method used ('split', 'cqr', 'jackknife+').
    """
    lower: np.ndarray
    upper: np.ndarray
    point: np.ndarray
    width: np.ndarray
    alpha: float
    method: str

    def coverage(self, y_true: np.ndarray) -> float:
        """Empirical coverage on test data."""
        covered = (y_true >= self.lower) & (y_true <= self.upper)
        return float(np.mean(covered))

    def mean_width(self) -> float:
        """Average interval width."""
        return float(np.mean(self.width))

    def summary(self) -> Dict[str, Any]:
        """Summary statistics of the prediction intervals."""
        return {
            "method": self.method,
            "alpha": self.alpha,
            "target_coverage": 1.0 - self.alpha,
            "mean_width": self.mean_width(),
            "median_width": float(np.median(self.width)),
            "max_width": float(np.max(self.width)),
            "min_width": float(np.min(self.width)),
            "n_predictions": len(self.point),
        }


@dataclass
class OODReport:
    """Out-of-distribution detection report.

    Attributes
    ----------
    ood_flags : ndarray of bool
        True where prediction set width exceeds threshold.
    ood_fraction : float
        Fraction of spatial points flagged as OOD.
    spatial_widths : ndarray
        Prediction interval width at each spatial point.
    threshold : float
        Width threshold used for OOD detection.
    """
    ood_flags: np.ndarray
    ood_fraction: float
    spatial_widths: np.ndarray
    threshold: float


# =============================================================================
# Nonconformity Score Functions
# =============================================================================

class NonconformityScore:
    """Abstract base for nonconformity score functions.

    The score s(x, y) measures disagreement between a model's prediction
    and the true value. Higher scores = worse agreement.
    """

    def __call__(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Compute nonconformity scores.

        Parameters
        ----------
        y_pred : ndarray (n,) or (n, d)
            Model predictions.
        y_true : ndarray (n,) or (n, d)
            Ground truth values.

        Returns
        -------
        scores : ndarray (n,)
            Nonconformity scores.
        """
        raise NotImplementedError


class AbsoluteResidualScore(NonconformityScore):
    """s(x, y) = |y − ŷ(x)|

    The most common score for regression. Produces symmetric intervals.
    """

    def __call__(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        residuals = np.abs(y_true - y_pred)
        if residuals.ndim > 1:
            # Multi-output: take max across outputs for conservative bound
            residuals = np.max(residuals, axis=-1)
        return residuals


class NormalizedResidualScore(NonconformityScore):
    """s(x, y) = |y − ŷ(x)| / σ̂(x)

    Normalized by estimated local uncertainty. Produces adaptive intervals
    that are tighter where the model is confident.

    Parameters
    ----------
    sigma_estimator : callable
        Function mapping predictions to estimated uncertainty σ̂.
        If None, uses |ŷ| + ε to produce relative-error intervals.
    epsilon : float
        Small constant to prevent division by zero.
    """

    def __init__(self, sigma_estimator: Optional[Callable] = None,
                 epsilon: float = 1e-6):
        self.sigma_estimator = sigma_estimator
        self.epsilon = epsilon

    def __call__(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        if self.sigma_estimator is not None:
            sigma = self.sigma_estimator(y_pred)
        else:
            sigma = np.abs(y_pred) + self.epsilon
        scores = np.abs(y_true - y_pred) / (sigma + self.epsilon)
        if scores.ndim > 1:
            scores = np.max(scores, axis=-1)
        return scores


class QuantileScore(NonconformityScore):
    """Score for Conformalized Quantile Regression.

    s(x, y) = max(q̂_lo(x) − y, y − q̂_hi(x))

    Parameters
    ----------
    quantile_lo : ndarray
        Lower quantile predictions.
    quantile_hi : ndarray
        Upper quantile predictions.
    """

    def __init__(self):
        self.quantile_lo = None
        self.quantile_hi = None

    def set_quantiles(self, q_lo: np.ndarray, q_hi: np.ndarray):
        """Set the quantile predictions for score computation."""
        self.quantile_lo = q_lo
        self.quantile_hi = q_hi

    def __call__(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        if self.quantile_lo is None or self.quantile_hi is None:
            raise ValueError("Quantile predictions not set. Call set_quantiles() first.")
        scores = np.maximum(
            self.quantile_lo - y_true,
            y_true - self.quantile_hi
        )
        if scores.ndim > 1:
            scores = np.max(scores, axis=-1)
        return scores


# =============================================================================
# Split Conformal Prediction
# =============================================================================

class SplitConformalPredictor:
    """Split Conformal Prediction for regression surrogates.

    Partitions held-out data into a calibration set. Computes the
    ⌈(1−α)(n+1)⌉-th smallest absolute residual as the conformal
    quantile q̂. At test time, constructs intervals [ŷ − q̂, ŷ + q̂].

    Parameters
    ----------
    alpha : float
        Miscoverage rate. Default 0.05 (95% coverage guarantee).
    score_fn : NonconformityScore or None
        Nonconformity score function. Default: AbsoluteResidualScore.

    Mathematical Guarantee
    ----------------------
    P(Y_{n+1} ∈ [ŷ(X_{n+1}) − q̂, ŷ(X_{n+1}) + q̂]) ≥ 1 − α

    This holds exactly for exchangeable data, regardless of the
    data distribution or the model architecture.
    """

    def __init__(self, alpha: float = 0.05,
                 score_fn: Optional[NonconformityScore] = None):
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0,1), got {alpha}")
        self.alpha = alpha
        self.score_fn = score_fn or AbsoluteResidualScore()
        self.q_hat = None
        self.cal_scores = None
        self._calibrated = False

    def calibrate(self, y_pred: np.ndarray, y_true: np.ndarray):
        """Calibrate conformal predictor on held-out calibration data.

        Parameters
        ----------
        y_pred : ndarray (n,) or (n, d)
            Calibration set predictions from the base model.
        y_true : ndarray (n,) or (n, d)
            Calibration set ground truth.
        """
        y_pred = np.asarray(y_pred, dtype=np.float64)
        y_true = np.asarray(y_true, dtype=np.float64)

        n = len(y_pred)
        if n < 2:
            raise ValueError(f"Need at least 2 calibration points, got {n}")

        # Compute nonconformity scores
        self.cal_scores = self.score_fn(y_pred, y_true)

        # Compute conformal quantile with finite-sample correction
        # q̂ = ⌈(1 − α)(n + 1)⌉-th smallest score
        quantile_level = np.ceil((1 - self.alpha) * (n + 1)) / n
        quantile_level = min(quantile_level, 1.0)

        self.q_hat = float(np.quantile(self.cal_scores, quantile_level,
                                        interpolation='higher'))
        self._calibrated = True

        logger.info(
            f"Calibrated split CP: n={n}, α={self.alpha}, "
            f"q̂={self.q_hat:.6f}, median_score={np.median(self.cal_scores):.6f}"
        )

    def predict_interval(self, y_pred: np.ndarray) -> PredictionInterval:
        """Generate conformal prediction intervals.

        Parameters
        ----------
        y_pred : ndarray (m,) or (m, d)
            Test set predictions from the base model.

        Returns
        -------
        PredictionInterval
            Guaranteed prediction intervals.
        """
        if not self._calibrated:
            raise RuntimeError("Must call calibrate() before predict_interval()")

        y_pred = np.asarray(y_pred, dtype=np.float64)
        lower = y_pred - self.q_hat
        upper = y_pred + self.q_hat
        width = upper - lower

        return PredictionInterval(
            lower=lower, upper=upper, point=y_pred,
            width=width, alpha=self.alpha, method="split"
        )


# =============================================================================
# Conformalized Quantile Regression (CQR)
# =============================================================================

class QuantileRegressor:
    """Simple quantile regression model (pinball loss minimization).

    Two-layer MLP trained with pinball loss for a given quantile τ.
    Uses numpy for portability — replace with torch for GPU training.

    Parameters
    ----------
    input_dim : int
        Input feature dimension.
    hidden_dim : int
        Hidden layer size.
    quantile : float
        Target quantile τ ∈ (0, 1).
    lr : float
        Learning rate for gradient descent.
    n_epochs : int
        Number of training epochs.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(self, input_dim: int = 10, hidden_dim: int = 64,
                 quantile: float = 0.5, lr: float = 1e-3,
                 n_epochs: int = 200, seed: int = 42):
        self.quantile = quantile
        self.lr = lr
        self.n_epochs = n_epochs

        rng = np.random.RandomState(seed)
        scale = np.sqrt(2.0 / input_dim)
        self.W1 = rng.randn(input_dim, hidden_dim) * scale
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.randn(hidden_dim, 1) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(1)

    def _relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass: X → hidden → output."""
        h = self._relu(X @ self.W1 + self.b1)
        return (h @ self.W2 + self.b2).ravel()

    def _pinball_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Pinball (quantile) loss: ρ_τ(y − ŷ)."""
        residual = y_true - y_pred
        return float(np.mean(np.where(
            residual >= 0,
            self.quantile * residual,
            (self.quantile - 1) * residual
        )))

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train via mini-batch gradient descent with pinball loss.

        Parameters
        ----------
        X : ndarray (n, d)
            Training features.
        y : ndarray (n,)
            Training targets.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n = len(X)

        for epoch in range(self.n_epochs):
            # Forward pass
            h = self._relu(X @ self.W1 + self.b1)
            y_pred = (h @ self.W2 + self.b2).ravel()

            # Pinball loss gradient
            residual = y - y_pred
            grad_output = np.where(
                residual >= 0,
                -self.quantile * np.ones(n),
                (1 - self.quantile) * np.ones(n)
            ) / n

            # Backprop through output layer
            dW2 = h.T @ grad_output.reshape(-1, 1)
            db2 = np.sum(grad_output)

            # Backprop through hidden layer
            dh = grad_output.reshape(-1, 1) @ self.W2.T
            dh *= (h > 0).astype(float)  # ReLU gradient
            dW1 = X.T @ dh
            db1 = np.sum(dh, axis=0)

            # Update
            self.W1 -= self.lr * dW1
            self.b1 -= self.lr * db1
            self.W2 -= self.lr * dW2
            self.b2 -= self.lr * db2

        final_loss = self._pinball_loss(y, self.forward(X))
        logger.debug(f"QuantileRegressor(τ={self.quantile:.2f}) "
                     f"trained: final_loss={final_loss:.6f}")


class ConformalizedQuantileRegression:
    """Conformalized Quantile Regression (CQR).

    Trains two quantile regression models for the lower (α/2) and
    upper (1 − α/2) quantiles, then calibrates the intervals using
    conformal prediction to ensure exact finite-sample coverage.

    The CQR score function is:
        s(x, y) = max(q̂_lo(x) − y, y − q̂_hi(x))

    This produces **adaptive** intervals that are tighter in
    well-predicted regions and wider in uncertain regions.

    Parameters
    ----------
    alpha : float
        Miscoverage rate. Default 0.05 (95% coverage).
    input_dim : int
        Feature dimension for quantile regressors.
    hidden_dim : int
        Hidden layer size.
    lr : float
        Learning rate.
    n_epochs : int
        Training epochs for quantile regressors.

    References
    ----------
    Romano, Patterson & Candès (2019) "Conformalized Quantile Regression"
    """

    def __init__(self, alpha: float = 0.05, input_dim: int = 10,
                 hidden_dim: int = 64, lr: float = 1e-3,
                 n_epochs: int = 200, seed: int = 42):
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0,1), got {alpha}")
        self.alpha = alpha
        self.q_lo_model = QuantileRegressor(
            input_dim=input_dim, hidden_dim=hidden_dim,
            quantile=alpha / 2, lr=lr, n_epochs=n_epochs, seed=seed
        )
        self.q_hi_model = QuantileRegressor(
            input_dim=input_dim, hidden_dim=hidden_dim,
            quantile=1 - alpha / 2, lr=lr, n_epochs=n_epochs, seed=seed + 1
        )
        self.q_hat = None
        self._calibrated = False

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the lower and upper quantile regressors.

        Parameters
        ----------
        X_train : ndarray (n_train, d)
        y_train : ndarray (n_train,)
        """
        self.q_lo_model.fit(X_train, y_train)
        self.q_hi_model.fit(X_train, y_train)
        logger.info("CQR quantile models trained.")

    def calibrate(self, X_cal: np.ndarray, y_cal: np.ndarray):
        """Calibrate CQR on held-out calibration data.

        Parameters
        ----------
        X_cal : ndarray (n_cal, d)
        y_cal : ndarray (n_cal,)
        """
        X_cal = np.asarray(X_cal, dtype=np.float64)
        y_cal = np.asarray(y_cal, dtype=np.float64)
        n = len(X_cal)

        q_lo = self.q_lo_model.forward(X_cal)
        q_hi = self.q_hi_model.forward(X_cal)

        # CQR nonconformity score: max(q̂_lo − y, y − q̂_hi)
        scores = np.maximum(q_lo - y_cal, y_cal - q_hi)

        # Conformal quantile with finite-sample correction
        quantile_level = np.ceil((1 - self.alpha) * (n + 1)) / n
        quantile_level = min(quantile_level, 1.0)
        self.q_hat = float(np.quantile(scores, quantile_level,
                                        interpolation='higher'))
        self._calibrated = True

        logger.info(
            f"CQR calibrated: n={n}, α={self.alpha}, q̂={self.q_hat:.6f}"
        )

    def predict_interval(self, X_test: np.ndarray,
                         y_point: Optional[np.ndarray] = None
                         ) -> PredictionInterval:
        """Generate CQR prediction intervals.

        Parameters
        ----------
        X_test : ndarray (m, d)
            Test features.
        y_point : ndarray (m,) or None
            Optional point predictions for the interval center.
            If None, uses the midpoint of quantile predictions.

        Returns
        -------
        PredictionInterval
            Adaptive prediction intervals with coverage guarantee.
        """
        if not self._calibrated:
            raise RuntimeError("Must call calibrate() before predict_interval()")

        X_test = np.asarray(X_test, dtype=np.float64)
        q_lo = self.q_lo_model.forward(X_test) - self.q_hat
        q_hi = self.q_hi_model.forward(X_test) + self.q_hat

        if y_point is None:
            y_point = (q_lo + q_hi) / 2

        return PredictionInterval(
            lower=q_lo, upper=q_hi, point=y_point,
            width=q_hi - q_lo, alpha=self.alpha, method="cqr"
        )


# =============================================================================
# Conformal Jackknife+
# =============================================================================

class ConformalJackknifeplus:
    """Conformal Jackknife+ for small calibration datasets.

    Fits n predictive models by systematically excluding the i-th
    observation, computing leave-one-out residuals. Ideal for small
    experimental datasets (e.g., Greenblatt PIV wall hump data) where
    withholding a large calibration split is infeasible.

    The Jackknife+ interval at test point x is:
        C(x) = [q̂_α/2({ŷ_{−i}(x) − R_i}), q̂_{1−α/2}({ŷ_{−i}(x) + R_i})]

    where R_i = |y_i − ŷ_{−i}(x_i)| is the LOO residual.

    Parameters
    ----------
    alpha : float
        Miscoverage rate.
    model_factory : callable
        Factory function that returns a fresh model instance.
        Must support .fit(X, y) and .predict(X) interfaces.

    References
    ----------
    Barber et al. (2021) "Predictive Inference with the Jackknife+"
    """

    def __init__(self, alpha: float = 0.05,
                 model_factory: Optional[Callable] = None):
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0,1), got {alpha}")
        self.alpha = alpha
        self.model_factory = model_factory or self._default_model_factory
        self.loo_residuals = None
        self.loo_models = None
        self._calibrated = False

    @staticmethod
    def _default_model_factory():
        """Default: simple linear regression model."""
        return _LinearRegressor()

    def calibrate(self, X: np.ndarray, y: np.ndarray):
        """Fit LOO models and compute LOO residuals.

        Parameters
        ----------
        X : ndarray (n, d)
        y : ndarray (n,)
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n = len(X)

        self.loo_residuals = np.zeros(n)
        self.loo_models = []

        for i in range(n):
            # Leave-one-out
            X_loo = np.delete(X, i, axis=0)
            y_loo = np.delete(y, i)

            model = self.model_factory()
            model.fit(X_loo, y_loo)
            self.loo_models.append(model)

            # LOO prediction and residual
            y_pred_i = model.predict(X[i:i + 1])[0]
            self.loo_residuals[i] = np.abs(y[i] - y_pred_i)

        self._calibrated = True
        logger.info(
            f"Jackknife+ calibrated: n={n}, α={self.alpha}, "
            f"median_LOO_residual={np.median(self.loo_residuals):.6f}"
        )

    def predict_interval(self, X_test: np.ndarray) -> PredictionInterval:
        """Generate Jackknife+ prediction intervals.

        Parameters
        ----------
        X_test : ndarray (m, d)

        Returns
        -------
        PredictionInterval
        """
        if not self._calibrated:
            raise RuntimeError("Must call calibrate() before predict_interval()")

        X_test = np.asarray(X_test, dtype=np.float64)
        m = len(X_test)
        n = len(self.loo_models)

        # Compute LOO predictions at each test point
        loo_preds = np.zeros((n, m))
        for i, model in enumerate(self.loo_models):
            loo_preds[i] = model.predict(X_test)

        # Jackknife+ intervals
        lower_candidates = loo_preds - self.loo_residuals[:, np.newaxis]
        upper_candidates = loo_preds + self.loo_residuals[:, np.newaxis]

        # Take quantiles across LOO models
        q_lo = self.alpha / 2
        q_hi = 1 - self.alpha / 2

        lower = np.quantile(lower_candidates, q_lo, axis=0)
        upper = np.quantile(upper_candidates, q_hi, axis=0)
        point = np.mean(loo_preds, axis=0)

        return PredictionInterval(
            lower=lower, upper=upper, point=point,
            width=upper - lower, alpha=self.alpha, method="jackknife+"
        )


class _LinearRegressor:
    """Minimal linear regression for Jackknife+ default model."""

    def __init__(self):
        self.w = None
        self.b = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        n, d = X.shape
        X_aug = np.column_stack([X, np.ones(n)])
        try:
            params = np.linalg.lstsq(X_aug, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            params = np.zeros(d + 1)
        self.w = params[:d]
        self.b = params[d]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.w + self.b


# =============================================================================
# OOD Flow Detector
# =============================================================================

class OODFlowDetector:
    """Out-of-Distribution flow physics detector using conformal prediction.

    Monitors the spatial variation of prediction interval widths across
    the flow domain. In regions of attached, well-behaved boundary layers,
    intervals are naturally narrow. As the flow approaches separation or
    encounters shock-induced interactions, intervals widen dramatically.

    This spatially heteroscedastic behavior serves as an automated mechanism
    for detecting where the Boussinesq approximation breaks down.

    Parameters
    ----------
    predictor : SplitConformalPredictor or ConformalizedQuantileRegression
        A calibrated conformal predictor.
    threshold_factor : float
        OOD flag triggered when interval width exceeds
        threshold_factor × median_width. Default 3.0 (3σ equivalent).
    """

    def __init__(self, predictor, threshold_factor: float = 3.0):
        self.predictor = predictor
        self.threshold_factor = threshold_factor

    def detect(self, y_pred: np.ndarray,
               X_test: Optional[np.ndarray] = None) -> OODReport:
        """Run OOD detection across spatial points.

        Parameters
        ----------
        y_pred : ndarray (n_points,)
            Point predictions at spatial locations.
        X_test : ndarray (n_points, d) or None
            Test features (required for CQR, optional for split CP).

        Returns
        -------
        OODReport
            Detection results with spatial OOD flags.
        """
        if isinstance(self.predictor, ConformalizedQuantileRegression):
            if X_test is None:
                raise ValueError("X_test required for CQR-based OOD detection")
            intervals = self.predictor.predict_interval(X_test, y_point=y_pred)
        else:
            intervals = self.predictor.predict_interval(y_pred)

        widths = intervals.width
        median_width = np.median(widths)
        threshold = self.threshold_factor * median_width

        ood_flags = widths > threshold
        ood_fraction = float(np.mean(ood_flags))

        if ood_fraction > 0:
            logger.warning(
                f"OOD detected: {ood_fraction:.1%} of spatial points flagged "
                f"(width > {threshold:.4f}, threshold_factor={self.threshold_factor})"
            )

        return OODReport(
            ood_flags=ood_flags,
            ood_fraction=ood_fraction,
            spatial_widths=widths,
            threshold=threshold,
        )


# =============================================================================
# Conformal PCE Wrapper (extends bayesian_pce_uq.py)
# =============================================================================

class ConformalPCEWrapper:
    """Wraps Polynomial Chaos Expansion predictions with conformal intervals.

    Instead of relying on the PCE variance estimate (which assumes
    polynomial approximation accuracy), this wrapper applies split
    conformal prediction to provide distribution-free coverage guarantees
    on PCE-predicted quantities of interest.

    Parameters
    ----------
    pce_model : object
        Fitted PCE surrogate with .predict(samples) method.
    alpha : float
        Miscoverage rate.
    """

    def __init__(self, pce_model, alpha: float = 0.05):
        self.pce_model = pce_model
        self.cp = SplitConformalPredictor(alpha=alpha)

    def calibrate(self, X_cal: np.ndarray, y_cal: np.ndarray):
        """Calibrate conformal wrapper on PCE residuals.

        Parameters
        ----------
        X_cal : ndarray (n, d)
            Calibration parameter samples.
        y_cal : ndarray (n,) or (n, q)
            True QoI values at calibration points.
        """
        y_pred = self.pce_model.predict(X_cal)
        if y_pred.ndim > 1 and y_pred.shape[1] == 1:
            y_pred = y_pred.ravel()
        if y_cal.ndim > 1 and y_cal.shape[1] == 1:
            y_cal = y_cal.ravel()
        self.cp.calibrate(y_pred, y_cal)

    def predict_interval(self, X_test: np.ndarray) -> PredictionInterval:
        """Generate conformal prediction intervals for PCE predictions."""
        y_pred = self.pce_model.predict(X_test)
        if y_pred.ndim > 1 and y_pred.shape[1] == 1:
            y_pred = y_pred.ravel()
        return self.cp.predict_interval(y_pred)


# =============================================================================
# Convenience: wrap any model with conformal prediction
# =============================================================================

def conformalize_surrogate(
    model_predict: Callable,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    alpha: float = 0.05,
    method: str = "split",
    score_fn: Optional[NonconformityScore] = None,
) -> SplitConformalPredictor:
    """One-liner to wrap any prediction function with conformal intervals.

    Parameters
    ----------
    model_predict : callable
        Function f(X) → predictions.
    X_cal : ndarray (n, d)
        Calibration features.
    y_cal : ndarray (n,)
        Calibration true values.
    alpha : float
        Miscoverage rate.
    method : str
        'split' (default).
    score_fn : NonconformityScore or None
        Custom score function.

    Returns
    -------
    SplitConformalPredictor
        Calibrated conformal predictor.

    Example
    -------
    >>> from scripts.ml_augmentation.surrogate_model import predict_cl_cd
    >>> cp = conformalize_surrogate(predict_cl_cd, X_cal, y_cal, alpha=0.05)
    >>> intervals = cp.predict_interval(predict_cl_cd(X_test))
    """
    y_pred_cal = model_predict(X_cal)
    cp = SplitConformalPredictor(alpha=alpha, score_fn=score_fn)
    cp.calibrate(y_pred_cal, y_cal)
    return cp
