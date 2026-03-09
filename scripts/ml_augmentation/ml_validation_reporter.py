#!/usr/bin/env python3
"""
ML Validation Reporter
=========================
Comprehensive validation metrics, overfitting analysis, and
model comparison for all ML surrogates in the CFD framework.

Features:
  - Hold-out set metrics: R^2, RMSE, MAE, MAPE per output
  - Train vs test learning curves with overfitting gap detection
  - k-fold cross-validation with variance
  - Model architecture comparison (GP vs MLP vs ensemble)
  - LaTeX table export

Usage
-----
    from scripts.ml_augmentation.ml_validation_reporter import (
        MLValidator, generate_validation_report,
    )
    validator = MLValidator()
    report = validator.full_report(model, X, y)
"""

import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


# =============================================================================
# Metrics
# =============================================================================
@dataclass
class ValidationMetrics:
    """Comprehensive validation metrics for a single output."""
    output_name: str
    R2: float = 0.0
    RMSE: float = 0.0
    MAE: float = 0.0
    MAPE: float = 0.0
    max_error: float = 0.0
    n_samples: int = 0
    bias: float = 0.0       # Mean signed error
    std_error: float = 0.0  # Std of errors


@dataclass
class OverfittingAnalysis:
    """Overfitting diagnostic results."""
    is_overfitting: bool = False
    train_R2: float = 0.0
    test_R2: float = 0.0
    R2_gap: float = 0.0
    train_RMSE: float = 0.0
    test_RMSE: float = 0.0
    RMSE_ratio: float = 1.0   # test/train — >1.5 suggests overfitting
    learning_curve_train: List[float] = field(default_factory=list)
    learning_curve_test: List[float] = field(default_factory=list)
    learning_curve_sizes: List[int] = field(default_factory=list)
    recommendation: str = ""


@dataclass
class CrossValidationResult:
    """k-fold cross-validation results."""
    k: int = 5
    R2_mean: float = 0.0
    R2_std: float = 0.0
    RMSE_mean: float = 0.0
    RMSE_std: float = 0.0
    fold_R2s: List[float] = field(default_factory=list)
    fold_RMSEs: List[float] = field(default_factory=list)


@dataclass
class ModelComparisonEntry:
    """Results for a single model in the comparison table."""
    model_name: str
    model_type: str
    n_params: int = 0
    train_R2: float = 0.0
    test_R2: float = 0.0
    test_RMSE: float = 0.0
    test_MAPE: float = 0.0
    cv_R2_mean: float = 0.0
    cv_R2_std: float = 0.0
    training_time_s: float = 0.0
    overfitting: bool = False


# =============================================================================
# Metric Computation
# =============================================================================
def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, output_name: str = "output",
) -> ValidationMetrics:
    """Compute comprehensive validation metrics."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    errors = y_pred - y_true

    # R^2
    ss_res = np.sum(errors**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    R2 = 1 - ss_res / max(ss_tot, 1e-15)

    # RMSE
    RMSE = float(np.sqrt(np.mean(errors**2)))

    # MAE
    MAE = float(np.mean(np.abs(errors)))

    # MAPE
    mask = np.abs(y_true) > 1e-6
    MAPE = float(100 * np.mean(np.abs(errors[mask] / y_true[mask]))) if np.any(mask) else 0.0

    return ValidationMetrics(
        output_name=output_name,
        R2=float(R2),
        RMSE=RMSE,
        MAE=MAE,
        MAPE=MAPE,
        max_error=float(np.max(np.abs(errors))),
        n_samples=len(y_true),
        bias=float(np.mean(errors)),
        std_error=float(np.std(errors)),
    )


# =============================================================================
# Overfitting Analysis
# =============================================================================
def analyze_overfitting(
    model_fit_func,
    model_predict_func,
    X: np.ndarray,
    y: np.ndarray,
    n_sizes: int = 8,
    test_fraction: float = 0.2,
) -> OverfittingAnalysis:
    """
    Analyze overfitting via learning curves.

    Trains model on increasing fractions of data and tracks
    train vs test error to detect the overfitting gap.

    Parameters
    ----------
    model_fit_func : callable
        Function(X_train, y_train) that fits a model.
    model_predict_func : callable
        Function(X) that returns predictions.
    X, y : ndarray
        Full dataset.
    n_sizes : int
        Number of training set sizes to evaluate.
    test_fraction : float
        Hold-out fraction.
    """
    n_total = len(X)
    n_test = max(int(n_total * test_fraction), 2)
    n_train_max = n_total - n_test

    # Fixed test set
    rng = np.random.RandomState(42)
    indices = rng.permutation(n_total)
    test_idx = indices[:n_test]
    train_pool = indices[n_test:]

    X_test, y_test = X[test_idx], y[test_idx]

    sizes = np.unique(np.linspace(
        max(5, n_train_max // 10), n_train_max, n_sizes,
    ).astype(int))

    train_rmses, test_rmses = [], []

    for size in sizes:
        train_idx = train_pool[:size]
        X_train, y_train = X[train_idx], y[train_idx]

        model_fit_func(X_train, y_train)

        y_train_pred = model_predict_func(X_train)
        y_test_pred = model_predict_func(X_test)

        train_rmse = np.sqrt(np.mean((y_train.ravel() - y_train_pred.ravel())**2))
        test_rmse = np.sqrt(np.mean((y_test.ravel() - y_test_pred.ravel())**2))

        train_rmses.append(float(train_rmse))
        test_rmses.append(float(test_rmse))

    # Final metrics
    train_R2 = _r2(y[train_pool].ravel(), model_predict_func(X[train_pool]).ravel())
    test_R2 = _r2(y_test.ravel(), model_predict_func(X_test).ravel())

    R2_gap = train_R2 - test_R2
    RMSE_ratio = test_rmses[-1] / max(train_rmses[-1], 1e-15) if train_rmses else 1.0

    is_overfitting = R2_gap > 0.05 or RMSE_ratio > 1.5

    if is_overfitting:
        rec = ("Model shows overfitting (R2 gap={:.3f}, RMSE ratio={:.2f}). "
               "Consider: regularization, early stopping, more training data, "
               "or simpler architecture.").format(R2_gap, RMSE_ratio)
    elif R2_gap > 0.02:
        rec = "Mild overfitting detected. Monitor with larger datasets."
    else:
        rec = "No significant overfitting. Train/test performance is consistent."

    return OverfittingAnalysis(
        is_overfitting=is_overfitting,
        train_R2=float(train_R2),
        test_R2=float(test_R2),
        R2_gap=float(R2_gap),
        train_RMSE=train_rmses[-1] if train_rmses else 0,
        test_RMSE=test_rmses[-1] if test_rmses else 0,
        RMSE_ratio=float(RMSE_ratio),
        learning_curve_train=train_rmses,
        learning_curve_test=test_rmses,
        learning_curve_sizes=[int(s) for s in sizes],
        recommendation=rec,
    )


def _r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - ss_res / max(ss_tot, 1e-15)


# =============================================================================
# Cross-Validation
# =============================================================================
def cross_validate(
    model_fit_func,
    model_predict_func,
    X: np.ndarray,
    y: np.ndarray,
    k: int = 5,
) -> CrossValidationResult:
    """k-fold cross-validation."""
    n = len(X)
    rng = np.random.RandomState(42)
    indices = rng.permutation(n)
    fold_size = n // k

    fold_R2s, fold_RMSEs = [], []

    for i in range(k):
        test_idx = indices[i * fold_size:(i + 1) * fold_size]
        train_idx = np.concatenate([indices[:i * fold_size], indices[(i + 1) * fold_size:]])

        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        model_fit_func(X_train, y_train)
        y_pred = model_predict_func(X_test)

        fold_R2s.append(float(_r2(y_test.ravel(), y_pred.ravel())))
        fold_RMSEs.append(float(np.sqrt(np.mean((y_test.ravel() - y_pred.ravel())**2))))

    return CrossValidationResult(
        k=k,
        R2_mean=float(np.mean(fold_R2s)),
        R2_std=float(np.std(fold_R2s)),
        RMSE_mean=float(np.mean(fold_RMSEs)),
        RMSE_std=float(np.std(fold_RMSEs)),
        fold_R2s=fold_R2s,
        fold_RMSEs=fold_RMSEs,
    )


# =============================================================================
# Model Comparison
# =============================================================================
def compare_models(
    models: Dict[str, Tuple],  # {name: (fit_func, predict_func, type_label)}
    X: np.ndarray,
    y: np.ndarray,
    test_fraction: float = 0.2,
) -> List[ModelComparisonEntry]:
    """
    Compare multiple model architectures on the same data.

    Parameters
    ----------
    models : dict
        {name: (fit_func, predict_func, model_type_label)}
    X, y : ndarray
        Dataset.
    """
    import time

    n_test = max(int(len(X) * test_fraction), 2)
    rng = np.random.RandomState(42)
    idx = rng.permutation(len(X))
    X_train, y_train = X[idx[n_test:]], y[idx[n_test:]]
    X_test, y_test = X[idx[:n_test]], y[idx[:n_test]]

    results = []

    for name, (fit_fn, pred_fn, model_type) in models.items():
        t0 = time.time()
        fit_fn(X_train, y_train)
        train_time = time.time() - t0

        y_train_pred = pred_fn(X_train)
        y_test_pred = pred_fn(X_test)

        train_R2 = _r2(y_train.ravel(), y_train_pred.ravel())
        test_R2 = _r2(y_test.ravel(), y_test_pred.ravel())
        test_RMSE = float(np.sqrt(np.mean((y_test.ravel() - y_test_pred.ravel())**2)))

        mask = np.abs(y_test.ravel()) > 1e-6
        test_MAPE = float(100 * np.mean(np.abs(
            (y_test.ravel()[mask] - y_test_pred.ravel()[mask]) / y_test.ravel()[mask]
        ))) if np.any(mask) else 0

        results.append(ModelComparisonEntry(
            model_name=name,
            model_type=model_type,
            train_R2=float(train_R2),
            test_R2=float(test_R2),
            test_RMSE=test_RMSE,
            test_MAPE=test_MAPE,
            training_time_s=train_time,
            overfitting=(train_R2 - test_R2) > 0.05,
        ))

    return sorted(results, key=lambda r: -r.test_R2)


# =============================================================================
# Report Generation
# =============================================================================
def generate_validation_report(
    metrics: List[ValidationMetrics],
    overfitting: Optional[OverfittingAnalysis] = None,
    cv_result: Optional[CrossValidationResult] = None,
    comparison: Optional[List[ModelComparisonEntry]] = None,
) -> str:
    """Generate comprehensive validation text report."""
    lines = [
        "=" * 80,
        "ML Model Validation Report",
        "=" * 80,
    ]

    # Per-output metrics
    lines.extend([
        "",
        "Hold-Out Set Metrics:",
        f"  {'Output':<20} {'R2':>8} {'RMSE':>12} {'MAE':>12} {'MAPE%':>8} {'Bias':>12}",
        "  " + "-" * 75,
    ])
    for m in metrics:
        lines.append(
            f"  {m.output_name:<20} {m.R2:>8.4f} {m.RMSE:>12.6f} "
            f"{m.MAE:>12.6f} {m.MAPE:>7.2f}% {m.bias:>12.6f}"
        )

    # Overfitting
    if overfitting:
        lines.extend([
            "",
            "Overfitting Analysis:",
            f"  Train R2:     {overfitting.train_R2:.4f}",
            f"  Test R2:      {overfitting.test_R2:.4f}",
            f"  R2 Gap:       {overfitting.R2_gap:.4f}",
            f"  RMSE Ratio:   {overfitting.RMSE_ratio:.2f} (test/train)",
            f"  Overfitting:  {'YES' if overfitting.is_overfitting else 'NO'}",
            f"  Recommendation: {overfitting.recommendation}",
        ])

    # Cross-validation
    if cv_result:
        lines.extend([
            "",
            f"{cv_result.k}-Fold Cross-Validation:",
            f"  R2:   {cv_result.R2_mean:.4f} +/- {cv_result.R2_std:.4f}",
            f"  RMSE: {cv_result.RMSE_mean:.6f} +/- {cv_result.RMSE_std:.6f}",
            f"  Per-fold R2: {[f'{r:.4f}' for r in cv_result.fold_R2s]}",
        ])

    # Model comparison
    if comparison:
        lines.extend([
            "",
            "Model Architecture Comparison:",
            f"  {'Model':<20} {'Type':<10} {'Train R2':>10} {'Test R2':>10} "
            f"{'RMSE':>10} {'MAPE%':>8} {'Overfit?':>9} {'Time(s)':>8}",
            "  " + "-" * 85,
        ])
        for c in comparison:
            of = "YES" if c.overfitting else "no"
            lines.append(
                f"  {c.model_name:<20} {c.model_type:<10} {c.train_R2:>10.4f} "
                f"{c.test_R2:>10.4f} {c.test_RMSE:>10.6f} {c.test_MAPE:>7.2f}% "
                f"{of:>9} {c.training_time_s:>7.2f}"
            )

    lines.append("\n" + "=" * 80)
    return "\n".join(lines)


def generate_latex_metrics_table(
    metrics: List[ValidationMetrics],
) -> str:
    """Generate LaTeX table for validation metrics."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{ML surrogate validation metrics on hold-out set.}",
        r"\label{tab:ml_validation}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Output & $R^2$ & RMSE & MAE & MAPE (\%) & Bias \\",
        r"\midrule",
    ]
    for m in metrics:
        lines.append(
            f"  {m.output_name} & {m.R2:.4f} & {m.RMSE:.6f} & "
            f"{m.MAE:.6f} & {m.MAPE:.2f} & {m.bias:.6f} \\\\"
        )
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


# =============================================================================
# Demo: Validate Existing Surrogates
# =============================================================================
def demo_validate_cl_surrogate():
    """Demo validation of the CL/CD surrogate."""
    from sklearn.neural_network import MLPRegressor

    # Synthetic CL data
    rng = np.random.RandomState(42)
    aoa = rng.uniform(-5, 18, 150)
    Re = 10**rng.uniform(5.5, 7, 150)
    X = np.column_stack([aoa, Re])
    CL = 0.11 * aoa * (1 - 0.02 * np.abs(aoa)) + rng.normal(0, 0.01, 150)
    y = CL.reshape(-1, 1)

    mlp = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=500, random_state=42)

    def fit_fn(X_t, y_t):
        mlp.fit(X_t, y_t.ravel())

    def pred_fn(X_p):
        return mlp.predict(X_p).reshape(-1, 1)

    # Metrics
    fit_fn(X[:120], y[:120])
    y_pred = pred_fn(X[120:])
    metrics = [compute_metrics(y[120:], y_pred, "CL")]

    # Overfitting
    overfitting = analyze_overfitting(fit_fn, pred_fn, X, y)

    # Cross-validation
    cv = cross_validate(fit_fn, pred_fn, X, y, k=5)

    report = generate_validation_report(metrics, overfitting, cv)
    print(report)

    return metrics, overfitting, cv


if __name__ == "__main__":
    demo_validate_cl_surrogate()
