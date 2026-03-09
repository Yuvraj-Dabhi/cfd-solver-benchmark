"""
ML Model Evaluation
===================
Comprehensive evaluation metrics and analysis for trained
turbulence correction models.

Includes:
- Standard regression metrics (RMSE, MAE, R²)
- Physics-consistency checks
- Generalization assessment (leave-one-case-out)
- Realizability constraints (Lumley triangle, positive k)
"""

import json
import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Container for model evaluation metrics."""
    model_name: str
    rmse: float = 0.0
    mae: float = 0.0
    r2: float = 0.0
    max_error: float = 0.0
    mape: float = 0.0
    # Physics metrics
    realizability_fraction: float = 0.0
    energy_positivity: float = 0.0
    # Per-component
    per_component: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # Generalization
    generalization: Dict[str, float] = field(default_factory=dict)


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: List[str] = None,
    model_name: str = "model",
) -> EvaluationResult:
    """
    Compute comprehensive evaluation metrics.

    Parameters
    ----------
    y_true : ndarray (N, n_targets)
        Ground truth values.
    y_pred : ndarray (N, n_targets)
        Model predictions.
    target_names : list of str
        Names of target components.
    model_name : str
        Identifier for this evaluation.
    """
    if y_true.ndim == 1:
        y_true = y_true[:, None]
        y_pred = y_pred[:, None]

    n_targets = y_true.shape[1]
    if target_names is None:
        target_names = [f"target_{i}" for i in range(n_targets)]

    # Global metrics
    residual = y_true - y_pred
    rmse = np.sqrt(np.mean(residual ** 2))
    mae = np.mean(np.abs(residual))
    max_error = np.max(np.abs(residual))

    # R² (coefficient of determination)
    ss_res = np.sum(residual ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true, axis=0)) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-15)

    # MAPE (handle zeros)
    mask = np.abs(y_true) > 1e-10
    if np.any(mask):
        mape = np.mean(np.abs(residual[mask] / y_true[mask])) * 100
    else:
        mape = 0.0

    result = EvaluationResult(
        model_name=model_name,
        rmse=rmse, mae=mae, r2=r2,
        max_error=max_error, mape=mape,
    )

    # Per-component metrics
    for i, name in enumerate(target_names):
        res_i = y_true[:, i] - y_pred[:, i]
        ss_res_i = np.sum(res_i ** 2)
        ss_tot_i = np.sum((y_true[:, i] - np.mean(y_true[:, i])) ** 2)
        result.per_component[name] = {
            "rmse": np.sqrt(np.mean(res_i ** 2)),
            "mae": np.mean(np.abs(res_i)),
            "r2": 1 - ss_res_i / (ss_tot_i + 1e-15),
        }

    return result


def check_realizability(
    b_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Check physical realizability of predicted anisotropy tensor.

    Constraints:
    1. Eigenvalues of b_ij must be ≥ -1/3 and ≤ 2/3
    2. Points must lie within Lumley triangle
    3. Trace of b_ij must be 0 (by definition)

    Parameters
    ----------
    b_pred : ndarray (N, 3) or (N, 6)
        Predicted anisotropy components [b11, b12, b22] or full tensor.
    """
    N = b_pred.shape[0]
    realizable = 0
    positive_tke = 0

    for i in range(N):
        if b_pred.shape[1] >= 3:
            b11, b12, b22 = b_pred[i, 0], b_pred[i, 1], b_pred[i, 2]
            b33 = -(b11 + b22)  # Trace-free

            # Build tensor
            b = np.array([
                [b11, b12, 0],
                [b12, b22, 0],
                [0, 0, b33],
            ])

            eigs = np.linalg.eigvalsh(b)

            # Constraint: -1/3 ≤ eigenvalues ≤ 2/3
            if np.all(eigs >= -1/3 - 1e-6) and np.all(eigs <= 2/3 + 1e-6):
                realizable += 1

            # Positive norm stresses
            if b11 > -1/3 and b22 > -1/3 and b33 > -1/3:
                positive_tke += 1

    return {
        "realizability_fraction": realizable / N * 100,
        "positive_tke_fraction": positive_tke / N * 100,
    }


def generalization_assessment(
    model: Callable,
    features: np.ndarray,
    targets: np.ndarray,
    case_labels: List[str],
) -> pd.DataFrame:
    """
    Leave-one-case-out cross-validation for generalization assessment.

    Parameters
    ----------
    model : callable
        Function that takes (X_train, y_train, X_test) → y_pred.
    features : ndarray (N, n_features)
    targets : ndarray (N, n_targets)
    case_labels : list of str
        Case label for each sample.

    Returns
    -------
    DataFrame with per-case generalization metrics.
    """
    unique_cases = sorted(set(case_labels))
    case_labels = np.array(case_labels)
    rows = []

    for test_case in unique_cases:
        test_mask = case_labels == test_case
        train_mask = ~test_mask

        X_train, y_train = features[train_mask], targets[train_mask]
        X_test, y_test = features[test_mask], targets[test_mask]

        try:
            y_pred = model(X_train, y_train, X_test)
            result = evaluate_predictions(y_test, y_pred, model_name=test_case)
            rows.append({
                "test_case": test_case,
                "n_test": int(np.sum(test_mask)),
                "rmse": result.rmse,
                "mae": result.mae,
                "r2": result.r2,
            })
        except Exception as e:
            logger.warning(f"Failed on case {test_case}: {e}")
            rows.append({
                "test_case": test_case,
                "n_test": int(np.sum(test_mask)),
                "rmse": np.nan,
                "mae": np.nan,
                "r2": np.nan,
            })

    df = pd.DataFrame(rows)
    logger.info(f"Generalization RMSE — mean: {df['rmse'].mean():.6f}, "
                f"std: {df['rmse'].std():.6f}")
    return df


def comparison_table(
    model_results: Dict[str, EvaluationResult],
) -> pd.DataFrame:
    """
    Build comparison table across multiple models.

    Parameters
    ----------
    model_results : dict
        {model_name: EvaluationResult}
    """
    rows = []
    for name, result in model_results.items():
        rows.append({
            "Model": name,
            "RMSE": result.rmse,
            "MAE": result.mae,
            "R²": result.r2,
            "MAPE (%)": result.mape,
            "Max Error": result.max_error,
            "Realizability (%)": result.realizability_fraction,
        })
    return pd.DataFrame(rows).sort_values("RMSE")


def print_evaluation_report(result: EvaluationResult) -> None:
    """Print formatted evaluation report."""
    print(f"\n{'='*50}")
    print(f"  Model Evaluation: {result.model_name}")
    print(f"{'='*50}")
    print(f"  RMSE:      {result.rmse:.6f}")
    print(f"  MAE:       {result.mae:.6f}")
    print(f"  R²:        {result.r2:.4f}")
    print(f"  MAPE:      {result.mape:.2f}%")
    print(f"  Max Error: {result.max_error:.6f}")

    if result.per_component:
        print(f"\n  Per-Component:")
        for name, metrics in result.per_component.items():
            print(f"    {name:>15s}: RMSE={metrics['rmse']:.6f}, R²={metrics['r2']:.4f}")

    if result.generalization:
        print(f"\n  Generalization:")
        for case, rmse in result.generalization.items():
            print(f"    {case:>25s}: RMSE={rmse:.6f}")

    print(f"{'='*50}")


def save_evaluation(result: EvaluationResult, path: Path) -> None:
    """Save evaluation results to JSON."""
    path = Path(path)
    data = {
        "model_name": result.model_name,
        "rmse": result.rmse,
        "mae": result.mae,
        "r2": result.r2,
        "mape": result.mape,
        "max_error": result.max_error,
        "per_component": result.per_component,
        "generalization": result.generalization,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Evaluation saved to {path}")
