"""
Error Metrics Module
====================
Comprehensive validation metrics for CFD-vs-experiment comparison.
Implements ASME V&V 20-2009 validation metric alongside standard
statistical measures (RMSE, MAE, MAPE, NRMSE, R²).
"""

import numpy as np
from typing import Dict, Optional, Tuple


def rmse(cfd: np.ndarray, exp: np.ndarray) -> float:
    """Root Mean Square Error."""
    return float(np.sqrt(np.mean((cfd - exp) ** 2)))


def mae(cfd: np.ndarray, exp: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(cfd - exp)))


def mape(cfd: np.ndarray, exp: np.ndarray, eps: float = 1e-10) -> float:
    """Mean Absolute Percentage Error (%)."""
    return float(np.mean(np.abs((cfd - exp) / (np.abs(exp) + eps))) * 100)


def nrmse(cfd: np.ndarray, exp: np.ndarray) -> float:
    """Normalized RMSE (by range of experimental data)."""
    data_range = np.max(exp) - np.min(exp)
    if data_range < 1e-15:
        return float("inf")
    return rmse(cfd, exp) / data_range


def correlation_coefficient(cfd: np.ndarray, exp: np.ndarray) -> float:
    """Pearson correlation coefficient R²."""
    if len(cfd) < 2:
        return 0.0
    r = np.corrcoef(cfd, exp)[0, 1]
    return float(r ** 2)


def max_error(cfd: np.ndarray, exp: np.ndarray) -> float:
    """Maximum absolute error."""
    return float(np.max(np.abs(cfd - exp)))


# =============================================================================
# ASME V&V 20-2009 Validation Metric (Section 9)
# =============================================================================
def asme_vv20_metric(
    cfd: np.ndarray,
    exp: np.ndarray,
    exp_uncertainty: np.ndarray,
    cfd_uncertainty: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    ASME V&V 20-2009 validation metric.

    Per ASME V&V 20 Section 9, validation is assessed by comparing the
    absolute comparison error |E| to the validation uncertainty u_val
    at each measurement point:

        E_i = S_i - D_i           (comparison error)
        u_val_i = sqrt(u_num_i^2 + u_D_i^2)   (validation uncertainty)

    If |E_i| <= u_val_i for all points, the model is validated at the
    achieved level of uncertainty.  If the inequality fails, there is
    a model-form deficiency that cannot be hidden by the uncertainties.

    **Important**: The exp_uncertainty and cfd_uncertainty inputs should
    already be at the desired confidence level (typically 95%, i.e. the
    coverage factor k=2 should already be included in the inputs).
    This function does NOT apply an additional coverage factor.

    Parameters
    ----------
    cfd : array
        CFD predicted values (S).
    exp : array
        Experimental reference values (D).
    exp_uncertainty : array
        Experimental expanded uncertainty (u_D) at each point.
        For Greenblatt wall-hump Cp, u_D ≈ 0.01 (±1% Cp).
    cfd_uncertainty : array, optional
        CFD numerical uncertainty (u_num), e.g. from GCI.
        If None, u_num = 0 (numerical error is neglected).

    Returns
    -------
    dict with keys:
        'E_mean'      : mean comparison error
        'E_rms'       : RMS comparison error
        'u_val_mean'  : mean validation uncertainty
        'u_val_rms'   : RMS validation uncertainty
        'metric_mean' : mean of |E_i| / u_val_i
        'metric_max'  : max of |E_i| / u_val_i
        'status'      : "VALIDATED" or "NOT VALIDATED"
        'warning'     : str, present if u_val >> |E| (inflated uncertainty)
    """
    E = cfd - exp  # Comparison error

    # Validation uncertainty = sqrt(u_exp^2 + u_cfd^2)
    u_exp2 = exp_uncertainty ** 2
    u_cfd2 = cfd_uncertainty ** 2 if cfd_uncertainty is not None else 0.0
    u_val = np.sqrt(u_exp2 + u_cfd2)

    # Metric: |E_i| / u_val_i (per ASME V&V 20 Section 9)
    # No additional coverage factor — uncertainties should already be
    # at the desired confidence level when passed to this function.
    metric = np.abs(E) / (u_val + 1e-15)

    # Overall status: all points must pass
    avg_metric = float(np.mean(metric))
    max_metric = float(np.max(metric))
    status = "VALIDATED" if max_metric < 1.0 else "NOT VALIDATED"

    E_rms = float(np.sqrt(np.mean(E ** 2)))
    u_val_rms = float(np.sqrt(np.mean(u_val ** 2)))
    u_val_mean = float(np.mean(u_val))

    result = {
        "E_mean": float(np.mean(E)),
        "E_rms": E_rms,
        "u_val_mean": u_val_mean,
        "u_val_rms": u_val_rms,
        "metric_mean": avg_metric,
        "metric_max": max_metric,
        "status": status,
    }

    # Sanity check: if u_val >> |E|, the metric is trivially satisfied
    # and the result is uninformative (inflated uncertainty)
    if E_rms > 1e-15 and u_val_mean / E_rms > 3.0:
        result["warning"] = (
            f"u_val/E_rms = {u_val_mean/E_rms:.1f}; validation uncertainty "
            f"is {u_val_mean/E_rms:.0f}x larger than the comparison error. "
            f"The metric is trivially satisfied — review u_D and u_num sources."
        )

    return result


# =============================================================================
# Feature-Based Metrics (Separation)
# =============================================================================
def separation_metrics(
    x: np.ndarray,
    Cf: np.ndarray,
    x_sep_exp: float,
    x_reat_exp: float,
) -> Dict[str, float]:
    """
    Compute separation-specific metrics.

    Parameters
    ----------
    x : array
        Streamwise coordinate.
    Cf : array
        Skin friction coefficient.
    x_sep_exp : float
        Experimental separation point.
    x_reat_exp : float
        Experimental reattachment point.

    Returns
    -------
    dict with sep/reat points, errors, and bubble length deviation.
    """
    # Find separation: Cf crosses zero from positive to negative
    x_sep_cfd = _find_zero_crossing(x, Cf, direction="negative")

    # Find reattachment: Cf crosses zero from negative to positive
    x_reat_cfd = _find_zero_crossing(x, Cf, direction="positive")

    bubble_exp = x_reat_exp - x_sep_exp
    bubble_cfd = (x_reat_cfd - x_sep_cfd) if (x_sep_cfd and x_reat_cfd) else None

    results = {
        "x_sep_cfd": x_sep_cfd,
        "x_reat_cfd": x_reat_cfd,
        "x_sep_error": abs(x_sep_cfd - x_sep_exp) if x_sep_cfd else None,
        "x_reat_error": abs(x_reat_cfd - x_reat_exp) if x_reat_cfd else None,
        "bubble_length_cfd": bubble_cfd,
        "bubble_length_exp": bubble_exp,
    }

    if bubble_cfd is not None and bubble_exp > 0:
        results["bubble_deviation_pct"] = abs(bubble_cfd - bubble_exp) / bubble_exp * 100

    return results


def _find_zero_crossing(
    x: np.ndarray, y: np.ndarray, direction: str = "negative"
) -> Optional[float]:
    """Find x where y crosses zero."""
    for i in range(len(y) - 1):
        if direction == "negative" and y[i] > 0 and y[i + 1] < 0:
            # Linear interpolation
            return float(x[i] - y[i] * (x[i + 1] - x[i]) / (y[i + 1] - y[i]))
        elif direction == "positive" and y[i] < 0 and y[i + 1] > 0:
            return float(x[i] - y[i] * (x[i + 1] - x[i]) / (y[i + 1] - y[i]))
    return None


# =============================================================================
# Comprehensive Report
# =============================================================================
def compute_all_metrics(
    cfd: np.ndarray,
    exp: np.ndarray,
    exp_uncertainty: Optional[np.ndarray] = None,
    label: str = "",
) -> Dict[str, float]:
    """
    Compute all available metrics for a CFD-vs-experiment comparison.

    Returns
    -------
    dict with RMSE, MAE, MAPE, NRMSE, R², max_error, and optionally ASME V&V.
    """
    metrics = {
        "label": label,
        "n_points": len(cfd),
        "RMSE": rmse(cfd, exp),
        "MAE": mae(cfd, exp),
        "MAPE": mape(cfd, exp),
        "NRMSE": nrmse(cfd, exp),
        "R2": correlation_coefficient(cfd, exp),
        "max_error": max_error(cfd, exp),
    }

    # Lowercase aliases for workflow script compatibility
    metrics["rmse"] = metrics["RMSE"]
    metrics["mae"] = metrics["MAE"]
    metrics["mape"] = metrics["MAPE"]
    metrics["nrmse"] = metrics["NRMSE"]
    metrics["r_squared"] = metrics["R2"]

    if exp_uncertainty is not None:
        vv = asme_vv20_metric(cfd, exp, exp_uncertainty)
        metrics["ASME_VV20_status"] = vv["status"]
        metrics["ASME_VV20_metric"] = vv["metric_mean"]

    return metrics


def print_metrics(metrics: Dict[str, float]) -> None:
    """Pretty-print validation metrics."""
    label = metrics.get("label", "")
    print(f"\n{'='*50}")
    if label:
        print(f"  Validation Metrics: {label}")
    print(f"{'='*50}")
    print(f"  Points:       {metrics.get('n_points', 'N/A')}")
    print(f"  RMSE:         {metrics.get('RMSE', 0):.6f}")
    print(f"  MAE:          {metrics.get('MAE', 0):.6f}")
    print(f"  MAPE:         {metrics.get('MAPE', 0):.2f}%")
    print(f"  NRMSE:        {metrics.get('NRMSE', 0):.4f}")
    print(f"  R²:           {metrics.get('R2', 0):.4f}")
    print(f"  Max Error:    {metrics.get('max_error', 0):.6f}")
    if "ASME_VV20_status" in metrics:
        print(f"  ASME V&V 20:  {metrics['ASME_VV20_status']} "
              f"(metric={metrics['ASME_VV20_metric']:.3f})")
    print(f"{'='*50}")
