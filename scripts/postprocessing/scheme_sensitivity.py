"""
Numerical Scheme Sensitivity Analysis
======================================
Compares CFD results across the 4-scheme sensitivity matrix
from config.py to quantify numerical discretization effects.

Scheme matrix:
  0: 1st-order upwind baseline
  1: 2nd-order linearUpwind (standard)
  2: 2nd-order limited
  3: Blended LUST
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class SchemeSensitivityResult:
    """Results from scheme sensitivity analysis."""
    quantity_name: str
    scheme_labels: List[str]
    values: Dict[str, float]  # {scheme_label: value}
    spread: float = 0.0      # max - min
    cv: float = 0.0          # coefficient of variation
    reference_scheme: str = "2nd-order standard"
    deviations: Dict[str, float] = field(default_factory=dict)  # % deviation from reference


def analyze_scheme_sensitivity(
    quantities: Dict[str, Dict[str, float]],
    reference_scheme: str = "2nd-order standard",
) -> Dict[str, SchemeSensitivityResult]:
    """
    Analyze sensitivity of output quantities to numerical scheme choice.

    Parameters
    ----------
    quantities : dict
        {quantity_name: {scheme_label: value}}
        e.g., {"x_reat": {"1st-order": 6.0, "2nd-order standard": 6.26, ...}}
    reference_scheme : str
        Label of the reference scheme for deviation computation.

    Returns
    -------
    dict of SchemeSensitivityResult for each quantity.
    """
    results = {}

    for qty_name, scheme_values in quantities.items():
        labels = list(scheme_values.keys())
        vals = np.array([scheme_values[l] for l in labels])

        result = SchemeSensitivityResult(
            quantity_name=qty_name,
            scheme_labels=labels,
            values=scheme_values,
            spread=np.max(vals) - np.min(vals),
            reference_scheme=reference_scheme,
        )

        # Coefficient of variation
        mean = np.mean(vals)
        if abs(mean) > 1e-15:
            result.cv = np.std(vals) / abs(mean) * 100  # %

        # Deviations from reference
        ref_val = scheme_values.get(reference_scheme)
        if ref_val is not None and abs(ref_val) > 1e-15:
            for label, val in scheme_values.items():
                result.deviations[label] = (val - ref_val) / abs(ref_val) * 100

        results[qty_name] = result

    return results


def scheme_order_study(
    profile_1st: np.ndarray,
    profile_2nd: np.ndarray,
    profile_lust: np.ndarray,
    x: np.ndarray,
) -> Dict[str, float]:
    """
    Compare profiles from different schemes to quantify numerical diffusion.

    Parameters
    ----------
    profile_1st, profile_2nd, profile_lust : ndarray
        Solution profiles from different schemes.
    x : ndarray
        Coordinate array.

    Returns
    -------
    dict with scheme comparison metrics.
    """
    # Numerical diffusion indicator: difference between 1st and higher order
    diff_1st_2nd = np.sqrt(np.mean((profile_1st - profile_2nd) ** 2))
    diff_2nd_lust = np.sqrt(np.mean((profile_2nd - profile_lust) ** 2))

    # If 1st→2nd change >> 2nd→LUST change, solution is scheme-converged
    if diff_1st_2nd > 1e-15:
        convergence_ratio = diff_2nd_lust / diff_1st_2nd
    else:
        convergence_ratio = 0.0

    return {
        "rmse_1st_vs_2nd": diff_1st_2nd,
        "rmse_2nd_vs_lust": diff_2nd_lust,
        "convergence_ratio": convergence_ratio,
        "scheme_converged": convergence_ratio < 0.3,
    }


def separation_scheme_sensitivity(
    scheme_data: Dict[str, Tuple[float, float]],
) -> pd.DataFrame:
    """
    Compare separation/reattachment points across schemes.

    Parameters
    ----------
    scheme_data : dict
        {scheme_label: (x_sep, x_reat)}

    Returns
    -------
    DataFrame with scheme comparison table.
    """
    rows = []
    for scheme, (x_sep, x_reat) in scheme_data.items():
        bubble = x_reat - x_sep if x_sep is not None and x_reat is not None else None
        rows.append({
            "Scheme": scheme,
            "x_sep": x_sep,
            "x_reat": x_reat,
            "Bubble Length": bubble,
        })

    df = pd.DataFrame(rows)

    # Add deviations from mean
    for col in ["x_sep", "x_reat", "Bubble Length"]:
        vals = df[col].dropna()
        if len(vals) > 0:
            mean_val = vals.mean()
            df[f"{col}_dev_%"] = ((df[col] - mean_val) / abs(mean_val) * 100).round(2)

    return df


def print_scheme_sensitivity_report(
    results: Dict[str, SchemeSensitivityResult],
) -> None:
    """Print formatted scheme sensitivity report."""
    print(f"\n{'='*65}")
    print(f"  Numerical Scheme Sensitivity Analysis")
    print(f"{'='*65}")

    for qty_name, result in results.items():
        print(f"\n  Quantity: {qty_name}")
        print(f"  {'Scheme':<25s} {'Value':>10s} {'Deviation':>10s}")
        print(f"  {'-'*47}")
        for label in result.scheme_labels:
            val = result.values[label]
            dev = result.deviations.get(label, 0.0)
            marker = " ←ref" if label == result.reference_scheme else ""
            print(f"  {label:<25s} {val:10.4f} {dev:+9.2f}%{marker}")
        print(f"  Spread: {result.spread:.4f}, CV: {result.cv:.2f}%")

        if result.cv > 5:
            print(f"  ⚠ High scheme sensitivity (CV > 5%) — refine mesh")
        elif result.cv < 1:
            print(f"  ✓ Scheme-independent (CV < 1%)")

    print(f"\n{'='*65}")
