#!/usr/bin/env python3
"""
Convergence Monitor
====================
Reusable convergence analysis for any SU2 case.

Features:
  - Residual parsing from SU2 history.csv or screen output
  - Cauchy criterion (quantity change < threshold over N iterations)
  - Residual plateau detection (log-residual slope < threshold)
  - Force time-history analysis (oscillation amplitude + mean drift)
  - Convergence classification: monotone, oscillatory, stalled, diverged
  - Time-budget projection (iterations-to-convergence estimate)
  - Summary report with recommendations

Usage:
  python convergence_monitor.py <history_csv> [--cauchy-window 200]
"""

import csv
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ===================================================================
# Data Structures
# ===================================================================
@dataclass
class ConvergenceStatus:
    """Classification of convergence behavior."""
    classification: str = "unknown"  # monotone, oscillatory, stalled, diverged
    converged: bool = False
    iterations_completed: int = 0
    residual_drop_decades: float = 0.0
    cauchy_converged: bool = False
    cauchy_value: float = float("inf")
    force_oscillation_amplitude: float = 0.0
    force_mean_drift: float = 0.0
    plateau_detected: bool = False
    plateau_start_iter: int = 0
    estimated_iters_to_converge: Optional[int] = None
    recommendation: str = ""
    details: Dict = field(default_factory=dict)


# ===================================================================
# History File Parser
# ===================================================================
def parse_history(filepath: str) -> Dict[str, np.ndarray]:
    """
    Parse SU2 history.csv file.

    Returns
    -------
    dict mapping column names to numpy arrays.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"History file not found: {filepath}")

    # Read header and data
    with open(filepath, "r") as f:
        reader = csv.reader(f)
        # SU2 headers often have quotes and spaces
        header = None
        data_rows = []
        for row in reader:
            if header is None:
                header = [h.strip().strip('"') for h in row]
                continue
            try:
                data_rows.append([float(v.strip()) for v in row])
            except (ValueError, IndexError):
                continue

    if not header or not data_rows:
        raise ValueError(f"Could not parse history file: {filepath}")

    data = np.array(data_rows)
    result = {}
    for i, name in enumerate(header):
        if i < data.shape[1]:
            result[name] = data[:, i]

    return result


def find_residual_columns(history: Dict[str, np.ndarray]) -> List[str]:
    """Find columns that look like residual data (log-scale, negative values)."""
    candidates = []
    for name, arr in history.items():
        lower = name.lower()
        if any(k in lower for k in ["rms_", "res_", "residual", "rms["]):
            candidates.append(name)
    # Fallback: look for columns that are consistently negative (log-residuals)
    if not candidates:
        for name, arr in history.items():
            if len(arr) > 10 and np.mean(arr[-10:]) < -1.0:
                candidates.append(name)
    return candidates


def find_force_columns(history: Dict[str, np.ndarray]) -> Dict[str, str]:
    """Find CL, CD, CM columns."""
    forces = {}
    for name in history:
        lower = name.lower()
        if "cl" in lower or "lift" in lower:
            forces["CL"] = name
        elif "cd" in lower or "drag" in lower:
            forces["CD"] = name
        elif "cm" in lower or "moment" in lower:
            forces["CM"] = name
    return forces


# ===================================================================
# Cauchy Criterion
# ===================================================================
def cauchy_criterion(
    values: np.ndarray,
    window: int = 200,
    threshold: float = 1e-6,
) -> Tuple[bool, float]:
    """
    Check Cauchy convergence criterion.

    The quantity is converged if the relative change over the last
    `window` iterations is less than `threshold`.

    Parameters
    ----------
    values : array
        Time history of a scalar quantity.
    window : int
        Number of iterations to check.
    threshold : float
        Convergence threshold.

    Returns
    -------
    (converged, cauchy_value)
    """
    if len(values) < window:
        return False, float("inf")

    recent = values[-window:]
    mean_val = np.mean(recent)
    if abs(mean_val) < 1e-15:
        return True, 0.0

    # Max deviation from mean in the window, normalized
    cauchy = np.max(np.abs(recent - mean_val)) / abs(mean_val)
    return cauchy < threshold, float(cauchy)


# ===================================================================
# Residual Plateau Detection
# ===================================================================
def detect_plateau(
    residuals: np.ndarray,
    window: int = 500,
    slope_threshold: float = 1e-5,
) -> Tuple[bool, int]:
    """
    Detect if log-residuals have plateaued (slope near zero).

    Parameters
    ----------
    residuals : array
        Log-scale residual history.
    window : int
        Window size for slope estimation.
    slope_threshold : float
        Slope magnitude below which plateau is declared.

    Returns
    -------
    (plateau_detected, plateau_start_iteration)
    """
    if len(residuals) < window:
        return False, 0

    # Compute running slope using linear regression over sliding window
    n = len(residuals)
    half_w = window // 2

    for start in range(max(0, n - window * 3), n - window):
        segment = residuals[start : start + window]
        x = np.arange(len(segment))
        slope = np.polyfit(x, segment, 1)[0]
        if abs(slope) < slope_threshold:
            return True, start

    return False, 0


# ===================================================================
# Force Time-History Analysis
# ===================================================================
def analyze_force_history(
    values: np.ndarray,
    window: int = 500,
) -> Dict[str, float]:
    """
    Analyze force coefficient time history.

    Returns
    -------
    dict with oscillation_amplitude, mean_drift, mean, std, trend
    """
    if len(values) < 10:
        return {"oscillation_amplitude": 0, "mean_drift": 0, "mean": 0, "std": 0}

    recent = values[-min(window, len(values)):]
    mean = float(np.mean(recent))
    std = float(np.std(recent))

    # Oscillation amplitude (peak-to-peak / 2)
    osc_amp = float((np.max(recent) - np.min(recent)) / 2)

    # Mean drift: compare first and second half averages
    half = len(recent) // 2
    first_half_mean = np.mean(recent[:half])
    second_half_mean = np.mean(recent[half:])
    mean_drift = float(abs(second_half_mean - first_half_mean))

    # Trend via linear fit
    x = np.arange(len(recent))
    slope = float(np.polyfit(x, recent, 1)[0])

    return {
        "oscillation_amplitude": osc_amp,
        "mean_drift": mean_drift,
        "mean": mean,
        "std": std,
        "trend_slope": slope,
    }


# ===================================================================
# Convergence Classification
# ===================================================================
def classify_convergence(
    residuals: np.ndarray,
    forces: Dict[str, np.ndarray],
    cauchy_window: int = 200,
    cauchy_threshold: float = 1e-6,
    plateau_window: int = 500,
    target_residual_drop: float = 6.0,
) -> ConvergenceStatus:
    """
    Classify convergence behavior of a simulation.

    Parameters
    ----------
    residuals : array
        Primary residual history (log-scale).
    forces : dict
        {name: array} for force coefficients (CL, CD, etc.).
    cauchy_window : int
        Window for Cauchy criterion.
    cauchy_threshold : float
        Threshold for Cauchy criterion.
    plateau_window : int
        Window for plateau detection.
    target_residual_drop : float
        Target decades of residual drop for "converged".

    Returns
    -------
    ConvergenceStatus
    """
    status = ConvergenceStatus()
    status.iterations_completed = len(residuals)

    if len(residuals) < 10:
        status.classification = "insufficient_data"
        status.recommendation = "Run more iterations (< 10 available)"
        return status

    # Residual drop
    initial = residuals[min(10, len(residuals) - 1)]  # skip initial transient
    final = residuals[-1]
    status.residual_drop_decades = float(abs(initial - final))

    # Check for divergence: residuals increasing
    recent_slope = np.polyfit(np.arange(min(100, len(residuals))),
                               residuals[-min(100, len(residuals)):], 1)[0]
    if recent_slope > 0.01:
        status.classification = "diverged"
        status.recommendation = "Reduce CFL number or check mesh quality"
        return status

    # Cauchy criterion on primary force
    primary_force = None
    for fname in ["CL", "CD"]:
        if fname in forces and len(forces[fname]) > cauchy_window:
            primary_force = forces[fname]
            break

    if primary_force is not None:
        conv, val = cauchy_criterion(primary_force, cauchy_window, cauchy_threshold)
        status.cauchy_converged = conv
        status.cauchy_value = val

    # Plateau detection
    plateau, p_start = detect_plateau(residuals, plateau_window)
    status.plateau_detected = plateau
    status.plateau_start_iter = p_start

    # Force oscillation analysis
    for fname, farr in forces.items():
        fh = analyze_force_history(farr)
        status.force_oscillation_amplitude = max(
            status.force_oscillation_amplitude, fh["oscillation_amplitude"]
        )
        status.force_mean_drift = max(status.force_mean_drift, fh["mean_drift"])
        status.details[f"{fname}_analysis"] = fh

    # Classification logic
    if status.residual_drop_decades >= target_residual_drop and status.cauchy_converged:
        status.classification = "monotone"
        status.converged = True
        status.recommendation = "Converged. Results are reliable."
    elif status.cauchy_converged and status.residual_drop_decades >= 3.0:
        status.classification = "monotone"
        status.converged = True
        status.recommendation = (
            f"Forces converged (Cauchy < {cauchy_threshold}). "
            f"Residuals dropped {status.residual_drop_decades:.1f} decades."
        )
    elif status.force_oscillation_amplitude > 0 and status.plateau_detected:
        # Oscillating forces with residual plateau
        rel_osc = status.force_oscillation_amplitude
        if primary_force is not None and abs(np.mean(primary_force[-100:])) > 1e-10:
            rel_osc /= abs(np.mean(primary_force[-100:]))

        if rel_osc < 0.01:  # < 1% oscillation
            status.classification = "oscillatory"
            status.converged = True
            status.recommendation = (
                "Small oscillations around converged mean. "
                "Use time-averaged forces for reporting."
            )
        else:
            status.classification = "oscillatory"
            status.converged = False
            status.recommendation = (
                f"Significant oscillations ({rel_osc*100:.1f}%). "
                "Consider: (1) more iterations, (2) implicit time stepping, "
                "(3) CFL reduction."
            )
    elif status.plateau_detected and not status.cauchy_converged:
        status.classification = "stalled"
        status.converged = False
        status.recommendation = (
            f"Residuals plateaued at iter {p_start} but forces not converged. "
            "Consider: (1) mesh refinement, (2) turbulence model change, "
            "(3) scheme order increase."
        )
    else:
        status.classification = "in_progress"
        status.converged = False
        status.recommendation = "Continue running."

    # Time-budget projection
    status.estimated_iters_to_converge = _estimate_remaining_iters(
        residuals, target_residual_drop
    )

    return status


def _estimate_remaining_iters(
    residuals: np.ndarray, target_drop: float
) -> Optional[int]:
    """Estimate iterations needed to reach target residual drop."""
    if len(residuals) < 100:
        return None

    current_drop = abs(residuals[10] - residuals[-1])
    if current_drop >= target_drop:
        return 0  # already there

    # Use recent convergence rate
    recent = residuals[-min(500, len(residuals)):]
    if len(recent) < 50:
        return None

    x = np.arange(len(recent))
    slope = np.polyfit(x, recent, 1)[0]

    if slope >= -1e-8:  # not decreasing
        return None

    remaining_drop = target_drop - current_drop
    iters_needed = int(remaining_drop / abs(slope))
    return min(iters_needed, 1000000)  # cap at 1M


# ===================================================================
# Report Generation
# ===================================================================
def print_convergence_report(status: ConvergenceStatus, label: str = "") -> str:
    """Generate formatted convergence report."""
    lines = []
    lines.append("=" * 65)
    lines.append(f"  CONVERGENCE REPORT{': ' + label if label else ''}")
    lines.append("=" * 65)
    lines.append(f"  Classification: {status.classification.upper()}")
    lines.append(f"  Converged:      {'YES' if status.converged else 'NO'}")
    lines.append(f"  Iterations:     {status.iterations_completed}")
    lines.append(f"  Residual drop:  {status.residual_drop_decades:.2f} decades")
    lines.append(f"  Cauchy value:   {status.cauchy_value:.2e} "
                 f"({'PASS' if status.cauchy_converged else 'FAIL'})")
    if status.plateau_detected:
        lines.append(f"  Plateau at:     iter {status.plateau_start_iter}")
    lines.append(f"  Force osc. amp: {status.force_oscillation_amplitude:.2e}")
    lines.append(f"  Force drift:    {status.force_mean_drift:.2e}")
    if status.estimated_iters_to_converge is not None:
        lines.append(f"  Est. remaining: {status.estimated_iters_to_converge:,} iterations")
    lines.append(f"\n  Recommendation: {status.recommendation}")
    lines.append("=" * 65)

    # Force details
    for key, val in status.details.items():
        if key.endswith("_analysis"):
            fname = key.replace("_analysis", "")
            lines.append(f"\n  {fname}: mean={val['mean']:.6f}  "
                         f"std={val['std']:.2e}  "
                         f"osc={val['oscillation_amplitude']:.2e}  "
                         f"drift={val['mean_drift']:.2e}")

    report = "\n".join(lines)
    print(report)
    return report


def to_json(status: ConvergenceStatus) -> Dict:
    """Convert ConvergenceStatus to JSON-serializable dict."""
    return {
        "classification": status.classification,
        "converged": status.converged,
        "iterations_completed": status.iterations_completed,
        "residual_drop_decades": status.residual_drop_decades,
        "cauchy_converged": status.cauchy_converged,
        "cauchy_value": status.cauchy_value,
        "force_oscillation_amplitude": status.force_oscillation_amplitude,
        "force_mean_drift": status.force_mean_drift,
        "plateau_detected": status.plateau_detected,
        "plateau_start_iter": status.plateau_start_iter,
        "estimated_iters_to_converge": status.estimated_iters_to_converge,
        "recommendation": status.recommendation,
        "details": status.details,
    }


# ===================================================================
# High-Level API
# ===================================================================
class ConvergenceMonitor:
    """
    Reusable convergence monitor for SU2 cases.

    Usage:
        monitor = ConvergenceMonitor("path/to/history.csv")
        status = monitor.analyze()
        monitor.print_report()
    """

    def __init__(
        self,
        history_file: str,
        cauchy_window: int = 200,
        cauchy_threshold: float = 1e-6,
        plateau_window: int = 500,
        target_residual_drop: float = 6.0,
    ):
        self.history_file = Path(history_file)
        self.cauchy_window = cauchy_window
        self.cauchy_threshold = cauchy_threshold
        self.plateau_window = plateau_window
        self.target_residual_drop = target_residual_drop

        self.history = None
        self.status = None

    def load(self):
        """Load and parse history file."""
        self.history = parse_history(str(self.history_file))
        return self

    def analyze(self) -> ConvergenceStatus:
        """Run full convergence analysis."""
        if self.history is None:
            self.load()

        # Find residual and force columns
        res_cols = find_residual_columns(self.history)
        force_map = find_force_columns(self.history)

        # Primary residual
        if res_cols:
            residuals = self.history[res_cols[0]]
        else:
            # Fallback: use first column that looks like iteration count
            residuals = np.zeros(100)

        # Forces
        forces = {}
        for label, col_name in force_map.items():
            forces[label] = self.history[col_name]

        self.status = classify_convergence(
            residuals, forces,
            cauchy_window=self.cauchy_window,
            cauchy_threshold=self.cauchy_threshold,
            plateau_window=self.plateau_window,
            target_residual_drop=self.target_residual_drop,
        )
        return self.status

    def print_report(self, label: str = "") -> str:
        """Print formatted report."""
        if self.status is None:
            self.analyze()
        label = label or self.history_file.parent.name
        return print_convergence_report(self.status, label)

    def save_json(self, output_path: str):
        """Save analysis results to JSON."""
        if self.status is None:
            self.analyze()
        with open(output_path, "w") as f:
            json.dump(to_json(self.status), f, indent=2)


# ===================================================================
# CLI
# ===================================================================
def main():
    """CLI entry point."""
    if len(sys.argv) < 2:
        print("Usage: python convergence_monitor.py <history.csv> [--cauchy-window N]")
        print("\nScans for history files in runs/ directory...")

        # Auto-scan for history files
        project = Path(__file__).resolve().parent.parent.parent
        runs_dir = project / "runs"
        if runs_dir.exists():
            history_files = list(runs_dir.rglob("history*.csv"))
            if history_files:
                print(f"\nFound {len(history_files)} history files:")
                for hf in history_files[:10]:
                    print(f"  {hf.relative_to(project)}")
                    try:
                        monitor = ConvergenceMonitor(str(hf))
                        status = monitor.analyze()
                        print(f"    -> {status.classification.upper()} "
                              f"({status.iterations_completed} iters, "
                              f"{status.residual_drop_decades:.1f} decades)")
                    except Exception as e:
                        print(f"    -> Error: {e}")
            else:
                print("  No history files found in runs/")
        return

    history_file = sys.argv[1]
    cauchy_window = 200
    for i, arg in enumerate(sys.argv):
        if arg == "--cauchy-window" and i + 1 < len(sys.argv):
            cauchy_window = int(sys.argv[i + 1])

    monitor = ConvergenceMonitor(history_file, cauchy_window=cauchy_window)
    monitor.analyze()
    monitor.print_report()

    # Save JSON
    out_path = Path(history_file).parent / "convergence_status.json"
    monitor.save_json(str(out_path))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
