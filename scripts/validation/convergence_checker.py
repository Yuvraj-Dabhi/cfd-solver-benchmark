#!/usr/bin/env python3
"""
Iterative Convergence Checker
================================
Monitors residual and force coefficient convergence for SU2 and OpenFOAM
simulations. Implements NASA/AIAA convergence criteria.

Usage
-----
    from scripts.validation.convergence_checker import ConvergenceChecker

    checker = ConvergenceChecker()
    checker.load_residual_history("history.csv")
    checker.load_force_history("history.csv")

    res_ok = checker.check_residual_convergence(target=1e-12)
    force_ok = checker.check_force_convergence(window=500, tolerance=0.001)
    report = checker.generate_convergence_report()
"""

import csv
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class ResidualStatus:
    """Status of residual convergence."""
    converged: bool = False
    final_residual: float = 0.0
    target_residual: float = 1e-12
    orders_of_magnitude: float = 0.0
    n_iterations: int = 0
    residual_history: np.ndarray = field(
        default_factory=lambda: np.array([]), repr=False
    )


@dataclass
class ForceStatus:
    """Status of force coefficient convergence."""
    converged: bool = False
    final_value: float = 0.0
    window_mean: float = 0.0
    window_std: float = 0.0
    relative_change: float = 0.0
    tolerance: float = 0.001
    window_size: int = 500
    force_history: np.ndarray = field(
        default_factory=lambda: np.array([]), repr=False
    )


@dataclass
class MonotoneStatus:
    """Status of monotone convergence check."""
    is_monotonic: bool = False
    is_oscillatory: bool = False
    trend: str = ""  # "decreasing", "increasing", "oscillatory", "flat"
    amplitude_decay_rate: float = 0.0


class ConvergenceChecker:
    """
    Monitors iterative convergence of CFD simulations.

    Supports SU2 history.csv and OpenFOAM postProcessing formats.
    """

    def __init__(self):
        self.residuals: Dict[str, np.ndarray] = {}
        self.forces: Dict[str, np.ndarray] = {}
        self.iterations: np.ndarray = np.array([])
        self._report: Dict[str, Any] = {}

    def load_residual_history(
        self,
        log_path: str,
        format: str = "auto",
    ) -> None:
        """
        Parse residual history from SU2 or OpenFOAM logs.

        Parameters
        ----------
        log_path : str
            Path to history.csv (SU2) or postProcessing/residuals (OpenFOAM).
        format : str
            'su2', 'openfoam', or 'auto' (detect from file).
        """
        log_path = Path(log_path)
        if not log_path.exists():
            raise FileNotFoundError(f"Log file not found: {log_path}")

        if format == "auto":
            format = self._detect_format(log_path)

        if format == "su2":
            self._parse_su2_history(log_path)
        elif format == "openfoam":
            self._parse_openfoam_residuals(log_path)
        else:
            raise ValueError(f"Unknown format: {format}")

    def load_force_history(
        self,
        history_path: str,
        format: str = "auto",
    ) -> None:
        """
        Parse force coefficient history.

        Parameters
        ----------
        history_path : str
            Path to SU2 history.csv or OpenFOAM forceCoeffs.
        format : str
            'su2', 'openfoam', or 'auto'.
        """
        history_path = Path(history_path)
        if not history_path.exists():
            raise FileNotFoundError(f"History file not found: {history_path}")

        if format == "auto":
            format = self._detect_format(history_path)

        if format == "su2":
            self._parse_su2_forces(history_path)
        elif format == "openfoam":
            self._parse_openfoam_forces(history_path)

    def check_residual_convergence(
        self,
        target: float = 1e-12,
        field_name: Optional[str] = None,
    ) -> ResidualStatus:
        """
        Check if residuals have reached the target level.

        Parameters
        ----------
        target : float
            Target residual level (NASA TMR recommends 1e-12).
        field_name : str, optional
            Specific field to check. If None, checks all fields.

        Returns
        -------
        ResidualStatus
        """
        if field_name and field_name in self.residuals:
            res = self.residuals[field_name]
        elif self.residuals:
            # Use the "worst" (largest) residual
            res = max(self.residuals.values(), key=lambda x: x[-1] if len(x) > 0 else 0)
        else:
            return ResidualStatus(converged=False, target_residual=target)

        if len(res) == 0:
            return ResidualStatus(converged=False, target_residual=target)

        final = res[-1]
        initial = res[0] if res[0] != 0 else 1.0
        orders = abs(np.log10(max(final, 1e-20)) - np.log10(max(initial, 1e-20)))

        return ResidualStatus(
            converged=final <= target,
            final_residual=final,
            target_residual=target,
            orders_of_magnitude=orders,
            n_iterations=len(res),
            residual_history=res,
        )

    def check_force_convergence(
        self,
        window: int = 500,
        tolerance: float = 0.001,
        force_name: Optional[str] = None,
    ) -> ForceStatus:
        """
        Check if force coefficients have stabilized.

        Uses the relative change criterion: the standard deviation of the
        last `window` iterations divided by the mean must be < `tolerance`.

        Parameters
        ----------
        window : int
            Number of iterations to assess stability.
        tolerance : float
            Relative change threshold (0.001 = 0.1%).
        force_name : str, optional
            Specific force to check (e.g., 'CL', 'CD'). If None, checks all.

        Returns
        -------
        ForceStatus
        """
        if force_name and force_name in self.forces:
            forces_to_check = {force_name: self.forces[force_name]}
        else:
            forces_to_check = self.forces

        if not forces_to_check:
            return ForceStatus(converged=False, tolerance=tolerance, window_size=window)

        # Check worst-case convergence
        worst_status = ForceStatus(converged=True, tolerance=tolerance, window_size=window)

        for name, history in forces_to_check.items():
            if len(history) < window:
                return ForceStatus(
                    converged=False,
                    final_value=history[-1] if len(history) > 0 else 0,
                    tolerance=tolerance,
                    window_size=window,
                    force_history=history,
                )

            tail = history[-window:]
            mean_val = np.mean(tail)
            std_val = np.std(tail)

            if abs(mean_val) > 1e-15:
                rel_change = std_val / abs(mean_val)
            else:
                rel_change = std_val

            status = ForceStatus(
                converged=rel_change < tolerance,
                final_value=history[-1],
                window_mean=mean_val,
                window_std=std_val,
                relative_change=rel_change,
                tolerance=tolerance,
                window_size=window,
                force_history=history,
            )

            if not status.converged:
                worst_status = status

        return worst_status if not worst_status.converged else status

    def check_monotone_convergence(
        self,
        quantity: np.ndarray,
        n_segments: int = 5,
    ) -> MonotoneStatus:
        """
        Detect monotonic vs oscillatory convergence.

        Splits the history into segments and checks if the segment
        means are monotonically changing.

        Parameters
        ----------
        quantity : ndarray
            History array.
        n_segments : int
            Number of segments to divide history into.

        Returns
        -------
        MonotoneStatus
        """
        if len(quantity) < n_segments * 10:
            return MonotoneStatus(trend="insufficient_data")

        segment_size = len(quantity) // n_segments
        means = [
            np.mean(quantity[i * segment_size : (i + 1) * segment_size])
            for i in range(n_segments)
        ]

        diffs = np.diff(means)

        if np.all(diffs < 0):
            trend = "decreasing"
            is_monotonic = True
        elif np.all(diffs > 0):
            trend = "increasing"
            is_monotonic = True
        elif np.std(diffs) < 1e-10:
            trend = "flat"
            is_monotonic = True
        else:
            trend = "oscillatory"
            is_monotonic = False

        is_oscillatory = not is_monotonic and trend == "oscillatory"

        # Compute amplitude decay for oscillatory signals
        decay_rate = 0.0
        if is_oscillatory and len(quantity) > 100:
            first_half = quantity[: len(quantity) // 2]
            second_half = quantity[len(quantity) // 2 :]
            amp1 = np.std(first_half)
            amp2 = np.std(second_half)
            if amp1 > 1e-15:
                decay_rate = (amp1 - amp2) / amp1

        return MonotoneStatus(
            is_monotonic=is_monotonic,
            is_oscillatory=is_oscillatory,
            trend=trend,
            amplitude_decay_rate=decay_rate,
        )

    def generate_convergence_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive convergence assessment.

        Returns
        -------
        dict with residual, force, and overall convergence status.
        """
        from config import CONVERGENCE_CRITERIA

        target_res = CONVERGENCE_CRITERIA.get("residual_target", 1e-12)
        window = CONVERGENCE_CRITERIA.get("force_stability_window", 500)
        tol = CONVERGENCE_CRITERIA.get("force_stability_tolerance", 0.001)

        report = {
            "criteria": {
                "residual_target": target_res,
                "force_window": window,
                "force_tolerance": tol,
            },
            "residuals": {},
            "forces": {},
            "overall_converged": True,
        }

        # Check each residual field
        for field_name in self.residuals:
            status = self.check_residual_convergence(target_res, field_name)
            report["residuals"][field_name] = {
                "converged": status.converged,
                "final_residual": float(status.final_residual),
                "orders_of_magnitude": float(status.orders_of_magnitude),
                "n_iterations": status.n_iterations,
            }
            if not status.converged:
                report["overall_converged"] = False

        # Check each force
        for force_name in self.forces:
            status = self.check_force_convergence(window, tol, force_name)
            mono = self.check_monotone_convergence(self.forces[force_name])
            report["forces"][force_name] = {
                "converged": status.converged,
                "final_value": float(status.final_value),
                "window_mean": float(status.window_mean),
                "relative_change": float(status.relative_change),
                "trend": mono.trend,
            }
            if not status.converged:
                report["overall_converged"] = False

        self._report = report
        return report

    def save_report(self, path: str) -> None:
        """Save convergence report to JSON."""
        if not self._report:
            self.generate_convergence_report()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self._report, f, indent=2)

    # ---- Internal parsers ----

    def _detect_format(self, path: Path) -> str:
        """Auto-detect file format."""
        content = path.read_text(errors="ignore")[:500]
        if "Inner_Iter" in content or '"Iteration"' in content or "rms[" in content.lower():
            return "su2"
        elif "Time" in content and "Ux" in content:
            return "openfoam"
        elif path.suffix == ".csv":
            return "su2"
        return "su2"  # Default

    def _parse_su2_history(self, path: Path) -> None:
        """Parse SU2 history.csv for residuals."""
        try:
            lines = path.read_text(errors="ignore").strip().split("\n")
            if not lines:
                return

            # Parse header — SU2 uses quoted column names with spaces
            header_line = lines[0]
            # Remove surrounding quotes and whitespace
            headers = [h.strip().strip('"').strip("'") for h in header_line.split(",")]

            # Find residual columns (rms_*)
            rms_cols = {}
            for i, h in enumerate(headers):
                h_lower = h.lower()
                if "rms" in h_lower or "res" in h_lower:
                    clean_name = re.sub(r'["\[\]\s]', '', h)
                    rms_cols[clean_name] = i

            # Parse data
            data_lines = [l for l in lines[1:] if l.strip() and not l.startswith("#")]
            if not data_lines:
                return

            for col_name, col_idx in rms_cols.items():
                values = []
                for line in data_lines:
                    parts = line.split(",")
                    if col_idx < len(parts):
                        try:
                            val = float(parts[col_idx].strip().strip('"'))
                            values.append(10 ** val if val < 0 else val)
                        except (ValueError, IndexError):
                            continue
                if values:
                    self.residuals[col_name] = np.array(values)

        except Exception as e:
            print(f"Warning: Could not parse SU2 residuals from {path}: {e}")

    def _parse_su2_forces(self, path: Path) -> None:
        """Parse SU2 history.csv for force coefficients."""
        try:
            lines = path.read_text(errors="ignore").strip().split("\n")
            if not lines:
                return

            headers = [h.strip().strip('"').strip("'") for h in lines[0].split(",")]

            # Find force columns (CL, CD, CMz, etc.)
            force_patterns = ["CL", "CD", "CMz", "CM", "CDrag", "CLift"]
            force_cols = {}
            for i, h in enumerate(headers):
                h_clean = re.sub(r'["\[\]\s]', '', h)
                for pat in force_patterns:
                    if pat.lower() == h_clean.lower() or h_clean.lower().endswith(pat.lower()):
                        force_cols[pat] = i
                        break

            data_lines = [l for l in lines[1:] if l.strip() and not l.startswith("#")]

            for col_name, col_idx in force_cols.items():
                values = []
                for line in data_lines:
                    parts = line.split(",")
                    if col_idx < len(parts):
                        try:
                            values.append(float(parts[col_idx].strip().strip('"')))
                        except (ValueError, IndexError):
                            continue
                if values:
                    self.forces[col_name] = np.array(values)

        except Exception as e:
            print(f"Warning: Could not parse SU2 forces from {path}: {e}")

    def _parse_openfoam_residuals(self, path: Path) -> None:
        """Parse OpenFOAM residual log."""
        try:
            content = path.read_text(errors="ignore")
            # Match patterns like "Solving for Ux, Initial residual = 1.234e-05"
            pattern = r"Solving for (\w+),\s*Initial residual\s*=\s*([\d.eE+-]+)"
            matches = re.findall(pattern, content)

            residual_dict: Dict[str, List[float]] = {}
            for field_name, value in matches:
                if field_name not in residual_dict:
                    residual_dict[field_name] = []
                try:
                    residual_dict[field_name].append(float(value))
                except ValueError:
                    continue

            for field_name, values in residual_dict.items():
                self.residuals[field_name] = np.array(values)

        except Exception as e:
            print(f"Warning: Could not parse OpenFOAM residuals: {e}")

    def _parse_openfoam_forces(self, path: Path) -> None:
        """Parse OpenFOAM forceCoeffs postProcessing."""
        try:
            lines = path.read_text(errors="ignore").strip().split("\n")
            data_lines = [l for l in lines if l.strip() and not l.startswith("#")]

            if not data_lines:
                return

            # OpenFOAM forceCoeffs: Time Cd Cs Cl CmRoll CmPitch CmYaw ...
            cl_vals, cd_vals = [], []
            for line in data_lines:
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        cd_vals.append(float(parts[1]))
                        cl_vals.append(float(parts[3]))
                    except (ValueError, IndexError):
                        continue

            if cl_vals:
                self.forces["CL"] = np.array(cl_vals)
            if cd_vals:
                self.forces["CD"] = np.array(cd_vals)

        except Exception as e:
            print(f"Warning: Could not parse OpenFOAM forces: {e}")


if __name__ == "__main__":
    # Demo with synthetic data
    print("=== Convergence Checker Demo ===\n")

    checker = ConvergenceChecker()

    # Simulate residual history (5 orders of magnitude drop)
    n_iter = 10000
    iters = np.arange(n_iter)
    res_p = 10 ** (-1.0 - 4.0 * iters / n_iter + 0.1 * np.random.randn(n_iter))
    checker.residuals["rms_Pressure"] = res_p
    checker.residuals["rms_Velocity"] = res_p * 0.1

    # Simulate force history (converging)
    cl = 1.09 + 0.01 * np.exp(-iters / 2000) * np.sin(0.01 * iters)
    cd = 0.0122 + 0.001 * np.exp(-iters / 2000)
    checker.forces["CL"] = cl
    checker.forces["CD"] = cd

    # Check convergence
    res_status = checker.check_residual_convergence(target=1e-5)
    print(f"Residual converged: {res_status.converged}")
    print(f"  Final: {res_status.final_residual:.2e}")
    print(f"  Orders dropped: {res_status.orders_of_magnitude:.1f}")

    force_status = checker.check_force_convergence(window=500, tolerance=0.001)
    print(f"\nForce converged: {force_status.converged}")
    print(f"  CL final: {force_status.final_value:.6f}")
    print(f"  Relative change: {force_status.relative_change:.6f}")

    report = checker.generate_convergence_report()
    print(f"\nOverall converged: {report['overall_converged']}")
