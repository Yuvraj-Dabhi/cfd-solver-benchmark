"""
TMR Reference Cross-Solver Comparison
======================================
Compare SU2 results against published CFL3D/FUN3D values from
NASA Turbulence Modeling Resource (TMR) for verification.

This enables detection of implementation differences between solvers
and validates that SU2 results fall within the accepted code-to-code
scatter band.

Usage:
    from scripts.comparison.tmr_reference_runner import TMRReferenceComparison
    comp = TMRReferenceComparison()
    report = comp.compare("naca_0012_stall", su2_results={"CL": 1.089, "CD": 0.01235})
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _load_config():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from config import TMR_CODE_RESULTS
    return TMR_CODE_RESULTS


@dataclass
class MetricComparison:
    """Single metric comparison between SU2 and a reference solver."""
    metric: str
    su2_value: float
    ref_solver: str
    ref_value: float
    deviation_pct: float
    within_tolerance: bool
    tolerance_pct: float = 2.0


@dataclass
class CaseComparisonReport:
    """Full comparison report for one benchmark case."""
    case_name: str
    su2_results: Dict[str, float]
    comparisons: List[MetricComparison] = field(default_factory=list)
    max_deviation_pct: float = 0.0
    all_within_tolerance: bool = True

    def summary(self) -> str:
        """Human-readable summary of the comparison."""
        lines = [
            f"Cross-Solver Comparison: {self.case_name}",
            "=" * 60,
        ]
        for c in self.comparisons:
            status = "✅" if c.within_tolerance else "⚠️"
            lines.append(
                f"  {status} {c.metric}: SU2={c.su2_value:.6f} vs "
                f"{c.ref_solver}={c.ref_value:.6f} "
                f"(Δ={c.deviation_pct:+.3f}%)"
            )
        lines.append(f"\n  Max deviation: {self.max_deviation_pct:.3f}%")
        lines.append(
            f"  Status: {'ALL WITHIN TOLERANCE' if self.all_within_tolerance else 'DEVIATIONS DETECTED'}"
        )
        return "\n".join(lines)


class TMRReferenceComparison:
    """
    Compare SU2 results against published CFL3D/FUN3D values from
    NASA Turbulence Modeling Resource.

    Parameters
    ----------
    tolerance_pct : float
        Maximum acceptable deviation (%) from reference codes.
        Default 2.0% follows TMR convention for grid-converged values.
    """

    def __init__(self, tolerance_pct: float = 2.0):
        self.tolerance_pct = tolerance_pct
        self.tmr_results = _load_config()

    def available_cases(self) -> List[str]:
        """List all cases with TMR reference data."""
        return list(self.tmr_results.keys())

    def available_solvers(self, case_name: str) -> List[str]:
        """List reference solvers available for a given case."""
        case_data = self.tmr_results.get(case_name, {})
        return list(case_data.keys())

    def compare(
        self,
        case_name: str,
        su2_results: Dict[str, float],
        ref_solver: Optional[str] = None,
    ) -> CaseComparisonReport:
        """
        Compare SU2 results against TMR reference data.

        Parameters
        ----------
        case_name : str
            Benchmark case key (e.g., "flat_plate", "naca_0012_stall").
        su2_results : dict
            SU2 metrics: {"CL": value, "CD": value, ...}.
        ref_solver : str, optional
            Specific reference solver to compare against.
            If None, compares against all available.

        Returns
        -------
        CaseComparisonReport
        """
        if case_name not in self.tmr_results:
            raise ValueError(
                f"No TMR reference data for '{case_name}'. "
                f"Available: {self.available_cases()}"
            )

        case_data = self.tmr_results[case_name]
        report = CaseComparisonReport(
            case_name=case_name,
            su2_results=su2_results,
        )

        solvers_to_check = (
            {ref_solver: case_data[ref_solver]}
            if ref_solver and ref_solver in case_data
            else case_data
        )

        for solver_name, ref_data in solvers_to_check.items():
            # Skip metadata keys
            for metric, su2_val in su2_results.items():
                if metric in ref_data and isinstance(ref_data[metric], (int, float)):
                    ref_val = ref_data[metric]
                    if abs(ref_val) > 1e-12:
                        dev_pct = abs(su2_val - ref_val) / abs(ref_val) * 100.0
                    else:
                        dev_pct = abs(su2_val - ref_val) * 100.0

                    within = dev_pct <= self.tolerance_pct
                    comp = MetricComparison(
                        metric=metric,
                        su2_value=su2_val,
                        ref_solver=solver_name,
                        ref_value=ref_val,
                        deviation_pct=dev_pct,
                        within_tolerance=within,
                        tolerance_pct=self.tolerance_pct,
                    )
                    report.comparisons.append(comp)

                    if dev_pct > report.max_deviation_pct:
                        report.max_deviation_pct = dev_pct
                    if not within:
                        report.all_within_tolerance = False

        return report

    def compare_all_cases(
        self, all_results: Dict[str, Dict[str, float]]
    ) -> Dict[str, CaseComparisonReport]:
        """
        Compare SU2 results for multiple cases.

        Parameters
        ----------
        all_results : dict
            {case_name: {metric: su2_value}}.

        Returns
        -------
        dict of CaseComparisonReport
        """
        reports = {}
        for case_name, su2_results in all_results.items():
            if case_name in self.tmr_results:
                reports[case_name] = self.compare(case_name, su2_results)
            else:
                logger.warning(f"No TMR reference for '{case_name}', skipping")
        return reports

    def generate_comparison_table(
        self, reports: Dict[str, CaseComparisonReport]
    ) -> str:
        """Generate a markdown comparison table from multiple reports."""
        lines = [
            "| Case | Metric | SU2 | Reference | Solver | Δ (%) | Status |",
            "|------|--------|-----|-----------|--------|-------|--------|",
        ]
        for case_name, report in reports.items():
            for c in report.comparisons:
                status = "✅" if c.within_tolerance else "⚠️"
                lines.append(
                    f"| {case_name} | {c.metric} | {c.su2_value:.6f} | "
                    f"{c.ref_value:.6f} | {c.ref_solver} | "
                    f"{c.deviation_pct:.3f} | {status} |"
                )
        return "\n".join(lines)


if __name__ == "__main__":
    # Demo: compare hypothetical SU2 results
    comp = TMRReferenceComparison()

    print("Available TMR cases:", comp.available_cases())

    # Hypothetical NACA 0012 results from SU2
    su2_naca = {"CL": 1.089, "CD": 0.01235}
    report = comp.compare("naca_0012_stall", su2_naca)
    print(report.summary())
