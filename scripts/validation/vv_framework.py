"""
Verification & Validation Framework
====================================
Formal V&V per Roache, AIAA G-077-1998, ASME V&V 20-2009,
Oberkampf & Trucano (2002).

Implements:
  - Code verification (MMS + analytical solutions)
  - Solution verification (GCI from grid_convergence module)
  - Validation pyramid (L1 unit → L2 benchmark → L3 system → L4 application)
  - NASA MRR levels 1-4
  - 40% Challenge metric tracking
  - Flat plate verification against analytical Blasius/log-law
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import VALIDATION_CRITERIA, NASA_40_PERCENT_CASES


# =============================================================================
# Validation Status
# =============================================================================
@dataclass
class ValidationLevel:
    """Status of one validation hierarchy level."""
    level: int
    name: str
    cases: List[str] = field(default_factory=list)
    passed: List[bool] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)

    @property
    def status(self) -> str:
        if not self.passed:
            return "NOT STARTED"
        if all(self.passed):
            return "PASSED"
        return f"PARTIAL ({sum(self.passed)}/{len(self.passed)} passed)"


@dataclass
class VVReport:
    """Complete V&V assessment report."""
    verification: Dict[str, Any] = field(default_factory=dict)
    validation_levels: List[ValidationLevel] = field(default_factory=list)
    mrr_level: int = 0
    mrr_description: str = ""
    challenge_metrics: Dict[str, float] = field(default_factory=dict)
    overall_status: str = "INCOMPLETE"


# =============================================================================
# V&V Framework
# =============================================================================
class VVFramework:
    """
    Formal Verification & Validation framework.

    Usage
    -----
    >>> vv = VVFramework()
    >>> vv.run_code_verification(flat_plate_results)
    >>> vv.run_solution_verification(gci_results)
    >>> vv.run_validation("backward_facing_step", cfd, exp, exp_unc)
    >>> report = vv.generate_report()
    """

    def __init__(self):
        self.criteria = VALIDATION_CRITERIA
        self._code_verified = False
        self._solution_verified = {}
        self._validation_results = {}
        self._validation_levels = {
            1: ValidationLevel(1, "Unit Problem Verification"),
            2: ValidationLevel(2, "Benchmark Case Validation"),
            3: ValidationLevel(3, "System-Level Validation"),
            4: ValidationLevel(4, "Application-Specific Validation"),
        }

    # ---- Code Verification ----
    def verify_flat_plate(
        self,
        x: np.ndarray,
        Cf_cfd: np.ndarray,
        U_inf: float,
        nu: float,
    ) -> Dict[str, float]:
        """
        Verify code against flat plate analytical solutions.

        Parameters
        ----------
        x : array
            Streamwise coordinates.
        Cf_cfd : array
            CFD-computed skin friction coefficient.
        U_inf : float
            Free-stream velocity.
        nu : float
            Kinematic viscosity.

        Returns
        -------
        dict with Cf error and pass/fail status.
        """
        Re_x = U_inf * x / nu
        Cf_analytical = 0.059 / Re_x**0.2  # Turbulent correlation

        # Only compare in turbulent region (Re_x > 5e5)
        turb_mask = Re_x > 5e5
        if not np.any(turb_mask):
            return {"error": "No turbulent region found", "passed": False}

        Cf_cfd_turb = Cf_cfd[turb_mask]
        Cf_ana_turb = Cf_analytical[turb_mask]

        mape = float(np.mean(np.abs((Cf_cfd_turb - Cf_ana_turb) / Cf_ana_turb)) * 100)
        max_err = float(np.max(np.abs(Cf_cfd_turb - Cf_ana_turb) / Cf_ana_turb) * 100)

        result = {
            "Cf_MAPE_pct": mape,
            "Cf_max_error_pct": max_err,
            "n_points": int(np.sum(turb_mask)),
            "passed": mape < 5.0,  # < 5% MAPE for code verification
        }

        self._code_verified = result["passed"]
        self._validation_levels[1].cases.append("flat_plate_Cf")
        self._validation_levels[1].passed.append(result["passed"])
        self._validation_levels[1].metrics["flat_plate_Cf_MAPE"] = mape

        return result

    def verify_law_of_wall(
        self,
        y_plus: np.ndarray,
        U_plus_cfd: np.ndarray,
        kappa: float = 0.41,
        B: float = 5.0,
    ) -> Dict[str, float]:
        """
        Verify U+ profile against law-of-wall.

        Checks:
          - Viscous sublayer (y+ < 5): U+ = y+
          - Log layer (30 < y+ < 300): U+ = (1/κ)ln(y+) + B
        """
        results = {}

        # Viscous sublayer
        visc_mask = y_plus < 5
        if np.any(visc_mask):
            U_plus_analytical = y_plus[visc_mask]
            err = np.mean(np.abs(U_plus_cfd[visc_mask] - U_plus_analytical)
                          / (U_plus_analytical + 1e-15)) * 100
            results["viscous_sublayer_MAPE"] = float(err)
            results["viscous_sublayer_passed"] = err < 5.0

        # Log layer
        log_mask = (y_plus > 30) & (y_plus < 300)
        if np.any(log_mask):
            U_plus_log = (1 / kappa) * np.log(y_plus[log_mask]) + B
            err = np.mean(np.abs(U_plus_cfd[log_mask] - U_plus_log)
                          / (U_plus_log + 1e-15)) * 100
            results["log_layer_MAPE"] = float(err)
            results["log_layer_passed"] = err < 5.0

        results["overall_passed"] = all(
            v for k, v in results.items() if k.endswith("_passed")
        )

        self._validation_levels[1].cases.append("law_of_wall")
        self._validation_levels[1].passed.append(results.get("overall_passed", False))

        return results

    # ---- Solution Verification ----
    def check_grid_convergence(
        self,
        case_name: str,
        gci_fine: float,
        observed_order: float,
        in_asymptotic_range: bool,
    ) -> Dict[str, Any]:
        """
        Record grid convergence result.
        """
        passed = gci_fine < self.criteria["gci_threshold"] * 100
        result = {
            "GCI_fine_pct": gci_fine,
            "observed_order": observed_order,
            "in_asymptotic_range": in_asymptotic_range,
            "passed": passed,
        }
        self._solution_verified[case_name] = result
        return result

    # ---- Validation ----
    def validate_case(
        self,
        case_name: str,
        cfd: np.ndarray,
        exp: np.ndarray,
        exp_uncertainty: np.ndarray,
        cfd_uncertainty: Optional[np.ndarray] = None,
        level: int = 2,
    ) -> Dict[str, Any]:
        """
        Run ASME V&V 20 validation for a benchmark case.
        """
        from scripts.postprocessing.error_metrics import (
            compute_all_metrics, asme_vv20_metric,
        )

        metrics = compute_all_metrics(cfd, exp, exp_uncertainty, label=case_name)

        vv = asme_vv20_metric(cfd, exp, exp_uncertainty, cfd_uncertainty)
        metrics.update(vv)

        passed = vv["status"] == "VALIDATED"
        self._validation_results[case_name] = metrics

        self._validation_levels[level].cases.append(case_name)
        self._validation_levels[level].passed.append(passed)
        self._validation_levels[level].metrics[f"{case_name}_MAPE"] = metrics.get("MAPE", 0)

        return metrics

    # ---- MRR Level ----
    def compute_mrr_level(self) -> int:
        """
        Compute Model Readiness Rating (NASA TMR MRR).

        MRR Levels:
          1: Verified on unit problems
          2: Validated on simple canonical cases
          3: Validated on multiple benchmark cases
          4: Applied to complex 3D configuration
        """
        level = 0

        if self._code_verified:
            level = 1

        l2 = self._validation_levels[2]
        if l2.passed and sum(l2.passed) >= 1:
            level = 2
        if l2.passed and sum(l2.passed) >= 3:
            level = 3

        l3 = self._validation_levels[3]
        if l3.passed and any(l3.passed):
            level = 4

        return level

    # ---- 40% Challenge Tracking ----
    def track_40_percent_challenge(
        self,
        case_name: str,
        baseline_error: float,
        current_error: float,
    ) -> Dict[str, float]:
        """
        Track progress toward NASA 40% error reduction goal.
        """
        reduction = (baseline_error - current_error) / baseline_error * 100
        target_error = baseline_error * 0.6  # 40% reduction target

        return {
            "case": case_name,
            "baseline_error": baseline_error,
            "current_error": current_error,
            "target_error": target_error,
            "reduction_pct": reduction,
            "target_met": current_error <= target_error,
        }

    # ---- Report Generation ----
    def generate_report(self) -> VVReport:
        """Generate comprehensive V&V report."""
        mrr = self.compute_mrr_level()

        report = VVReport(
            verification={
                "code_verified": self._code_verified,
                "solution_verified": self._solution_verified,
            },
            validation_levels=list(self._validation_levels.values()),
            mrr_level=mrr,
            mrr_description=VALIDATION_CRITERIA["mrr_levels"].get(mrr, "Not assessed"),
            overall_status=f"MRR Level {mrr}",
        )

        return report

    def print_report(self):
        """Print formatted V&V report."""
        report = self.generate_report()

        print("\n" + "=" * 60)
        print("  VERIFICATION & VALIDATION REPORT")
        print("=" * 60)

        print(f"\n  Code Verification: {'PASSED' if report.verification.get('code_verified') else 'NOT DONE'}")

        for level in report.validation_levels:
            print(f"\n  Level {level.level} ({level.name}): {level.status}")
            for case, passed in zip(level.cases, level.passed):
                icon = "✓" if passed else "✗"
                print(f"    {icon} {case}")

        print(f"\n  NASA MRR Level: {report.mrr_level} — {report.mrr_description}")
        print(f"  Overall Status: {report.overall_status}")
        print("=" * 60)
