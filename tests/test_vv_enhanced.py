#!/usr/bin/env python3
"""
Tests for V&V Rigor Infrastructure (Component 2)
====================================================
Tests GCI harness, convergence checker, and new config entries.
"""

import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestGCIStudy:
    """Tests for the reusable GCI harness."""

    def test_gci_monotonic_convergence(self):
        """GCI with known 2nd-order monotonic convergence data."""
        from scripts.validation.gci_harness import GCIStudy

        # f(h) = f_exact + C*h^2, with f_exact=1.0, C=1.0, r=2
        # fine: f(1) = 2.0, medium: f(2) = 5.0, coarse: f(4) = 17.0
        study = GCIStudy(r21=2.0, r32=2.0)
        study.add_quantity("test", f_coarse=17.0, f_medium=5.0, f_fine=2.0)
        results = study.compute()

        assert "test" in results
        r = results["test"]
        assert r.convergence_type == "monotonic"
        assert r.observed_order > 1.5  # Should be ~2.0
        assert r.gci_fine_pct < 100  # Should have a finite GCI (large for extreme test ratios)

    def test_gci_detects_divergent(self):
        """GCI should detect divergent grid refinement."""
        from scripts.validation.gci_harness import GCIStudy

        # Values getting worse with refinement
        study = GCIStudy(r21=2.0, r32=2.0)
        study.add_quantity("div_test", f_coarse=1.0, f_medium=1.5, f_fine=2.5)
        results = study.compute()
        # eps_32 = 0.5, eps_21 = 1.0 → ratio = 0.5 < 1 → divergent
        assert results["div_test"].convergence_type == "divergent"

    def test_gci_identical_values(self):
        """GCI should handle identical fine/medium values gracefully."""
        from scripts.validation.gci_harness import GCIStudy

        study = GCIStudy(r21=2.0, r32=2.0)
        study.add_quantity("converged", f_coarse=1.01, f_medium=1.0, f_fine=1.0)
        results = study.compute()
        assert results["converged"].gci_fine_pct == 0.0

    def test_gci_summary_table(self):
        """Summary table should include quantity names and column headers."""
        from scripts.validation.gci_harness import GCIStudy

        study = GCIStudy(r21=1.5, r32=1.5)
        study.add_quantity("x_sep", f_coarse=0.70, f_medium=0.68, f_fine=0.67)
        study.compute()
        table = study.summary_table()

        assert "x_sep" in table
        assert "GCI_fine%" in table
        assert "p_obs" in table

    def test_gci_to_json(self):
        """GCI results should serialize/deserialize via JSON."""
        from scripts.validation.gci_harness import GCIStudy
        import tempfile

        study = GCIStudy(r21=2.0, r32=2.0)
        study.add_quantity("qty1", f_coarse=3.0, f_medium=2.0, f_fine=1.5)
        study.compute()

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "gci_test.json"
            study.to_json(json_path)
            data = json.loads(json_path.read_text())
            assert "quantities" in data
            assert "qty1" in data["quantities"]
            assert "observed_order" in data["quantities"]["qty1"]

    def test_gci_from_cell_counts(self):
        """compute_from_cell_counts should give correct refinement ratios."""
        from scripts.validation.gci_harness import compute_from_cell_counts

        r21, r32 = compute_from_cell_counts(1000, 4000, 16000, ndim=2)
        assert abs(r21 - 2.0) < 0.01
        assert abs(r32 - 2.0) < 0.01

    def test_profile_gci(self):
        """Profile GCI should compute L2-based metrics."""
        from scripts.validation.gci_harness import compute_profile_gci

        x = np.linspace(0, 1, 100)
        f_fine = np.sin(x)
        f_medium = np.sin(x) + 0.01
        f_coarse = np.sin(x) + 0.04

        result = compute_profile_gci(f_coarse, f_medium, f_fine)
        assert result["gci_fine_pct"] > 0
        assert result["observed_order"] > 0


class TestConvergenceChecker:
    """Tests for the iterative convergence checker."""

    def _make_checker_with_synthetic_data(self):
        """Create a checker with synthetic convergence data."""
        from scripts.validation.convergence_checker import ConvergenceChecker

        checker = ConvergenceChecker()
        n = 5000
        iters = np.arange(n)

        # Residuals dropping 5 orders
        checker.residuals["rms_Pressure"] = 10 ** (-1.0 - 4.0 * iters / n)
        checker.residuals["rms_Velocity"] = 10 ** (-1.5 - 4.5 * iters / n)

        # Forces converging
        checker.forces["CL"] = 1.09 + 0.01 * np.exp(-iters / 500)
        checker.forces["CD"] = 0.012 + 0.001 * np.exp(-iters / 500)

        return checker

    def test_residual_convergence_target_met(self):
        """Should detect when residual target is met."""
        checker = self._make_checker_with_synthetic_data()
        status = checker.check_residual_convergence(target=1e-4)
        assert status.converged
        assert status.final_residual < 1e-4

    def test_residual_convergence_target_not_met(self):
        """Should detect when residual target is NOT met."""
        checker = self._make_checker_with_synthetic_data()
        status = checker.check_residual_convergence(target=1e-12)
        assert not status.converged
        assert status.orders_of_magnitude > 3

    def test_force_convergence_with_stabilized_data(self):
        """Should detect force convergence with stable data."""
        checker = self._make_checker_with_synthetic_data()
        status = checker.check_force_convergence(window=500, tolerance=0.01)
        assert status.converged

    def test_force_convergence_insufficient_data(self):
        """Should not converge if window > data length."""
        from scripts.validation.convergence_checker import ConvergenceChecker
        checker = ConvergenceChecker()
        checker.forces["CL"] = np.array([1.0, 1.1, 1.05])
        status = checker.check_force_convergence(window=500)
        assert not status.converged

    def test_monotone_convergence_decreasing(self):
        """Should detect monotonically decreasing trend."""
        checker = self._make_checker_with_synthetic_data()
        res = checker.residuals["rms_Pressure"]
        mono = checker.check_monotone_convergence(res)
        assert mono.is_monotonic
        assert mono.trend == "decreasing"

    def test_monotone_convergence_oscillatory(self):
        """Should detect oscillatory convergence."""
        from scripts.validation.convergence_checker import ConvergenceChecker
        checker = ConvergenceChecker()
        n = 2000
        x = np.arange(n, dtype=float)
        osc = np.sin(0.1 * x) * np.exp(-x / 500) + 1.0
        mono = checker.check_monotone_convergence(osc)
        assert mono.is_oscillatory or mono.trend == "oscillatory"

    def test_convergence_report_structure(self):
        """Report should contain residuals, forces, and overall status."""
        checker = self._make_checker_with_synthetic_data()
        report = checker.generate_convergence_report()
        assert "residuals" in report
        assert "forces" in report
        assert "overall_converged" in report
        assert "criteria" in report


class TestConvergenceCriteriaConfig:
    """Tests for new config entries."""

    def test_convergence_criteria_exists(self):
        """CONVERGENCE_CRITERIA should exist in config."""
        from config import CONVERGENCE_CRITERIA
        assert isinstance(CONVERGENCE_CRITERIA, dict)

    def test_convergence_criteria_keys(self):
        """CONVERGENCE_CRITERIA should have all required keys."""
        from config import CONVERGENCE_CRITERIA
        required = [
            "residual_target", "force_stability_window",
            "force_stability_tolerance", "gci_threshold",
        ]
        for key in required:
            assert key in CONVERGENCE_CRITERIA

    def test_strict_solver_defaults(self):
        """SOLVER_DEFAULTS should have strict convergence entries."""
        from config import SOLVER_DEFAULTS
        assert "strict_convergence_residual" in SOLVER_DEFAULTS
        assert SOLVER_DEFAULTS["strict_convergence_residual"] == 1e-12
        assert "strict_max_iterations" in SOLVER_DEFAULTS
        assert SOLVER_DEFAULTS["strict_max_iterations"] == 50000
