"""
Tests for Numerical Schemes and TMR Cross-Solver Comparison
============================================================
Validates:
- SU2_NUMERICAL_DEFAULTS enforce second-order everywhere
- TMR_CODE_RESULTS registry is populated for core cases
- TMRReferenceComparison produces correct deviation reports

Run: pytest tests/test_numerical_schemes.py -v
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestSU2NumericalDefaults:
    """Verify that SU2 numerical defaults enforce second-order schemes."""

    def test_import(self):
        from config import SU2_NUMERICAL_DEFAULTS
        assert isinstance(SU2_NUMERICAL_DEFAULTS, dict)

    def test_second_order_spatial(self):
        from config import SU2_NUMERICAL_DEFAULTS
        assert SU2_NUMERICAL_DEFAULTS["spatial_order"] == 2

    def test_muscl_flow_enabled(self):
        from config import SU2_NUMERICAL_DEFAULTS
        assert SU2_NUMERICAL_DEFAULTS["muscl_flow"] is True

    def test_muscl_turb_enabled(self):
        from config import SU2_NUMERICAL_DEFAULTS
        assert SU2_NUMERICAL_DEFAULTS["muscl_turb"] is True, (
            "NASA TMR: first-order turbulence is inadequate for verification"
        )

    def test_convergence_stricter_than_minus_8(self):
        from config import SU2_NUMERICAL_DEFAULTS
        assert SU2_NUMERICAL_DEFAULTS["conv_residual_minval"] <= -10

    def test_has_rationale_notes(self):
        from config import SU2_NUMERICAL_DEFAULTS
        assert "notes" in SU2_NUMERICAL_DEFAULTS
        assert "second_order_rationale" in SU2_NUMERICAL_DEFAULTS["notes"]
        assert "limiter_choice" in SU2_NUMERICAL_DEFAULTS["notes"]
        assert "line_implicit_reference" in SU2_NUMERICAL_DEFAULTS["notes"]


class TestTMRCodeResults:
    """Verify TMR reference data registry for cross-solver comparison."""

    def test_import(self):
        from config import TMR_CODE_RESULTS
        assert isinstance(TMR_CODE_RESULTS, dict)

    def test_core_cases_present(self):
        from config import TMR_CODE_RESULTS
        for case in ["flat_plate", "naca_0012_stall", "nasa_hump", "backward_facing_step"]:
            assert case in TMR_CODE_RESULTS, f"Missing TMR reference for {case}"

    def test_flat_plate_cfl3d(self):
        from config import TMR_CODE_RESULTS
        fp = TMR_CODE_RESULTS["flat_plate"]
        assert "CFL3D" in fp
        assert fp["CFL3D"]["Cf_x5"] > 0

    def test_naca_0012_has_fun3d(self):
        from config import TMR_CODE_RESULTS
        naca = TMR_CODE_RESULTS["naca_0012_stall"]
        assert "FUN3D_SA_alpha10" in naca
        assert naca["FUN3D_SA_alpha10"]["CL"] > 1.0

    def test_all_entries_have_source(self):
        from config import TMR_CODE_RESULTS
        for case_name, solvers in TMR_CODE_RESULTS.items():
            for solver_name, data in solvers.items():
                assert "source" in data, f"{case_name}/{solver_name} missing source URL"


class TestTMRReferenceComparison:
    """Tests for the TMR cross-solver comparison module."""

    def test_import(self):
        from scripts.comparison.tmr_reference_runner import TMRReferenceComparison
        comp = TMRReferenceComparison()
        assert len(comp.available_cases()) >= 4

    def test_compare_within_tolerance(self):
        from scripts.comparison.tmr_reference_runner import TMRReferenceComparison
        comp = TMRReferenceComparison(tolerance_pct=2.0)
        # Values very close to CFL3D
        report = comp.compare("flat_plate", {"Cf_x5": 0.002960})
        assert len(report.comparisons) > 0
        assert report.all_within_tolerance

    def test_compare_outside_tolerance(self):
        from scripts.comparison.tmr_reference_runner import TMRReferenceComparison
        comp = TMRReferenceComparison(tolerance_pct=0.1)
        # Value deliberately far from reference
        report = comp.compare("flat_plate", {"Cf_x5": 0.003500})
        assert not report.all_within_tolerance

    def test_compare_invalid_case(self):
        from scripts.comparison.tmr_reference_runner import TMRReferenceComparison
        comp = TMRReferenceComparison()
        with pytest.raises(ValueError, match="No TMR reference data"):
            comp.compare("nonexistent_case", {"CL": 1.0})

    def test_summary_generation(self):
        from scripts.comparison.tmr_reference_runner import TMRReferenceComparison
        comp = TMRReferenceComparison()
        report = comp.compare("flat_plate", {"Cf_x5": 0.002960})
        summary = report.summary()
        assert "Cross-Solver Comparison" in summary
        assert "flat_plate" in summary

    def test_comparison_table(self):
        from scripts.comparison.tmr_reference_runner import TMRReferenceComparison
        comp = TMRReferenceComparison()
        report = comp.compare("flat_plate", {"Cf_x5": 0.002960})
        table = comp.generate_comparison_table({"flat_plate": report})
        assert "| flat_plate |" in table
        assert "Cf_x5" in table
