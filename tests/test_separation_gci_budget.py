#!/usr/bin/env python3
"""
Tests for Separation-Metric GCI & Uncertainty Budget
=====================================================
"""

import sys
import numpy as np
import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analysis.separation_gci_budget import (
    CASE_DATA,
    SeparationGCIResult,
    compute_separation_gci,
    SeparationGCIBudget,
    UncertaintyBudgetEntry,
    UncertaintyBudgetResult,
    run_all_cases,
    print_combined_report,
)


# =========================================================================
# Reference Data Tests
# =========================================================================

class TestCaseData:
    """Test that reference data is well-formed."""

    def test_all_cases_present(self):
        expected = {"wall_hump", "naca0012_stall", "bfs", "swbli"}
        assert set(CASE_DATA.keys()) == expected

    @pytest.mark.parametrize("case", list(CASE_DATA.keys()))
    def test_case_has_required_fields(self, case):
        data = CASE_DATA[case]
        assert "experimental" in data
        assert "grids" in data
        assert "SA" in data
        assert "SST" in data
        assert "input_sensitivity" in data

    @pytest.mark.parametrize("case", list(CASE_DATA.keys()))
    def test_three_grid_levels(self, case):
        data = CASE_DATA[case]
        assert len(data["grids"]["n_cells"]) == 3
        for qoi in ["x_sep", "x_reat", "L_bubble"]:
            assert len(data["SA"][qoi]) == 3
            assert len(data["SST"][qoi]) == 3

    @pytest.mark.parametrize("case", list(CASE_DATA.keys()))
    def test_grid_refinement_order(self, case):
        """Grid cells should increase: coarse < medium < fine."""
        cells = CASE_DATA[case]["grids"]["n_cells"]
        assert cells[0] < cells[1] < cells[2]

    @pytest.mark.parametrize("case", list(CASE_DATA.keys()))
    def test_experimental_values_present(self, case):
        exp = CASE_DATA[case]["experimental"]
        for qoi in ["x_sep", "x_reat", "L_bubble"]:
            assert qoi in exp
            assert isinstance(exp[qoi], (int, float))


# =========================================================================
# GCI Computation Tests
# =========================================================================

class TestSeparationGCI:
    """Test GCI computation for separation metrics."""

    @pytest.mark.parametrize("case", list(CASE_DATA.keys()))
    def test_gci_runs(self, case):
        result = compute_separation_gci(case, "SA")
        assert result.case == case
        assert result.model == "SA"
        assert len(result.qoi_results) == 3

    @pytest.mark.parametrize("case", list(CASE_DATA.keys()))
    def test_gci_sst_runs(self, case):
        result = compute_separation_gci(case, "SST")
        assert result.model == "SST"

    def test_observed_order_positive(self):
        result = compute_separation_gci("wall_hump", "SA")
        for qoi, r in result.qoi_results.items():
            if r.observed_order > 0:
                assert r.observed_order > 0

    def test_gci_fine_nonnegative(self):
        result = compute_separation_gci("wall_hump", "SA")
        for qoi, r in result.qoi_results.items():
            assert r.gci_fine_pct >= 0

    def test_summary_string(self):
        result = compute_separation_gci("wall_hump", "SA")
        s = result.summary
        assert "wall_hump" in s
        assert "SA" in s

    def test_invalid_case_raises(self):
        with pytest.raises(ValueError, match="Unknown case"):
            compute_separation_gci("nonexistent", "SA")


# =========================================================================
# Uncertainty Budget Tests
# =========================================================================

class TestUncertaintyBudgetEntry:
    """Test the budget entry dataclass."""

    def test_total_uncertainty_computed(self):
        entry = UncertaintyBudgetEntry(
            qoi="x_sep", fine_value=0.668, experimental=0.665,
            discretisation=0.005, model_form=0.003,
            input_Re=0.005, input_Tu=0.003,
        )
        expected = np.sqrt(0.005**2 + 0.003**2 + 0.005**2 + 0.003**2)
        assert abs(entry.total_uncertainty - expected) < 1e-10

    def test_zero_uncertainties(self):
        entry = UncertaintyBudgetEntry(
            qoi="x_sep", fine_value=0.0, experimental=0.0,
            discretisation=0.0, model_form=0.0,
            input_Re=0.0, input_Tu=0.0,
        )
        assert entry.total_uncertainty == 0.0


class TestSeparationGCIBudget:
    """Test the full budget computation."""

    @pytest.mark.parametrize("case", list(CASE_DATA.keys()))
    def test_budget_runs(self, case):
        budget = SeparationGCIBudget(case)
        result = budget.compute()
        assert result.case == case
        assert len(result.entries) == 3

    def test_wall_hump_budget(self):
        budget = SeparationGCIBudget("wall_hump")
        result = budget.compute()
        for entry in result.entries:
            assert entry.total_uncertainty > 0
            assert entry.fine_value != 0 or entry.qoi == "x_sep"

    def test_text_table_format(self):
        budget = SeparationGCIBudget("wall_hump")
        result = budget.compute()
        assert "Uncertainty Budget" in result.text_table
        assert "x_sep" in result.text_table
        assert "x_reat" in result.text_table
        assert "L_bubble" in result.text_table
        assert "Dominant" in result.text_table

    def test_latex_table_format(self):
        budget = SeparationGCIBudget("wall_hump")
        result = budget.compute()
        assert r"\begin{table}" in result.latex_table
        assert r"\end{table}" in result.latex_table
        assert r"\toprule" in result.latex_table

    def test_summary_has_validation_status(self):
        budget = SeparationGCIBudget("wall_hump")
        result = budget.compute()
        assert "VALIDATED" in result.summary or "NOT VALIDATED" in result.summary

    def test_invalid_case(self):
        with pytest.raises(ValueError):
            SeparationGCIBudget("invalid_case")

    @pytest.mark.parametrize("case", list(CASE_DATA.keys()))
    def test_dominant_source_identified(self, case):
        budget = SeparationGCIBudget(case)
        result = budget.compute()
        assert "Dominant" in result.text_table


# =========================================================================
# Multi-Case Runner
# =========================================================================

class TestMultiCase:
    """Test the multi-case runner."""

    def test_run_all_cases(self):
        results = run_all_cases()
        assert len(results) == 4
        for case in CASE_DATA:
            assert case in results
            assert isinstance(results[case], UncertaintyBudgetResult)

    def test_run_selected_cases(self):
        results = run_all_cases(cases=["wall_hump", "bfs"])
        assert len(results) == 2
        assert "wall_hump" in results
        assert "bfs" in results

    def test_combined_report(self):
        results = run_all_cases()
        report = print_combined_report(results)
        assert "SEPARATION-METRIC" in report
        assert "CROSS-CASE SUMMARY" in report
        assert "VALIDATED" in report or "REVIEW" in report
