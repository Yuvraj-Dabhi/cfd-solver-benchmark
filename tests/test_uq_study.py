#!/usr/bin/env python3
"""
Tests for UQ & Sensitivity Analysis Pipeline
================================================
Tests multi-case GCI orchestration, input-parameter UQ sweeps,
and the combined UQ summary report generator.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =========================================================================
# Multi-Case GCI Tests
# =========================================================================
class TestMultiCaseGCI:
    """Tests for run_gci_all_cases module."""

    def test_eligible_cases_not_empty(self):
        """At least 5 cases should be eligible for GCI."""
        from scripts.analysis.run_gci_all_cases import get_eligible_cases
        eligible = get_eligible_cases()
        assert len(eligible) >= 5

    def test_gci_for_single_case(self):
        """GCI should compute for nasa_hump."""
        from scripts.analysis.run_gci_all_cases import run_gci_for_case
        result = run_gci_for_case("nasa_hump")
        assert result.case_name == "nasa_hump"
        assert len(result.quantities) > 0
        for qty_name, r in result.quantities.items():
            assert r.observed_order > 0
            assert r.gci_fine_pct >= 0

    def test_gci_for_bfs(self):
        """GCI should compute for backward_facing_step."""
        from scripts.analysis.run_gci_all_cases import run_gci_for_case
        result = run_gci_for_case("backward_facing_step")
        assert "x_reat_xH" in result.quantities

    def test_run_all_cases(self):
        """run_all_cases should return results for all eligible cases."""
        from scripts.analysis.run_gci_all_cases import run_all_cases, get_eligible_cases
        eligible = get_eligible_cases()
        results = run_all_cases()
        assert len(results) == len(eligible)

    def test_master_summary_table(self):
        """Summary table should be a non-empty string."""
        from scripts.analysis.run_gci_all_cases import run_all_cases, master_summary_table
        results = run_all_cases()
        table = master_summary_table(results)
        assert "GCI" in table
        assert len(table) > 100

    def test_latex_export(self):
        """LaTeX export should produce valid table markup."""
        from scripts.analysis.run_gci_all_cases import run_all_cases, to_latex
        results = run_all_cases()
        latex = to_latex(results)
        assert r"\begin{table}" in latex
        assert r"\end{table}" in latex

    def test_gci_naca0012(self):
        """NACA 0012 should have GCI for CL and CD."""
        from scripts.analysis.run_gci_all_cases import run_gci_for_case
        result = run_gci_for_case("naca_0012_stall")
        assert "CL_alpha10" in result.quantities
        assert "CD_alpha10" in result.quantities


# =========================================================================
# Input-Parameter UQ Tests
# =========================================================================
class TestInputUQStudy:
    """Tests for run_input_uq_study module."""

    def test_hump_uq_sweep(self):
        """NASA hump UQ sweep should return valid sensitivities."""
        from scripts.analysis.run_input_uq_study import run_uq_sweep
        result = run_uq_sweep("nasa_hump")
        assert len(result.sensitivities) > 0
        assert result.dominant_parameter != ""
        assert result.max_output_variation_pct > 0

    def test_bfs_uq_sweep(self):
        """BFS UQ sweep should return results."""
        from scripts.analysis.run_input_uq_study import run_uq_sweep
        result = run_uq_sweep("backward_facing_step")
        assert any(s.output_name == "x_reat_xH" for s in result.sensitivities)

    def test_naca0012_uq_sweep(self):
        """NACA 0012 should have CL/CD sensitivities."""
        from scripts.analysis.run_input_uq_study import run_uq_sweep
        result = run_uq_sweep("naca_0012_stall")
        outputs = {s.output_name for s in result.sensitivities}
        assert "CL" in outputs
        assert "CD" in outputs

    def test_jet_uq_sweep(self):
        """Jet UQ should include core_length and spreading_rate."""
        from scripts.analysis.run_input_uq_study import run_uq_sweep
        result = run_uq_sweep("axisymmetric_jet")
        outputs = {s.output_name for s in result.sensitivities}
        assert "core_length_xD" in outputs
        assert "spreading_rate" in outputs

    def test_all_uq_cases(self):
        """run_all_uq should produce results for all 4 cases."""
        from scripts.analysis.run_input_uq_study import run_all_uq
        results = run_all_uq()
        assert len(results) == 4

    def test_sensitivity_coefficients_finite(self):
        """All sensitivity coefficients should be finite."""
        from scripts.analysis.run_input_uq_study import run_all_uq
        results = run_all_uq()
        for case_name, cr in results.items():
            for s in cr.sensitivities:
                assert np.isfinite(s.sensitivity_coeff), \
                    f"Non-finite sensitivity for {case_name}/{s.output_name}/{s.parameter_name}"

    def test_perturbation_symmetric(self):
        """Low and high outputs should bracket the baseline."""
        from scripts.analysis.run_input_uq_study import run_uq_sweep
        result = run_uq_sweep("nasa_hump")
        for s in result.sensitivities:
            # Low and high should generally bracket baseline
            # (not always exactly due to nonlinearity, but close)
            mid = (s.low_output + s.high_output) / 2
            assert abs(mid - s.baseline_output) / max(abs(s.baseline_output), 1e-10) < 0.5

    def test_sensitivity_summary_table(self):
        """Summary table should reference NASA 40% Challenge."""
        from scripts.analysis.run_input_uq_study import run_all_uq, sensitivity_summary_table
        results = run_all_uq()
        table = sensitivity_summary_table(results)
        assert "40%" in table
        assert "Slotnick" in table


# =========================================================================
# UQ Summary Report Tests
# =========================================================================
class TestUQSummaryReport:
    """Tests for the combined UQ summary report."""

    def test_build_error_budget_standalone(self):
        """Error budget should build from model uncertainty alone."""
        from scripts.analysis.uq_summary_report import build_error_budget
        summary = build_error_budget()
        assert summary.n_cases > 0
        assert summary.n_quantities > 0

    def test_build_error_budget_with_gci(self):
        """Error budget with GCI should have numerical uncertainty."""
        from scripts.analysis.run_gci_all_cases import run_all_cases
        from scripts.analysis.uq_summary_report import build_error_budget
        gci = run_all_cases()
        summary = build_error_budget(gci_results=gci)
        gci_entries = [e for e in summary.entries if e.gci_fine_pct > 0]
        assert len(gci_entries) > 0

    def test_build_error_budget_with_uq(self):
        """Error budget with UQ should have input uncertainty."""
        from scripts.analysis.run_input_uq_study import run_all_uq
        from scripts.analysis.uq_summary_report import build_error_budget
        uq = run_all_uq()
        summary = build_error_budget(uq_results=uq)
        input_entries = [e for e in summary.entries if e.input_max_variation_pct > 0]
        assert len(input_entries) > 0

    def test_rss_combination(self):
        """Total uncertainty should be RSS of components."""
        from scripts.analysis.uq_summary_report import build_error_budget
        summary = build_error_budget()
        for e in summary.entries:
            expected_rss = np.sqrt(
                e.gci_fine_pct**2 + e.model_spread_pct**2 +
                e.input_max_variation_pct**2 + (e.ml_epistemic_pct or 0.0)**2 +
                (e.ai_model_pct or 0.0)**2
            )
            assert abs(e.total_uncertainty_pct - expected_rss) < 0.01

    def test_text_report_generation(self):
        """Text report should be generated."""
        from scripts.analysis.uq_summary_report import (
            build_error_budget, generate_text_report,
        )
        summary = build_error_budget()
        report = generate_text_report(summary)
        assert "ASME V&V" in report
        assert "Vision 2030" in report
        assert len(report) > 200

    def test_latex_table_generation(self):
        """LaTeX table should produce valid markup."""
        from scripts.analysis.uq_summary_report import (
            build_error_budget, generate_latex_table,
        )
        summary = build_error_budget()
        latex = generate_latex_table(summary)
        assert r"\begin{table}" in latex
        assert r"U_\text{total}" in latex

    def test_worst_case_identified(self):
        """Worst case should be identified in summary."""
        from scripts.analysis.uq_summary_report import build_error_budget
        summary = build_error_budget()
        assert summary.worst_case != ""
        assert summary.worst_uncertainty_pct > 0
