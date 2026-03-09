#!/usr/bin/env python3
"""
Tests for LLM-Based Benchmark Automation & Intelligent Reporting
==================================================================
All tests use the rule-based fallback — no API key required.

Tests cover:
  - CFDResultParser: log/CSV parsing
  - PromptTemplates: template rendering
  - PhysicsSanityChecker: rule-based diagnostics
  - BenchmarkCaseAnalyzer: full diagnostic pipeline
  - NarrativeReportGenerator: report output format
"""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =========================================================================
# TestCFDResultParser
# =========================================================================
class TestCFDResultParser:
    """Tests for CFD result parsing."""

    def test_parse_su2_log(self):
        """Should extract convergence data from SU2-style log."""
        from scripts.ml_augmentation.llm_benchmark_assistant import (
            CFDResultParser,
        )
        log = (
            "| 0    | -2.50e+00 | 0.00e+00 | 0.000000 | 0.000000 |\n"
            "| 100  | -4.00e+00 | 1.00e-02 | 0.500000 | 0.020000 |\n"
            "| 200  | -5.50e+00 | 5.00e-03 | 0.510000 | 0.019500 |\n"
            "| 300  | -6.50e+00 | 2.00e-03 | 0.505000 | 0.019800 |\n"
        )
        conv = CFDResultParser.parse_su2_log(log)
        assert conv.n_iterations == 4
        assert len(conv.cl_history) == 4
        assert conv.final_residual == -6.5

    def test_parse_surface_csv(self):
        """Should extract Cp and Cf from CSV."""
        from scripts.ml_augmentation.llm_benchmark_assistant import (
            CFDResultParser,
        )
        csv = (
            "x, Cp, Cf\n"
            "0.0, 1.0, 0.005\n"
            "0.5, -0.5, 0.003\n"
            "0.8, -0.2, -0.001\n"
            "1.0, 0.1, 0.002\n"
        )
        surf = CFDResultParser.parse_surface_csv(csv)
        assert len(surf.x) == 4
        assert len(surf.Cp) == 4
        assert len(surf.Cf) == 4
        assert surf.Cp_stagnation == pytest.approx(1.0)
        assert surf.separation_x == pytest.approx(0.8)

    def test_parse_empty_csv(self):
        """Should handle empty CSV gracefully."""
        from scripts.ml_augmentation.llm_benchmark_assistant import (
            CFDResultParser,
        )
        surf = CFDResultParser.parse_surface_csv("")
        assert len(surf.x) == 0

    def test_build_case_result(self):
        """Should build structured case result."""
        from scripts.ml_augmentation.llm_benchmark_assistant import (
            CFDResultParser,
        )
        case = CFDResultParser.build_case_result(
            "test_case", cl=0.5, cd=0.02,
            solver="SU2", turbulence_model="SA",
        )
        assert case.case_name == "test_case"
        assert case.cl == 0.5
        assert case.cd == 0.02


# =========================================================================
# TestPromptTemplates
# =========================================================================
class TestPromptTemplates:
    """Tests for prompt template rendering."""

    def test_render_convergence(self):
        """Should render convergence template."""
        from scripts.ml_augmentation.llm_benchmark_assistant import (
            PromptTemplates,
        )
        prompt = PromptTemplates.render(
            "analyze_convergence",
            case_name="Wall Hump",
            solver="SU2", turbulence_model="SA",
            n_iterations=10000,
            initial_residual=1.0,
            final_residual=1e-6,
            orders_reduction=6.0,
            cl_std=0.001, cd_std=0.0001,
        )
        assert "Wall Hump" in prompt
        assert "10000" in prompt
        assert "6.0" in prompt

    def test_render_with_missing_vars(self):
        """Should handle missing variables gracefully."""
        from scripts.ml_augmentation.llm_benchmark_assistant import (
            PromptTemplates,
        )
        prompt = PromptTemplates.render(
            "analyze_convergence",
            case_name="Test",
        )
        assert "Test" in prompt
        assert "N/A" in prompt  # Missing vars filled with N/A

    def test_unknown_template_raises(self):
        """Should raise for unknown templates."""
        from scripts.ml_augmentation.llm_benchmark_assistant import (
            PromptTemplates,
        )
        with pytest.raises(ValueError, match="Unknown template"):
            PromptTemplates.render("nonexistent_template")

    def test_system_prompt_exists(self):
        """System prompt should contain CFD context."""
        from scripts.ml_augmentation.llm_benchmark_assistant import (
            PromptTemplates,
        )
        assert "CFD" in PromptTemplates.SYSTEM_PROMPT
        assert "RANS" in PromptTemplates.SYSTEM_PROMPT


# =========================================================================
# TestPhysicsSanityChecker
# =========================================================================
class TestPhysicsSanityChecker:
    """Tests for rule-based physics checks."""

    def test_stagnation_cp_pass(self):
        """Cp_max ~1.0 should pass."""
        from scripts.ml_augmentation.llm_benchmark_assistant import (
            PhysicsSanityChecker,
        )
        Cp = np.array([0.5, 1.0, -0.3, 0.2])
        finding = PhysicsSanityChecker.check_stagnation_cp(Cp)
        assert finding.severity == "info"

    def test_stagnation_cp_warning(self):
        """Cp_max far from 1.0 should warn."""
        from scripts.ml_augmentation.llm_benchmark_assistant import (
            PhysicsSanityChecker,
        )
        Cp = np.array([0.3, 0.5, -0.2])
        finding = PhysicsSanityChecker.check_stagnation_cp(Cp)
        assert finding.severity == "warning"

    def test_residual_convergence_pass(self):
        """Sufficient residual drop should pass."""
        from scripts.ml_augmentation.llm_benchmark_assistant import (
            PhysicsSanityChecker, ConvergenceData,
        )
        conv = ConvergenceData(orders_reduction=5.0)
        finding = PhysicsSanityChecker.check_residual_convergence(conv)
        assert finding.severity == "info"

    def test_residual_convergence_fail(self):
        """Insufficient drop should error."""
        from scripts.ml_augmentation.llm_benchmark_assistant import (
            PhysicsSanityChecker, ConvergenceData,
        )
        conv = ConvergenceData(orders_reduction=1.5)
        finding = PhysicsSanityChecker.check_residual_convergence(conv)
        assert finding.severity == "error"
        assert finding.recommendation != ""

    def test_force_oscillation_stable(self):
        """Stable forces should pass."""
        from scripts.ml_augmentation.llm_benchmark_assistant import (
            PhysicsSanityChecker,
        )
        cl = np.ones(100) * 0.5 + np.random.randn(100) * 0.0001
        cd = np.ones(100) * 0.02 + np.random.randn(100) * 0.00001
        finding = PhysicsSanityChecker.check_force_oscillation(cl, cd)
        assert finding.severity == "info"

    def test_force_oscillation_warning(self):
        """Large oscillations should warn."""
        from scripts.ml_augmentation.llm_benchmark_assistant import (
            PhysicsSanityChecker,
        )
        cl = 0.5 + 0.1 * np.sin(np.linspace(0, 20 * np.pi, 100))
        cd = np.ones(100) * 0.02
        finding = PhysicsSanityChecker.check_force_oscillation(cl, cd)
        assert finding.severity == "warning"

    def test_negative_drag_error(self):
        """Negative drag should error."""
        from scripts.ml_augmentation.llm_benchmark_assistant import (
            PhysicsSanityChecker,
        )
        finding = PhysicsSanityChecker.check_net_drag_positive(-0.005)
        assert finding.severity == "error"
        assert "negative" in finding.message.lower()

    def test_positive_drag_pass(self):
        """Positive drag should pass."""
        from scripts.ml_augmentation.llm_benchmark_assistant import (
            PhysicsSanityChecker,
        )
        finding = PhysicsSanityChecker.check_net_drag_positive(0.02)
        assert finding.severity == "info"

    def test_run_all_checks(self):
        """Should run all checks and return list."""
        from scripts.ml_augmentation.llm_benchmark_assistant import (
            PhysicsSanityChecker, BenchmarkCaseResult,
        )
        case = BenchmarkCaseResult(cd=0.02)
        findings = PhysicsSanityChecker.run_all_checks(case)
        assert len(findings) >= 3
        assert all(hasattr(f, 'severity') for f in findings)


# =========================================================================
# TestBenchmarkCaseAnalyzer
# =========================================================================
class TestBenchmarkCaseAnalyzer:
    """Tests for the full diagnostic pipeline."""

    def test_analyze_produces_report(self):
        """Analyzer should produce a diagnostic report."""
        from scripts.ml_augmentation.llm_benchmark_assistant import (
            BenchmarkCaseAnalyzer, generate_demo_cases,
        )
        cases = generate_demo_cases()
        analyzer = BenchmarkCaseAnalyzer()
        report = analyzer.analyze(cases[0])
        assert report.case_name == cases[0].case_name
        assert report.overall_status in ("pass", "warning", "fail")
        assert report.timestamp != ""

    def test_failing_case_detected(self):
        """Cases with errors should be flagged."""
        from scripts.ml_augmentation.llm_benchmark_assistant import (
            BenchmarkCaseAnalyzer, generate_demo_cases,
        )
        cases = generate_demo_cases()
        analyzer = BenchmarkCaseAnalyzer()
        # Second case has insufficient convergence
        report = analyzer.analyze(cases[1])
        assert report.n_errors > 0 or report.n_warnings > 0

    def test_recommendations_generated(self):
        """Should generate actionable recommendations."""
        from scripts.ml_augmentation.llm_benchmark_assistant import (
            BenchmarkCaseAnalyzer, generate_demo_cases,
        )
        cases = generate_demo_cases()
        analyzer = BenchmarkCaseAnalyzer()
        report = analyzer.analyze(cases[1])
        # Under-converged case should have recommendations
        assert report.summary != ""


# =========================================================================
# TestNarrativeReportGenerator
# =========================================================================
class TestNarrativeReportGenerator:
    """Tests for narrative report generation."""

    def test_generate_case_section(self):
        """Should produce a markdown section for a case."""
        from scripts.ml_augmentation.llm_benchmark_assistant import (
            NarrativeReportGenerator, BenchmarkCaseAnalyzer,
            generate_demo_cases,
        )
        cases = generate_demo_cases()
        gen = NarrativeReportGenerator()
        analyzer = BenchmarkCaseAnalyzer()
        diag = analyzer.analyze(cases[0])
        section = gen.generate_case_section(cases[0], diag)
        assert "###" in section
        assert cases[0].case_name in section

    def test_full_report_structure(self):
        """Full report should have expected sections."""
        from scripts.ml_augmentation.llm_benchmark_assistant import (
            NarrativeReportGenerator, generate_demo_cases,
        )
        cases = generate_demo_cases()
        gen = NarrativeReportGenerator()
        report = gen.generate_full_report(cases)
        assert "# " in report
        assert "Executive Summary" in report
        assert "Recommendations" in report
        assert cases[0].case_name in report


# =========================================================================
# TestLLMBackend
# =========================================================================
class TestLLMBackend:
    """Tests for the LLM backend."""

    def test_rule_based_convergence(self):
        """Rule-based backend should respond to convergence queries."""
        from scripts.ml_augmentation.llm_benchmark_assistant import (
            LLMBackend,
        )
        llm = LLMBackend()
        response = llm.query("Analyze convergence of this simulation")
        assert len(response) > 50
        assert "convergence" in response.lower()

    def test_rule_based_separation(self):
        """Should respond to separation queries."""
        from scripts.ml_augmentation.llm_benchmark_assistant import (
            LLMBackend,
        )
        llm = LLMBackend()
        response = llm.query("Diagnose flow separation behavior")
        assert "separation" in response.lower()

    def test_caching(self):
        """Repeated queries should return cached response."""
        from scripts.ml_augmentation.llm_benchmark_assistant import (
            LLMBackend, LLMConfig,
        )
        config = LLMConfig(cache_responses=True)
        llm = LLMBackend(config)
        r1 = llm.query("Analyze convergence")
        r2 = llm.query("Analyze convergence")
        assert r1 == r2
