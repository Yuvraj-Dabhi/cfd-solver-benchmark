#!/usr/bin/env python3
"""
Tests for Reproducible V&V Benchmark Harness
==============================================
"""

import json
import sys
import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.benchmark_harness import (
    BENCHMARK_CASES,
    CaseMetrics,
    BenchmarkSummary,
    BenchmarkHarness,
)


# =========================================================================
# Data Structure Tests
# =========================================================================

class TestCaseMetrics:
    """Test CaseMetrics dataclass."""

    def test_defaults(self):
        m = CaseMetrics(case_name="test")
        assert m.status == "PENDING"
        assert m.CL is None
        assert m.notes == []

    def test_to_dict_excludes_none(self):
        m = CaseMetrics(case_name="test", CL=1.5, x_sep=0.665)
        d = m.to_dict()
        assert "CL" in d
        assert "x_sep" in d
        assert "CD" not in d  # None excluded

    def test_separation_errors(self):
        m = CaseMetrics(
            case_name="test",
            x_sep=0.668, x_sep_exp=0.665,
            x_sep_error=0.003,
        )
        assert m.x_sep_error == 0.003


class TestBenchmarkSummary:
    """Test BenchmarkSummary dataclass."""

    def test_defaults(self):
        s = BenchmarkSummary()
        assert s.n_cases == 0
        assert s.overall_status == ""

    def test_to_dict(self):
        s = BenchmarkSummary(n_cases=5, n_passed=3, n_warned=1, n_failed=1)
        s.cases["test"] = CaseMetrics(case_name="test", status="PASS")
        d = s.to_dict()
        assert d["n_cases"] == 5
        assert "test" in d["cases"]


# =========================================================================
# Benchmark Registry
# =========================================================================

class TestBenchmarkCases:
    """Test the benchmark case registry."""

    def test_all_cases_present(self):
        expected = {"flat_plate", "wall_hump", "naca0012", "bfs", "swbli"}
        assert set(BENCHMARK_CASES.keys()) == expected

    @pytest.mark.parametrize("case", list(BENCHMARK_CASES.keys()))
    def test_case_has_required_fields(self, case):
        data = BENCHMARK_CASES[case]
        assert "label" in data
        assert "type" in data
        assert "reference" in data
        assert "metrics" in data


# =========================================================================
# Harness Execution
# =========================================================================

class TestBenchmarkHarness:
    """Test the full harness execution."""

    @pytest.fixture(scope="class")
    def harness_result(self, tmp_path_factory):
        out = tmp_path_factory.mktemp("output")
        harness = BenchmarkHarness(output_dir=str(out))
        summary = harness.run()
        harness.write_all()
        return harness, summary, out

    def test_runs_all_cases(self, harness_result):
        _, summary, _ = harness_result
        assert summary.n_cases == 5

    def test_status_counts(self, harness_result):
        _, summary, _ = harness_result
        total = summary.n_passed + summary.n_warned + summary.n_failed
        assert total == summary.n_cases

    def test_overall_status_set(self, harness_result):
        _, summary, _ = harness_result
        assert summary.overall_status in [
            "ALL PASS", "PASS WITH WARNINGS", "FAILURES PRESENT",
        ]

    def test_wall_hump_metrics(self, harness_result):
        _, summary, _ = harness_result
        m = summary.cases["wall_hump"]
        assert m.x_sep is not None
        assert m.x_reat is not None
        assert m.L_bubble is not None
        assert m.Cp_RMSE is not None

    def test_gci_collected(self, harness_result):
        _, summary, _ = harness_result
        m = summary.cases["wall_hump"]
        assert m.GCI_fine_pct is not None
        assert m.observed_order is not None

    def test_uncertainty_budget_collected(self, harness_result):
        _, summary, _ = harness_result
        m = summary.cases["wall_hump"]
        assert m.U_total is not None
        assert m.U_disc is not None

    def test_flat_plate_is_verification(self, harness_result):
        _, summary, _ = harness_result
        m = summary.cases["flat_plate"]
        assert m.case_type == "verification"

    def test_json_output(self, harness_result):
        _, _, out = harness_result
        p = out / "benchmark_summary.json"
        assert p.exists()
        with open(p) as f:
            data = json.load(f)
        assert "n_cases" in data
        assert "cases" in data
        assert len(data["cases"]) == 5

    def test_csv_output(self, harness_result):
        _, _, out = harness_result
        p = out / "benchmark_summary.csv"
        assert p.exists()
        lines = p.read_text().strip().split("\n")
        assert len(lines) >= 2  # header + at least 1 data row

    def test_markdown_output(self, harness_result):
        _, _, out = harness_result
        p = out / "benchmark_report.md"
        assert p.exists()
        content = p.read_text()
        assert "Executive Summary" in content
        assert "Per-Case Results" in content
        assert "Grid Convergence" in content
        assert "Uncertainty Budgets" in content
        assert "Separation Bubble" in content

    def test_per_case_json(self, harness_result):
        _, _, out = harness_result
        for case in BENCHMARK_CASES:
            p = out / "per_case" / f"{case}.json"
            assert p.exists(), f"Missing per-case JSON for {case}"


class TestHarnessSelectedCases:
    """Test running with specific cases."""

    def test_selected_cases(self, tmp_path):
        harness = BenchmarkHarness(
            cases=["wall_hump", "bfs"],
            output_dir=str(tmp_path),
        )
        summary = harness.run()
        assert summary.n_cases == 2
        assert "wall_hump" in summary.cases
        assert "bfs" in summary.cases
        assert "naca0012" not in summary.cases
