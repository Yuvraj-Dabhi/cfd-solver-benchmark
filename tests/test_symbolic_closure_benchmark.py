#!/usr/bin/env python3
"""
Tests: Symbolic Closure Benchmark
==================================
Validates unit-constrained GEP, cross-flow evaluation, interpretability
analysis, and end-to-end benchmark pipeline.

Run: pytest tests/test_symbolic_closure_benchmark.py -v --tb=short
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ml_augmentation.gep_explicit_closure import (
    ExprNode,
    ClosureFormula,
    GEPClosureDiscovery,
    generate_synthetic_closure_data,
    _random_tree,
)
from scripts.ml_augmentation.symbolic_closure_benchmark import (
    UnitConstrainedGEP,
    SymbolicClosureEvaluator,
    InterpretabilityAnalyzer,
    SymbolicClosureBenchmarkRunner,
    FlowCaseData,
    ModelResult,
    _generate_case_data,
    CANONICAL_CASES,
)


# ============================================================================
# Unit Constraint Tests
# ============================================================================

class TestUnitConstraints:
    """Test dimensional analysis and invariant enforcement."""

    def test_constant_is_dimensionless(self):
        node = ExprNode("const", value=3.14)
        assert node.is_dimensionless()

    def test_invariant_var_is_dimensionless(self):
        node = ExprNode("var", var_idx=0)
        assert node.is_dimensionless([0, 0, 0, 0, 0])

    def test_dimensional_var_is_not_dimensionless(self):
        node = ExprNode("var", var_idx=2)
        assert not node.is_dimensionless([0, 0, 1, 0, 0])

    def test_add_same_dims_ok(self):
        left = ExprNode("var", var_idx=0)
        right = ExprNode("var", var_idx=1)
        node = ExprNode("add", [left, right])
        assert node.is_dimensionless([0, 0])

    def test_add_mixed_dims_fails(self):
        left = ExprNode("var", var_idx=0)
        right = ExprNode("var", var_idx=1)
        node = ExprNode("add", [left, right])
        assert not node.is_dimensionless([0, 1])

    def test_mul_preserves_dimensionless(self):
        left = ExprNode("var", var_idx=0)
        right = ExprNode("const", value=2.0)
        node = ExprNode("mul", [left, right])
        assert node.is_dimensionless([0])

    def test_uses_only_invariants_pass(self):
        node = ExprNode("mul", [
            ExprNode("var", var_idx=0),
            ExprNode("var", var_idx=4),
        ])
        assert node.uses_only_invariants(5)

    def test_uses_only_invariants_fail(self):
        node = ExprNode("mul", [
            ExprNode("var", var_idx=0),
            ExprNode("var", var_idx=7),
        ])
        assert not node.uses_only_invariants(5)

    def test_complex_dimensionless_tree(self):
        # (λ₁ * λ₂) / (λ₃ + const)
        tree = ExprNode("div", [
            ExprNode("mul", [
                ExprNode("var", var_idx=0),
                ExprNode("var", var_idx=1),
            ]),
            ExprNode("add", [
                ExprNode("var", var_idx=2),
                ExprNode("const", value=1.0),
            ]),
        ])
        assert tree.is_dimensionless([0, 0, 0, 0, 0])
        assert tree.uses_only_invariants(5)


# ============================================================================
# Unit-Constrained GEP Tests
# ============================================================================

class TestUnitConstrainedGEP:
    """Test UnitConstrainedGEP class."""

    def test_import(self):
        gep = UnitConstrainedGEP(n_bases=2, max_depth=3)
        assert gep.n_bases == 2

    def test_discover_runs(self):
        data = generate_synthetic_closure_data(n_points=100, seed=42)
        gep = UnitConstrainedGEP(n_bases=2, max_depth=3, seed=42)
        formula = gep.discover(
            data["invariants"], data["b_target"], data["tensor_bases"],
            n_generations=5, population_size=30,
        )
        assert isinstance(formula, ClosureFormula)
        assert len(formula.coefficient_trees) == 2
        assert formula.fitness < float("inf")

    def test_unit_report(self):
        data = generate_synthetic_closure_data(n_points=100, seed=42)
        gep = UnitConstrainedGEP(n_bases=2, max_depth=3, seed=42)
        formula = gep.discover(
            data["invariants"], data["b_target"], data["tensor_bases"],
            n_generations=5, population_size=30,
        )
        report = gep.get_unit_report(formula)
        assert "coefficients" in report
        assert "all_pass" in report
        assert "g1" in report["coefficients"]
        assert "dimensionless" in report["coefficients"]["g1"]

    def test_unit_check_consistency(self):
        gep = UnitConstrainedGEP(n_bases=2, n_invariants=5)
        # Dimensionless tree
        tree_good = ExprNode("mul", [
            ExprNode("const", value=-0.1),
            ExprNode("var", var_idx=0),
        ])
        assert gep._check_unit_consistency(tree_good)

        # Tree using out-of-range variable
        tree_bad = ExprNode("var", var_idx=7)
        assert not gep._check_unit_consistency(tree_bad)


# ============================================================================
# Evaluator Tests
# ============================================================================

class TestSymbolicClosureEvaluator:
    """Test cross-flow evaluation."""

    @pytest.fixture
    def evaluator(self):
        return SymbolicClosureEvaluator(
            cases=["periodic_hill", "backward_facing_step"],
            n_points=100,
        )

    @pytest.fixture
    def simple_formula(self):
        trees = [
            ExprNode("mul", [
                ExprNode("const", value=-0.1),
                ExprNode("var", var_idx=0),
            ]),
            ExprNode("mul", [
                ExprNode("const", value=0.05),
                ExprNode("var", var_idx=1),
            ]),
        ]
        return ClosureFormula(
            coefficient_trees=trees, n_bases=2,
            fitness=0.01, complexity_score=6,
        )

    def test_evaluator_cases_loaded(self, evaluator):
        assert len(evaluator.cases) == 2
        assert "periodic_hill" in evaluator.cases

    def test_evaluate_symbolic(self, evaluator, simple_formula):
        results = evaluator.evaluate_symbolic(simple_formula, "test_GEP")
        assert len(results) == 2
        assert all(isinstance(r, ModelResult) for r in results)
        assert all(r.model_type == "symbolic" for r in results)
        assert all(r.b_rmse >= 0 for r in results)

    def test_evaluate_baseline_rans(self, evaluator):
        results = evaluator.evaluate_baseline_rans()
        assert len(results) == 4  # 2 models × 2 cases
        model_names = set(r.model_name for r in results)
        assert "SA_baseline" in model_names
        assert "SST_baseline" in model_names

    def test_evaluate_blackbox_ml(self, evaluator):
        results = evaluator.evaluate_blackbox_ml()
        assert len(results) == 4  # 2 models × 2 cases
        model_names = set(r.model_name for r in results)
        assert "TBNN" in model_names
        assert "FIML" in model_names

    def test_pareto_frontier(self, evaluator, simple_formula):
        evaluator.evaluate_symbolic(simple_formula, "test_GEP")
        evaluator.evaluate_baseline_rans()
        evaluator.evaluate_blackbox_ml()
        frontier = evaluator.get_pareto_frontier()
        assert len(frontier) >= 1
        # Frontier should be sorted by complexity
        complexities = [f.complexity for f in frontier]
        assert complexities == sorted(complexities)

    def test_format_results_markdown(self, evaluator, simple_formula):
        evaluator.evaluate_symbolic(simple_formula, "test_GEP")
        evaluator.evaluate_baseline_rans()
        md = evaluator.format_results_markdown()
        assert "## Symbolic Closure Benchmark Results" in md
        assert "test_GEP" in md
        assert "SA_baseline" in md

    def test_compute_metrics(self, evaluator):
        b_pred = np.zeros((50, 3, 3))
        b_true = np.ones((50, 3, 3)) * 0.01
        rmse, mae, rv = evaluator._compute_metrics(b_pred, b_true)
        assert rmse > 0
        assert mae > 0
        assert 0.0 <= rv <= 1.0

    def test_realizability_all_zero(self, evaluator):
        b_pred = np.zeros((10, 3, 3))
        b_true = np.zeros((10, 3, 3))
        rmse, mae, rv = evaluator._compute_metrics(b_pred, b_true)
        assert rmse == 0.0
        assert rv == 0.0  # Zero tensor is realizable


# ============================================================================
# Interpretability Tests
# ============================================================================

class TestInterpretabilityAnalyzer:
    """Test interpretability analysis."""

    @pytest.fixture
    def formula_and_data(self):
        data = generate_synthetic_closure_data(n_points=200, seed=42)
        trees = [
            ExprNode("mul", [
                ExprNode("const", value=-0.1),
                ExprNode("div", [
                    ExprNode("var", var_idx=0),
                    ExprNode("add", [
                        ExprNode("const", value=1.0),
                        ExprNode("var", var_idx=0),
                    ]),
                ]),
            ]),
            ExprNode("mul", [
                ExprNode("const", value=0.05),
                ExprNode("var", var_idx=1),
            ]),
        ]
        formula = ClosureFormula(
            coefficient_trees=trees, n_bases=2,
            fitness=0.005, complexity_score=9,
        )
        return formula, data

    def test_term_contributions(self, formula_and_data):
        formula, data = formula_and_data
        analyzer = InterpretabilityAnalyzer(formula)
        contribs = analyzer.analyse_term_contributions(
            data["invariants"], data["tensor_bases"], data["b_target"]
        )
        assert "g1" in contribs
        assert "g2" in contribs
        assert contribs["g1"]["fraction_of_total"] >= 0
        assert contribs["g1"]["complexity"] > 0

    def test_physics_alignment(self, formula_and_data):
        formula, data = formula_and_data
        analyzer = InterpretabilityAnalyzer(formula)
        alignment = analyzer.analyse_physics_alignment(data["invariants"])
        assert "overall_score" in alignment
        assert alignment["overall_score"]["total"] > 0

    def test_g1_negative_under_apg(self, formula_and_data):
        formula, data = formula_and_data
        analyzer = InterpretabilityAnalyzer(formula)
        alignment = analyzer.analyse_physics_alignment(data["invariants"])
        # g1 = -0.1 * λ₁/(1+λ₁) should be negative under APG
        if "g1_under_APG" in alignment:
            assert alignment["g1_under_APG"]["actual_sign"] == "negative"

    def test_region_response(self, formula_and_data):
        formula, data = formula_and_data
        analyzer = InterpretabilityAnalyzer(formula)
        regions = analyzer.analyse_region_response(data["invariants"])
        assert len(regions) > 0
        assert "attached_bl" in regions or "strain_dominated" in regions

    def test_classical_comparison(self, formula_and_data):
        formula, _ = formula_and_data
        analyzer = InterpretabilityAnalyzer(formula)
        comparisons = analyzer.classical_comparison()
        assert "g1" in comparisons
        assert "model_class" in comparisons["g1"]
        assert "classical_analogue" in comparisons["g1"]

    def test_generate_report(self, formula_and_data):
        formula, data = formula_and_data
        analyzer = InterpretabilityAnalyzer(formula)
        report = analyzer.generate_report(
            data["invariants"], data["tensor_bases"], data["b_target"]
        )
        assert "## Interpretability Analysis" in report
        assert "Term Contributions" in report
        assert "Physics Alignment" in report
        assert "Classical Model Comparison" in report

    def test_report_contains_expressions(self, formula_and_data):
        formula, data = formula_and_data
        analyzer = InterpretabilityAnalyzer(formula)
        report = analyzer.generate_report(
            data["invariants"], data["tensor_bases"], data["b_target"]
        )
        assert "g^{" in report  # LaTeX expressions

    def test_zero_coefficient_handling(self):
        """Test with a degenerate zero-coefficient formula."""
        trees = [ExprNode("const", value=0.0)]
        formula = ClosureFormula(coefficient_trees=trees, n_bases=1)
        data = generate_synthetic_closure_data(n_points=50)
        analyzer = InterpretabilityAnalyzer(formula)
        contribs = analyzer.analyse_term_contributions(
            data["invariants"], data["tensor_bases"], data["b_target"]
        )
        assert contribs["g1"]["rms_contribution"] == 0.0


# ============================================================================
# Flow Case Data Tests
# ============================================================================

class TestFlowCaseData:
    """Test case data generation."""

    def test_generate_case_data(self):
        data = _generate_case_data("periodic_hill", n_points=50, seed=42)
        assert isinstance(data, FlowCaseData)
        assert data.n_points == 50
        assert data.invariants.shape == (50, 5)
        assert data.tensor_bases.shape == (50, 10, 3, 3)
        assert data.b_target.shape == (50, 3, 3)

    def test_canonical_cases(self):
        assert len(CANONICAL_CASES) == 4
        for case in CANONICAL_CASES:
            data = _generate_case_data(case, n_points=20)
            assert data.case_name == case
            assert data.n_points == 20

    def test_different_cases_different_data(self):
        d1 = _generate_case_data("periodic_hill", n_points=50, seed=42)
        d2 = _generate_case_data("nasa_hump", n_points=50, seed=1042)
        # Different seed → different invariants
        assert not np.allclose(d1.invariants, d2.invariants)

    def test_unknown_case_uses_default(self):
        data = _generate_case_data("unknown_case", n_points=20)
        assert data.n_points == 20  # Should not crash


# ============================================================================
# Integration Tests
# ============================================================================

class TestBenchmarkRunner:
    """End-to-end integration tests."""

    def test_runner_import(self):
        runner = SymbolicClosureBenchmarkRunner(fast_mode=True)
        assert runner.fast_mode is True

    def test_runner_fast(self, tmp_path):
        runner = SymbolicClosureBenchmarkRunner(fast_mode=True)
        results = runner.run(output_dir=tmp_path / "output")
        assert isinstance(results, dict)
        assert "formulas" in results
        assert "evaluation" in results
        assert "pareto_frontier" in results
        assert (tmp_path / "output" / "benchmark_report.md").exists()
        assert (tmp_path / "output" / "benchmark_results.json").exists()

    def test_runner_json_valid(self, tmp_path):
        runner = SymbolicClosureBenchmarkRunner(fast_mode=True)
        runner.run(output_dir=tmp_path / "output")
        with open(tmp_path / "output" / "benchmark_results.json") as f:
            data = json.load(f)
        assert "formulas" in data
        assert "minimal" in data["formulas"]
        assert "complex" in data["formulas"]

    def test_runner_report_content(self, tmp_path):
        runner = SymbolicClosureBenchmarkRunner(fast_mode=True)
        runner.run(output_dir=tmp_path / "output")
        report = (tmp_path / "output" / "benchmark_report.md").read_text()
        assert "Symbolic Closure Benchmark Report" in report
        assert "SU2 C++ Export" in report
        assert "Interpretability Analysis" in report
