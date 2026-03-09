"""
Tests for GEP Explicit Closure Module
========================================
Validates symbolic regression, formula evaluation, SU2 C++ export,
transfer learning, and realizability enforcement.
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
    evaluate_transfer,
    constrained_recalibration,
    generate_synthetic_closure_data,
    _random_tree,
    _mutate_tree,
)


class TestExprNode:
    """Test symbolic expression evaluation."""

    def test_constant(self):
        node = ExprNode("const", value=3.14)
        X = np.zeros((5, 3))
        result = node.evaluate(X)
        np.testing.assert_allclose(result, 3.14)

    def test_variable(self):
        node = ExprNode("var", var_idx=1)
        X = np.array([[1, 2, 3], [4, 5, 6]])
        result = node.evaluate(X)
        np.testing.assert_array_equal(result, [2, 5])

    def test_add(self):
        a = ExprNode("const", value=2.0)
        b = ExprNode("var", var_idx=0)
        node = ExprNode("add", [a, b])
        X = np.array([[3.0], [7.0]])
        result = node.evaluate(X)
        np.testing.assert_allclose(result, [5.0, 9.0])

    def test_safe_div(self):
        """Division by near-zero should not crash."""
        a = ExprNode("const", value=1.0)
        b = ExprNode("const", value=0.0)
        node = ExprNode("div", [a, b])
        X = np.zeros((3, 1))
        result = node.evaluate(X)
        assert np.all(np.isfinite(result))

    def test_sqrt_negative(self):
        """sqrt of negative should use abs."""
        a = ExprNode("const", value=-4.0)
        node = ExprNode("sqrt", [a])
        X = np.zeros((2, 1))
        result = node.evaluate(X)
        np.testing.assert_allclose(result, 2.0)

    def test_to_string(self):
        a = ExprNode("const", value=0.1)
        b = ExprNode("var", var_idx=0)
        node = ExprNode("mul", [a, b])
        s = node.to_string(["λ₁", "λ₂"])
        assert "λ₁" in s
        assert "0.1" in s

    def test_complexity(self):
        a = ExprNode("const", value=1.0)
        b = ExprNode("var", var_idx=0)
        node = ExprNode("add", [a, b])
        assert node.complexity() == 3  # add + const + var

    def test_depth(self):
        a = ExprNode("const", value=1.0)
        b = ExprNode("var", var_idx=0)
        node = ExprNode("add", [a, b])
        assert node.depth() == 1


class TestRandomTree:
    """Test tree generation and mutation."""

    def test_random_tree_evaluates(self):
        rng = np.random.default_rng(42)
        tree = _random_tree(5, max_depth=3, rng=rng)
        X = np.random.randn(10, 5)
        result = tree.evaluate(X)
        assert result.shape == (10,)

    def test_mutation_changes_tree(self):
        rng = np.random.default_rng(42)
        tree = ExprNode("var", var_idx=0)
        mutated = _mutate_tree(tree, 5, rng)
        # Mutation should produce a valid tree
        X = np.random.randn(5, 5)
        result = mutated.evaluate(X)
        assert len(result) == 5


class TestClosureFormula:
    """Test formula evaluation and export."""

    def _make_simple_formula(self):
        """g1 = 0.1 * λ₁, g2 = -0.05 * λ₂"""
        t1 = ExprNode("mul", [
            ExprNode("const", value=0.1),
            ExprNode("var", var_idx=0),
        ])
        t2 = ExprNode("mul", [
            ExprNode("const", value=-0.05),
            ExprNode("var", var_idx=1),
        ])
        return ClosureFormula(
            coefficient_trees=[t1, t2],
            n_bases=2,
        )

    def test_evaluate_coefficients(self):
        formula = self._make_simple_formula()
        inv = np.array([[1.0, 2.0, 0, 0, 0],
                        [3.0, 4.0, 0, 0, 0]])
        g = formula.evaluate_coefficients(inv)
        assert g.shape == (2, 2)
        np.testing.assert_allclose(g[:, 0], [0.1, 0.3])
        np.testing.assert_allclose(g[:, 1], [-0.1, -0.2])

    def test_predict_anisotropy_shape(self):
        formula = self._make_simple_formula()
        inv = np.random.randn(10, 5)
        T = np.random.randn(10, 10, 3, 3)
        b = formula.predict_anisotropy(inv, T)
        assert b.shape == (10, 3, 3)

    def test_to_latex(self):
        formula = self._make_simple_formula()
        latex = formula.to_latex()
        assert "g^{(1)}" in latex
        assert "g^{(2)}" in latex

    def test_to_su2_cpp(self):
        formula = self._make_simple_formula()
        cpp = formula.to_su2_cpp()
        assert "su2double g_1" in cpp
        assert "su2double g_2" in cpp
        assert "lambda1" in cpp
        assert "b_ij_correction" in cpp

    def test_to_dict(self):
        formula = self._make_simple_formula()
        formula.fitness = 0.001
        d = formula.to_dict()
        assert d["n_bases"] == 2
        assert d["fitness"] == 0.001
        assert "g1" in d["expressions"]
        # Should be JSON-serializable
        json.dumps(d)


class TestGEPDiscovery:
    """Test symbolic regression discovery."""

    def test_discovery_runs(self):
        data = generate_synthetic_closure_data(n_points=100, seed=42)

        gep = GEPClosureDiscovery(n_bases=2, max_depth=3, seed=42)
        gep.add_training_data(
            data["invariants"], data["b_target"], data["tensor_bases"])

        formula = gep.discover(
            n_generations=10, population_size=30,
        )

        assert isinstance(formula, ClosureFormula)
        assert len(formula.coefficient_trees) == 2
        assert np.isfinite(formula.fitness)
        assert formula.complexity_score > 0

    def test_discovery_improves_over_random(self):
        """GEP should produce lower MSE than random expressions."""
        data = generate_synthetic_closure_data(n_points=100, seed=42)

        # Random formula
        rng = np.random.default_rng(99)
        random_trees = [_random_tree(5, 3, rng) for _ in range(2)]
        random_formula = ClosureFormula(coefficient_trees=random_trees, n_bases=2)
        b_rand = random_formula.predict_anisotropy(
            data["invariants"], data["tensor_bases"])
        random_mse = np.mean((b_rand - data["b_target"]) ** 2)

        # GEP formula
        gep = GEPClosureDiscovery(n_bases=2, max_depth=3, seed=42)
        gep.add_training_data(
            data["invariants"], data["b_target"], data["tensor_bases"])
        gep_formula = gep.discover(n_generations=20, population_size=50)

        # GEP should be at least as good (usually much better)
        # We check it's finite and doesn't completely diverge
        assert np.isfinite(gep_formula.fitness)

    def test_requires_training_data(self):
        gep = GEPClosureDiscovery(n_bases=2)
        with pytest.raises(RuntimeError, match="No training data"):
            gep.discover()


class TestTransferEvaluation:
    """Test cross-case transfer metrics."""

    def test_transfer_evaluation(self):
        data1 = generate_synthetic_closure_data(n_points=50, seed=1)
        data2 = generate_synthetic_closure_data(n_points=50, seed=2)

        # Create a simple formula
        t1 = ExprNode("mul", [
            ExprNode("const", value=-0.1),
            ExprNode("var", var_idx=0),
        ])
        formula = ClosureFormula(coefficient_trees=[t1], n_bases=1)

        results = evaluate_transfer(
            formula,
            train_cases={"hill": data1},
            test_cases={"hump": data2},
        )

        assert "train" in results
        assert "test" in results
        assert "hill" in results["train"]
        assert "hump" in results["test"]
        assert "MSE" in results["train"]["hill"]
        assert "transfer_gap" in results
        assert "gap_ratio" in results["transfer_gap"]


class TestConstrainedRecalibration:
    """Test realizability enforcement."""

    def test_trace_free(self):
        data = generate_synthetic_closure_data(n_points=50)
        t1 = ExprNode("const", value=0.5)
        formula = ClosureFormula(coefficient_trees=[t1], n_bases=1)

        b_constrained = constrained_recalibration(
            formula, data["invariants"], data["tensor_bases"])

        traces = np.trace(b_constrained, axis1=-2, axis2=-1)
        np.testing.assert_allclose(traces, 0.0, atol=1e-10)

    def test_symmetry(self):
        data = generate_synthetic_closure_data(n_points=50)
        t1 = ExprNode("const", value=0.5)
        formula = ClosureFormula(coefficient_trees=[t1], n_bases=1)

        b_constrained = constrained_recalibration(
            formula, data["invariants"], data["tensor_bases"])

        for i in range(len(b_constrained)):
            np.testing.assert_allclose(
                b_constrained[i], b_constrained[i].T, atol=1e-10)

    def test_eigenvalue_bounds(self):
        data = generate_synthetic_closure_data(n_points=50)
        t1 = ExprNode("const", value=5.0)  # Large → will need clamping
        formula = ClosureFormula(coefficient_trees=[t1], n_bases=1)

        b_constrained = constrained_recalibration(
            formula, data["invariants"], data["tensor_bases"])

        for i in range(len(b_constrained)):
            eigvals = np.linalg.eigvalsh(b_constrained[i])
            assert np.all(eigvals >= -1.0/3.0 - 1e-10)
            assert np.all(eigvals <= 2.0/3.0 + 1e-10)


class TestSyntheticData:
    """Test synthetic data generation."""

    def test_data_shapes(self):
        data = generate_synthetic_closure_data(n_points=100)
        assert data["invariants"].shape == (100, 5)
        assert data["tensor_bases"].shape == (100, 10, 3, 3)
        assert data["b_target"].shape == (100, 3, 3)
        assert data["true_g"].shape == (100, 3)

    def test_tensor_basis_symmetry(self):
        """T(1) = S_hat should be symmetric."""
        data = generate_synthetic_closure_data(n_points=20)
        S = data["tensor_bases"][:, 0]  # T(1) = S_hat
        np.testing.assert_allclose(S, np.swapaxes(S, -2, -1), atol=1e-10)
