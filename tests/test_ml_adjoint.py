"""
Tests for ML-Augmented Adjoint Optimization
=============================================
Validates adjoint sensitivity, surrogate gradient, hybrid optimizer,
β-field initializer, and Rosenbrock convergence.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ml_augmentation.ml_adjoint_optimization import (
    AdjointSensitivity,
    SurrogateGradient,
    HybridOptimizer,
    OptimizationResult,
    BetaFieldInitializer,
    generate_rosenbrock_problem,
    generate_synthetic_adjoint_data,
)


class TestAdjointSensitivity:
    """Test adjoint sensitivity extraction."""

    def test_from_arrays(self):
        adj = AdjointSensitivity.from_arrays(
            ["param1", "param2"], np.array([0.5, -0.3]), "CL")
        assert adj.n_vars == 2
        assert adj.objective == "CL"
        np.testing.assert_allclose(adj.gradients, [0.5, -0.3])

    def test_top_sensitivities(self):
        adj = generate_synthetic_adjoint_data(n_vars=10)
        top = adj.top_sensitivities(n=3)
        assert len(top) == 3
        # First should have largest magnitude
        assert abs(top[0][1]) >= abs(top[1][1])

    def test_empty_sensitivity(self):
        adj = AdjointSensitivity()
        assert adj.top_sensitivities() == []

    def test_from_csv(self, tmp_path):
        csv_path = tmp_path / "adj_sens.csv"
        csv_path.write_text("design_var,dJ/dx\nhick_0,0.5\nhick_1,-0.3\n")
        adj = AdjointSensitivity.from_su2_history(csv_path)
        assert adj.n_vars == 2

    def test_from_nonexistent_file(self):
        adj = AdjointSensitivity.from_su2_history("/nonexistent/path.csv")
        assert adj.n_vars == 0


class TestSurrogateGradient:
    """Test surrogate gradient computation."""

    def test_quadratic_gradient(self):
        """Gradient of x² should be 2x."""
        func = lambda x: float(np.sum(x**2))
        sg = SurrogateGradient(func, eps=1e-5)

        x = np.array([3.0, -2.0])
        grad = sg.gradient(x)
        np.testing.assert_allclose(grad, [6.0, -4.0], atol=1e-3)

    def test_linear_gradient(self):
        """Gradient of 2x + 3y should be [2, 3]."""
        func = lambda x: 2*x[0] + 3*x[1]
        sg = SurrogateGradient(func)

        grad = sg.gradient(np.array([0.0, 0.0]))
        np.testing.assert_allclose(grad, [2.0, 3.0], atol=1e-3)


class TestHybridOptimizer:
    """Test hybrid optimization."""

    def test_rosenbrock_descent(self):
        """Optimizer should reduce objective on Rosenbrock."""
        problem = generate_rosenbrock_problem(n_vars=2)
        optimizer = HybridOptimizer(alpha=0.5, lr=0.0005, bounds=[(-5, 5)] * 2)

        x0 = np.array([0.0, 0.0])
        result = optimizer.optimize(
            x0=x0,
            adjoint_grad_func=problem["gradient"],
            surrogate_grad_func=problem["gradient"],
            objective_func=problem["objective"],
            n_steps=50,
        )

        assert result.n_iterations > 0
        # Objective should decrease from initial
        assert result.history[-1]["f"] < result.history[0]["f"]

    def test_pure_adjoint(self):
        """alpha=1.0 uses only adjoint gradient."""
        func = lambda x: np.sum(x**2)
        grad = lambda x: 2 * x

        optimizer = HybridOptimizer(alpha=1.0, lr=0.1)
        result = optimizer.optimize(
            x0=np.array([5.0]),
            adjoint_grad_func=grad,
            surrogate_grad_func=lambda x: np.zeros_like(x),  # Ignored
            objective_func=func,
            n_steps=20,
        )
        assert result.f_opt < 25.0  # Should improve from f(5)=25

    def test_pure_surrogate(self):
        """alpha=0.0 uses only surrogate gradient."""
        func = lambda x: np.sum(x**2)
        grad = lambda x: 2 * x

        optimizer = HybridOptimizer(alpha=0.0, lr=0.1)
        result = optimizer.optimize(
            x0=np.array([5.0]),
            adjoint_grad_func=lambda x: np.zeros_like(x),
            surrogate_grad_func=grad,
            objective_func=func,
            n_steps=20,
        )
        assert result.f_opt < 25.0

    def test_bounds_enforcement(self):
        """Variables should stay within bounds."""
        func = lambda x: np.sum(x**2)
        grad = lambda x: 2 * x + 10  # Always pushes left

        optimizer = HybridOptimizer(alpha=1.0, lr=1.0, bounds=[(-2, 2)])
        result = optimizer.optimize(
            x0=np.array([0.0]),
            adjoint_grad_func=grad,
            objective_func=func,
            n_steps=10,
        )
        assert result.x_opt[0] >= -2.0
        assert result.x_opt[0] <= 2.0

    def test_convergence_detection(self):
        """Should detect convergence at optimum."""
        func = lambda x: np.sum(x**2)
        grad = lambda x: 2 * x

        optimizer = HybridOptimizer(alpha=1.0, lr=0.4)
        result = optimizer.optimize(
            x0=np.array([0.001]),
            adjoint_grad_func=grad,
            objective_func=func,
            n_steps=100,
            tol=1e-4,
        )
        assert result.converged

    def test_result_history(self):
        problem = generate_rosenbrock_problem(n_vars=2)
        optimizer = HybridOptimizer(alpha=0.5, lr=0.001)
        result = optimizer.optimize(
            x0=problem["x0"],
            adjoint_grad_func=problem["gradient"],
            surrogate_grad_func=problem["gradient"],
            objective_func=problem["objective"],
            n_steps=10,
        )
        assert len(result.history) == 10
        assert "f" in result.history[0]
        assert "grad_norm" in result.history[0]


class TestBetaFieldInitializer:
    """Test β-field prior initialization."""

    def test_fit_and_sample(self):
        init = BetaFieldInitializer()
        init.add_converged_field(np.array([1.0, 1.1, 0.9, 1.05]))
        init.add_converged_field(np.array([1.0, 1.2, 0.85, 1.1]))
        init.add_converged_field(np.array([1.0, 1.15, 0.88, 1.08]))
        init.fit()

        beta0 = init.sample_initial()
        assert len(beta0) == 4
        assert np.all(np.isfinite(beta0))

    def test_interpolation_to_different_size(self):
        init = BetaFieldInitializer()
        init.add_converged_field(np.ones(20))
        init.add_converged_field(np.ones(20) * 1.1)
        init.fit()

        beta0 = init.sample_initial(n_points=50)
        assert len(beta0) == 50

    def test_statistics(self):
        init = BetaFieldInitializer()
        init.add_converged_field(np.ones(10) * 1.2)
        init.add_converged_field(np.ones(10) * 1.1)
        init.fit()

        stats = init.get_statistics()
        assert stats["n_samples"] == 2
        assert stats["mean_deviation_from_unity"] > 0  # mean=1.15, deviation=0.15

    def test_requires_data(self):
        init = BetaFieldInitializer()
        with pytest.raises(RuntimeError, match="No β-fields"):
            init.fit()
