#!/usr/bin/env python3
"""
GEP Explicit Closure — Symbolic Regression for RANS Augmentation
==================================================================
Discovers interpretable algebraic relationships for the Reynolds stress
discrepancy using Gene Expression Programming (symbolic regression).

The key advantage over black-box ML (TBNN, FIML NN):
  - Explicit formula is Galilean-invariant by construction
  - Can be copy-pasted into SU2's SA/SST source code
  - No runtime neural network inference required
  - Physically interpretable and analyzable

Architecture:
    Pope invariants (λ₁...λ₅) → Symbolic Regression → g^(n)(λ)
    b_ij = Σ g^(n)(λ) · T^(n)_ij

References:
    - Zhao et al. (Frontiers in Physics, 2024): GEP for explicit closure
    - Srivastava et al. (AIAA SciTech, 2024): Generalized FIML
    - Bin et al. (TAML, 2024): Constrained recalibration for SST
    - Pope (1975): Tensor basis representation

Usage:
    from scripts.ml_augmentation.gep_explicit_closure import (
        GEPClosureDiscovery, ClosureFormula,
    )
    gep = GEPClosureDiscovery()
    gep.add_training_data(invariants, b_ij_target, tensor_bases)
    formula = gep.discover(n_generations=100, population_size=500)
    print(formula.to_latex())
    print(formula.to_su2_cpp())
"""

import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)
PROJECT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT))

# Optional: gplearn for production symbolic regression
try:
    from gplearn.genetic import SymbolicRegressor
    _HAS_GPLEARN = True
except ImportError:
    _HAS_GPLEARN = False


# =============================================================================
# Symbolic Expression Tree (numpy-only implementation)
# =============================================================================
class ExprNode:
    """
    Node in a symbolic expression tree.

    Operations: +, -, *, /, sqrt, abs, square, sin, cos, const, var
    """

    def __init__(self, op: str, children: List = None,
                 value: float = None, var_idx: int = None):
        self.op = op
        self.children = children or []
        self.value = value
        self.var_idx = var_idx

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate expression tree on input array X (N, n_vars).

        Returns ndarray (N,).
        """
        if self.op == "const":
            return np.full(len(X), self.value)
        elif self.op == "var":
            return X[:, self.var_idx]
        elif self.op == "add":
            return self.children[0].evaluate(X) + self.children[1].evaluate(X)
        elif self.op == "sub":
            return self.children[0].evaluate(X) - self.children[1].evaluate(X)
        elif self.op == "mul":
            return self.children[0].evaluate(X) * self.children[1].evaluate(X)
        elif self.op == "div":
            denom = self.children[1].evaluate(X)
            return self.children[0].evaluate(X) / np.where(
                np.abs(denom) < 1e-10, 1.0, denom)
        elif self.op == "sqrt":
            return np.sqrt(np.abs(self.children[0].evaluate(X)))
        elif self.op == "abs":
            return np.abs(self.children[0].evaluate(X))
        elif self.op == "square":
            return self.children[0].evaluate(X) ** 2
        elif self.op == "neg":
            return -self.children[0].evaluate(X)
        else:
            raise ValueError(f"Unknown op: {self.op}")

    def to_string(self, var_names: List[str] = None) -> str:
        """Convert to human-readable string."""
        if var_names is None:
            var_names = [f"λ{i+1}" for i in range(10)]

        if self.op == "const":
            return f"{self.value:.4g}"
        elif self.op == "var":
            return var_names[self.var_idx]
        elif self.op == "add":
            return f"({self.children[0].to_string(var_names)} + {self.children[1].to_string(var_names)})"
        elif self.op == "sub":
            return f"({self.children[0].to_string(var_names)} - {self.children[1].to_string(var_names)})"
        elif self.op == "mul":
            return f"({self.children[0].to_string(var_names)} * {self.children[1].to_string(var_names)})"
        elif self.op == "div":
            return f"({self.children[0].to_string(var_names)} / {self.children[1].to_string(var_names)})"
        elif self.op == "sqrt":
            return f"sqrt(|{self.children[0].to_string(var_names)}|)"
        elif self.op == "abs":
            return f"|{self.children[0].to_string(var_names)}|"
        elif self.op == "square":
            return f"({self.children[0].to_string(var_names)})²"
        elif self.op == "neg":
            return f"-{self.children[0].to_string(var_names)}"
        else:
            return f"?({self.op})"

    def depth(self) -> int:
        if not self.children:
            return 0
        return 1 + max(c.depth() for c in self.children)

    def complexity(self) -> int:
        """Count total nodes in the expression tree."""
        return 1 + sum(c.complexity() for c in self.children)

    def get_unit_dimensions(self, var_dims: List[int] = None) -> int:
        """
        Propagate dimensional analysis through the expression tree.

        Uses a simplified dimension tag:
          0 = dimensionless
          1 = has dimension (cannot be combined with dim-0 via +/-)

        Parameters
        ----------
        var_dims : list of int, optional
            Dimension tag per input variable. Defaults to all-0 (dimensionless
            Pope invariants).

        Returns
        -------
        int : 0 if dimensionless, 1 if dimensional.
        """
        if var_dims is None:
            var_dims = [0] * 10  # All Pope invariants are dimensionless

        if self.op == "const":
            return 0
        elif self.op == "var":
            idx = self.var_idx if self.var_idx < len(var_dims) else 0
            return var_dims[idx]
        elif self.op in ("add", "sub"):
            d0 = self.children[0].get_unit_dimensions(var_dims)
            d1 = self.children[1].get_unit_dimensions(var_dims)
            # Adding quantities of different dimensions → dimensional mismatch
            if d0 != d1:
                return 1
            return d0
        elif self.op == "mul":
            d0 = self.children[0].get_unit_dimensions(var_dims)
            d1 = self.children[1].get_unit_dimensions(var_dims)
            return max(d0, d1)  # Dim × dim = dim; dim × nodim = dim
        elif self.op == "div":
            d0 = self.children[0].get_unit_dimensions(var_dims)
            d1 = self.children[1].get_unit_dimensions(var_dims)
            return max(d0, d1)
        elif self.op in ("sqrt", "abs", "neg"):
            return self.children[0].get_unit_dimensions(var_dims)
        elif self.op == "square":
            return self.children[0].get_unit_dimensions(var_dims)
        return 1  # Unknown op → conservative

    def is_dimensionless(self, var_dims: List[int] = None) -> bool:
        """Check if the expression is unit-consistent (dimensionless)."""
        return self.get_unit_dimensions(var_dims) == 0

    def uses_only_invariants(self, n_invariants: int = 5) -> bool:
        """Check if tree only references invariant variables (indices 0..n-1)."""
        if self.op == "var":
            return self.var_idx < n_invariants
        return all(c.uses_only_invariants(n_invariants) for c in self.children)


def _random_tree(n_vars: int, max_depth: int, rng: np.random.Generator,
                 terminal_prob: float = 0.3) -> ExprNode:
    """Generate a random expression tree."""
    if max_depth <= 0 or rng.random() < terminal_prob:
        # Terminal node
        if rng.random() < 0.5:
            return ExprNode("var", var_idx=rng.integers(0, n_vars))
        else:
            return ExprNode("const", value=round(rng.uniform(-2, 2), 3))

    # Function node
    ops_unary = ["sqrt", "abs", "square", "neg"]
    ops_binary = ["add", "sub", "mul", "div"]
    all_ops = ops_unary + ops_binary

    op = rng.choice(all_ops)
    if op in ops_unary:
        child = _random_tree(n_vars, max_depth - 1, rng, terminal_prob)
        return ExprNode(op, [child])
    else:
        left = _random_tree(n_vars, max_depth - 1, rng, terminal_prob)
        right = _random_tree(n_vars, max_depth - 1, rng, terminal_prob)
        return ExprNode(op, [left, right])


def _mutate_tree(tree: ExprNode, n_vars: int,
                 rng: np.random.Generator) -> ExprNode:
    """Point mutation on a random node."""
    import copy
    tree = copy.deepcopy(tree)

    # Choose mutation type
    if rng.random() < 0.3:
        # Replace entire subtree
        return _random_tree(n_vars, max_depth=3, rng=rng)
    elif rng.random() < 0.5 and tree.op == "const":
        # Perturb constant
        tree.value += rng.normal(0, 0.5)
        tree.value = round(tree.value, 4)
        return tree
    elif tree.children:
        # Mutate a random child
        idx = rng.integers(0, len(tree.children))
        tree.children[idx] = _mutate_tree(tree.children[idx], n_vars, rng)
    return tree


def _crossover_trees(parent1: ExprNode, parent2: ExprNode,
                     rng: np.random.Generator) -> ExprNode:
    """Subtree crossover."""
    import copy
    child = copy.deepcopy(parent1)

    if child.children and rng.random() < 0.7:
        idx = rng.integers(0, len(child.children))
        # Replace a subtree with one from parent2
        donor = copy.deepcopy(parent2)
        child.children[idx] = donor
    return child


# =============================================================================
# Closure Formula
# =============================================================================
@dataclass
class ClosureFormula:
    """
    Discovered algebraic closure formula.

    Represents g^(n)(λ₁,...,λ₅) coefficients as symbolic expressions.
    """
    coefficient_trees: List[ExprNode]  # g^(1)...g^(n) expressions
    n_bases: int = 10
    fitness: float = float("inf")
    complexity_score: int = 0
    var_names: List[str] = field(
        default_factory=lambda: ["λ₁", "λ₂", "λ₃", "λ₄", "λ₅"])

    def evaluate_coefficients(self, invariants: np.ndarray) -> np.ndarray:
        """
        Evaluate all g^(n) coefficients.

        Parameters
        ----------
        invariants : (N, 5) Pope invariants

        Returns
        -------
        g : (N, n_bases) coefficient values
        """
        N = len(invariants)
        g = np.zeros((N, len(self.coefficient_trees)))
        for n, tree in enumerate(self.coefficient_trees):
            g[:, n] = tree.evaluate(invariants)
        return g

    def predict_anisotropy(self, invariants: np.ndarray,
                           tensor_bases: np.ndarray) -> np.ndarray:
        """
        Predict Reynolds stress anisotropy b_ij.

        b_ij = Σ g^(n)(λ) · T^(n)_ij

        Parameters
        ----------
        invariants : (N, 5)
        tensor_bases : (N, 10, 3, 3)

        Returns
        -------
        b_ij : (N, 3, 3)
        """
        g = self.evaluate_coefficients(invariants)
        n_active = min(g.shape[1], tensor_bases.shape[1])
        b = np.einsum("ni,nijk->njk", g[:, :n_active],
                       tensor_bases[:, :n_active, :, :])
        return b

    def to_latex(self) -> str:
        """Convert formula to LaTeX representation."""
        lines = []
        for n, tree in enumerate(self.coefficient_trees):
            expr = tree.to_string(self.var_names)
            lines.append(f"g^{{({n+1})}} = {expr}")
        return "\n".join(lines)

    def to_su2_cpp(self) -> str:
        """
        Export as SU2-compatible C++ code snippet.

        Generates code that can be inserted into SU2's turbulence
        model source (e.g., CTurbSASolver.cpp or CTurbSSTSolver.cpp).
        """
        lines = [
            "// ============================================================",
            "// GEP-discovered explicit Reynolds stress correction",
            "// Auto-generated by gep_explicit_closure.py",
            "// Insert into CTurbSASolver::Source_Residual or equivalent",
            "// ============================================================",
            "",
            "// Input invariants (already computed in SU2's SA/SST solver):",
            "// lambda1 = trace(S_hat * S_hat);",
            "// lambda2 = trace(Omega_hat * Omega_hat);",
            "// lambda3 = trace(S_hat * S_hat * S_hat);",
            "// lambda4 = trace(Omega_hat * Omega_hat * S_hat);",
            "// lambda5 = trace(Omega_hat * Omega_hat * S_hat * S_hat);",
            "",
        ]

        c_var_names = ["lambda1", "lambda2", "lambda3", "lambda4", "lambda5"]

        for n, tree in enumerate(self.coefficient_trees):
            expr = self._tree_to_cpp(tree, c_var_names)
            lines.append(f"su2double g_{n+1} = {expr};")

        lines.extend([
            "",
            "// Reconstruct anisotropy correction:",
            "// b_ij_correction = sum_n(g_n * T_n_ij);",
            "// Add to Reynolds stress: tau_ij += 2*k * b_ij_correction;",
        ])

        return "\n".join(lines)

    def _tree_to_cpp(self, node: ExprNode,
                     var_names: List[str]) -> str:
        """Convert expression tree to C++ expression string."""
        if node.op == "const":
            return f"{node.value:.6g}"
        elif node.op == "var":
            return var_names[node.var_idx] if node.var_idx < len(var_names) else "0.0"
        elif node.op == "add":
            return f"({self._tree_to_cpp(node.children[0], var_names)} + {self._tree_to_cpp(node.children[1], var_names)})"
        elif node.op == "sub":
            return f"({self._tree_to_cpp(node.children[0], var_names)} - {self._tree_to_cpp(node.children[1], var_names)})"
        elif node.op == "mul":
            return f"({self._tree_to_cpp(node.children[0], var_names)} * {self._tree_to_cpp(node.children[1], var_names)})"
        elif node.op == "div":
            denom = self._tree_to_cpp(node.children[1], var_names)
            return f"({self._tree_to_cpp(node.children[0], var_names)} / max(fabs({denom}), 1e-10))"
        elif node.op == "sqrt":
            return f"sqrt(fabs({self._tree_to_cpp(node.children[0], var_names)}))"
        elif node.op == "abs":
            return f"fabs({self._tree_to_cpp(node.children[0], var_names)})"
        elif node.op == "square":
            inner = self._tree_to_cpp(node.children[0], var_names)
            return f"pow({inner}, 2)"
        elif node.op == "neg":
            return f"(-{self._tree_to_cpp(node.children[0], var_names)})"
        return "0.0"

    def to_dict(self) -> Dict:
        """Serializable summary."""
        return {
            "n_bases": self.n_bases,
            "n_active_coefficients": len(self.coefficient_trees),
            "fitness": float(self.fitness),
            "complexity": self.complexity_score,
            "expressions": {
                f"g{n+1}": tree.to_string(self.var_names)
                for n, tree in enumerate(self.coefficient_trees)
            },
        }


# =============================================================================
# GEP Closure Discovery (built-in symbolic regression)
# =============================================================================
class GEPClosureDiscovery:
    """
    Genetic Expression Programming for discovering explicit closure models.

    Evolves algebraic expressions for the tensor basis coefficients
    g^(n)(λ₁,...,λ₅) using evolutionary symbolic regression.
    """

    def __init__(self, n_bases: int = 3, max_depth: int = 4,
                 parsimony_coefficient: float = 0.01,
                 seed: int = 42):
        """
        Parameters
        ----------
        n_bases : int
            Number of tensor basis coefficients to evolve (default: 3).
            Using 3 instead of 10 reduces search space while capturing
            dominant corrections (g1, g2, g3 are typically most important).
        max_depth : int
            Maximum depth of expression trees.
        parsimony_coefficient : float
            Complexity penalty to favor simpler expressions.
        """
        self.n_bases = n_bases
        self.max_depth = max_depth
        self.parsimony_coefficient = parsimony_coefficient
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self._invariants = None
        self._b_target = None
        self._tensor_bases = None
        self._best_formula = None

    def add_training_data(self, invariants: np.ndarray,
                          b_target: np.ndarray,
                          tensor_bases: np.ndarray):
        """
        Add training data.

        Parameters
        ----------
        invariants : (N, 5) Pope invariants λ₁...λ₅
        b_target : (N, 3, 3) target anisotropy tensors
        tensor_bases : (N, 10, 3, 3) Pope tensor bases
        """
        self._invariants = invariants
        self._b_target = b_target
        self._tensor_bases = tensor_bases

    def discover(self, n_generations: int = 50,
                 population_size: int = 200,
                 tournament_size: int = 5,
                 mutation_rate: float = 0.3,
                 crossover_rate: float = 0.5) -> ClosureFormula:
        """
        Run symbolic regression to discover closure formula.

        Returns
        -------
        ClosureFormula with the best discovered expression.
        """
        if self._invariants is None:
            raise RuntimeError("No training data. Call add_training_data() first.")

        logger.info(
            "Starting GEP discovery: %d bases, pop=%d, gens=%d",
            self.n_bases, population_size, n_generations,
        )

        n_vars = self._invariants.shape[1]
        best_overall = None
        best_fitness = float("inf")

        # Evolve each g^(n) coefficient independently
        best_trees = []

        for basis_idx in range(self.n_bases):
            logger.info("Evolving g^(%d)...", basis_idx + 1)

            # Target: project b_target onto basis n to get scalar target
            # g^(n)_target ≈ (b_target : T^(n)) / (T^(n) : T^(n))
            T_n = self._tensor_bases[:, basis_idx]  # (N, 3, 3)
            TT = np.einsum("nij,nij->n", T_n, T_n) + 1e-10
            g_target = np.einsum("nij,nij->n", self._b_target, T_n) / TT

            # Initialize population
            population = [
                _random_tree(n_vars, self.max_depth, self.rng)
                for _ in range(population_size)
            ]

            for gen in range(n_generations):
                # Evaluate fitness
                fitnesses = []
                for tree in population:
                    try:
                        pred = tree.evaluate(self._invariants)
                        if not np.all(np.isfinite(pred)):
                            fitnesses.append(float("inf"))
                            continue
                        mse = float(np.mean((pred - g_target) ** 2))
                        penalty = self.parsimony_coefficient * tree.complexity()
                        fitnesses.append(mse + penalty)
                    except Exception:
                        fitnesses.append(float("inf"))

                fitnesses = np.array(fitnesses)

                # Track best
                best_idx = np.argmin(fitnesses)
                if fitnesses[best_idx] < best_fitness:
                    import copy
                    best_overall_tree = copy.deepcopy(population[best_idx])

                # Selection + reproduction
                new_population = [
                    population[best_idx],  # Elitism
                ]

                for _ in range(population_size - 1):
                    if self.rng.random() < crossover_rate:
                        p1 = self._tournament_select(
                            population, fitnesses, tournament_size)
                        p2 = self._tournament_select(
                            population, fitnesses, tournament_size)
                        child = _crossover_trees(p1, p2, self.rng)
                    elif self.rng.random() < mutation_rate:
                        parent = self._tournament_select(
                            population, fitnesses, tournament_size)
                        child = _mutate_tree(parent, n_vars, self.rng)
                    else:
                        child = self._tournament_select(
                            population, fitnesses, tournament_size)
                        import copy
                        child = copy.deepcopy(child)

                    # Limit depth
                    if child.depth() > self.max_depth + 2:
                        child = _random_tree(n_vars, self.max_depth, self.rng)

                    new_population.append(child)

                population = new_population

            # Best tree for this basis coefficient
            final_fitnesses = []
            for tree in population:
                try:
                    pred = tree.evaluate(self._invariants)
                    if not np.all(np.isfinite(pred)):
                        final_fitnesses.append(float("inf"))
                        continue
                    mse = float(np.mean((pred - g_target) ** 2))
                    final_fitnesses.append(mse)
                except Exception:
                    final_fitnesses.append(float("inf"))

            import copy
            best_idx = int(np.argmin(final_fitnesses))
            best_trees.append(copy.deepcopy(population[best_idx]))
            logger.info(
                "  g^(%d) best MSE: %.4e, complexity: %d",
                basis_idx + 1, final_fitnesses[best_idx],
                population[best_idx].complexity(),
            )

        # Assemble final formula
        formula = ClosureFormula(
            coefficient_trees=best_trees,
            n_bases=self.n_bases,
        )

        # Evaluate overall fitness
        b_pred = formula.predict_anisotropy(
            self._invariants, self._tensor_bases)
        overall_mse = float(np.mean((b_pred - self._b_target) ** 2))
        total_complexity = sum(t.complexity() for t in best_trees)
        formula.fitness = overall_mse
        formula.complexity_score = total_complexity

        self._best_formula = formula

        logger.info(
            "GEP discovery complete: fitness=%.4e, complexity=%d",
            formula.fitness, total_complexity,
        )

        return formula

    def _tournament_select(self, population: List[ExprNode],
                           fitnesses: np.ndarray,
                           tournament_size: int) -> ExprNode:
        """Tournament selection."""
        indices = self.rng.choice(
            len(population), size=tournament_size, replace=False)
        best = indices[np.argmin(fitnesses[indices])]
        return population[best]


# =============================================================================
# GEP with gplearn Backend (when available)
# =============================================================================
class GEPGPLearnBackend:
    """
    Alternative backend using gplearn's SymbolicRegressor.

    Requires: pip install gplearn
    """

    def __init__(self, population_size: int = 500,
                 generations: int = 50, parsimony: float = 0.01,
                 seed: int = 42):
        if not _HAS_GPLEARN:
            raise ImportError("gplearn not installed: pip install gplearn")

        self.population_size = population_size
        self.generations = generations
        self.parsimony = parsimony
        self.seed = seed
        self.models = []

    def fit(self, invariants: np.ndarray,
            b_target: np.ndarray,
            tensor_bases: np.ndarray,
            n_bases: int = 3) -> List[str]:
        """
        Fit symbolic regressors for each tensor basis coefficient.

        Returns list of discovered formula strings.
        """
        formulas = []

        for n in range(n_bases):
            T_n = tensor_bases[:, n]
            TT = np.einsum("nij,nij->n", T_n, T_n) + 1e-10
            g_target = np.einsum("nij,nij->n", b_target, T_n) / TT

            reg = SymbolicRegressor(
                population_size=self.population_size,
                generations=self.generations,
                parsimony_coefficient=self.parsimony,
                function_set=["add", "sub", "mul", "div", "sqrt", "abs"],
                feature_names=["λ₁", "λ₂", "λ₃", "λ₄", "λ₅"],
                random_state=self.seed + n,
                verbose=0,
            )
            reg.fit(invariants, g_target)
            self.models.append(reg)
            formulas.append(str(reg._program))

            logger.info("g^(%d) = %s", n + 1, formulas[-1])

        return formulas


# =============================================================================
# Transfer Learning Evaluation
# =============================================================================
def evaluate_transfer(
    formula: ClosureFormula,
    train_cases: Dict[str, Dict[str, np.ndarray]],
    test_cases: Dict[str, Dict[str, np.ndarray]],
) -> Dict:
    """
    Evaluate transfer performance of a discovered closure.

    Parameters
    ----------
    formula : ClosureFormula
    train_cases : dict of {case_name: {"invariants": ..., "b_target": ..., "tensor_bases": ...}}
    test_cases : same format

    Returns
    -------
    Dict with per-case and aggregate metrics.
    """
    results = {"train": {}, "test": {}, "transfer_gap": {}}

    for split_name, cases in [("train", train_cases), ("test", test_cases)]:
        for name, data in cases.items():
            b_pred = formula.predict_anisotropy(
                data["invariants"], data["tensor_bases"])
            b_true = data["b_target"]

            mse = float(np.mean((b_pred - b_true) ** 2))
            # Relative error
            denom = float(np.mean(b_true ** 2)) + 1e-10
            rel_err = np.sqrt(mse / denom)

            results[split_name][name] = {
                "MSE": mse,
                "relative_error": float(rel_err),
                "n_points": len(data["invariants"]),
            }

    # Compute transfer gap
    train_mses = [v["MSE"] for v in results["train"].values()]
    test_mses = [v["MSE"] for v in results["test"].values()]
    if train_mses and test_mses:
        results["transfer_gap"] = {
            "train_mean_MSE": float(np.mean(train_mses)),
            "test_mean_MSE": float(np.mean(test_mses)),
            "gap_ratio": float(np.mean(test_mses) / (np.mean(train_mses) + 1e-10)),
        }

    return results


# =============================================================================
# Constrained Recalibration (Bin et al., 2024)
# =============================================================================
def constrained_recalibration(
    formula: ClosureFormula,
    invariants: np.ndarray,
    tensor_bases: np.ndarray,
) -> np.ndarray:
    """
    Apply realizability constraints to the GEP prediction.

    Enforces:
      1. Trace-free: b_kk = 0
      2. Symmetric: b_ij = b_ji
      3. Eigenvalue bounds: -1/3 ≤ eigenvalues ≤ 2/3 (Lumley triangle)

    Returns
    -------
    b_realizable : (N, 3, 3) constrained anisotropy
    """
    b_raw = formula.predict_anisotropy(invariants, tensor_bases)

    N = b_raw.shape[0]
    b_out = np.zeros_like(b_raw)

    for i in range(N):
        b = b_raw[i]

        # 1. Symmetry
        b = 0.5 * (b + b.T)

        # 2. Trace-free
        b -= np.trace(b) / 3.0 * np.eye(3)

        # 3. Iterative eigenvalue clamping + trace-free projection
        eigvals, eigvecs = np.linalg.eigh(b)
        for _ in range(50):  # Iterate until converged
            eigvals_clamped = np.clip(eigvals, -1.0 / 3.0, 2.0 / 3.0)
            eigvals = eigvals_clamped - np.mean(eigvals_clamped)
            if (np.all(eigvals >= -1.0/3.0 - 1e-14)
                    and np.all(eigvals <= 2.0/3.0 + 1e-14)):
                break
        eigvals = np.clip(eigvals, -1.0 / 3.0, 2.0 / 3.0)
        b = eigvecs @ np.diag(eigvals) @ eigvecs.T

        b_out[i] = b

    return b_out


# =============================================================================
# Synthetic Data for Testing
# =============================================================================
def generate_synthetic_closure_data(
    n_points: int = 500,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Generate synthetic closure training data.

    Creates invariants, tensor bases, and target anisotropy
    with a known algebraic relationship for testing.

    Known relationship:
        g1 = -0.1 * λ₁ / (1 + λ₁)
        g2 = 0.05 * λ₂ / (1 + |λ₂|)
        g3 = 0.01 * λ₃

    Returns dict with invariants, tensor_bases, b_target.
    """
    rng = np.random.default_rng(seed)

    # Random invariants
    invariants = rng.standard_normal((n_points, 5))
    invariants[:, 0] = np.abs(invariants[:, 0])  # λ₁ ≥ 0

    # Construct normalized S_hat and O_hat
    S_hat = rng.standard_normal((n_points, 3, 3)) * 0.5
    S_hat = 0.5 * (S_hat + np.swapaxes(S_hat, -2, -1))  # Symmetric
    # Remove trace
    for i in range(n_points):
        S_hat[i] -= np.trace(S_hat[i]) / 3 * np.eye(3)

    O_hat = rng.standard_normal((n_points, 3, 3)) * 0.5
    O_hat = 0.5 * (O_hat - np.swapaxes(O_hat, -2, -1))  # Anti-symmetric

    # Build 10 tensor bases (simplified: use first 3)
    T = np.zeros((n_points, 10, 3, 3))
    T[:, 0] = S_hat  # T(1) = Ŝ
    T[:, 1] = np.einsum("nij,njk->nik", S_hat, O_hat) - np.einsum(
        "nij,njk->nik", O_hat, S_hat)  # T(2) = ŜΩ̂ - Ω̂Ŝ
    T[:, 2] = np.einsum("nij,njk->nik", S_hat, S_hat) - (
        np.einsum("nij,nji->n", S_hat, S_hat)[:, None, None] / 3 * np.eye(3))

    # Known g-coefficients
    l1, l2, l3 = invariants[:, 0], invariants[:, 1], invariants[:, 2]
    g1 = -0.1 * l1 / (1 + l1)
    g2 = 0.05 * l2 / (1 + np.abs(l2))
    g3 = 0.01 * l3

    # Target anisotropy
    b_target = (g1[:, None, None] * T[:, 0]
                + g2[:, None, None] * T[:, 1]
                + g3[:, None, None] * T[:, 2])

    # Add small noise
    b_target += rng.normal(0, 0.001, b_target.shape)

    return {
        "invariants": invariants,
        "tensor_bases": T,
        "b_target": b_target,
        "S_hat": S_hat,
        "O_hat": O_hat,
        "true_g": np.column_stack([g1, g2, g3]),
    }


# =============================================================================
# CLI
# =============================================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="GEP Explicit Closure Discovery")
    parser.add_argument("--n-bases", type=int, default=3)
    parser.add_argument("--generations", type=int, default=30)
    parser.add_argument("--population", type=int, default=100)
    parser.add_argument("--n-points", type=int, default=300)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    print("=== GEP Explicit Closure Discovery ===\n")

    # Generate data
    data = generate_synthetic_closure_data(n_points=args.n_points)

    # Discover
    gep = GEPClosureDiscovery(n_bases=args.n_bases, max_depth=4)
    gep.add_training_data(
        data["invariants"], data["b_target"], data["tensor_bases"])

    formula = gep.discover(
        n_generations=args.generations,
        population_size=args.population,
    )

    # Print results
    print(f"\nDiscovered formula (fitness={formula.fitness:.4e}):")
    print(formula.to_latex())
    print(f"\nComplexity: {formula.complexity_score} nodes")

    # Show C++ export
    print("\n--- SU2 C++ Export ---")
    print(formula.to_su2_cpp())

    # Show summary
    print("\n--- Summary ---")
    print(json.dumps(formula.to_dict(), indent=2))


if __name__ == "__main__":
    main()
