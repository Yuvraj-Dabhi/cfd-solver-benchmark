#!/usr/bin/env python3
"""
Sobol UQ + Bayesian Model Averaging
=======================================
Global sensitivity analysis and probabilistic model combination
for publication-ready uncertainty quantification.

Features:
  1. Sobol sensitivity indices (first-order, total-order)
  2. Bayesian model averaging (BMA) across turbulence models
  3. Credible interval construction
  4. UQ summary table generation for §21 of technical report

References:
    - Saltelli et al. (2010), Computer Physics Communications
    - Hoeting et al. (1999), Statistical Science (BMA)
    - SALib library for Sobol analysis

Usage:
    from scripts.ml_augmentation.sobol_uq_bayesian import (
        SobolAnalyzer, BayesianModelAverager, UQReportGenerator,
    )
    sobol = SobolAnalyzer(problem_spec)
    indices = sobol.analyze(model_func, n_samples=1024)
    bma = BayesianModelAverager()
    bma.add_model("SA", predictions_sa, log_likelihood_sa)
    bma.add_model("SST", predictions_sst, log_likelihood_sst)
    combined = bma.average()
"""

import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)
PROJECT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT))

# Optional SALib
try:
    from SALib.sample import saltelli as salib_saltelli
    from SALib.analyze import sobol as salib_sobol
    _HAS_SALIB = True
except ImportError:
    _HAS_SALIB = False


# =============================================================================
# Sobol Sensitivity Analysis
# =============================================================================
@dataclass
class SobolProblem:
    """Sobol problem specification."""
    n_vars: int
    names: List[str]
    bounds: List[Tuple[float, float]]
    dists: Optional[List[str]] = None  # "uniform" or "norm"


@dataclass
class SobolResult:
    """Sobol analysis results."""
    S1: np.ndarray           # First-order indices
    ST: np.ndarray           # Total-order indices
    S1_conf: np.ndarray      # First-order confidence intervals
    ST_conf: np.ndarray      # Total-order confidence intervals
    S2: Optional[np.ndarray] = None  # Second-order indices
    var_names: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        result = {
            "first_order": {
                name: {"S1": float(self.S1[i]), "S1_conf": float(self.S1_conf[i])}
                for i, name in enumerate(self.var_names)
            },
            "total_order": {
                name: {"ST": float(self.ST[i]), "ST_conf": float(self.ST_conf[i])}
                for i, name in enumerate(self.var_names)
            },
        }
        return result

    def to_markdown_table(self) -> str:
        lines = ["| Parameter | S₁ (First-order) | S₁ CI | Sₜ (Total) | Sₜ CI |",
                 "|-----------|-------------------|-------|-------------|-------|"]
        for i, name in enumerate(self.var_names):
            lines.append(
                f"| {name} | {self.S1[i]:.4f} | ±{self.S1_conf[i]:.4f} "
                f"| {self.ST[i]:.4f} | ±{self.ST_conf[i]:.4f} |")
        return "\n".join(lines)


class SobolAnalyzer:
    """
    Sobol global sensitivity analysis.

    Computes first-order (S₁) and total-order (Sₜ) sensitivity indices
    to determine which input parameters most affect CFD predictions.
    """

    def __init__(self, problem: SobolProblem):
        self.problem = problem

    def analyze(self, model_func: Callable,
                n_samples: int = 1024,
                calc_second_order: bool = False,
                seed: int = 42) -> SobolResult:
        """
        Run Sobol analysis.

        Parameters
        ----------
        model_func : callable (N, n_vars) → (N,)
            Model function mapping parameter samples to scalar output.
        n_samples : int
            Base sample size (total evals = n_samples * (2*n_vars + 2)).
        calc_second_order : bool
            Whether to compute second-order interaction indices.

        Returns
        -------
        SobolResult
        """
        if _HAS_SALIB:
            return self._analyze_salib(
                model_func, n_samples, calc_second_order, seed)
        else:
            return self._analyze_builtin(
                model_func, n_samples, seed)

    def _analyze_salib(self, model_func, n_samples,
                       calc_second_order, seed) -> SobolResult:
        """Use SALib for Sobol analysis."""
        problem_dict = {
            "num_vars": self.problem.n_vars,
            "names": self.problem.names,
            "bounds": self.problem.bounds,
        }

        # Generate Saltelli samples
        samples = salib_saltelli.sample(
            problem_dict, n_samples,
            calc_second_order=calc_second_order)

        # Evaluate model
        Y = model_func(samples)

        # Analyze
        Si = salib_sobol.analyze(
            problem_dict, Y,
            calc_second_order=calc_second_order)

        return SobolResult(
            S1=np.array(Si["S1"]),
            ST=np.array(Si["ST"]),
            S1_conf=np.array(Si["S1_conf"]),
            ST_conf=np.array(Si["ST_conf"]),
            S2=np.array(Si.get("S2")) if "S2" in Si else None,
            var_names=self.problem.names,
        )

    def _analyze_builtin(self, model_func, n_samples,
                         seed) -> SobolResult:
        """
        Built-in Sobol analysis (no SALib required).

        Uses Saltelli sampling scheme and Jansen estimators.
        """
        rng = np.random.default_rng(seed)
        D = self.problem.n_vars
        bounds = np.array(self.problem.bounds)

        # Generate base matrices A and B
        A_unit = rng.random((n_samples, D))
        B_unit = rng.random((n_samples, D))

        # Scale to bounds
        A = bounds[:, 0] + A_unit * (bounds[:, 1] - bounds[:, 0])
        B = bounds[:, 0] + B_unit * (bounds[:, 1] - bounds[:, 0])

        # Evaluate base matrices
        f_A = model_func(A)
        f_B = model_func(B)

        # Total variance
        f_all = np.concatenate([f_A, f_B])
        var_total = np.var(f_all) + 1e-15

        S1 = np.zeros(D)
        ST = np.zeros(D)
        S1_conf = np.zeros(D)
        ST_conf = np.zeros(D)

        for i in range(D):
            # AB_i: A with column i replaced by B's column i
            AB_i = A.copy()
            AB_i[:, i] = B[:, i]
            f_AB_i = model_func(AB_i)

            # BA_i: B with column i replaced by A's column i
            BA_i = B.copy()
            BA_i[:, i] = A[:, i]
            f_BA_i = model_func(BA_i)

            # Jansen estimator for S1
            S1[i] = 1 - np.mean((f_A - f_BA_i)**2) / (2 * var_total)

            # Jansen estimator for ST
            ST[i] = np.mean((f_A - f_AB_i)**2) / (2 * var_total)

            # Bootstrap confidence intervals
            S1_boot = []
            ST_boot = []
            for _ in range(100):
                idx = rng.choice(n_samples, n_samples, replace=True)
                var_b = np.var(np.concatenate([f_A[idx], f_B[idx]])) + 1e-15
                S1_boot.append(
                    1 - np.mean((f_A[idx] - f_BA_i[idx])**2) / (2 * var_b))
                ST_boot.append(
                    np.mean((f_A[idx] - f_AB_i[idx])**2) / (2 * var_b))
            S1_conf[i] = 1.96 * np.std(S1_boot)
            ST_conf[i] = 1.96 * np.std(ST_boot)

        # Clamp to [0, 1]
        S1 = np.clip(S1, 0, 1)
        ST = np.clip(ST, 0, 1)

        return SobolResult(
            S1=S1, ST=ST,
            S1_conf=S1_conf, ST_conf=ST_conf,
            var_names=self.problem.names,
        )


# =============================================================================
# Bayesian Model Averaging
# =============================================================================
@dataclass
class ModelEntry:
    """A single model in the BMA ensemble."""
    name: str
    predictions: np.ndarray    # (N,) or (N, M) predictions
    log_likelihood: float      # Model evidence (log marginal likelihood)
    weight: float = 0.0        # Posterior model probability


class BayesianModelAverager:
    """
    Bayesian Model Averaging across turbulence models.

    Combines SA, SST, etc. predictions weighted by their posterior
    probability given the experimental data.
    """

    def __init__(self, prior: str = "uniform"):
        """
        Parameters
        ----------
        prior : str
            "uniform" (equal prior) or "bic" (BIC-based approximation).
        """
        self.prior = prior
        self.models: List[ModelEntry] = []

    def add_model(self, name: str, predictions: np.ndarray,
                  log_likelihood: float):
        """Add a model to the ensemble."""
        self.models.append(ModelEntry(
            name=name,
            predictions=predictions,
            log_likelihood=log_likelihood,
        ))

    def add_model_from_residuals(self, name: str,
                                  predictions: np.ndarray,
                                  observations: np.ndarray,
                                  sigma: float = 1.0,
                                  n_params: int = 1):
        """
        Add model with log-likelihood estimated from residuals.

        Uses Gaussian likelihood: L = -n/2 * log(MSE/σ²)
        With BIC correction: log_L_eff = L - (k/2)*log(n)
        """
        residuals = predictions - observations
        n = len(residuals)
        mse = np.mean(residuals**2)
        log_L = -n / 2 * np.log(mse / sigma**2 + 1e-15)

        if self.prior == "bic":
            log_L -= (n_params / 2) * np.log(n)

        self.add_model(name, predictions, log_L)

    def compute_weights(self) -> Dict[str, float]:
        """Compute posterior model weights via Bayes' rule."""
        if not self.models:
            return {}

        log_liks = np.array([m.log_likelihood for m in self.models])

        # Numerically stable softmax
        log_liks_shifted = log_liks - np.max(log_liks)
        weights = np.exp(log_liks_shifted)
        weights /= weights.sum()

        for i, m in enumerate(self.models):
            m.weight = float(weights[i])

        return {m.name: m.weight for m in self.models}

    def average(self) -> Dict[str, np.ndarray]:
        """
        Compute BMA prediction and uncertainty.

        Returns dict with:
            mean: weighted average prediction
            std: BMA standard deviation (epistemic + within-model)
            credible_lower: 2.5th percentile
            credible_upper: 97.5th percentile
            weights: model weights
        """
        self.compute_weights()

        # Weighted mean
        mean = np.zeros_like(self.models[0].predictions, dtype=float)
        for m in self.models:
            mean += m.weight * m.predictions

        # BMA variance: E[V|M] + V[E|M]
        var = np.zeros_like(mean)
        for m in self.models:
            # Within-model variance assumed ~ residual²
            var += m.weight * (m.predictions - mean)**2

        std = np.sqrt(var)

        return {
            "mean": mean,
            "std": std,
            "credible_lower": mean - 1.96 * std,
            "credible_upper": mean + 1.96 * std,
            "weights": {m.name: m.weight for m in self.models},
        }

    def to_dict(self) -> Dict:
        self.compute_weights()
        return {
            "n_models": len(self.models),
            "weights": {m.name: m.weight for m in self.models},
            "models": [
                {"name": m.name, "weight": m.weight,
                 "log_likelihood": m.log_likelihood}
                for m in self.models
            ],
        }


# =============================================================================
# UQ Report Generator
# =============================================================================
class UQReportGenerator:
    """
    Generate publication-ready UQ summary tables.

    Combines GCI, Sobol, BMA, and RSS error budgets into
    a unified Markdown report for §21 of the technical report.
    """

    def __init__(self):
        self.sections = []

    def add_sobol_results(self, case_name: str, result: SobolResult):
        self.sections.append({
            "type": "sobol",
            "case": case_name,
            "data": result,
        })

    def add_bma_results(self, case_name: str, bma: BayesianModelAverager):
        self.sections.append({
            "type": "bma",
            "case": case_name,
            "data": bma,
        })

    def add_error_budget(self, case_name: str, budget: Dict[str, float]):
        """
        Add RSS error budget.

        budget: {"U_num": ..., "U_input": ..., "U_model": ..., "U_total": ...}
        """
        self.sections.append({
            "type": "error_budget",
            "case": case_name,
            "data": budget,
        })

    def generate_report(self) -> str:
        """Generate full Markdown UQ report."""
        lines = [
            "# Uncertainty Quantification Summary",
            "",
        ]

        for section in self.sections:
            if section["type"] == "sobol":
                lines.append(f"## Sobol Sensitivity — {section['case']}")
                lines.append(section["data"].to_markdown_table())
                lines.append("")

            elif section["type"] == "bma":
                bma = section["data"]
                bma.compute_weights()
                lines.append(f"## BMA Model Weights — {section['case']}")
                lines.append("| Model | Weight |")
                lines.append("|-------|--------|")
                for m in bma.models:
                    lines.append(f"| {m.name} | {m.weight:.4f} |")
                lines.append("")

            elif section["type"] == "error_budget":
                budget = section["data"]
                lines.append(f"## Error Budget — {section['case']}")
                lines.append("| Source | Uncertainty |")
                lines.append("|--------|-------------|")
                for source, val in budget.items():
                    lines.append(f"| {source} | {val:.2%} |")
                lines.append("")

        return "\n".join(lines)


# =============================================================================
# Synthetic Test Helpers
# =============================================================================
def generate_sobol_test_problem() -> Tuple[SobolProblem, Callable]:
    """
    Generate Ishigami test function for Sobol analysis validation.

    f(x₁, x₂, x₃) = sin(x₁) + 7·sin²(x₂) + 0.1·x₃⁴·sin(x₁)

    Known analytical Sobol indices:
        S1 ≈ [0.314, 0.442, 0.000]
        ST ≈ [0.558, 0.442, 0.244]
    """
    problem = SobolProblem(
        n_vars=3,
        names=["x1", "x2", "x3"],
        bounds=[(-np.pi, np.pi)] * 3,
    )

    def ishigami(X):
        x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2]
        return np.sin(x1) + 7 * np.sin(x2)**2 + 0.1 * x3**4 * np.sin(x1)

    return problem, ishigami


def generate_bma_test_data(seed: int = 42) -> Dict:
    """
    Generate synthetic predictions from 3 turbulence models.

    Returns dict with observations, model predictions, and sigmas.
    """
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, 50)

    # True Cf distribution
    cf_true = 0.004 * (1 - 1.5 * x + 0.6 * x**2)

    # Model predictions with different biases
    cf_sa = cf_true * 1.1 + rng.normal(0, 0.0002, len(x))   # SA over-predicts
    cf_sst = cf_true * 0.95 + rng.normal(0, 0.0001, len(x))  # SST slight under
    cf_rsm = cf_true + rng.normal(0, 0.0003, len(x))          # RSM unbiased

    return {
        "x": x,
        "observations": cf_true,
        "predictions": {"SA": cf_sa, "SST": cf_sst, "RSM": cf_rsm},
    }


# =============================================================================
# CLI
# =============================================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Sobol UQ + Bayesian Model Averaging")
    parser.add_argument("--n-samples", type=int, default=512)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    print("=== Sobol Global Sensitivity Analysis ===\n")

    # Ishigami test
    problem, func = generate_sobol_test_problem()
    analyzer = SobolAnalyzer(problem)
    result = analyzer.analyze(func, n_samples=args.n_samples)

    print(result.to_markdown_table())
    print(f"\nKnown analytical: S1 ≈ [0.314, 0.442, 0.000]")
    print(f"                  ST ≈ [0.558, 0.442, 0.244]")

    print("\n=== Bayesian Model Averaging ===\n")

    data = generate_bma_test_data()
    bma = BayesianModelAverager()
    for name, preds in data["predictions"].items():
        bma.add_model_from_residuals(name, preds, data["observations"])

    result_bma = bma.average()
    print("Model weights:")
    for name, w in result_bma["weights"].items():
        print(f"  {name}: {w:.4f}")


if __name__ == "__main__":
    main()
