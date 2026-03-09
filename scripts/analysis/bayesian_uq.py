#!/usr/bin/env python3
"""
Bayesian / Probabilistic UQ Upgrades
=======================================
Extends the UQ pipeline beyond RSS combination (ASME V&V 20) with:

1. **Bayesian Model Averaging** — weighted ensemble of turbulence model
   predictions with posterior credible intervals.
2. **Active Subspace Detection** — eigendecomposition of the gradient
   outer-product matrix for dimensionality reduction in high-dimensional
   parameter spaces.
3. **Aleatoric/Epistemic Decomposition** — separates reducible (epistemic)
   from irreducible (aleatoric) uncertainty using ensemble variance and
   data noise estimation.
4. **OAT vs Sobol Comparison** — quantifies interaction effects missed by
   one-at-a-time sensitivity analysis.
5. **Probabilistic Validation Metrics** — CRPS, ELPD, calibration error,
   and sharpness for proper probabilistic scoring.

References
----------
  - Levine & McKeon (2023), Phys. Fluids: probabilistic data-driven closures
  - Lakshminarayanan et al. (2017): Deep Ensembles for epistemic UQ
  - Constantine (2015): Active Subspaces — Emerging Ideas for Dimension
    Reduction in Parameter Studies
  - Gneiting & Raftery (2007): Strictly proper scoring rules
"""

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


# ============================================================================
# Data Structures
# ============================================================================
@dataclass
class BMAResult:
    """Result from Bayesian Model Averaging."""
    model_names: List[str] = field(default_factory=list)
    weights: np.ndarray = field(default_factory=lambda: np.array([]))
    weighted_mean: float = 0.0
    weighted_std: float = 0.0
    credible_interval_95: Tuple[float, float] = (0.0, 0.0)
    credible_interval_99: Tuple[float, float] = (0.0, 0.0)
    individual_predictions: np.ndarray = field(
        default_factory=lambda: np.array([])
    )
    individual_errors: np.ndarray = field(
        default_factory=lambda: np.array([])
    )


@dataclass
class ActiveSubspaceResult:
    """Result from active subspace analysis."""
    eigenvalues: np.ndarray = field(default_factory=lambda: np.array([]))
    eigenvectors: np.ndarray = field(default_factory=lambda: np.array([]))
    subspace_dim: int = 1
    spectral_gap: float = 0.0
    activity_scores: np.ndarray = field(default_factory=lambda: np.array([]))
    parameter_names: List[str] = field(default_factory=list)
    explained_variance_ratio: np.ndarray = field(
        default_factory=lambda: np.array([])
    )


@dataclass
class UncertaintyDecomposition:
    """Aleatoric/Epistemic uncertainty decomposition."""
    total_variance: float = 0.0
    epistemic_variance: float = 0.0
    aleatoric_variance: float = 0.0
    epistemic_fraction: float = 0.0
    aleatoric_fraction: float = 0.0
    # Per-point decomposition (if available)
    epistemic_per_point: np.ndarray = field(
        default_factory=lambda: np.array([])
    )
    aleatoric_per_point: np.ndarray = field(
        default_factory=lambda: np.array([])
    )


@dataclass
class SensitivityComparison:
    """OAT vs Sobol sensitivity comparison."""
    parameter_names: List[str] = field(default_factory=list)
    oat_sensitivities: np.ndarray = field(
        default_factory=lambda: np.array([])
    )
    sobol_S1: np.ndarray = field(default_factory=lambda: np.array([]))
    sobol_ST: np.ndarray = field(default_factory=lambda: np.array([]))
    oat_ranking: List[str] = field(default_factory=list)
    sobol_ranking: List[str] = field(default_factory=list)
    rank_correlation: float = 0.0
    interaction_index: float = 0.0


@dataclass
class ProbabilisticScores:
    """Probabilistic validation metric scores."""
    crps: float = 0.0
    elpd: float = 0.0
    calibration_error: float = 0.0
    sharpness: float = 0.0
    coverage_90: float = 0.0
    coverage_95: float = 0.0


# ============================================================================
# 1. Bayesian Model Averaging
# ============================================================================
class BayesianModelAveraging:
    """
    Bayesian Model Averaging over turbulence model predictions.

    Weights models by inverse validation error:
        w_i ∝ exp(-|E_i|² / 2σ²)

    Posterior predictive:
        p(y|D) = Σ w_i · p(y|M_i, D)

    Usage:
        bma = BayesianModelAveraging()
        bma.add_model("SA", prediction=0.65, error=0.05)
        bma.add_model("SST", prediction=0.70, error=0.02)
        result = bma.compute()
    """

    def __init__(self, sigma: float = 0.1):
        """
        Parameters
        ----------
        sigma : float
            Scale parameter for weighting. Smaller σ → sharper weights.
        """
        self.sigma = sigma
        self._models: List[Dict] = []

    def add_model(
        self,
        name: str,
        prediction: float,
        error: float,
        uncertainty: float = 0.0,
    ):
        """
        Add a turbulence model prediction.

        Parameters
        ----------
        name : str
            Model name (e.g. "SA", "SST", "k-epsilon").
        prediction : float
            Model prediction for the QoI.
        error : float
            Absolute validation error |pred - exp|.
        uncertainty : float
            Model's own uncertainty estimate (if available).
        """
        self._models.append({
            "name": name,
            "prediction": prediction,
            "error": error,
            "uncertainty": uncertainty,
        })

    def compute(self) -> BMAResult:
        """
        Compute BMA weighted prediction and credible intervals.

        Returns
        -------
        BMAResult
        """
        if not self._models:
            raise ValueError("No models added. Call add_model() first.")

        names = [m["name"] for m in self._models]
        preds = np.array([m["prediction"] for m in self._models])
        errors = np.array([m["error"] for m in self._models])
        uncerts = np.array([m["uncertainty"] for m in self._models])

        # Compute weights: w_i ∝ exp(-E_i² / 2σ²)
        log_weights = -(errors ** 2) / (2 * self.sigma ** 2)
        log_weights -= np.max(log_weights)  # Numerical stability
        weights = np.exp(log_weights)
        weights /= weights.sum()

        # Weighted mean
        w_mean = float(np.sum(weights * preds))

        # Weighted variance: within-model + between-model
        var_within = float(np.sum(weights * uncerts ** 2))
        var_between = float(np.sum(weights * (preds - w_mean) ** 2))
        w_std = float(np.sqrt(var_within + var_between))

        # Credible intervals (Gaussian approximation)
        ci_95 = (w_mean - 1.96 * w_std, w_mean + 1.96 * w_std)
        ci_99 = (w_mean - 2.576 * w_std, w_mean + 2.576 * w_std)

        return BMAResult(
            model_names=names,
            weights=weights,
            weighted_mean=w_mean,
            weighted_std=w_std,
            credible_interval_95=ci_95,
            credible_interval_99=ci_99,
            individual_predictions=preds,
            individual_errors=errors,
        )

    def compute_distribution(
        self, n_samples: int = 10000, seed: int = 42
    ) -> np.ndarray:
        """
        Sample from the BMA posterior predictive distribution.

        Returns
        -------
        samples : ndarray (n_samples,)
        """
        rng = np.random.default_rng(seed)
        result = self.compute()

        samples = []
        for _ in range(n_samples):
            # Select model proportional to weight
            idx = rng.choice(len(result.weights), p=result.weights)
            pred = result.individual_predictions[idx]
            # Sample from model's predictive distribution
            unc = max(
                self._models[idx]["uncertainty"],
                result.weighted_std * 0.1,  # Minimum spread
            )
            samples.append(rng.normal(pred, unc))

        return np.array(samples)


# ============================================================================
# 2. Active Subspace Detection
# ============================================================================
class ActiveSubspace:
    """
    Active subspace analysis for dimensionality reduction in UQ.

    Identifies low-dimensional directions in parameter space that
    capture most of the variability in the model output.

    Algorithm:
        1. Estimate gradient ∇f at M random parameter points
        2. Form C = (1/M) Σ ∇f_i ∇f_iᵀ
        3. Eigendecompose C = W Λ Wᵀ
        4. Subspace dimension from spectral gap
    """

    def __init__(
        self,
        parameter_names: Optional[List[str]] = None,
        parameter_bounds: Optional[np.ndarray] = None,
    ):
        self.parameter_names = parameter_names or []
        self.parameter_bounds = parameter_bounds

    def compute(
        self,
        model_func: Callable[[np.ndarray], float],
        n_samples: int = 100,
        n_params: Optional[int] = None,
        seed: int = 42,
    ) -> ActiveSubspaceResult:
        """
        Compute active subspace via gradient sampling.

        Parameters
        ----------
        model_func : callable
            Function f(x) → scalar, where x is (n_params,).
        n_samples : int
            Number of gradient samples.
        n_params : int, optional
            Parameter dimension (inferred from bounds if not given).
        seed : int
            Random seed.

        Returns
        -------
        ActiveSubspaceResult
        """
        rng = np.random.default_rng(seed)

        if n_params is None:
            if self.parameter_bounds is not None:
                n_params = len(self.parameter_bounds)
            else:
                n_params = len(self.parameter_names) if self.parameter_names else 4

        if not self.parameter_names:
            self.parameter_names = [f"x{i}" for i in range(n_params)]

        # Sample parameter points
        if self.parameter_bounds is not None:
            bounds = np.array(self.parameter_bounds)
            X = rng.uniform(bounds[:, 0], bounds[:, 1], size=(n_samples, n_params))
        else:
            X = rng.uniform(-1, 1, size=(n_samples, n_params))

        # Estimate gradients via finite differences
        eps = 1e-5
        gradients = np.zeros((n_samples, n_params))
        for i in range(n_samples):
            f0 = model_func(X[i])
            for j in range(n_params):
                x_plus = X[i].copy()
                x_plus[j] += eps
                gradients[i, j] = (model_func(x_plus) - f0) / eps

        # Form gradient outer product matrix C = (1/M) Σ ∇f ∇fᵀ
        C = gradients.T @ gradients / n_samples

        # Eigendecompose
        eigenvalues, eigenvectors = np.linalg.eigh(C)
        # Sort descending
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Non-negative eigenvalues
        eigenvalues = np.maximum(eigenvalues, 0)

        # Explained variance ratio
        total = eigenvalues.sum()
        evr = eigenvalues / max(total, 1e-15)

        # Spectral gap — find largest gap between consecutive eigenvalues
        if len(eigenvalues) > 1:
            gaps = eigenvalues[:-1] / np.maximum(eigenvalues[1:], 1e-15)
            spectral_gap = float(np.max(gaps))
            subspace_dim = int(np.argmax(gaps)) + 1
        else:
            spectral_gap = 1.0
            subspace_dim = 1

        # Activity scores: diagonal of eigenvector matrix squared,
        # weighted by eigenvalues
        activity_scores = np.sum(
            eigenvectors ** 2 * eigenvalues[np.newaxis, :], axis=1
        )
        activity_scores /= max(activity_scores.sum(), 1e-15)

        return ActiveSubspaceResult(
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            subspace_dim=subspace_dim,
            spectral_gap=spectral_gap,
            activity_scores=activity_scores,
            parameter_names=self.parameter_names[:n_params],
            explained_variance_ratio=evr,
        )


# ============================================================================
# 3. Aleatoric / Epistemic Decomposition
# ============================================================================
class UncertaintyDecomposer:
    """
    Decomposes total uncertainty into epistemic and aleatoric components.

    - Epistemic (reducible): disagreement between ensemble members
    - Aleatoric (irreducible): inherent noise / model-form error

    Uses the deep ensemble decomposition:
        σ²_epistemic = Var[μ_i]  (variance of member means)
        σ²_aleatoric = E[σ²_i]  (mean of member variances, or noise floor)
    """

    @staticmethod
    def decompose_ensemble(
        predictions: np.ndarray,
        individual_variances: Optional[np.ndarray] = None,
        noise_floor: float = 0.0,
    ) -> UncertaintyDecomposition:
        """
        Decompose uncertainty from ensemble predictions.

        Parameters
        ----------
        predictions : ndarray (n_members, n_points)
            Predictions from each ensemble member.
        individual_variances : ndarray, optional (n_members, n_points)
            Per-member predictive variances (heteroscedastic NLL).
        noise_floor : float
            Minimum aleatoric variance estimate.

        Returns
        -------
        UncertaintyDecomposition
        """
        n_members = predictions.shape[0]

        # Epistemic: variance of member means
        member_means = predictions  # (n_members, n_points)
        ensemble_mean = np.mean(member_means, axis=0)
        epistemic_per_point = np.var(member_means, axis=0)

        # Aleatoric: mean of member variances (or noise floor)
        if individual_variances is not None:
            aleatoric_per_point = np.mean(individual_variances, axis=0)
        else:
            aleatoric_per_point = np.full_like(
                epistemic_per_point, noise_floor
            )
        aleatoric_per_point = np.maximum(aleatoric_per_point, noise_floor)

        # Aggregate
        epistemic_var = float(np.mean(epistemic_per_point))
        aleatoric_var = float(np.mean(aleatoric_per_point))
        total_var = epistemic_var + aleatoric_var

        return UncertaintyDecomposition(
            total_variance=total_var,
            epistemic_variance=epistemic_var,
            aleatoric_variance=aleatoric_var,
            epistemic_fraction=epistemic_var / max(total_var, 1e-15),
            aleatoric_fraction=aleatoric_var / max(total_var, 1e-15),
            epistemic_per_point=epistemic_per_point,
            aleatoric_per_point=aleatoric_per_point,
        )

    @staticmethod
    def estimate_noise_floor(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        window: int = 5,
    ) -> float:
        """
        Estimate aleatoric noise floor from residuals.

        Uses local variance of residuals as noise estimate.
        """
        residuals = y_true - y_pred
        if len(residuals) < window:
            return float(np.var(residuals))

        # Rolling variance
        local_vars = []
        for i in range(len(residuals) - window + 1):
            local_vars.append(np.var(residuals[i:i + window]))

        return float(np.median(local_vars))


# ============================================================================
# 4. OAT vs Sobol Sensitivity Comparison
# ============================================================================
class SensitivityComparator:
    """
    Compares One-At-a-Time (OAT) and Sobol global sensitivity analysis.

    OAT: perturb each parameter ±Δ while holding others at baseline
    Sobol: variance-based decomposition via Saltelli sampling

    The interaction index I = 1 - Σ S1_i / Σ ST_i quantifies
    how much the OAT approach misses parameter interactions.
    """

    @staticmethod
    def oat_sensitivity(
        model_func: Callable[[np.ndarray], float],
        baseline: np.ndarray,
        perturbation: float = 0.10,
        parameter_names: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Compute OAT sensitivities.

        Parameters
        ----------
        model_func : callable
            f(x) → scalar.
        baseline : ndarray
            Baseline parameter values.
        perturbation : float
            Fractional perturbation (e.g. 0.10 for ±10%).
        parameter_names : list, optional

        Returns
        -------
        sensitivities : ndarray
            Normalized sensitivity |Δf/f₀| / (Δx/x₀) for each parameter.
        names : list
        """
        n_params = len(baseline)
        if parameter_names is None:
            parameter_names = [f"x{i}" for i in range(n_params)]

        f_base = model_func(baseline)
        sensitivities = np.zeros(n_params)

        for i in range(n_params):
            x_plus = baseline.copy()
            x_minus = baseline.copy()
            dx = max(abs(baseline[i]) * perturbation, 1e-8)
            x_plus[i] += dx
            x_minus[i] -= dx

            f_plus = model_func(x_plus)
            f_minus = model_func(x_minus)

            # Central difference sensitivity
            df = (f_plus - f_minus) / (2 * dx)
            # Normalize: elasticity = (x/f) * df/dx
            if abs(f_base) > 1e-15 and abs(baseline[i]) > 1e-15:
                sensitivities[i] = abs(df * baseline[i] / f_base)
            else:
                sensitivities[i] = abs(df)

        return sensitivities, parameter_names

    @staticmethod
    def compare(
        model_func: Callable[[np.ndarray], float],
        baseline: np.ndarray,
        parameter_names: Optional[List[str]] = None,
        perturbation: float = 0.10,
        sobol_S1: Optional[np.ndarray] = None,
        sobol_ST: Optional[np.ndarray] = None,
    ) -> SensitivityComparison:
        """
        Compare OAT and Sobol sensitivity rankings.

        Parameters
        ----------
        model_func : callable
        baseline : ndarray
        sobol_S1, sobol_ST : ndarray, optional
            Pre-computed Sobol indices. If None, approximate via
            perturbation-based variance decomposition.

        Returns
        -------
        SensitivityComparison
        """
        n_params = len(baseline)
        if parameter_names is None:
            parameter_names = [f"x{i}" for i in range(n_params)]

        # OAT
        oat_sens, _ = SensitivityComparator.oat_sensitivity(
            model_func, baseline, perturbation, parameter_names
        )

        # If Sobol not provided, approximate
        if sobol_S1 is None or sobol_ST is None:
            sobol_S1, sobol_ST = SensitivityComparator._approximate_sobol(
                model_func, baseline, perturbation
            )

        # Rankings
        oat_rank = [
            parameter_names[i]
            for i in np.argsort(oat_sens)[::-1]
        ]
        sobol_rank = [
            parameter_names[i]
            for i in np.argsort(sobol_ST)[::-1]
        ]

        # Rank correlation (Spearman)
        from scipy.stats import spearmanr
        rank_corr, _ = spearmanr(
            np.argsort(np.argsort(oat_sens)[::-1]),
            np.argsort(np.argsort(sobol_ST)[::-1]),
        )

        # Interaction index: I = 1 - Σ S1 / Σ ST
        sum_s1 = np.sum(sobol_S1)
        sum_st = np.sum(sobol_ST)
        interaction = 1.0 - sum_s1 / max(sum_st, 1e-15)

        return SensitivityComparison(
            parameter_names=parameter_names,
            oat_sensitivities=oat_sens,
            sobol_S1=sobol_S1,
            sobol_ST=sobol_ST,
            oat_ranking=oat_rank,
            sobol_ranking=sobol_rank,
            rank_correlation=float(rank_corr),
            interaction_index=float(interaction),
        )

    @staticmethod
    def _approximate_sobol(
        model_func: Callable[[np.ndarray], float],
        baseline: np.ndarray,
        perturbation: float = 0.10,
        n_samples: int = 200,
        seed: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Approximate Sobol indices when SALib is not available.

        Uses random perturbation-based variance decomposition.
        """
        rng = np.random.default_rng(seed)
        n_params = len(baseline)

        # Total variance from random samples
        X = np.tile(baseline, (n_samples, 1))
        for i in range(n_params):
            delta = max(abs(baseline[i]) * perturbation, 1e-8)
            X[:, i] += rng.uniform(-delta, delta, n_samples)

        Y = np.array([model_func(x) for x in X])
        var_total = np.var(Y)

        # Approximate first-order: freeze each param and measure
        # remaining variance
        S1 = np.zeros(n_params)
        ST = np.zeros(n_params)

        for i in range(n_params):
            # Variance with param i fixed at baseline
            X_fixed = X.copy()
            X_fixed[:, i] = baseline[i]
            Y_fixed = np.array([model_func(x) for x in X_fixed])
            var_fixed = np.var(Y_fixed)

            # S1_i ≈ (Var_total - Var_fixed) / Var_total
            S1[i] = max(0, (var_total - var_fixed) / max(var_total, 1e-15))

            # Variance with only param i varying
            X_only = np.tile(baseline, (n_samples, 1))
            delta = max(abs(baseline[i]) * perturbation, 1e-8)
            X_only[:, i] += rng.uniform(-delta, delta, n_samples)
            Y_only = np.array([model_func(x) for x in X_only])
            var_only = np.var(Y_only)

            # ST_i ≈ var_only / var_total
            ST[i] = var_only / max(var_total, 1e-15)

        # Normalize
        ST = np.maximum(ST, S1)

        return S1, ST


# ============================================================================
# 5. Probabilistic Validation Metrics
# ============================================================================
class ProbabilisticMetrics:
    """
    Proper scoring rules for probabilistic predictions.

    Implements CRPS, ELPD, calibration error, and sharpness.
    """

    @staticmethod
    def crps_gaussian(
        y_true: np.ndarray,
        y_mean: np.ndarray,
        y_std: np.ndarray,
    ) -> float:
        """
        Continuous Ranked Probability Score for Gaussian predictions.

        CRPS(F, y) = σ [z·(2Φ(z)-1) + 2φ(z) - 1/√π]

        where z = (y - μ) / σ, Φ is CDF, φ is PDF.

        Lower is better. CRPS = 0 for perfect probabilistic forecast.
        """
        y_true = np.asarray(y_true).ravel()
        y_mean = np.asarray(y_mean).ravel()
        y_std = np.maximum(np.asarray(y_std).ravel(), 1e-10)

        z = (y_true - y_mean) / y_std
        crps_vals = y_std * (
            z * (2 * stats.norm.cdf(z) - 1)
            + 2 * stats.norm.pdf(z)
            - 1.0 / np.sqrt(np.pi)
        )
        return float(np.mean(crps_vals))

    @staticmethod
    def elpd(
        y_true: np.ndarray,
        y_mean: np.ndarray,
        y_std: np.ndarray,
    ) -> float:
        """
        Expected Log Predictive Density.

        ELPD = Σ log p(y_i | μ_i, σ_i)

        Higher is better. Measures how well the predictive distribution
        concentrates around observed values.
        """
        y_true = np.asarray(y_true).ravel()
        y_mean = np.asarray(y_mean).ravel()
        y_std = np.maximum(np.asarray(y_std).ravel(), 1e-10)

        log_probs = stats.norm.logpdf(y_true, loc=y_mean, scale=y_std)
        return float(np.mean(log_probs))

    @staticmethod
    def calibration_error(
        y_true: np.ndarray,
        y_mean: np.ndarray,
        y_std: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """
        Expected Calibration Error for regression.

        For each confidence level p, checks if empirical coverage
        matches nominal coverage.
        """
        y_true = np.asarray(y_true).ravel()
        y_mean = np.asarray(y_mean).ravel()
        y_std = np.maximum(np.asarray(y_std).ravel(), 1e-10)

        confidence_levels = np.linspace(0.1, 0.99, n_bins)
        cal_errors = []

        for p in confidence_levels:
            z = stats.norm.ppf((1 + p) / 2)
            lower = y_mean - z * y_std
            upper = y_mean + z * y_std
            empirical = np.mean((y_true >= lower) & (y_true <= upper))
            cal_errors.append(abs(empirical - p))

        return float(np.mean(cal_errors))

    @staticmethod
    def sharpness(y_std: np.ndarray) -> float:
        """
        Sharpness: mean width of 95% credible intervals.

        Narrower (sharper) intervals are better, conditional on
        being well-calibrated.
        """
        y_std = np.maximum(np.asarray(y_std).ravel(), 1e-10)
        ci_width = 2 * 1.96 * y_std
        return float(np.mean(ci_width))

    @staticmethod
    def coverage(
        y_true: np.ndarray,
        y_mean: np.ndarray,
        y_std: np.ndarray,
        level: float = 0.95,
    ) -> float:
        """Empirical coverage at given confidence level."""
        y_true = np.asarray(y_true).ravel()
        y_mean = np.asarray(y_mean).ravel()
        y_std = np.maximum(np.asarray(y_std).ravel(), 1e-10)

        z = stats.norm.ppf((1 + level) / 2)
        lower = y_mean - z * y_std
        upper = y_mean + z * y_std
        return float(np.mean((y_true >= lower) & (y_true <= upper)))

    @classmethod
    def compute_all(
        cls,
        y_true: np.ndarray,
        y_mean: np.ndarray,
        y_std: np.ndarray,
    ) -> ProbabilisticScores:
        """Compute all probabilistic metrics."""
        return ProbabilisticScores(
            crps=cls.crps_gaussian(y_true, y_mean, y_std),
            elpd=cls.elpd(y_true, y_mean, y_std),
            calibration_error=cls.calibration_error(y_true, y_mean, y_std),
            sharpness=cls.sharpness(y_std),
            coverage_90=cls.coverage(y_true, y_mean, y_std, 0.90),
            coverage_95=cls.coverage(y_true, y_mean, y_std, 0.95),
        )


# ============================================================================
# Demo
# ============================================================================
def _demo():
    """Demonstrate Bayesian UQ upgrades."""
    print("=" * 70)
    print("  Bayesian / Probabilistic UQ Upgrades")
    print("=" * 70)

    # 1. BMA
    print("\n  [1] Bayesian Model Averaging")
    bma = BayesianModelAveraging(sigma=0.05)
    bma.add_model("SA", prediction=0.65, error=0.08, uncertainty=0.03)
    bma.add_model("SST", prediction=0.70, error=0.02, uncertainty=0.02)
    bma.add_model("k-epsilon", prediction=0.60, error=0.12, uncertainty=0.04)
    result = bma.compute()
    print(f"  Weights: {dict(zip(result.model_names, result.weights.round(3)))}")
    print(f"  BMA mean: {result.weighted_mean:.4f} ± {result.weighted_std:.4f}")
    print(f"  95% CI: [{result.credible_interval_95[0]:.4f}, "
          f"{result.credible_interval_95[1]:.4f}]")

    # 2. Active Subspace
    print("\n  [2] Active Subspace Detection")
    def test_func(x):
        return 3.0 * x[0] + 0.5 * x[1] ** 2 + 0.1 * x[2] + 0.01 * x[3]
    asub = ActiveSubspace(
        parameter_names=["Re", "TI", "Mach", "AoA"],
        parameter_bounds=np.array([[0, 1]] * 4),
    )
    as_result = asub.compute(test_func, n_samples=200)
    print(f"  Subspace dim: {as_result.subspace_dim}")
    print(f"  Spectral gap: {as_result.spectral_gap:.1f}")
    print(f"  Activity scores: {dict(zip(as_result.parameter_names, as_result.activity_scores.round(3)))}")

    # 3. Aleatoric/Epistemic Decomposition
    print("\n  [3] Aleatoric/Epistemic Decomposition")
    rng = np.random.default_rng(42)
    preds = rng.normal(0.7, 0.02, size=(5, 50))  # 5 members, 50 points
    decomp = UncertaintyDecomposer.decompose_ensemble(preds, noise_floor=0.001)
    print(f"  Epistemic fraction: {decomp.epistemic_fraction:.1%}")
    print(f"  Aleatoric fraction: {decomp.aleatoric_fraction:.1%}")

    # 4. OAT vs Sobol
    print("\n  [4] OAT vs Sobol Comparison")
    baseline = np.array([0.5, 0.3, 0.8, 0.1])
    comp = SensitivityComparator.compare(
        test_func, baseline, ["Re", "TI", "Mach", "AoA"]
    )
    print(f"  OAT ranking: {comp.oat_ranking}")
    print(f"  Sobol ranking: {comp.sobol_ranking}")
    print(f"  Rank correlation: {comp.rank_correlation:.3f}")
    print(f"  Interaction index: {comp.interaction_index:.3f}")

    # 5. Probabilistic Metrics
    print("\n  [5] Probabilistic Validation Metrics")
    y_true = rng.normal(0.7, 0.05, 100)
    y_mean = y_true + rng.normal(0, 0.02, 100)
    y_std = np.full(100, 0.05)
    scores = ProbabilisticMetrics.compute_all(y_true, y_mean, y_std)
    print(f"  CRPS: {scores.crps:.6f}")
    print(f"  ELPD: {scores.elpd:.4f}")
    print(f"  Calibration error: {scores.calibration_error:.4f}")
    print(f"  Sharpness (95% CI width): {scores.sharpness:.4f}")
    print(f"  Coverage (95%): {scores.coverage_95:.1%}")

    print(f"\n{'=' * 70}")
    print("  Demo complete!")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _demo()
