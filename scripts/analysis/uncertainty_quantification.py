"""
Uncertainty Quantification Module
==================================
Monte Carlo propagation of input uncertainties through CFD models.
Implements combined CFD+experimental uncertainty validation per ASME V&V 20-2009.

Usage:
    uq = UncertaintyQuantification()
    uq.define_input_uncertainties()
    uq.propagate_uncertainty(model_func, n_samples=500)
    results = uq.analyze()
    validation = uq.compare_with_experiment(exp_value=6.5, exp_uncertainty=0.3)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import UQ_INPUT_DISTRIBUTIONS


@dataclass
class UQResult:
    """Container for uncertainty quantification results."""
    mean: float = 0.0
    std: float = 0.0
    ci_95: Tuple[float, float] = (0.0, 0.0)
    ci_99: Tuple[float, float] = (0.0, 0.0)
    coefficient_of_variation: float = 0.0
    n_samples: int = 0
    distribution: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class ValidationResult:
    """Result of CFD-vs-experiment validation with uncertainty."""
    cfd_mean: float = 0.0
    cfd_std: float = 0.0
    exp_value: float = 0.0
    exp_uncertainty: float = 0.0
    error: float = 0.0
    combined_uncertainty: float = 0.0
    status: str = "UNKNOWN"  # VALIDATED or NOT VALIDATED


class UncertaintyQuantification:
    """
    Monte Carlo uncertainty quantification for CFD benchmarking.

    Propagates input parameter uncertainties through the model and computes
    output statistics. Validates against experimental data using combined
    CFD+experimental uncertainty as per ASME V&V 20-2009.
    """

    def __init__(self, input_distributions: Optional[Dict] = None):
        """
        Parameters
        ----------
        input_distributions : dict, optional
            Custom distributions. Uses config.UQ_INPUT_DISTRIBUTIONS if None.
            Format: {param_name: {"distribution": "normal"/"uniform", ...}}
        """
        self.input_dists = input_distributions or UQ_INPUT_DISTRIBUTIONS
        self.samples: Optional[np.ndarray] = None
        self.output_distribution: Optional[np.ndarray] = None

    def define_input_uncertainties(self, custom: Optional[Dict] = None):
        """Set or update input uncertainty distributions."""
        if custom:
            self.input_dists.update(custom)

    def generate_samples(self, n_samples: int = 500) -> np.ndarray:
        """
        Generate Monte Carlo samples from input distributions.

        Parameters
        ----------
        n_samples : int
            Number of Monte Carlo samples.

        Returns
        -------
        array, shape (n_samples, n_params)
            Input parameter samples.
        """
        rng = np.random.default_rng(42)  # Reproducible
        param_samples = []

        for name, dist in self.input_dists.items():
            if dist["distribution"] == "normal":
                s = rng.normal(dist["mean"], dist["std"], n_samples)
            elif dist["distribution"] == "uniform":
                s = rng.uniform(dist["lower"], dist["upper"], n_samples)
            elif dist["distribution"] == "lognormal":
                mu = np.log(dist["mean"]**2 / np.sqrt(dist["std"]**2 + dist["mean"]**2))
                sigma = np.sqrt(np.log(1 + dist["std"]**2 / dist["mean"]**2))
                s = rng.lognormal(mu, sigma, n_samples)
            else:
                raise ValueError(f"Unknown distribution: {dist['distribution']}")
            param_samples.append(s)

        self.samples = np.column_stack(param_samples)
        self._param_names = list(self.input_dists.keys())
        return self.samples

    def propagate_uncertainty(
        self,
        model_func: Callable[[np.ndarray], float],
        n_samples: int = 500,
        parallel: bool = False,
    ) -> np.ndarray:
        """
        Propagate input uncertainty through the model.

        Parameters
        ----------
        model_func : callable
            Function taking parameter vector → scalar output.
        n_samples : int
            Number of MC samples.
        parallel : bool
            Use multiprocessing.

        Returns
        -------
        array
            Output distribution.
        """
        if self.samples is None or len(self.samples) != n_samples:
            self.generate_samples(n_samples)

        if parallel:
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor() as pool:
                self.output_distribution = np.array(
                    list(pool.map(model_func, self.samples))
                )
        else:
            self.output_distribution = np.array(
                [model_func(x) for x in self.samples]
            )

        return self.output_distribution

    def analyze(self) -> UQResult:
        """
        Analyze output distribution: mean, std, confidence intervals.

        Returns
        -------
        UQResult
            Statistical summary of output uncertainty.
        """
        if self.output_distribution is None:
            raise RuntimeError("Run propagate_uncertainty() first.")

        dist = self.output_distribution
        mean = float(np.mean(dist))
        std = float(np.std(dist))

        return UQResult(
            mean=mean,
            std=std,
            ci_95=(float(np.percentile(dist, 2.5)), float(np.percentile(dist, 97.5))),
            ci_99=(float(np.percentile(dist, 0.5)), float(np.percentile(dist, 99.5))),
            coefficient_of_variation=std / abs(mean) if abs(mean) > 1e-15 else float("inf"),
            n_samples=len(dist),
            distribution=dist,
        )

    def compare_with_experiment(
        self,
        exp_value: float,
        exp_uncertainty: float,
    ) -> ValidationResult:
        """
        Validate CFD against experiment accounting for uncertainties.

        Uses ASME V&V 20-2009 approach:
            combined_uncertainty = sqrt(σ_CFD² + σ_exp²)
            status = VALIDATED if |E| < 2 * combined_uncertainty

        Parameters
        ----------
        exp_value : float
            Experimental reference value.
        exp_uncertainty : float
            Experimental uncertainty (±).

        Returns
        -------
        ValidationResult
            Validation status with combined uncertainty.
        """
        if self.output_distribution is None:
            raise RuntimeError("Run propagate_uncertainty() first.")

        cfd_mean = float(np.mean(self.output_distribution))
        cfd_std = float(np.std(self.output_distribution))

        combined = np.sqrt(cfd_std**2 + exp_uncertainty**2)
        error = cfd_mean - exp_value

        # Validated if error within 2σ combined (95% confidence)
        status = "VALIDATED" if abs(error) < 2 * combined else "NOT VALIDATED"

        return ValidationResult(
            cfd_mean=cfd_mean,
            cfd_std=cfd_std,
            exp_value=exp_value,
            exp_uncertainty=exp_uncertainty,
            error=error,
            combined_uncertainty=combined,
            status=status,
        )

    def plot(
        self,
        result: UQResult,
        exp_value: Optional[float] = None,
        exp_uncertainty: Optional[float] = None,
        save_path: Optional[str] = None,
    ):
        """Histogram + Q-Q plot of output distribution."""
        import matplotlib.pyplot as plt
        from scipy import stats

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Histogram
        axes[0].hist(result.distribution, bins=30, density=True,
                     alpha=0.7, color="steelblue", edgecolor="white")
        axes[0].axvline(result.mean, color="red", linestyle="--",
                        label=f"Mean = {result.mean:.3f}")
        axes[0].axvline(result.ci_95[0], color="green", linestyle="--",
                        label="95% CI")
        axes[0].axvline(result.ci_95[1], color="green", linestyle="--")

        if exp_value is not None:
            axes[0].axvline(exp_value, color="orange", linewidth=2,
                            label=f"Experiment = {exp_value:.3f}")
            if exp_uncertainty:
                axes[0].axvspan(exp_value - exp_uncertainty,
                                exp_value + exp_uncertainty,
                                alpha=0.15, color="orange", label="Exp. uncertainty")

        axes[0].set_xlabel("Output Value")
        axes[0].set_ylabel("Probability Density")
        axes[0].set_title("Monte Carlo Output Distribution")
        axes[0].legend(fontsize=8)

        # Q-Q plot
        stats.probplot(result.distribution, dist="norm", plot=axes[1])
        axes[1].set_title("Q-Q Plot (Normality Check)")

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300)
        return fig
