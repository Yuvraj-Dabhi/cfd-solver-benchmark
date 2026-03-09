"""
Sensitivity Analysis Module
============================
Global sensitivity analysis using Sobol indices to identify critical
input parameters affecting CFD simulation results.

Uses SALib for Saltelli sampling and Sobol analysis.
Identifies parameters with total-order index ST > 0.1 as critical.

Reference: Sobol, I.M. (2001) "Global sensitivity indices for nonlinear
mathematical models and their Monte Carlo estimates"
"""

import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import SENSITIVITY_PARAMETERS


@dataclass
class SobolResult:
    """Container for Sobol sensitivity analysis results."""
    parameter_names: List[str] = field(default_factory=list)
    S1: np.ndarray = field(default_factory=lambda: np.array([]))  # First-order
    S1_conf: np.ndarray = field(default_factory=lambda: np.array([]))
    ST: np.ndarray = field(default_factory=lambda: np.array([]))  # Total-order
    ST_conf: np.ndarray = field(default_factory=lambda: np.array([]))
    n_samples: int = 0
    critical_parameters: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = ["Sobol Sensitivity Analysis Results", "=" * 50]
        lines.append(f"Samples: {self.n_samples}")
        lines.append(f"{'Parameter':<25} {'S1':>8} {'ST':>8} {'Critical?':>10}")
        lines.append("-" * 55)
        for i, name in enumerate(self.parameter_names):
            s1 = self.S1[i] if i < len(self.S1) else 0
            st = self.ST[i] if i < len(self.ST) else 0
            crit = "YES" if name in self.critical_parameters else ""
            lines.append(f"{name:<25} {s1:>8.4f} {st:>8.4f} {crit:>10}")
        return "\n".join(lines)


class ParametricSensitivityAnalysis:
    """
    Perform global sensitivity analysis using Sobol indices.

    Usage
    -----
    >>> sa = ParametricSensitivityAnalysis()
    >>> result = sa.run(model_func=my_cfd_wrapper, n_samples=1024)
    >>> print(result.summary())
    """

    CRITICAL_THRESHOLD = 0.1  # ST > 0.1 → critical parameter

    def __init__(self, problem: Optional[Dict] = None):
        """
        Parameters
        ----------
        problem : dict, optional
            SALib problem definition. Uses config.SENSITIVITY_PARAMETERS if None.
        """
        self.problem = problem or SENSITIVITY_PARAMETERS

    def generate_samples(self, n_samples: int = 1024) -> np.ndarray:
        """
        Generate Saltelli samples for Sobol analysis.

        Parameters
        ----------
        n_samples : int
            Base sample size. Total evaluations = N * (2D + 2).

        Returns
        -------
        array, shape (n_total, n_params)
            Parameter samples.
        """
        from SALib.sample import saltelli

        self.samples = saltelli.sample(self.problem, n_samples, calc_second_order=False)
        self._n_base = n_samples
        return self.samples

    def evaluate(
        self,
        model_func: Callable[[np.ndarray], float],
        samples: Optional[np.ndarray] = None,
        parallel: bool = False,
    ) -> np.ndarray:
        """
        Evaluate the model function at all sample points.

        Parameters
        ----------
        model_func : callable
            Function that takes a 1D parameter array and returns a scalar output.
        samples : array, optional
            Override samples from generate_samples().
        parallel : bool
            Use multiprocessing.

        Returns
        -------
        array
            Model output for each sample.
        """
        if samples is None:
            if not hasattr(self, "samples"):
                raise RuntimeError("Call generate_samples() first or provide samples.")
            samples = self.samples

        if parallel:
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor() as pool:
                self.outputs = np.array(list(pool.map(model_func, samples)))
        else:
            self.outputs = np.array([model_func(x) for x in samples])

        return self.outputs

    def analyze(self, outputs: Optional[np.ndarray] = None) -> SobolResult:
        """
        Compute Sobol indices from model outputs.

        Returns
        -------
        SobolResult
            First-order and total-order indices with confidence intervals.
        """
        from SALib.analyze import sobol

        if outputs is None:
            outputs = self.outputs

        si = sobol.analyze(
            self.problem, outputs, calc_second_order=False,
            print_to_console=False,
        )

        result = SobolResult(
            parameter_names=self.problem["names"],
            S1=np.array(si["S1"]),
            S1_conf=np.array(si["S1_conf"]),
            ST=np.array(si["ST"]),
            ST_conf=np.array(si["ST_conf"]),
            n_samples=len(outputs),
        )

        # Identify critical parameters
        result.critical_parameters = [
            name for name, st in zip(result.parameter_names, result.ST)
            if st > self.CRITICAL_THRESHOLD
        ]

        return result

    def run(
        self,
        model_func: Callable[[np.ndarray], float],
        n_samples: int = 1024,
        parallel: bool = False,
    ) -> SobolResult:
        """Complete sensitivity analysis: sample → evaluate → analyze."""
        self.generate_samples(n_samples)
        self.evaluate(model_func, parallel=parallel)
        return self.analyze()

    def plot(self, result: SobolResult, save_path: Optional[str] = None):
        """Bar chart of Sobol indices."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        n = len(result.parameter_names)
        x = np.arange(n)

        # First-order indices
        axes[0].bar(x, result.S1, yerr=result.S1_conf, capsize=3,
                    color="steelblue", alpha=0.8)
        axes[0].axhline(self.CRITICAL_THRESHOLD, color="red", linestyle="--",
                        linewidth=0.8, label=f"Threshold ({self.CRITICAL_THRESHOLD})")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(result.parameter_names, rotation=45, ha="right")
        axes[0].set_ylabel("S1 (First-Order)")
        axes[0].set_title("First-Order Sobol Indices")
        axes[0].legend()

        # Total-order indices
        axes[1].bar(x, result.ST, yerr=result.ST_conf, capsize=3,
                    color="coral", alpha=0.8)
        axes[1].axhline(self.CRITICAL_THRESHOLD, color="red", linestyle="--",
                        linewidth=0.8, label=f"Threshold ({self.CRITICAL_THRESHOLD})")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(result.parameter_names, rotation=45, ha="right")
        axes[1].set_ylabel("ST (Total-Order)")
        axes[1].set_title("Total-Order Sobol Indices")
        axes[1].legend()

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300)
        return fig
