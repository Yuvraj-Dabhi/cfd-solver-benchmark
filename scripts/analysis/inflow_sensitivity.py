"""
Parametric Inflow Sensitivity Analysis
========================================
Sweep inlet boundary condition parameters to quantify their
influence on separation prediction accuracy.

Parameters studied (per Implementation Plan §2.3 and §7.1):
  - Turbulence intensity (TI): 0.5% – 5%
  - Momentum thickness Reynolds number (Re_θ): 6454 – 7200
  - Inlet velocity profile shape (power-law exponent)
  - Freestream velocity perturbation

The research literature identifies inflow conditions as a major
source of uncertainty in separated flow predictions.

Usage:
    sweep = InflowSweep()
    sweep.add_model(my_solver_func)
    results = sweep.run()
    sweep.analyze(results)
    sweep.plot_tornado(results)
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple
from itertools import product

logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================
@dataclass
class InflowParameter:
    """Definition of an inflow boundary condition parameter."""
    name: str
    baseline: float
    low: float
    high: float
    units: str = ""
    n_points: int = 5
    description: str = ""

    def sweep_values(self) -> np.ndarray:
        """Generate evenly-spaced sweep values."""
        return np.linspace(self.low, self.high, self.n_points)

    def normalized_range(self) -> Tuple[float, float]:
        """Return range normalized to baseline."""
        if abs(self.baseline) < 1e-15:
            return (self.low, self.high)
        return (
            (self.low - self.baseline) / abs(self.baseline),
            (self.high - self.baseline) / abs(self.baseline),
        )


@dataclass
class SweepResult:
    """Result from a single inflow parameter evaluation."""
    parameter_name: str
    parameter_value: float
    x_separation: Optional[float] = None
    x_reattachment: Optional[float] = None
    bubble_length: Optional[float] = None
    Cp_rmse: Optional[float] = None
    Cf_rmse: Optional[float] = None
    converged: bool = True
    metadata: Dict = field(default_factory=dict)


@dataclass
class InfluenceResult:
    """Quantified influence of a parameter on outputs."""
    parameter_name: str
    output_name: str
    sensitivity: float  # ∂output/∂param (normalized)
    influence_rank: int = 0
    baseline_output: float = 0.0
    output_range: float = 0.0
    cv_pct: float = 0.0  # coefficient of variation


# =============================================================================
# Pre-defined Parameter Sets
# =============================================================================
# NASA hump case parameters from research (Implementation Plan §2.3)
NASA_HUMP_PARAMETERS = [
    InflowParameter(
        name="turbulence_intensity",
        baseline=1.0,
        low=0.5,
        high=5.0,
        units="%",
        n_points=7,
        description="Freestream turbulence intensity (TI)",
    ),
    InflowParameter(
        name="Re_theta",
        baseline=6827,
        low=6454,
        high=7200,
        units="",
        n_points=5,
        description="Momentum thickness Reynolds number at inlet",
    ),
    InflowParameter(
        name="profile_exponent",
        baseline=7.0,
        low=5.0,
        high=9.0,
        units="",
        n_points=5,
        description="Power-law profile exponent (1/n in U/U_inf = (y/δ)^(1/n))",
    ),
    InflowParameter(
        name="U_inf",
        baseline=34.6,
        low=33.5,
        high=35.5,
        units="m/s",
        n_points=5,
        description="Freestream velocity (±3% uncertainty range)",
    ),
    InflowParameter(
        name="eddy_viscosity_ratio",
        baseline=10.0,
        low=1.0,
        high=100.0,
        units="",
        n_points=5,
        description="Inlet eddy viscosity ratio (ν_t/ν)",
    ),
]

# Backward-facing step parameters
BFS_PARAMETERS = [
    InflowParameter(
        name="turbulence_intensity",
        baseline=1.5,
        low=0.5,
        high=5.0,
        units="%",
        n_points=7,
        description="Freestream turbulence intensity",
    ),
    InflowParameter(
        name="Re_h",
        baseline=37400,
        low=35000,
        high=40000,
        units="",
        n_points=5,
        description="Step-height Reynolds number",
    ),
    InflowParameter(
        name="profile_exponent",
        baseline=7.0,
        low=5.0,
        high=9.0,
        units="",
        n_points=5,
        description="Power-law profile exponent",
    ),
]


# =============================================================================
# Simplified Flow Model for Parameter Sweeps
# =============================================================================
def _default_hump_model(params: Dict[str, float]) -> SweepResult:
    """
    Built-in simplified NASA hump flow model for parametric sweeps.

    Models the sensitivity of separation/reattachment to inflow
    conditions based on correlations from the research literature.

    In production, replace this with an actual CFD solver call.
    """
    TI = params.get("turbulence_intensity", 1.0)
    Re_theta = params.get("Re_theta", 6827)
    n_exp = params.get("profile_exponent", 7.0)
    U_inf = params.get("U_inf", 34.6)
    evt_ratio = params.get("eddy_viscosity_ratio", 10.0)

    # Separation point: sensitive to Re_θ and profile shape
    # Higher Re_θ → more momentum → later separation
    x_sep_base = 0.665
    x_sep = x_sep_base - 0.02 * (TI - 1.0) / 4.0     # Higher TI → earlier sep
    x_sep += 0.015 * (Re_theta - 6827) / 746           # Higher Re_θ → later sep
    x_sep += 0.008 * (n_exp - 7.0) / 2.0               # Fuller profile → later sep
    x_sep -= 0.003 * (U_inf - 34.6) / 1.0              # Minor velocity effect
    x_sep += 0.005 * np.log10(evt_ratio / 10.0 + 1e-10)  # EVR effect

    # Reattachment point: primarily sensitive to TI and Re_θ
    x_reat_base = 1.11
    x_reat = x_reat_base + 0.04 * (TI - 1.0) / 4.0    # Higher TI → earlier reat
    x_reat -= 0.025 * (Re_theta - 6827) / 746
    x_reat -= 0.015 * (n_exp - 7.0) / 2.0
    x_reat += 0.005 * (U_inf - 34.6) / 1.0

    bubble = x_reat - x_sep

    return SweepResult(
        parameter_name="combined",
        parameter_value=0.0,
        x_separation=x_sep,
        x_reattachment=x_reat,
        bubble_length=bubble,
        converged=True,
    )


# =============================================================================
# Inflow Sweep Engine
# =============================================================================
class InflowSweep:
    """
    Parametric inflow condition sweep for separation sensitivity analysis.

    Sweeps each inflow parameter independently (one-at-a-time OAT)
    while holding others at baseline, then computes influence rankings.

    For full interaction effects, use the coupled factorial sweep mode.

    Usage:
        sweep = InflowSweep(case="nasa_hump")
        results = sweep.run()
        influence = sweep.analyze(results)
        sweep.print_report(influence)
    """

    PARAMETER_PRESETS = {
        "nasa_hump": NASA_HUMP_PARAMETERS,
        "backward_facing_step": BFS_PARAMETERS,
    }

    MODEL_PRESETS = {
        "nasa_hump": _default_hump_model,
    }

    def __init__(
        self,
        case: str = "nasa_hump",
        parameters: Optional[List[InflowParameter]] = None,
        model_func: Optional[Callable] = None,
    ):
        """
        Parameters
        ----------
        case : str
            Benchmark case name (for parameter/model presets).
        parameters : list of InflowParameter, optional
            Custom parameters (overrides preset).
        model_func : callable, optional
            Solver function: f(params_dict) -> SweepResult.
            If None, uses built-in simplified model.
        """
        self.case = case
        self.parameters = parameters or self.PARAMETER_PRESETS.get(case, [])
        self.model_func = model_func or self.MODEL_PRESETS.get(
            case, _default_hump_model
        )

    def _get_baseline_params(self) -> Dict[str, float]:
        """Get dict of baseline parameter values."""
        return {p.name: p.baseline for p in self.parameters}

    def run(self) -> Dict[str, List[SweepResult]]:
        """
        Run one-at-a-time (OAT) parameter sweep.

        For each parameter, sweeps through its range while holding
        all other parameters at their baseline values.

        Returns
        -------
        dict : {parameter_name: [SweepResult, ...]}
        """
        results = {}
        baseline = self._get_baseline_params()

        for param in self.parameters:
            logger.info(f"Sweeping {param.name}: {param.low} → {param.high} "
                        f"({param.n_points} points)")
            param_results = []

            for value in param.sweep_values():
                # Set this parameter, keep others at baseline
                params = baseline.copy()
                params[param.name] = value

                try:
                    result = self.model_func(params)
                    result.parameter_name = param.name
                    result.parameter_value = value
                    param_results.append(result)
                except Exception as e:
                    logger.warning(f"Failed at {param.name}={value}: {e}")
                    param_results.append(SweepResult(
                        parameter_name=param.name,
                        parameter_value=value,
                        converged=False,
                    ))

            results[param.name] = param_results
            logger.info(f"  → {len(param_results)} evaluations completed")

        return results

    def run_factorial(
        self,
        n_levels: int = 3,
    ) -> List[SweepResult]:
        """
        Run full factorial sweep for interaction effects.

        Warning: grows as n_levels^n_params — use only for
        small parameter sets.

        Parameters
        ----------
        n_levels : int
            Number of levels per parameter.

        Returns
        -------
        List of SweepResults for all combinations.
        """
        # Override n_points for factorial
        levels = {}
        for p in self.parameters:
            levels[p.name] = np.linspace(p.low, p.high, n_levels)

        names = list(levels.keys())
        values_list = [levels[n] for n in names]

        results = []
        total = np.prod([len(v) for v in values_list])
        logger.info(f"Running factorial sweep: {int(total)} evaluations")

        for combo in product(*values_list):
            params = dict(zip(names, combo))
            try:
                result = self.model_func(params)
                result.parameter_name = "factorial"
                result.metadata = params.copy()
                results.append(result)
            except Exception as e:
                logger.warning(f"Failed: {e}")

        return results

    # =========================================================================
    # Analysis
    # =========================================================================
    def analyze(
        self,
        sweep_results: Dict[str, List[SweepResult]],
    ) -> List[InfluenceResult]:
        """
        Compute influence of each parameter on each output quantity.

        Uses finite-difference sensitivity estimation:
        S_i = (∂Q/∂p_i) * (Δp_i / Q_baseline)

        Parameters
        ----------
        sweep_results : dict from run()

        Returns
        -------
        List of InfluenceResult, sorted by absolute sensitivity.
        """
        outputs = ["x_separation", "x_reattachment", "bubble_length"]
        influences = []

        for param in self.parameters:
            results = sweep_results.get(param.name, [])
            converged = [r for r in results if r.converged]
            if len(converged) < 2:
                continue

            for output_name in outputs:
                values = [getattr(r, output_name) for r in converged]
                if any(v is None for v in values):
                    continue

                values = np.array(values)
                param_values = np.array([r.parameter_value for r in converged])

                # Baseline output (at baseline parameter value)
                baseline_idx = np.argmin(
                    np.abs(param_values - param.baseline)
                )
                Q_base = values[baseline_idx]

                # Sensitivity: slope of output vs parameter
                if abs(Q_base) > 1e-15 and len(param_values) > 1:
                    # Linear regression for sensitivity
                    p_norm = (param_values - param.baseline) / abs(param.baseline)
                    q_norm = (values - Q_base) / abs(Q_base)
                    if len(p_norm) > 1 and np.std(p_norm) > 1e-15:
                        sensitivity = np.polyfit(p_norm, q_norm, 1)[0]
                    else:
                        sensitivity = 0.0
                else:
                    sensitivity = 0.0

                output_range = float(np.max(values) - np.min(values))
                cv = float(np.std(values) / abs(np.mean(values)) * 100
                           if abs(np.mean(values)) > 1e-15 else 0.0)

                influences.append(InfluenceResult(
                    parameter_name=param.name,
                    output_name=output_name,
                    sensitivity=sensitivity,
                    baseline_output=Q_base,
                    output_range=output_range,
                    cv_pct=cv,
                ))

        # Rank by absolute sensitivity within each output
        for output_name in outputs:
            group = [i for i in influences if i.output_name == output_name]
            group.sort(key=lambda i: abs(i.sensitivity), reverse=True)
            for rank, inf in enumerate(group, 1):
                inf.influence_rank = rank

        return influences

    # =========================================================================
    # Visualization
    # =========================================================================
    def plot_tornado(
        self,
        influences: List[InfluenceResult],
        output_name: str = "bubble_length",
        save_path: Optional[str] = None,
    ) -> None:
        """
        Generate tornado chart showing parameter influence on an output.

        Parameters
        ----------
        influences : list of InfluenceResult
        output_name : str
            Which output to plot.
        save_path : str, optional
            Save plot to file.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available for plotting")
            return

        relevant = [i for i in influences if i.output_name == output_name]
        relevant.sort(key=lambda i: abs(i.sensitivity))

        names = [i.parameter_name.replace("_", " ").title() for i in relevant]
        sensitivities = [i.sensitivity for i in relevant]

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ["#e74c3c" if s > 0 else "#3498db" for s in sensitivities]
        bars = ax.barh(names, sensitivities, color=colors, edgecolor="white",
                       linewidth=0.5)

        ax.set_xlabel("Normalized Sensitivity (∂Q/Q)/(∂p/p)", fontsize=12)
        ax.set_title(
            f"Inflow Parameter Influence on {output_name.replace('_', ' ').title()}",
            fontsize=14, fontweight="bold"
        )
        ax.axvline(x=0, color="black", linewidth=0.8)

        # Add value labels
        for bar, val in zip(bars, sensitivities):
            x_pos = bar.get_width()
            ax.text(
                x_pos + 0.01 * np.sign(x_pos), bar.get_y() + bar.get_height() / 2,
                f"{val:+.3f}", va="center", fontsize=10,
            )

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Tornado chart saved to {save_path}")
        plt.close()

    def plot_response_surface(
        self,
        sweep_results: Dict[str, List[SweepResult]],
        output_name: str = "bubble_length",
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot individual parameter response curves.

        Parameters
        ----------
        sweep_results : dict from run()
        output_name : str
            Output quantity to plot.
        save_path : str, optional
            Save plot to file.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available for plotting")
            return

        n_params = len(sweep_results)
        fig, axes = plt.subplots(1, n_params, figsize=(4 * n_params, 4),
                                 squeeze=False)
        axes = axes[0]

        for ax, (param_name, results) in zip(axes, sweep_results.items()):
            converged = [r for r in results if r.converged]
            x = [r.parameter_value for r in converged]
            y = [getattr(r, output_name) for r in converged]

            if any(v is None for v in y):
                continue

            ax.plot(x, y, "o-", color="#2c3e50", linewidth=2, markersize=6)
            ax.set_xlabel(param_name.replace("_", " ").title(), fontsize=10)
            ax.set_ylabel(output_name.replace("_", " ").title(), fontsize=10)
            ax.grid(True, alpha=0.3)

            # Mark baseline
            param_obj = next(
                (p for p in self.parameters if p.name == param_name), None
            )
            if param_obj:
                ax.axvline(
                    x=param_obj.baseline, color="#e74c3c",
                    linestyle="--", alpha=0.7, label="Baseline"
                )
                ax.legend(fontsize=8)

        fig.suptitle(
            f"Inflow Parameter Response: {output_name.replace('_', ' ').title()}",
            fontsize=14, fontweight="bold"
        )
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Response surface saved to {save_path}")
        plt.close()

    # =========================================================================
    # Reporting
    # =========================================================================
    def print_report(self, influences: List[InfluenceResult]) -> None:
        """Print formatted inflow sensitivity report."""
        print(f"\n{'='*70}")
        print(f"  Parametric Inflow Sensitivity Analysis — {self.case}")
        print(f"{'='*70}")

        # Group by output
        outputs = sorted(set(i.output_name for i in influences))

        for output_name in outputs:
            group = [i for i in influences if i.output_name == output_name]
            group.sort(key=lambda i: i.influence_rank)

            print(f"\n  Output: {output_name}")
            print(f"  {'Rank':<5s} {'Parameter':<25s} {'Sensitivity':>12s} "
                  f"{'Range':>10s} {'CV%':>7s}")
            print(f"  {'-'*61}")

            for inf in group:
                print(
                    f"  {inf.influence_rank:<5d} "
                    f"{inf.parameter_name:<25s} "
                    f"{inf.sensitivity:+12.4f} "
                    f"{inf.output_range:10.4f} "
                    f"{inf.cv_pct:7.2f}"
                )

            # Most influential
            if group:
                top = group[0]
                print(f"\n  → Most influential: {top.parameter_name} "
                      f"(sensitivity = {top.sensitivity:+.4f})")

        print(f"\n{'='*70}")

    def get_summary_dict(
        self,
        influences: List[InfluenceResult],
    ) -> Dict[str, Dict]:
        """Return machine-readable summary of influence analysis."""
        summary = {}
        for inf in influences:
            key = f"{inf.output_name}__{inf.parameter_name}"
            summary[key] = {
                "parameter": inf.parameter_name,
                "output": inf.output_name,
                "sensitivity": inf.sensitivity,
                "rank": inf.influence_rank,
                "baseline_output": inf.baseline_output,
                "output_range": inf.output_range,
                "cv_pct": inf.cv_pct,
            }
        return summary
