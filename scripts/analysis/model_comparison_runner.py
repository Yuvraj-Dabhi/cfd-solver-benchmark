"""
Multi-Model Comparison Runner
==============================
Generate SU2 configs for multiple turbulence models on the same case,
then produce cross-model comparison tables with metrics and TMR/RSM
reference data.

Usage:
    runner = ModelComparisonRunner("nasa_hump")
    configs = runner.generate_all_configs(output_dir=Path("runs"))
    table = runner.build_comparison_table(results_dict)
"""

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


def _load_config():
    from config import (
        MODEL_COMPARISON_MATRIX, BENCHMARK_CASES, TURBULENCE_MODELS,
        TMR_CODE_RESULTS, RSM_REFERENCE_RESULTS,
    )
    return (MODEL_COMPARISON_MATRIX, BENCHMARK_CASES, TURBULENCE_MODELS,
            TMR_CODE_RESULTS, RSM_REFERENCE_RESULTS)


@dataclass
class ModelResult:
    """Result metrics for one model on one case."""
    model: str
    case: str
    cl: float = 0.0
    cd: float = 0.0
    x_reat: float = 0.0
    bubble_length: float = 0.0
    converged: bool = False
    source: str = "SU2"  # "SU2", "CFL3D", "FUN3D", "RSM-TMR", "experiment"


@dataclass
class ComparisonTable:
    """Cross-model comparison for one case."""
    case_name: str
    results: List[ModelResult] = field(default_factory=list)
    experimental: Optional[Dict[str, float]] = None

    def to_markdown(self) -> str:
        """Generate markdown comparison table."""
        lines = [
            f"## Model Comparison: {self.case_name}",
            "",
            "| Model | Source | CL | CD | x_reat | Bubble Len | Converged |",
            "|-------|--------|-----|-----|--------|------------|-----------|",
        ]
        for r in self.results:
            lines.append(
                f"| {r.model} | {r.source} | {r.cl:.4f} | {r.cd:.5f} | "
                f"{r.x_reat:.3f} | {r.bubble_length:.3f} | "
                f"{'✅' if r.converged else '—'} |"
            )
        if self.experimental:
            exp_line = "| **Experiment** | Exp | "
            exp_line += f"{self.experimental.get('CL', 0):.4f} | "
            exp_line += f"{self.experimental.get('CD', 0):.5f} | "
            exp_line += f"{self.experimental.get('x_reat', 0):.3f} | "
            exp_line += f"{self.experimental.get('bubble_length', 0):.3f} | — |"
            lines.append(exp_line)
        return "\n".join(lines)


class ModelComparisonRunner:
    """
    Generate and compare multiple turbulence models on one benchmark case.

    Parameters
    ----------
    case_name : str
        Benchmark case key from config.
    """

    def __init__(self, case_name: str):
        (self.matrix, self.cases, self.models,
         self.tmr_results, self.rsm_results) = _load_config()
        self.case_name = case_name

        if case_name not in self.matrix:
            raise ValueError(
                f"No model comparison defined for '{case_name}'. "
                f"Available: {list(self.matrix.keys())}"
            )
        self.model_list = self.matrix[case_name]

    def generate_all_configs(
        self,
        output_dir: Path,
        mesh_file: str = "mesh.su2",
        **runner_kwargs,
    ) -> Dict[str, Path]:
        """
        Generate SU2 config files for all models in the comparison.

        Returns dict: {model_name: config_file_path}
        """
        from scripts.solvers.su2_runner import SU2Runner

        configs = {}
        for model in self.model_list:
            case_dir = output_dir / f"{self.case_name}_{model}"
            runner = SU2Runner(
                case_dir=case_dir,
                case_name=self.case_name,
                model=model,
                mesh_file=mesh_file,
                **runner_kwargs,
            )
            runner.setup_case()
            configs[model] = case_dir / "config.cfg"
            logger.info(f"  Generated config: {model} -> {configs[model]}")

        return configs

    def build_comparison_table(
        self,
        su2_results: Optional[Dict[str, Dict[str, float]]] = None,
        experimental: Optional[Dict[str, float]] = None,
    ) -> ComparisonTable:
        """
        Build comparison table combining SU2 results, TMR code reference,
        and RSM reference data.

        Parameters
        ----------
        su2_results : dict, optional
            {model: {"CL": val, "CD": val, ...}} from actual SU2 runs.
        experimental : dict, optional
            Experimental reference values.
        """
        table = ComparisonTable(
            case_name=self.case_name,
            experimental=experimental,
        )

        # Add SU2 results
        if su2_results:
            for model, metrics in su2_results.items():
                table.results.append(ModelResult(
                    model=model, case=self.case_name, source="SU2",
                    cl=metrics.get("CL", 0), cd=metrics.get("CD", 0),
                    x_reat=metrics.get("x_reat", 0),
                    bubble_length=metrics.get("bubble_length", 0),
                    converged=metrics.get("converged", True),
                ))

        # Add TMR CFL3D/FUN3D reference
        if self.case_name in self.tmr_results:
            for solver_key, data in self.tmr_results[self.case_name].items():
                table.results.append(ModelResult(
                    model=f"{solver_key}",
                    case=self.case_name,
                    source="TMR",
                    cl=data.get("CL", 0),
                    cd=data.get("CD", 0),
                    x_reat=data.get("x_reat", data.get("x_reat_xH", 0)),
                    bubble_length=data.get("bubble_length", 0),
                    converged=True,
                ))

        # Add RSM reference (SU2 can't run RSM, so we use TMR published data)
        if self.case_name in self.rsm_results:
            for rsm_name, data in self.rsm_results[self.case_name].items():
                table.results.append(ModelResult(
                    model=f"RSM ({rsm_name})",
                    case=self.case_name,
                    source="TMR-RSM",
                    x_reat=data.get("x_reat", data.get("x_reat_xH", 0)),
                    bubble_length=data.get("bubble_length", 0),
                    converged=True,
                ))

        return table

    def get_expected_results(self) -> Dict[str, str]:
        """Get expected RANS error baselines from config for this case."""
        case = self.cases.get(self.case_name)
        if case and case.rans_error_baseline:
            return case.rans_error_baseline
        return {}

    def generate_summary_report(
        self,
        su2_results: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> str:
        """Generate a complete markdown summary report."""
        table = self.build_comparison_table(su2_results)
        expected = self.get_expected_results()

        lines = [table.to_markdown(), ""]

        if expected:
            lines.extend([
                "### Expected RANS Error Baselines",
                "",
                "| Model | Expected Performance |",
                "|-------|---------------------|",
            ])
            for model, perf in expected.items():
                lines.append(f"| {model} | {perf} |")

        return "\n".join(lines)


def list_comparison_cases() -> Dict[str, List[str]]:
    """List all cases with defined model comparisons."""
    matrix, *_ = _load_config()
    return matrix


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Demo: generate configs for nasa_hump comparison
    runner = ModelComparisonRunner("nasa_hump")
    output = PROJECT_ROOT / "runs" / "model_comparison"
    configs = runner.generate_all_configs(output)
    print(f"\nGenerated {len(configs)} configs:")
    for model, path in configs.items():
        print(f"  {model}: {path}")

    # Demo: build table with hypothetical results
    report = runner.generate_summary_report(su2_results={
        "SA":  {"CL": 0.0, "CD": 0.0, "x_reat": 1.28, "bubble_length": 0.615},
        "SST": {"CL": 0.0, "CD": 0.0, "x_reat": 1.17, "bubble_length": 0.505},
    })
    print(report)
