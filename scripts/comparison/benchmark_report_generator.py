"""
Benchmark Report Generator
============================
Produces consolidated Markdown and JSON reports aggregating all
pipeline results: GCI, scheme sensitivity, model rankings,
V&V status, 40% Challenge progress, and experimental comparisons.

Usage:
    report = BenchmarkReportGenerator()
    report.add_gci_results(gci_data)
    report.add_model_ranking(ranking_data)
    report.generate_markdown("benchmark_report.md")
    report.generate_json("benchmark_report.json")
"""

import json
import logging
import numpy as np
from datetime import datetime
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures for Report Sections
# =============================================================================
@dataclass
class GCIEntry:
    """Grid Convergence Index result for one case."""
    case_name: str
    quantity: str
    gci_fine: float
    gci_coarse: float
    observed_order: float
    asymptotic_ratio: float
    in_asymptotic_range: bool
    grid_levels: int = 3


@dataclass
class ModelRankEntry:
    """Ranking entry for one turbulence model on one case."""
    case_name: str
    model_name: str
    x_sep_error: float = 0.0
    x_reat_error: float = 0.0
    Cp_rmse: float = 0.0
    Cf_rmse: float = 0.0
    composite_score: float = 0.0
    rank: int = 0


@dataclass
class VVEntry:
    """V&V validation result for one case."""
    case_name: str
    validation_level: int
    status: str  # PASS, FAIL, PARTIAL
    error_metric: float = 0.0
    within_uncertainty: bool = False
    mrr_level: int = 0


@dataclass
class ChallengeEntry:
    """NASA 40% Challenge progress for one metric."""
    metric: str
    baseline_error: float
    current_error: float
    reduction_pct: float = 0.0
    target_met: bool = False


@dataclass
class SchemeSensEntry:
    """Scheme sensitivity for one quantity."""
    quantity: str
    cv_pct: float
    spread: float
    scheme_converged: bool


# =============================================================================
# Report Generator
# =============================================================================
class BenchmarkReportGenerator:
    """
    Collects results from all benchmark pipeline phases and produces
    formatted Markdown and JSON reports.

    Follows the documentation structure outlined in the Implementation
    Plan §4.1 and the CFD Benchmark Review §8.
    """

    def __init__(self, title: str = "CFD Solver Benchmark Report"):
        self.title = title
        self.timestamp = datetime.now().isoformat()
        self.gci_results: List[GCIEntry] = []
        self.model_rankings: List[ModelRankEntry] = []
        self.vv_results: List[VVEntry] = []
        self.challenge_progress: List[ChallengeEntry] = []
        self.scheme_sensitivity: List[SchemeSensEntry] = []
        self.experimental_comparisons: Dict[str, Dict[str, float]] = {}
        self.notes: List[str] = []

    # =========================================================================
    # Data Collection Methods
    # =========================================================================
    def add_gci_results(
        self,
        case_name: str,
        quantity: str,
        gci_fine: float,
        gci_coarse: float,
        observed_order: float,
        asymptotic_ratio: float,
        grid_levels: int = 3,
    ) -> None:
        """Add Grid Convergence Index results for a case/quantity."""
        self.gci_results.append(GCIEntry(
            case_name=case_name,
            quantity=quantity,
            gci_fine=gci_fine,
            gci_coarse=gci_coarse,
            observed_order=observed_order,
            asymptotic_ratio=asymptotic_ratio,
            in_asymptotic_range=abs(asymptotic_ratio - 1.0) < 0.1,
            grid_levels=grid_levels,
        ))

    def add_model_ranking(
        self,
        case_name: str,
        model_name: str,
        x_sep_error: float = 0.0,
        x_reat_error: float = 0.0,
        Cp_rmse: float = 0.0,
        Cf_rmse: float = 0.0,
    ) -> None:
        """Add turbulence model result for ranking."""
        # Composite score: weighted combination
        composite = (
            0.30 * abs(x_sep_error) +
            0.30 * abs(x_reat_error) +
            0.20 * Cp_rmse +
            0.20 * Cf_rmse
        )
        self.model_rankings.append(ModelRankEntry(
            case_name=case_name,
            model_name=model_name,
            x_sep_error=x_sep_error,
            x_reat_error=x_reat_error,
            Cp_rmse=Cp_rmse,
            Cf_rmse=Cf_rmse,
            composite_score=composite,
        ))

    def add_vv_result(
        self,
        case_name: str,
        validation_level: int,
        status: str,
        error_metric: float = 0.0,
        within_uncertainty: bool = False,
        mrr_level: int = 0,
    ) -> None:
        """Add V&V validation result."""
        self.vv_results.append(VVEntry(
            case_name=case_name,
            validation_level=validation_level,
            status=status,
            error_metric=error_metric,
            within_uncertainty=within_uncertainty,
            mrr_level=mrr_level,
        ))

    def add_challenge_progress(
        self,
        metric: str,
        baseline_error: float,
        current_error: float,
    ) -> None:
        """Track NASA 40% error reduction challenge."""
        if abs(baseline_error) > 1e-15:
            reduction = (baseline_error - current_error) / baseline_error * 100
        else:
            reduction = 0.0
        self.challenge_progress.append(ChallengeEntry(
            metric=metric,
            baseline_error=baseline_error,
            current_error=current_error,
            reduction_pct=reduction,
            target_met=reduction >= 40.0,
        ))

    def add_scheme_sensitivity(
        self,
        quantity: str,
        cv_pct: float,
        spread: float,
    ) -> None:
        """Add numerical scheme sensitivity result."""
        self.scheme_sensitivity.append(SchemeSensEntry(
            quantity=quantity,
            cv_pct=cv_pct,
            spread=spread,
            scheme_converged=cv_pct < 5.0,
        ))

    def add_experimental_comparison(
        self,
        case_name: str,
        metrics: Dict[str, float],
    ) -> None:
        """Add experimental comparison metrics for a case."""
        self.experimental_comparisons[case_name] = metrics

    def add_note(self, note: str) -> None:
        """Add an analyst note to the report."""
        self.notes.append(note)

    # =========================================================================
    # Ranking Computation
    # =========================================================================
    def _compute_rankings(self) -> Dict[str, List[ModelRankEntry]]:
        """Compute per-case model rankings by composite score."""
        rankings: Dict[str, List[ModelRankEntry]] = {}
        for entry in self.model_rankings:
            if entry.case_name not in rankings:
                rankings[entry.case_name] = []
            rankings[entry.case_name].append(entry)

        for case_entries in rankings.values():
            case_entries.sort(key=lambda e: e.composite_score)
            for i, entry in enumerate(case_entries):
                entry.rank = i + 1

        return rankings

    # =========================================================================
    # Markdown Report Generation
    # =========================================================================
    def generate_markdown(self, output_path: Optional[str] = None) -> str:
        """
        Generate comprehensive Markdown benchmark report.

        Parameters
        ----------
        output_path : str, optional
            If provided, write report to this file.

        Returns
        -------
        str : The Markdown content.
        """
        sections = []
        sections.append(f"# {self.title}\n")
        sections.append(f"Generated: {self.timestamp}\n")

        # Executive Summary
        sections.append(self._section_executive_summary())

        # GCI Results
        if self.gci_results:
            sections.append(self._section_gci())

        # Scheme Sensitivity
        if self.scheme_sensitivity:
            sections.append(self._section_scheme_sensitivity())

        # Model Rankings
        if self.model_rankings:
            sections.append(self._section_model_rankings())

        # V&V Status
        if self.vv_results:
            sections.append(self._section_vv_status())

        # 40% Challenge
        if self.challenge_progress:
            sections.append(self._section_challenge())

        # Experimental Comparisons
        if self.experimental_comparisons:
            sections.append(self._section_experimental())

        # Notes
        if self.notes:
            sections.append(self._section_notes())

        content = "\n".join(sections)

        if output_path:
            Path(output_path).write_text(content, encoding="utf-8")
            logger.info(f"Report written to {output_path}")

        return content

    def _section_executive_summary(self) -> str:
        """Generate executive summary."""
        lines = ["## Executive Summary\n"]

        n_gci = len(self.gci_results)
        n_asym = sum(1 for g in self.gci_results if g.in_asymptotic_range)
        n_models = len(set(e.model_name for e in self.model_rankings))
        n_cases = len(set(e.case_name for e in self.model_rankings))
        n_vv_pass = sum(1 for v in self.vv_results if v.status == "PASS")
        n_challenge_met = sum(1 for c in self.challenge_progress if c.target_met)

        lines.append(f"- **Grid studies**: {n_gci} completed, "
                      f"{n_asym}/{n_gci} in asymptotic range")
        if n_models > 0:
            lines.append(f"- **Models evaluated**: {n_models} across {n_cases} cases")
        if self.vv_results:
            lines.append(f"- **Validation**: {n_vv_pass}/{len(self.vv_results)} passed")
        if self.challenge_progress:
            lines.append(f"- **40% Challenge**: {n_challenge_met}/"
                          f"{len(self.challenge_progress)} targets met")

        lines.append("")
        return "\n".join(lines)

    def _section_gci(self) -> str:
        """Generate GCI results table."""
        lines = [
            "## Grid Convergence Index (GCI)\n",
            "| Case | Quantity | GCI_fine (%) | GCI_coarse (%) | "
            "p_obs | Asymptotic? |",
            "|---|---|---|---|---|---|",
        ]
        for g in self.gci_results:
            asym = "✓" if g.in_asymptotic_range else "✗"
            lines.append(
                f"| {g.case_name} | {g.quantity} | "
                f"{g.gci_fine:.2f} | {g.gci_coarse:.2f} | "
                f"{g.observed_order:.2f} | {asym} |"
            )
        lines.append("")
        return "\n".join(lines)

    def _section_scheme_sensitivity(self) -> str:
        """Generate scheme sensitivity table."""
        lines = [
            "## Numerical Scheme Sensitivity\n",
            "| Quantity | CV (%) | Spread | Converged? |",
            "|---|---|---|---|",
        ]
        for s in self.scheme_sensitivity:
            conv = "✓" if s.scheme_converged else "⚠ Refine mesh"
            lines.append(
                f"| {s.quantity} | {s.cv_pct:.2f} | "
                f"{s.spread:.4f} | {conv} |"
            )
        lines.append("")
        return "\n".join(lines)

    def _section_model_rankings(self) -> str:
        """Generate model ranking tables per case."""
        rankings = self._compute_rankings()
        lines = ["## Turbulence Model Rankings\n"]

        for case_name, entries in rankings.items():
            lines.append(f"### {case_name}\n")
            lines.append(
                "| Rank | Model | x_sep err | x_reat err | "
                "Cp RMSE | Cf RMSE | Score |"
            )
            lines.append("|---|---|---|---|---|---|---|")
            for e in entries:
                lines.append(
                    f"| {e.rank} | {e.model_name} | "
                    f"{e.x_sep_error:+.4f} | {e.x_reat_error:+.4f} | "
                    f"{e.Cp_rmse:.4f} | {e.Cf_rmse:.4f} | "
                    f"{e.composite_score:.4f} |"
                )
            lines.append("")

        return "\n".join(lines)

    def _section_vv_status(self) -> str:
        """Generate V&V status table."""
        lines = [
            "## Verification & Validation Status\n",
            "| Case | Level | Status | Error | Within σ? | MRR |",
            "|---|---|---|---|---|---|",
        ]
        for v in self.vv_results:
            icon = {"PASS": "✓", "FAIL": "✗", "PARTIAL": "◐"}.get(
                v.status, "?"
            )
            unc = "✓" if v.within_uncertainty else "✗"
            lines.append(
                f"| {v.case_name} | {v.validation_level} | "
                f"{icon} {v.status} | {v.error_metric:.4f} | {unc} | "
                f"{v.mrr_level} |"
            )
        lines.append("")
        return "\n".join(lines)

    def _section_challenge(self) -> str:
        """Generate 40% Challenge progress table."""
        lines = [
            "## NASA 40% Error Reduction Challenge\n",
            "| Metric | Baseline | Current | Reduction | Target? |",
            "|---|---|---|---|---|",
        ]
        for c in self.challenge_progress:
            met = "✓ MET" if c.target_met else f"✗ ({c.reduction_pct:.1f}%)"
            lines.append(
                f"| {c.metric} | {c.baseline_error:.4f} | "
                f"{c.current_error:.4f} | {c.reduction_pct:.1f}% | {met} |"
            )
        lines.append("")
        return "\n".join(lines)

    def _section_experimental(self) -> str:
        """Generate experimental comparison summary."""
        lines = ["## Experimental Comparisons\n"]

        for case_name, metrics in self.experimental_comparisons.items():
            lines.append(f"### {case_name}\n")
            for metric_name, value in metrics.items():
                lines.append(f"- **{metric_name}**: {value:.4f}")
            lines.append("")

        return "\n".join(lines)

    def _section_notes(self) -> str:
        """Generate analyst notes section."""
        lines = ["## Analyst Notes\n"]
        for i, note in enumerate(self.notes, 1):
            lines.append(f"{i}. {note}")
        lines.append("")
        return "\n".join(lines)

    # =========================================================================
    # JSON Report Generation
    # =========================================================================
    def generate_json(self, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate machine-readable JSON summary.

        Parameters
        ----------
        output_path : str, optional
            If provided, write JSON to this file.

        Returns
        -------
        dict : Report data.
        """
        rankings = self._compute_rankings()

        data = {
            "title": self.title,
            "timestamp": self.timestamp,
            "gci": [asdict(g) for g in self.gci_results],
            "scheme_sensitivity": [asdict(s) for s in self.scheme_sensitivity],
            "model_rankings": {
                case: [asdict(e) for e in entries]
                for case, entries in rankings.items()
            },
            "vv_status": [asdict(v) for v in self.vv_results],
            "challenge_40pct": [asdict(c) for c in self.challenge_progress],
            "experimental_comparisons": self.experimental_comparisons,
            "notes": self.notes,
            "summary": {
                "n_gci_studies": len(self.gci_results),
                "n_models": len(set(e.model_name for e in self.model_rankings)),
                "n_cases": len(set(e.case_name for e in self.model_rankings)),
                "n_vv_pass": sum(1 for v in self.vv_results if v.status == "PASS"),
                "n_challenge_met": sum(
                    1 for c in self.challenge_progress if c.target_met
                ),
            },
        }

        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
            logger.info(f"JSON report written to {output_path}")

        return data


# =============================================================================
# Convenience Function
# =============================================================================
def print_report_summary(report: BenchmarkReportGenerator) -> None:
    """Print a quick console summary of the report."""
    print(f"\n{'='*65}")
    print(f"  {report.title}")
    print(f"  {report.timestamp}")
    print(f"{'='*65}")

    if report.gci_results:
        n_ok = sum(1 for g in report.gci_results if g.in_asymptotic_range)
        print(f"\n  Grid Studies: {n_ok}/{len(report.gci_results)} in asymptotic range")

    if report.model_rankings:
        rankings = report._compute_rankings()
        for case, entries in rankings.items():
            best = entries[0]
            print(f"\n  Best model for {case}: {best.model_name} "
                  f"(score={best.composite_score:.4f})")

    if report.challenge_progress:
        for c in report.challenge_progress:
            icon = "[PASS]" if c.target_met else "[FAIL]"
            print(f"\n  {icon} {c.metric}: {c.reduction_pct:.1f}% reduction")

    print(f"\n{'='*65}")
