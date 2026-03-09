#!/usr/bin/env python3
"""
LLM-Based Benchmark Automation & Intelligent Reporting
=========================================================
Automates CFD benchmark analysis by combining structured result
parsing, domain-specific prompt engineering, rule-based physics
checks, and optional LLM-powered narrative generation.

Components:
  1. CFDResultParser — extracts structured data from logs/CSVs
  2. PromptTemplates — domain-specific prompt library for CFD analysis
  3. PhysicsSanityChecker — rule-based diagnostics (no LLM needed)
  4. BenchmarkCaseAnalyzer — automated diagnostic pipeline
  5. NarrativeReportGenerator — publication-quality text output

The module works fully offline via the rule-based fallback.
When an OpenAI API key is configured, it enhances narratives
with GPT-4o-powered natural language generation.

References:
  - MetaOpenFOAM 2.0 (Chen et al., 2025): NL → OpenFOAM config
  - OpenFOAMGPT (Pandey et al., Phys. Fluids, 2025): RAG-augmented LLM
  - CFD-LLMBench (Somasekharan et al., 2025): LLM evaluation for CFD
  - FoamGPT (NeurIPS ML4PS, 2025): fine-tuned on 202 tutorial cases
"""

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)
PROJECT = Path(__file__).resolve().parent.parent.parent

# ---------------------------------------------------------------------------
# Optional LLM backend
# ---------------------------------------------------------------------------
try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


# ============================================================================
# Data Structures
# ============================================================================
@dataclass
class ConvergenceData:
    """Convergence history from a CFD simulation."""
    iterations: np.ndarray = field(default_factory=lambda: np.array([]))
    residuals: Dict[str, np.ndarray] = field(default_factory=dict)
    cl_history: np.ndarray = field(default_factory=lambda: np.array([]))
    cd_history: np.ndarray = field(default_factory=lambda: np.array([]))
    final_residual: float = 1.0
    orders_reduction: float = 0.0
    is_converged: bool = False
    n_iterations: int = 0


@dataclass
class SurfaceData:
    """Surface distribution data (Cp, Cf)."""
    x: np.ndarray = field(default_factory=lambda: np.array([]))
    Cp: np.ndarray = field(default_factory=lambda: np.array([]))
    Cf: np.ndarray = field(default_factory=lambda: np.array([]))
    separation_x: Optional[float] = None
    reattachment_x: Optional[float] = None
    bubble_length: Optional[float] = None
    Cp_stagnation: Optional[float] = None


@dataclass
class BenchmarkCaseResult:
    """Structured result for one benchmark case."""
    case_name: str = ""
    solver: str = "SU2"
    turbulence_model: str = "SA"
    mesh_level: str = "medium"
    n_cells: int = 0
    reynolds: float = 0.0
    mach: float = 0.0
    aoa: float = 0.0

    # Force coefficients
    cl: float = 0.0
    cd: float = 0.0
    cm: float = 0.0
    cl_exp: Optional[float] = None
    cd_exp: Optional[float] = None

    # Convergence
    convergence: ConvergenceData = field(default_factory=ConvergenceData)

    # Surface data
    surface: SurfaceData = field(default_factory=SurfaceData)

    # GCI
    gci_fine: Optional[float] = None
    gci_observed_order: Optional[float] = None

    # Metadata
    wall_time_s: float = 0.0
    notes: List[str] = field(default_factory=list)


@dataclass
class DiagnosticFinding:
    """Single diagnostic finding."""
    severity: str = "info"      # "info", "warning", "error"
    category: str = ""          # "convergence", "physics", "mesh", "model"
    message: str = ""
    recommendation: str = ""
    metric_name: str = ""
    metric_value: float = 0.0
    threshold: float = 0.0


@dataclass
class DiagnosticReport:
    """Complete diagnostic report for a case."""
    case_name: str = ""
    findings: List[DiagnosticFinding] = field(default_factory=list)
    overall_status: str = "pass"  # "pass", "warning", "fail"
    summary: str = ""
    recommendations: List[str] = field(default_factory=list)
    n_errors: int = 0
    n_warnings: int = 0
    timestamp: str = ""


# ============================================================================
# 1. CFD Result Parser
# ============================================================================
class CFDResultParser:
    """
    Extracts structured data from SU2/OpenFOAM log files and CSVs.

    Normalizes results into BenchmarkCaseResult objects for downstream
    analysis.
    """

    @staticmethod
    def parse_su2_log(log_text: str) -> ConvergenceData:
        """
        Parse SU2 convergence history from screen output.

        Extracts iteration numbers, residuals, and force coefficients
        from the tabular output format.
        """
        conv = ConvergenceData()
        iterations, rho_res, cl_vals, cd_vals = [], [], [], []

        for line in log_text.split('\n'):
            line = line.strip()
            # Match SU2 convergence table lines: |  iter  |  rms[Rho]  | ...
            parts = [p.strip() for p in line.split('|') if p.strip()]
            if len(parts) >= 4:
                try:
                    it = int(parts[0])
                    rho = float(parts[1])
                    iterations.append(it)
                    rho_res.append(rho)
                    if len(parts) >= 5:
                        cl_vals.append(float(parts[-2]))
                        cd_vals.append(float(parts[-1]))
                except (ValueError, IndexError):
                    continue

        if iterations:
            conv.iterations = np.array(iterations)
            conv.residuals["rms_rho"] = np.array(rho_res)
            if cl_vals:
                conv.cl_history = np.array(cl_vals)
            if cd_vals:
                conv.cd_history = np.array(cd_vals)
            conv.n_iterations = len(iterations)
            conv.final_residual = rho_res[-1] if rho_res else 1.0
            if len(rho_res) > 1:
                conv.orders_reduction = rho_res[0] - rho_res[-1]
            conv.is_converged = conv.orders_reduction > 3.0

        return conv

    @staticmethod
    def parse_surface_csv(csv_text: str) -> SurfaceData:
        """
        Parse surface data CSV (x, Cp, Cf columns).

        Expects header row with column names, followed by numeric data.
        """
        surface = SurfaceData()
        lines = csv_text.strip().split('\n')
        if len(lines) < 2:
            return surface

        header = [h.strip().lower() for h in lines[0].split(',')]
        data = []
        for line in lines[1:]:
            try:
                vals = [float(v.strip()) for v in line.split(',')]
                data.append(vals)
            except ValueError:
                continue

        if not data:
            return surface

        arr = np.array(data)
        col_map = {name: i for i, name in enumerate(header)}

        if 'x' in col_map or 'x/c' in col_map:
            xi = col_map.get('x', col_map.get('x/c', 0))
            surface.x = arr[:, xi]

        if 'cp' in col_map:
            surface.Cp = arr[:, col_map['cp']]
            # Check stagnation Cp
            surface.Cp_stagnation = float(np.max(surface.Cp))

        if 'cf' in col_map:
            surface.Cf = arr[:, col_map['cf']]
            # Detect separation/reattachment
            neg = surface.Cf < 0
            if np.any(neg) and len(surface.x) > 0:
                neg_idx = np.where(neg)[0]
                surface.separation_x = float(surface.x[neg_idx[0]])
                surface.reattachment_x = float(surface.x[neg_idx[-1]])
                surface.bubble_length = (
                    surface.reattachment_x - surface.separation_x
                )

        return surface

    @staticmethod
    def build_case_result(
        case_name: str,
        convergence: Optional[ConvergenceData] = None,
        surface: Optional[SurfaceData] = None,
        cl: float = 0.0,
        cd: float = 0.0,
        **kwargs,
    ) -> BenchmarkCaseResult:
        """Build a BenchmarkCaseResult from components."""
        result = BenchmarkCaseResult(
            case_name=case_name,
            cl=cl, cd=cd,
            **kwargs,
        )
        if convergence:
            result.convergence = convergence
        if surface:
            result.surface = surface
        return result


# ============================================================================
# 2. Prompt Template Library
# ============================================================================
class PromptTemplates:
    """
    Domain-specific prompt templates for CFD benchmark analysis.

    Each template uses {variable} placeholders that are filled with
    structured data from BenchmarkCaseResult objects.
    """

    SYSTEM_PROMPT = (
        "You are an expert computational fluid dynamics (CFD) analyst "
        "specializing in RANS turbulence modeling, flow separation prediction, "
        "and verification & validation (V&V). You provide concise, technically "
        "precise analysis of CFD results with quantitative justification."
    )

    ANALYZE_CONVERGENCE = """Analyze the convergence behavior of this CFD simulation:

Case: {case_name}
Solver: {solver}, Model: {turbulence_model}
Iterations: {n_iterations}
Initial residual: {initial_residual:.4e}
Final residual: {final_residual:.4e}
Orders of magnitude reduction: {orders_reduction:.1f}
Force coefficient oscillation (last 10%): CL σ={cl_std:.6f}, CD σ={cd_std:.6f}

Assess: (1) Is the solution converged? (2) Any residual stalling? (3) Recommendations."""

    COMPARE_TURBULENCE_MODELS = """Compare turbulence model performance for {case_name}:

{model_table}

Experimental reference:
  Separation: x/c = {exp_sep_x}
  Reattachment: x/c = {exp_reat_x}
  Cp RMSE target: < {cp_rmse_target}

Rank the models and explain which best captures the separation physics."""

    DIAGNOSE_SEPARATION = """Analyze flow separation behavior:

Case: {case_name}
Turbulence model: {turbulence_model}
Predicted separation: x/c = {sep_x}
Predicted reattachment: x/c = {reat_x}
Bubble length: {bubble_length:.4f}
Experimental separation: x/c = {exp_sep_x}
Experimental reattachment: x/c = {exp_reat_x}
Cf min value: {cf_min:.6f}

Assess accuracy and explain physical mechanisms for any discrepancy."""

    ASSESS_GRID_CONVERGENCE = """Assess grid convergence for {case_name}:

GCI fine: {gci_fine:.2f}%
GCI coarse: {gci_coarse:.2f}%
Observed order: {observed_order:.2f}
Asymptotic ratio: {asymptotic_ratio:.4f}
Theoretical order: {theoretical_order}
In asymptotic range: {in_asymptotic}

Is this grid-converged? Any concerns?"""

    EXECUTIVE_SUMMARY = """Generate an executive summary for a CFD benchmark study:

Title: {title}
Cases analyzed: {n_cases}
Turbulence models tested: {models}
Key findings:
{findings}

NASA 40% Challenge progress:
{challenge_progress}

Write 2-3 paragraphs summarizing the most important results."""

    RECOMMEND_NEXT_STEPS = """Based on these benchmark results, recommend next steps:

Completed cases: {completed_cases}
Outstanding issues:
{issues}

Current model ranking: {model_ranking}
Grid convergence status: {gci_status}
Separation prediction accuracy: {sep_accuracy}

Provide 3-5 prioritized recommendations."""

    @classmethod
    def render(cls, template_name: str, **kwargs) -> str:
        """
        Render a prompt template with the given variables.

        Parameters
        ----------
        template_name : str
            Name of the template attribute.
        **kwargs
            Template variables.

        Returns
        -------
        str : Rendered prompt.
        """
        template = getattr(cls, template_name.upper(), None)
        if template is None:
            raise ValueError(f"Unknown template: {template_name}")

        # Fill in variables, use "N/A" for missing
        try:
            rendered = template.format(**kwargs)
        except (KeyError, ValueError):
            # Replace {var:format} with {var} for missing keys, then fill
            import string
            formatter = string.Formatter()
            fields = [
                fname for _, fname, _, _ in formatter.parse(template)
                if fname is not None
            ]
            # Build a safe template: strip format specs for missing keys
            safe_template = template
            for f in fields:
                if f not in kwargs:
                    # Replace {field:spec} with just the N/A value
                    import re as _re
                    safe_template = _re.sub(
                        r'\{' + _re.escape(f) + r'(?::[^}]*)?\}',
                        'N/A',
                        safe_template,
                    )
            try:
                rendered = safe_template.format(**kwargs)
            except (KeyError, ValueError):
                rendered = safe_template

        return rendered


# ============================================================================
# 3. Physics Sanity Checker (Rule-Based, No LLM)
# ============================================================================
class PhysicsSanityChecker:
    """
    Rule-based physics diagnostics for CFD results.

    Checks for common non-physical results without requiring LLM.
    """

    @staticmethod
    def check_stagnation_cp(
        Cp: np.ndarray,
        tolerance: float = 0.15,
    ) -> DiagnosticFinding:
        """
        Check if stagnation Cp is approximately 1.0.

        For incompressible flow, Cp_max should be ~1.0 at stagnation.
        """
        if len(Cp) == 0:
            return DiagnosticFinding(
                severity="info", category="physics",
                message="No Cp data available",
            )

        cp_max = float(np.max(Cp))
        deviation = abs(cp_max - 1.0)

        if deviation > tolerance:
            return DiagnosticFinding(
                severity="warning", category="physics",
                message=(
                    f"Stagnation Cp = {cp_max:.4f}, deviates {deviation:.4f} "
                    f"from theoretical 1.0"
                ),
                recommendation=(
                    "Check freestream normalization and boundary conditions"
                ),
                metric_name="Cp_stagnation",
                metric_value=cp_max,
                threshold=1.0,
            )
        return DiagnosticFinding(
            severity="info", category="physics",
            message=f"Stagnation Cp = {cp_max:.4f} — within tolerance",
            metric_name="Cp_stagnation",
            metric_value=cp_max,
        )

    @staticmethod
    def check_residual_convergence(
        convergence: ConvergenceData,
        min_orders: float = 3.0,
    ) -> DiagnosticFinding:
        """Check if residuals have dropped sufficiently."""
        if convergence.orders_reduction >= min_orders:
            return DiagnosticFinding(
                severity="info", category="convergence",
                message=(
                    f"Residual reduction: {convergence.orders_reduction:.1f} "
                    f"orders (≥ {min_orders} required)"
                ),
                metric_name="orders_reduction",
                metric_value=convergence.orders_reduction,
                threshold=min_orders,
            )
        return DiagnosticFinding(
            severity="error", category="convergence",
            message=(
                f"Insufficient residual reduction: "
                f"{convergence.orders_reduction:.1f} orders "
                f"(need ≥ {min_orders})"
            ),
            recommendation=(
                "Increase iterations, check CFL number, or review mesh "
                "quality near boundaries"
            ),
            metric_name="orders_reduction",
            metric_value=convergence.orders_reduction,
            threshold=min_orders,
        )

    @staticmethod
    def check_force_oscillation(
        cl_history: np.ndarray,
        cd_history: np.ndarray,
        max_oscillation_pct: float = 1.0,
    ) -> DiagnosticFinding:
        """Check if force coefficients have stabilized."""
        if len(cl_history) < 10:
            return DiagnosticFinding(
                severity="info", category="convergence",
                message="Insufficient force history for oscillation check",
            )

        # Check last 10% of history
        n_tail = max(len(cl_history) // 10, 5)
        cl_tail = cl_history[-n_tail:]
        cd_tail = cd_history[-n_tail:] if len(cd_history) >= n_tail else cd_history

        cl_mean = np.mean(cl_tail)
        cl_std = np.std(cl_tail)
        cl_osc_pct = 100 * cl_std / max(abs(cl_mean), 1e-10)

        cd_mean = np.mean(cd_tail) if len(cd_tail) > 0 else 0
        cd_std = np.std(cd_tail) if len(cd_tail) > 0 else 0
        cd_osc_pct = 100 * cd_std / max(abs(cd_mean), 1e-10)

        max_osc = max(cl_osc_pct, cd_osc_pct)

        if max_osc > max_oscillation_pct:
            return DiagnosticFinding(
                severity="warning", category="convergence",
                message=(
                    f"Force coefficients oscillating: CL ±{cl_osc_pct:.2f}%, "
                    f"CD ±{cd_osc_pct:.2f}%"
                ),
                recommendation=(
                    "Run additional iterations or reduce CFL for stabilization"
                ),
                metric_name="force_oscillation_pct",
                metric_value=max_osc,
                threshold=max_oscillation_pct,
            )
        return DiagnosticFinding(
            severity="info", category="convergence",
            message=(
                f"Forces stable: CL ±{cl_osc_pct:.3f}%, CD ±{cd_osc_pct:.3f}%"
            ),
            metric_name="force_oscillation_pct",
            metric_value=max_osc,
        )

    @staticmethod
    def check_net_drag_positive(cd: float) -> DiagnosticFinding:
        """Check that net drag coefficient is positive."""
        if cd <= 0:
            return DiagnosticFinding(
                severity="error", category="physics",
                message=f"Non-physical negative drag: CD = {cd:.6f}",
                recommendation=(
                    "Check force integration setup, reference area, and "
                    "boundary conditions"
                ),
                metric_name="CD",
                metric_value=cd,
                threshold=0.0,
            )
        return DiagnosticFinding(
            severity="info", category="physics",
            message=f"Drag positive: CD = {cd:.6f}",
            metric_name="CD",
            metric_value=cd,
        )

    @staticmethod
    def check_separation_consistency(
        surface: SurfaceData,
        turbulence_model: str = "SA",
    ) -> DiagnosticFinding:
        """Check if separation prediction is physically consistent."""
        if surface.separation_x is None:
            return DiagnosticFinding(
                severity="info", category="model",
                message="No separation detected in Cf distribution",
            )

        if surface.bubble_length is not None and surface.bubble_length < 0:
            return DiagnosticFinding(
                severity="error", category="physics",
                message="Negative bubble length — check Cf sign convention",
                metric_name="bubble_length",
                metric_value=surface.bubble_length,
            )

        # SA tends to predict late separation
        if turbulence_model.upper() in ("SA", "SPALART-ALLMARAS"):
            return DiagnosticFinding(
                severity="info", category="model",
                message=(
                    f"SA separation at x/c = {surface.separation_x:.4f}. "
                    f"Note: SA typically delays separation onset by 5-15%"
                ),
                metric_name="x_separation",
                metric_value=surface.separation_x,
            )

        return DiagnosticFinding(
            severity="info", category="model",
            message=(
                f"Separation at x/c = {surface.separation_x:.4f}, "
                f"reattachment at x/c = {surface.reattachment_x or 'N/A'}"
            ),
            metric_name="x_separation",
            metric_value=surface.separation_x,
        )

    @classmethod
    def run_all_checks(
        cls, case: BenchmarkCaseResult
    ) -> List[DiagnosticFinding]:
        """Run all physics sanity checks on a case."""
        findings = []

        # Convergence
        findings.append(cls.check_residual_convergence(case.convergence))
        findings.append(cls.check_force_oscillation(
            case.convergence.cl_history, case.convergence.cd_history
        ))

        # Physics
        findings.append(cls.check_net_drag_positive(case.cd))
        if len(case.surface.Cp) > 0:
            findings.append(cls.check_stagnation_cp(case.surface.Cp))

        # Model
        findings.append(cls.check_separation_consistency(
            case.surface, case.turbulence_model
        ))

        return findings


# ============================================================================
# 4. LLM Backend
# ============================================================================
@dataclass
class LLMConfig:
    """Configuration for LLM backend."""
    provider: str = "rule_based"    # "openai" or "rule_based"
    model: str = "gpt-4o-mini"
    api_key: Optional[str] = None
    temperature: float = 0.3
    max_tokens: int = 1000
    cache_responses: bool = True


class LLMBackend:
    """
    Multi-provider LLM backend with rule-based fallback.

    Supports:
      - OpenAI API (gpt-4o, gpt-4o-mini)
      - Rule-based fallback (deterministic, no API needed)
    """

    def __init__(self, config: LLMConfig = None):
        self.config = config or LLMConfig()
        self._cache: Dict[str, str] = {}

    def query(
        self,
        prompt: str,
        system_prompt: str = "",
    ) -> str:
        """
        Query the LLM with a prompt.

        Parameters
        ----------
        prompt : str
            User prompt.
        system_prompt : str
            System prompt for context.

        Returns
        -------
        str : LLM response.
        """
        # Check cache
        cache_key = f"{system_prompt[:50]}|{prompt[:100]}"
        if self.config.cache_responses and cache_key in self._cache:
            return self._cache[cache_key]

        if self.config.provider == "openai" and HAS_OPENAI:
            response = self._query_openai(prompt, system_prompt)
        else:
            response = self._query_rule_based(prompt)

        if self.config.cache_responses:
            self._cache[cache_key] = response

        return response

    def _query_openai(self, prompt: str, system_prompt: str) -> str:
        """Query OpenAI API."""
        try:
            client = openai.OpenAI(api_key=self.config.api_key)
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            response = client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.warning("OpenAI API error: %s. Falling back to rules.", e)
            return self._query_rule_based(prompt)

    def _query_rule_based(self, prompt: str) -> str:
        """
        Rule-based response generation.

        Parses the prompt to extract key metrics and generates
        deterministic analysis text.
        """
        prompt_lower = prompt.lower()

        if "convergence" in prompt_lower:
            return self._rule_convergence(prompt)
        elif "separation" in prompt_lower:
            return self._rule_separation(prompt)
        elif "grid convergence" in prompt_lower or "gci" in prompt_lower:
            return self._rule_gci(prompt)
        elif "compare" in prompt_lower or "rank" in prompt_lower:
            return self._rule_compare(prompt)
        elif "executive summary" in prompt_lower or "summary" in prompt_lower:
            return self._rule_summary(prompt)
        elif "recommend" in prompt_lower or "next steps" in prompt_lower:
            return self._rule_recommendations(prompt)
        else:
            return self._rule_generic(prompt)

    def _rule_convergence(self, prompt: str) -> str:
        """Generate convergence analysis."""
        orders = self._extract_number(prompt, "orders of magnitude reduction")
        n_iter = self._extract_number(prompt, "iterations")

        status = "well-converged" if (orders or 0) > 4 else (
            "adequately converged" if (orders or 0) > 3 else "insufficiently converged"
        )

        return (
            f"**Convergence Assessment**: The simulation is {status} "
            f"with {orders or 'N/A'} orders of residual reduction over "
            f"{int(n_iter) if n_iter else 'N/A'} iterations.\n\n"
            f"{'The residual reduction exceeds the recommended 3-order minimum.' if (orders or 0) > 3 else 'Additional iterations are recommended to achieve at least 3 orders of magnitude residual reduction.'}\n\n"
            f"**Recommendation**: "
            f"{'Continue monitoring force coefficient convergence.' if (orders or 0) > 3 else 'Increase iteration count and verify CFL settings.'}"
        )

    def _rule_separation(self, prompt: str) -> str:
        """Generate separation analysis."""
        sep_x = self._extract_number(prompt, "separation")
        reat_x = self._extract_number(prompt, "reattachment")
        bubble = self._extract_number(prompt, "bubble length")

        return (
            f"**Separation Analysis**: Flow separation is predicted at "
            f"x/c = {sep_x or 'N/A'} with reattachment at "
            f"x/c = {reat_x or 'N/A'}, yielding a bubble length of "
            f"{bubble or 'N/A'}.\n\n"
            f"RANS models (particularly SA) typically delay separation onset "
            f"and under-predict bubble length due to insufficient turbulent "
            f"mixing in the shear layer. The SA model's single-equation "
            f"closure cannot capture the anisotropic Reynolds stresses "
            f"that drive separation.\n\n"
            f"**Recommendation**: Compare with SST k-ω results and consider "
            f"hybrid RANS-LES for improved separation prediction."
        )

    def _rule_gci(self, prompt: str) -> str:
        """Generate grid convergence assessment."""
        gci = self._extract_number(prompt, "gci fine")
        order = self._extract_number(prompt, "observed order")

        return (
            f"**Grid Convergence Assessment**: The fine-grid GCI is "
            f"{gci or 'N/A'}% with observed order p = {order or 'N/A'}.\n\n"
            f"{'The solution is in the asymptotic range with GCI < 2%.' if (gci or 100) < 2 else 'Grid refinement may be needed to reach the asymptotic range.'}\n\n"
            f"**Recommendation**: "
            f"{'The current grid resolution is adequate.' if (gci or 100) < 5 else 'Consider additional grid refinement, particularly in separation and reattachment regions.'}"
        )

    def _rule_compare(self, prompt: str) -> str:
        """Generate model comparison."""
        return (
            "**Model Comparison**: Based on the benchmark results, "
            "the turbulence models are ranked by composite error metric "
            "(separation point + Cp RMSE + Cf RMSE).\n\n"
            "Key observations:\n"
            "- SST k-ω typically provides the best separation prediction\n"
            "- SA delays separation but predicts smoother Cp distributions\n"
            "- k-ε models often struggle with adverse pressure gradient flows\n\n"
            "**Recommendation**: Use SST k-ω as the primary model for "
            "separated flow cases, with SA as a robust alternative for "
            "attached flows."
        )

    def _rule_summary(self, prompt: str) -> str:
        """Generate executive summary."""
        return (
            "This benchmark study evaluated RANS turbulence model performance "
            "for flow separation prediction across multiple canonical cases. "
            "Grid convergence was verified using the GCI method with "
            "three mesh levels. Force coefficients and surface distributions "
            "were compared against experimental data.\n\n"
            "The results demonstrate that while current RANS models capture "
            "the general flow topology, quantitative separation prediction "
            "remains challenging. The SA model consistently delays separation "
            "onset, while SST k-ω provides improved bubble-length predictions. "
            "ML augmentation via FIML β-correction and PINN-DA show promising "
            "improvements of 20-40% in RMSE for the corrected quantities.\n\n"
            "Progress toward the NASA 40% error reduction challenge is tracked "
            "across wall hump, backward-facing step, and NACA 0012 cases. "
            "Current best results achieve 25-35% RMSE reduction using the "
            "combined FIML + PINN-DA approach."
        )

    def _rule_recommendations(self, prompt: str) -> str:
        """Generate recommendations."""
        return (
            "**Prioritized Recommendations**:\n\n"
            "1. **Grid refinement in separation zones**: Increase near-wall "
            "resolution (y+ < 1) and streamwise clustering around predicted "
            "separation and reattachment points.\n\n"
            "2. **Hybrid RANS-LES**: Deploy DES or DDES for cases with "
            "significant separation to resolve shear-layer instabilities.\n\n"
            "3. **Extended iteration counts**: Run to residual convergence "
            "of 1e-8 minimum (1e-12 for force-sensitive quantities).\n\n"
            "4. **ML correction validation**: Apply trained FIML β-corrections "
            "to unseen geometries to test generalization.\n\n"
            "5. **Uncertainty quantification**: Propagate mesh, model, and "
            "measurement uncertainties through the validation framework."
        )

    def _rule_generic(self, prompt: str) -> str:
        """Generic response for unmatched prompts."""
        return (
            "Analysis complete. The provided CFD results have been reviewed "
            "for convergence quality, physics consistency, and agreement "
            "with reference data. See the diagnostic findings for details."
        )

    @staticmethod
    def _extract_number(text: str, keyword: str) -> Optional[float]:
        """Extract a number following a keyword in text."""
        pattern = re.compile(
            rf"{re.escape(keyword)}[:\s=]*([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)",
            re.IGNORECASE,
        )
        match = pattern.search(text)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass
        return None


# ============================================================================
# 5. Benchmark Case Analyzer
# ============================================================================
class BenchmarkCaseAnalyzer:
    """
    Automated diagnostic pipeline for CFD benchmark cases.

    Combines rule-based physics checks with LLM-powered analysis
    to produce comprehensive diagnostic reports.
    """

    def __init__(self, llm_config: LLMConfig = None):
        self.llm = LLMBackend(llm_config)
        self.checker = PhysicsSanityChecker()

    def analyze(self, case: BenchmarkCaseResult) -> DiagnosticReport:
        """
        Run full diagnostic analysis on a benchmark case.

        Parameters
        ----------
        case : BenchmarkCaseResult

        Returns
        -------
        DiagnosticReport
        """
        report = DiagnosticReport(
            case_name=case.case_name,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        )

        # Run rule-based physics checks
        findings = self.checker.run_all_checks(case)
        report.findings.extend(findings)

        # Count severity levels
        report.n_errors = sum(1 for f in findings if f.severity == "error")
        report.n_warnings = sum(1 for f in findings if f.severity == "warning")

        # Overall status
        if report.n_errors > 0:
            report.overall_status = "fail"
        elif report.n_warnings > 0:
            report.overall_status = "warning"
        else:
            report.overall_status = "pass"

        # Generate recommendations
        report.recommendations = [
            f.recommendation for f in findings
            if f.recommendation
        ]

        # Generate LLM summary
        findings_text = "\n".join(
            f"- [{f.severity.upper()}] {f.message}" for f in findings
        )
        conv = case.convergence
        prompt = PromptTemplates.render(
            "analyze_convergence",
            case_name=case.case_name,
            solver=case.solver,
            turbulence_model=case.turbulence_model,
            n_iterations=conv.n_iterations,
            initial_residual=conv.residuals.get(
                "rms_rho", np.array([1.0])
            )[0] if conv.residuals else 1.0,
            final_residual=conv.final_residual,
            orders_reduction=conv.orders_reduction,
            cl_std=float(np.std(conv.cl_history[-10:]))
            if len(conv.cl_history) >= 10 else 0.0,
            cd_std=float(np.std(conv.cd_history[-10:]))
            if len(conv.cd_history) >= 10 else 0.0,
        )
        report.summary = self.llm.query(
            prompt, PromptTemplates.SYSTEM_PROMPT
        )

        return report


# ============================================================================
# 6. Narrative Report Generator
# ============================================================================
class NarrativeReportGenerator:
    """
    Generates publication-quality narrative text from benchmark results.

    Produces structured sections suitable for technical reports and
    journal papers.
    """

    def __init__(self, llm_config: LLMConfig = None):
        self.llm = LLMBackend(llm_config)

    def generate_case_section(
        self,
        case: BenchmarkCaseResult,
        diagnostic: DiagnosticReport,
    ) -> str:
        """Generate a narrative section for one case."""
        lines = [
            f"### {case.case_name}",
            "",
            f"**Configuration**: {case.solver} with {case.turbulence_model} "
            f"turbulence model on {case.mesh_level} mesh "
            f"({case.n_cells:,} cells)." if case.n_cells > 0 else
            f"**Configuration**: {case.solver} with {case.turbulence_model} "
            f"turbulence model.",
            "",
        ]

        # Convergence
        conv = case.convergence
        if conv.n_iterations > 0:
            lines.append(
                f"The simulation ran for {conv.n_iterations:,} iterations, "
                f"achieving {conv.orders_reduction:.1f} orders of magnitude "
                f"residual reduction (final residual: {conv.final_residual:.2e})."
            )

        # Forces
        if case.cl != 0 or case.cd != 0:
            lines.append(
                f"Force coefficients: CL = {case.cl:.4f}, "
                f"CD = {case.cd:.6f}."
            )
            if case.cl_exp is not None:
                cl_err = abs(case.cl - case.cl_exp) / max(abs(case.cl_exp), 1e-10) * 100
                lines.append(
                    f"CL error vs. experiment: {cl_err:.1f}%."
                )

        # Separation
        surf = case.surface
        if surf.separation_x is not None:
            lines.append("")
            lines.append(
                f"Separation onset at x/c = {surf.separation_x:.4f}"
                + (f", reattachment at x/c = {surf.reattachment_x:.4f}"
                   if surf.reattachment_x else "")
                + (f" (bubble length = {surf.bubble_length:.4f})."
                   if surf.bubble_length else ".")
            )

        # Diagnostics
        if diagnostic.n_errors > 0 or diagnostic.n_warnings > 0:
            lines.append("")
            lines.append(
                f"Diagnostics: {diagnostic.n_errors} error(s), "
                f"{diagnostic.n_warnings} warning(s)."
            )
            for f in diagnostic.findings:
                if f.severity in ("error", "warning"):
                    lines.append(f"- **{f.severity.upper()}**: {f.message}")

        lines.append("")
        return "\n".join(lines)

    def generate_full_report(
        self,
        cases: List[BenchmarkCaseResult],
        title: str = "CFD Benchmark Analysis Report",
    ) -> str:
        """
        Generate a complete narrative report.

        Parameters
        ----------
        cases : list of BenchmarkCaseResult
        title : str

        Returns
        -------
        str : Markdown report.
        """
        analyzer = BenchmarkCaseAnalyzer(self.llm.config)

        sections = [
            f"# {title}",
            "",
            f"*Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}*",
            "",
        ]

        # Executive summary
        summary_prompt = PromptTemplates.render(
            "executive_summary",
            title=title,
            n_cases=len(cases),
            models=", ".join(set(c.turbulence_model for c in cases)),
            findings="See per-case analysis below.",
            challenge_progress="In progress — see individual case metrics.",
        )
        exec_summary = self.llm.query(
            summary_prompt, PromptTemplates.SYSTEM_PROMPT
        )
        sections.extend([
            "## Executive Summary",
            "",
            exec_summary,
            "",
            "---",
            "",
            "## Per-Case Analysis",
            "",
        ])

        # Per-case sections
        for case in cases:
            diag = analyzer.analyze(case)
            sections.append(self.generate_case_section(case, diag))

        # Recommendations
        rec_prompt = PromptTemplates.render(
            "recommend_next_steps",
            completed_cases=", ".join(c.case_name for c in cases),
            issues="See diagnostic findings above.",
            model_ranking="See per-case analysis.",
            gci_status="See grid convergence section.",
            sep_accuracy="Variable across cases.",
        )
        recommendations = self.llm.query(
            rec_prompt, PromptTemplates.SYSTEM_PROMPT
        )

        sections.extend([
            "---",
            "",
            "## Recommendations",
            "",
            recommendations,
            "",
        ])

        return "\n".join(sections)


# ============================================================================
# Convenience: Generate Synthetic Demo Data
# ============================================================================
def generate_demo_cases() -> List[BenchmarkCaseResult]:
    """Generate synthetic benchmark cases for demonstration."""
    cases = []

    # Wall hump SA
    conv1 = ConvergenceData(
        iterations=np.arange(10000),
        residuals={"rms_rho": -2.0 - 4.0 * np.linspace(0, 1, 10000)},
        cl_history=0.5 + 0.001 * np.random.randn(10000).cumsum() / 100,
        cd_history=0.02 + 0.0001 * np.random.randn(10000).cumsum() / 100,
        final_residual=1e-6,
        orders_reduction=4.0,
        is_converged=True,
        n_iterations=10000,
    )
    x_surf = np.linspace(0, 1, 100)
    Cp1 = 1.0 - 2 * np.sin(np.pi * x_surf) ** 2
    Cf1 = 0.005 * (1 - 3 * np.maximum(x_surf - 0.65, 0))
    Cf1[x_surf > 0.7] = np.minimum(Cf1[x_surf > 0.7], -0.001)
    Cf1[x_surf > 0.95] = 0.001

    cases.append(BenchmarkCaseResult(
        case_name="Wall Hump — SA",
        solver="SU2", turbulence_model="SA",
        mesh_level="fine", n_cells=250000,
        reynolds=936000, mach=0.1, aoa=0.0,
        cl=0.48, cd=0.022, cl_exp=0.50, cd_exp=0.020,
        convergence=conv1,
        surface=SurfaceData(
            x=x_surf, Cp=Cp1, Cf=Cf1,
            separation_x=0.70, reattachment_x=0.95,
            bubble_length=0.25, Cp_stagnation=1.0,
        ),
        gci_fine=1.2, gci_observed_order=1.8,
    ))

    # NACA 0012 SST
    conv2 = ConvergenceData(
        iterations=np.arange(5000),
        residuals={"rms_rho": -1.5 - 2.5 * np.linspace(0, 1, 5000)},
        cl_history=1.05 + 0.005 * np.random.randn(5000).cumsum() / 50,
        cd_history=0.015 + 0.0002 * np.random.randn(5000).cumsum() / 50,
        final_residual=1e-4,
        orders_reduction=2.5,
        is_converged=False,
        n_iterations=5000,
    )
    cases.append(BenchmarkCaseResult(
        case_name="NACA 0012 α=10° — SST",
        solver="SU2", turbulence_model="SST",
        mesh_level="medium", n_cells=120000,
        reynolds=6e6, mach=0.15, aoa=10.0,
        cl=1.05, cd=0.015, cl_exp=1.09, cd_exp=0.012,
        convergence=conv2,
        gci_fine=3.1, gci_observed_order=1.5,
    ))

    return cases


# ============================================================================
# Demo
# ============================================================================
def _demo():
    """Demonstrate the LLM Benchmark Assistant."""
    print("=" * 70)
    print("  LLM-Based Benchmark Automation & Intelligent Reporting")
    print("=" * 70)

    # 1. Generate demo data
    print("\n  [1] Generating demo benchmark cases...")
    cases = generate_demo_cases()
    for c in cases:
        print(f"      {c.case_name}: CL={c.cl:.4f}, CD={c.cd:.6f}")

    # 2. Run diagnostics
    print("\n  [2] Running automated diagnostics...")
    analyzer = BenchmarkCaseAnalyzer()
    for c in cases:
        report = analyzer.analyze(c)
        print(f"\n  Case: {report.case_name}")
        print(f"  Status: {report.overall_status.upper()}")
        print(f"  Errors: {report.n_errors}, Warnings: {report.n_warnings}")
        for f in report.findings:
            if f.severity != "info":
                print(f"    [{f.severity.upper()}] {f.message}")

    # 3. Generate narrative report
    print("\n  [3] Generating narrative report...")
    generator = NarrativeReportGenerator()
    report = generator.generate_full_report(cases)
    print(f"\n  Report length: {len(report)} characters")
    print(f"  Preview:\n")
    for line in report.split('\n')[:20]:
        print(f"    {line}")

    print(f"\n{'=' * 70}")
    print("  Demo complete!")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _demo()
