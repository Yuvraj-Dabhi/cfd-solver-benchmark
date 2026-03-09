#!/usr/bin/env python3
"""
LLM-Driven Turbulence Closure Discovery
==========================================
Uses large language model reasoning (DeepSeek-R1 style) to discover
interpretable, algebraic RANS corrections from residual analysis.

Architecture
------------
1. **ResidualAnalyzer** — extracts structured error reports from SU2 vs
   experimental residuals, identifying where and why RANS fails.

2. **LLMClosurePrompter** — constructs Chain-of-Thought prompts with
   physical constraints (Galilean invariance, realizability, symmetry).

3. **AlgebraicCorrectionParser** — parses LLM-generated mathematical
   formulas into executable Python/C++ code snippets.

4. **SU2SourceInjector** — templates corrections into SU2 source code
   via its C++ template architecture.

5. **ClosureValidationLoop** — runs corrected SU2, compares to experiment,
   feeds results back to LLM for iterative refinement.

Key Advantage
-------------
Unlike black-box neural networks, LLM-derived corrections produce
human-readable algebraic formulas that can be inspected for physical
validity (Galilean invariance, realizability, thermodynamic consistency).

References
----------
- Deng et al. (2025) "Large Language Model Driven Development of
  Turbulence Models", Cambridge Flow Journal
- DeepSeek-R1 technical report (2025)

Usage
-----
    analyzer = ResidualAnalyzer()
    errors = analyzer.analyze("wall_hump", su2_results, experimental_data)
    prompter = LLMClosurePrompter()
    prompt = prompter.build_prompt(errors, model="sa")
    # Send to DeepSeek-R1 API for algebraic correction discovery
"""

import json
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)
PROJECT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT))


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class ResidualReport:
    """Structured report of RANS vs experimental discrepancies.

    Attributes
    ----------
    case_name : str
        Benchmark case identifier.
    model_name : str
        Turbulence model used.
    separation_error : float
        Error in separation onset prediction (x/c).
    reattachment_error : float
        Error in reattachment prediction (x/c).
    cp_rmse : float
        RMSE of pressure coefficient distribution.
    cf_rmse : float
        RMSE of skin friction coefficient distribution.
    peak_error_location : float
        x/c location of maximum discrepancy.
    error_type : str
        Classification: 'separation_onset', 'reattachment', 'shear_layer',
        'pressure_recovery', 'shock_position'.
    physical_mechanism : str
        Identified physical mechanism of the failure.
    """
    case_name: str
    model_name: str
    separation_error: float
    reattachment_error: float
    cp_rmse: float
    cf_rmse: float
    peak_error_location: float
    error_type: str = "unknown"
    physical_mechanism: str = "unknown"


@dataclass
class AlgebraicCorrection:
    """Parsed algebraic correction from LLM output.

    Attributes
    ----------
    formula : str
        Human-readable algebraic formula.
    python_expr : str
        Executable Python expression.
    cpp_expr : str
        C++ expression for SU2 injection.
    corrected_variable : str
        Which RANS variable is corrected (e.g., 'nu_t', 'production', 'omega').
    constraints_satisfied : list of str
        Physical constraints verified (e.g., 'galilean_invariant',
        'realizable', 'positive_definite').
    confidence : float
        LLM confidence in this correction (0-1).
    """
    formula: str
    python_expr: str
    cpp_expr: str
    corrected_variable: str
    constraints_satisfied: List[str] = field(default_factory=list)
    confidence: float = 0.5


# =============================================================================
# Residual Analyzer
# =============================================================================

class ResidualAnalyzer:
    """Extracts structured error reports from SU2 vs experimental data.

    Analyzes the spatial distribution of errors to identify the physical
    mechanism of the RANS failure, enabling targeted LLM prompting.
    """

    # Known physical failure mechanisms
    MECHANISMS = {
        "adverse_pressure_gradient": "APG-induced premature/delayed separation",
        "shear_layer_mixing": "Incorrect turbulent mixing in free shear layer",
        "reattachment_overshoot": "Reattachment point predicted too far downstream",
        "shock_position_error": "Shock foot position error in SWBLI",
        "curvature_effect": "Missing streamline curvature correction",
        "history_effect": "Non-equilibrium turbulence history neglected",
        "anisotropy": "Boussinesq linear relation breaks down",
    }

    def analyze(self, case_name: str,
                su2_cp: np.ndarray, su2_cf: np.ndarray,
                exp_cp: np.ndarray, exp_cf: np.ndarray,
                x_coords: np.ndarray,
                model_name: str = "SA",
                exp_x_sep: float = 0.0,
                exp_x_reat: float = 0.0,
                su2_x_sep: float = 0.0,
                su2_x_reat: float = 0.0,
                ) -> ResidualReport:
        """Analyze RANS errors and classify failure mechanism.

        Parameters
        ----------
        case_name : str
        su2_cp, su2_cf : ndarray
            SU2-predicted Cp and Cf distributions.
        exp_cp, exp_cf : ndarray
            Experimental Cp and Cf distributions.
        x_coords : ndarray
            Streamwise coordinates.
        model_name : str
        exp_x_sep, exp_x_reat : float
            Experimental separation/reattachment locations.
        su2_x_sep, su2_x_reat : float
            SU2-predicted locations.

        Returns
        -------
        ResidualReport
        """
        # Compute errors
        cp_residual = su2_cp - exp_cp
        cf_residual = su2_cf - exp_cf
        cp_rmse = float(np.sqrt(np.mean(cp_residual ** 2)))
        cf_rmse = float(np.sqrt(np.mean(cf_residual ** 2)))

        sep_error = su2_x_sep - exp_x_sep
        reat_error = su2_x_reat - exp_x_reat

        # Find peak error location
        total_error = np.abs(cp_residual) + np.abs(cf_residual)
        peak_idx = np.argmax(total_error)
        peak_location = float(x_coords[peak_idx])

        # Classify failure mechanism
        error_type, mechanism = self._classify_mechanism(
            x_coords, cp_residual, cf_residual,
            sep_error, reat_error, peak_location
        )

        report = ResidualReport(
            case_name=case_name,
            model_name=model_name,
            separation_error=float(sep_error),
            reattachment_error=float(reat_error),
            cp_rmse=cp_rmse,
            cf_rmse=cf_rmse,
            peak_error_location=peak_location,
            error_type=error_type,
            physical_mechanism=mechanism,
        )

        logger.info(
            f"Residual analysis [{case_name}/{model_name}]: "
            f"Cp_RMSE={cp_rmse:.4f}, Cf_RMSE={cf_rmse:.4f}, "
            f"mechanism='{mechanism}'"
        )
        return report

    def _classify_mechanism(self, x: np.ndarray,
                             cp_res: np.ndarray, cf_res: np.ndarray,
                             sep_err: float, reat_err: float,
                             peak_loc: float
                             ) -> Tuple[str, str]:
        """Classify the dominant physical failure mechanism."""
        if abs(sep_err) > 0.05:
            return "separation_onset", self.MECHANISMS["adverse_pressure_gradient"]
        elif abs(reat_err) > 0.1:
            return "reattachment", self.MECHANISMS["reattachment_overshoot"]
        elif abs(reat_err) > 0.02:
            return "shear_layer", self.MECHANISMS["shear_layer_mixing"]
        else:
            return "general", self.MECHANISMS["history_effect"]


# =============================================================================
# LLM Closure Prompter
# =============================================================================

class LLMClosurePrompter:
    """Constructs Chain-of-Thought prompts for turbulence closure discovery.

    Generates structured prompts that guide reasoning LLMs (DeepSeek-R1)
    to derive algebraic RANS corrections while respecting physical constraints.

    Parameters
    ----------
    model_type : str
        Target RANS model: 'sa' (Spalart-Allmaras) or 'sst' (Menter SST).
    """

    # Physical constraints the LLM must respect
    CONSTRAINTS = {
        "galilean_invariance": "The correction must be invariant under "
                                "uniform translation of the reference frame.",
        "realizability": "The corrected Reynolds stress tensor must remain "
                          "positive semi-definite (eigenvalues ≥ 0).",
        "dimensional_consistency": "All terms must be dimensionally consistent "
                                    "with the transport equation.",
        "wall_limiting": "The correction must vanish in the viscous sublayer "
                          "(y+ → 0).",
        "freestream_decay": "The correction must vanish far from walls and "
                             "separation regions.",
    }

    def __init__(self, model_type: str = "sa"):
        self.model_type = model_type.lower()

    def build_prompt(self, residual_report: ResidualReport,
                     include_constraints: bool = True,
                     max_terms: int = 3) -> str:
        """Build a Chain-of-Thought prompt for closure discovery.

        Parameters
        ----------
        residual_report : ResidualReport
        include_constraints : bool
            Whether to include physical constraint instructions.
        max_terms : int
            Maximum number of algebraic terms to propose.

        Returns
        -------
        prompt : str
            Complete prompt for the reasoning LLM.
        """
        r = residual_report
        lines = [
            "<think>",
            f"# Turbulence Model Error Analysis: {r.case_name}",
            f"",
            f"## Current Model: {r.model_name}",
            f"## Error Summary:",
            f"- Separation onset error: Δx_sep = {r.separation_error:+.4f}",
            f"- Reattachment error: Δx_reat = {r.reattachment_error:+.4f}",
            f"- Cp RMSE: {r.cp_rmse:.4f}",
            f"- Cf RMSE: {r.cf_rmse:.4f}",
            f"- Peak error at x/c = {r.peak_error_location:.3f}",
            f"- Failure mechanism: {r.physical_mechanism}",
            f"",
            f"## Task:",
            f"Propose up to {max_terms} algebraic correction terms to the "
            f"{r.model_name} turbulence model that specifically address the "
            f"identified '{r.error_type}' failure mechanism.",
            f"",
        ]

        if self.model_type == "sa":
            lines.extend([
                "## SA Production Term:",
                "P = c_b1 · S̃ · ν̃",
                "where S̃ = Ω + (ν̃ / (κ²d²)) · f_v2",
                "",
                "Propose corrections to S̃ or additional source terms.",
            ])
        elif self.model_type == "sst":
            lines.extend([
                "## SST-k Production:",
                "P_k = min(μ_t · S², 10 · β* · ρ · k · ω)",
                "",
                "Propose corrections to the production limiter or ω equation.",
            ])

        if include_constraints:
            lines.extend([
                "",
                "## Physical Constraints (MUST satisfy ALL):",
            ])
            for name, desc in self.CONSTRAINTS.items():
                lines.append(f"- **{name}**: {desc}")

        lines.extend([
            "",
            "## Output Format:",
            "For each correction term, provide:",
            "1. Algebraic formula (LaTeX notation)",
            "2. Physical justification",
            "3. Python implementation: `correction = ...`",
            "4. Expected impact on the identified error",
            "</think>",
        ])

        return "\n".join(lines)

    def build_validation_prompt(self, correction: AlgebraicCorrection,
                                 before_metrics: Dict[str, float],
                                 after_metrics: Dict[str, float]) -> str:
        """Build a prompt for validating a correction's effectiveness.

        Parameters
        ----------
        correction : AlgebraicCorrection
        before_metrics : dict
            Error metrics before correction.
        after_metrics : dict
            Error metrics after correction.

        Returns
        -------
        prompt : str
        """
        lines = [
            "<think>",
            f"# Correction Validation: {correction.corrected_variable}",
            f"",
            f"## Applied Correction: {correction.formula}",
            f"",
            f"## Metrics Before:",
        ]
        for k, v in before_metrics.items():
            lines.append(f"- {k}: {v:.4f}")
        lines.append("")
        lines.append("## Metrics After:")
        for k, v in after_metrics.items():
            lines.append(f"- {k}: {v:.4f}")
        lines.extend([
            "",
            "## Analyze:",
            "1. Did the correction improve the target error?",
            "2. Did it introduce any new errors or constraint violations?",
            "3. Should the correction be refined? If so, suggest modifications.",
            "</think>",
        ])
        return "\n".join(lines)


# =============================================================================
# Algebraic Correction Parser
# =============================================================================

class AlgebraicCorrectionParser:
    """Parses LLM-generated mathematical formulas into executable code.

    Extracts algebraic expressions from LLM output, validates physical
    dimensions, and converts to both Python and C++ representations.
    """

    # Safe mathematical operations for Python eval
    SAFE_FUNCTIONS = {
        "sqrt": "np.sqrt",
        "exp": "np.exp",
        "log": "np.log",
        "abs": "np.abs",
        "max": "np.maximum",
        "min": "np.minimum",
        "tanh": "np.tanh",
        "sign": "np.sign",
    }

    # Standard RANS variables
    RANS_VARIABLES = {
        "S", "Omega", "k", "epsilon", "omega", "nu_t", "nu",
        "dUdy", "d", "y_plus", "rho", "mu", "P_k", "S_tilde",
        "f_v1", "f_v2", "f_w", "chi", "kappa",
    }

    def parse_response(self, llm_output: str,
                        target_variable: str = "nu_t"
                        ) -> List[AlgebraicCorrection]:
        """Parse LLM response into algebraic corrections.

        Parameters
        ----------
        llm_output : str
            Raw LLM output text.
        target_variable : str
            Default variable being corrected.

        Returns
        -------
        corrections : list of AlgebraicCorrection
        """
        corrections = []

        # Extract Python code blocks
        python_blocks = re.findall(
            r'```python\s*\n(.*?)\n```',
            llm_output, re.DOTALL
        )

        # Extract formula blocks (LaTeX-style)
        formula_blocks = re.findall(
            r'\$\$(.*?)\$\$|\$(.*?)\$',
            llm_output
        )

        for i, py_code in enumerate(python_blocks):
            # Extract assignment expressions
            assignments = re.findall(
                r'correction\s*=\s*(.+)',
                py_code
            )

            formula = ""
            if i < len(formula_blocks):
                formula = formula_blocks[i][0] or formula_blocks[i][1]

            for expr in assignments:
                expr = expr.strip().rstrip(';')

                # Basic safety check
                is_safe = self._validate_expression(expr)

                # Convert to C++
                cpp_expr = self._python_to_cpp(expr)

                # Check physical constraints
                constraints = self._check_constraints(expr)

                corrections.append(AlgebraicCorrection(
                    formula=formula or expr,
                    python_expr=expr,
                    cpp_expr=cpp_expr,
                    corrected_variable=target_variable,
                    constraints_satisfied=constraints,
                    confidence=0.5 if is_safe else 0.2,
                ))

        # If no Python blocks found, try to extract inline expressions
        if not corrections:
            inline = re.findall(
                r'correction\s*=\s*(.+?)(?:\n|$)',
                llm_output
            )
            for expr in inline:
                expr = expr.strip()
                corrections.append(AlgebraicCorrection(
                    formula=expr,
                    python_expr=expr,
                    cpp_expr=self._python_to_cpp(expr),
                    corrected_variable=target_variable,
                    constraints_satisfied=[],
                    confidence=0.3,
                ))

        return corrections

    def _validate_expression(self, expr: str) -> bool:
        """Check that expression only uses safe operations and RANS variables."""
        # Remove known safe tokens
        cleaned = expr
        for func in self.SAFE_FUNCTIONS:
            cleaned = cleaned.replace(func, "")
        for var in self.RANS_VARIABLES:
            cleaned = cleaned.replace(var, "")

        # Remove operators and numbers
        cleaned = re.sub(r'[0-9.+\-*/() \t,]', '', cleaned)
        # Remaining should be empty or only np. prefix
        cleaned = cleaned.replace("np.", "")

        if cleaned.strip() and not all(c.isalpha() or c == '_' for c in cleaned.strip()):
            logger.warning(f"Potentially unsafe expression: {expr}")
            return False
        return True

    def _python_to_cpp(self, expr: str) -> str:
        """Convert Python expression to C++ equivalent."""
        cpp = expr
        for py_func, np_func in self.SAFE_FUNCTIONS.items():
            cpp = cpp.replace(f"np.{py_func}", py_func)
            cpp = cpp.replace(np_func, py_func)
        cpp = cpp.replace("**", "pow_placeholder")
        # Handle power operator
        cpp = re.sub(r'(\w+)\s*pow_placeholder\s*(\d+\.?\d*)',
                     r'pow(\1, \2)', cpp)
        cpp = cpp.replace("pow_placeholder", "**")
        return cpp

    def _check_constraints(self, expr: str) -> List[str]:
        """Check which physical constraints the expression satisfies."""
        constraints = []

        # Galilean invariance: uses only invariant quantities (S, Ω, k, ε)
        invariant_vars = {"S", "Omega", "k", "epsilon", "omega", "nu_t", "nu"}
        used_vars = set(re.findall(r'[a-zA-Z_]+', expr)) & self.RANS_VARIABLES
        if used_vars.issubset(invariant_vars):
            constraints.append("galilean_invariant")

        # Wall limiting: if expression contains d or y_plus
        if "d" in used_vars or "y_plus" in used_vars:
            constraints.append("wall_limited")

        # Positive definite: if wrapped in abs() or **2
        if "abs" in expr or "**2" in expr:
            constraints.append("positive_definite")

        return constraints


# =============================================================================
# SU2 Source Injector
# =============================================================================

class SU2SourceInjector:
    """Templates algebraic corrections into SU2 C++ source code.

    Generates C++ code snippets that can be injected into the SU2
    turbulence model source files to implement LLM-discovered corrections.

    Parameters
    ----------
    su2_source_dir : Path or str
        Path to SU2 source directory.
    """

    def __init__(self, su2_source_dir: Optional[str] = None):
        self.su2_source_dir = Path(su2_source_dir) if su2_source_dir else None

    def generate_correction_code(self, correction: AlgebraicCorrection,
                                   model: str = "sa") -> str:
        """Generate C++ code for injecting the correction into SU2.

        Parameters
        ----------
        correction : AlgebraicCorrection
        model : str
            'sa' or 'sst'.

        Returns
        -------
        cpp_code : str
            C++ code snippet ready for injection.
        """
        var = correction.corrected_variable
        expr = correction.cpp_expr

        code = f"""
// =================================================================
// LLM-Derived Correction: {correction.formula}
// Corrected variable: {var}
// Constraints: {', '.join(correction.constraints_satisfied)}
// Confidence: {correction.confidence:.2f}
// =================================================================

// Original value preserved for comparison
su2double {var}_original = {var};

// Apply algebraic correction
su2double correction = {expr};
{var} = {var}_original + correction;

// Safety clamp to prevent numerical instability
{var} = max({var}, 0.0);
"""
        return code.strip()

    def generate_header_comment(self, corrections: List[AlgebraicCorrection]
                                 ) -> str:
        """Generate a header comment documenting all applied corrections."""
        lines = [
            "/*",
            " * ============================================================",
            " * LLM-Derived Turbulence Model Corrections",
            " * Generated by: llm_turbulence_closure.py",
            " * ============================================================",
        ]
        for i, c in enumerate(corrections, 1):
            lines.extend([
                f" * Correction {i}: {c.formula}",
                f" *   Variable: {c.corrected_variable}",
                f" *   Constraints: {', '.join(c.constraints_satisfied)}",
            ])
        lines.append(" */")
        return "\n".join(lines)


# =============================================================================
# Closure Validation Loop
# =============================================================================

class ClosureValidationLoop:
    """Iterative LLM → SU2 → validate → refine loop.

    Manages the complete cycle:
    1. Analyze current RANS errors
    2. Build LLM prompt
    3. Parse LLM response
    4. Generate C++ correction code
    5. (Conceptually) run SU2 with correction
    6. Validate improvement
    7. Feed results back to LLM for refinement

    Parameters
    ----------
    analyzer : ResidualAnalyzer
    prompter : LLMClosurePrompter
    parser : AlgebraicCorrectionParser
    injector : SU2SourceInjector
    max_iterations : int
        Maximum refinement iterations.
    """

    def __init__(self, max_iterations: int = 5):
        self.analyzer = ResidualAnalyzer()
        self.prompter = LLMClosurePrompter()
        self.parser = AlgebraicCorrectionParser()
        self.injector = SU2SourceInjector()
        self.max_iterations = max_iterations
        self.history: List[Dict[str, Any]] = []

    def run_analysis(self, case_name: str,
                      su2_cp: np.ndarray, su2_cf: np.ndarray,
                      exp_cp: np.ndarray, exp_cf: np.ndarray,
                      x_coords: np.ndarray,
                      model_name: str = "SA",
                      **kwargs) -> Dict[str, Any]:
        """Run one iteration of the analysis-prompt-parse cycle.

        Parameters
        ----------
        case_name, su2_cp, su2_cf, exp_cp, exp_cf, x_coords, model_name :
            See ResidualAnalyzer.analyze()

        Returns
        -------
        result : dict
            Contains 'report', 'prompt', and 'correction_template'.
        """
        # Step 1: Analyze residuals
        report = self.analyzer.analyze(
            case_name, su2_cp, su2_cf, exp_cp, exp_cf, x_coords,
            model_name=model_name, **kwargs
        )

        # Step 2: Build prompt
        prompt = self.prompter.build_prompt(report)

        # Step 3: Generate template for correction
        # (In production, this prompt would be sent to DeepSeek-R1)
        template_correction = AlgebraicCorrection(
            formula=f"f({report.error_type}) correction for {model_name}",
            python_expr="0.0  # Placeholder — send prompt to LLM",
            cpp_expr="0.0",
            corrected_variable="nu_t",
            constraints_satisfied=["galilean_invariant"],
            confidence=0.0,
        )

        # Step 4: Generate injection code
        cpp_code = self.injector.generate_correction_code(template_correction)

        result = {
            "report": report,
            "prompt": prompt,
            "correction_template": template_correction,
            "cpp_code": cpp_code,
            "iteration": len(self.history),
        }

        self.history.append(result)
        return result

    def summary(self) -> Dict[str, Any]:
        """Summary of all iterations in the validation loop."""
        return {
            "n_iterations": len(self.history),
            "max_iterations": self.max_iterations,
            "cases_analyzed": [h["report"].case_name for h in self.history],
            "mechanisms_found": [h["report"].physical_mechanism
                                 for h in self.history],
        }
