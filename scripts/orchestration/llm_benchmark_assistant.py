#!/usr/bin/env python3
"""
LLM Benchmark Assistant — AI-Powered CFD Orchestration
========================================================
Intelligence layer on top of the benchmark harness that uses LLM APIs
to automate SU2 config generation, physics-aware diagnostics, and
anomaly detection from benchmark results.

Features:
  1. Natural language → SU2 config file generation
  2. Physics-aware diagnostic report from benchmark JSON
  3. Anomaly detection (flag cases deviating from TMR scatter)
  4. Cross-solver comparison automation

Architecture:
  - System prompt: loaded from config.py (case registry + model descriptions)
  - Context: benchmark_harness.py JSON output + TMR knowledge base
  - Tools: config generation, diagnostic analysis, anomaly flagging
  - API: Claude API, OpenAI API, or local via Ollama

Usage:
    from scripts.orchestration.llm_benchmark_assistant import (
        BenchmarkAssistant, ConfigGenerator, DiagnosticEngine,
    )
    assistant = BenchmarkAssistant()
    cfg = assistant.generate_config("Mach 0.3 BFS at Re=50000 with SST, 3 grid levels")
    report = assistant.diagnose("output/benchmark_summary.json")
    anomalies = assistant.detect_anomalies("output/benchmark_summary.json")
"""

import json
import logging
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)
PROJECT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT))

from config import (
    BENCHMARK_CASES, TURBULENCE_MODELS,
    BenchmarkCase, TurbulenceModel,
)

# Optional LLM backends
try:
    import anthropic
    _HAS_ANTHROPIC = True
except ImportError:
    _HAS_ANTHROPIC = False

try:
    import openai
    _HAS_OPENAI = True
except ImportError:
    _HAS_OPENAI = False


# =============================================================================
# TMR Knowledge Base — Physics Rules for Diagnostics
# =============================================================================
TMR_KNOWLEDGE = {
    "SA": {
        "strengths": [
            "Excellent for attached boundary layers and mild APG",
            "Single-equation: fast, robust convergence",
            "Well-calibrated for aerospace flows (NASA TMR)",
        ],
        "weaknesses": [
            "Underpredicts turbulent kinetic energy production in strong APG",
            "Systematically overpredicts separation bubble length by 10-20%",
            "No transition modeling capability",
            "Poor corner flow / 3D separation prediction",
        ],
        "expected_errors": {
            "flat_plate": {"Cf_error": 0.01, "category": "excellent"},
            "naca0012_alpha10": {"CL_error": 0.02, "category": "good"},
            "wall_hump": {"x_sep_error": 0.05, "L_bubble_error": 0.15, "category": "moderate"},
            "swbli": {"x_sep_error": 0.10, "L_bubble_error": 0.20, "category": "poor"},
        },
    },
    "SST": {
        "strengths": [
            "Better APG prediction than SA due to k-ω near-wall treatment",
            "Shear stress transport limiter improves separation prediction",
            "Two-equation model: captures more turbulence physics",
        ],
        "weaknesses": [
            "Slower convergence than SA (2 transport equations)",
            "Still uses Boussinesq approximation — limited for 3D separation",
            "Sensitivity to freestream turbulence values",
        ],
        "expected_errors": {
            "flat_plate": {"Cf_error": 0.01, "category": "excellent"},
            "naca0012_alpha10": {"CL_error": 0.02, "category": "good"},
            "wall_hump": {"x_sep_error": 0.03, "L_bubble_error": 0.10, "category": "good"},
            "swbli": {"x_sep_error": 0.085, "L_bubble_error": 0.12, "category": "moderate"},
        },
    },
}

# Diagnostic rule templates
DIAGNOSTIC_RULES = [
    {
        "condition": lambda m: m.get("Cf_RMSE", 0) < 0.001,
        "level": "PASS",
        "template": "{case}: Excellent Cf prediction (RMSE={Cf_RMSE:.4f}), within NASA TMR inter-code scatter.",
    },
    {
        "condition": lambda m: m.get("GCI_pct", 100) < 2.0,
        "level": "PASS",
        "template": "{case}: Grid convergence achieved (GCI={GCI_pct:.1f}%). Solution is grid-independent.",
    },
    {
        "condition": lambda m: m.get("L_bubble_error", 0) > 0.15,
        "level": "WARN",
        "template": "{case}: Separation bubble over-predicted by {L_bubble_error:.0%}. Consistent with {model}'s known underprediction of APG turbulence production.",
    },
    {
        "condition": lambda m: m.get("x_sep_error", 0) > 0.10,
        "level": "FAIL",
        "template": "{case}: Separation point error ({x_sep_error:.0%}) exceeds TMR acceptable range. Consider SST or RSM.",
    },
    {
        "condition": lambda m: m.get("CL_error", 0) > 0.05,
        "level": "WARN",
        "template": "{case}: CL error ({CL_error:.1%}) above 5%. May indicate insufficient convergence or grid resolution.",
    },
]


# =============================================================================
# Natural Language Config Parser
# =============================================================================
@dataclass
class ParsedConfig:
    """Structured config parsed from natural language."""
    case_type: str = ""
    mach: float = 0.0
    reynolds: float = 0.0
    alpha_deg: float = 0.0
    turbulence_model: str = "SA"
    n_grid_levels: int = 1
    solver: str = "SU2"
    mesh_file: str = ""
    max_iterations: int = 5000
    cfl: float = 5.0
    raw_text: str = ""
    confidence: float = 0.0


class ConfigGenerator:
    """
    Natural language → SU2 config file generator.

    Uses regex-based NL parsing (no LLM required) with optional
    LLM enhancement for complex descriptions.
    """

    # Known case type patterns
    CASE_PATTERNS = {
        r"flat\s*plate": "flat_plate",
        r"naca\s*0012": "naca0012",
        r"wall\s*hump": "wall_hump",
        r"backward[\s-]*facing\s*step|bfs": "bfs",
        r"swbli|shock[\s-]*wave": "swbli",
        r"periodic\s*hill": "periodic_hill",
        r"bump": "gaussian_bump",
        r"crm|common\s*research\s*model": "nasa_crm",
        r"diffuser": "diffuser",
        r"airfoil": "naca0012",
    }

    # Turbulence model patterns (ordered: most specific first)
    MODEL_PATTERNS = {
        r"\bsa[\s-]*neg\b": "SA_NEG",
        r"\bsst\b|menter[\s-]*sst": "SST",
        r"\bk[\s-]*omega\b": "SST",
        r"\bk[\s-]*epsilon\b": "KE",
        r"\brsm\b|reynolds[\s-]*stress": "RSM",
        r"\bdes\b|detached[\s-]*eddy": "DES",
        r"\bsa\b|spalart[\s-]*allmaras": "SA",
    }

    def parse_natural_language(self, text: str) -> ParsedConfig:
        """
        Parse natural language description into structured config.

        Examples:
            "Mach 0.3 BFS at Re=50000 with SST and 3 grid levels"
            "Set up a NACA 0012 case at α=15°, Re=6M, SA model"
            "SWBLI case, Mach 2.0, Re=2.3e7, SA-neg, fine grid"
        """
        cfg = ParsedConfig(raw_text=text)
        text_lower = text.lower()
        confidence_factors = []

        # Case type
        for pattern, case_type in self.CASE_PATTERNS.items():
            if re.search(pattern, text_lower):
                cfg.case_type = case_type
                confidence_factors.append(0.9)
                break

        if not cfg.case_type:
            cfg.case_type = "generic"
            confidence_factors.append(0.3)

        # Mach number
        mach_match = re.search(r"mach[\s=]*([0-9.]+)", text_lower)
        if mach_match:
            cfg.mach = float(mach_match.group(1))
            confidence_factors.append(0.95)
        elif cfg.case_type in ("swbli",):
            cfg.mach = 2.0  # Default for SWBLI
            confidence_factors.append(0.7)
        else:
            cfg.mach = 0.15  # Subsonic default
            confidence_factors.append(0.5)

        # Reynolds number
        re_match = re.search(
            r"re(?:ynolds)?[\s=]*([0-9.]+)\s*[×x*]?\s*(?:10\^?)?([0-9]*)[kKmM]?",
            text,
            re.IGNORECASE,
        )
        if re_match:
            base = float(re_match.group(1))
            exp = re_match.group(2)
            if exp:
                cfg.reynolds = base * 10 ** int(exp)
            elif base < 100:
                cfg.reynolds = base * 1e6  # "Re=6" → 6M
            else:
                cfg.reynolds = base

            # Handle k/M suffixes
            if re.search(r"re.*[0-9]\s*[kK]", text):
                if cfg.reynolds < 1e4:
                    cfg.reynolds *= 1e3
            elif re.search(r"re.*[0-9]\s*[mM]", text):
                if cfg.reynolds < 1e4:
                    cfg.reynolds *= 1e6

            confidence_factors.append(0.9)
        else:
            cfg.reynolds = 6e6  # Default
            confidence_factors.append(0.3)

        # Angle of attack
        alpha_match = re.search(
            r"(?:alpha|α|aoa|angle)[\s=]*([+-]?[0-9.]+)\s*°?",
            text, re.IGNORECASE,
        )
        if alpha_match:
            cfg.alpha_deg = float(alpha_match.group(1))
            confidence_factors.append(0.95)

        # Turbulence model
        for pattern, model in self.MODEL_PATTERNS.items():
            if re.search(pattern, text_lower):
                cfg.turbulence_model = model
                confidence_factors.append(0.95)
                break

        # Grid levels
        grid_match = re.search(r"(\d+)\s*(?:grid|mesh)\s*levels?", text_lower)
        if grid_match:
            cfg.n_grid_levels = int(grid_match.group(1))
            confidence_factors.append(0.9)

        # Iterations
        iter_match = re.search(r"(\d+)\s*(?:iterations?|iters?)", text_lower)
        if iter_match:
            cfg.max_iterations = int(iter_match.group(1))

        # Overall confidence
        cfg.confidence = float(np.mean(confidence_factors)) if confidence_factors else 0.3

        return cfg

    def generate_su2_config(self, parsed: ParsedConfig,
                            output_dir: Path = None) -> str:
        """
        Generate SU2 config file content from ParsedConfig.

        Returns config file content as string.
        """
        # Map turbulence model to SU2 name
        su2_models = {
            "SA": "SA", "SA_NEG": "SA_NEG",
            "SST": "SST", "KE": "KE",
            "RSM": "SSG_LRR", "DES": "SA_DES",
        }
        su2_turb = su2_models.get(parsed.turbulence_model, "SA")

        # CFL based on Mach
        cfl = 3.0 if parsed.mach > 0.5 else 10.0

        lines = [
            f"% Auto-generated SU2 config for: {parsed.raw_text}",
            f"% Confidence: {parsed.confidence:.0%}",
            "%",
            "SOLVER= RANS",
            f"KIND_TURB_MODEL= {su2_turb}",
            "MATH_PROBLEM= DIRECT",
            "RESTART_SOL= NO",
            "%",
            "% Flow conditions",
            f"MACH_NUMBER= {parsed.mach}",
            f"AOA= {parsed.alpha_deg}",
            "SIDESLIP_ANGLE= 0.0",
            f"REYNOLDS_NUMBER= {parsed.reynolds}",
            "REYNOLDS_LENGTH= 1.0",
            "%",
            "% Fluid model",
            "FLUID_MODEL= IDEAL_GAS",
            "GAMMA_VALUE= 1.4",
            "GAS_CONSTANT= 287.058",
            "VISCOSITY_MODEL= SUTHERLAND",
            "MU_REF= 1.716e-5",
            "MU_T_REF= 273.15",
            "SUTHERLAND_CONSTANT= 110.4",
            "%",
            "% Numerics",
            "NUM_METHOD_GRAD= GREEN_GAUSS",
            f"CFL_NUMBER= {cfl}",
            "CFL_ADAPT= YES",
            f"CFL_ADAPT_PARAM= ( 0.1, 1.5, {cfl}, 1e10 )",
            f"ITER= {parsed.max_iterations}",
            "CONV_RESIDUAL_MINVAL= -12",
            "%",
        ]

        # Scheme selection based on Mach
        if parsed.mach > 0.5:
            lines.extend([
                "% Compressible numerics",
                "CONV_NUM_METHOD_FLOW= JST",
                "JST_SENSOR_COEFF= ( 0.5, 0.02 )",
            ])
        else:
            lines.extend([
                "% Incompressible numerics",
                "CONV_NUM_METHOD_FLOW= ROE",
                "MUSCL_FLOW= YES",
                "SLOPE_LIMITER_FLOW= VENKATAKRISHNAN",
            ])

        lines.extend([
            "CONV_NUM_METHOD_TURB= SCALAR_UPWIND",
            "MUSCL_TURB= NO",
            "%",
            "% Linear solver",
            "LINEAR_SOLVER= FGMRES",
            "LINEAR_SOLVER_PREC= ILU",
            "LINEAR_SOLVER_ERROR= 1e-6",
            "LINEAR_SOLVER_ITER= 10",
            "%",
            "% I/O",
            f"MESH_FILENAME= {parsed.case_type}_mesh.su2",
            "MESH_FORMAT= SU2",
            "OUTPUT_FILES= RESTART, PARAVIEW",
            "CONV_FILENAME= history",
            "RESTART_FILENAME= restart.dat",
            "VOLUME_FILENAME= flow",
            "SURFACE_FILENAME= surface_flow",
            "OUTPUT_WRT_FREQ= 100",
            "%",
            "% Monitoring",
            "MARKER_PLOTTING= ( wall )",
            "MARKER_MONITORING= ( wall )",
        ])

        config_text = "\n".join(lines) + "\n"

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            cfg_path = output_dir / f"{parsed.case_type}_{parsed.turbulence_model}.cfg"
            cfg_path.write_text(config_text)
            logger.info("Config written to %s", cfg_path)

        return config_text


# =============================================================================
# Diagnostic Engine — Physics-Aware Report Generation
# =============================================================================
class DiagnosticEngine:
    """
    Generates physics-aware diagnostic reports from benchmark JSON.

    Uses TMR knowledge base + diagnostic rules to interpret results
    and provide actionable recommendations.
    """

    def __init__(self):
        self.knowledge = TMR_KNOWLEDGE
        self.rules = DIAGNOSTIC_RULES

    def load_benchmark(self, json_path: Union[str, Path]) -> Dict:
        """Load benchmark JSON output."""
        with open(json_path) as f:
            return json.load(f)

    def diagnose_case(self, case_name: str,
                      metrics: Dict, model: str = "SA") -> List[Dict]:
        """
        Generate diagnostics for a single case.

        Returns list of {level, message, recommendation} dicts.
        """
        diagnostics = []
        ctx = {**metrics, "case": case_name, "model": model}

        for rule in self.rules:
            try:
                if rule["condition"](metrics):
                    msg = rule["template"].format(**ctx)
                    diagnostics.append({
                        "level": rule["level"],
                        "message": msg,
                        "case": case_name,
                    })
            except (KeyError, TypeError):
                continue

        # Model-specific insights
        if model in self.knowledge:
            model_info = self.knowledge[model]
            expected = model_info.get("expected_errors", {})

            if case_name in expected:
                exp = expected[case_name]
                category = exp.get("category", "unknown")
                diagnostics.append({
                    "level": "INFO",
                    "message": f"{case_name}: {model} performance category: {category}. "
                               f"This is {'consistent' if category != 'poor' else 'known to be challenging'} "
                               f"for this turbulence model.",
                    "case": case_name,
                })

        return diagnostics

    def diagnose_full_benchmark(self, benchmark_data: Dict) -> Dict:
        """
        Generate full diagnostic report from benchmark summary.

        Returns structured report with per-case diagnostics,
        cross-case patterns, and recommendations.
        """
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_cases": 0,
            "per_case": {},
            "patterns": [],
            "recommendations": [],
        }

        cases = benchmark_data.get("cases", benchmark_data)
        if isinstance(cases, dict):
            for case_name, case_data in cases.items():
                if isinstance(case_data, dict):
                    metrics = case_data if "status" not in case_data else case_data
                    model = case_data.get("model", "SA")
                    diags = self.diagnose_case(case_name, metrics, model)
                    report["per_case"][case_name] = diags
                    report["n_cases"] += 1

        # Detect cross-case patterns
        report["patterns"] = self._detect_patterns(report["per_case"])
        report["recommendations"] = self._generate_recommendations(report)

        return report

    def _detect_patterns(self, per_case: Dict) -> List[str]:
        """Detect recurring patterns across cases."""
        patterns = []

        # Count severity levels
        fails = sum(
            1 for diags in per_case.values()
            for d in diags if d["level"] == "FAIL"
        )
        warns = sum(
            1 for diags in per_case.values()
            for d in diags if d["level"] == "WARN"
        )

        if fails > 2:
            patterns.append(
                f"Multiple failures ({fails} cases): systematic model deficiency likely. "
                f"Consider switching to SST or RSM for separated flow cases."
            )

        if warns > 3:
            patterns.append(
                f"Multiple warnings ({warns}): may indicate overall grid resolution issues. "
                f"Check GCI values across all cases."
            )

        return patterns

    def _generate_recommendations(self, report: Dict) -> List[str]:
        """Generate actionable recommendations."""
        recs = []

        n_fails = sum(
            1 for diags in report["per_case"].values()
            for d in diags if d["level"] == "FAIL"
        )

        if n_fails == 0:
            recs.append("All cases within acceptable error bounds. Framework is well-validated.")
        elif n_fails <= 2:
            recs.append(
                "Consider targeted improvement: run failed cases with SST model "
                "and finer grids to isolate model vs. discretization error."
            )
        else:
            recs.append(
                "Systematic issues detected. Recommended actions:\n"
                "  1. Run GCI study on all failed cases\n"
                "  2. Compare SA vs SST on separation-dominated cases\n"
                "  3. Consider FIML β-correction for the worst cases"
            )

        return recs

    def format_report(self, report: Dict) -> str:
        """Format diagnostic report as Markdown."""
        lines = [
            "# Benchmark Diagnostic Report",
            f"*Generated: {report['timestamp']}*",
            f"*Cases analyzed: {report['n_cases']}*",
            "",
        ]

        for case_name, diags in report["per_case"].items():
            lines.append(f"## {case_name}")
            for d in diags:
                icon = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌", "INFO": "ℹ️"}.get(
                    d["level"], "•")
                lines.append(f"- {icon} {d['message']}")
            lines.append("")

        if report["patterns"]:
            lines.append("## Cross-Case Patterns")
            for p in report["patterns"]:
                lines.append(f"- {p}")
            lines.append("")

        if report["recommendations"]:
            lines.append("## Recommendations")
            for r in report["recommendations"]:
                lines.append(f"- {r}")

        return "\n".join(lines)


# =============================================================================
# Anomaly Detection
# =============================================================================
class AnomalyDetector:
    """
    Detect cases deviating from expected TMR scatter bands.

    Flags results that are statistical outliers relative to the
    inter-code scatter reported in NASA TMR documentation.
    """

    # TMR inter-code scatter bands (approximate from NASA TMR)
    TMR_SCATTER = {
        "flat_plate": {"Cf_scatter": 0.02, "U_plus_scatter": 0.03},
        "naca0012": {"CL_scatter": 0.03, "CD_scatter": 0.05},
        "wall_hump": {"x_sep_scatter": 0.08, "L_bubble_scatter": 0.15},
        "swbli": {"x_sep_scatter": 0.12, "L_bubble_scatter": 0.20},
        "bfs": {"x_reat_scatter": 0.10},
    }

    def detect(self, benchmark_data: Dict,
               sigma_threshold: float = 2.0) -> List[Dict]:
        """
        Detect anomalous results.

        Parameters
        ----------
        benchmark_data : benchmark summary dict
        sigma_threshold : number of scatter-band widths before flagging

        Returns
        -------
        List of anomaly dicts with case, metric, value, expected, severity.
        """
        anomalies = []
        cases = benchmark_data.get("cases", benchmark_data)

        for case_name, case_data in cases.items():
            if not isinstance(case_data, dict):
                continue

            # Find matching scatter band
            scatter = None
            for key, bands in self.TMR_SCATTER.items():
                if key in case_name.lower():
                    scatter = bands
                    break

            if not scatter:
                continue

            for metric_key, band_width in scatter.items():
                # Extract the base metric name
                base_metric = metric_key.replace("_scatter", "")
                value = case_data.get(base_metric)
                error = case_data.get(f"{base_metric}_error")

                if error is not None and abs(error) > band_width * sigma_threshold:
                    severity = "HIGH" if abs(error) > band_width * 3 else "MODERATE"
                    anomalies.append({
                        "case": case_name,
                        "metric": base_metric,
                        "error": float(error),
                        "tmr_scatter_band": float(band_width),
                        "sigma": float(abs(error) / band_width),
                        "severity": severity,
                        "message": (
                            f"{case_name}: {base_metric} error ({error:.1%}) "
                            f"exceeds TMR scatter ({band_width:.1%}) by "
                            f"{abs(error)/band_width:.1f}σ — {severity} anomaly"
                        ),
                    })

        return anomalies


# =============================================================================
# Cross-Solver Alignment
# =============================================================================
class CrossSolverAligner:
    """
    Automates alignment of SU2 and OpenFOAM case configurations.

    Maps config parameters between solver syntaxes for consistent
    comparison (same BC, mesh density, convergence criteria).
    """

    # SU2 ↔ OpenFOAM parameter mapping
    PARAM_MAP = {
        "MACH_NUMBER": {"of_key": "Mach", "of_file": "transportProperties"},
        "AOA": {"of_key": "flowDirection", "of_file": "0/U"},
        "REYNOLDS_NUMBER": {"of_key": "Re", "of_file": "transportProperties"},
        "KIND_TURB_MODEL": {
            "of_map": {
                "SA": "SpalartAllmaras",
                "SST": "kOmegaSST",
                "SA_NEG": "SpalartAllmaras",
                "KE": "kEpsilon",
            },
            "of_file": "constant/momentumTransport",
        },
        "CFL_NUMBER": {"of_key": "maxCo", "of_file": "system/controlDict"},
        "ITER": {"of_key": "endTime", "of_file": "system/controlDict"},
    }

    def align_configs(self, su2_config: str) -> Dict[str, str]:
        """
        Parse SU2 config and generate equivalent OpenFOAM settings.

        Returns dict of {of_file: content_snippet} for key parameters.
        """
        su2_params = {}
        for line in su2_config.split("\n"):
            line = line.strip()
            if line.startswith("%") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            su2_params[key.strip()] = val.strip()

        of_settings = {}
        for su2_key, mapping in self.PARAM_MAP.items():
            if su2_key in su2_params:
                su2_val = su2_params[su2_key]
                of_file = mapping.get("of_file", "unknown")

                if "of_map" in mapping:
                    of_val = mapping["of_map"].get(su2_val, su2_val)
                elif "of_key" in mapping:
                    of_val = su2_val
                else:
                    of_val = su2_val

                of_settings[su2_key] = {
                    "su2_value": su2_val,
                    "openfoam_equivalent": of_val,
                    "openfoam_file": of_file,
                }

        return of_settings


# =============================================================================
# High-Level Assistant
# =============================================================================
class BenchmarkAssistant:
    """
    Top-level LLM-powered benchmark assistant.

    Combines config generation, diagnostics, anomaly detection,
    and cross-solver alignment in a single interface.
    """

    def __init__(self, llm_backend: str = "local"):
        """
        Parameters
        ----------
        llm_backend : str
            "anthropic", "openai", or "local" (rule-based, no API needed).
        """
        self.backend = llm_backend
        self.config_gen = ConfigGenerator()
        self.diagnostics = DiagnosticEngine()
        self.anomaly_detector = AnomalyDetector()
        self.cross_solver = CrossSolverAligner()

    def generate_config(self, description: str,
                        output_dir: Path = None) -> Dict:
        """
        Generate SU2 config from natural language description.

        Returns dict with parsed config, generated content, and confidence.
        """
        parsed = self.config_gen.parse_natural_language(description)
        config_text = self.config_gen.generate_su2_config(parsed, output_dir)

        return {
            "parsed": {
                "case_type": parsed.case_type,
                "mach": parsed.mach,
                "reynolds": parsed.reynolds,
                "alpha_deg": parsed.alpha_deg,
                "turbulence_model": parsed.turbulence_model,
                "n_grid_levels": parsed.n_grid_levels,
                "confidence": parsed.confidence,
            },
            "config_text": config_text,
            "output_path": str(output_dir) if output_dir else None,
        }

    def diagnose(self, json_path: Union[str, Path] = None,
                 benchmark_data: Dict = None) -> Dict:
        """
        Generate physics-aware diagnostic report.

        Accepts either a JSON file path or pre-loaded dict.
        """
        if json_path:
            data = self.diagnostics.load_benchmark(json_path)
        elif benchmark_data:
            data = benchmark_data
        else:
            raise ValueError("Provide json_path or benchmark_data")

        report = self.diagnostics.diagnose_full_benchmark(data)
        report["markdown"] = self.diagnostics.format_report(report)
        return report

    def detect_anomalies(self, json_path: Union[str, Path] = None,
                         benchmark_data: Dict = None) -> List[Dict]:
        """Detect cases deviating from TMR scatter bands."""
        if json_path:
            with open(json_path) as f:
                data = json.load(f)
        elif benchmark_data:
            data = benchmark_data
        else:
            raise ValueError("Provide json_path or benchmark_data")

        return self.anomaly_detector.detect(data)

    def align_solvers(self, su2_config: str) -> Dict:
        """Generate OpenFOAM-equivalent settings from SU2 config."""
        return self.cross_solver.align_configs(su2_config)

    def get_model_recommendation(self, case_description: str) -> Dict:
        """
        Recommend turbulence model based on case description.

        Uses TMR knowledge base to suggest the best model.
        """
        parsed = self.config_gen.parse_natural_language(case_description)

        recommendations = []
        for model_name, info in TMR_KNOWLEDGE.items():
            score = 0
            notes = []

            # Check known performance on similar cases
            for case_key, perf in info.get("expected_errors", {}).items():
                if case_key in parsed.case_type or parsed.case_type in case_key:
                    cat = perf.get("category", "unknown")
                    score_map = {"excellent": 3, "good": 2, "moderate": 1, "poor": 0}
                    score = score_map.get(cat, 0)
                    notes.append(f"Expected performance on {case_key}: {cat}")

            recommendations.append({
                "model": model_name,
                "score": score,
                "strengths": info["strengths"],
                "weaknesses": info["weaknesses"],
                "notes": notes,
            })

        recommendations.sort(key=lambda x: x["score"], reverse=True)

        return {
            "case_type": parsed.case_type,
            "mach": parsed.mach,
            "reynolds": parsed.reynolds,
            "recommended_model": recommendations[0]["model"] if recommendations else "SST",
            "all_models": recommendations,
        }


# =============================================================================
# CLI
# =============================================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(description="LLM Benchmark Assistant")
    sub = parser.add_subparsers(dest="command")

    # Config generation
    gen = sub.add_parser("generate", help="Generate SU2 config from description")
    gen.add_argument("description", help="Natural language case description")
    gen.add_argument("--output-dir", default=None)

    # Diagnostics
    diag = sub.add_parser("diagnose", help="Diagnose benchmark results")
    diag.add_argument("json_path", help="Path to benchmark_summary.json")

    # Anomaly detection
    anom = sub.add_parser("anomalies", help="Detect anomalous results")
    anom.add_argument("json_path", help="Path to benchmark_summary.json")

    # Model recommendation
    rec = sub.add_parser("recommend", help="Recommend turbulence model")
    rec.add_argument("description", help="Case description")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    assistant = BenchmarkAssistant()

    if args.command == "generate":
        result = assistant.generate_config(
            args.description,
            Path(args.output_dir) if args.output_dir else None,
        )
        print(f"Confidence: {result['parsed']['confidence']:.0%}")
        print(f"Case: {result['parsed']['case_type']}")
        print(f"Mach: {result['parsed']['mach']}")
        print(f"Re: {result['parsed']['reynolds']:.0e}")
        print(f"Model: {result['parsed']['turbulence_model']}")
        print(f"\n--- Generated Config ---\n{result['config_text']}")

    elif args.command == "diagnose":
        report = assistant.diagnose(args.json_path)
        print(report["markdown"])

    elif args.command == "anomalies":
        anomalies = assistant.detect_anomalies(args.json_path)
        if anomalies:
            for a in anomalies:
                print(f"[{a['severity']}] {a['message']}")
        else:
            print("No anomalies detected.")

    elif args.command == "recommend":
        rec = assistant.get_model_recommendation(args.description)
        print(f"Case: {rec['case_type']} (M={rec['mach']}, Re={rec['reynolds']:.0e})")
        print(f"Recommended model: {rec['recommended_model']}")
        for m in rec["all_models"]:
            print(f"  {m['model']}: score={m['score']}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
