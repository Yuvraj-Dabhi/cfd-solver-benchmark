"""
Tests for LLM Benchmark Assistant
====================================
Validates NL config parsing, SU2 generation, diagnostics,
anomaly detection, and cross-solver alignment.
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.orchestration.llm_benchmark_assistant import (
    ConfigGenerator,
    ParsedConfig,
    DiagnosticEngine,
    AnomalyDetector,
    CrossSolverAligner,
    BenchmarkAssistant,
)


class TestConfigParser:
    """Test natural language → structured config parsing."""

    def test_parse_bfs_case(self):
        gen = ConfigGenerator()
        parsed = gen.parse_natural_language(
            "Mach 0.3 BFS at Re=50000 with SST and 3 grid levels")

        assert parsed.case_type == "bfs"
        assert abs(parsed.mach - 0.3) < 1e-6
        assert parsed.reynolds == 50000
        assert parsed.turbulence_model == "SST"
        assert parsed.n_grid_levels == 3
        assert parsed.confidence > 0.5

    def test_parse_naca_case(self):
        gen = ConfigGenerator()
        parsed = gen.parse_natural_language(
            "NACA 0012 at α=15°, Re=6M, SA model")

        assert parsed.case_type == "naca0012"
        assert parsed.alpha_deg == 15.0
        assert parsed.turbulence_model == "SA"

    def test_parse_swbli_case(self):
        gen = ConfigGenerator()
        parsed = gen.parse_natural_language(
            "SWBLI case, Mach 2.0, Re=2.3e7, SA-neg")

        assert parsed.case_type == "swbli"
        assert abs(parsed.mach - 2.0) < 1e-6
        assert parsed.turbulence_model == "SA_NEG"

    def test_parse_wall_hump(self):
        gen = ConfigGenerator()
        parsed = gen.parse_natural_language(
            "Wall hump simulation, Re=936000, SST model, 5000 iterations")

        assert parsed.case_type == "wall_hump"
        assert parsed.turbulence_model == "SST"
        assert parsed.max_iterations == 5000

    def test_unknown_case_low_confidence(self):
        gen = ConfigGenerator()
        parsed = gen.parse_natural_language("some random text")
        assert parsed.case_type == "generic"
        assert parsed.confidence < 0.6


class TestSU2ConfigGeneration:
    """Test SU2 config file generation."""

    def test_generates_valid_config(self):
        gen = ConfigGenerator()
        parsed = gen.parse_natural_language("BFS at Mach 0.1, Re=50000, SST")
        config = gen.generate_su2_config(parsed)

        assert "SOLVER= RANS" in config
        assert "SST" in config
        assert "MACH_NUMBER= 0.1" in config
        assert "REYNOLDS_NUMBER= 50000" in config

    def test_compressible_scheme(self):
        gen = ConfigGenerator()
        parsed = ParsedConfig(mach=2.0, case_type="swbli", turbulence_model="SA")
        config = gen.generate_su2_config(parsed)
        assert "JST" in config  # Compressible scheme

    def test_incompressible_scheme(self):
        gen = ConfigGenerator()
        parsed = ParsedConfig(mach=0.1, case_type="bfs", turbulence_model="SST")
        config = gen.generate_su2_config(parsed)
        assert "ROE" in config  # Incompressible scheme

    def test_write_to_file(self, tmp_path):
        gen = ConfigGenerator()
        parsed = gen.parse_natural_language("flat plate Re=5e6 SA")
        gen.generate_su2_config(parsed, output_dir=tmp_path)

        cfg_files = list(tmp_path.glob("*.cfg"))
        assert len(cfg_files) == 1
        assert cfg_files[0].stat().st_size > 100


class TestDiagnosticEngine:
    """Test physics-aware diagnostics."""

    def test_diagnose_passing_case(self):
        engine = DiagnosticEngine()
        metrics = {"Cf_RMSE": 0.0005, "GCI_pct": 1.5}
        diags = engine.diagnose_case("flat_plate", metrics, model="SA")

        levels = [d["level"] for d in diags]
        assert "PASS" in levels

    def test_diagnose_failing_case(self):
        engine = DiagnosticEngine()
        metrics = {"x_sep_error": 0.15, "L_bubble_error": 0.25}
        diags = engine.diagnose_case("swbli", metrics, model="SA")

        levels = [d["level"] for d in diags]
        assert "FAIL" in levels or "WARN" in levels

    def test_full_benchmark_report(self):
        engine = DiagnosticEngine()
        benchmark = {
            "cases": {
                "flat_plate": {"Cf_RMSE": 0.0003, "status": "PASS"},
                "wall_hump": {"x_sep_error": 0.12, "L_bubble_error": 0.18},
            }
        }
        report = engine.diagnose_full_benchmark(benchmark)

        assert report["n_cases"] == 2
        assert "flat_plate" in report["per_case"]
        assert "wall_hump" in report["per_case"]

    def test_format_report_markdown(self):
        engine = DiagnosticEngine()
        benchmark = {
            "cases": {
                "flat_plate": {"Cf_RMSE": 0.0003},
            }
        }
        report = engine.diagnose_full_benchmark(benchmark)
        md = engine.format_report(report)
        assert "# Benchmark Diagnostic Report" in md
        assert "flat_plate" in md


class TestAnomalyDetector:
    """Test TMR scatter band anomaly detection."""

    def test_no_anomalies(self):
        detector = AnomalyDetector()
        data = {
            "cases": {
                "flat_plate_SA": {"Cf_error": 0.01},
            }
        }
        anomalies = detector.detect(data)
        assert len(anomalies) == 0

    def test_detects_anomaly(self):
        detector = AnomalyDetector()
        data = {
            "cases": {
                "wall_hump_SA": {"x_sep_error": 0.50},  # Way over scatter
            }
        }
        anomalies = detector.detect(data)
        assert len(anomalies) > 0
        assert anomalies[0]["severity"] in ("MODERATE", "HIGH")

    def test_severity_levels(self):
        detector = AnomalyDetector()
        data = {
            "cases": {
                "swbli_SA": {"x_sep_error": 0.80},  # Extreme
            }
        }
        anomalies = detector.detect(data)
        if anomalies:
            assert anomalies[0]["severity"] == "HIGH"


class TestCrossSolverAligner:
    """Test SU2 → OpenFOAM config alignment."""

    def test_align_basic_config(self):
        aligner = CrossSolverAligner()
        su2_config = """
SOLVER= RANS
KIND_TURB_MODEL= SST
MACH_NUMBER= 0.15
REYNOLDS_NUMBER= 936000
AOA= 0.0
CFL_NUMBER= 10.0
ITER= 5000
"""
        result = aligner.align_configs(su2_config)

        assert "KIND_TURB_MODEL" in result
        assert result["KIND_TURB_MODEL"]["openfoam_equivalent"] == "kOmegaSST"
        assert "MACH_NUMBER" in result
        assert "REYNOLDS_NUMBER" in result

    def test_sa_to_openfoam(self):
        aligner = CrossSolverAligner()
        su2_config = "KIND_TURB_MODEL= SA\n"
        result = aligner.align_configs(su2_config)
        assert result["KIND_TURB_MODEL"]["openfoam_equivalent"] == "SpalartAllmaras"


class TestBenchmarkAssistant:
    """Test high-level assistant."""

    def test_generate_config(self):
        assistant = BenchmarkAssistant()
        result = assistant.generate_config(
            "Backward facing step, Mach 0.1, Re=36000, SST, 3 grid levels")

        assert result["parsed"]["case_type"] == "bfs"
        assert result["parsed"]["turbulence_model"] == "SST"
        assert "SOLVER= RANS" in result["config_text"]

    def test_diagnose_from_dict(self):
        assistant = BenchmarkAssistant()
        data = {"cases": {"flat_plate": {"Cf_RMSE": 0.0003}}}
        report = assistant.diagnose(benchmark_data=data)

        assert "markdown" in report
        assert report["n_cases"] == 1

    def test_model_recommendation(self):
        assistant = BenchmarkAssistant()
        rec = assistant.get_model_recommendation(
            "wall hump case, Re=936000")

        assert rec["case_type"] == "wall_hump"
        assert rec["recommended_model"] in ("SA", "SST")
        assert len(rec["all_models"]) > 0

    def test_align_solvers(self):
        assistant = BenchmarkAssistant()
        su2_cfg = "KIND_TURB_MODEL= SST\nMACH_NUMBER= 0.15\n"
        result = assistant.align_solvers(su2_cfg)
        assert "KIND_TURB_MODEL" in result
