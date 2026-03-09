"""
Tests for Hypersonic & Variable-Property Extrapolation
======================================================
Validates heat flux / profile extraction and the OOD ML evaluation metrics.
"""

import sys
from pathlib import Path
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Test extractors
from run_swbli import HypersonicExtractor
from run_heated_jet import VariablePropertyExtractor
from scripts.ml_augmentation.hypersonic_extrapolation import (
    VariablePropertyMLTester,
    HypersonicExtrapolationReport,
    run_benchmark_suite
)


class TestHypersonicExtractor:
    def test_extract_heat_flux_synthetic(self):
        data = {}
        result = HypersonicExtractor.extract_heat_flux(data)
        assert "x" in result
        assert "qw" in result
        assert len(result["x"]) == 100
        assert len(result["qw"]) == 100

    def test_extract_heat_flux_real(self):
        x = np.array([0.1, 0.2, 0.3])
        qw = np.array([50.0, 60.0, 70.0])
        data = {"x": x, "Heat_Flux": qw}
        result = HypersonicExtractor.extract_heat_flux(data)
        assert np.array_equal(result["qw"], qw)

    def test_extract_profiles(self):
        data = {}
        result = HypersonicExtractor.extract_boundary_layer_profile(data, 0.5)
        for key in ["y", "u", "T", "rho", "mach"]:
            assert key in result
            assert len(result[key]) == 50


class TestVariablePropertyExtractor:
    def test_extract_heat_flux_synthetic(self):
        data = {}
        result = VariablePropertyExtractor.extract_heat_flux(data)
        assert "x" in result
        assert "qw" in result
        assert len(result["x"]) == 50

    def test_extract_jet_profiles(self):
        data = {}
        result = VariablePropertyExtractor.extract_jet_profile(data, 0.5)
        for key in ["r", "u", "T", "rho", "mach"]:
            assert key in result
            assert len(result[key]) == 50


class TestVariablePropertyMLTester:
    @pytest.fixture
    def tester(self):
        return VariablePropertyMLTester(seed=42)

    def test_evaluate_swbli_degradation(self, tester):
        x = np.linspace(0, 1, 50)
        data = {
            "name": "Test SWBLI",
            "x": x,
            "cf_true": np.ones_like(x),
            "cf_base": np.zeros_like(x),
            "qw_true": np.ones_like(x) * 100,
            "qw_base": np.zeros_like(x),
        }
        res = tester.evaluate_swbli_case("TestModel", data)
        assert res.model_name == "TestModel"
        assert res.flow_type == "hypersonic_swbli"
        # Since it's OOD, qw improvement should be poor or negative
        assert res.is_ood_failure is True
        assert "Catastrophic degradation in Wall Heat-Flux" in res.failure_mode

    def test_evaluate_heated_jet_degradation(self, tester):
        y = np.linspace(0, 1, 50)
        data = {
            "name": "Test Jet",
            "y": y,
            "u_true": np.ones_like(y),
            "u_base": np.zeros_like(y),
            "T_true": np.ones_like(y) * 300,
            "T_base": np.zeros_like(y),
        }
        res = tester.evaluate_heated_jet_case("TestModel", data)
        assert res.model_name == "TestModel"
        assert res.flow_type == "variable_property_jet"
        assert res.is_ood_failure is True
        assert "Failure to transfer" in res.failure_mode


class TestExtrapolationReport:
    def test_generate_markdown(self):
        tester = VariablePropertyMLTester(seed=42)
        x = np.linspace(0, 1, 10)
        swbli_data = {
            "x": x, "cf_true": x, "cf_base": x*0.5, "qw_true": x*100, "qw_base": x*50
        }
        res1 = tester.evaluate_swbli_case("Model1", swbli_data)
        
        y = np.linspace(0, 1, 10)
        jet_data = {
            "y": y, "u_true": y, "u_base": y*0.5, "T_true": y*300, "T_base": y*150
        }
        res2 = tester.evaluate_heated_jet_case("Model1", jet_data)
        
        report = HypersonicExtrapolationReport([res1, res2])
        md = report.generate_markdown()
        
        assert "Out-Of-Comfort-Zone Extrapolation Benchmark" in md
        assert "Model1 on SWBLI_M5" in md
        assert "Model1 on Heated_Jet_M1.63" in md
        assert "EXTRAPOLATION FAILURE DETECTED" in md

    def test_run_benchmark_suite(self):
        out = run_benchmark_suite(["TBNN"])
        assert "results" in out
        assert "markdown_report" in out
        assert len(out["results"]) == 2  # 1 model * 2 cases
        assert "TBNN" in out["markdown_report"]
