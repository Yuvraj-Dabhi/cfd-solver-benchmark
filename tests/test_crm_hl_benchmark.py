#!/usr/bin/env python3
"""
Tests for CRM-HL / HLPW-5 Benchmark Module
=============================================
Validates warm-start strategy, drag decomposition, numerical
dissipation study, WMLES configuration, and full benchmark runner.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ml_augmentation.crm_hl_benchmark import (
    CRMHLBenchmarkRunner,
    CRMHLConfiguration,
    DragDecompositionAnalyzer,
    NumericalDissipationStudy,
    WarmStartManager,
    WMLESConfiguration,
)


# =========================================================================
# TestCRMHLConfiguration
# =========================================================================
class TestCRMHLConfiguration:
    """Tests for CRM-HL geometry configuration."""

    def test_default_values(self):
        config = CRMHLConfiguration()
        assert config.mach == pytest.approx(0.2)
        assert config.slat_deflection == pytest.approx(30.0)
        assert config.flap_deflection == pytest.approx(40.0)
        assert config.include_nacelle is True

    def test_su2_config_generation(self):
        config = CRMHLConfiguration()
        su2_cfg = config.to_su2_config(alpha=8.0)
        assert su2_cfg["SOLVER"] == "RANS"
        assert su2_cfg["AOA"] == 8.0
        assert su2_cfg["MACH_NUMBER"] == pytest.approx(0.2)
        assert su2_cfg["REYNOLDS_NUMBER"] == pytest.approx(5.49e6)

    def test_grid_levels(self):
        config = CRMHLConfiguration()
        assert config.get_grid_size("coarse") == 25_000_000
        assert config.get_grid_size("medium") == 55_000_000
        assert config.get_grid_size("fine") == 130_000_000

    def test_alpha_range(self):
        config = CRMHLConfiguration()
        assert len(config.alpha_range) >= 5
        assert 0 in config.alpha_range
        assert any(a >= 20 for a in config.alpha_range)


# =========================================================================
# TestWarmStartManager
# =========================================================================
class TestWarmStartManager:
    """Tests for warm-start α-sweep strategy."""

    def test_sweep_plan_generation(self):
        ws = WarmStartManager(alpha_sequence=[0, 4, 8, 12, 16, 20])
        plan = ws.generate_sweep_plan()
        assert len(plan) == 6
        assert plan[0]["restart_from"] is None
        assert plan[1]["restart_from"] == 0

    def test_near_stall_detection(self):
        ws = WarmStartManager(alpha_sequence=[0, 8, 16, 19, 21])
        plan = ws.generate_sweep_plan()
        assert plan[3]["near_stall"] is True  # α = 19
        assert plan[0]["near_stall"] is False  # α = 0

    def test_convergence_logging(self):
        ws = WarmStartManager()
        ws.log_convergence(8.0, True, residual=1e-8, cl=1.5, cd=0.06)
        assert 8.0 in ws.convergence_log
        assert ws.convergence_log[8.0]["converged"] is True

    def test_restart_config(self):
        ws = WarmStartManager()
        ws.log_convergence(0, True, cl=0.3, cd=0.02)
        ws.log_convergence(4, True, cl=0.8, cd=0.04)
        config = ws.get_restart_config(8.0)
        assert config["RESTART_SOL"] == "YES"
        assert config["AOA"] == 8.0

    def test_first_alpha_no_restart(self):
        ws = WarmStartManager()
        config = ws.get_restart_config(0.0)
        assert config["RESTART_SOL"] == "NO"

    def test_summary(self):
        ws = WarmStartManager()
        ws.log_convergence(8.0, True, cl=1.5, cd=0.06)
        s = ws.summary()
        assert "8.0" in s


# =========================================================================
# TestDragDecompositionAnalyzer
# =========================================================================
class TestDragDecompositionAnalyzer:
    """Tests for drag decomposition."""

    def test_decompose_shape(self):
        analyzer = DragDecompositionAnalyzer()
        n = 100
        result = analyzer.decompose(
            surface_cp=np.random.randn(n),
            surface_cf=np.abs(np.random.randn(n)) * 0.003,
            surface_normals=np.random.randn(n, 3),
            surface_areas=np.abs(np.random.randn(n)) * 0.01,
            alpha=8.0,
        )
        assert "CD_total" in result
        assert "CD_pressure" in result
        assert "CD_viscous" in result
        assert "CL" in result

    def test_pressure_viscous_sum(self):
        analyzer = DragDecompositionAnalyzer()
        n = 50
        result = analyzer.decompose(
            surface_cp=np.ones(n) * -0.5,
            surface_cf=np.ones(n) * 0.003,
            surface_normals=np.tile([1, 0, 0], (n, 1)).astype(float),
            surface_areas=np.ones(n) * 0.01,
        )
        # Total should equal pressure + viscous
        assert result["CD_total"] == pytest.approx(
            result["CD_pressure"] + result["CD_viscous"], abs=1e-10)

    def test_induced_drag(self):
        analyzer = DragDecompositionAnalyzer()
        cdi = analyzer.compute_induced_drag(CL=1.5, AR=9.0, e=0.85)
        assert cdi > 0
        # CDi = 1.5² / (π * 0.85 * 9) ≈ 0.094
        assert cdi == pytest.approx(0.094, abs=0.01)

    def test_scheme_sensitivity(self):
        analyzer = DragDecompositionAnalyzer()
        result = analyzer.scheme_sensitivity(cd_jst=0.0310, cd_roe=0.0295)
        assert result["difference"] > 0
        assert "preferred_scheme" in result


# =========================================================================
# TestNumericalDissipationStudy
# =========================================================================
class TestNumericalDissipationStudy:
    """Tests for numerical dissipation study."""

    def test_jst_config_generation(self):
        study = NumericalDissipationStudy()
        configs = study.generate_jst_configs()
        assert len(configs) > 1
        # Should include Roe reference
        assert any("Roe" in c.get("label", "") for c in configs)

    def test_analyze_results(self):
        study = NumericalDissipationStudy()
        cd_values = {
            "JST_k2=0.5_k4=0.02": 0.032,
            "JST_k2=1.0_k4=0.04": 0.034,
            "Roe_MUSCL": 0.030,
        }
        result = study.analyze_results(cd_values)
        assert result["best_scheme"] == "Roe_MUSCL"
        assert result["cd_spread"] > 0


# =========================================================================
# TestWMLESConfiguration
# =========================================================================
class TestWMLESConfiguration:
    """Tests for WMLES configuration."""

    def test_su2_config(self):
        wmles = WMLESConfiguration()
        cfg = wmles.to_su2_config()
        assert cfg["TIME_DOMAIN"] == "YES"
        assert cfg["SGS_MODEL"] == "WALE"
        assert cfg["TIME_STEP"] == pytest.approx(1e-5)

    def test_cost_estimate(self):
        wmles = WMLESConfiguration(n_timesteps=10000)
        cost = wmles.estimate_cost(n_cells=50_000_000)
        assert cost["n_cells"] == 50_000_000
        assert cost["total_hours"] > 0
        assert cost["averaging_steps"] > 0


# =========================================================================
# TestCRMHLBenchmarkRunner
# =========================================================================
class TestCRMHLBenchmarkRunner:
    """Tests for the full benchmark runner."""

    def test_alpha_sweep_configs(self):
        runner = CRMHLBenchmarkRunner()
        configs = runner.generate_alpha_sweep_configs()
        assert len(configs) == len(runner.config.alpha_range)

    def test_log_result_and_predict_clmax(self):
        runner = CRMHLBenchmarkRunner()
        # Simulate α-sweep
        for alpha, cl in [(0, 0.3), (4, 0.8), (8, 1.3), (12, 1.7),
                          (16, 2.0), (20, 2.1), (21, 1.9)]:
            runner.log_alpha_result(alpha, cl=cl, cd=0.03 + 0.001 * alpha)
        clmax = runner.predict_clmax()
        assert clmax["CLmax"] == pytest.approx(2.1, abs=0.01)
        assert clmax["alpha_stall"] == 20

    def test_hlpw5_compliance(self):
        runner = CRMHLBenchmarkRunner()
        runner.log_alpha_result(8.0, cl=1.3, cd=0.04)
        compliance = runner.hlpw5_compliance_check()
        assert "geometry" in compliance
        assert "conditions" in compliance
        assert compliance["geometry"]["slat_angle"] is True

    def test_report(self):
        runner = CRMHLBenchmarkRunner()
        runner.log_alpha_result(8.0, cl=1.3, cd=0.04)
        report = runner.report()
        assert "CRM-HL" in report
        assert "α=" in report


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
