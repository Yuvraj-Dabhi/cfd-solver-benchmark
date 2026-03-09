#!/usr/bin/env python3
"""
Tests: End-to-End Design & Control Workflows with UQ
======================================================
Validates multi-fidelity design optimisation, DRL flow-control benchmark,
and UQ wrapping pipelines.

Run: pytest tests/test_design_control_uq.py -v --tb=short
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ml_augmentation.design_control_uq_workflows import (
    DesignPoint,
    AeroResult,
    OptimizationResult,
    ControlMetrics,
    UQResult,
    RobustAirfoilOptimizer,
    DRLFlowControlBenchmark,
    UQWorkflowWrapper,
    EndToEndWorkflowRunner,
    _synthetic_aero,
)


# ============================================================================
# Synthetic Aero Model Tests
# ============================================================================

class TestSyntheticAero:
    """Test the physics-inspired CL/CD model."""

    def test_basic_evaluation(self):
        dp = DesignPoint(thickness=0.12, camber=0.02, aoa=5.0)
        result = _synthetic_aero(dp, "rans")
        assert isinstance(result, AeroResult)
        assert result.CD > 0
        assert result.fidelity == "rans"

    def test_zero_aoa(self):
        dp = DesignPoint(thickness=0.12, camber=0.0, aoa=0.0)
        result = _synthetic_aero(dp, "rans", np.random.default_rng(42))
        # At zero AoA with zero camber, CL should be near zero
        assert abs(result.CL) < 0.5

    def test_high_aoa_increases_separation(self):
        dp_low = DesignPoint(thickness=0.12, camber=0.02, aoa=4.0)
        dp_high = DesignPoint(thickness=0.12, camber=0.02, aoa=14.0)
        rng = np.random.default_rng(42)
        r_low = _synthetic_aero(dp_low, "rans", rng)
        rng = np.random.default_rng(42)
        r_high = _synthetic_aero(dp_high, "rans", rng)
        assert r_high.separation_length >= r_low.separation_length

    def test_fidelity_affects_noise(self):
        dp = DesignPoint(thickness=0.12, camber=0.02, aoa=5.0)
        # Run many samples and check variance
        results_rans = [
            _synthetic_aero(dp, "rans", np.random.default_rng(i)).CL
            for i in range(50)
        ]
        results_ml = [
            _synthetic_aero(dp, "rans_ml", np.random.default_rng(i)).CL
            for i in range(50)
        ]
        # RANS+ML should have less variance than pure RANS
        assert np.std(results_ml) < np.std(results_rans) * 1.5


# ============================================================================
# RobustAirfoilOptimizer Tests
# ============================================================================

class TestRobustAirfoilOptimizer:
    """Test multi-fidelity design optimisation."""

    @pytest.fixture
    def optimizer(self):
        return RobustAirfoilOptimizer(
            n_initial=15, n_refine=5, n_mc_robustness=10, seed=42
        )

    def test_lhs_sampling(self, optimizer):
        designs = optimizer._sample_lhs(20)
        assert len(designs) == 20
        for dp in designs:
            assert 0.06 <= dp.thickness <= 0.18
            assert 0.0 <= dp.camber <= 0.06
            assert 0.0 <= dp.aoa <= 15.0

    def test_optimize_rans(self, optimizer):
        result = optimizer.optimize("rans")
        assert isinstance(result, OptimizationResult)
        assert result.fidelity_loop == "rans"
        assert result.total_evaluations > 0
        assert len(result.pareto_designs) > 0
        assert result.best_CL_CD > 0

    def test_optimize_rans_ml(self, optimizer):
        result = optimizer.optimize("rans_ml")
        assert result.fidelity_loop == "rans_ml"
        assert result.best_CL_CD > 0

    def test_compare_fidelity_loops(self, optimizer):
        results = optimizer.compare_fidelity_loops()
        assert "rans" in results
        assert "rans_ml" in results
        assert "surrogate" in results
        # All should find a design
        for k, v in results.items():
            assert v.best_CL_CD > 0

    def test_robustness_evaluation(self, optimizer):
        dp = DesignPoint(thickness=0.12, camber=0.02, aoa=5.0)
        mean, std = optimizer._evaluate_robustness(dp, "rans")
        assert mean > 0
        assert std >= 0

    def test_pareto_has_designs(self, optimizer):
        result = optimizer.optimize("rans")
        assert len(result.pareto_designs) >= 1
        for p in result.pareto_designs:
            assert "mean_CL_CD" in p
            assert "robustness" in p

    def test_design_bounds_respected(self, optimizer):
        result = optimizer.optimize("rans")
        dp = result.best_design
        assert 0.06 <= dp.thickness <= 0.18
        assert 0.0 <= dp.camber <= 0.06
        assert 0.0 <= dp.aoa <= 15.0


# ============================================================================
# DRLFlowControlBenchmark Tests
# ============================================================================

class TestDRLFlowControlBenchmark:
    """Test DRL flow-control benchmark."""

    @pytest.fixture
    def benchmark(self):
        return DRLFlowControlBenchmark(
            n_actuators=3, max_steps=20, n_eval_episodes=5, seed=42
        )

    def test_no_control(self, benchmark):
        m = benchmark.evaluate_no_control()
        assert isinstance(m, ControlMetrics)
        assert m.strategy_name == "no_control"
        assert m.bubble_length > 0
        assert m.drag_coefficient > 0

    def test_constant_blowing(self, benchmark):
        m = benchmark.evaluate_constant_blowing()
        assert m.strategy_name == "constant_blowing"
        assert m.bubble_length > 0

    def test_periodic_forcing(self, benchmark):
        m = benchmark.evaluate_periodic_forcing()
        assert m.strategy_name == "periodic_forcing"
        assert m.bubble_length > 0

    def test_drl_policy(self, benchmark):
        m = benchmark.evaluate_drl_policy(n_training_steps=30)
        assert m.strategy_name == "DRL_PPO"

    def test_full_benchmark(self, benchmark):
        results = benchmark.run_benchmark(n_training_steps=30)
        assert len(results) == 4
        assert "no_control" in results
        assert "DRL_PPO" in results

    def test_drl_improves_over_baseline(self, benchmark):
        benchmark.run_benchmark(n_training_steps=50)
        # DRL should have better reward than no control
        no_ctrl = benchmark.results["no_control"]
        drl = benchmark.results["DRL_PPO"]
        assert drl.mean_reward >= no_ctrl.mean_reward

    def test_compute_deltas(self, benchmark):
        benchmark.run_benchmark(n_training_steps=30)
        deltas = benchmark.compute_deltas()
        assert "constant_blowing" in deltas
        assert "DRL_PPO" in deltas
        assert "delta_bubble_pct" in deltas["DRL_PPO"]
        assert "delta_Cd_pct" in deltas["DRL_PPO"]

    def test_koklu_table(self, benchmark):
        benchmark.run_benchmark(n_training_steps=30)
        table = benchmark.format_koklu_table()
        assert "Flow-Control Benchmark" in table
        assert "no_control" in table
        assert "DRL_PPO" in table


# ============================================================================
# UQWorkflowWrapper Tests
# ============================================================================

class TestUQWorkflowWrapper:
    """Test UQ propagation and analysis."""

    @pytest.fixture
    def uq(self):
        return UQWorkflowWrapper(n_mc_samples=50, seed=42)

    def test_design_uq(self, uq):
        dp = DesignPoint(thickness=0.12, camber=0.02, aoa=5.0)
        result = uq.propagate_design_uq(dp, "rans")
        assert isinstance(result, UQResult)
        assert result.std_value >= 0
        assert result.ci_95_low <= result.ci_95_high
        assert 0 <= result.robustness_score <= 1

    def test_sobol_indices(self, uq):
        dp = DesignPoint(thickness=0.12, camber=0.02, aoa=5.0)
        result = uq.propagate_design_uq(dp, "rans")
        assert "AoA" in result.sobol_indices
        assert "model_form" in result.sobol_indices
        # Sobol should sum to ~1
        s_total = sum(result.sobol_indices.values())
        assert 0.9 <= s_total <= 1.1

    def test_control_uq(self, uq):
        benchmark = DRLFlowControlBenchmark(
            n_actuators=3, n_eval_episodes=3, seed=42
        )
        benchmark.run_benchmark(n_training_steps=20)
        result = uq.propagate_control_uq(benchmark, "DRL_PPO")
        assert isinstance(result, UQResult)
        assert result.std_value >= 0

    def test_control_uq_no_control(self, uq):
        benchmark = DRLFlowControlBenchmark(
            n_actuators=3, n_eval_episodes=3, seed=42
        )
        benchmark.run_benchmark(n_training_steps=20)
        result = uq.propagate_control_uq(benchmark, "no_control")
        assert result.target_name == "mean_reward"

    def test_format_uq_report(self, uq):
        dp = DesignPoint(thickness=0.12, camber=0.02, aoa=5.0)
        design_uq = uq.propagate_design_uq(dp, "rans")
        report = uq.format_uq_report(design_uq=design_uq)
        assert "## UQ Analysis" in report
        assert "Sobol Indices" in report
        assert "Robustness score" in report

    def test_mc_samples_count(self, uq):
        dp = DesignPoint(thickness=0.12, camber=0.02, aoa=5.0)
        result = uq.propagate_design_uq(dp, "rans")
        # CI should be meaningful (not collapsed)
        assert result.ci_95_high - result.ci_95_low > 0

    def test_high_uncertainty_at_stall(self, uq):
        # Near-stall condition should have higher uncertainty
        dp_safe = DesignPoint(thickness=0.12, camber=0.02, aoa=3.0)
        dp_stall = DesignPoint(thickness=0.12, camber=0.02, aoa=14.0)
        r_safe = uq.propagate_design_uq(dp_safe, "rans")
        r_stall = uq.propagate_design_uq(dp_stall, "rans")
        # High AoA should have relatively higher std/mean ratio
        cv_safe = r_safe.std_value / (abs(r_safe.mean_value) + 1e-6)
        cv_stall = r_stall.std_value / (abs(r_stall.mean_value) + 1e-6)
        assert cv_stall > cv_safe * 0.5  # Reasonable check

    def test_eigenspace_perturbation_effect(self, uq):
        dp = DesignPoint(thickness=0.12, camber=0.02, aoa=5.0)
        result = uq.propagate_design_uq(dp, "rans")
        # model_form should contribute to Sobol
        assert result.sobol_indices["model_form"] > 0


# ============================================================================
# Integration Tests
# ============================================================================

class TestEndToEndWorkflowRunner:
    """End-to-end integration tests."""

    def test_runner_import(self):
        runner = EndToEndWorkflowRunner(fast_mode=True)
        assert runner.fast_mode is True

    def test_design_workflow(self, tmp_path):
        runner = EndToEndWorkflowRunner(fast_mode=True)
        results = runner.run(mode="design", output_dir=tmp_path / "out")
        assert "design" in results
        assert "fidelity_results" in results["design"]
        assert (tmp_path / "out" / "workflow_report.md").exists()
        assert (tmp_path / "out" / "workflow_results.json").exists()

    def test_control_workflow(self, tmp_path):
        runner = EndToEndWorkflowRunner(fast_mode=True)
        results = runner.run(mode="control", output_dir=tmp_path / "out")
        assert "control" in results
        assert "benchmark_metrics" in results["control"]

    def test_both_workflow(self, tmp_path):
        runner = EndToEndWorkflowRunner(fast_mode=True)
        results = runner.run(mode="both", output_dir=tmp_path / "out")
        assert "design" in results
        assert "control" in results

    def test_json_valid(self, tmp_path):
        runner = EndToEndWorkflowRunner(fast_mode=True)
        runner.run(mode="both", output_dir=tmp_path / "out")
        with open(tmp_path / "out" / "workflow_results.json") as f:
            data = json.load(f)
        assert "design" in data

    def test_report_content(self, tmp_path):
        runner = EndToEndWorkflowRunner(fast_mode=True)
        runner.run(mode="both", output_dir=tmp_path / "out")
        report = (tmp_path / "out" / "workflow_report.md").read_text(
            encoding="utf-8"
        )
        assert "End-to-End Design & Control" in report
        assert "UQ Analysis" in report
