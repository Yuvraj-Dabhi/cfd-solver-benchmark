#!/usr/bin/env python3
"""
Tests for Operator-Learning & Temporal Case Studies
=====================================================
Validates DeepONet vs FNO comparison, ConvLSTM temporal surrogate,
unsteady BFS study, and design screening integration.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ml_augmentation.operator_temporal_case_studies import (
    ConvLSTMCell,
    DesignScreeningDemo,
    OperatorCaseConfig,
    OperatorComparisonResult,
    OperatorComparisonStudy,
    OperatorTemporalReport,
    SWBLIDataGenerator,
    TemporalStudyResult,
    TemporalSurrogate,
    UnsteadyBFSConfig,
    UnsteadyBFSDataGenerator,
    UnsteadyBFSStudy,
    run_all_case_studies,
)


# =========================================================================
# TestSWBLIDataGenerator
# =========================================================================
class TestSWBLIDataGenerator:
    """Tests for SWBLI data generation."""

    def test_shapes(self):
        cfg = OperatorCaseConfig("test", "T", n_train=30, n_test=10, n_sensors=20, n_query=40)
        gen = SWBLIDataGenerator(cfg)
        data = gen.generate()
        assert data["input_functions"].shape == (30, 20)
        assert data["query_coords"].shape == (40, 1)
        assert data["target_fields"].shape == (30, 2, 40)
        assert data["test_input"].shape == (10, 20)
        assert data["test_target"].shape == (10, 2, 40)

    def test_mach_in_range(self):
        cfg = OperatorCaseConfig("test", "T", mach_range=(2.0, 5.0), n_train=20, n_test=5)
        gen = SWBLIDataGenerator(cfg)
        data = gen.generate()
        assert np.all(data["mach_numbers"] >= 2.0)
        assert np.all(data["mach_numbers"] <= 5.0)


# =========================================================================
# TestOperatorComparisonStudy
# =========================================================================
class TestOperatorComparisonStudy:
    """Tests for DeepONet vs FNO comparison."""

    def test_mini_comparison(self):
        cfg = [OperatorCaseConfig(
            "mini", "Mini", n_train=30, n_test=10,
            n_sensors=20, n_query=30, seed=42,
        )]
        study = OperatorComparisonStudy(configs=cfg)
        results = study.run()
        assert len(results) == 1
        r = results[0]
        assert np.isfinite(r.deeponet_rmse)
        assert np.isfinite(r.fno_rmse)
        assert r.winner in ("DeepONet", "FNO")
        assert r.speedup_ratio > 0

    def test_two_cases(self):
        cfgs = [
            OperatorCaseConfig("a", "A", n_train=20, n_test=5, n_sensors=15, n_query=20),
            OperatorCaseConfig("b", "B", n_train=20, n_test=5, n_sensors=15, n_query=20),
        ]
        study = OperatorComparisonStudy(configs=cfgs)
        results = study.run()
        assert len(results) == 2


# =========================================================================
# TestConvLSTMCell
# =========================================================================
class TestConvLSTMCell:
    """Tests for ConvLSTM cell."""

    def test_output_shapes(self):
        cell = ConvLSTMCell(input_channels=2, hidden_channels=4, kernel_size=3)
        x = np.random.randn(2, 2, 20)      # batch=2, ch=2, spatial=20
        h = np.zeros((2, 4, 20))
        c = np.zeros((2, 4, 20))
        h_new, c_new = cell.forward(x, h, c)
        assert h_new.shape == (2, 4, 20)
        assert c_new.shape == (2, 4, 20)

    def test_cell_state_updates(self):
        cell = ConvLSTMCell(input_channels=1, hidden_channels=2, kernel_size=3)
        x = np.ones((1, 1, 10))
        h = np.zeros((1, 2, 10))
        c = np.zeros((1, 2, 10))
        h1, c1 = cell.forward(x, h, c)
        h2, c2 = cell.forward(x, h1, c1)
        # Cell state should evolve
        assert not np.allclose(c1, c2)

    def test_forget_gate_bias(self):
        cell = ConvLSTMCell(input_channels=1, hidden_channels=2)
        assert np.allclose(cell.b[1], 1.0)  # forget gate bias = 1


# =========================================================================
# TestTemporalSurrogate
# =========================================================================
class TestTemporalSurrogate:
    """Tests for temporal surrogate model."""

    def test_predict_shapes(self):
        model = TemporalSurrogate(input_channels=2, hidden_channels=4, output_channels=2)
        X = np.random.randn(10, 2, 2, 30)  # T=10, batch=2, ch=2, spatial=30
        Y = model.predict_sequence(X)
        assert Y.shape == (10, 2, 2, 30)

    def test_output_finite(self):
        model = TemporalSurrogate(input_channels=2, hidden_channels=4, output_channels=2)
        X = np.random.randn(5, 1, 2, 20)
        Y = model.predict_sequence(X)
        assert np.all(np.isfinite(Y))


# =========================================================================
# TestUnsteadyBFSStudy
# =========================================================================
class TestUnsteadyBFSStudy:
    """Tests for unsteady BFS data gen and study."""

    def test_data_generation(self):
        cfg = UnsteadyBFSConfig(n_spatial=30, n_timesteps=10)
        gen = UnsteadyBFSDataGenerator(cfg)
        data = gen.generate()
        assert data["sequence"].shape == (10, 1, 2, 30)
        assert data["x_coord"].shape == (30,)
        assert data["time"].shape == (10,)

    def test_study_runs(self):
        cfg = UnsteadyBFSConfig(n_spatial=20, n_timesteps=15)
        study = UnsteadyBFSStudy(config=cfg)
        result = study.run()
        assert np.isfinite(result.temporal_rmse)
        assert np.isfinite(result.phase_error_rad)
        assert result.n_timesteps == 14  # T-1


# =========================================================================
# TestDesignScreeningDemo
# =========================================================================
class TestDesignScreeningDemo:
    """Tests for design screening integration."""

    def test_screening_runs(self):
        demo = DesignScreeningDemo(n_design_points=20, seed=42)
        result = demo.run()
        assert result["n_design_points"] == 20
        assert result["speedup_factor"] > 0
        assert np.isfinite(result["surrogate_rmse"])
        assert np.isfinite(result["surrogate_r2"])

    def test_uncertainty_finite(self):
        demo = DesignScreeningDemo(n_design_points=10)
        result = demo.run()
        assert np.isfinite(result["surrogate_uncertainty_pct"])


# =========================================================================
# TestOperatorTemporalReport
# =========================================================================
class TestOperatorTemporalReport:
    """Tests for report generation."""

    @pytest.fixture
    def report(self):
        op_results = [
            OperatorComparisonResult("test_case", deeponet_rmse=0.05, fno_rmse=0.04, winner="FNO"),
        ]
        temp_result = TemporalStudyResult(temporal_rmse=0.01, phase_error_rad=0.1, n_timesteps=20, n_spatial=30)
        screen_result = {"n_design_points": 50, "rans_time_s": 1.0, "surrogate_time_s": 0.01,
                         "speedup_factor": 100.0, "surrogate_rmse": 0.02, "surrogate_r2": 0.95,
                         "surrogate_uncertainty_pct": 5.0}
        return OperatorTemporalReport(op_results, temp_result, screen_result)

    def test_markdown(self, report):
        md = report.generate_markdown()
        assert "DeepONet vs FNO" in md
        assert "Temporal Surrogate" in md
        assert "Design Screening" in md

    def test_json_serializable(self, report):
        j = report.to_json()
        data = json.loads(j)
        assert "operator_comparison" in data
        assert "temporal_study" in data
        assert "design_screening" in data

    def test_summary(self, report):
        s = report.summary()
        assert "Operator/Temporal Studies" in s


# =========================================================================
# TestRunAllCaseStudies
# =========================================================================
class TestRunAllCaseStudies:
    """Test the convenience runner."""

    def test_mini_all(self):
        report = run_all_case_studies()
        assert "Operator/Temporal Studies" in report.summary()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
