#!/usr/bin/env python3
"""
Test Suite: Curated ML-Turbulence Benchmark
============================================
Comprehensive tests for the benchmark case registry, target definitions,
metrics contract API, and end-to-end orchestrator.

Run: pytest tests/test_curated_benchmark_suite.py -v --tb=short
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# Registry Tests
# ============================================================================

class TestCuratedCaseRegistry:
    """Test benchmark_case_registry module."""

    def test_import(self):
        from scripts.ml_augmentation.benchmark_case_registry import (
            CURATED_CASE_REGISTRY,
            CuratedCase,
        )
        assert CURATED_CASE_REGISTRY is not None
        assert len(CURATED_CASE_REGISTRY) >= 5

    def test_registry_keys(self):
        from scripts.ml_augmentation.benchmark_case_registry import (
            CURATED_CASE_REGISTRY,
        )
        expected = {
            "periodic_hill", "nasa_hump", "backward_facing_step",
            "boeing_gaussian_bump", "juncture_flow",
        }
        assert expected.issubset(set(CURATED_CASE_REGISTRY.keys()))

    def test_curated_case_fields(self):
        from scripts.ml_augmentation.benchmark_case_registry import (
            CURATED_CASE_REGISTRY,
        )
        for key, cc in CURATED_CASE_REGISTRY.items():
            assert cc.case_key == key
            assert isinstance(cc.curated_geometry, str) and cc.curated_geometry
            assert cc.reynolds_number > 0
            assert isinstance(cc.reference_source, str) and cc.reference_source
            assert isinstance(cc.reference_fields, list) and len(cc.reference_fields) > 0
            assert isinstance(cc.curated_rans_models, list)

    def test_get_matched_cases(self):
        from scripts.ml_augmentation.benchmark_case_registry import (
            get_matched_cases,
        )
        cases = get_matched_cases()
        assert isinstance(cases, dict)
        assert len(cases) >= 5

    def test_get_curated_case(self):
        from scripts.ml_augmentation.benchmark_case_registry import (
            get_curated_case,
        )
        cc = get_curated_case("periodic_hill")
        assert cc is not None
        assert cc.reynolds_number == 10_595
        assert get_curated_case("nonexistent") is None

    def test_list_matched_case_keys(self):
        from scripts.ml_augmentation.benchmark_case_registry import (
            list_matched_case_keys,
        )
        keys = list_matched_case_keys()
        assert "periodic_hill" in keys
        assert "backward_facing_step" in keys

    def test_get_field_intersection(self):
        from scripts.ml_augmentation.benchmark_case_registry import (
            get_field_intersection,
        )
        common = get_field_intersection()
        assert isinstance(common, list)
        # Ux and Uy should be common across most cases
        assert "Ux" in common
        assert "Uy" in common

    def test_get_field_intersection_subset(self):
        from scripts.ml_augmentation.benchmark_case_registry import (
            get_field_intersection,
        )
        common = get_field_intersection(["periodic_hill", "backward_facing_step"])
        assert "Ux" in common
        assert "Uy" in common

    def test_export_curated_structure(self, tmp_path):
        from scripts.ml_augmentation.benchmark_case_registry import (
            export_curated_structure,
        )
        out = export_curated_structure(
            output_dir=tmp_path / "export",
            case_keys=["periodic_hill"],
            synthetic=True,
        )
        assert out.exists()
        assert (out / "metadata.json").exists()
        assert (out / "periodic_hill" / "dns_reference" / "fields.npz").exists()

        # Check metadata
        with open(out / "metadata.json") as f:
            meta = json.load(f)
        assert "periodic_hill" in meta["cases"]
        assert meta["cases"]["periodic_hill"]["Re"] == 10_595


# ============================================================================
# Targets Tests
# ============================================================================

class TestBenchmarkTargets:
    """Test benchmark_targets module."""

    def test_import(self):
        from scripts.ml_augmentation.benchmark_targets import (
            BENCHMARK_TASKS,
            BASELINE_ERROR_TABLE,
            BenchmarkTask,
        )
        assert BENCHMARK_TASKS is not None
        assert len(BENCHMARK_TASKS) >= 8

    def test_task_fields(self):
        from scripts.ml_augmentation.benchmark_targets import BENCHMARK_TASKS
        for tid, task in BENCHMARK_TASKS.items():
            assert task.task_id == tid
            assert isinstance(task.case_key, str) and task.case_key
            assert isinstance(task.description, str) and task.description
            assert isinstance(task.target_quantities, list)
            assert len(task.target_quantities) > 0
            assert task.metric_type in ("RMSE", "MAE", "relative_error", "topology")

    def test_baseline_errors_populated(self):
        from scripts.ml_augmentation.benchmark_targets import BENCHMARK_TASKS
        for task in BENCHMARK_TASKS.values():
            assert isinstance(task.baseline_errors, dict)
            # At least RANS baselines should exist
            if task.metric_type != "topology":
                assert "SA" in task.baseline_errors or "SST" in task.baseline_errors

    def test_baseline_error_table(self):
        from scripts.ml_augmentation.benchmark_targets import (
            BASELINE_ERROR_TABLE,
            ALL_MODEL_NAMES,
        )
        assert len(BASELINE_ERROR_TABLE) == len(ALL_MODEL_NAMES)
        for model in ALL_MODEL_NAMES:
            assert model in BASELINE_ERROR_TABLE

    def test_get_tasks_for_case(self):
        from scripts.ml_augmentation.benchmark_targets import get_tasks_for_case
        ph_tasks = get_tasks_for_case("periodic_hill")
        assert len(ph_tasks) >= 2
        assert all(t.case_key == "periodic_hill" for t in ph_tasks)

    def test_get_baseline_table(self):
        from scripts.ml_augmentation.benchmark_targets import get_baseline_table
        full = get_baseline_table()
        assert "SA" in full
        assert "TBNN" in full

        sa_row = get_baseline_table("SA")
        assert isinstance(sa_row, dict)

    def test_get_model_ranking(self):
        from scripts.ml_augmentation.benchmark_targets import get_model_ranking
        ranking = get_model_ranking("PH_RS_field")
        assert len(ranking) > 0
        # Should be sorted ascending
        errors = [e for _, e in ranking]
        assert errors == sorted(errors)

    def test_format_baseline_table_markdown(self):
        from scripts.ml_augmentation.benchmark_targets import (
            format_baseline_table_markdown,
        )
        md = format_baseline_table_markdown()
        assert "| Model |" in md
        assert "SA" in md
        assert "TBNN" in md


# ============================================================================
# Metrics Contract Tests
# ============================================================================

class TestMetricsContract:
    """Test CuratedBenchmarkEvaluator and BenchmarkMetricsContract."""

    @pytest.fixture
    def target_names(self):
        return ["Ux", "Uy", "k_dns", "uu_dns", "uv_dns", "vv_dns"]

    @pytest.fixture
    def sample_data(self, target_names):
        np.random.seed(42)
        n = 200
        targets = np.random.randn(n, len(target_names)).astype(np.float32)
        predictions_good = targets + np.random.randn(n, len(target_names)) * 0.01
        predictions_bad = targets + np.random.randn(n, len(target_names)) * 0.5
        features = np.random.randn(n, 10).astype(np.float32)
        return features, targets, predictions_good, predictions_bad

    def test_evaluator_rmse(self, target_names, sample_data):
        from scripts.ml_augmentation.curated_benchmark_evaluator import (
            CuratedBenchmarkEvaluator,
        )
        _, targets, pred_good, pred_bad = sample_data
        ev = CuratedBenchmarkEvaluator(target_names)

        result_good = ev.evaluate_predictions("good", pred_good, targets)
        result_bad = ev.evaluate_predictions("bad", pred_bad, targets)

        # Good model should have lower RMSE
        assert result_good["overall"]["U_mag_RMS"] < result_bad["overall"]["U_mag_RMS"]

    def test_evaluator_mae(self, target_names, sample_data):
        from scripts.ml_augmentation.curated_benchmark_evaluator import (
            CuratedBenchmarkEvaluator,
        )
        _, targets, pred_good, _ = sample_data
        ev = CuratedBenchmarkEvaluator(target_names)

        result = ev.evaluate_predictions("test", pred_good, targets)
        assert "U_mag_MAE" in result["overall"]
        assert "kMAE" in result["overall"]
        assert result["overall"]["U_mag_MAE"] >= 0

    def test_evaluator_realizability(self, target_names, sample_data):
        from scripts.ml_augmentation.curated_benchmark_evaluator import (
            CuratedBenchmarkEvaluator,
        )
        _, targets, _, _ = sample_data
        ev = CuratedBenchmarkEvaluator(target_names)

        # All-positive normal stresses → low violation rate
        good_pred = np.abs(targets.copy())
        result = ev.evaluate_predictions("realizable", good_pred, targets)
        assert "realizability_violation" in result["overall"]

    def test_evaluator_per_case(self, target_names, sample_data):
        from scripts.ml_augmentation.curated_benchmark_evaluator import (
            CuratedBenchmarkEvaluator,
        )
        _, targets, pred_good, _ = sample_data
        ev = CuratedBenchmarkEvaluator(target_names)
        cases = ["case_A"] * 100 + ["case_B"] * 100

        result = ev.evaluate_predictions("test", pred_good, targets, case_labels=cases)
        assert "per_case" in result
        assert "case_A" in result["per_case"]
        assert "case_B" in result["per_case"]

    def test_evaluator_shape_mismatch(self, target_names):
        from scripts.ml_augmentation.curated_benchmark_evaluator import (
            CuratedBenchmarkEvaluator,
        )
        ev = CuratedBenchmarkEvaluator(target_names)
        with pytest.raises(ValueError, match="shape"):
            ev.evaluate_predictions(
                "bad", np.zeros((10, 3)), np.zeros((10, 6))
            )

    def test_evaluator_markdown(self, target_names, sample_data):
        from scripts.ml_augmentation.curated_benchmark_evaluator import (
            CuratedBenchmarkEvaluator,
        )
        _, targets, pred_good, pred_bad = sample_data
        ev = CuratedBenchmarkEvaluator(target_names)

        evals = [
            ev.evaluate_predictions("good", pred_good, targets),
            ev.evaluate_predictions("bad", pred_bad, targets),
        ]
        md = ev.format_as_markdown_table(evals)
        assert "| Model |" in md
        assert "good" in md
        assert "bad" in md

    # ---- BenchmarkMetricsContract tests ----

    def test_contract_register(self, target_names):
        from scripts.ml_augmentation.curated_benchmark_evaluator import (
            BenchmarkMetricsContract,
        )
        contract = BenchmarkMetricsContract(target_names)
        contract.register_model("test_model", lambda x: np.zeros((len(x), 6)))
        assert "test_model" in contract.list_models()

    def test_contract_evaluate_all(self, target_names, sample_data):
        from scripts.ml_augmentation.curated_benchmark_evaluator import (
            BenchmarkMetricsContract,
        )
        features, targets, _, _ = sample_data
        contract = BenchmarkMetricsContract(target_names)

        contract.register_model("zero_model", lambda x: np.zeros((len(x), 6)))
        contract.register_model("noise_model", lambda x: np.random.randn(len(x), 6))

        results = contract.evaluate_all(features, targets)
        assert len(results) == 2
        assert results[0]["model"] == "zero_model"
        assert results[1]["model"] == "noise_model"

    def test_contract_evaluate_direct(self, target_names, sample_data):
        from scripts.ml_augmentation.curated_benchmark_evaluator import (
            BenchmarkMetricsContract,
        )
        _, targets, pred_good, _ = sample_data
        contract = BenchmarkMetricsContract(target_names)

        ev = contract.evaluate_predictions_direct("direct", pred_good, targets)
        assert ev["model"] == "direct"
        assert "overall" in ev
        assert len(contract.get_results()) == 1

    def test_contract_export_json(self, target_names, sample_data, tmp_path):
        from scripts.ml_augmentation.curated_benchmark_evaluator import (
            BenchmarkMetricsContract,
        )
        _, targets, pred_good, _ = sample_data
        contract = BenchmarkMetricsContract(target_names)
        contract.evaluate_predictions_direct("test", pred_good, targets)

        path = contract.export_results(tmp_path, fmt="json")
        assert path.exists()
        with open(path) as f:
            data = json.load(f)
        assert len(data) == 1
        assert data[0]["model"] == "test"

    def test_contract_export_csv(self, target_names, sample_data, tmp_path):
        from scripts.ml_augmentation.curated_benchmark_evaluator import (
            BenchmarkMetricsContract,
        )
        _, targets, pred_good, _ = sample_data
        contract = BenchmarkMetricsContract(target_names)
        contract.evaluate_predictions_direct("test", pred_good, targets)

        path = contract.export_results(tmp_path, fmt="csv")
        assert path.exists()
        content = path.read_text()
        assert "model" in content
        assert "test" in content

    def test_contract_export_markdown(self, target_names, sample_data, tmp_path):
        from scripts.ml_augmentation.curated_benchmark_evaluator import (
            BenchmarkMetricsContract,
        )
        _, targets, pred_good, _ = sample_data
        contract = BenchmarkMetricsContract(target_names)
        contract.evaluate_predictions_direct("test", pred_good, targets)

        path = contract.export_results(tmp_path, fmt="markdown")
        assert path.exists()
        content = path.read_text()
        assert "# ML-Turbulence Benchmark Results" in content

    def test_contract_export_all(self, target_names, sample_data, tmp_path):
        from scripts.ml_augmentation.curated_benchmark_evaluator import (
            BenchmarkMetricsContract,
        )
        _, targets, pred_good, _ = sample_data
        contract = BenchmarkMetricsContract(target_names)
        contract.evaluate_predictions_direct("test", pred_good, targets)

        contract.export_results(tmp_path, fmt="all")
        assert (tmp_path / "benchmark_results.json").exists()
        assert (tmp_path / "benchmark_results.csv").exists()
        assert (tmp_path / "benchmark_results.md").exists()


# ============================================================================
# Separation Metrics Tests
# ============================================================================

class TestSeparationMetrics:
    """Test Cf-based separation detection in the evaluator."""

    def test_cf_separation_detection(self):
        from scripts.ml_augmentation.curated_benchmark_evaluator import (
            CuratedBenchmarkEvaluator,
        )
        target_names = ["Cf"]
        ev = CuratedBenchmarkEvaluator(target_names)

        # Create Cf that goes negative (separated) and back positive (reattached)
        n = 100
        x = np.linspace(0, 1, n)
        cf_target = np.where((x > 0.3) & (x < 0.7), -0.01, 0.01)
        cf_pred = np.where((x > 0.32) & (x < 0.68), -0.01, 0.01)

        targets = cf_target.reshape(-1, 1)
        predictions = cf_pred.reshape(-1, 1)

        result = ev.evaluate_predictions(
            "test", predictions, targets, x_coords=x
        )
        assert "Cf_RMSE" in result["overall"]
        assert "Cf_MAE" in result["overall"]
        assert "sep_point_error" in result["overall"]
        assert "reat_point_error" in result["overall"]
        assert "bubble_length_error" in result["overall"]


# ============================================================================
# End-to-End Integration Tests
# ============================================================================

class TestEndToEnd:
    """Integration tests for the full benchmark pipeline."""

    def test_import_orchestrator(self):
        from scripts.run_curated_benchmark import run_benchmark
        assert callable(run_benchmark)

    def test_synthetic_dataset_build(self):
        from scripts.run_curated_benchmark import _build_synthetic_dataset
        X, Y, cases, models, names = _build_synthetic_dataset(
            ["periodic_hill", "backward_facing_step"], n_points=50
        )
        assert X.shape[0] > 0
        assert Y.shape[0] == X.shape[0]
        assert len(cases) == X.shape[0]
        assert len(names) == 6

    def test_full_pipeline_fast(self, tmp_path, monkeypatch):
        """Run the full benchmark in fast mode with synthetic data."""
        import scripts.run_curated_benchmark as runner

        # Redirect output to tmp
        monkeypatch.setattr(
            runner, "PROJECT_ROOT", tmp_path
        )
        (tmp_path / "results").mkdir(exist_ok=True)

        runner.run_benchmark(
            fast_mode=True,
            case_keys=["periodic_hill"],
            export_structure=False,
        )

        # Check output was generated
        results_dir = tmp_path / "results" / "curated_benchmark"
        assert results_dir.exists()
        assert (results_dir / "benchmark_results.json").exists()
        assert (results_dir / "benchmark_results.csv").exists()
        assert (results_dir / "benchmark_report.md").exists()


# ============================================================================
# Physics-Constraint Tests
# ============================================================================

class TestPhysicsConstraints:
    """Test realizability constraints."""

    def test_realizable_predictions(self):
        from scripts.ml_augmentation.curated_benchmark_evaluator import (
            CuratedBenchmarkEvaluator,
        )
        ev = CuratedBenchmarkEvaluator(["uu_dns", "uv_dns", "vv_dns"])

        # Perfectly realizable: positive diagonals, Cauchy-Schwarz satisfied
        n = 100
        predictions = np.column_stack([
            np.ones(n) * 1.0,      # uu
            np.zeros(n),            # uv = 0 (trivially satisfies C-S)
            np.ones(n) * 1.0,      # vv
        ])
        targets = predictions.copy()

        result = ev.evaluate_predictions("real", predictions, targets)
        violation = result["overall"]["realizability_violation"]
        assert violation == 0.0

    def test_unrealizable_predictions(self):
        from scripts.ml_augmentation.curated_benchmark_evaluator import (
            CuratedBenchmarkEvaluator,
        )
        ev = CuratedBenchmarkEvaluator(["uu_dns", "uv_dns", "vv_dns"])

        # All negative normal stresses → unrealizable
        n = 100
        predictions = np.column_stack([
            -np.ones(n),    # uu < 0
            np.zeros(n),    # uv
            -np.ones(n),    # vv < 0
        ])
        targets = np.abs(predictions)

        result = ev.evaluate_predictions("unreal", predictions, targets)
        violation = result["overall"]["realizability_violation"]
        assert violation == 1.0  # 100% violation
