"""
Tests for Cross-Case Generalization Study
============================================
Validates LOO splits, fold evaluation, ranking, reporting, and the
full end-to-end pipeline.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ml_augmentation.cross_case_generalization import (
    BenchmarkCase,
    MLArchitecture,
    LOOSplit,
    FoldResult,
    GeneralizationStudy,
    GeneralizationReport,
    BENCHMARK_CASES,
    generate_synthetic_case_data,
    create_baseline_architectures,
)


# =========================================================================
# BenchmarkCase
# =========================================================================
class TestBenchmarkCase:
    """Test case descriptors."""

    def test_creation(self):
        case = BenchmarkCase("wall_hump", "NASA Hump", "smooth_body", Re=9.36e5)
        assert case.name == "wall_hump"
        assert case.Re == 9.36e5

    def test_default_cases_exist(self):
        assert len(BENCHMARK_CASES) == 5
        names = [c.name for c in BENCHMARK_CASES]
        assert "wall_hump" in names
        assert "periodic_hill" in names


# =========================================================================
# LOOSplit
# =========================================================================
class TestLOOSplit:
    """Test split generation."""

    def test_creation(self):
        split = LOOSplit("wall_hump", ["bfs", "swbli", "phill"], fold_index=0)
        assert split.held_out_case == "wall_hump"
        assert len(split.train_cases) == 3


# =========================================================================
# FoldResult
# =========================================================================
class TestFoldResult:
    """Test fold result dataclass."""

    def test_creation(self):
        r = FoldResult("MLP", "wall_hump", fold_index=0, Cf_RMSE=0.01)
        assert r.architecture == "MLP"
        assert r.Cf_RMSE == 0.01

    def test_to_dict(self):
        r = FoldResult("MLP", "wall_hump", Cf_RMSE=0.01, R2=0.95)
        d = r.to_dict()
        assert d["architecture"] == "MLP"
        assert d["Cf_RMSE"] == 0.01


# =========================================================================
# GeneralizationStudy
# =========================================================================
class TestGeneralizationStudy:
    """Test full LOO study engine."""

    @pytest.fixture
    def study_with_data(self):
        study = GeneralizationStudy(cases=BENCHMARK_CASES[:3])
        data = generate_synthetic_case_data(
            cases=BENCHMARK_CASES[:3], n_points=50, seed=42
        )
        for name, d in data.items():
            study.register_case_data(name, d["X"], d["Y"])
        return study

    def test_loo_splits(self, study_with_data):
        splits = study_with_data.generate_loo_splits()
        assert len(splits) == 3
        # Each split holds out one case
        held_out = [s.held_out_case for s in splits]
        assert len(set(held_out)) == 3

    def test_each_split_has_correct_train_count(self, study_with_data):
        splits = study_with_data.generate_loo_splits()
        for split in splits:
            assert len(split.train_cases) == 2

    def test_run_requires_architectures(self, study_with_data):
        with pytest.raises(RuntimeError, match="No architectures"):
            study_with_data.run()

    def test_run_requires_data(self):
        study = GeneralizationStudy()
        study.register_architecture(
            MLArchitecture("test", fit_fn=lambda X, Y: None,
                           predict_fn=lambda m, X: np.zeros((X.shape[0], 2)))
        )
        with pytest.raises(RuntimeError, match="No case data"):
            study.run()

    def test_full_run(self, study_with_data):
        archs = create_baseline_architectures()
        for a in archs:
            study_with_data.register_architecture(a)
        results = study_with_data.run()
        # 3 cases × 3 architectures = 9 results
        assert len(results) == 9
        assert all(r.status == "DONE" for r in results)

    def test_results_have_metrics(self, study_with_data):
        archs = create_baseline_architectures()
        for a in archs:
            study_with_data.register_architecture(a)
        results = study_with_data.run()
        for r in results:
            assert np.isfinite(r.Cf_RMSE)
            assert np.isfinite(r.R2)
            assert np.isfinite(r.MAE)


# =========================================================================
# GeneralizationReport
# =========================================================================
class TestGeneralizationReport:
    """Test report generation and ranking."""

    @pytest.fixture
    def report(self):
        study = GeneralizationStudy(cases=BENCHMARK_CASES[:3])
        data = generate_synthetic_case_data(
            cases=BENCHMARK_CASES[:3], n_points=50
        )
        for name, d in data.items():
            study.register_case_data(name, d["X"], d["Y"])
        for a in create_baseline_architectures():
            study.register_architecture(a)
        results = study.run()
        return GeneralizationReport(results)

    def test_rank_architectures(self, report):
        ranking = report.rank_architectures("Cf_RMSE")
        assert len(ranking) == 3
        assert ranking[0]["rank"] == 1
        # All have valid means
        assert all(np.isfinite(r["mean"]) for r in ranking)

    def test_per_case_table(self, report):
        table = report.per_case_table()
        assert len(table) == 3  # 3 cases
        for case_name, arch_results in table.items():
            assert len(arch_results) == 3  # 3 architectures

    def test_transferability_analysis(self, report):
        analysis = report.identify_transferable_features()
        assert len(analysis) >= 1
        for case, info in analysis.items():
            assert "mean_Cf_RMSE" in info
            assert np.isfinite(info["mean_Cf_RMSE"])

    def test_markdown_report(self, report):
        md = report.generate_markdown_report()
        assert "Cross-Case Generalization Study" in md
        assert "Architecture Ranking" in md
        assert "Transferability" in md

    def test_json_output(self, report):
        j = report.to_json()
        import json
        data = json.loads(j)
        assert len(data) == 9  # 3 cases × 3 architectures

    def test_summary_string(self, report):
        s = report.summary()
        assert "Cross-Case" in s
        assert "LOO Folds" in s


# =========================================================================
# Baseline Architectures
# =========================================================================
class TestBaselineArchitectures:
    """Test default architecture creation."""

    def test_creates_three_architectures(self):
        archs = create_baseline_architectures()
        assert len(archs) == 3
        names = [a.name for a in archs]
        assert "MLP" in names
        assert "LinearRegression" in names
        assert "MeanPredictor" in names

    def test_mlp_fit_predict(self):
        archs = create_baseline_architectures()
        mlp = next(a for a in archs if a.name == "MLP")
        X = np.random.randn(50, 5)
        Y = np.random.randn(50, 2)
        model = mlp.fit_fn(X, Y)
        pred = mlp.predict_fn(model, X[:10])
        assert pred.shape == (10, 2)
        assert np.all(np.isfinite(pred))

    def test_linear_fit_predict(self):
        archs = create_baseline_architectures()
        lr = next(a for a in archs if a.name == "LinearRegression")
        X = np.random.randn(50, 5)
        Y = np.random.randn(50, 2)
        model = lr.fit_fn(X, Y)
        pred = lr.predict_fn(model, X[:10])
        assert pred.shape == (10, 2)

    def test_mean_predictor(self):
        archs = create_baseline_architectures()
        mp = next(a for a in archs if a.name == "MeanPredictor")
        X = np.random.randn(50, 5)
        Y = np.ones((50, 2)) * 3.0
        model = mp.fit_fn(X, Y)
        pred = mp.predict_fn(model, X[:10])
        np.testing.assert_allclose(pred, 3.0, atol=1e-10)


# =========================================================================
# Synthetic Data Generation
# =========================================================================
class TestDataGeneration:
    """Test synthetic case data generator."""

    def test_generates_all_cases(self):
        data = generate_synthetic_case_data()
        assert len(data) == 5
        for case in BENCHMARK_CASES:
            assert case.name in data

    def test_shapes(self):
        data = generate_synthetic_case_data(n_points=60, n_features=4, n_targets=2)
        for name, d in data.items():
            assert d["X"].shape == (60, 4)
            assert d["Y"].shape == (60, 2)

    def test_finite_values(self):
        data = generate_synthetic_case_data()
        for name, d in data.items():
            assert np.all(np.isfinite(d["X"]))
            assert np.all(np.isfinite(d["Y"]))

    def test_cases_differ(self):
        data = generate_synthetic_case_data(n_points=50)
        names = list(data.keys())
        Y0 = data[names[0]]["Y"]
        Y1 = data[names[1]]["Y"]
        assert not np.allclose(Y0, Y1)


# =========================================================================
# Separation Detection
# =========================================================================
class TestSeparationDetection:
    """Test separation/reattachment point finding."""

    def test_finds_separation(self):
        Cf = np.array([0.004, 0.003, 0.001, -0.001, -0.002, 0.001, 0.003])
        x_sep = GeneralizationStudy._find_separation(Cf)
        assert x_sep is not None
        assert 0 < x_sep < 1

    def test_finds_reattachment(self):
        Cf = np.array([0.004, 0.003, 0.001, -0.001, -0.002, 0.001, 0.003])
        x_reat = GeneralizationStudy._find_reattachment(Cf)
        assert x_reat is not None
        assert 0 < x_reat < 1

    def test_no_separation_returns_none(self):
        Cf = np.array([0.004, 0.003, 0.002, 0.001])
        assert GeneralizationStudy._find_separation(Cf) is None


# =========================================================================
# Advanced Architectures
# =========================================================================
from scripts.ml_augmentation.cross_case_generalization import create_advanced_architectures

class TestAdvancedArchitectures:
    """Test wrappers for advanced architectures (TBNN, GNN, Zonal)."""

    def test_creates_three_architectures(self):
        archs = create_advanced_architectures()
        assert len(archs) == 4
        names = [a.name for a in archs]
        assert any("TBNN" in n for n in names)
        assert any("GNN" in n for n in names)
        assert any("Zonal" in n for n in names)

    def test_arch_fit_predict(self):
        archs = create_advanced_architectures(seed=42)
        X = np.random.randn(50, 5)
        Y = np.random.randn(50, 2)
        for arch in archs:
            model = arch.fit_fn(X, Y)
            assert model is not None
            pred = arch.predict_fn(model, X[:10])
            if arch.name == "DoMINO_MoE_25_11":
                assert pred.shape == (10, 1)
            else:
                assert pred.shape == (10, 2)
            assert np.all(np.isfinite(pred))


# =========================================================================
# Zonal vs Global Reporting
# =========================================================================
class TestZonalVsGlobalReporting:
    """Test explicit reporting of Zonal vs Global architecture performance."""

    def test_zonal_vs_global_extraction(self):
        # Create a dummy study and manually add populated FoldResults
        study = GeneralizationStudy()
        study.results = [
            FoldResult("TBNN_Global", "wall_hump", status="DONE", Cf_RMSE=0.04),
            FoldResult("GNN_FIML_Global", "wall_hump", status="DONE", Cf_RMSE=0.035),
            FoldResult("Spatial_Blended_Zonal", "wall_hump", status="DONE", Cf_RMSE=0.02),
        ]
        
        report = GeneralizationReport(study.results)
        zvg_text = report._zonal_vs_global_comparison()
        
        assert "Zonal vs Global Architecture Performance" in zvg_text
        assert "Mean Zonal Cf RMSE" in zvg_text
        assert "Mean Global Cf RMSE" in zvg_text
        assert "OUTPERFORM" in zvg_text
        assert "error reduction" in zvg_text

    def test_zonal_vs_global_markdown(self):
        study = GeneralizationStudy()
        study.results = [
            FoldResult("TBNN_Global", "wall_hump", status="DONE", Cf_RMSE=0.05),
            FoldResult("Spatial_Blended_Zonal", "wall_hump", status="DONE", Cf_RMSE=0.01),
        ]
        
        report = GeneralizationReport(study.results)
        md = report.generate_markdown_report()
        assert "Zonal vs Global Architecture Performance" in md


# =========================================================================
# Physics Stress Testing
# =========================================================================
from scripts.ml_augmentation.cross_case_generalization import PhysicsStressTest

class TestPhysicsStressTest:
    """Test out-of-distribution physical scenario evaluations."""

    def test_add_and_run_stress_test(self):
        study = GeneralizationStudy()
        stress = PhysicsStressTest(study)
        
        X_test = np.random.randn(20, 5)
        Y_test = np.random.randn(20, 2)
        
        stress.add_stress_case("High_Re", "Re=10M (10x higher)", X_test, Y_test)
        
        arch = create_advanced_architectures()[0]
        model = arch.fit_fn(np.random.randn(50, 5), np.random.randn(50, 2))
        
        res = stress.run_stress_test(arch, model)
        assert res["architecture"] == arch.name
        assert "High_Re" in res["cases"]
        assert res["cases"]["High_Re"]["status"] == "DONE"
        assert "extrapolation_failure" in res["cases"]["High_Re"]
        
    def test_generate_stress_report(self):
        study = GeneralizationStudy()
        stress = PhysicsStressTest(study)
        
        X_test = np.ones((10, 5))
        Y_test = np.ones((10, 2)) # Error will be high due to random model
        
        stress.add_stress_case("Compressible", "M=0.8", X_test, Y_test)
        
        arch = create_advanced_architectures()[0]
        model = arch.fit_fn(np.zeros((50, 5)), np.zeros((50, 2)))
        stress.run_stress_test(arch, model)
        
        report = stress.generate_stress_report()
        assert "Physics Stress Testing (Extrapolation Constraints)" in report
        assert "Compressible" in report
        assert "M=0.8" in report
        assert "extrapolation_failure" not in report # It should use YES/NO in the markdown
        assert "YES" in report or "NO" in report

