#!/usr/bin/env python3
"""
Tests for LOO Extrapolation Study
===================================
Validates the systematic leave-one-flow-out generalization experiment:
  - config, data generation, architecture wrappers
  - LOO experiment runner, failure-mode analysis
  - spatial UQ mapping, report generation
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ml_augmentation.loo_extrapolation_study import (
    ExtrapolationFoldResult,
    ExtrapolationReport,
    FailureModeAnalyzer,
    FlowFamily,
    GlobalMLPWrapper,
    LOOExperiment,
    LOOExperimentConfig,
    PGGNNWrapper,
    SpatialBlendWrapper,
    SpatialUQMapper,
    SyntheticFlowGenerator,
    TBNNWrapper,
    create_default_architectures,
    run_full_study,
)


# =========================================================================
# TestLOOExperimentConfig
# =========================================================================
class TestLOOExperimentConfig:
    """Tests for experiment configuration."""

    def test_default_config(self):
        cfg = LOOExperimentConfig()
        assert len(cfg.flow_families) == 6
        assert cfg.n_ensemble_members == 5
        assert cfg.n_features == 5

    def test_custom_config(self):
        cfg = LOOExperimentConfig(n_points_per_case=50, n_ensemble_members=3)
        assert cfg.n_points_per_case == 50
        assert cfg.n_ensemble_members == 3

    def test_flow_family_names(self):
        cfg = LOOExperimentConfig()
        names = {f.name for f in cfg.flow_families}
        assert "periodic_hill" in names
        assert "wall_hump" in names
        assert "bfs" in names
        assert "gaussian_bump" in names
        assert "beverli_hill" in names
        assert "swbli_low_mach" in names


# =========================================================================
# TestSyntheticFlowGenerator
# =========================================================================
class TestSyntheticFlowGenerator:
    """Tests for synthetic data generation."""

    def test_all_cases_generated(self):
        cfg = LOOExperimentConfig(n_points_per_case=50)
        gen = SyntheticFlowGenerator(cfg)
        data = gen.generate_all()
        assert len(data) == 6

    def test_shapes(self):
        cfg = LOOExperimentConfig(n_points_per_case=50, n_features=5, n_targets=6)
        gen = SyntheticFlowGenerator(cfg)
        data = gen.generate_all()
        for name, d in data.items():
            assert d["X"].shape == (50, 5), f"{name} X shape wrong"
            assert d["Y_dns"].shape == (50, 6), f"{name} Y_dns shape wrong"
            assert d["Y_rans"].shape == (50, 6), f"{name} Y_rans shape wrong"
            assert d["x_coord"].shape == (50,), f"{name} x_coord shape wrong"

    def test_case_features_differ(self):
        """Different flow families should have distinct feature distributions."""
        cfg = LOOExperimentConfig(n_points_per_case=200, seed=42)
        gen = SyntheticFlowGenerator(cfg)
        data = gen.generate_all()
        means = {name: np.mean(d["X"]) for name, d in data.items()}
        # Not all cases should have identical mean features
        vals = list(means.values())
        assert max(vals) - min(vals) > 0.01

    def test_feature_coverage_distances(self):
        cfg = LOOExperimentConfig(n_points_per_case=50)
        gen = SyntheticFlowGenerator(cfg)
        data = gen.generate_all()
        train = data["wall_hump"]["X"]
        test = data["periodic_hill"]["X"]
        dists = gen.compute_feature_coverage(train, test)
        assert dists.shape == (50,)
        assert np.all(dists >= 0)


# =========================================================================
# TestArchitectureWrappers
# =========================================================================
class TestArchitectureWrappers:
    """Tests for each architecture wrapper."""

    @pytest.fixture
    def train_data(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 5))
        Y = rng.standard_normal((100, 6))
        return X, Y

    @pytest.fixture
    def test_data(self):
        rng = np.random.default_rng(99)
        return rng.standard_normal((30, 5))

    def test_global_mlp(self, train_data, test_data):
        X, Y = train_data
        wrapper = GlobalMLPWrapper(n_ensemble=2, seed=42)
        wrapper.fit(X, Y)
        pred = wrapper.predict(test_data)
        assert pred.shape == (30, 6)
        mean, std = wrapper.predict_with_uncertainty(test_data)
        assert mean.shape == (30, 6)
        assert std.shape == (30, 6)
        assert np.all(std >= 0)

    def test_tbnn(self, train_data, test_data):
        X, Y = train_data
        wrapper = TBNNWrapper(n_ensemble=2, seed=42)
        wrapper.fit(X, Y)
        pred = wrapper.predict(test_data)
        assert pred.shape == (30, 6)
        # Check trace-free enforcement on first 3 components
        trace = pred[:, :3].sum(axis=1)
        np.testing.assert_allclose(trace, 0.0, atol=1e-10)

    def test_pg_gnn(self, train_data, test_data):
        X, Y = train_data
        wrapper = PGGNNWrapper(n_ensemble=2, seed=42)
        wrapper.fit(X, Y)
        pred = wrapper.predict(test_data)
        assert pred.shape == (30, 6)
        mean, std = wrapper.predict_with_uncertainty(test_data)
        assert np.all(std >= 0)

    def test_spatial_blend(self, train_data, test_data):
        X, Y = train_data
        wrapper = SpatialBlendWrapper(n_ensemble=2, seed=42)
        wrapper.fit(X, Y)
        pred = wrapper.predict(test_data)
        assert pred.shape == (30, 6)
        mean, std = wrapper.predict_with_uncertainty(test_data)
        assert np.all(std >= 0)

    def test_create_default(self):
        archs = create_default_architectures(n_ensemble=2)
        assert len(archs) == 4
        names = {a.name for a in archs}
        assert names == {"GlobalMLP", "TBNN", "PG-GNN", "SpatialBlend"}


# =========================================================================
# TestLOOExperiment
# =========================================================================
class TestLOOExperiment:
    """Tests for the LOO experiment runner."""

    def _make_mini_experiment(self):
        """Create a mini experiment with 2 cases and 2 architectures."""
        families = [
            FlowFamily("case_a", "Case A", "type_a", Re=1e5),
            FlowFamily("case_b", "Case B", "type_b", Re=2e5),
        ]
        cfg = LOOExperimentConfig(
            flow_families=families,
            n_points_per_case=40,
            n_ensemble_members=2,
            n_features=3,
            n_targets=2,
        )
        archs = [
            GlobalMLPWrapper(n_ensemble=2, seed=42),
            TBNNWrapper(n_ensemble=2, seed=42),
        ]
        return LOOExperiment(config=cfg, architectures=archs)

    def test_mini_loo_runs(self):
        exp = self._make_mini_experiment()
        results = exp.run()
        # 2 cases × 2 architectures = 4 results
        assert len(results) == 4
        assert all(r.status == "OK" for r in results)

    def test_metrics_are_finite(self):
        exp = self._make_mini_experiment()
        results = exp.run()
        for r in results:
            assert np.isfinite(r.rmse)
            assert np.isfinite(r.r_squared)
            assert np.isfinite(r.improvement_pct)
            assert r.n_train > 0
            assert r.n_test > 0

    def test_failure_analyses_populated(self):
        exp = self._make_mini_experiment()
        exp.run()
        assert len(exp.failure_analyses) == 2  # 2 held-out cases
        for case, archs in exp.failure_analyses.items():
            for arch_name, fm in archs.items():
                assert "counts" in fm
                assert "degradation_frac" in fm

    def test_spatial_maps_populated(self):
        exp = self._make_mini_experiment()
        exp.run()
        assert len(exp.spatial_maps) == 2
        for case, archs in exp.spatial_maps.items():
            for arch_name, sm in archs.items():
                assert "relative_error" in sm
                assert "epistemic_std" in sm


# =========================================================================
# TestFailureModeAnalyzer
# =========================================================================
class TestFailureModeAnalyzer:
    """Tests for failure mode classification."""

    def test_perfect_prediction(self):
        n = 50
        Y_dns = np.ones((n, 3))
        Y_ml = np.ones((n, 3))
        Y_rans = np.ones((n, 3)) * 0.5  # RANS is worse
        epi_std = np.ones((n, 3)) * 0.01

        fma = FailureModeAnalyzer()
        result = fma.analyze(Y_dns, Y_rans, Y_ml, epi_std)
        assert result["degradation_frac"] == 0.0
        assert result["counts"]["worse_than_rans"] == 0

    def test_full_degradation(self):
        n = 50
        rng = np.random.default_rng(42)
        Y_dns = rng.standard_normal((n, 3))
        Y_rans = Y_dns + 0.01  # RANS is close
        Y_ml = Y_dns + 10.0    # ML is far off
        epi_std = np.ones((n, 3)) * 0.01

        fma = FailureModeAnalyzer()
        result = fma.analyze(Y_dns, Y_rans, Y_ml, epi_std)
        assert result["degradation_frac"] > 0.9

    def test_failure_labels(self):
        n = 100
        rng = np.random.default_rng(42)
        Y_dns = rng.standard_normal((n, 2))
        Y_rans = Y_dns + 0.5
        Y_ml = Y_dns + rng.standard_normal((n, 2)) * 0.8
        epi_std = rng.uniform(0.01, 1.0, (n, 2))

        fma = FailureModeAnalyzer()
        result = fma.analyze(Y_dns, Y_rans, Y_ml, epi_std)
        assert len(result["labels"]) == n
        assert set(result["labels"]).issubset(
            {"ok", "confident_wrong", "high_unc_wrong", "high_unc_correct"}
        )


# =========================================================================
# TestSpatialUQMapper
# =========================================================================
class TestSpatialUQMapper:
    """Tests for spatial UQ mapping."""

    def test_map_shapes(self):
        n = 50
        mapper = SpatialUQMapper()
        maps = mapper.compute_maps(
            x_coord=np.linspace(0, 1, n),
            Y_dns=np.ones((n, 3)),
            Y_ml=np.ones((n, 3)) * 1.1,
            epistemic_std=np.ones((n, 3)) * 0.1,
            feature_distances=np.ones(n) * 0.5,
        )
        assert maps["x_coord"].shape == (n,)
        assert maps["relative_error"].shape == (n,)
        assert maps["epistemic_std"].shape == (n,)
        assert maps["feature_distance"].shape == (n,)

    def test_correlation_finite(self):
        n = 100
        rng = np.random.default_rng(42)
        mapper = SpatialUQMapper()
        maps = mapper.compute_maps(
            x_coord=np.linspace(0, 1, n),
            Y_dns=rng.standard_normal((n, 2)),
            Y_ml=rng.standard_normal((n, 2)),
            epistemic_std=np.abs(rng.standard_normal((n, 2))),
            feature_distances=rng.uniform(0, 1, n),
        )
        assert np.isfinite(maps["error_unc_correlation"])


# =========================================================================
# TestExtrapolationReport
# =========================================================================
class TestExtrapolationReport:
    """Tests for report generation."""

    @pytest.fixture
    def report(self):
        families = [
            FlowFamily("case_a", "A", "t1", Re=1e5),
            FlowFamily("case_b", "B", "t2", Re=2e5),
        ]
        cfg = LOOExperimentConfig(
            flow_families=families,
            n_points_per_case=40,
            n_ensemble_members=2,
            n_features=3,
            n_targets=2,
        )
        archs = [
            GlobalMLPWrapper(n_ensemble=2, seed=42),
            SpatialBlendWrapper(n_ensemble=2, seed=42),
        ]
        exp = LOOExperiment(config=cfg, architectures=archs)
        exp.run()
        return ExtrapolationReport(exp)

    def test_rankings_sorted(self, report):
        ranking = report.rank_architectures("rmse", ascending=True)
        assert len(ranking) == 2
        assert ranking[0]["mean"] <= ranking[1]["mean"]
        assert ranking[0]["rank"] == 1

    def test_per_case_table(self, report):
        table = report.per_case_table()
        assert len(table) == 2  # 2 cases
        for case, archs in table.items():
            for arch, metrics in archs.items():
                assert "rmse" in metrics
                assert np.isfinite(metrics["rmse"])

    def test_failure_mode_summary(self, report):
        fm = report.failure_mode_summary()
        assert len(fm) > 0

    def test_zonal_vs_global(self, report):
        comp = report.zonal_vs_global_comparison()
        assert np.isfinite(comp["global_mean_rmse"])
        assert np.isfinite(comp["zonal_mean_rmse"])

    def test_markdown_generation(self, report):
        md = report.generate_markdown()
        assert "# Systematic LOO Generalization" in md
        assert "Architecture Rankings" in md
        assert "Per-Case Results" in md
        assert "Failure Mode" in md

    def test_json_serializable(self, report):
        j = report.to_json()
        data = json.loads(j)
        assert "rankings" in data
        assert "results" in data

    def test_summary(self, report):
        s = report.summary()
        assert "LOO Extrapolation Study" in s
        assert "OK" in s


# =========================================================================
# TestRunFullStudy
# =========================================================================
class TestRunFullStudy:
    """Test the convenience runner."""

    def test_full_study_mini(self):
        """Run a minimal 2-case study end-to-end."""
        families = [
            FlowFamily("a", "A", "t1", Re=1e5),
            FlowFamily("b", "B", "t2", Re=2e5),
        ]
        cfg = LOOExperimentConfig(
            flow_families=families,
            n_points_per_case=30,
            n_ensemble_members=2,
            n_features=3,
            n_targets=2,
        )
        report = run_full_study(config=cfg)
        assert "LOO Extrapolation Study" in report.summary()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
