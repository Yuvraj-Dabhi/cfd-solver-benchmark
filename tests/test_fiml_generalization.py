"""
Tests for FIML Cross-Geometry Generalization (Gap 1)
=====================================================
Validates the multi-geometry FIML pipeline: new BFS/curved-bump generators,
leave-one-geometry-out protocol, and JSON summary output.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ml_augmentation.fiml_generalization_test import (
    generate_periodic_hill_case,
    generate_wall_hump_case,
    generate_bfs_case,
    generate_curved_bump_case,
    LeaveOneGeometryOut,
    get_generalization_summary,
    run_generalization_test,
)
from scripts.ml_augmentation.fiml_nn_embedding import EmbeddingConfig


class TestNewGeometryGenerators:
    """Verify BFS and curved bump synthetic data generators."""

    def test_bfs_case_shapes(self):
        case = generate_bfs_case(n_points=100)
        assert case.name == "backward_facing_step"
        assert case.features.shape == (100, 5)
        assert case.beta_target.shape == (100,)
        assert case.x_coords.shape == (100,)
        assert case.y_coords.shape == (100,)

    def test_bfs_case_beta_range(self):
        case = generate_bfs_case(n_points=500)
        assert np.all(case.beta_target >= 0.5)
        assert np.all(case.beta_target <= 3.0)

    def test_bfs_case_metadata(self):
        case = generate_bfs_case()
        assert case.metadata["geometry"] == "bfs"
        assert case.metadata["Re_H"] == 36000
        assert case.metadata["x_sep_exp"] == 0.0
        assert case.metadata["x_reat_exp"] == 6.28

    def test_curved_bump_case_shapes(self):
        case = generate_curved_bump_case(n_points=150)
        assert case.name == "curved_bump"
        assert case.features.shape == (150, 5)
        assert case.beta_target.shape == (150,)
        assert case.x_coords.shape == (150,)

    def test_curved_bump_case_beta_range(self):
        case = generate_curved_bump_case(n_points=500)
        assert np.all(case.beta_target >= 0.5)
        assert np.all(case.beta_target <= 3.0)

    def test_curved_bump_case_metadata(self):
        case = generate_curved_bump_case()
        assert case.metadata["geometry"] == "curved_bump"
        assert case.metadata["Re_L"] == 2000000
        assert case.metadata["x_sep_exp"] == 0.72
        assert case.metadata["x_reat_exp"] == 0.93

    def test_different_seeds_different_data(self):
        c1 = generate_bfs_case(n_points=50, seed=1)
        c2 = generate_bfs_case(n_points=50, seed=2)
        assert not np.allclose(c1.features, c2.features)


class TestLeaveOneGeometryOut:
    """Verify LOGO protocol with small cases for speed."""

    @pytest.fixture
    def small_logo(self):
        """Create LOGO with small cases for fast testing."""
        cases = [
            generate_periodic_hill_case(n_points=100),
            generate_wall_hump_case(n_points=100),
            generate_bfs_case(n_points=100),
            generate_curved_bump_case(n_points=100),
        ]
        config = EmbeddingConfig(
            hidden_layers=(16, 16), max_epochs=50,
        )
        return LeaveOneGeometryOut(cases=cases, config=config)

    def test_leave_one_geometry_out_runs(self, small_logo):
        results = small_logo.run()
        assert len(results) == 4
        # Each result should have the test case name from a known geometry
        test_names = {r.test_case for r in results}
        assert "periodic_hill" in test_names
        assert "wall_hump" in test_names
        assert "backward_facing_step" in test_names
        assert "curved_bump" in test_names

    def test_generalization_r2_populated(self, small_logo):
        results = small_logo.run()
        for r in results:
            # R² should be a real number (can be negative but not NaN/Inf)
            assert np.isfinite(r.test_r2_beta), f"Non-finite R² for {r.test_case}"

    def test_results_stored(self, small_logo):
        small_logo.run()
        assert len(small_logo.results) == 4


class TestGeneralizationSummary:
    """Verify JSON-serializable summary output."""

    def test_summary_json_serializable(self):
        # Run a minimal LOGO
        cases = [
            generate_periodic_hill_case(n_points=50),
            generate_wall_hump_case(n_points=50),
        ]
        config = EmbeddingConfig(hidden_layers=(8,), max_epochs=20)
        logo = LeaveOneGeometryOut(cases=cases, config=config)
        results = logo.run()

        summary = get_generalization_summary(results)
        # Must be JSON-serializable
        json_str = json.dumps(summary)
        assert len(json_str) > 0

    def test_summary_structure(self):
        cases = [
            generate_periodic_hill_case(n_points=50),
            generate_wall_hump_case(n_points=50),
        ]
        config = EmbeddingConfig(hidden_layers=(8,), max_epochs=20)
        logo = LeaveOneGeometryOut(cases=cases, config=config)
        results = logo.run()

        summary = get_generalization_summary(results)
        assert summary["protocol"] == "leave_one_geometry_out"
        assert summary["n_geometries"] == 2
        assert "per_case" in summary
        assert "mean_r2_beta" in summary
        assert "mean_bubble_improvement_pct" in summary

    def test_summary_per_case_keys(self):
        cases = [
            generate_periodic_hill_case(n_points=50),
            generate_wall_hump_case(n_points=50),
        ]
        config = EmbeddingConfig(hidden_layers=(8,), max_epochs=20)
        logo = LeaveOneGeometryOut(cases=cases, config=config)
        results = logo.run()

        summary = get_generalization_summary(results)
        for case_name, case_data in summary["per_case"].items():
            assert "test_r2_beta" in case_data
            assert "bubble_improvement_pct" in case_data
            assert "passed" in case_data
