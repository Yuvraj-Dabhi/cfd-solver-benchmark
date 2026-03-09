"""
Tests for ML-Corrected RANS Solver-Loop Integration (Gap 2)
=============================================================
Tests the iterative outer-loop coupling of ML β-correction
with the flow solver using dry-run mode.
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ml_augmentation.solver_coupling import (
    MLCorrectedRANSLoop,
    SolverCouplingConfig,
    CouplingHistory,
    extract_coupling_features,
    _generate_synthetic_flow,
)


class TestDryRunCoupling:
    """Test the ML-RANS coupling loop in dry-run mode."""

    def test_dry_run_completes(self):
        config = SolverCouplingConfig(
            mode="dry_run",
            max_outer_iterations=3,
            n_mesh_nodes=500,
            n_wall_points=50,
        )
        loop = MLCorrectedRANSLoop(config)
        history = loop.run()

        assert isinstance(history, CouplingHistory)
        assert history.n_iterations > 0
        assert history.n_iterations <= 3
        assert len(history.iterations) == history.n_iterations

    def test_convergence_detection(self):
        """Inject near-converged β and verify loop terminates early."""
        config = SolverCouplingConfig(
            mode="dry_run",
            max_outer_iterations=20,
            convergence_tol=0.5,  # Very loose tolerance
            n_mesh_nodes=100,
            n_wall_points=50,
        )
        loop = MLCorrectedRANSLoop(config)
        history = loop.run()

        # Should converge before max iterations with loose tol
        assert history.n_iterations < 20

    def test_beta_relaxation_clamps(self):
        """Verify β stays within [beta_min, beta_max] bounds."""
        config = SolverCouplingConfig(
            mode="dry_run",
            max_outer_iterations=5,
            beta_min=0.5,
            beta_max=2.0,
            n_mesh_nodes=200,
        )
        loop = MLCorrectedRANSLoop(config)
        loop.run()

        beta = loop._beta
        assert np.all(beta >= config.beta_min)
        assert np.all(beta <= config.beta_max)

    def test_history_populated(self):
        config = SolverCouplingConfig(
            mode="dry_run", max_outer_iterations=3,
            n_mesh_nodes=200,
        )
        loop = MLCorrectedRANSLoop(config)
        history = loop.run()

        for it in history.iterations:
            assert it.beta_norm > 0
            assert np.isfinite(it.cf_rmse)
            assert it.wall_time_s >= 0

    def test_history_export_json(self):
        config = SolverCouplingConfig(
            mode="dry_run", max_outer_iterations=2,
            n_mesh_nodes=100,
        )
        loop = MLCorrectedRANSLoop(config)
        history = loop.run()

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            loop.save_history(f.name)

        with open(f.name) as fp:
            data = json.load(fp)

        assert "n_iterations" in data
        assert "per_iteration" in data
        assert len(data["per_iteration"]) == history.n_iterations


class TestSyntheticFlow:
    """Test synthetic flow data generation."""

    def test_flow_data_shapes(self):
        beta = np.ones(500)
        flow = _generate_synthetic_flow(500, 100, beta)

        assert flow["x"].shape == (500,)
        assert flow["nu_t"].shape == (500,)
        assert flow["Cf"].shape == (100,)
        assert flow["Cf_exp"].shape == (100,)
        assert flow["x_wall"].shape == (100,)

    def test_flow_responds_to_beta(self):
        beta_low = np.ones(500)
        beta_high = np.ones(500) * 1.5

        flow_low = _generate_synthetic_flow(500, 100, beta_low, seed=1)
        flow_high = _generate_synthetic_flow(500, 100, beta_high, seed=1)

        # Higher beta should change the flow
        assert not np.allclose(flow_low["Cf"], flow_high["Cf"])


class TestFeatureExtraction:
    """Test coupling feature extraction."""

    def test_feature_shapes(self):
        beta = np.ones(300)
        flow = _generate_synthetic_flow(300, 50, beta)
        features = extract_coupling_features(flow)

        assert features.shape == (300, 5)
        assert np.all(np.isfinite(features))
