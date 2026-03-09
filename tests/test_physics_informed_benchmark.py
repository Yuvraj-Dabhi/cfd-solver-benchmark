"""
Tests for Physics-Informed ML Invariance vs Realizability Benchmark
=====================================================================
Validates the benchmark harness and expected constraint violations
for the 3 compared architectures.
"""

import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ml_augmentation.physics_informed_benchmark import (
    PhysicsAwarenessBenchmark,
    PhysicsBenchmarkResult
)

class TestPhysicsAwarenessBenchmark:
    
    def test_synthetic_data_generation(self):
        bm = PhysicsAwarenessBenchmark(seed=42)
        data = bm._generate_synthetic_flow_field(n_points=10)
        
        assert "S" in data
        assert "Omega" in data
        assert "b_true" in data
        assert "cf_true" in data
        
        assert data["S"].shape == (10, 3, 3)
        assert data["Omega"].shape == (10, 3, 3)
        assert data["b_true"].shape == (10, 3, 3)
        
        # Verify b_true is realizable (trace free & symmetric)
        traces = np.trace(data["b_true"], axis1=1, axis2=2)
        assert np.allclose(traces, 0, atol=1e-5)
        
        sym_diff = data["b_true"] - data["b_true"].transpose(0, 2, 1)
        assert np.allclose(sym_diff, 0, atol=1e-5)

    def test_vanilla_mlp_violates_physics(self):
        bm = PhysicsAwarenessBenchmark(seed=42)
        data = bm._generate_synthetic_flow_field(n_points=50)
        
        res = bm.evaluate_vanilla_mlp(data)
        
        assert res.architecture == "Vanilla_MLP_Raw"
        assert res.anisotropy_rmse > 0
        
        # Vanilla MLP should massively fail physics constraints
        assert res.invariance_violation_pct > 0
        assert res.trace_violation_pct > 0
        assert res.symmetry_violation_pct > 0
        
    def test_invariant_mlp_violates_some_physics(self):
        bm = PhysicsAwarenessBenchmark(seed=42)
        data = bm._generate_synthetic_flow_field(n_points=50)
        
        res = bm.evaluate_invariant_mlp(data)
        
        assert res.architecture == "Invariant_MLP_Scalars"
        assert res.anisotropy_rmse > 0
        
        # Invariant MLP fails trace and symmetry because it maps scalars to 9 independent tensor components
        assert res.trace_violation_pct > 0
        assert res.symmetry_violation_pct > 0
        
    def test_tbnn_satisfies_physics(self):
        bm = PhysicsAwarenessBenchmark(seed=42)
        data = bm._generate_synthetic_flow_field(n_points=50)
        
        res = bm.evaluate_tbnn(data)
        
        assert res.architecture == "TBNN_Tensor_Basis"
        
        # TBNN satisfies exact symmetry and trace because of the Pope basis
        assert res.invariance_violation_pct == 0.0
        assert res.trace_violation_pct == 0.0
        assert res.symmetry_violation_pct == 0.0
        
    def test_run_benchmark_and_markdown(self):
        bm = PhysicsAwarenessBenchmark(seed=42)
        results = bm.run_benchmark()
        
        assert len(results) == 3
        
        md = bm.generate_markdown_report()
        assert "Vanilla MLP" in md
        assert "Invariant MLP" in md
        assert "TBNN" in md
        assert "RMSE Accuracy" in md
