"""
Tests for PhysicsNeMo DoMINO Integration
=========================================
Validates the mocks for VTU-to-Zarr conversion, DoMINONIM wrapper,
Mixture-of-Experts routing, and benchmarking pipeline.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.ml_augmentation.physicsnemo_domino_integration import (
    VTUtoZarrConverter,
    DoMINONIMConfig,
    DoMINONIM,
    MixtureOfExpertsDoMINO,
    benchmark_domino,
)


class TestVTUtoZarrConverter:
    """Test PhysicsNeMo-Curator mock converter."""

    def test_convert_creates_zarr(self, tmp_path):
        """Conversion generates a .zarr directory structure."""
        converter = VTUtoZarrConverter(output_dir=tmp_path)
        vtu_files = ["case1.vtu", "case2.vtu"]
        
        zarr_path = Path(converter.convert_dataset(vtu_files))
        
        assert zarr_path.exists()
        assert zarr_path.is_dir()
        assert zarr_path.name == "dataset.zarr"
        assert (zarr_path / ".zgroup").exists()
        assert len(converter.converted_files) == 1


class TestDoMINONIM:
    """Test DoMINO Neural Inference Microservice wrapper."""

    def setup_method(self):
        self.config = DoMINONIMConfig(latent_dim=128)
        self.nim = DoMINONIM(config=self.config)
        self.pc = np.random.rand(100, 3)  # Mock 100-node point cloud

    def test_initial_state(self):
        """Model starts un-fine-tuned."""
        assert not self.nim.is_fine_tuned
        assert self.nim.training_distribution_centroid is None

    def test_fine_tune_updates_state(self):
        """Fine-tuning marks model as ready and calculates centroid."""
        history = self.nim.fine_tune("dummy.zarr", epochs=2)
        assert self.nim.is_fine_tuned
        assert self.nim.training_distribution_centroid is not None
        assert "final_loss" in history

    def test_zero_shot_predict(self):
        """Zero-shot inference returns valid shape array and confidence."""
        preds, conf = self.nim.predict(self.pc, mach=0.1, reynolds=1e5, aoa=0.0)
        assert preds.shape == (100, 2)  # Cp, Cf for 100 points
        assert 0.0 <= conf <= 1.0

    def test_fine_tuned_predict_confidence(self):
        """Fine-tuned model produces OOD confidence scores based on distance."""
        self.nim.fine_tune("dummy.zarr", epochs=1)
        
        # In-distribution-ish (mach 0.1)
        _, conf_id = self.nim.predict(self.pc, mach=0.1, reynolds=1e5, aoa=0.0)
        
        # Highly OOD (mach 5.0)
        _, conf_ood = self.nim.predict(self.pc, mach=5.0, reynolds=1e5, aoa=0.0)
        
        assert 0.0 <= conf_id <= 1.0
        assert 0.0 <= conf_ood <= 1.0
        assert conf_ood != conf_id  # Confidences should differ


class TestMixtureOfExpertsDoMINO:
    """Test 25.11 regime-specific MoE routing."""

    def setup_method(self):
        self.moe = MixtureOfExpertsDoMINO()
        self.pc = np.random.rand(50, 3)

    def test_router_logic(self):
        """Gate network routes to appropriate experts."""
        assert self.moe.gate_network(mach=0.9, reynolds=1e5, aoa=0.0) == "shock_induced"
        assert self.moe.gate_network(mach=0.1, reynolds=1e5, aoa=15.0) == "sharp_separation"
        assert self.moe.gate_network(mach=0.1, reynolds=1e5, aoa=0.0) == "attached"

    def test_fine_tune_experts(self):
        """Fine-tuning routes to correct internal models."""
        datasets = {"attached": "att.zarr", "sharp_separation": "sep.zarr"}
        self.moe.fine_tune_experts(datasets, epochs=1)
        
        assert self.moe.experts["attached"].is_fine_tuned
        assert self.moe.experts["sharp_separation"].is_fine_tuned
        assert not self.moe.experts["smooth_separation"].is_fine_tuned

    def test_predict_routing(self):
        """Predict method returns expert name along with data."""
        preds, conf, expert_name = self.moe.predict(
            self.pc, mach=0.1, reynolds=1e5, aoa=15.0
        )
        assert expert_name == "sharp_separation"
        assert preds.shape == (50, 2)


class TestBenchmarkDoMINO:
    """Test benchmark pipeline."""

    def test_benchmark_returns_structured_results(self, tmp_path):
        """Pipeline computes metrics and saves to JSON."""
        results = benchmark_domino(output_dir=str(tmp_path))
        
        assert "dataset" in results
        assert "metrics" in results
        assert "DoMINO (25.08)" in results["metrics"]
        assert "DoMINO MoE (25.11)" in results["metrics"]
        
        moe_metrics = results["metrics"]["DoMINO MoE (25.11)"]
        assert "MAPE_Cp" in moe_metrics
        assert "GPU_latency_ms" in moe_metrics
        assert moe_metrics["supports_ood_estimation"] is True
        
        assert (tmp_path / "domino_benchmark.json").exists()
