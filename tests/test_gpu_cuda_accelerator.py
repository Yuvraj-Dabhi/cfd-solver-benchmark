#!/usr/bin/env python3
"""
Tests for GPU/CUDA Acceleration Module
==========================================
Validates CUDA config generation, memory estimation, NvBLAS config,
hybrid workflow orchestration, and benchmark reporting.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ml_augmentation.gpu_cuda_accelerator import (
    CUDAConfigGenerator,
    GPUBenchmarkRunner,
    GPUMemoryEstimator,
    GPUSpec,
    HybridWorkflowManager,
    NvBLASConfigurator,
    PyTorchDeviceManager,
)


# =========================================================================
# TestGPUSpec
# =========================================================================
class TestGPUSpec:
    """Tests for GPU hardware specifications."""

    def test_a100_preset(self):
        gpu = GPUSpec.a100_80gb()
        assert gpu.vram_gb == 80.0
        assert gpu.fp64_tflops == pytest.approx(9.7)

    def test_v100_preset(self):
        gpu = GPUSpec.v100_32gb()
        assert gpu.vram_gb == 32.0

    def test_h100_preset(self):
        gpu = GPUSpec.h100_80gb()
        assert gpu.fp64_tflops > 30.0


# =========================================================================
# TestCUDAConfigGenerator
# =========================================================================
class TestCUDAConfigGenerator:
    """Tests for SU2 CUDA config generation."""

    def test_generates_config(self):
        gen = CUDAConfigGenerator(mesh_size=10e6)
        config = gen.generate()
        assert config["SOLVER"] == "RANS"
        assert config["LINEAR_SOLVER"] == "FGMRES"
        assert config["LINEAR_SOLVER_PREC"] == "ILU"

    def test_build_flags(self):
        gen = CUDAConfigGenerator()
        flags = gen.generate_build_flags()
        assert "SU2_ENABLE_CUDA" in flags
        assert flags["SU2_ENABLE_CUDA"] == "ON"
        assert "CMAKE_CUDA_ARCHITECTURES" in flags

    def test_build_command(self):
        gen = CUDAConfigGenerator()
        cmd = gen.generate_build_command()
        assert "cmake" in cmd
        assert "SU2_ENABLE_CUDA=ON" in cmd

    def test_validate_small_mesh_fits(self):
        gpu = GPUSpec.a100_80gb()
        gen = CUDAConfigGenerator(mesh_size=1e6, gpu=gpu)
        result = gen.validate_gpu_fit()
        assert result["fits_in_vram"] is True
        assert result["estimated_vram_gb"] < 80

    def test_validate_huge_mesh_does_not_fit(self):
        gpu = GPUSpec.v100_32gb()
        gen = CUDAConfigGenerator(mesh_size=500e6, gpu=gpu)
        result = gen.validate_gpu_fit()
        # 500M cells on V100-32GB should not fit
        assert result["utilization_pct"] > 90 or not result["fits_in_vram"]


# =========================================================================
# TestGPUMemoryEstimator
# =========================================================================
class TestGPUMemoryEstimator:
    """Tests for GPU memory estimation."""

    def test_estimate_returns_positive(self):
        estimator = GPUMemoryEstimator()
        result = estimator.estimate(10e6)
        assert result["total_gb"] > 0

    def test_memory_scales_with_mesh(self):
        estimator = GPUMemoryEstimator()
        small = estimator.estimate(1e6)
        large = estimator.estimate(10e6)
        assert large["total_gb"] > small["total_gb"]

    def test_memory_breakdown_sums(self):
        estimator = GPUMemoryEstimator()
        result = estimator.estimate(5e6)
        components = (result["conservative_gb"] + result["gradients_gb"] +
                      result["turbulence_gb"] + result["linear_solver_gb"] +
                      result["sparse_matrix_gb"] + result["preconditioner_gb"])
        assert result["total_gb"] == pytest.approx(components, rel=0.01)

    def test_max_mesh_positive(self):
        estimator = GPUMemoryEstimator()
        result = estimator.estimate(10e6)
        assert result["max_mesh_for_gpu"] > 0


# =========================================================================
# TestNvBLASConfigurator
# =========================================================================
class TestNvBLASConfigurator:
    """Tests for NvBLAS configuration."""

    def test_config_format(self):
        config = NvBLASConfigurator(gpu_list=[0, 1])
        content = config.generate_config()
        assert "NVBLAS_GPU_LIST" in content
        assert "0 1" in content

    def test_launch_command(self):
        config = NvBLASConfigurator()
        cmd = config.generate_launch_command(n_mpi=4)
        assert "LD_PRELOAD" in cmd
        assert "mpirun -np 4" in cmd

    def test_single_gpu(self):
        config = NvBLASConfigurator(gpu_list=[0])
        content = config.generate_config()
        assert "NVBLAS_TILE_DIM" in content


# =========================================================================
# TestHybridWorkflowManager
# =========================================================================
class TestHybridWorkflowManager:
    """Tests for hybrid GPU/CPU workflow."""

    def test_primal_config(self):
        manager = HybridWorkflowManager()
        config = manager.generate_primal_config()
        assert config["MATH_PROBLEM"] == "DIRECT"
        assert config["_EXECUTION"] == "GPU"

    def test_adjoint_config(self):
        manager = HybridWorkflowManager(n_cpu_cores=32)
        config = manager.generate_adjoint_config()
        assert config["MATH_PROBLEM"] == "DISCRETE_ADJOINT"
        assert config["_EXECUTION"] == "CPU"

    def test_workflow_time_estimate(self):
        manager = HybridWorkflowManager()
        estimate = manager.estimate_workflow_time(mesh_size=10e6)
        assert estimate["total_h"] > 0
        assert estimate["speedup_vs_cpu_only"] > 1.0


# =========================================================================
# TestGPUBenchmarkRunner
# =========================================================================
class TestGPUBenchmarkRunner:
    """Tests for GPU benchmark runner."""

    def test_performance_estimate(self):
        runner = GPUBenchmarkRunner()
        results = runner.estimate_performance()
        assert len(results["benchmarks"]) > 0
        assert results["avg_speedup"] > 1.0

    def test_report(self):
        runner = GPUBenchmarkRunner()
        runner.estimate_performance(mesh_sizes=[1e6, 10e6])
        report = runner.report()
        assert "GPU Benchmark Report" in report
        assert "Speedup" in report

    def test_custom_gpu(self):
        runner = GPUBenchmarkRunner(
            gpu=GPUSpec.h100_80gb(), n_cpu_cores=128)
        results = runner.estimate_performance(mesh_sizes=[10e6])
        assert results["gpu"] == "NVIDIA H100-SXM5-80GB"


# =========================================================================
# TestPyTorchDeviceManager
# =========================================================================
class TestPyTorchDeviceManager:
    """Tests for PyTorch device management."""

    def test_memory_allocation(self):
        manager = PyTorchDeviceManager()
        alloc = manager.get_ml_allocation()
        assert alloc["total_vram_gb"] == 80.0
        assert alloc["ml_available_gb"] > 0
        assert alloc["ml_available_gb"] < alloc["total_vram_gb"]

    def test_max_gnn_nodes(self):
        manager = PyTorchDeviceManager()
        max_nodes = manager.estimate_max_gnn_nodes()
        assert max_nodes > 0

    def test_pytorch_env(self):
        manager = PyTorchDeviceManager()
        env = manager.generate_pytorch_env()
        assert "PYTORCH_CUDA_ALLOC_CONF" in env
        assert "CUDA_VISIBLE_DEVICES" in env


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
