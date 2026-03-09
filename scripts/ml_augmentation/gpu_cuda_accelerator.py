#!/usr/bin/env python3
"""
GPU/CUDA Acceleration Configuration
=======================================
CUDA acceleration and GPU workflow management for SU2 v8.4.0
and co-resident ML (PyTorch/GNN) workloads.

Key features:
  - CUDAConfigGenerator: SU2 config with GPU FGMRES linear solver
  - HybridWorkflowManager: GPU primal + CPU adjoint orchestration
  - GPUMemoryEstimator: VRAM requirement estimation
  - NvBLASConfigurator: NvBLAS library config for BLAS interception
  - GPUBenchmarkRunner: CPU vs GPU speedup comparison
  - PyTorchDeviceManager: GPU memory management for ML co-residency

Architecture reference:
  - Economon et al. (2023): SU2 v8+ GPU-native solver
  - Giles et al. (2022): GPU adjoint for design optimization
  - NVIDIA (2024): NvBLAS drop-in GPU BLAS acceleration

Usage:
    from scripts.ml_augmentation.gpu_cuda_accelerator import (
        CUDAConfigGenerator, GPUBenchmarkRunner,
    )
    gen = CUDAConfigGenerator(mesh_size=50e6)
    config = gen.generate()
"""

import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)
PROJECT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT))


# =============================================================================
# GPU Specifications
# =============================================================================
@dataclass
class GPUSpec:
    """GPU hardware specification."""
    name: str = "NVIDIA A100"
    vram_gb: float = 80.0
    fp64_tflops: float = 9.7
    fp32_tflops: float = 19.5
    memory_bandwidth_gb_s: float = 2039.0  # GB/s
    pcie_gen: int = 4
    nvlink: bool = True
    compute_capability: Tuple[int, int] = (8, 0)

    # Common GPU presets
    @classmethod
    def a100_80gb(cls):
        return cls(name="NVIDIA A100-80GB-SXM", vram_gb=80.0,
                   fp64_tflops=9.7, fp32_tflops=19.5)

    @classmethod
    def v100_32gb(cls):
        return cls(name="NVIDIA V100-SXM2-32GB", vram_gb=32.0,
                   fp64_tflops=7.8, fp32_tflops=15.7,
                   memory_bandwidth_gb_s=900.0,
                   compute_capability=(7, 0))

    @classmethod
    def h100_80gb(cls):
        return cls(name="NVIDIA H100-SXM5-80GB", vram_gb=80.0,
                   fp64_tflops=34.0, fp32_tflops=67.0,
                   memory_bandwidth_gb_s=3350.0,
                   compute_capability=(9, 0))


# =============================================================================
# CUDA Config Generator
# =============================================================================
class CUDAConfigGenerator:
    """
    Generates SU2 configuration for GPU-accelerated RANS simulations.

    SU2 v8.4.0+ supports GPU-native linear solvers via CUDA:
      - FGMRES with ILU(0) preconditioner on GPU
      - Mixed-precision (FP32 residual, FP64 solution)
      - GPU-resident data (minimal CPU↔GPU transfers)

    Parameters
    ----------
    mesh_size : float
        Number of mesh points (e.g., 50e6).
    gpu : GPUSpec
        GPU hardware specification.
    mixed_precision : bool
        Use FP32 for residual computation (2x speedup).
    """

    def __init__(self, mesh_size: float = 50e6, gpu: GPUSpec = None,
                 mixed_precision: bool = True):
        self.mesh_size = mesh_size
        self.gpu = gpu or GPUSpec.a100_80gb()
        self.mixed_precision = mixed_precision

    def generate(self) -> Dict[str, Any]:
        """
        Generate SU2 CUDA-enabled configuration.

        Returns
        -------
        Dict with SU2 config key-value pairs for GPU execution.
        """
        config = {
            # Core solver
            "SOLVER": "RANS",
            "KIND_TURB_MODEL": "SST",
            "MATH_PROBLEM": "DIRECT",

            # GPU Linear solver
            "LINEAR_SOLVER": "FGMRES",
            "LINEAR_SOLVER_PREC": "ILU",
            "LINEAR_SOLVER_ILU_FILL_IN": 0,
            "LINEAR_SOLVER_ERROR": 1e-6,
            "LINEAR_SOLVER_ITER": 20,

            # GPU acceleration flags
            "USE_VECTORIZATION": "YES",
            "DIRECT_DIFF": "NO",

            # Numerics
            "NUM_METHOD_GRAD": "GREEN_GAUSS",
            "CONV_NUM_METHOD_FLOW": "ROE",
            "MUSCL_FLOW": "YES",
            "SLOPE_LIMITER_FLOW": "VENKATAKRISHNAN",
            "VENKAT_LIMITER_COEFF": 0.05,

            # Convergence
            "ITER": 10000,
            "CFL_NUMBER": 10.0,
            "CFL_ADAPT": "YES",
            "CFL_ADAPT_PARAM": "(0.1, 2.0, 5.0, 200.0)",
            "CONV_RESIDUAL_MINVAL": -10,

            # Output
            "OUTPUT_FILES": "(RESTART, PARAVIEW_MULTIBLOCK)",
            "OUTPUT_WRT_FREQ": 250,
        }

        # Mixed precision note
        if self.mixed_precision:
            config["_COMMENT_MIXED_PREC"] = \
                "Build SU2 with -DENABLE_MIXED_PRECISION=ON for 2x speedup"

        return config

    def generate_build_flags(self) -> Dict[str, str]:
        """
        Generate CMake build flags for GPU-enabled SU2.

        Returns
        -------
        Dict with CMake variable → value mappings.
        """
        cc = f"{self.gpu.compute_capability[0]}{self.gpu.compute_capability[1]}"
        return {
            "CMAKE_BUILD_TYPE": "Release",
            "SU2_ENABLE_CUDA": "ON",
            "CMAKE_CUDA_ARCHITECTURES": cc,
            "ENABLE_MIXED_PRECISION": "ON" if self.mixed_precision else "OFF",
            "SU2_ENABLE_MPI": "ON",
            "SU2_ENABLE_CGNS": "ON",
            "SU2_ENABLE_TECIO": "ON",
            "CUDA_TOOLKIT_ROOT_DIR": "${CUDA_HOME}",
        }

    def generate_build_command(self) -> str:
        """Generate the CMake build command string."""
        flags = self.generate_build_flags()
        flag_str = " ".join(f"-D{k}={v}" for k, v in flags.items())
        return f"cmake {flag_str} .. && make -j$(nproc)"

    def validate_gpu_fit(self) -> Dict[str, Any]:
        """
        Check if the mesh fits in GPU VRAM.

        Returns
        -------
        Dict with memory analysis.
        """
        estimator = GPUMemoryEstimator(self.gpu)
        estimate = estimator.estimate(self.mesh_size)
        fits = estimate["total_gb"] < self.gpu.vram_gb * 0.9  # 90% utilization
        return {
            "fits_in_vram": fits,
            "estimated_vram_gb": estimate["total_gb"],
            "available_vram_gb": self.gpu.vram_gb,
            "utilization_pct": estimate["total_gb"] / self.gpu.vram_gb * 100,
            **estimate,
        }


# =============================================================================
# GPU Memory Estimator
# =============================================================================
class GPUMemoryEstimator:
    """
    Estimates GPU VRAM requirements for CFD simulations.

    Memory model for SU2 RANS:
      - Conservative variables: 5 × n_cells × 8 bytes (FP64)
      - Gradient storage: 5 × 3 × n_cells × 8 bytes
      - Turbulence model: 2 × n_cells × 8 bytes (SST: k, ω)
      - Linear solver workspace: ~3× solution vectors
      - Sparse matrix (CSR): ~200 × n_cells bytes

    Parameters
    ----------
    gpu : GPUSpec
    """

    def __init__(self, gpu: GPUSpec = None):
        self.gpu = gpu or GPUSpec.a100_80gb()

    def estimate(self, n_cells: float, n_vars: int = 5,
                 turb_vars: int = 2) -> Dict[str, float]:
        """
        Estimate VRAM requirement.

        Parameters
        ----------
        n_cells : float
            Number of mesh cells.
        n_vars : int
            Number of conservative variables.
        turb_vars : int
            Number of turbulence variables.

        Returns
        -------
        Dict with memory breakdown in GB.
        """
        bytes_per_fp64 = 8
        n = int(n_cells)

        # Solution vectors
        conservative = n * n_vars * bytes_per_fp64
        gradients = n * n_vars * 3 * bytes_per_fp64
        turbulence = n * turb_vars * bytes_per_fp64
        turb_gradients = n * turb_vars * 3 * bytes_per_fp64

        # Linear solver (FGMRES workspace: ~3-5 Krylov vectors)
        krylov_vectors = 5
        linear_solver = n * (n_vars + turb_vars) * bytes_per_fp64 * krylov_vectors

        # Sparse matrix (CSR format, ~7 non-zeros per row avg)
        nnz_per_row = 7
        sparse_matrix = n * nnz_per_row * bytes_per_fp64 * 2  # Values + indices

        # Preconditioner (ILU0 ≈ same as matrix)
        preconditioner = sparse_matrix

        total_bytes = (conservative + gradients + turbulence + turb_gradients +
                       linear_solver + sparse_matrix + preconditioner)
        total_gb = total_bytes / (1024 ** 3)

        return {
            "conservative_gb": conservative / (1024 ** 3),
            "gradients_gb": gradients / (1024 ** 3),
            "turbulence_gb": (turbulence + turb_gradients) / (1024 ** 3),
            "linear_solver_gb": linear_solver / (1024 ** 3),
            "sparse_matrix_gb": sparse_matrix / (1024 ** 3),
            "preconditioner_gb": preconditioner / (1024 ** 3),
            "total_gb": total_gb,
            "max_mesh_for_gpu": int(self.gpu.vram_gb * 0.9 * (1024 ** 3) /
                                     (total_bytes / n + 1e-15)),
        }


# =============================================================================
# NvBLAS Configurator
# =============================================================================
class NvBLASConfigurator:
    """
    NvBLAS drop-in GPU BLAS acceleration configuration.

    NvBLAS intercepts Level 3 BLAS calls (DGEMM, DSYRK, etc.)
    and executes them on GPUs transparently.

    Parameters
    ----------
    gpu_list : list of int
        GPU device indices to use.
    autopin : bool
        Enable automatic memory pinning.
    tile_dim : int
        Tile dimension for GPU DGEMM.
    """

    def __init__(self, gpu_list: List[int] = None, autopin: bool = True,
                 tile_dim: int = 2048):
        self.gpu_list = gpu_list or [0]
        self.autopin = autopin
        self.tile_dim = tile_dim

    def generate_config(self) -> str:
        """
        Generate nvblas.conf file content.

        Returns
        -------
        Config file content as string.
        """
        gpu_str = " ".join(str(g) for g in self.gpu_list)
        lines = [
            "# NvBLAS Configuration for SU2 GPU Acceleration",
            f"NVBLAS_LOGFILE   nvblas.log",
            f"NVBLAS_GPU_LIST  {gpu_str}",
            f"NVBLAS_TILE_DIM  {self.tile_dim}",
            f"NVBLAS_AUTOPIN_MEM_ENABLED" if self.autopin else "",
            "",
            "# CPU BLAS Fallback Library",
            "NVBLAS_CPU_BLAS_LIB  /usr/lib/x86_64-linux-gnu/libopenblas.so",
            "",
            "# Intercepted BLAS routines",
            "NVBLAS_GPU_DISABLED_SGEMM",
            "NVBLAS_GPU_DISABLED_CGEMM",
            "NVBLAS_GPU_DISABLED_ZGEMM",
        ]
        return "\n".join(l for l in lines if l is not None)

    def generate_launch_command(self, su2_binary: str = "SU2_CFD",
                                config_file: str = "config.cfg",
                                n_mpi: int = 1) -> str:
        """Generate launch command with NvBLAS preload."""
        return (
            f"LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnvblas.so "
            f"NVBLAS_CONFIG_FILE=nvblas.conf "
            f"mpirun -np {n_mpi} {su2_binary} {config_file}"
        )


# =============================================================================
# Hybrid Workflow Manager
# =============================================================================
class HybridWorkflowManager:
    """
    Orchestrates GPU primal / CPU adjoint hybrid workflows.

    SU2 adjoint solver may not be fully GPU-optimized, so a hybrid
    approach runs the forward solver on GPU and the adjoint on CPU.

    Parameters
    ----------
    gpu : GPUSpec
    n_cpu_cores : int
    config_base : dict
    """

    def __init__(self, gpu: GPUSpec = None, n_cpu_cores: int = 64,
                 config_base: Dict[str, Any] = None):
        self.gpu = gpu or GPUSpec.a100_80gb()
        self.n_cpu_cores = n_cpu_cores
        self.config_base = config_base or {}

    def generate_primal_config(self) -> Dict[str, Any]:
        """Generate GPU-optimized primal (forward) solver config."""
        config = dict(self.config_base)
        config.update({
            "MATH_PROBLEM": "DIRECT",
            "LINEAR_SOLVER": "FGMRES",
            "LINEAR_SOLVER_PREC": "ILU",
            "LINEAR_SOLVER_ILU_FILL_IN": 0,
            "CFL_NUMBER": 10.0,
            "_EXECUTION": "GPU",
        })
        return config

    def generate_adjoint_config(self) -> Dict[str, Any]:
        """Generate CPU-optimized adjoint solver config."""
        config = dict(self.config_base)
        config.update({
            "MATH_PROBLEM": "DISCRETE_ADJOINT",
            "LINEAR_SOLVER": "FGMRES",
            "LINEAR_SOLVER_PREC": "ILU",
            "LINEAR_SOLVER_ILU_FILL_IN": 1,
            "CFL_NUMBER": 5.0,
            "ITER": 500,
            "_EXECUTION": "CPU",
            "_COMMENT": f"Run with mpirun -np {self.n_cpu_cores}",
        })
        return config

    def estimate_workflow_time(self, mesh_size: float,
                               n_design_iterations: int = 10) -> Dict[str, float]:
        """
        Estimate total workflow time for design optimization.

        Parameters
        ----------
        mesh_size : float
        n_design_iterations : int

        Returns
        -------
        Dict with time estimates in hours.
        """
        # Rough estimate: 1M cells/GPU TFLOP per iteration
        primal_per_iter_s = mesh_size / (self.gpu.fp64_tflops * 1e6) * 10
        adjoint_per_iter_s = mesh_size / (self.n_cpu_cores * 1e4) * 5

        primal_total = primal_per_iter_s * n_design_iterations / 3600
        adjoint_total = adjoint_per_iter_s * n_design_iterations / 3600

        return {
            "primal_per_iter_h": primal_per_iter_s / 3600,
            "adjoint_per_iter_h": adjoint_per_iter_s / 3600,
            "primal_total_h": primal_total,
            "adjoint_total_h": adjoint_total,
            "total_h": primal_total + adjoint_total,
            "speedup_vs_cpu_only": (
                (mesh_size / (self.n_cpu_cores * 1e4) * 15 * n_design_iterations / 3600) /
                max(primal_total + adjoint_total, 1e-15)
            ),
        }


# =============================================================================
# GPU Benchmark Runner
# =============================================================================
class GPUBenchmarkRunner:
    """
    Benchmarks CPU vs GPU performance for SU2 RANS simulations.

    Generates performance comparison reports with speedup metrics.

    Parameters
    ----------
    gpu : GPUSpec
    n_cpu_cores : int
    """

    def __init__(self, gpu: GPUSpec = None, n_cpu_cores: int = 64):
        self.gpu = gpu or GPUSpec.a100_80gb()
        self.n_cpu_cores = n_cpu_cores
        self._results = {}

    def estimate_performance(self, mesh_sizes: List[float] = None
                              ) -> Dict[str, Any]:
        """
        Estimate CPU vs GPU performance across mesh sizes.

        Parameters
        ----------
        mesh_sizes : list of float
            Mesh sizes to benchmark.

        Returns
        -------
        Performance comparison dict.
        """
        if mesh_sizes is None:
            mesh_sizes = [1e6, 5e6, 10e6, 50e6, 100e6]

        results = []
        for n in mesh_sizes:
            cpu_time = n / (self.n_cpu_cores * 5e3)   # Rough: cells / (cores * rate)
            gpu_time = n / (self.gpu.fp64_tflops * 5e5)  # Rough: cells / (TFLOPS * rate)
            speedup = cpu_time / max(gpu_time, 1e-15)

            results.append({
                "mesh_size": int(n),
                "cpu_time_s": cpu_time,
                "gpu_time_s": gpu_time,
                "speedup": speedup,
                "fits_in_vram": GPUMemoryEstimator(self.gpu).estimate(n)["total_gb"] < self.gpu.vram_gb * 0.9,
            })

        self._results = {
            "gpu": self.gpu.name,
            "n_cpu_cores": self.n_cpu_cores,
            "benchmarks": results,
            "avg_speedup": float(np.mean([r["speedup"] for r in results])),
        }
        return self._results

    def report(self) -> str:
        """Generate performance comparison report."""
        if not self._results:
            return "No benchmark results."

        lines = [
            "GPU Benchmark Report",
            "=" * 50,
            f"GPU: {self._results['gpu']}",
            f"CPU: {self._results['n_cpu_cores']} cores",
            "",
            f"{'Mesh Size':>12} {'CPU (s)':>10} {'GPU (s)':>10} {'Speedup':>8} {'Fits':>5}",
            "-" * 50,
        ]
        for r in self._results["benchmarks"]:
            fit = "✓" if r["fits_in_vram"] else "✗"
            lines.append(
                f"{r['mesh_size']:>12,d} {r['cpu_time_s']:>10.1f} "
                f"{r['gpu_time_s']:>10.1f} {r['speedup']:>7.1f}x {fit:>5}")
        lines.append(f"\nAvg speedup: {self._results['avg_speedup']:.1f}x")
        return "\n".join(lines)


# =============================================================================
# PyTorch Device Manager
# =============================================================================
class PyTorchDeviceManager:
    """
    Manages GPU memory for co-resident SU2 and PyTorch (GNN) workloads.

    When running ML inference alongside SU2, GPU memory must be
    partitioned to avoid OOM errors.

    Parameters
    ----------
    gpu : GPUSpec
    su2_vram_fraction : float
        Fraction of VRAM reserved for SU2 solver.
    """

    def __init__(self, gpu: GPUSpec = None, su2_vram_fraction: float = 0.6):
        self.gpu = gpu or GPUSpec.a100_80gb()
        self.su2_vram_fraction = su2_vram_fraction

    def get_ml_allocation(self) -> Dict[str, float]:
        """
        Calculate available VRAM for ML workloads.

        Returns
        -------
        Dict with memory allocation in GB.
        """
        su2_gb = self.gpu.vram_gb * self.su2_vram_fraction
        ml_gb = self.gpu.vram_gb * (1 - self.su2_vram_fraction)
        overhead_gb = 2.0  # CUDA runtime overhead

        return {
            "total_vram_gb": self.gpu.vram_gb,
            "su2_reserved_gb": su2_gb,
            "ml_available_gb": ml_gb - overhead_gb,
            "cuda_overhead_gb": overhead_gb,
        }

    def estimate_max_gnn_nodes(self, hidden_dim: int = 128,
                                 n_message_passing: int = 6) -> int:
        """
        Estimate maximum GNN graph size for available memory.

        Parameters
        ----------
        hidden_dim : int
        n_message_passing : int

        Returns
        -------
        Maximum number of graph nodes.
        """
        alloc = self.get_ml_allocation()
        available_bytes = alloc["ml_available_gb"] * (1024 ** 3)

        # Memory per node: features + messages + hidden states
        bytes_per_node = hidden_dim * 4 * (n_message_passing + 2) * 4  # FP32
        return int(available_bytes / max(bytes_per_node, 1))

    def generate_pytorch_env(self) -> Dict[str, str]:
        """Generate environment variables for PyTorch memory management."""
        alloc = self.get_ml_allocation()
        fraction = (1 - self.su2_vram_fraction) * 0.95  # 95% of ML allocation
        return {
            "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512",
            "CUDA_VISIBLE_DEVICES": "0",
            "PYTORCH_NO_CUDA_MEMORY_CACHING": "0",
            "_MAX_GPU_MEMORY_GB": f"{alloc['ml_available_gb']:.1f}",
            "_RECOMMENDED_BATCH_SIZE": "1",
        }
