#!/usr/bin/env python3
"""
ML-Solver Coupling Latency Benchmark
=====================================
Quantifies the Python-C++ API boundary overhead for in-situ ML-CFD coupling.

Benchmark matrix:

| Method                     | Expected Latency | Overhead Source          |
|----------------------------|------------------|--------------------------|
| Python PyTorch loop        | ~40 us/iter      | GIL + interpreter        |
| torch.no_grad batch        | ~15 us/iter      | API boundary             |
| TorchScript traced         | ~10 us/iter      | Reduced interpreter      |
| TorchScript + zero-copy    | ~8 us/iter       | Minimal copy             |
| NumPy matmul (sklearn)     | ~4 us/iter       | No framework overhead    |
| C++ libtorch (generated)   | ~2.5 us/iter     | Native (no Python)       |
| Raw C++ (ideal)            | ~0.002 us/iter   | Theoretical minimum      |

Usage:
    python -m scripts.utils.coupling_benchmark
    python -m scripts.utils.coupling_benchmark --n-points 50000 --n-repeats 200
"""

import argparse
import logging
import time
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

logger = logging.getLogger(__name__)

# Check for PyTorch
_HAS_TORCH = False
try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except ImportError:
    pass


@dataclass
class BenchmarkRow:
    """Single row of the coupling benchmark table."""
    method: str
    latency_us: float
    python_multiplier: float
    cpp_multiplier: float
    notes: str = ""


def _build_simple_nn(n_features: int = 5, hidden: int = 64):
    """Build a simple NN representative of FIML beta correction."""
    if not _HAS_TORCH:
        return None
    return nn.Sequential(
        nn.Linear(n_features, hidden),
        nn.Tanh(),
        nn.Linear(hidden, hidden),
        nn.Tanh(),
        nn.Linear(hidden, 1),
        nn.Softplus(),
    )


def run_coupling_benchmark(
    n_points: int = 10000,
    n_features: int = 5,
    n_repeats: int = 100,
) -> List[BenchmarkRow]:
    """
    Run the full coupling latency benchmark.

    Returns list of BenchmarkRow sorted by latency.
    """
    results = []
    features_np = np.random.randn(n_points, n_features).astype(np.float32)

    # Reference: raw C++ estimate (from user benchmark data)
    RAW_CPP_US = 0.00168

    # --- Method 1: NumPy matmul (sklearn-equivalent) ---
    W1 = np.random.randn(n_features, 64).astype(np.float32)
    b1 = np.random.randn(64).astype(np.float32)
    W2 = np.random.randn(64, 64).astype(np.float32)
    b2 = np.random.randn(64).astype(np.float32)
    W3 = np.random.randn(64, 1).astype(np.float32)
    b3 = np.random.randn(1).astype(np.float32)

    # Warmup
    for _ in range(5):
        h = np.maximum(features_np @ W1 + b1, 0)
        h = np.maximum(h @ W2 + b2, 0)
        _ = h @ W3 + b3

    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        h = np.maximum(features_np @ W1 + b1, 0)
        h = np.maximum(h @ W2 + b2, 0)
        _ = h @ W3 + b3
        times.append(time.perf_counter() - t0)

    numpy_us = np.median(times) * 1e6
    results.append(BenchmarkRow(
        method="NumPy matmul",
        latency_us=numpy_us,
        python_multiplier=numpy_us / max(RAW_CPP_US, 1e-15),
        cpp_multiplier=numpy_us / 2.55,  # vs PyTorch C++ API
        notes="sklearn-equivalent, no framework overhead",
    ))

    if not _HAS_TORCH:
        # Can't benchmark PyTorch methods
        results.append(BenchmarkRow(
            method="(PyTorch not available)",
            latency_us=0, python_multiplier=0, cpp_multiplier=0,
            notes="Install torch for full benchmark",
        ))
        return results

    model = _build_simple_nn(n_features)
    model.eval()
    features_t = torch.tensor(features_np)

    # --- Method 2: Python torch.no_grad batch ---
    for _ in range(5):
        with torch.no_grad():
            _ = model(features_t)

    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(features_t)
        times.append(time.perf_counter() - t0)

    nograd_us = np.median(times) * 1e6
    results.append(BenchmarkRow(
        method="PyTorch torch.no_grad",
        latency_us=nograd_us,
        python_multiplier=nograd_us / max(RAW_CPP_US, 1e-15),
        cpp_multiplier=nograd_us / 2.55,
        notes="Standard Python inference",
    ))

    # --- Method 3: torch.inference_mode (faster than no_grad) ---
    for _ in range(5):
        with torch.inference_mode():
            _ = model(features_t)

    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        with torch.inference_mode():
            _ = model(features_t)
        times.append(time.perf_counter() - t0)

    infmode_us = np.median(times) * 1e6
    results.append(BenchmarkRow(
        method="PyTorch inference_mode",
        latency_us=infmode_us,
        python_multiplier=infmode_us / max(RAW_CPP_US, 1e-15),
        cpp_multiplier=infmode_us / 2.55,
        notes="Disables view tracking + version counters",
    ))

    # --- Method 4: TorchScript traced ---
    traced = torch.jit.trace(model, features_t)
    traced.eval()
    for _ in range(5):
        with torch.inference_mode():
            _ = traced(features_t)

    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        with torch.inference_mode():
            _ = traced(features_t)
        times.append(time.perf_counter() - t0)

    ts_us = np.median(times) * 1e6
    results.append(BenchmarkRow(
        method="TorchScript traced",
        latency_us=ts_us,
        python_multiplier=ts_us / max(RAW_CPP_US, 1e-15),
        cpp_multiplier=ts_us / 2.55,
        notes="JIT-compiled, reduced interpreter overhead",
    ))

    # --- Method 5: TorchScript + zero-copy buffer ---
    from scripts.ml_augmentation.native_inference import SharedMemoryBuffer
    buffer = SharedMemoryBuffer(n_points, n_features)
    for _ in range(5):
        inp = buffer.load_features(features_np)
        with torch.inference_mode():
            _ = traced(inp)

    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        inp = buffer.load_features(features_np)
        with torch.inference_mode():
            _ = traced(inp)
        times.append(time.perf_counter() - t0)

    zc_us = np.median(times) * 1e6
    results.append(BenchmarkRow(
        method="TorchScript + zero-copy",
        latency_us=zc_us,
        python_multiplier=zc_us / max(RAW_CPP_US, 1e-15),
        cpp_multiplier=zc_us / 2.55,
        notes="Pre-allocated buffer, no allocation per call",
    ))

    # --- Reference rows (from user's benchmark data) ---
    results.append(BenchmarkRow(
        method="PyTorch C++ API (ref)",
        latency_us=2.55,
        python_multiplier=2.55 / max(RAW_CPP_US, 1e-15),
        cpp_multiplier=1.0,
        notes="libtorch native (generated .cpp)",
    ))
    results.append(BenchmarkRow(
        method="Raw C++ native (ref)",
        latency_us=RAW_CPP_US,
        python_multiplier=1.0,
        cpp_multiplier=RAW_CPP_US / 2.55,
        notes="Theoretical lower bound",
    ))

    # Sort by latency
    results.sort(key=lambda r: r.latency_us)
    return results


def format_benchmark_table(results: List[BenchmarkRow]) -> str:
    """Format results as a markdown-style table."""
    lines = [
        "",
        "ML-Solver Coupling Latency Benchmark",
        "=" * 90,
        f"{'Method':<30} {'Latency (us)':<15} {'vs Raw C++':<15} {'vs libtorch':<15} {'Notes'}",
        "-" * 90,
    ]
    for r in results:
        lines.append(
            f"{r.method:<30} {r.latency_us:<15.3f} "
            f"{r.python_multiplier:<15.1f}x "
            f"{r.cpp_multiplier:<15.2f}x "
            f"{r.notes}"
        )
    lines.append("-" * 90)
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ML-Solver coupling benchmark")
    parser.add_argument("--n-points", type=int, default=10000)
    parser.add_argument("--n-features", type=int, default=5)
    parser.add_argument("--n-repeats", type=int, default=100)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    print(f"\nBenchmarking with {args.n_points} points, {args.n_repeats} repeats...")
    results = run_coupling_benchmark(
        n_points=args.n_points,
        n_features=args.n_features,
        n_repeats=args.n_repeats,
    )
    print(format_benchmark_table(results))
