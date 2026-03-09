"""
Performance Profiling Utilities
================================
Tools for profiling and benchmarking the data-processing pipeline.

Features:
  - @profile_function decorator (cProfile-based)
  - benchmark_vectorization() — compares loop vs vectorized performance
  - memory_usage() — tracemalloc-based peak memory tracking

Usage:
    from scripts.utils.profiling import profile_function, benchmark_vectorization

    @profile_function
    def expensive_computation(data):
        ...

    benchmark_vectorization(n_points=100000)
"""

import cProfile
import functools
import io
import logging
import pstats
import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ProfileResult:
    """Result from profiling a function."""
    function_name: str
    wall_time_s: float
    cpu_time_s: float
    n_calls: int
    top_functions: str  # Top-10 cumulative time summary


@dataclass
class BenchmarkResult:
    """Result from vectorization benchmark."""
    name: str
    loop_time_s: float
    vectorized_time_s: float
    speedup: float
    n_points: int
    results_match: bool


def profile_function(func: Optional[Callable] = None, *, top_n: int = 10):
    """
    Decorator to profile a function with cProfile.

    Usage:
        @profile_function
        def my_func(x, y):
            ...

        @profile_function(top_n=20)
        def my_func(x, y):
            ...
    """
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            profiler = cProfile.Profile()
            t0 = time.perf_counter()

            try:
                profiler.enable()
                result = fn(*args, **kwargs)
                profiler.disable()
                wall_time = time.perf_counter() - t0

                # Extract stats
                stream = io.StringIO()
                stats = pstats.Stats(profiler, stream=stream)
                stats.sort_stats("cumulative")
                stats.print_stats(top_n)

                profile_result = ProfileResult(
                    function_name=fn.__name__,
                    wall_time_s=wall_time,
                    cpu_time_s=stats.total_tt,
                    n_calls=stats.total_calls,
                    top_functions=stream.getvalue(),
                )
            except ValueError:
                # Another profiler is active (e.g. pytest-cov) — fallback
                result = fn(*args, **kwargs)
                wall_time = time.perf_counter() - t0
                profile_result = ProfileResult(
                    function_name=fn.__name__,
                    wall_time_s=wall_time,
                    cpu_time_s=wall_time,
                    n_calls=1,
                    top_functions="(profiling skipped: another profiler active)",
                )

            logger.info(
                f"PROFILE {fn.__name__}: {profile_result.wall_time_s:.4f}s wall"
            )
            wrapper._last_profile = profile_result
            return result

        wrapper._last_profile = None
        return wrapper

    if func is not None:
        return decorator(func)
    return decorator


@contextmanager
def timer(label: str = ""):
    """Context manager for timing code blocks."""
    t0 = time.perf_counter()
    yield
    elapsed = time.perf_counter() - t0
    logger.info(f"TIMER {label}: {elapsed:.4f}s")


@contextmanager
def memory_tracker(label: str = ""):
    """Context manager for tracking peak memory usage."""
    tracemalloc.start()
    yield
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    logger.info(
        f"MEMORY {label}: current={current / 1024:.1f} KB, "
        f"peak={peak / 1024:.1f} KB"
    )


def benchmark_vectorization(n_points: int = 10000) -> Dict[str, BenchmarkResult]:
    """
    Benchmark loop-based vs vectorized implementations.

    Tests key operations from the framework:
    1. Tensor contraction (P/epsilon computation)
    2. Anisotropy invariants (Lumley triangle)
    3. Zero-crossing detection (separation point)

    Returns dict of BenchmarkResult.
    """
    results = {}

    # --- 1. Tensor contraction ---
    tau = np.random.randn(n_points, 3, 3)
    S = np.random.randn(n_points, 3, 3)

    # Loop version
    t0 = time.perf_counter()
    P_loop = np.zeros(n_points)
    for i in range(n_points):
        P_loop[i] = -np.sum(tau[i] * S[i])
    t_loop = time.perf_counter() - t0

    # Vectorized version
    t0 = time.perf_counter()
    P_vec = -np.einsum('nij,nij->n', tau, S)
    t_vec = time.perf_counter() - t0

    results["tensor_contraction"] = BenchmarkResult(
        name="Tensor contraction (P = -tau:S)",
        loop_time_s=t_loop,
        vectorized_time_s=t_vec,
        speedup=t_loop / max(t_vec, 1e-15),
        n_points=n_points,
        results_match=np.allclose(P_loop, P_vec, atol=1e-12),
    )

    # --- 2. Batch determinant ---
    A = np.random.randn(n_points, 3, 3)

    t0 = time.perf_counter()
    det_loop = np.array([np.linalg.det(A[i]) for i in range(n_points)])
    t_loop = time.perf_counter() - t0

    t0 = time.perf_counter()
    det_vec = np.linalg.det(A)
    t_vec = time.perf_counter() - t0

    results["batch_determinant"] = BenchmarkResult(
        name="Batch 3x3 determinant",
        loop_time_s=t_loop,
        vectorized_time_s=t_vec,
        speedup=t_loop / max(t_vec, 1e-15),
        n_points=n_points,
        results_match=np.allclose(det_loop, det_vec, atol=1e-10),
    )

    # --- 3. Zero-crossing detection ---
    x = np.linspace(0, 10, n_points)
    Cf = np.sin(x)

    t0 = time.perf_counter()
    sep_loop = None
    for i in range(len(Cf) - 1):
        if Cf[i] > 0 and Cf[i + 1] < 0:
            sep_loop = x[i] - Cf[i] * (x[i + 1] - x[i]) / (Cf[i + 1] - Cf[i])
            break
    t_loop = time.perf_counter() - t0

    t0 = time.perf_counter()
    sign_change = np.diff(np.sign(Cf))
    crossings = np.where(sign_change < 0)[0]
    sep_vec = None
    if len(crossings) > 0:
        i = crossings[0]
        sep_vec = x[i] - Cf[i] * (x[i + 1] - x[i]) / (Cf[i + 1] - Cf[i])
    t_vec = time.perf_counter() - t0

    results["zero_crossing"] = BenchmarkResult(
        name="Zero-crossing detection (separation point)",
        loop_time_s=t_loop,
        vectorized_time_s=t_vec,
        speedup=t_loop / max(t_vec, 1e-15),
        n_points=n_points,
        results_match=(sep_loop is None and sep_vec is None) or (
            sep_loop is not None and sep_vec is not None and
            abs(sep_loop - sep_vec) < 1e-10
        ),
    )

    return results


def print_benchmark_report(results: Dict[str, BenchmarkResult]) -> str:
    """Generate formatted benchmark report."""
    lines = [
        "Vectorization Benchmark Report",
        "=" * 65,
        f"{'Operation':<40} {'Loop (s)':<12} {'Vec (s)':<12} {'Speedup':<10} {'Match'}",
        "-" * 65,
    ]
    for key, r in results.items():
        match = "OK" if r.results_match else "FAIL"
        lines.append(
            f"{r.name:<40} {r.loop_time_s:<12.6f} "
            f"{r.vectorized_time_s:<12.6f} {r.speedup:<10.1f}x {match}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Running vectorization benchmark...")
    results = benchmark_vectorization(n_points=50000)
    print(print_benchmark_report(results))
