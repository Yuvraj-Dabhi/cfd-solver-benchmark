#!/usr/bin/env python3
"""
GPU & CPU Diagnostics — Hardware Detection and Acceleration Strategy
====================================================================

Detects available hardware (GPU, CPU cores, MPI), benchmarks performance,
and reports the optimal parallelization strategy for SU2 simulations.

This project uses TWO acceleration paths:
  1. SU2 solver: OpenMP threads (-t N) on CPU cores
  2. Pre/post-processing: NumPy on CPU (GPU only benefits at >10M elements)
"""

import os
import sys
import time
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Optional, Tuple


def detect_gpu() -> Dict:
    """Detect NVIDIA GPU and CUDA capabilities."""
    gpu_info = {
        "available": False,
        "name": None,
        "vram_gb": 0.0,
        "compute_capability": None,
        "cuda_driver": None,
        "cupy_available": False,
    }

    # Check nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(", ")
            gpu_info["available"] = True
            gpu_info["name"] = parts[0].strip() if len(parts) > 0 else "Unknown"
            gpu_info["vram_gb"] = float(parts[1]) / 1024 if len(parts) > 1 else 0
            gpu_info["cuda_driver"] = parts[2].strip() if len(parts) > 2 else None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Check CuPy
    try:
        import cupy as cp
        gpu_info["cupy_available"] = True
        d = cp.cuda.Device(0)
        gpu_info["compute_capability"] = str(d.compute_capability)
        free, total = cp.cuda.runtime.memGetInfo()
        gpu_info["vram_gb"] = total / (1024**3)
    except Exception:
        pass

    return gpu_info


def detect_cpu() -> Dict:
    """Detect CPU capabilities."""
    return {
        "cores": os.cpu_count() or 1,
        "recommended_threads": max(1, (os.cpu_count() or 1) - 1),
    }


def detect_su2() -> Dict:
    """Detect SU2 installation and capabilities."""
    su2_info = {
        "found": False,
        "path": None,
        "version": None,
        "openmp": False,
        "mpi": False,
    }

    su2_path = shutil.which("SU2_CFD") or shutil.which("SU2_CFD.exe")
    if not su2_path:
        # Check SU2_RUN env
        su2_run = os.environ.get("SU2_RUN", "")
        if su2_run:
            candidate = Path(su2_run) / "SU2_CFD.exe"
            if candidate.exists():
                su2_path = str(candidate)

    if not su2_path:
        return su2_info

    su2_info["found"] = True
    su2_info["path"] = su2_path

    # Get version and capabilities
    try:
        result = subprocess.run(
            [su2_path, "--help"],
            capture_output=True, text=True, timeout=10
        )
        output = result.stdout + result.stderr
        # Extract version
        for line in output.split("\n"):
            if "SU2 v" in line:
                su2_info["version"] = line.strip()
                break
        # Check for OpenMP support
        if "--threads" in output or "-t" in output:
            su2_info["openmp"] = True
    except Exception:
        pass

    # Check MPI
    mpi_path = shutil.which("mpiexec") or shutil.which("mpirun")
    su2_info["mpi"] = mpi_path is not None

    return su2_info


def recommend_strategy(gpu: Dict, cpu: Dict, su2: Dict) -> Dict:
    """
    Recommend optimal parallelization strategy.

    For SU2 CFD with an MX450-class GPU:
    - The GPU is too small for direct CFD acceleration
    - OpenMP threads on CPU cores is the best approach
    - MPI can be used if available for domain decomposition
    """
    strategy = {
        "solver_method": "serial",
        "n_threads": 1,
        "n_mpi_procs": 1,
        "preprocessing": "numpy",
        "postprocessing": "numpy",
        "rationale": [],
    }

    # SU2 solver strategy
    if su2.get("openmp"):
        # Use OpenMP threads — best for single-machine parallelism
        n_threads = cpu["recommended_threads"]
        strategy["solver_method"] = "openmp"
        strategy["n_threads"] = n_threads
        strategy["rationale"].append(
            f"SU2 OpenMP: {n_threads} threads on {cpu['cores']}-core CPU "
            f"(1 core reserved for OS)"
        )
    elif su2.get("mpi"):
        # Fallback to MPI
        n_procs = min(4, cpu["cores"])
        strategy["solver_method"] = "mpi"
        strategy["n_mpi_procs"] = n_procs
        strategy["rationale"].append(
            f"SU2 MPI: {n_procs} processes on {cpu['cores']}-core CPU"
        )

    # Pre/post-processing strategy
    if gpu.get("available") and gpu.get("vram_gb", 0) >= 4.0:
        strategy["preprocessing"] = "cupy"
        strategy["postprocessing"] = "cupy"
        strategy["rationale"].append(
            f"GPU ({gpu['name']}, {gpu['vram_gb']:.0f}GB): Used for "
            f"preprocessing/postprocessing"
        )
    else:
        reason = "NumPy on CPU"
        if gpu.get("available"):
            reason += (
                f" (GPU {gpu.get('name', '?')} has only "
                f"{gpu.get('vram_gb', 0):.1f}GB VRAM -- "
                f"CPU is faster for these problem sizes)"
            )
        strategy["rationale"].append(reason)

    return strategy


def run_diagnostics(verbose: bool = True) -> Dict:
    """Run full hardware diagnostics and report optimal strategy."""
    if verbose:
        print("=" * 70)
        print("  HARDWARE DIAGNOSTICS & ACCELERATION STRATEGY")
        print("=" * 70)

    gpu = detect_gpu()
    cpu = detect_cpu()
    su2 = detect_su2()
    strategy = recommend_strategy(gpu, cpu, su2)

    if verbose:
        # GPU
        print(f"\n  GPU:")
        if gpu["available"]:
            print(f"    Device:     {gpu['name']}")
            print(f"    VRAM:       {gpu['vram_gb']:.1f} GB")
            print(f"    CUDA:       Driver {gpu.get('cuda_driver', 'N/A')}")
            print(f"    Compute:    {gpu.get('compute_capability', 'N/A')}")
            print(f"    CuPy:       {'[Y]' if gpu['cupy_available'] else '[N]'}")
        else:
            print("    Not detected")

        # CPU
        print(f"\n  CPU:")
        print(f"    Cores:      {cpu['cores']}")
        print(f"    Threads:    {cpu['recommended_threads']} (recommended)")

        # SU2
        print(f"\n  SU2:")
        if su2["found"]:
            print(f"    Path:       {su2['path']}")
            print(f"    Version:    {su2.get('version', 'Unknown')}")
            print(f"    OpenMP:     {'[Y]' if su2['openmp'] else '[N]'}")
            print(f"    MPI:        {'[Y]' if su2['mpi'] else '[N]'}")
        else:
            print("    Not found on PATH")

        # Strategy
        print(f"\n  {'-' * 60}")
        print(f"  RECOMMENDED STRATEGY:")
        print(f"  {'-' * 60}")
        for r in strategy["rationale"]:
            print(f"    * {r}")

        # SU2 command
        if strategy["solver_method"] == "openmp":
            print(f"\n  SU2 command:")
            print(f"    SU2_CFD -t {strategy['n_threads']} naca0012.cfg")
        elif strategy["solver_method"] == "mpi":
            print(f"\n  SU2 command:")
            print(f"    mpiexec -np {strategy['n_mpi_procs']} SU2_CFD naca0012.cfg")

        print()

    return {
        "gpu": gpu,
        "cpu": cpu,
        "su2": su2,
        "strategy": strategy,
    }


if __name__ == "__main__":
    run_diagnostics()
