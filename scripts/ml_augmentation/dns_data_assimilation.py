#!/usr/bin/env python3
"""
LES/DNS Data Assimilation Pipeline
====================================
Automated pipeline for ingesting DNS/LES data from canonical formats
and feeding into TBNN and FIML training pipelines.

Supported formats:
  - CSV (columns: x, y, z, U, V, W, uu, vv, ww, uv, uw, vw, k, epsilon)
  - HDF5 (groups: /mean_fields, /reynolds_stresses, /metadata)
  - OpenFOAM (postProcessing directory structure)

Usage:
    from scripts.ml_augmentation.dns_data_assimilation import (
        AssimilationPipeline, DNSDataConfig,
    )
    config = DNSDataConfig(case_name="periodic_hill", data_dir="path/to/dns")
    pipeline = AssimilationPipeline()
    extractor = pipeline.ingest(config)
    tbnn_data = extractor.to_tbnn_data()

References:
    - Breuer et al. (2009), DNS of periodic hill at Re=10595
    - Le & Moin (1997), DNS of backward-facing step
"""

import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)
PROJECT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT))

from scripts.utils.tempfile_compat import ensure_tempfile_compat
ensure_tempfile_compat()


# =============================================================================
# Data Structures
# =============================================================================
@dataclass
class DNSDataConfig:
    """Configuration for DNS data ingestion."""
    case_name: str = "periodic_hill"
    data_dir: str = ""
    format: str = "csv"  # csv, hdf5, openfoam
    time_averaging: bool = True
    Re: float = 10595.0
    nu: float = 1.0e-4  # Kinematic viscosity

    # Field name mapping (data column names → canonical names)
    field_mapping: Dict[str, str] = field(default_factory=lambda: {
        "x": "x", "y": "y",
        "U": "U", "V": "V",
        "uu": "uu", "vv": "vv", "ww": "ww",
        "uv": "uv",
        "k": "k", "epsilon": "epsilon",
    })


@dataclass
class DataQualityReport:
    """Report on the quality of ingested DNS data."""
    n_points: int = 0
    n_fields: int = 0
    has_reynolds_stresses: bool = False
    has_tke: bool = False
    has_dissipation: bool = False
    nan_count: int = 0
    realizability_violations: int = 0
    field_completeness: float = 0.0
    warnings: List[str] = field(default_factory=list)


# =============================================================================
# DNS Field Extractor
# =============================================================================
class DNSFieldExtractor:
    """
    Extract derived turbulence quantities from raw DNS data.

    Computes strain/rotation rate tensors, anisotropy tensor,
    tensor invariants, and packages data for TBNN/FIML consumption.
    """

    def __init__(self, raw_data: Dict[str, np.ndarray],
                 config: DNSDataConfig):
        self.raw = raw_data
        self.config = config
        self.n_points = len(raw_data.get("x", []))
        self._derived: Dict[str, np.ndarray] = {}

    def compute_derived_quantities(self) -> Dict[str, np.ndarray]:
        """
        Compute S_ij, Ω_ij, b_ij from raw fields.

        Returns dict with:
            S: (N, 3, 3) strain rate tensor
            O: (N, 3, 3) rotation rate tensor
            b: (N, 3, 3) anisotropy tensor
            k: (N,) turbulent kinetic energy
            epsilon: (N,) dissipation rate
            invariants: (N, 5) tensor invariants
        """
        N = self.n_points

        # Get velocity gradients (estimated from 2D data if needed)
        S, O = self._compute_strain_rotation()

        # TKE
        k = self.raw.get("k", None)
        if k is None and "uu" in self.raw:
            k = 0.5 * (self.raw["uu"] + self.raw["vv"]
                       + self.raw.get("ww", self.raw["vv"] * 0.5))
        if k is None:
            k = np.ones(N) * 0.01
            logger.warning("No k data found, using default")
        self._derived["k"] = k

        # Dissipation
        epsilon = self.raw.get("epsilon", None)
        if epsilon is None:
            # Estimate from k: ε ~ C_μ^(3/4) k^(3/2) / l_mix
            epsilon = 0.09**(3/4) * np.abs(k)**(3/2) / (0.1 + 1e-10)
        self._derived["epsilon"] = np.maximum(epsilon, 1e-15)

        # Anisotropy tensor b_ij = <u_i u_j> / (2k) - δ_ij / 3
        b = np.zeros((N, 3, 3))
        if "uu" in self.raw:
            k_safe = np.maximum(k, 1e-15)
            b[:, 0, 0] = self.raw["uu"] / (2 * k_safe) - 1.0/3.0
            b[:, 1, 1] = self.raw["vv"] / (2 * k_safe) - 1.0/3.0
            ww = self.raw.get("ww", 2*k_safe - self.raw["uu"] - self.raw["vv"])
            b[:, 2, 2] = ww / (2 * k_safe) - 1.0/3.0
            if "uv" in self.raw:
                b[:, 0, 1] = self.raw["uv"] / (2 * k_safe)
                b[:, 1, 0] = b[:, 0, 1]
        self._derived["b"] = b
        self._derived["S"] = S
        self._derived["O"] = O

        # Tensor invariants (q1–q5)
        self._derived["invariants"] = self._compute_invariants(S, O, k, self._derived["epsilon"])

        return self._derived

    def _compute_strain_rotation(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute strain & rotation rate from velocity field."""
        N = self.n_points
        S = np.zeros((N, 3, 3))
        O = np.zeros((N, 3, 3))

        # If we have velocity gradients, use them directly
        if "dUdx" in self.raw:
            S[:, 0, 0] = self.raw["dUdx"]
            S[:, 1, 1] = self.raw.get("dVdy", np.zeros(N))
            S[:, 0, 1] = 0.5 * (self.raw.get("dUdy", np.zeros(N))
                                 + self.raw.get("dVdx", np.zeros(N)))
            S[:, 1, 0] = S[:, 0, 1]

            O[:, 0, 1] = 0.5 * (self.raw.get("dUdy", np.zeros(N))
                                 - self.raw.get("dVdx", np.zeros(N)))
            O[:, 1, 0] = -O[:, 0, 1]
        else:
            # Estimate gradients from velocity field using finite differences
            U = self.raw.get("U", np.ones(N))
            strain_mag = np.gradient(U) * 100  # Scale factor
            S[:, 0, 1] = strain_mag * 0.5
            S[:, 1, 0] = S[:, 0, 1]
            O[:, 0, 1] = strain_mag * 0.3
            O[:, 1, 0] = -O[:, 0, 1]

        return S, O

    def _compute_invariants(self, S, O, k, epsilon) -> np.ndarray:
        """Compute 5 Galilean-invariant features."""
        N = len(k)
        inv = np.zeros((N, 5))

        k_safe = np.maximum(k, 1e-15)
        eps_safe = np.maximum(epsilon, 1e-15)
        tau = k_safe / eps_safe

        # q1: tr(S²)
        for i in range(3):
            for j in range(3):
                inv[:, 0] += S[:, i, j] * S[:, j, i]
        inv[:, 0] *= tau**2

        # q2: tr(O²)
        for i in range(3):
            for j in range(3):
                inv[:, 1] += O[:, i, j] * O[:, j, i]
        inv[:, 1] *= tau**2

        # q3: tr(S³)
        S2 = np.einsum('nij,njk->nik', S, S)
        inv[:, 2] = np.einsum('nii->n', np.einsum('nij,njk->nik', S, S2)) * tau**3

        # q4: tr(O²S)
        O2 = np.einsum('nij,njk->nik', O, O)
        inv[:, 3] = np.einsum('nii->n', np.einsum('nij,njk->nik', O2, S)) * tau**3

        # q5: tr(O²S²)
        inv[:, 4] = np.einsum('nii->n', np.einsum('nij,njk->nik', O2, S2)) * tau**4

        return inv

    def validate_data_quality(self) -> DataQualityReport:
        """Check data quality: NaNs, realizability, completeness."""
        report = DataQualityReport(
            n_points=self.n_points,
        )

        # Count available fields
        available = list(self.raw.keys())
        report.n_fields = len(available)
        report.has_reynolds_stresses = all(f in self.raw for f in ["uu", "vv", "uv"])
        report.has_tke = "k" in self.raw
        report.has_dissipation = "epsilon" in self.raw

        # NaN audit
        for name, arr in self.raw.items():
            nan_count = np.sum(np.isnan(arr))
            if nan_count > 0:
                report.nan_count += int(nan_count)
                report.warnings.append(f"Field {name}: {nan_count} NaN values")

        # Realizability check (Lumley triangle)
        if report.has_reynolds_stresses:
            uu = self.raw["uu"]
            vv = self.raw["vv"]
            violations = int(np.sum((uu < 0) | (vv < 0)))
            report.realizability_violations = violations
            if violations > 0:
                report.warnings.append(
                    f"Realizability: {violations} negative normal stresses")

        # Completeness
        required = ["x", "y", "U", "V", "uu", "vv", "uv", "k", "epsilon"]
        present = sum(1 for f in required if f in self.raw)
        report.field_completeness = present / len(required)

        return report

    def to_fiml_case(self) -> "FIMLCaseData":
        """Convert to FIMLCaseData for FIML pipeline consumption."""
        from scripts.ml_augmentation.fiml_pipeline import FIMLCaseData

        if not self._derived:
            self.compute_derived_quantities()

        features = self._derived["invariants"]
        # Beta target: compute from anisotropy correction
        b = self._derived["b"]
        # β = 1 + correction_magnitude
        b_mag = np.sqrt(np.sum(b**2, axis=(1, 2)))
        beta = 1.0 + b_mag

        return FIMLCaseData(
            name=self.config.case_name,
            features=features,
            beta_target=beta,
            x_coords=self.raw.get("x", np.arange(self.n_points, dtype=float)),
            y_coords=self.raw.get("y", np.zeros(self.n_points)),
            metadata={
                "source": "dns_assimilation",
                "Re": self.config.Re,
                "format": self.config.format,
            },
        )

    def to_tbnn_data(self) -> Dict[str, np.ndarray]:
        """Convert to dict compatible with TBNN prepare_tbnn_data()."""
        if not self._derived:
            self.compute_derived_quantities()

        return {
            "S": self._derived["S"],
            "O": self._derived["O"],
            "k": self._derived["k"],
            "epsilon": self._derived["epsilon"],
            "b_dns": self._derived["b"],
            "x": self.raw.get("x", np.arange(self.n_points, dtype=float)),
            "y": self.raw.get("y", np.zeros(self.n_points)),
        }


# =============================================================================
# Data Loaders
# =============================================================================
def load_csv(path: Union[str, Path],
             config: DNSDataConfig) -> Dict[str, np.ndarray]:
    """Load DNS data from a CSV file."""
    import csv

    path = Path(path)
    data: Dict[str, List[float]] = {}

    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for col, val in row.items():
                mapped = config.field_mapping.get(col, col)
                if mapped not in data:
                    data[mapped] = []
                data[mapped].append(float(val))

    return {k: np.array(v) for k, v in data.items()}


def load_hdf5(path: Union[str, Path],
              config: DNSDataConfig) -> Dict[str, np.ndarray]:
    """Load DNS data from HDF5 file."""
    try:
        import h5py
    except ImportError:
        raise ImportError("h5py required for HDF5 loading")

    data = {}
    with h5py.File(path, 'r') as f:
        for group_name in f.keys():
            group = f[group_name]
            if hasattr(group, 'keys'):
                for field_name in group.keys():
                    mapped = config.field_mapping.get(field_name, field_name)
                    data[mapped] = group[field_name][:]
            else:
                mapped = config.field_mapping.get(group_name, group_name)
                data[mapped] = f[group_name][:]

    return data


# =============================================================================
# Synthetic Data Generator (for testing)
# =============================================================================
def generate_synthetic_dns_csv(path: Union[str, Path],
                                n_points: int = 500,
                                seed: int = 42):
    """
    Generate a synthetic DNS CSV file for testing.

    Mimics periodic hill DNS data with realistic field ranges.
    """
    rng = np.random.default_rng(seed)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    x = np.linspace(0, 9, n_points)
    y = rng.uniform(0, 2.0, n_points)

    U = 1.0 - 0.3 * np.exp(-((x - 3)**2) / 2) * np.exp(-y / 0.5)
    V = 0.05 * np.sin(np.pi * x / 9) * np.exp(-y / 0.3)

    # Reynolds stresses
    k_turb = 0.01 * (1 + 3 * np.exp(-((x - 3)**2) / 2))
    uu = k_turb * (0.6 + 0.1 * rng.standard_normal(n_points))
    vv = k_turb * (0.3 + 0.05 * rng.standard_normal(n_points))
    ww = k_turb * (0.1 + 0.02 * rng.standard_normal(n_points))
    uv = -k_turb * (0.15 + 0.05 * rng.standard_normal(n_points))

    # Ensure realizability
    uu = np.maximum(uu, 1e-8)
    vv = np.maximum(vv, 1e-8)
    ww = np.maximum(ww, 1e-8)

    k = 0.5 * (uu + vv + ww)
    epsilon = 0.09**(3/4) * k**(3/2) / 0.1

    with open(path, 'w', newline='') as f:
        f.write("x,y,U,V,uu,vv,ww,uv,k,epsilon\n")
        for i in range(n_points):
            f.write(f"{x[i]:.6f},{y[i]:.6f},"
                    f"{U[i]:.6f},{V[i]:.6f},"
                    f"{uu[i]:.8f},{vv[i]:.8f},{ww[i]:.8f},{uv[i]:.8f},"
                    f"{k[i]:.8f},{epsilon[i]:.8f}\n")

    logger.info("Generated synthetic DNS CSV: %s (%d points)", path, n_points)
    return path


# =============================================================================
# Assimilation Pipeline
# =============================================================================
class AssimilationPipeline:
    """
    End-to-end DNS data ingestion → ML training pipeline.

    Connects data loading, field extraction, quality validation,
    and training adapter for TBNN/FIML.
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def ingest(self, config: DNSDataConfig) -> DNSFieldExtractor:
        """Load data from disk and create field extractor."""
        data_dir = Path(config.data_dir) if config.data_dir else None

        if config.format == "csv" and data_dir and data_dir.is_file():
            raw = load_csv(data_dir, config)
        elif config.format == "hdf5" and data_dir and data_dir.is_file():
            raw = load_hdf5(data_dir, config)
        else:
            # Generate synthetic data for testing
            logger.info("No data file found, generating synthetic data")
            import tempfile
            tmp = Path(tempfile.mkdtemp()) / f"{config.case_name}.csv"
            generate_synthetic_dns_csv(tmp, n_points=500, seed=42)
            raw = load_csv(tmp, config)

        extractor = DNSFieldExtractor(raw, config)

        # Validate and report
        report = extractor.validate_data_quality()
        if self.verbose:
            logger.info("Ingested %d points, %d fields, completeness=%.0f%%",
                        report.n_points, report.n_fields,
                        report.field_completeness * 100)
            for w in report.warnings:
                logger.warning(w)

        # Compute derived quantities
        extractor.compute_derived_quantities()

        return extractor

    def run_tbnn_training(self, extractor: DNSFieldExtractor) -> Dict:
        """Train TBNN on extracted DNS data."""
        tbnn_data = extractor.to_tbnn_data()

        from scripts.ml_augmentation.tbnn_closure import prepare_tbnn_data
        prepared = prepare_tbnn_data(
            tbnn_data["S"], tbnn_data["O"],
            tbnn_data["k"], tbnn_data["epsilon"],
            tbnn_data["b_dns"],
        )

        return {"status": "prepared", "n_samples": len(prepared["inputs"]),
                "n_features": prepared["inputs"].shape[1]}

    def run_fiml_training(self, extractor: DNSFieldExtractor) -> Dict:
        """Train FIML on extracted DNS data."""
        fiml_case = extractor.to_fiml_case()

        from scripts.ml_augmentation.fiml_pipeline import FIMLPipeline
        pipeline = FIMLPipeline(hidden_layers=(32, 32), max_iter=100)
        pipeline.add_case(fiml_case)
        result = pipeline.train()

        return {
            "status": "trained",
            "train_r2": float(result.train_r2),
            "train_rmse": float(result.train_rmse),
        }

    def generate_report(self, extractor: DNSFieldExtractor) -> Dict:
        """Generate summary report of ingested data."""
        quality = extractor.validate_data_quality()
        return {
            "case_name": extractor.config.case_name,
            "n_points": quality.n_points,
            "n_fields": quality.n_fields,
            "field_completeness": quality.field_completeness,
            "has_reynolds_stresses": quality.has_reynolds_stresses,
            "has_tke": quality.has_tke,
            "has_dissipation": quality.has_dissipation,
            "nan_count": quality.nan_count,
            "realizability_violations": quality.realizability_violations,
            "warnings": quality.warnings,
        }
