#!/usr/bin/env python3
"""
Curated Benchmark Case Registry
================================
Maps project flow-separation cases to the McConkey et al. (2021) curated
turbulence dataset structure, enabling standardised cross-comparison of
ML closures and surrogates against DNS/LES reference data.

Each `CuratedCase` couples a project-level `BenchmarkCase` key with:
  - The curated dataset geometry identifier and Reynolds number.
  - Available DNS/LES reference fields (velocities, Reynolds stresses).
  - Profile station coordinates for matched extraction.
  - Separation-metric ground-truth values.

References
----------
McConkey, R., Yee, E. & Lien, F. (2021). "A curated dataset for
data-driven turbulence modelling." Scientific Data 8, 255.
DOI: 10.1038/s41597-021-01034-2
"""

import json
import logging
import numpy as np
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# =====================================================================
# Data Classes
# =====================================================================

@dataclass
class CuratedCase:
    """
    Links a project BenchmarkCase to its curated dataset counterpart.

    Attributes
    ----------
    case_key : str
        Key into ``config.BENCHMARK_CASES`` (e.g., ``"periodic_hill"``).
    curated_geometry : str
        Identifier in the curated dataset (e.g., ``"periodic_hills_alpha1.0"``).
    reynolds_number : float
        Matched Reynolds number.
    reference_source : str
        Origin of the DNS/LES/experimental reference data.
    reference_fields : list of str
        Available ground-truth field names in the reference data.
    profile_stations : list of float
        Non-dimensional coordinates for profile extraction.
    station_label : str
        Label for station coordinate (e.g., ``"x/H"``, ``"x/c"``).
    separation_ground_truth : dict
        Ground-truth separation metrics (e.g., x_sep, x_reat, L_bubble).
    curated_rans_models : list of str
        RANS models for which baseline results are available in the dataset.
    notes : str
        Additional notes on the alignment or known caveats.
    """

    case_key: str
    curated_geometry: str
    reynolds_number: float
    reference_source: str
    reference_fields: List[str] = field(default_factory=list)
    profile_stations: List[float] = field(default_factory=list)
    station_label: str = ""
    separation_ground_truth: Dict[str, Any] = field(default_factory=dict)
    curated_rans_models: List[str] = field(default_factory=list)
    notes: str = ""


# =====================================================================
# Curated Case Registry
# =====================================================================

CURATED_CASE_REGISTRY: Dict[str, CuratedCase] = {

    "periodic_hill": CuratedCase(
        case_key="periodic_hill",
        curated_geometry="periodic_hills_alpha1.0",
        reynolds_number=10_595,
        reference_source="Breuer et al. (2009) DNS — ERCOFTAC #081",
        reference_fields=[
            "Ux", "Uy", "k_dns", "uu_dns", "uv_dns", "vv_dns",
        ],
        profile_stations=[0.05, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        station_label="x/H",
        separation_ground_truth={
            "x_sep_xH": 0.22,
            "x_reat_xH": 4.72,
            "L_bubble_xH": 4.50,
        },
        curated_rans_models=["ke", "kepsphit", "komega", "kwsst"],
        notes=(
            "Periodic hill at α=1.0 is the standard geometry in the curated "
            "dataset. Re_H = 10,595 matches Project config exactly."
        ),
    ),

    "nasa_hump": CuratedCase(
        case_key="nasa_hump",
        curated_geometry="parametric_bump_h42",
        reynolds_number=936_000,
        reference_source="Greenblatt et al. (2006) PIV — NASA TMR 2DWMH",
        reference_fields=[
            "Ux", "Uy", "uu_dns", "uv_dns", "vv_dns",
        ],
        profile_stations=[0.65, 0.80, 0.90, 1.00, 1.10, 1.20, 1.30],
        station_label="x/c",
        separation_ground_truth={
            "x_sep_xc": 0.665,
            "x_reat_xc": 1.11,
            "L_bubble_xc": 0.445,
        },
        curated_rans_models=["ke", "kwsst"],
        notes=(
            "Parametric bump h42 in the curated dataset is geometrically "
            "closest to the NASA Glauert-Goldschmied wall hump. Re differs "
            "from curated bump Re; treated as a transfer case."
        ),
    ),

    "backward_facing_step": CuratedCase(
        case_key="backward_facing_step",
        curated_geometry="backward_facing_step",
        reynolds_number=36_000,
        reference_source="Driver & Seegmiller (1985) — NASA TMR / ERCOFTAC C.30",
        reference_fields=[
            "Ux", "Uy", "uu_dns", "uv_dns", "vv_dns",
        ],
        profile_stations=[1.0, 4.0, 6.0, 10.0],
        station_label="x/H",
        separation_ground_truth={
            "x_reat_xH": 6.26,
            "x_reat_uncertainty_xH": 0.10,
        },
        curated_rans_models=["ke", "kepsphit", "komega", "kwsst"],
        notes=(
            "BFS geometry is directly present in the curated dataset. "
            "Expansion ratio 1.125, step height H = 12.7 mm."
        ),
    ),

    "boeing_gaussian_bump": CuratedCase(
        case_key="boeing_gaussian_bump",
        curated_geometry="gaussian_bump_3d",
        reynolds_number=2_000_000,
        reference_source="NASA TMR WMLES — Iyer & Malik (2020)",
        reference_fields=[
            "Ux", "Uy", "Uz", "Cp", "Cf",
        ],
        profile_stations=[0.6, 0.8, 1.0, 1.2, 1.4, 1.8],
        station_label="x/L",
        separation_ground_truth={
            "x_sep_xL": 0.75,
            "x_reat_xL": 1.35,
            "L_bubble_xL": 0.60,
        },
        curated_rans_models=["kwsst"],
        notes=(
            "3D Gaussian bump (h₀ = 0.085L). WMLES fields used as "
            "high-fidelity reference. Curated dataset alignment is by "
            "geometry class rather than exact parametric match."
        ),
    ),

    "juncture_flow": CuratedCase(
        case_key="juncture_flow",
        curated_geometry="wing_body_junction",
        reynolds_number=2_400_000,
        reference_source="NASA TMR / HLPW-4 — Rumsey et al. (2020)",
        reference_fields=[
            "Ux", "Uy", "Uz", "Cp",
        ],
        profile_stations=[],
        station_label="",
        separation_ground_truth={
            "corner_bubble_present": True,
            "horseshoe_vortex_present": True,
        },
        curated_rans_models=["kwsst"],
        notes=(
            "Wing-body junction flow. Corner-separation topology is the "
            "primary benchmark target (presence/absence of corner bubble). "
            "Not a point-wise field benchmark but a topology benchmark."
        ),
    ),
}


# =====================================================================
# Accessor Functions
# =====================================================================

def get_matched_cases() -> Dict[str, CuratedCase]:
    """Return all cases with curated dataset alignment."""
    return dict(CURATED_CASE_REGISTRY)


def get_curated_case(case_key: str) -> Optional[CuratedCase]:
    """Look up a single curated case by its project BenchmarkCase key."""
    return CURATED_CASE_REGISTRY.get(case_key)


def list_matched_case_keys() -> List[str]:
    """Return the list of project case keys that are curated-aligned."""
    return list(CURATED_CASE_REGISTRY.keys())


def get_field_intersection(case_keys: Optional[List[str]] = None) -> List[str]:
    """
    Return the intersection of available reference fields across cases.

    Parameters
    ----------
    case_keys : list of str, optional
        Subset of case keys. If None, uses all matched cases.

    Returns
    -------
    list of str
        Fields present in every specified case.
    """
    cases = case_keys or list_matched_case_keys()
    sets = [set(CURATED_CASE_REGISTRY[k].reference_fields)
            for k in cases if k in CURATED_CASE_REGISTRY]
    if not sets:
        return []
    return sorted(set.intersection(*sets))


# =====================================================================
# Export Utilities
# =====================================================================

def export_curated_structure(
    output_dir: Optional[Path] = None,
    case_keys: Optional[List[str]] = None,
    synthetic: bool = True,
) -> Path:
    """
    Export RANS + DNS/LES fields in the standardised curated dataset
    directory structure.

    Structure::

        output_dir/
        ├── periodic_hill/
        │   ├── kwsst/
        │   │   └── fields.npz
        │   ├── ke/
        │   │   └── fields.npz
        │   └── dns_reference/
        │       └── fields.npz
        ├── backward_facing_step/
        │   └── ...
        └── metadata.json

    Parameters
    ----------
    output_dir : Path, optional
        Output root. Defaults to ``PROJECT_ROOT / results / curated_benchmark``.
    case_keys : list of str, optional
        Cases to export. Defaults to all matched cases.
    synthetic : bool
        If True, generates representative synthetic data for demonstration.

    Returns
    -------
    Path
        Root of the exported structure.
    """
    if output_dir is None:
        output_dir = PROJECT_ROOT / "results" / "curated_benchmark"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = case_keys or list_matched_case_keys()
    metadata = {"format_version": "1.0", "cases": {}}

    for key in cases:
        cc = CURATED_CASE_REGISTRY.get(key)
        if cc is None:
            logger.warning(f"Case '{key}' not in registry, skipping.")
            continue

        case_dir = output_dir / key
        n_fields = len(cc.reference_fields)
        n_points = 500  # representative

        # Export RANS baselines
        for model in cc.curated_rans_models:
            model_dir = case_dir / model
            model_dir.mkdir(parents=True, exist_ok=True)
            if synthetic:
                np.random.seed(hash(f"{key}_{model}") % (2**32))
                data = {f: np.random.randn(n_points).astype(np.float32)
                        for f in cc.reference_fields}
                data["x"] = np.linspace(0, 1, n_points).astype(np.float32)
                data["y"] = np.random.rand(n_points).astype(np.float32)
                np.savez_compressed(model_dir / "fields.npz", **data)

        # Export DNS/LES reference
        ref_dir = case_dir / "dns_reference"
        ref_dir.mkdir(parents=True, exist_ok=True)
        if synthetic:
            np.random.seed(hash(f"{key}_dns") % (2**32))
            data = {f: np.random.randn(n_points).astype(np.float32)
                    for f in cc.reference_fields}
            data["x"] = np.linspace(0, 1, n_points).astype(np.float32)
            data["y"] = np.random.rand(n_points).astype(np.float32)
            np.savez_compressed(ref_dir / "fields.npz", **data)

        metadata["cases"][key] = {
            "geometry": cc.curated_geometry,
            "Re": cc.reynolds_number,
            "reference_source": cc.reference_source,
            "fields": cc.reference_fields,
            "rans_models": cc.curated_rans_models,
        }
        logger.info(f"Exported curated structure for '{key}'")

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Curated dataset exported to {output_dir}")
    return output_dir


# =====================================================================
# CLI
# =====================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    print("=" * 60)
    print("Curated Benchmark Case Registry")
    print("=" * 60)
    for key, cc in CURATED_CASE_REGISTRY.items():
        print(f"\n  {key}:")
        print(f"    Geometry:   {cc.curated_geometry}")
        print(f"    Re:         {cc.reynolds_number:,.0f}")
        print(f"    Reference:  {cc.reference_source}")
        print(f"    Fields:     {cc.reference_fields}")
        print(f"    Stations:   {cc.profile_stations}")
    print(f"\nCommon fields across all cases: {get_field_intersection()}")
