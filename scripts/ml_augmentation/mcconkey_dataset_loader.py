#!/usr/bin/env python3
"""
McConkey et al. (2021) Curated Turbulence Dataset Loader
=========================================================
Loads and integrates the curated turbulence modelling dataset from:

    McConkey, R., Yee, E. & Lien, F. (2021).
    "A curated dataset for data-driven turbulence modelling."
    Scientific Data, 8(1). DOI: 10.1038/s41597-021-01034-2

    Data DOI: 10.34740/kaggle/dsv/2637500  (Kaggle)
    Metadata DOI: 10.6084/m9.figshare.15124857

Dataset summary:
    - 895,640 spatial datapoints
    - 29 cases per turbulence model
    - 4 RANS models: k-ε, k-ε-ϕt-f, k-ω, k-ω SST
    - DNS/LES reference labels
    - Geometries: periodic hills, square duct, parametric bumps,
      converging-diverging channel, curved backward-facing step

Download instructions:
    1. pip install kaggle  (if not installed)
    2. kaggle datasets download -d ryleymcconkey/ml-turbulence-dataset
    3. Unzip into: <project>/experimental_data/mcconkey_dataset/
    OR
    1. Visit https://www.kaggle.com/datasets/ryleymcconkey/ml-turbulence-dataset
    2. Download and extract to the path above

The parametric bump cases are geometrically closest to the NASA wall hump
and provide the most physically relevant training data for the FIML
correction model (fiml_correction.py).

This module is used as the primary external ML training data source,
replacing the need to generate thousands of synthetic NACA cases.

References:
    - Srivastava et al. (2024), NASA TM-20240012512 (FIML methodology)
    - Parish & Duraisamy (2016), JCP 305, 758-774 (FIML framework)
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add project root for imports
PROJECT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT))

try:
    from scripts.ml_augmentation.dataset import CFDDataset, DatasetBuilder
except ImportError:
    CFDDataset = None
    DatasetBuilder = None

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------
DEFAULT_DATA_DIR = PROJECT / "experimental_data" / "mcconkey_dataset"

# Known case names in the McConkey dataset
# Parametric bumps are the most relevant for wall hump validation
BUMP_CASES = [
    "bump_h20", "bump_h26", "bump_h31", "bump_h38", "bump_h42",
]

ALL_RANS_MODELS = ["ke", "kepsphit", "komega", "kwsst"]

# Feature columns expected in the dataset
FEATURE_COLUMNS = [
    "x", "y",              # Coordinates
    "Ux", "Uy",            # Mean velocity components
    "p",                   # Mean pressure
    "k",                   # Turbulent kinetic energy (RANS)
    "omega",               # Specific dissipation rate (if available)
    "epsilon",             # Dissipation rate (if available)
    "nut",                 # Eddy viscosity
    "grad_u_xx", "grad_u_xy", "grad_u_yx", "grad_u_yy",  # Velocity gradients
]

# DNS/LES target columns
TARGET_COLUMNS = [
    "k_dns",               # DNS/LES turbulent kinetic energy
    "uu_dns", "uv_dns", "vv_dns",  # DNS/LES Reynolds stresses
]


# ===================================================================
# Data Loading
# ===================================================================
def load_mcconkey_dataset(
    data_dir: Optional[Path] = None,
    models: Optional[List[str]] = None,
    cases: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Load the McConkey et al. (2021) curated turbulence dataset.

    Parameters
    ----------
    data_dir : Path, optional
        Directory containing the extracted dataset files.
        Default: <project>/experimental_data/mcconkey_dataset/
    models : list of str, optional
        RANS models to load. Default: all 4 models.
    cases : list of str, optional
        Case names to load. Default: all available cases.

    Returns
    -------
    dict
        Mapping from (model, case) tuple keys to DataFrames,
        e.g. {("kwsst", "bump_h20"): DataFrame, ...}

    Raises
    ------
    FileNotFoundError
        If the dataset directory does not exist (with download instructions).
    """
    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    models = models or ALL_RANS_MODELS
    cases = cases or None  # None = load all found

    if not data_dir.exists():
        raise FileNotFoundError(
            f"McConkey dataset not found at: {data_dir}\n"
            f"\n"
            f"Download instructions:\n"
            f"  Option 1 (Kaggle CLI):\n"
            f"    pip install kaggle\n"
            f"    kaggle datasets download -d ryleymcconkey/ml-turbulence-dataset\n"
            f"    Unzip to: {data_dir}\n"
            f"\n"
            f"  Option 2 (Browser):\n"
            f"    Visit: https://www.kaggle.com/datasets/ryleymcconkey/ml-turbulence-dataset\n"
            f"    Download and extract to: {data_dir}\n"
            f"\n"
            f"  Dataset DOI: 10.34740/kaggle/dsv/2637500\n"
            f"  Paper DOI: 10.1038/s41597-021-01034-2\n"
        )

    result = {}

    # Try multiple file organization patterns the dataset may use
    for model in models:
        model_dir = data_dir / model
        if not model_dir.exists():
            # Try flat CSV naming: model_case.csv
            _load_flat_csvs(data_dir, model, cases, result)
            continue

        # Subdirectory structure: model/case/data.csv
        case_dirs = sorted(model_dir.iterdir()) if model_dir.is_dir() else []
        for case_dir in case_dirs:
            if not case_dir.is_dir():
                continue
            case_name = case_dir.name

            if cases and case_name not in cases:
                continue

            # Look for CSV or parquet files
            df = _load_case_data(case_dir, model, case_name)
            if df is not None:
                result[(model, case_name)] = df
                logger.info(f"Loaded {model}/{case_name}: {len(df)} points")

    # Also try loading from a single combined CSV/parquet
    if not result:
        combined = _try_combined_file(data_dir, models, cases)
        if combined:
            result.update(combined)

    if not result:
        logger.warning(
            f"No data files found in {data_dir}. "
            f"Checked subdirectory and flat CSV patterns. "
            f"Please verify the dataset was extracted correctly."
        )

    return result


def _load_case_data(
    case_dir: Path, model: str, case_name: str
) -> Optional[pd.DataFrame]:
    """Load data from a single case directory."""
    for ext in [".csv", ".parquet", ".csv.gz"]:
        candidates = list(case_dir.glob(f"*{ext}"))
        if candidates:
            fpath = candidates[0]
            try:
                if ext == ".parquet":
                    df = pd.read_parquet(fpath)
                else:
                    df = pd.read_csv(fpath)
                df["model"] = model
                df["case"] = case_name
                return df
            except Exception as e:
                logger.warning(f"Failed to load {fpath}: {e}")
    return None


def _load_flat_csvs(
    data_dir: Path, model: str, cases: Optional[List[str]],
    result: dict
) -> None:
    """Load from flat CSV files named model_case.csv."""
    for fpath in sorted(data_dir.glob(f"{model}_*.csv")):
        case_name = fpath.stem.replace(f"{model}_", "")
        if cases and case_name not in cases:
            continue
        try:
            df = pd.read_csv(fpath)
            df["model"] = model
            df["case"] = case_name
            result[(model, case_name)] = df
            logger.info(f"Loaded {model}/{case_name}: {len(df)} points")
        except Exception as e:
            logger.warning(f"Failed to load {fpath}: {e}")


def _try_combined_file(
    data_dir: Path, models: List[str], cases: Optional[List[str]]
) -> Dict:
    """Try loading from a single combined file."""
    result = {}
    for name in ["combined.csv", "dataset.csv", "all_data.csv",
                  "combined.parquet", "dataset.parquet"]:
        fpath = data_dir / name
        if fpath.exists():
            try:
                if name.endswith(".parquet"):
                    df = pd.read_parquet(fpath)
                else:
                    df = pd.read_csv(fpath)

                # Split by model and case columns
                if "model" in df.columns and "case" in df.columns:
                    for (m, c), group in df.groupby(["model", "case"]):
                        if m in models and (cases is None or c in cases):
                            result[(m, c)] = group.reset_index(drop=True)
                    logger.info(f"Loaded combined file: {len(result)} model-case pairs")
                    return result
            except Exception as e:
                logger.warning(f"Failed to load {fpath}: {e}")
    return result


# ===================================================================
# Filtering and Feature Extraction
# ===================================================================
def extract_bump_cases(
    dataset: Dict[str, pd.DataFrame],
    bump_prefixes: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Filter dataset to include only parametric bump cases.

    These are geometrically closest to the NASA wall hump and provide
    the most physically relevant training data for FIML correction.

    Parameters
    ----------
    dataset : dict
        Output from load_mcconkey_dataset().
    bump_prefixes : list of str, optional
        Case name prefixes to match. Default: ["bump"].

    Returns
    -------
    dict
        Filtered dataset containing only bump cases.
    """
    prefixes = bump_prefixes or ["bump"]
    filtered = {}
    for (model, case), df in dataset.items():
        if any(case.startswith(p) for p in prefixes):
            filtered[(model, case)] = df
    logger.info(f"Filtered to {len(filtered)} bump cases from {len(dataset)} total")
    return filtered


def compute_invariant_features(df: pd.DataFrame) -> np.ndarray:
    """
    Compute Galilean-invariant features from flow data, aligned with
    the q1–q5 feature set from Srivastava et al. (2024).

    Parameters
    ----------
    df : DataFrame
        Must contain velocity, TKE, eddy viscosity, and gradient columns.

    Returns
    -------
    array, shape (N, 5)
        Features [q1, q2, q3, q4, q5] for each point.
    """
    # Ensure required columns exist
    required = ["Ux", "Uy", "k", "nut"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        logger.warning(f"Missing columns for invariant features: {missing}")
        return np.full((len(df), 5), np.nan)

    U_mag = np.sqrt(df["Ux"].values**2 + df["Uy"].values**2) + 1e-15
    k = np.maximum(df["k"].values, 1e-15)
    nut = np.maximum(df["nut"].values, 1e-15)

    # Molecular viscosity estimate (if not present, use typical air)
    nu = df["nu"].values if "nu" in df.columns else np.full(len(df), 1.5e-5)

    # Wall distance (if available)
    d = df["y"].values if "y" in df.columns else np.ones(len(df))
    d = np.maximum(np.abs(d), 1e-10)

    # Compute strain magnitude from gradients if available
    if all(c in df.columns for c in ["grad_u_xx", "grad_u_xy", "grad_u_yx", "grad_u_yy"]):
        S11 = df["grad_u_xx"].values
        S12 = 0.5 * (df["grad_u_xy"].values + df["grad_u_yx"].values)
        S22 = df["grad_u_yy"].values
        S_mag = np.sqrt(2.0 * (S11**2 + 2*S12**2 + S22**2) + 1e-20)

        O12 = 0.5 * (df["grad_u_xy"].values - df["grad_u_yx"].values)
    else:
        S_mag = np.ones(len(df))
        O12 = np.zeros(len(df))
        S12 = np.zeros(len(df))

    # q1: turbulence-to-mean-strain ratio
    q1 = np.clip(nut / (nu * S_mag * d + 1e-15), -10, 10)

    # q2: wall-distance Reynolds number
    q2 = np.minimum(np.sqrt(nut) * d / (nu + 1e-15) / 50.0, 2.0)

    # q3: strain-rotation ratio
    q3 = np.clip((S12 * O12) / (S_mag**2 + 1e-15), -5, 5)

    # q4: pressure-gradient alignment (simplified, requires ∇p)
    if "p" in df.columns and "grad_p_x" in df.columns:
        dp_stream = df["grad_p_x"].values  # streamwise approximation
        rho = df["rho"].values if "rho" in df.columns else np.ones(len(df)) * 1.225
        q4 = np.clip(dp_stream / (0.5 * rho * U_mag**2 + 1e-15), -10, 10)
    else:
        q4 = np.zeros(len(df))

    # q5: eddy viscosity ratio (log scale)
    q5 = np.clip(np.log10(nut / (nu + 1e-15) + 1.0), 0, 6)

    return np.column_stack([q1, q2, q3, q4, q5])


# ===================================================================
# Integration with project's CFDDataset
# ===================================================================
def to_cfd_dataset(
    data: Dict[str, pd.DataFrame],
    feature_names: Optional[List[str]] = None,
    target_names: Optional[List[str]] = None,
    dataset_name: str = "mcconkey_2021",
) -> Optional["CFDDataset"]:
    """
    Convert loaded McConkey data to the project's CFDDataset format.

    Parameters
    ----------
    data : dict
        Output from load_mcconkey_dataset() or extract_bump_cases().
    feature_names : list of str, optional
        Feature column names. Default: compute q1–q5 invariants.
    target_names : list of str, optional
        Target column names. Default: ["Ux", "Uy", "k_dns", "uu_dns", "uv_dns", "vv_dns"] if available.
    dataset_name : str
        Name for the dataset.

    Returns
    -------
    CFDDataset or None
        Dataset ready for ML training, or None if CFDDataset not available.
    """
    if CFDDataset is None:
        logger.error("Cannot import CFDDataset from dataset.py")
        return None

    all_features = []
    all_targets = []
    case_labels = []
    model_labels = []

    for (model, case), df in data.items():
        # Compute invariant features
        features = compute_invariant_features(df)

        # Get targets (DNS/LES reference + velocities for curated benchmarking)
        if target_names is not None:
            target_cols = target_names
        else:
            # Default to full state extraction for strict benchmarking
            target_cols = ["Ux", "Uy", "k_dns", "uu_dns", "uv_dns", "vv_dns"]
            target_cols = [c for c in target_cols if c in df.columns]
            
        if not target_cols:
            # Fallback: use TKE ratio as correction target
            if "k" in df.columns:
                targets = np.ones((len(df), 1))  # β = 1 (unity correction placeholder)
                target_cols = ["beta_correction"]
            else:
                continue
        else:
            targets = df[target_cols].values

        valid = np.all(np.isfinite(features), axis=1) & np.all(np.isfinite(targets), axis=1)
        all_features.append(features[valid])
        all_targets.append(targets[valid])
        case_labels.extend([case] * valid.sum())
        model_labels.extend([model] * valid.sum())

    if not all_features:
        logger.warning("No valid data found to build CFDDataset")
        return None

    X = np.vstack(all_features)
    y = np.vstack(all_targets)

    fname = feature_names or ["q1_strain_turbulence", "q2_wall_distance",
                               "q3_strain_rotation", "q4_pressure_gradient",
                               "q5_viscosity_ratio"]
    tname = target_names or target_cols

    return CFDDataset(
        name=dataset_name,
        features=X,
        targets=y,
        feature_names=fname,
        target_names=tname[:y.shape[1]],
        case_labels=case_labels,
        model_labels=model_labels,
    )


def split_by_case(
    dataset: "CFDDataset", 
    train_cases: List[str],
    val_cases: Optional[List[str]] = None,
    test_cases: Optional[List[str]] = None,
) -> Tuple["CFDDataset", Optional["CFDDataset"], Optional["CFDDataset"]]:
    """
    Split a CFDDataset strictly by case labels to evaluate cross-case generalization.

    This aligns with the curated dataset benchmarking strategy of evaluating models
    on completely unseen parametric geometries.

    Parameters
    ----------
    dataset : CFDDataset
        The full dataset structure.
    train_cases : list of str
        Identifiers of cases to include in the training set.
    val_cases : list of str, optional
        Identifiers of cases to include in validation.
    test_cases : list of str, optional
        Identifiers of cases to include in testing.

    Returns
    -------
    train_ds, val_ds, test_ds : CFDDataset (val/test may be None if not provided)
    """
    import copy

    def _filter_cases(cases: List[str], suffix: str) -> Optional["CFDDataset"]:
        if not cases:
            return None
        
        mask = np.isin(dataset.case_labels, cases)
        if not np.any(mask):
            return None
            
        ds_subset = copy.copy(dataset)
        ds_subset.name = f"{dataset.name}_{suffix}"
        ds_subset.features = dataset.features[mask]
        ds_subset.targets = dataset.targets[mask]
        ds_subset.case_labels = [c for m, c in zip(mask, dataset.case_labels) if m]
        ds_subset.model_labels = [m for mask_val, m in zip(mask, dataset.model_labels) if mask_val]
        return ds_subset

    train_ds = _filter_cases(train_cases, "train")
    if train_ds is None:
         raise ValueError(f"No samples found for train_cases: {train_cases}")
         
    val_ds = _filter_cases(val_cases, "val") if val_cases else None
    test_ds = _filter_cases(test_cases, "test") if test_cases else None

    return train_ds, val_ds, test_ds


# ===================================================================
# Dataset Summary
# ===================================================================
def get_dataset_summary(
    data_dir: Optional[Path] = None
) -> Dict:
    """
    Return metadata about the McConkey dataset availability.

    Returns
    -------
    dict with keys: available, n_files, n_points, models, cases, path
    """
    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    summary = {
        "available": data_dir.exists(),
        "path": str(data_dir),
        "paper_doi": "10.1038/s41597-021-01034-2",
        "data_doi": "10.34740/kaggle/dsv/2637500",
        "description": "McConkey et al. (2021) curated turbulence dataset",
        "total_points": 895_640,
        "n_rans_models": 4,
        "rans_models": ALL_RANS_MODELS,
        "n_cases_per_model": 29,
    }

    if data_dir.exists():
        csv_files = list(data_dir.rglob("*.csv"))
        parquet_files = list(data_dir.rglob("*.parquet"))
        summary["n_csv_files"] = len(csv_files)
        summary["n_parquet_files"] = len(parquet_files)
        summary["n_files"] = len(csv_files) + len(parquet_files)
    else:
        summary["n_files"] = 0
        summary["download_instructions"] = (
            "kaggle datasets download -d ryleymcconkey/ml-turbulence-dataset "
            f"&& unzip to {data_dir}"
        )

    return summary


# ===================================================================
# CLI
# ===================================================================
def main():
    """Print dataset summary and attempt to load bump cases."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("=" * 60)
    print("  McConkey et al. (2021) Curated Turbulence Dataset")
    print("  DOI: 10.1038/s41597-021-01034-2")
    print("=" * 60)

    summary = get_dataset_summary()
    print(f"\n  Path:      {summary['path']}")
    print(f"  Available: {summary['available']}")
    print(f"  Files:     {summary.get('n_files', 'N/A')}")
    print(f"  Expected:  {summary['total_points']:,} points")
    print(f"  Models:    {', '.join(summary['rans_models'])}")

    if not summary["available"]:
        print(f"\n  Dataset not found. Download with:")
        print(f"    kaggle datasets download -d ryleymcconkey/ml-turbulence-dataset")
        print(f"    Unzip to: {summary['path']}")
        return

    print("\n  Loading dataset...")
    try:
        data = load_mcconkey_dataset()
        print(f"  Loaded {len(data)} model-case pairs")

        bump_data = extract_bump_cases(data)
        print(f"  Bump cases: {len(bump_data)}")

        total_points = sum(len(df) for df in data.values())
        bump_points = sum(len(df) for df in bump_data.values())
        print(f"  Total points: {total_points:,}")
        print(f"  Bump points:  {bump_points:,}")

        # Try building CFDDataset
        if CFDDataset is not None:
            ds = to_cfd_dataset(bump_data)
            if ds is not None:
                print(f"\n  CFDDataset built: {ds.n_samples} samples, "
                      f"{ds.n_features} features")
    except FileNotFoundError as e:
        print(f"\n  {e}")
    except Exception as e:
        print(f"\n  Error loading dataset: {e}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
