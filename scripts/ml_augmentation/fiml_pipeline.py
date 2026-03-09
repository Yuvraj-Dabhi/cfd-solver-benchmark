#!/usr/bin/env python3
"""
FIML (Field Inversion and Machine Learning) Reusable Pipeline
==============================================================
Consolidates the FIML β-correction workflow from fiml_correction.py into
a reusable, cross-case training pipeline.

The FIML methodology (Parish & Duraisamy 2016):
  1. Run baseline RANS simulation
  2. Extract Galilean-invariant features (q1–q5) from the RANS field
  3. Compute β correction: β(x) = Cf_exp(x) / Cf_RANS(x) at the wall,
     propagated into the field via exponential decay
  4. Train MLP: q1...q5 → β
  5. Apply β-correction to new cases

This pipeline supports:
  - Multi-case training (wall hump + bump + BFS)
  - Leave-one-case-out cross-validation
  - Integration with the McConkey et al. (2021) curated turbulence dataset
  - Quantitative comparison of baseline vs. corrected predictions

References
----------
- Parish & Duraisamy (2016), JCP 305, pp. 758–774
- Srivastava et al. (2024), NASA TM-20240012512
- McConkey et al. (2021), Scientific Data 8, DOI:10.1038/s41597-021-01034-2
"""

import json
import logging
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Project root
PROJECT = Path(__file__).resolve().parent.parent.parent

# Try sklearn for fallback training
try:
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import KFold
    from sklearn.metrics import mean_squared_error, r2_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# Try PyTorch
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class FIMLCaseData:
    """Data for a single FIML case."""
    name: str
    features: np.ndarray      # (N, n_features) — Galilean-invariant q1–q5
    beta_target: np.ndarray   # (N,) — correction factor
    x_coords: np.ndarray      # (N,) — streamwise coordinates
    y_coords: np.ndarray      # (N,) — wall-normal coordinates
    cf_baseline: Optional[np.ndarray] = None   # Wall Cf from RANS
    cf_experimental: Optional[np.ndarray] = None  # Wall Cf from experiment
    x_wall: Optional[np.ndarray] = None          # Wall x coordinates
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FIMLResult:
    """Result from FIML training/evaluation."""
    model_type: str = "sklearn_mlp"
    train_cases: List[str] = field(default_factory=list)
    test_cases: List[str] = field(default_factory=list)
    train_rmse: float = 0.0
    test_rmse: float = 0.0
    train_r2: float = 0.0
    test_r2: float = 0.0
    cf_improvement: Dict[str, float] = field(default_factory=dict)
    # Cf improvement = (baseline_error - corrected_error) / baseline_error
    summary: str = ""


# =============================================================================
# Feature Extraction (Galilean-Invariant q1–q5)
# =============================================================================

def extract_fiml_features(
    nu_t: np.ndarray,
    nu: float,
    strain_mag: np.ndarray,
    wall_distance: np.ndarray,
    strain_rotation_ratio: np.ndarray,
    pressure_gradient_indicator: np.ndarray,
) -> np.ndarray:
    """
    Extract the 5 Galilean-invariant features used in the FIML framework.

    q1: Turbulence-to-mean-strain ratio — ν_t / (ν · |S| · d²)
    q2: Wall-distance Reynolds number — min(√(ν_t) · d / (50ν), 2)
    q3: Strain-rotation ratio — Ŝ_ij Ω̂_ij / |Ŝ|²
    q4: Pressure-gradient alignment — (∇p · ê_s) / (ρU²/c)
    q5: Turbulent viscosity ratio — ν_t / ν

    Parameters
    ----------
    nu_t : ndarray (N,)
        Turbulent eddy viscosity.
    nu : float
        Molecular kinematic viscosity.
    strain_mag : ndarray (N,)
        Magnitude of strain rate tensor |S|.
    wall_distance : ndarray (N,)
        Distance to nearest wall.
    strain_rotation_ratio : ndarray (N,)
        Pre-computed strain-rotation ratio.
    pressure_gradient_indicator : ndarray (N,)
        Pre-computed pressure gradient indicator.

    Returns
    -------
    features : ndarray (N, 5)
    """
    d_safe = np.maximum(wall_distance, 1e-15)
    S_safe = np.maximum(strain_mag, 1e-15)
    nut_safe = np.maximum(nu_t, 0.0)

    # q1: turbulence-to-strain ratio
    q1 = nut_safe / (nu * S_safe * d_safe**2 + 1e-15)
    q1 = np.clip(q1, 0, 10)

    # q2: wall-distance Reynolds number
    q2 = np.sqrt(nut_safe) * d_safe / (50.0 * nu + 1e-15)
    q2 = np.minimum(q2, 2.0)

    # q3: strain-rotation ratio
    q3 = np.clip(strain_rotation_ratio, -2, 2)

    # q4: pressure gradient indicator
    q4 = np.clip(pressure_gradient_indicator, -5, 5)

    # q5: eddy viscosity ratio
    q5 = nut_safe / nu
    q5 = np.log1p(q5)  # Log-transform for better scaling

    return np.column_stack([q1, q2, q3, q4, q5])


# =============================================================================
# FIML Pipeline
# =============================================================================

class FIMLPipeline:
    """
    End-to-end FIML training pipeline.

    Usage:
        pipeline = FIMLPipeline()
        pipeline.add_case(case_data)
        result = pipeline.train(test_case="wall_hump")
        beta_pred = pipeline.predict(new_features)
    """

    def __init__(
        self,
        hidden_layers: Tuple[int, ...] = (64, 64, 32),
        max_iter: int = 1000,
        learning_rate_init: float = 1e-3,
        random_state: int = 42,
    ):
        self.cases: Dict[str, FIMLCaseData] = {}
        self.hidden_layers = hidden_layers
        self.max_iter = max_iter
        self.learning_rate_init = learning_rate_init
        self.random_state = random_state

        self.model = None
        self.scaler = None
        self._trained = False

    def add_case(self, case: FIMLCaseData):
        """Add a case to the training pool."""
        self.cases[case.name] = case
        logger.info(f"Added case '{case.name}': {case.features.shape[0]} points, "
                     f"{case.features.shape[1]} features")

    def _assemble_data(
        self,
        case_names: List[str],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Stack features and targets from multiple cases."""
        X_list, y_list = [], []
        for name in case_names:
            case = self.cases[name]
            X_list.append(case.features)
            y_list.append(case.beta_target)
        return np.vstack(X_list), np.concatenate(y_list)

    def train(
        self,
        test_case: Optional[str] = None,
        train_cases: Optional[List[str]] = None,
    ) -> FIMLResult:
        """
        Train the FIML correction model.

        Parameters
        ----------
        test_case : str, optional
            Case name to exclude from training (leave-one-case-out).
        train_cases : list of str, optional
            Explicit list of training cases. If None, use all except test_case.

        Returns
        -------
        FIMLResult
        """
        if not HAS_SKLEARN:
            raise ImportError("scikit-learn required for FIML training")

        all_cases = list(self.cases.keys())

        if train_cases is None:
            if test_case:
                train_cases = [c for c in all_cases if c != test_case]
            else:
                train_cases = all_cases

        test_cases = [test_case] if test_case else []

        # Assemble training data
        X_train, y_train = self._assemble_data(train_cases)

        # Normalize
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)

        # Train model
        self.model = MLPRegressor(
            hidden_layer_sizes=self.hidden_layers,
            activation="relu",
            solver="adam",
            max_iter=self.max_iter,
            learning_rate_init=self.learning_rate_init,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=self.random_state,
            verbose=False,
        )
        self.model.fit(X_train_scaled, y_train)
        self._trained = True

        # Evaluate on training data
        y_train_pred = self.model.predict(X_train_scaled)
        result = FIMLResult(
            train_cases=train_cases,
            test_cases=test_cases,
            train_rmse=float(np.sqrt(mean_squared_error(y_train, y_train_pred))),
            train_r2=float(r2_score(y_train, y_train_pred)),
        )

        # Evaluate on test case
        if test_case and test_case in self.cases:
            X_test, y_test = self._assemble_data([test_case])
            X_test_scaled = self.scaler.transform(X_test)
            y_test_pred = self.model.predict(X_test_scaled)
            result.test_rmse = float(np.sqrt(mean_squared_error(y_test, y_test_pred)))
            result.test_r2 = float(r2_score(y_test, y_test_pred))

            # Compute Cf improvement if available
            case = self.cases[test_case]
            if case.cf_baseline is not None and case.cf_experimental is not None:
                improvement = self._compute_cf_improvement(case, y_test_pred)
                result.cf_improvement[test_case] = improvement

        result.summary = (
            f"FIML trained on {train_cases}: "
            f"R²_train={result.train_r2:.4f}, R²_test={result.test_r2:.4f}"
        )

        logger.info(result.summary)
        return result

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict β correction field for new features."""
        if not self._trained:
            raise RuntimeError("Pipeline not trained. Call train() first.")
        X_scaled = self.scaler.transform(features)
        return self.model.predict(X_scaled)

    def _compute_cf_improvement(
        self,
        case: FIMLCaseData,
        beta_predicted: np.ndarray,
    ) -> float:
        """
        Compute Cf improvement metric.

        Returns fractional reduction in Cf error after β-correction.
        """
        if case.cf_baseline is None or case.cf_experimental is None:
            return 0.0

        # Only compare at matching wall points
        n_wall = min(len(case.cf_baseline), len(case.cf_experimental))
        cf_base = case.cf_baseline[:n_wall]
        cf_exp = case.cf_experimental[:n_wall]

        # Baseline error
        err_base = np.sqrt(np.mean((cf_base - cf_exp)**2))

        # Corrected Cf: approximate β at wall (first n_wall points)
        beta_wall = beta_predicted[:n_wall] if len(beta_predicted) >= n_wall else beta_predicted
        if len(beta_wall) < n_wall:
            return 0.0

        cf_corrected = cf_base * beta_wall
        err_corrected = np.sqrt(np.mean((cf_corrected - cf_exp)**2))

        # Fractional improvement
        if err_base > 1e-10:
            return float((err_base - err_corrected) / err_base)
        return 0.0

    def cross_validate(self, n_folds: int = None) -> Dict[str, FIMLResult]:
        """
        Leave-one-case-out cross-validation.

        Returns dict mapping test case name → FIMLResult.
        """
        results = {}
        all_cases = list(self.cases.keys())

        if n_folds is None:
            # Leave-one-case-out
            for test_case in all_cases:
                logger.info(f"Cross-validation: holding out '{test_case}'")
                result = self.train(test_case=test_case)
                results[test_case] = result
        else:
            # K-fold (split by case)
            kf = KFold(n_splits=min(n_folds, len(all_cases)), shuffle=True,
                        random_state=self.random_state)
            for fold, (train_idx, test_idx) in enumerate(kf.split(all_cases)):
                train_names = [all_cases[i] for i in train_idx]
                test_name = all_cases[test_idx[0]]
                result = self.train(test_case=test_name, train_cases=train_names)
                results[f"fold_{fold}_{test_name}"] = result

        return results

    def save(self, output_dir: Path):
        """Save pipeline state."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if HAS_SKLEARN and self._trained:
            import pickle
            with open(output_dir / "fiml_model.pkl", "wb") as f:
                pickle.dump(self.model, f)
            with open(output_dir / "fiml_scaler.pkl", "wb") as f:
                pickle.dump(self.scaler, f)

        meta = {
            "hidden_layers": list(self.hidden_layers),
            "max_iter": self.max_iter,
            "cases": list(self.cases.keys()),
            "n_points_per_case": {
                name: int(case.features.shape[0])
                for name, case in self.cases.items()
            },
        }
        with open(output_dir / "fiml_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        logger.info(f"FIML pipeline saved to {output_dir}")

    def load(self, model_dir: Path):
        """Load pipeline state."""
        import pickle
        model_dir = Path(model_dir)
        with open(model_dir / "fiml_model.pkl", "rb") as f:
            self.model = pickle.load(f)
        with open(model_dir / "fiml_scaler.pkl", "rb") as f:
            self.scaler = pickle.load(f)
        self._trained = True
        logger.info(f"FIML pipeline loaded from {model_dir}")


# =============================================================================
# Synthetic Data Generator (for testing / demonstration)
# =============================================================================

def generate_synthetic_fiml_case(
    case_name: str = "synthetic_hump",
    n_points: int = 2000,
    seed: int = 42,
) -> FIMLCaseData:
    """
    Generate a synthetic FIML test case with known β correction.

    Creates synthetic flow features and a β field that mimics the
    wall hump correction pattern: β ≈ 1 in attached regions,
    β > 1 in the separation region (where RANS underpredicts Cf).

    Parameters
    ----------
    case_name : str
        Name for the case.
    n_points : int
        Number of spatial points.
    seed : int
        Random seed.

    Returns
    -------
    FIMLCaseData
    """
    rng = np.random.default_rng(seed)

    # Spatial coordinates
    x = np.linspace(0.0, 2.0, n_points)
    y = rng.uniform(0.0, 0.1, n_points)

    # Synthetic features (q1–q5)
    q1 = 0.5 * np.exp(-((x - 1.0)**2) / 0.1) + rng.normal(0, 0.02, n_points)
    q2 = np.clip(np.sqrt(y) * 10, 0, 2) + rng.normal(0, 0.01, n_points)
    q3 = 0.2 * np.sin(2 * np.pi * x) + rng.normal(0, 0.05, n_points)
    q4 = -0.5 * np.exp(-((x - 0.7)**2) / 0.05) + rng.normal(0, 0.02, n_points)
    q5 = np.log1p(100 * np.exp(-y * 50)) + rng.normal(0, 0.1, n_points)

    features = np.column_stack([q1, q2, q3, q4, q5])

    # Synthetic β: elevated in separation region (0.65 < x < 1.1)
    beta = np.ones(n_points)
    sep_mask = (x > 0.65) & (x < 1.1)
    beta[sep_mask] = 1.0 + 0.3 * np.sin(np.pi * (x[sep_mask] - 0.65) / 0.45)
    beta += rng.normal(0, 0.02, n_points)

    # Synthetic Cf
    cf_base = 0.004 * (1 - 0.5 * np.exp(-((x - 0.9)**2) / 0.01))
    cf_exp = cf_base * beta

    return FIMLCaseData(
        name=case_name,
        features=features,
        beta_target=beta,
        x_coords=x,
        y_coords=y,
        cf_baseline=cf_base[:200],
        cf_experimental=cf_exp[:200],
        x_wall=x[:200],
        metadata={"synthetic": True, "seed": seed},
    )


# =============================================================================
# Smoke Test
# =============================================================================

def _smoke_test():
    """Quick self-check with synthetic data."""
    print("=" * 60)
    print("  FIML Pipeline — Smoke Test")
    print("=" * 60)

    # Create synthetic cases
    case1 = generate_synthetic_fiml_case("hump_like", n_points=1000, seed=42)
    case2 = generate_synthetic_fiml_case("bump_like", n_points=800, seed=43)
    case3 = generate_synthetic_fiml_case("bfs_like", n_points=600, seed=44)

    print(f"[OK] Synthetic cases: {case1.name}={case1.features.shape}, "
          f"{case2.name}={case2.features.shape}, "
          f"{case3.name}={case3.features.shape}")

    # Feature extraction test
    features = extract_fiml_features(
        nu_t=np.abs(np.random.randn(100)) * 1e-4,
        nu=1.5e-5,
        strain_mag=np.abs(np.random.randn(100)) * 100,
        wall_distance=np.abs(np.random.randn(100)) * 0.01,
        strain_rotation_ratio=np.random.randn(100) * 0.5,
        pressure_gradient_indicator=np.random.randn(100) * 0.3,
    )
    assert features.shape == (100, 5)
    print(f"[OK] Feature extraction: shape={features.shape}")

    # Pipeline test
    pipeline = FIMLPipeline(hidden_layers=(32, 32), max_iter=200)
    pipeline.add_case(case1)
    pipeline.add_case(case2)
    pipeline.add_case(case3)

    # Train with leave-one-out
    result = pipeline.train(test_case="bfs_like")
    print(f"[OK] Training: R²_train={result.train_r2:.4f}, R²_test={result.test_r2:.4f}")
    assert result.train_r2 > 0, "Training R² should be positive"

    # Prediction
    beta_pred = pipeline.predict(case3.features)
    assert len(beta_pred) == len(case3.beta_target)
    print(f"[OK] Prediction: shape={beta_pred.shape}, "
          f"mean beta={np.mean(beta_pred):.4f}")

    # Cross-validation
    cv_results = pipeline.cross_validate()
    for name, res in cv_results.items():
        print(f"[OK] CV holdout '{name}': R²_test={res.test_r2:.4f}")

    print(f"\n{'=' * 60}")
    print("  ALL FIML PIPELINE TESTS PASSED")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _smoke_test()
