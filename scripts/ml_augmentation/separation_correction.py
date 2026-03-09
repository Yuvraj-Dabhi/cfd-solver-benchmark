#!/usr/bin/env python3
"""
Non-Intrusive ML Separation Correction for Wall-Hump RANS
===========================================================
Phase 2 ML-CFD extension aligned with "Turbulence in the Age of Data"
(Duraisamy et al., 2019, Ann. Rev. Fluid Mech.).

Corrects RANS separation metrics *without modifying the solver* by
training an MLP on physics-informed features extracted from the
surface Cp/Cf distributions.

Architecture:
    RANS Cp/Cf → SeparationFeatureExtractor → 8 features
    → SeparationCorrectionModel (MLP + realizability) → 4 corrections:
        Δx_sep, Δx_reatt, ΔCf_min, ΔL_bubble

Target: 40% error reduction on separation location (per NASA TMR concept).

Physics-informed features:
  f1: Cp slope at separation (adverse pressure gradient strength)
  f2: Cf minimum depth (reverse flow intensity)
  f3: Cf minimum location / chord
  f4: Pressure recovery rate (∂Cp/∂x post-reattachment)
  f5: Clauser APG parameter (β_c = δ*/τ_w · dP/dx)
  f6: Cp peak suction magnitude
  f7: Bubble aspect ratio (L_bubble / Cp_amplitude)
  f8: Model indicator (SA=0, SST=1, kEpsilon=2)

References:
  - Duraisamy, Iaccarino, Xiao (2019), "Turbulence Modeling in the
    Age of Data", Ann. Rev. Fluid Mech. 51:357-377
  - Singh et al. (2017), AIAA J., ML-augmented RANS for separated flows
  - Parish & Duraisamy (2016), JCP, Field inversion framework
  - Weatheritt & Sandberg (2016), JFM, Gene-expression programming
    for Reynolds stress closures

Usage:
    python -m scripts.ml_augmentation.separation_correction
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)
PROJECT = Path(__file__).resolve().parent.parent.parent

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import KFold
    from sklearn.metrics import mean_squared_error, r2_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# =============================================================================
# Data Structures
# =============================================================================
@dataclass
class SeparationMetrics:
    """Raw separation metrics from a RANS solution."""
    x_sep: float = np.nan
    x_reatt: float = np.nan
    L_bubble: float = np.nan
    Cf_min: float = np.nan
    model: str = ""
    grid: str = ""
    converged: bool = True


@dataclass
class CorrectionResult:
    """Result of applying ML correction."""
    raw_metrics: SeparationMetrics
    corrected_x_sep: float = np.nan
    corrected_x_reatt: float = np.nan
    corrected_Cf_min: float = np.nan
    corrected_L_bubble: float = np.nan
    exp_x_sep: float = 0.665
    exp_x_reatt: float = 1.10
    exp_L_bubble: float = 0.435

    @property
    def raw_sep_error(self) -> float:
        return abs(self.raw_metrics.x_sep - self.exp_x_sep)

    @property
    def corrected_sep_error(self) -> float:
        return abs(self.corrected_x_sep - self.exp_x_sep)

    @property
    def sep_error_reduction_pct(self) -> float:
        if self.raw_sep_error < 1e-15:
            return 0.0
        return (1 - self.corrected_sep_error / self.raw_sep_error) * 100

    @property
    def raw_reatt_error(self) -> float:
        return abs(self.raw_metrics.x_reatt - self.exp_x_reatt)

    @property
    def corrected_reatt_error(self) -> float:
        return abs(self.corrected_x_reatt - self.exp_x_reatt)

    @property
    def reatt_error_reduction_pct(self) -> float:
        if self.raw_reatt_error < 1e-15:
            return 0.0
        return (1 - self.corrected_reatt_error / self.raw_reatt_error) * 100


@dataclass
class TrainingResult:
    """Result from correction model training."""
    n_samples: int = 0
    n_features: int = 0
    train_r2: float = np.nan
    val_r2: float = np.nan
    train_rmse: float = np.nan
    val_rmse: float = np.nan
    cv_r2_mean: float = np.nan
    cv_r2_std: float = np.nan
    mean_error_reduction_pct: float = np.nan


# =============================================================================
# 1. Separation Feature Extractor
# =============================================================================
class SeparationFeatureExtractor:
    """
    Extract physics-informed features from RANS surface Cp/Cf.

    These features encode the flow physics that determine separation
    behavior, following the invariance and physics-informed targets
    emphasized in Duraisamy et al. (2019).

    Features (8D):
      f1: ∂Cp/∂x at separation onset (APG strength)
      f2: Cf minimum depth (recirculation intensity)
      f3: x-location of Cf minimum / chord
      f4: ∂Cp/∂x in recovery region (pressure recovery rate)
      f5: Clauser β_c estimate (APG severity parameter)
      f6: Peak suction Cp magnitude
      f7: Bubble aspect ratio (L / Cp_amplitude)
      f8: Model class indicator (SA=0, SST=1, kε=2)
    """

    MODEL_ENCODING = {"SA": 0.0, "SST": 1.0, "kEpsilon": 2.0}

    def extract(
        self,
        x: np.ndarray,
        Cp: np.ndarray,
        Cf: np.ndarray,
        model: str = "SA",
        x_sep_approx: float = 0.665,
    ) -> np.ndarray:
        """
        Extract 8 physics-informed features from surface distributions.

        Parameters
        ----------
        x : ndarray
            Streamwise coordinate (x/c).
        Cp, Cf : ndarray
            Pressure coefficient and skin friction coefficient.
        model : str
            Turbulence model name.
        x_sep_approx : float
            Approximate separation location for gradient computation.

        Returns
        -------
        features : ndarray (8,)
        """
        features = np.zeros(8)

        # f1: Cp gradient at separation (APG strength)
        sep_mask = (x >= x_sep_approx - 0.1) & (x <= x_sep_approx + 0.05)
        if sep_mask.sum() > 2:
            x_s, Cp_s = x[sep_mask], Cp[sep_mask]
            features[0] = np.polyfit(x_s, Cp_s, 1)[0]  # dCp/dx

        # f2: Cf minimum depth
        sep_region = (x >= 0.5) & (x <= 1.4)
        if sep_region.any():
            features[1] = np.min(Cf[sep_region])

        # f3: x-location of Cf minimum
        if sep_region.any():
            min_idx = np.argmin(Cf[sep_region])
            features[2] = x[sep_region][min_idx]

        # f4: Pressure recovery rate (post-reattachment Cp gradient)
        recov_mask = (x >= 1.1) & (x <= 1.5)
        if recov_mask.sum() > 2:
            x_r, Cp_r = x[recov_mask], Cp[recov_mask]
            features[3] = np.polyfit(x_r, Cp_r, 1)[0]

        # f5: Clauser β_c estimate (simplified)
        # β_c ≈ (δ*/τ_w) × dP/dx ≈ (Cp_slope / Cf_mean) in non-dim form
        Cf_mean_sep = np.mean(np.abs(Cf[sep_region])) if sep_region.any() else 1e-6
        features[4] = features[0] / max(Cf_mean_sep, 1e-8)

        # f6: Peak suction Cp magnitude
        hump_region = (x >= 0) & (x <= 0.7)
        if hump_region.any():
            features[5] = np.min(Cp[hump_region])  # Most negative Cp

        # f7: Bubble aspect ratio (L_bubble / Cp_amplitude)
        Cp_amplitude = abs(features[5]) if abs(features[5]) > 1e-6 else 1e-6
        # Estimate bubble length from Cf zero crossings
        L_est = self._estimate_bubble_length(x, Cf) or 0.4
        features[6] = L_est / Cp_amplitude

        # f8: Model indicator
        features[7] = self.MODEL_ENCODING.get(model, 0.0)

        return features

    def _estimate_bubble_length(
        self, x: np.ndarray, Cf: np.ndarray,
    ) -> Optional[float]:
        """Estimate bubble length from Cf zero crossings."""
        sep_region = (x >= 0.5) & (x <= 1.5)
        x_r, Cf_r = x[sep_region], Cf[sep_region]
        if len(Cf_r) < 3:
            return None

        sign_changes = np.diff(np.sign(Cf_r))
        neg_cross = np.where(sign_changes < 0)[0]
        pos_cross = np.where(sign_changes > 0)[0]

        if len(neg_cross) > 0 and len(pos_cross) > 0:
            x_sep = x_r[neg_cross[0]]
            x_reat = x_r[pos_cross[-1]]
            if x_reat > x_sep:
                return float(x_reat - x_sep)
        return None

    def extract_batch(
        self,
        samples: List[Dict],
    ) -> np.ndarray:
        """
        Extract features from multiple RANS solutions.

        Parameters
        ----------
        samples : list of dicts with 'x', 'Cp', 'Cf', 'model' keys.

        Returns
        -------
        features : ndarray (N, 8)
        """
        return np.array([
            self.extract(s["x"], s["Cp"], s["Cf"], s.get("model", "SA"))
            for s in samples
        ])


# =============================================================================
# 2. Separation Correction Model
# =============================================================================
class SeparationCorrectionModel:
    """
    MLP that predicts corrections to RANS separation metrics.

    Targets (4D):
      Δx_sep    = x_sep_true - x_sep_RANS
      Δx_reatt  = x_reatt_true - x_reatt_RANS
      ΔCf_min   = Cf_min_true - Cf_min_RANS
      ΔL_bubble = L_true - L_RANS

    Physics constraint (realizability):
      corrected x_sep < corrected x_reatt (enforced post-hoc if needed)

    Architecture: 8 → 64(ReLU) → 64(ReLU) → 32(ReLU) → 4
    """

    def __init__(
        self,
        hidden_layers: Tuple[int, ...] = (64, 64, 32),
        n_ensemble: int = 5,
    ):
        self.hidden_layers = hidden_layers
        self.n_ensemble = n_ensemble
        self.scaler_X = None
        self.scaler_y = None
        self.models = []
        self._trained = False

    def train(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        val_split: float = 0.2,
    ) -> TrainingResult:
        """
        Train the correction model (ensemble of MLPs).

        Parameters
        ----------
        features : ndarray (N, 8)
            Physics-informed features.
        targets : ndarray (N, 4)
            Target corrections [Δx_sep, Δx_reatt, ΔCf_min, ΔL_bubble].
        val_split : float
            Validation fraction.
        """
        n = len(features)
        n_val = max(int(n * val_split), 1)

        # Normalize
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        X_norm = self.scaler_X.fit_transform(features)
        y_norm = self.scaler_y.fit_transform(targets)

        # Split
        perm = np.random.RandomState(42).permutation(n)
        X_train = X_norm[perm[n_val:]]
        y_train = y_norm[perm[n_val:]]
        X_val = X_norm[perm[:n_val]]
        y_val = y_norm[perm[:n_val]]

        # Train ensemble
        self.models = []
        train_scores = []
        val_scores = []

        for i in range(self.n_ensemble):
            model = MLPRegressor(
                hidden_layer_sizes=self.hidden_layers,
                activation='relu',
                max_iter=2000,
                learning_rate_init=1e-3,
                early_stopping=True,
                validation_fraction=0.15,
                random_state=42 + i,  # Different initialization per member
                tol=1e-6,
            )
            model.fit(X_train, y_train)
            self.models.append(model)

            train_scores.append(r2_score(y_train, model.predict(X_train),
                                         multioutput='uniform_average'))
            val_scores.append(r2_score(y_val, model.predict(X_val),
                                       multioutput='uniform_average'))

        self._trained = True

        # Cross-validation
        cv_r2s = self._cross_validate(X_norm, y_norm)

        # Ensemble prediction on validation set
        y_pred_val = self._predict_raw(X_val)
        val_r2 = r2_score(
            self.scaler_y.inverse_transform(y_val),
            self.scaler_y.inverse_transform(y_pred_val),
            multioutput='uniform_average',
        )

        result = TrainingResult(
            n_samples=n,
            n_features=features.shape[1],
            train_r2=float(np.mean(train_scores)),
            val_r2=float(val_r2),
            train_rmse=float(np.sqrt(mean_squared_error(
                y_train, self.models[0].predict(X_train)))),
            val_rmse=float(np.sqrt(mean_squared_error(
                y_val, self.models[0].predict(X_val)))),
            cv_r2_mean=float(np.mean(cv_r2s)),
            cv_r2_std=float(np.std(cv_r2s)),
        )

        logger.info(
            f"Trained {self.n_ensemble}-member ensemble: "
            f"val_R²={result.val_r2:.4f}, CV_R²={result.cv_r2_mean:.4f}±{result.cv_r2_std:.4f}"
        )
        return result

    def _predict_raw(self, X_norm: np.ndarray) -> np.ndarray:
        """Ensemble-averaged prediction on normalized data."""
        preds = np.array([m.predict(X_norm) for m in self.models])
        return np.mean(preds, axis=0)

    def predict(
        self,
        features: np.ndarray,
        raw_metrics: Optional[SeparationMetrics] = None,
    ) -> np.ndarray:
        """
        Predict corrections from features.

        Parameters
        ----------
        features : ndarray (8,) or (N, 8)

        Returns
        -------
        corrections : ndarray (4,) or (N, 4) — [Δx_sep, Δx_reatt, ΔCf_min, ΔL_bubble]
        """
        if not self._trained:
            raise RuntimeError("Model not trained")

        single = features.ndim == 1
        if single:
            features = features.reshape(1, -1)

        X_norm = self.scaler_X.transform(features)
        y_norm = self._predict_raw(X_norm)
        corrections = self.scaler_y.inverse_transform(y_norm)

        # Realizability check: corrected x_sep < corrected x_reatt
        if raw_metrics is not None:
            x_sep_corr = raw_metrics.x_sep + corrections[0, 0]
            x_reat_corr = raw_metrics.x_reatt + corrections[0, 1]
            if x_sep_corr >= x_reat_corr:
                # Enforce: slightly adjust to maintain ordering
                mid = (x_sep_corr + x_reat_corr) / 2
                corrections[0, 0] = mid - 0.01 - raw_metrics.x_sep
                corrections[0, 1] = mid + 0.01 - raw_metrics.x_reatt

        return corrections[0] if single else corrections

    def predict_with_uncertainty(
        self, features: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict corrections with epistemic uncertainty from ensemble.

        Returns (mean_correction, std_correction).
        """
        if not self._trained:
            raise RuntimeError("Model not trained")

        single = features.ndim == 1
        if single:
            features = features.reshape(1, -1)

        X_norm = self.scaler_X.transform(features)
        preds = np.array([
            self.scaler_y.inverse_transform(m.predict(X_norm).reshape(-1, 4))
            for m in self.models
        ])  # (n_ensemble, N, 4)

        mean = np.mean(preds, axis=0)
        std = np.std(preds, axis=0)

        return (mean[0], std[0]) if single else (mean, std)

    def _cross_validate(
        self, X: np.ndarray, y: np.ndarray, k: int = 5,
    ) -> List[float]:
        """k-fold cross-validation R² scores."""
        kf = KFold(n_splits=min(k, len(X)), shuffle=True, random_state=42)
        scores = []
        for train_idx, val_idx in kf.split(X):
            model = MLPRegressor(
                hidden_layer_sizes=self.hidden_layers,
                activation='relu', max_iter=1000,
                random_state=42, early_stopping=True,
            )
            model.fit(X[train_idx], y[train_idx])
            score = r2_score(y[val_idx], model.predict(X[val_idx]),
                             multioutput='uniform_average')
            scores.append(score)
        return scores


# =============================================================================
# 3. Training Data Generator
# =============================================================================
class CorrectionTrainer:
    """
    Generate multi-model training data and train the correction model.

    Creates synthetic paired data:
      (RANS_features, RANS_metrics) → (true_metrics - RANS_metrics) = corrections

    Varies:
      - Turbulence model (SA, SST, k-ε)
      - Reynolds number perturbation (±20%)
      - APG strength perturbation (±15%)
    """

    def __init__(
        self,
        n_samples_per_model: int = 100,
        noise_scale: float = 0.02,
    ):
        self.n_samples_per_model = n_samples_per_model
        self.noise_scale = noise_scale
        self.extractor = SeparationFeatureExtractor()

    def generate_training_data(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate training data: (features, corrections, raw_metrics).

        Returns
        -------
        features : (N, 8)
        corrections : (N, 4) — [Δx_sep, Δx_reatt, ΔCf_min, ΔL_bubble]
        raw_metrics : (N, 4) — [x_sep, x_reatt, Cf_min, L_bubble]
        """
        # Experimental ground truth
        EXP = {"x_sep": 0.665, "x_reatt": 1.10, "Cf_min": -0.00170, "L_bubble": 0.435}

        # Model-specific biases (mean RANS error)
        MODEL_BIASES = {
            "SA":       {"x_sep": 0.000, "x_reatt": 0.00, "Cf_min": -0.00015, "L_bubble": 0.000},
            "SST":      {"x_sep": -0.005, "x_reatt": 0.05, "Cf_min": -0.00080, "L_bubble": 0.055},
            "kEpsilon": {"x_sep": 0.005, "x_reatt": -0.02, "Cf_min": 0.00010, "L_bubble": -0.025},
        }

        rng = np.random.default_rng(42)
        all_features = []
        all_corrections = []
        all_raw = []

        for model, bias in MODEL_BIASES.items():
            for i in range(self.n_samples_per_model):
                # Perturbed RANS metrics (bias + noise)
                noise = rng.normal(0, self.noise_scale, 4)
                rans_x_sep = EXP["x_sep"] + bias["x_sep"] + noise[0] * 0.02
                rans_x_reatt = EXP["x_reatt"] + bias["x_reatt"] + noise[1] * 0.05
                rans_Cf_min = EXP["Cf_min"] + bias["Cf_min"] + noise[2] * 0.0005
                rans_L = rans_x_reatt - rans_x_sep

                # Corrections = truth - RANS
                corr = np.array([
                    EXP["x_sep"] - rans_x_sep,
                    EXP["x_reatt"] - rans_x_reatt,
                    EXP["Cf_min"] - rans_Cf_min,
                    EXP["L_bubble"] - rans_L,
                ])

                # Generate synthetic Cp/Cf for feature extraction
                x, Cp, Cf = self._generate_perturbed_surface(
                    model, rans_x_sep, rans_x_reatt, rans_Cf_min, rng
                )
                features = self.extractor.extract(x, Cp, Cf, model, rans_x_sep)

                all_features.append(features)
                all_corrections.append(corr)
                all_raw.append([rans_x_sep, rans_x_reatt, rans_Cf_min, rans_L])

        return (
            np.array(all_features),
            np.array(all_corrections),
            np.array(all_raw),
        )

    def _generate_perturbed_surface(
        self, model: str, x_sep: float, x_reatt: float,
        Cf_min: float, rng: np.random.Generator,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate synthetic Cp/Cf with model-specific characteristics."""
        n = 300
        x = np.linspace(-0.5, 2.0, n)

        # Cp: hump shape with perturbation
        Cp = np.zeros_like(x)
        hump = (x >= 0) & (x <= 0.65)
        Cp[hump] = -0.8 * np.sin(np.pi * x[hump] / 0.65)
        recov = x > 0.65
        Cp[recov] = -0.8 * np.exp(-2.5 * (x[recov] - 0.65))
        Cp += rng.normal(0, 0.01, n)

        # Cf: with separation bubble
        Cf = np.ones_like(x) * 0.003
        bubble = (x >= x_sep) & (x <= x_reatt)
        if bubble.any():
            t = (x[bubble] - x_sep) / max(x_reatt - x_sep, 0.01)
            Cf[bubble] = Cf_min * np.sin(np.pi * t)
        # Smooth approach
        approach = (x >= x_sep - 0.1) & (x < x_sep)
        if approach.any():
            t = (x[approach] - (x_sep - 0.1)) / 0.1
            Cf[approach] = 0.003 * (1 - t)
        # Recovery
        recovery = x > x_reatt
        if recovery.any():
            Cf[recovery] = 0.002 * (1 - np.exp(-3 * (x[recovery] - x_reatt)))
        Cf += rng.normal(0, 0.0001, n)

        return x, Cp, Cf


# =============================================================================
# 4. Correction Evaluator
# =============================================================================
class CorrectionEvaluator:
    """
    Evaluate ML correction: produce 40% error-reduction table.

    Compares:
      Raw RANS → ML-corrected → Experiment
    """

    def __init__(
        self,
        model: SeparationCorrectionModel,
        extractor: SeparationFeatureExtractor,
    ):
        self.model = model
        self.extractor = extractor

    def evaluate_model(
        self,
        model_name: str,
        x: np.ndarray,
        Cp: np.ndarray,
        Cf: np.ndarray,
        rans_metrics: SeparationMetrics,
    ) -> CorrectionResult:
        """
        Evaluate correction for a single RANS solution.
        """
        features = self.extractor.extract(x, Cp, Cf, model_name, rans_metrics.x_sep)
        corrections = self.model.predict(features, rans_metrics)

        result = CorrectionResult(
            raw_metrics=rans_metrics,
            corrected_x_sep=rans_metrics.x_sep + corrections[0],
            corrected_x_reatt=rans_metrics.x_reatt + corrections[1],
            corrected_Cf_min=rans_metrics.Cf_min + corrections[2],
            corrected_L_bubble=(rans_metrics.x_sep + corrections[0]) -
                               (rans_metrics.x_reatt + corrections[1]),
        )
        # Fix L_bubble sign
        result.corrected_L_bubble = abs(
            result.corrected_x_reatt - result.corrected_x_sep
        )

        return result

    def evaluate_all_models(self) -> Dict[str, CorrectionResult]:
        """Evaluate on SA, SST, and k-ε with default data."""
        from scripts.analysis.wall_hump_cross_model import (
            generate_synthetic_hump_data, extract_metrics_from_surface_data,
        )

        results = {}
        for model_name in ["SA", "SST", "kEpsilon"]:
            x, Cp, Cf = generate_synthetic_hump_data(model_name)
            raw_met = extract_metrics_from_surface_data(x, Cp, Cf, model_name)
            rans_met = SeparationMetrics(
                x_sep=raw_met.x_sep,
                x_reatt=raw_met.x_reatt,
                L_bubble=raw_met.bubble_length,
                Cf_min=raw_met.Cf_min,
                model=model_name,
            )
            result = self.evaluate_model(model_name, x, Cp, Cf, rans_met)
            results[model_name] = result

        return results

    @staticmethod
    def format_error_reduction_table(
        results: Dict[str, CorrectionResult],
    ) -> str:
        """Format the 40% error-reduction table."""
        lines = [
            "",
            "Non-Intrusive ML Separation Correction — Error Reduction",
            "=" * 85,
            f"{'Model':<10} {'|'} {'x_sep err (raw)':<16} {'x_sep err (ML)':<16} "
            f"{'Reduction':<12} {'|'} {'x_reatt err (raw)':<18} {'x_reatt err (ML)':<18} {'Reduction':<10}",
            "-" * 85,
        ]

        for model, result in results.items():
            lines.append(
                f"{model:<10} {'|':} "
                f"{result.raw_sep_error:<16.4f} {result.corrected_sep_error:<16.4f} "
                f"{result.sep_error_reduction_pct:<12.1f}% {'|':} "
                f"{result.raw_reatt_error:<18.4f} {result.corrected_reatt_error:<18.4f} "
                f"{result.reatt_error_reduction_pct:<10.1f}%"
            )

        # Average
        avg_sep = np.mean([r.sep_error_reduction_pct for r in results.values()])
        avg_reat = np.mean([r.reatt_error_reduction_pct for r in results.values()])
        lines.append("-" * 85)
        lines.append(f"{'AVERAGE':<10} {'|':} {'':16} {'':16} {avg_sep:<12.1f}% "
                      f"{'|':} {'':18} {'':18} {avg_reat:<10.1f}%")
        lines.append("")
        lines.append(f"  Target: ≥40% error reduction (NASA TMR concept)")
        lines.append(f"  Achieved: {avg_sep:.1f}% on x_sep, {avg_reat:.1f}% on x_reatt")

        return "\n".join(lines)


# =============================================================================
# 5. Full Pipeline
# =============================================================================
def run_separation_correction_pipeline(
    n_samples: int = 100,
    n_ensemble: int = 5,
) -> Dict:
    """
    Run the complete non-intrusive separation correction pipeline.

    1. Generate multi-model training data
    2. Train correction model (ensemble MLP)
    3. Evaluate on held-out data
    4. Print 40% error-reduction table
    """
    print("=" * 70)
    print("  Non-Intrusive ML Separation Correction (Phase 2)")
    print("  Aligned with: Duraisamy et al. (2019), 'Turbulence in")
    print("  the Age of Data', Ann. Rev. Fluid Mech. 51:357–377")
    print("=" * 70)

    # 1. Generate training data
    print("\n  Phase 1: Generating multi-model training data...")
    trainer = CorrectionTrainer(n_samples_per_model=n_samples)
    features, corrections, raw_metrics = trainer.generate_training_data()
    print(f"    Generated {len(features)} samples ({features.shape[1]} features)")
    print(f"    Models: SA ({n_samples}), SST ({n_samples}), k-ε ({n_samples})")

    # 2. Train
    print("\n  Phase 2: Training {}-member correction ensemble...".format(n_ensemble))
    model = SeparationCorrectionModel(n_ensemble=n_ensemble)
    train_result = model.train(features, corrections)
    print(f"    Train R²: {train_result.train_r2:.4f}")
    print(f"    Val R²:   {train_result.val_r2:.4f}")
    print(f"    CV R²:    {train_result.cv_r2_mean:.4f} ± {train_result.cv_r2_std:.4f}")

    # 3. Evaluate
    print("\n  Phase 3: Evaluating error reduction...")
    extractor = SeparationFeatureExtractor()
    evaluator = CorrectionEvaluator(model, extractor)
    results = evaluator.evaluate_all_models()

    # 4. Print table
    table = CorrectionEvaluator.format_error_reduction_table(results)
    print(table)

    # 5. Save report
    output_dir = PROJECT / "results" / "ml_separation_correction"
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "training": {
            "n_samples": train_result.n_samples,
            "n_features": train_result.n_features,
            "train_r2": train_result.train_r2,
            "val_r2": train_result.val_r2,
            "cv_r2_mean": train_result.cv_r2_mean,
            "cv_r2_std": train_result.cv_r2_std,
        },
        "results": {
            model: {
                "raw_x_sep": r.raw_metrics.x_sep,
                "corrected_x_sep": r.corrected_x_sep,
                "raw_x_reatt": r.raw_metrics.x_reatt,
                "corrected_x_reatt": r.corrected_x_reatt,
                "sep_error_reduction_pct": r.sep_error_reduction_pct,
                "reatt_error_reduction_pct": r.reatt_error_reduction_pct,
            }
            for model, r in results.items()
        },
    }

    with open(output_dir / "correction_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    return {"train_result": train_result, "results": results}


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_separation_correction_pipeline()
