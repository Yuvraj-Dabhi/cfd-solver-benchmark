"""
ML Training-Validation Pipeline
================================
End-to-end pipeline for turbulence model correction:
  1. Generate synthetic CFD-like training data from multiple cases
  2. Extract invariant features
  3. Train TurbulenceModelCorrection model
  4. Leave-one-out cross-validation across cases
  5. Report per-case and overall accuracy

Usage:
    pipeline = MLTrainValidatePipeline()
    results = pipeline.run_cross_validation(n_samples_per_case=500)
    print(results.summary())

    # Or train on all and predict
    pipeline.train_all(n_samples_per_case=500)
    corrections = pipeline.predict(new_features)
"""

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


@dataclass
class CaseData:
    """Training data for one flow case."""
    case_name: str
    features: np.ndarray       # (N, n_features)
    targets: np.ndarray        # (N, n_outputs) — correction labels
    n_samples: int = 0


@dataclass
class CVFoldResult:
    """Result from one cross-validation fold."""
    test_case: str
    train_cases: List[str]
    n_train: int
    n_test: int
    mape: float                # Mean absolute percentage error
    rmse: float                # Root mean squared error
    r2: float                  # Coefficient of determination
    max_error: float           # Maximum absolute error


@dataclass
class PipelineResult:
    """Full pipeline results."""
    fold_results: List[CVFoldResult] = field(default_factory=list)
    overall_mape: float = 0.0
    overall_r2: float = 0.0
    model_path: Optional[str] = None

    def summary(self) -> str:
        """Human-readable summary of cross-validation results."""
        lines = [
            "ML Training-Validation Pipeline Results",
            "=" * 60,
            f"{'Test Case':<25} {'MAPE (%)':<12} {'R²':<10} {'RMSE':<10} {'N_test'}",
            "-" * 60,
        ]
        for f in self.fold_results:
            lines.append(
                f"{f.test_case:<25} {f.mape:<12.3f} {f.r2:<10.4f} "
                f"{f.rmse:<10.6f} {f.n_test}"
            )
        lines.append("-" * 60)
        lines.append(f"{'OVERALL':<25} {self.overall_mape:<12.3f} {self.overall_r2:<10.4f}")
        return "\n".join(lines)


class MLTrainValidatePipeline:
    """
    End-to-end ML turbulence model correction pipeline.

    Parameters
    ----------
    n_features : int
        Number of invariant features (default 7 per Pope 1975).
    n_outputs : int
        Number of correction outputs (1 for Δν_t, 6 for anisotropy).
    hidden_layers : list
        NN hidden layer sizes.
    """

    def __init__(
        self,
        n_features: int = 7,
        n_outputs: int = 1,
        hidden_layers: Optional[List[int]] = None,
    ):
        self.n_features = n_features
        self.n_outputs = n_outputs
        self.hidden_layers = hidden_layers or [64, 64, 32]
        self._model = None
        self._scaler = None

    def generate_case_data(
        self, case_name: str, n_samples: int = 500
    ) -> CaseData:
        """
        Generate synthetic training data for one case.

        Uses physics-informed synthetic profiles that mimic BL characteristics.
        """
        np.random.seed(hash(case_name) % 2**32)

        # Flow-regime-dependent feature distributions
        regime_params = {
            "flat_plate": {"apg_strength": 0.0, "curvature": 0.0, "Re_factor": 1.0},
            "backward_facing_step": {"apg_strength": 0.3, "curvature": 0.1, "Re_factor": 0.5},
            "nasa_hump": {"apg_strength": 0.6, "curvature": 0.3, "Re_factor": 1.2},
            "periodic_hill": {"apg_strength": 0.8, "curvature": 0.5, "Re_factor": 0.3},
            "swbli": {"apg_strength": 0.9, "curvature": 0.1, "Re_factor": 2.0},
        }
        params = regime_params.get(
            case_name,
            {"apg_strength": 0.5, "curvature": 0.2, "Re_factor": 1.0},
        )

        # Generate invariant features
        # λ1-λ5: strain/rotation invariants, q6: pressure gradient, q7: wall distance
        features = np.zeros((n_samples, self.n_features))
        for i in range(min(5, self.n_features)):
            features[:, i] = np.random.randn(n_samples) * (0.1 + 0.05 * i)
        if self.n_features > 5:
            features[:, 5] = params["apg_strength"] * np.random.exponential(0.5, n_samples)
        if self.n_features > 6:
            features[:, 6] = np.random.exponential(0.3, n_samples) * params["Re_factor"]

        # Generate correction targets (synthetic ground truth)
        # Model: correction depends on APG + curvature indicators
        target_base = (
            0.1 * params["apg_strength"] * features[:, 0]
            + 0.2 * params["curvature"] * features[:, 1] ** 2
            + 0.05 * np.sin(2 * np.pi * features[:, 5] if self.n_features > 5 else 0)
        )
        noise = 0.02 * np.random.randn(n_samples)
        targets = target_base + noise

        if self.n_outputs > 1:
            targets = np.column_stack([targets] + [
                targets * (0.8 + 0.1 * j + 0.05 * np.random.randn(n_samples))
                for j in range(1, self.n_outputs)
            ])
        else:
            targets = targets.reshape(-1, 1)

        return CaseData(
            case_name=case_name,
            features=features,
            targets=targets,
            n_samples=n_samples,
        )

    def run_cross_validation(
        self,
        case_names: Optional[List[str]] = None,
        n_samples_per_case: int = 500,
    ) -> PipelineResult:
        """
        Leave-one-out cross-validation across cases.

        Parameters
        ----------
        case_names : list, optional
            Cases to include. Default: standard set.
        n_samples_per_case : int
            Samples per case.
        """
        if case_names is None:
            case_names = [
                "flat_plate", "backward_facing_step", "nasa_hump",
                "periodic_hill", "swbli",
            ]

        # Generate data for all cases
        all_data = {
            name: self.generate_case_data(name, n_samples_per_case)
            for name in case_names
        }

        result = PipelineResult()

        for test_name in case_names:
            # Train on all except test case
            train_cases = [n for n in case_names if n != test_name]
            X_train = np.vstack([all_data[n].features for n in train_cases])
            y_train = np.vstack([all_data[n].targets for n in train_cases])

            # Test on held-out case
            X_test = all_data[test_name].features
            y_test = all_data[test_name].targets

            # Standardize
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            # Train a simple model
            from sklearn.neural_network import MLPRegressor
            model = MLPRegressor(
                hidden_layer_sizes=tuple(self.hidden_layers),
                max_iter=200,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.15,
            )
            model.fit(X_train_s, y_train.ravel() if self.n_outputs == 1 else y_train)

            # Predict
            y_pred = model.predict(X_test_s)
            if y_pred.ndim == 1:
                y_pred = y_pred.reshape(-1, 1)

            # Metrics
            mape = float(np.mean(
                np.abs(y_pred - y_test) / np.maximum(np.abs(y_test), 1e-10)
            ) * 100)
            rmse = float(np.sqrt(np.mean((y_pred - y_test) ** 2)))
            ss_res = np.sum((y_test - y_pred) ** 2)
            ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
            r2 = float(1 - ss_res / max(ss_tot, 1e-15))
            max_err = float(np.max(np.abs(y_pred - y_test)))

            fold = CVFoldResult(
                test_case=test_name,
                train_cases=train_cases,
                n_train=len(X_train),
                n_test=len(X_test),
                mape=mape,
                rmse=rmse,
                r2=r2,
                max_error=max_err,
            )
            result.fold_results.append(fold)
            logger.info(f"  {test_name}: MAPE={mape:.2f}%, R²={r2:.4f}")

        # Overall metrics
        result.overall_mape = float(np.mean([f.mape for f in result.fold_results]))
        result.overall_r2 = float(np.mean([f.r2 for f in result.fold_results]))

        return result

    def train_all(
        self,
        case_names: Optional[List[str]] = None,
        n_samples_per_case: int = 500,
        save_path: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Train on all available cases.

        Returns
        -------
        dict with training metrics.
        """
        if case_names is None:
            case_names = [
                "flat_plate", "backward_facing_step", "nasa_hump",
                "periodic_hill", "swbli",
            ]

        all_data = {
            name: self.generate_case_data(name, n_samples_per_case)
            for name in case_names
        }

        X = np.vstack([d.features for d in all_data.values()])
        y = np.vstack([d.targets for d in all_data.values()])

        from sklearn.preprocessing import StandardScaler
        from sklearn.neural_network import MLPRegressor

        self._scaler = StandardScaler()
        X_s = self._scaler.fit_transform(X)

        self._model = MLPRegressor(
            hidden_layer_sizes=tuple(self.hidden_layers),
            max_iter=300,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.15,
        )
        self._model.fit(X_s, y.ravel() if self.n_outputs == 1 else y)

        y_pred = self._model.predict(X_s)
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)

        rmse = float(np.sqrt(np.mean((y_pred - y) ** 2)))
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = float(1 - ss_res / max(ss_tot, 1e-15))

        metrics = {"rmse": rmse, "r2": r2, "n_samples": len(X)}

        if save_path:
            import joblib
            joblib.dump({"model": self._model, "scaler": self._scaler}, save_path)
            metrics["model_path"] = save_path
            logger.info(f"Model saved to {save_path}")

        return metrics

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict corrections for new features."""
        if self._model is None or self._scaler is None:
            raise RuntimeError("Model not trained. Call train_all() first.")
        X_s = self._scaler.transform(features)
        return self._model.predict(X_s)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("ML Turbulence Model Correction Pipeline")
    print("=" * 60)

    pipeline = MLTrainValidatePipeline()

    # Cross-validation
    results = pipeline.run_cross_validation(n_samples_per_case=300)
    print(results.summary())

    # Train on all
    print("\n\nTraining on all cases...")
    metrics = pipeline.train_all(n_samples_per_case=500)
    print(f"  Train R²: {metrics['r2']:.4f}, RMSE: {metrics['rmse']:.6f}")
