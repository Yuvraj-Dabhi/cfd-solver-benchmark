#!/usr/bin/env python3
"""
CL/CD Surrogate Model
========================
Gaussian Process (GP) or MLP surrogate for predicting aerodynamic
coefficients (CL, CD, CM) from flow parameters (AoA, Re, Mach).

Supports k-fold cross-validation and provides uncertainty estimates
(GP posterior variance or ensemble variance for MLP).

Usage
-----
    from scripts.ml_augmentation.surrogate_model import SurrogateModel

    model = SurrogateModel(model_type='gp')
    model.fit(X_train, y_train)
    y_pred, y_std = model.predict(X_test)
    metrics = model.evaluate(X_test, y_test)
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


class SurrogateModel:
    """
    CL/CD surrogate from AoA, Re, and Mach.

    Parameters
    ----------
    model_type : str
        Model type: 'gp' (Gaussian Process) or 'mlp' (Neural Network).
    normalize : bool
        Whether to normalize features.
    """

    def __init__(self, model_type: str = "gp", normalize: bool = True):
        if model_type not in ("gp", "mlp"):
            raise ValueError(f"model_type must be 'gp' or 'mlp', got '{model_type}'")
        self.model_type = model_type
        self.normalize = normalize
        self._model = None
        self._scaler_X = None
        self._scaler_y = None
        self._is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict:
        """
        Train the surrogate model.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Feature matrix. Columns: [AoA_deg, Re, Mach, ...].
        y : ndarray, shape (n_samples,) or (n_samples, n_targets)
            Target values (CL, CD, etc.).

        Returns
        -------
        dict with training metrics (R², RMSE).
        """
        from sklearn.preprocessing import StandardScaler

        X = np.asarray(X)
        y = np.asarray(y)

        if self.normalize:
            self._scaler_X = StandardScaler()
            self._scaler_y = StandardScaler()
            X_scaled = self._scaler_X.fit_transform(X)
            if y.ndim == 1:
                y_scaled = self._scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
            else:
                y_scaled = self._scaler_y.fit_transform(y)
        else:
            X_scaled = X
            y_scaled = y

        if self.model_type == "gp":
            metrics = self._fit_gp(X_scaled, y_scaled, **kwargs)
        else:
            metrics = self._fit_mlp(X_scaled, y_scaled, **kwargs)

        self._is_fitted = True
        return metrics

    def predict(
        self, X: np.ndarray
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Predict with the surrogate model.

        Parameters
        ----------
        X : ndarray
            Feature matrix.

        Returns
        -------
        For GP: (mean, std) tuple.
        For MLP: mean array.
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X = np.asarray(X)
        if self.normalize and self._scaler_X is not None:
            X_scaled = self._scaler_X.transform(X)
        else:
            X_scaled = X

        if self.model_type == "gp":
            mean_s, std_s = self._model.predict(X_scaled, return_std=True)
            if self.normalize and self._scaler_y is not None:
                if mean_s.ndim == 1:
                    mean = self._scaler_y.inverse_transform(
                        mean_s.reshape(-1, 1)
                    ).ravel()
                    std = std_s * self._scaler_y.scale_[0]
                else:
                    mean = self._scaler_y.inverse_transform(mean_s)
                    std = std_s * self._scaler_y.scale_
            else:
                mean, std = mean_s, std_s
            return mean, std
        else:
            pred_s = self._model.predict(X_scaled)
            if self.normalize and self._scaler_y is not None:
                if pred_s.ndim == 1:
                    pred = self._scaler_y.inverse_transform(
                        pred_s.reshape(-1, 1)
                    ).ravel()
                else:
                    pred = self._scaler_y.inverse_transform(pred_s)
            else:
                pred = pred_s
            return pred

    def evaluate(
        self, X_test: np.ndarray, y_test: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate model on test data.

        Returns
        -------
        dict with RMSE, R², MAPE.
        """
        from sklearn.metrics import mean_squared_error, r2_score

        y_test = np.asarray(y_test)
        result = self.predict(X_test)

        if isinstance(result, tuple):
            y_pred, y_std = result
        else:
            y_pred = result

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        # MAPE (avoiding division by zero)
        mask = np.abs(y_test) > 1e-10
        if mask.any():
            mape = np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100
        else:
            mape = 0.0

        return {"RMSE": rmse, "R2": r2, "MAPE": mape}

    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        k: int = 5,
    ) -> Dict[str, List[float]]:
        """
        k-fold cross-validation.

        Returns
        -------
        dict with lists of per-fold RMSE, R², MAPE.
        """
        from sklearn.model_selection import KFold

        X = np.asarray(X)
        y = np.asarray(y)
        kf = KFold(n_splits=k, shuffle=True, random_state=42)

        metrics = {"RMSE": [], "R2": [], "MAPE": []}

        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            fold_model = SurrogateModel(
                model_type=self.model_type, normalize=self.normalize
            )
            fold_model.fit(X_train, y_train)
            fold_metrics = fold_model.evaluate(X_test, y_test)

            for key in metrics:
                metrics[key].append(fold_metrics[key])

        return metrics

    def _fit_gp(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict:
        """Fit Gaussian Process."""
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import (
            ConstantKernel,
            Matern,
            WhiteKernel,
        )

        kernel = ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(noise_level=0.01)
        self._model = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=kwargs.get("n_restarts", 5),
            normalize_y=False,
            random_state=42,
        )
        self._model.fit(X, y)

        y_pred = self._model.predict(X)
        from sklearn.metrics import r2_score, mean_squared_error

        return {
            "train_R2": r2_score(y, y_pred),
            "train_RMSE": np.sqrt(mean_squared_error(y, y_pred)),
        }

    def _fit_mlp(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict:
        """Fit MLP regressor."""
        from sklearn.neural_network import MLPRegressor

        self._model = MLPRegressor(
            hidden_layer_sizes=kwargs.get("hidden", (64, 64, 32)),
            max_iter=kwargs.get("max_iter", 2000),
            early_stopping=True,
            validation_fraction=0.15,
            random_state=42,
            learning_rate_init=kwargs.get("lr", 0.001),
        )
        self._model.fit(X, y)

        y_pred = self._model.predict(X)
        from sklearn.metrics import r2_score, mean_squared_error

        return {
            "train_R2": r2_score(y, y_pred),
            "train_RMSE": np.sqrt(mean_squared_error(y, y_pred)),
        }


def train_naca0012_surrogate(model_type: str = "gp") -> Dict:
    """
    Demo: train surrogate on NACA 0012 data.

    Uses synthetic CL/CD data across angles of attack.
    """
    # Load real experimental data: Ladson NACA 0012
    data_path = PROJECT_ROOT / "experimental_data" / "naca0012" / "csv" / "ladson_forces.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Real experimental data not found: {data_path}")
        
    import pandas as pd
    df = pd.read_csv(data_path)
    
    # Ladson data is at Re = 6 million, Mach = 0.15 
    # Use alpha and Re as features
    aoa = df["alpha"].values
    Re = np.full(len(aoa), 6e6)
    
    CL = df["CL"].values
    CD = df["CD"].values

    X = np.column_stack([aoa, Re / 1e6])

    model = SurrogateModel(model_type=model_type)
    train_metrics = model.fit(X, CL)
    print(f"Training: {train_metrics}")

    # Hold-out evaluation
    aoa_test = np.array([5, 10, 12, 15])
    Re_test = np.full(4, 6.0)
    X_test = np.column_stack([aoa_test, Re_test])

    result = model.predict(X_test)
    if isinstance(result, tuple):
        pred, std = result
        print(f"Predictions: {pred}")
        print(f"Uncertainty: {std}")
    else:
        print(f"Predictions: {result}")

    # Cross-validate
    cv = model.cross_validate(X, CL, k=5)
    print(f"CV R²: {np.mean(cv['R2']):.4f} ± {np.std(cv['R2']):.4f}")

    return train_metrics


def plot_surrogate_predictions(
    model: SurrogateModel,
    X_test: np.ndarray,
    y_test: np.ndarray,
    save_path: Optional[str] = None,
):
    """Plot parity and residual plots for surrogate validation."""
    import matplotlib.pyplot as plt

    result = model.predict(X_test)
    if isinstance(result, tuple):
        y_pred, y_std = result
    else:
        y_pred = result
        y_std = None

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Parity plot
    ax = axes[0]
    ax.scatter(y_test, y_pred, c="steelblue", alpha=0.7, edgecolors="k", linewidths=0.5)
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, "k--", linewidth=1, label="Perfect prediction")
    if y_std is not None:
        ax.errorbar(y_test, y_pred, yerr=2 * y_std, fmt="none", color="gray", alpha=0.3)
    ax.set_xlabel("True value")
    ax.set_ylabel("Predicted value")
    ax.set_title("Parity Plot")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Residual plot
    ax = axes[1]
    residuals = y_pred - y_test
    ax.scatter(y_test, residuals, c="coral", alpha=0.7, edgecolors="k", linewidths=0.5)
    ax.axhline(y=0, color="k", linestyle="--", linewidth=1)
    ax.set_xlabel("True value")
    ax.set_ylabel("Residual (pred - true)")
    ax.set_title("Residual Plot")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    print("=== Surrogate Model Demo (GP) ===")
    train_naca0012_surrogate("gp")
    print("\n=== Surrogate Model Demo (MLP) ===")
    train_naca0012_surrogate("mlp")
