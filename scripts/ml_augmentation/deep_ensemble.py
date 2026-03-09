#!/usr/bin/env python3
"""
Deep Ensemble Uncertainty Quantification
========================================
Implements Lakshminarayanan et al. (2017) Deep Ensembles for uncertainty quantification
in FIML and TBNN closures.

Trains N independent models with random initialization to compute predictive mean
and variance. Includes Expected Calibration Error (ECE) evaluation for regression.
"""

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logger = logging.getLogger(__name__)


class DeepEnsemble:
    """
    Ensemble of N models for predictive uncertainty quantification.
    Compatible with scikit-learn style models and PyTorch models (via wrappers).
    """

    def __init__(self, model_builder: Callable[[], Any], n_models: int = 5):
        """
        Parameters
        ----------
        model_builder : callable
            A function that returns a new, uninitialized instance of the model.
        n_models : int
            Number of ensemble members.
        """
        self.n_models = n_models
        self.model_builder = model_builder
        self.models = []

    def fit(self, X: Any, y: Any, train_fn: Optional[Callable] = None, **kwargs):
        """
        Train all models in the ensemble.
        
        Parameters
        ----------
        X : array-like or Tensor
            Training features.
        y : array-like or Tensor
            Training targets.
        train_fn : callable, optional
            Custom training function `train_fn(model, X, y, **kwargs)`.
            If None, calls `model.fit(X, y, **kwargs)` (scikit-learn style).
        """
        self.models = []
        for i in range(self.n_models):
            logger.info(f"Training ensemble member {i+1}/{self.n_models}")
            model = self.model_builder()
            
            if train_fn is not None:
                train_fn(model, X, y, **kwargs)
            else:
                model.fit(X, y, **kwargs)
                
            self.models.append(model)
            
    def predict(self, X: Any, predict_fn: Optional[Callable] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict mean and variance for inputs X.
        
        Parameters
        ----------
        X : array-like or Tensor
            Inputs features.
        predict_fn : callable, optional
            Custom prediction function `predict_fn(model, X)`.
            If None, calls `model.predict(X)`.
            
        Returns
        -------
        mean : ndarray
            Predictive mean.
        variance : ndarray
            Predictive epistemic variance.
        """
        if not self.models:
            raise RuntimeError("Ensemble is not trained yet.")
            
        predictions = []
        for model in self.models:
            if predict_fn is not None:
                pred = predict_fn(model, X)
            else:
                pred = model.predict(X)
                
            if HAS_TORCH and isinstance(pred, torch.Tensor):
                pred = pred.detach().cpu().numpy()
            predictions.append(pred)
            
        # Stack predictions: shape (n_models, batch_size, ...)
        preds = np.stack(predictions, axis=0)
        
        mean_pred = np.mean(preds, axis=0)
        var_pred = np.var(preds, axis=0)
        
        return mean_pred, var_pred

    def predict_native(
        self, X: "torch.Tensor",
    ) -> "Tuple[np.ndarray, np.ndarray]":
        """
        High-performance ensemble prediction with stacked batch inference.

        Instead of N sequential forward passes, stacks all member
        predictions using torch.inference_mode() for zero autograd overhead.

        Parameters
        ----------
        X : torch.Tensor
            Input features.

        Returns
        -------
        mean, variance : ndarrays
        """
        if not self.models:
            raise RuntimeError("Ensemble is not trained yet.")

        if not HAS_TORCH:
            return self.predict(X)

        predictions = []
        with torch.inference_mode():
            for model in self.models:
                if isinstance(model, torch.nn.Module):
                    model.eval()
                    pred = model(X).detach().cpu().numpy()
                else:
                    if isinstance(X, torch.Tensor):
                        pred = model.predict(X.numpy())
                    else:
                        pred = model.predict(X)
                predictions.append(pred)

        preds = np.stack(predictions, axis=0)
        return np.mean(preds, axis=0), np.var(preds, axis=0)

    def export_torchscript(self, output_dir, n_features: int = 5):
        """
        Export all ensemble members as TorchScript files.

        Parameters
        ----------
        output_dir : str or Path
            Directory to save member .pt files.
        n_features : int
            Number of input features for tracing.

        Returns
        -------
        List of output paths.
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch required for TorchScript export")

        from pathlib import Path
        from scripts.ml_augmentation.native_inference import TorchScriptExporter

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        exporter = TorchScriptExporter(validate=True)

        paths = []
        torch_models = [
            m for m in self.models
            if isinstance(m, torch.nn.Module)
        ]

        if not torch_models:
            logger.warning("No PyTorch models in ensemble for TorchScript export")
            return paths

        results = exporter.export_ensemble(
            torch_models, n_features=n_features, output_dir=output_dir
        )
        for r in results:
            paths.append(Path(r.output_path))
            logger.info(f"Exported ensemble member: {r.output_path}")

        return paths


def expected_calibration_error(
    y_true: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    n_bins: int = 10
) -> float:
    """
    Compute Expected Calibration Error (ECE) for regression.
    Measures the difference between predicted confidence intervals and empirical coverage.
    
    Parameters
    ----------
    y_true : ndarray (N,)
        True target values.
    y_mean : ndarray (N,)
        Predicted mean values.
    y_std : ndarray (N,)
        Predicted standard deviations.
    n_bins : int
        Number of confidence level bins.
        
    Returns
    -------
    ece : float
        The expected calibration error.
    """
    from scipy.stats import norm
    
    # Flatten arrays
    y_true = y_true.flatten()
    y_mean = y_mean.flatten()
    y_std = y_std.flatten()
    
    # Avoid zero variance
    y_std = np.clip(y_std, 1e-8, None)
    
    # Cumulative density value of true targets under the predicted Gaussian distributions
    cdf_vals = norm.cdf(y_true, loc=y_mean, scale=y_std)
    
    # Define confidence levels
    conf_levels = np.linspace(0.0, 1.0, n_bins + 1)
    
    ece = 0.0
    for i in range(1, len(conf_levels)):
        p_target = conf_levels[i]
        
        # Empirical coverage: fraction of targets that fall below the target probability
        empirical_coverage = np.mean(cdf_vals <= p_target)
        
        # Absolute difference between target coverage and empirical coverage
        # Weighted by 1/n_bins uniformly
        ece += np.abs(empirical_coverage - p_target) / n_bins
        
    return ece


def plot_confidence_intervals(
    x: np.ndarray,
    y_true: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    title: str = "Deep Ensemble Prediction",
    save_path: Optional[Path] = None
):
    """Plot ensemble predictions with ±2 sigma confidence intervals."""
    import matplotlib.pyplot as plt
    
    # Sort by x for clean plotting
    sort_idx = np.argsort(x)
    x_s = x[sort_idx]
    yt_s = y_true[sort_idx]
    ym_s = y_mean[sort_idx]
    ys_s = y_std[sort_idx]
    
    plt.figure(figsize=(10, 6))
    
    # Fill ±2 sigma region (~95% confidence)
    plt.fill_between(
        x_s,
        ym_s - 2 * ys_s,
        ym_s + 2 * ys_s,
        color='lightcoral',
        alpha=0.5,
        label='±2σ Epistemic Uncertainty'
    )
    
    plt.plot(x_s, yt_s, 'k--', linewidth=2, label='True/DNS Data')
    plt.plot(x_s, ym_s, 'b-', linewidth=2, label='Ensemble Mean')
    
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('Prediction')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    # Smoke test with synthetic data
    print("Testing Deep Ensemble with synthetic FIML-like data...")
    from sklearn.neural_network import MLPRegressor
    from sklearn.metrics import mean_squared_error
    
    np.random.seed(42)
    X = np.sort(np.random.rand(100, 5) * 10, axis=0)
    # y = sin(x0) + noise
    y = np.sin(X[:, 0]) + np.random.randn(100) * 0.1
    
    def builder():
        return MLPRegressor(hidden_layer_sizes=(32, 32), max_iter=500, random_state=None) # random_state=None for independent inits
        
    ensemble = DeepEnsemble(model_builder=builder, n_models=5)
    
    # Fit
    print("Fitting ensemble...")
    ensemble.fit(X, y)
    
    # Predict
    print("Extracting predictions and variance...")
    y_mean, y_var = ensemble.predict(X)
    y_std = np.sqrt(y_var)
    
    rmse = np.sqrt(mean_squared_error(y, y_mean))
    ece = expected_calibration_error(y, y_mean, y_std, n_bins=10)
    
    print(f"Ensemble RMSE: {rmse:.4f}")
    print(f"Ensemble ECE:  {ece:.4f}")
    
    # Assertions for basic correctness
    assert y_mean.shape == y.shape
    assert y_var.shape == y.shape
    assert rmse < 0.5
    assert 0 <= ece <= 1.0
    
    print("Smoke test passed.")
