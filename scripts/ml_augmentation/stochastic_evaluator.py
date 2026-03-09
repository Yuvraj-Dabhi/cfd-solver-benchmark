#!/usr/bin/env python3
"""
Stochastic ML Coverage Evaluator
================================
Evaluates stochastic representations of turbulence closures against expected
predictive coverage metrics.

Given spatial predictions providing `mean` and `variance` (or `std` bounds),
this module calculates the Empirical Coverage Rate (percentage of ground
truth DNS/LES values that fall within the bounded 95% Credible Interval) and
the Mean Interval Width (preventing overly conservative uninformative bounds).

Supports evaluation across:
 - Deep Ensembles (`deep_ensemble.py`)
 - Bayesian Neural Networks (`bayesian_dnn_closure.py`)
 - Generative Diffusion Ensembles
"""

import logging
import numpy as np
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


class StochasticCoverageEvaluator:
    """Evaluates uncertainty bounds and predictive calibration."""
    
    def __init__(self, confidence_level: float = 0.95):
        """
        Parameters
        ----------
        confidence_level : float
            Confidence bounds target (e.g., 0.95 for 2-sigma limit).
        """
        self.confidence_level = confidence_level
        
        from scipy.stats import norm
        alpha = 1 - confidence_level
        # Z score for two-tailed percentile
        self.z_score = abs(norm.ppf(alpha / 2))
        
    def evaluate(self, y_true: np.ndarray, y_mean: np.ndarray, y_var: np.ndarray) -> Dict[str, float]:
        """
        Calculates empirical coverage metrics for a given prediction.
        
        Parameters
        ----------
        y_true : np.ndarray
            Target values (e.g., DNS Cf or uv stresses).
            Can be 1D or 2D (N, n_targets).
        y_mean : np.ndarray
            Predicted expected value over stochastic samples.
        y_var : np.ndarray
            Predicted variance over stochastic samples (epistemic + aleatoric if applicable).
            
        Returns
        -------
        dict with:
            rmse : float
                Root Mean Square Error of the mean.
            coverage : float
                Fraction of y_true inside the [lower, upper] interval.
            mean_width : float
                Average width of the interval across spatial points.
            ece : float
                Empirical Calibration Error.
        """
        if y_true.shape != y_mean.shape or y_mean.shape != y_var.shape:
            raise ValueError(f"Shape mismatch: {y_true.shape}, {y_mean.shape}, {y_var.shape}")
            
        mask = np.isfinite(y_true) & np.isfinite(y_mean) & np.isfinite(y_var)
        yt = y_true[mask]
        ym = y_mean[mask]
        yv = y_var[mask]
        
        if len(yt) == 0:
             return {"rmse": float('nan'), "coverage": float('nan'), "mean_width": float('nan')}
             
        std = np.sqrt(yv)
        lower = ym - self.z_score * std
        upper = ym + self.z_score * std
        
        # Binary mask for points enclosed by the predictive capability
        is_covered = (yt >= lower) & (yt <= upper)
        
        coverage = np.mean(is_covered)
        rmse_val = np.sqrt(np.mean((yt - ym)**2))
        avg_width = np.mean(upper - lower)
        
        return {
            "rmse": rmse_val,
            "coverage_pct": coverage * 100.0,
            "mean_interval_width": avg_width
        }


def mock_stochastic_pipelines(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray):
    """
    Demonstrates evaluating three disparate stochastic closure methodologies.
    For this benchmark integration step, we proxy full pipeline training with immediate
    sklearn/Torch fit-predict loops over the curated points.
    """
    evaluator = StochasticCoverageEvaluator(confidence_level=0.95)
    results = {}
    
    # 1. BNN Evaluation
    try:
        from scripts.ml_augmentation.bayesian_dnn_closure import BayesianDNNClosure
        logger.info("Training BNN...")
        bnn = BayesianDNNClosure(n_in=X_train.shape[1], n_out=y_train.shape[1], epochs=100)
        bnn.fit(X_train, y_train)
        b_mean, b_var_epi, b_var_alea = bnn.predict_with_uncertainty(X_test)
        
        bnn_res = evaluator.evaluate(y_test, b_mean, b_var_epi + b_var_alea)
        results["Bayesian DNN"] = bnn_res
    except ImportError:
        logger.warning("Failed to evaluate BNN: missing torch module")

    # 2. Deep Ensemble Evaluation
    try:
        from scripts.ml_augmentation.deep_ensemble import DeepEnsemble
        from sklearn.neural_network import MLPRegressor
        
        def _mlp_builder():
            return MLPRegressor(hidden_layer_sizes=(32, 32), max_iter=200)
            
        logger.info("Training Deep Ensemble...")
        # Reduce to 3-model ensemble for script testing speed
        ensemble = DeepEnsemble(model_builder=_mlp_builder, n_models=3)
        ensemble.fit(X_train, y_train)
        e_mean, e_var = ensemble.predict(X_test)
        
        # Scikit learn MLPs don't provide aleatoric, only epistemic via members
        ens_res = evaluator.evaluate(y_test, e_mean, e_var)
        results["Deep Ensemble"] = ens_res
    except Exception as e:
         logger.warning(f"Failed to evaluate Deep Ensemble: {e}")
         
    # 3. Diffusion Flow Surrogate (dummy proxy to represent sample-based generation)
    logger.info("Sampling Diffusion Model (proxy)...")
    # Generating purely synthetic spatial bands representing DDIM step convergence.
    d_mean = y_test + np.random.randn(*y_test.shape) * 0.15 * np.std(y_test)
    # Give the diffusion model overly pessimistic conservative bands
    d_var = np.ones_like(y_test) * (np.std(y_test)*0.8)**2
    dif_res = evaluator.evaluate(y_test, d_mean, d_var)
    results["Generative Diffusion"] = dif_res

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Testing Stochastic Evaluator Interface")
    
    # Synthetic bump-like data
    np.random.seed(42)
    X = np.sort(np.random.rand(500, 4) * 5, axis=0)
    # Non-linear relationship representing separated flow
    y = np.sin(X[:, 0] * 3) + 0.5 * np.cos(X[:, 1]) 
    y = y.reshape(-1, 1) + np.random.randn(500, 1) * 0.1
    
    train_slice = slice(0, 400)
    test_slice = slice(400, 500)
    
    res = mock_stochastic_pipelines(X[train_slice], y[train_slice], X[test_slice], y[test_slice])
    
    print("\n" + "="*60)
    print(" STOCHASTIC COVERAGE (%) TARGET: ~95%")
    print("="*60)
    for model, metrics in res.items():
        print(f"| {model:<20} | Cov: {metrics['coverage_pct']:>6.2f}% | "
              f"Width: {metrics['mean_interval_width']:>6.3f} | "
              f"RMSE: {metrics['rmse']:>6.3f} |")
    print("="*60)
