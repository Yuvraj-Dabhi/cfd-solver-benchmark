#!/usr/bin/env python3
"""
Bayesian Neural Network Closure
===============================
Implements a Bayesian Neural Network (BNN) for turbulence modeling closures,
providing predictive mean and epistemic uncertainty bounds.

This module provides a robust Monte Carlo Dropout (MC-Dropout) implementation
as a scalable approximation to full Variational Inference for BNNs, which is
a standard approach in deep learning for fluid dynamics (e.g., Gal & Ghahramani).

Key features:
- MC-Dropout for epistemic uncertainty estimation.
- Handles (mean, variance) predictions to construct 95% credible intervals.
- Compatible with sklearn-like APIs.
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logger = logging.getLogger(__name__)


class MCDropoutBNN(nn.Module):
    """
    Monte Carlo Dropout BNN architecture.
    A standard MLP where dropout is applied before every weight layer.
    """
    def __init__(self, n_in: int, n_out: int, hidden: List[int] = [64, 64], p_drop: float = 0.1):
        super().__init__()
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for MCDropoutBNN.")
            
        self.p_drop = p_drop
        layers = []
        in_features = n_in
        
        for h in hidden:
            layers.append(nn.Linear(in_features, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p_drop))
            in_features = h
            
        layers.append(nn.Linear(in_features, n_out))
        self.net = nn.Sequential(*layers)
        
    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.net(x)


class BayesianDNNClosure:
    """
    Scikit-learn style wrapper for Bayesian Deep Neural Network.
    
    Provides `.fit` and `.predict_with_uncertainty` methods.
    """
    def __init__(
        self, 
        n_in: int, 
        n_out: int = 1,
        hidden: List[int] = [64, 64], 
        p_drop: float = 0.1,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4, # Prior length-scale implicitly via L2
        epochs: int = 100,
        batch_size: int = 128,
        mc_samples: int = 50,
        device: str = "auto"
    ):
        self.n_in = n_in
        self.n_out = n_out
        self.hidden = hidden
        self.p_drop = p_drop
        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.mc_samples = mc_samples
        
        if not HAS_TORCH:
            logger.warning("PyTorch not installed. BayesianDNNClosure will run in dummy mode.")
            self.model = None
            self.device = None
            return

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        self.model = MCDropoutBNN(n_in, n_out, hidden, p_drop).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.lr, 
            weight_decay=self.weight_decay
        )
        self.criterion = nn.MSELoss()
        
    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = False):
        """Train the BNN."""
        if not HAS_TORCH:
            logger.warning("Dummy fit. PyTorch missing.")
            return self
            
        dataset = torch.utils.data.TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32)
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                self.optimizer.zero_grad()
                pred = self.model(batch_x)
                if pred.shape != batch_y.shape:
                    if batch_y.ndim == 1 and pred.shape[1] == 1:
                        batch_y = batch_y.unsqueeze(1)
                
                loss = self.criterion(pred, batch_y)
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                
            if verbose and (epoch + 1) % max(1, self.epochs // 10) == 0:
                logger.info(f"BNN Epoch {epoch+1}/{self.epochs} | Loss: {epoch_loss/len(loader):.4e}")
                
        return self
        
    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Produce predictions with model epistemic uncertainty via MC-Dropout sampling.
        
        Parameters
        ----------
        X : np.ndarray
            Features (N, n_in)
            
        Returns
        -------
        mean : np.ndarray (N, n_out)
        epistemic_variance : np.ndarray (N, n_out)
        aleatoric_variance : np.ndarray (N, n_out)  (assumed homoscedastic noise here)
        """
        if not HAS_TORCH:
            logger.warning("Dummy BNN predict_with_uncertainty.")
            N = X.shape[0]
            # Dummy data for missing PyTorch env
            return (
                np.zeros((N, self.n_out)),
                np.ones((N, self.n_out)) * 0.1,
                np.ones((N, self.n_out)) * 0.05
            )
            
        self.model.train()  # Critically important: leave dropout ON!
        
        # We need inverse precision for aleatoric (tau), approximating from weight decay
        # following Gal & Ghahramani (2016).
        # tau^-1 ~ 2 * N_train * weight_decay / (1 - p_drop)
        # Using a fixed homoscedastic estimate for simplicity here.
        aleatoric_var = 0.01  
        
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        preds = []
        
        with torch.no_grad():
            for _ in range(self.mc_samples):
                p = self.model(X_t).cpu().numpy()
                preds.append(p)
                
        preds = np.stack(preds, axis=0) # (MC, N, n_out)
        
        mean = np.mean(preds, axis=0)
        epistemic_var = np.var(preds, axis=0)
        
        aleatoric_var_arr = np.ones_like(epistemic_var) * aleatoric_var
        
        return mean, epistemic_var, aleatoric_var_arr

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Standard prediction (returns only mean)."""
        mean, _, _ = self.predict_with_uncertainty(X)
        return mean


if __name__ == "__main__":
    print("Testing Bayesian DNN Closure (MC-Dropout)...")
    
    np.random.seed(42)
    X = np.sort(np.random.rand(200, 3) * 10, axis=0)
    y = np.sin(X[:, 0]) + np.random.randn(200) * 0.1
    y = y.reshape(-1, 1)
    
    bnn = BayesianDNNClosure(n_in=3, n_out=1, hidden=[64, 64], epochs=300, learning_rate=5e-3)
    bnn.fit(X, y, verbose=False)
    
    mean, epi_var, alea_var = bnn.predict_with_uncertainty(X)
    
    print(f"Mean shape: {mean.shape}")
    print(f"Epi_var shape: {epi_var.shape}")
    print(f"Epi_var mean scale: {np.mean(epi_var):.4f}")
    
    # 95% Confidence interval
    total_std = np.sqrt(epi_var + alea_var)
    upper = mean + 1.96 * total_std
    lower = mean - 1.96 * total_std
    
    coverage = np.mean((y >= lower) & (y <= upper))
    print(f"Empirical 95% Coverage: {coverage*100:.1f}%")
    assert coverage > 0.70
    print("BNN Component Test Passed.")
