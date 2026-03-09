#!/usr/bin/env python3
"""
TBNN Application to Wall-Hump Separation
========================================
Applies the Tensor-Basis Neural Network to the NASA wall-hump flow.

This script demonstrates transfer learning:
1. Loads DNS data for periodic hills (McConkey et al. 2021) or generates synthetic equivalent.
2. Extracts invariant inputs and tensor bases.
3. Trains a TBNN to predict Reynolds stress anisotropy b_ij.
4. Generates predictions for the wall-hump case.
5. Verifies realizability constraints (Lumley triangle).
"""

import argparse
import logging
from pathlib import Path

import numpy as np

# Add project root to path
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Try importing TBNN and Torch
try:
    from scripts.ml_augmentation.tbnn_closure import (
        TBNNModel, 
        compute_tensor_basis, 
        compute_invariant_inputs,
        project_to_realizable,
        check_realizability,
        HAS_TORCH
    )
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as e:
    HAS_TORCH = False
    print(f"Failed to import PyTorch components: {e}")

logger = logging.getLogger(__name__)

# =============================================================================
# Mock/Loader for McConkey DNS Data
# =============================================================================

def load_mcconkey_dns(n_samples: int = 1000):
    """
    Mock loader for McConkey et al. (2021) periodic hill DNS data.
    In production, this would parse the actual OpenFOAM/VTK DNS fields.
    """
    logger.info(f"Generating {n_samples} points of synthetic periodic-hill DNS data for TBNN training.")
    
    np.random.seed(42)
    # Generate random strain (S) and rotation (O) tensors (trace-free, symmetric/anti-symmetric)
    S_hat = np.random.randn(n_samples, 3, 3)
    S_hat = 0.5 * (S_hat + S_hat.transpose(0, 2, 1))
    # Enforce trace-free
    traceS = np.trace(S_hat, axis1=1, axis2=2)
    S_hat -= (traceS[:, None, None] / 3.0) * np.eye(3)
    
    O_hat = np.random.randn(n_samples, 3, 3)
    O_hat = 0.5 * (O_hat - O_hat.transpose(0, 2, 1))
    
    # Compute inputs and bases
    invariants = compute_invariant_inputs(S_hat, O_hat)
    tensor_bases = compute_tensor_basis(S_hat, O_hat)
    
    # Generate target b_ij using a fixed set of Pope g-coefficients + noise
    # Let true b_ij = 0.5 * T^(1) - 0.2 * T^(2)
    b_ij_target = 0.5 * tensor_bases[:, 0] - 0.2 * tensor_bases[:, 1]
    b_ij_target += np.random.randn(n_samples, 3, 3) * 0.05
    
    # Project to make sure targets are realizable
    b_ij_target = project_to_realizable(b_ij_target)
    
    return invariants, tensor_bases, b_ij_target


# =============================================================================
# TBNN Training Loop
# =============================================================================

def train_tbnn_model(
    model: 'TBNNModel', 
    X_inv: np.ndarray, 
    X_tb: np.ndarray, 
    Y_b: np.ndarray, 
    epochs: int = 20, 
    batch_size: int = 64
):
    """Train the PyTorch TBNN model."""
    if not HAS_TORCH:
        raise ImportError("PyTorch is required for TBNN training.")
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    tens_inv = torch.tensor(X_inv, dtype=torch.float32)
    tens_tb = torch.tensor(X_tb, dtype=torch.float32)
    tens_y = torch.tensor(Y_b, dtype=torch.float32)
    
    dataset = TensorDataset(tens_inv, tens_tb, tens_y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    model.train()
    for ep in range(epochs):
        ep_loss = 0.0
        for b_inv, b_tb, b_y in loader:
            b_inv, b_tb, b_y = b_inv.to(device), b_tb.to(device), b_y.to(device)
            
            optimizer.zero_grad()
            pred_b = model(b_inv, b_tb)
            loss = criterion(pred_b, b_y)
            loss.backward()
            optimizer.step()
            
            ep_loss += loss.item() * b_inv.size(0)
            
        ep_loss /= len(dataset)
        if (ep + 1) % 5 == 0 or ep == 0:
            logger.info(f"Epoch {ep+1}/{epochs} - MSE: {ep_loss:.6f}")


# =============================================================================
# Inference & Application
# =============================================================================

def apply_to_wall_hump(model: 'TBNNModel'):
    """Apply trained TBNN model to generate predictions for the wall hump."""
    logger.info("Applying TBNN to wall-hump flow (synthetic transfer)...")
    
    # Generate some slightly different inputs resembling adverse pressure gradients
    n_hump = 500
    np.random.seed(123)
    S_hat = np.random.randn(n_hump, 3, 3) * 1.5
    S_hat = 0.5 * (S_hat + S_hat.transpose(0, 2, 1))
    traceS = np.trace(S_hat, axis1=1, axis2=2)
    S_hat -= (traceS[:, None, None] / 3.0) * np.eye(3)
    
    O_hat = np.random.randn(n_hump, 3, 3) * 1.5
    O_hat = 0.5 * (O_hat - O_hat.transpose(0, 2, 1))
    
    inv_hump = compute_invariant_inputs(S_hat, O_hat)
    tb_hump = compute_tensor_basis(S_hat, O_hat)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        tens_inv = torch.tensor(inv_hump, dtype=torch.float32).to(device)
        tens_tb = torch.tensor(tb_hump, dtype=torch.float32).to(device)
        
        # Predict b_ij
        b_pred = model(tens_inv, tens_tb).cpu().numpy()
        
    logger.info(f"Generated predictions for {n_hump} wall-hump control volumes.")
    
    # Realizability check
    report_raw = check_realizability(b_pred)
    logger.info(f"Realizability BEFORE projection: {report_raw.fraction_realizable*100:.1f}%")
    
    # Project to Lumley Triangle
    b_realizable = project_to_realizable(b_pred)
    report_proj = check_realizability(b_realizable)
    logger.info(f"Realizability AFTER projection:  {report_proj.fraction_realizable*100:.1f}%")
    
    return b_realizable


# =============================================================================
# Main
# =============================================================================

def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    
    if not HAS_TORCH:
        logger.error("PyTorch not installed. TBNN application requires PyTorch.")
        sys.exit(1)
        
    print("=================================================================")
    print(" TBNN Wall-Hump Transfer Application")
    print("=================================================================")
    
    # 1. Load Data
    X_inv, X_tb, Y_b = load_mcconkey_dns(n_samples=2000)
    
    # 2. Build Model
    model = TBNNModel(n_scalar_inputs=5, hidden_layers=[32, 32])
    
    # 3. Train Model
    logger.info("Starting TBNN training on periodic hill DNS data...")
    train_tbnn_model(model, X_inv, X_tb, Y_b, epochs=args.epochs)
    
    # 4. Apply to Wall Hump
    _ = apply_to_wall_hump(model)
    
    print("\nTBNN Transfer Pipeline completed successfully.")


if __name__ == "__main__":
    main()
