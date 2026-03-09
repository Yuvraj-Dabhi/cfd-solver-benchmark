#!/usr/bin/env python3
"""
Multi-Fidelity Operator Learning Harness
========================================
Trains and compares DeepONet, FNO, and HUFNO surrogate models on
synthetic field data (mimicking transonic airfoil physics).

1. Generates transonic training data (Cp fields).
2. Trains DeepONet, FNO, and HUFNO models.
3. Compares their relative L2 errors and plots the results.
"""

import sys
import argparse
import logging
from pathlib import Path
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.ml_augmentation.deeponet_surrogate import (
    DeepONetSurrogate, DeepONetConfig, generate_transonic_airfoil_data
)
from scripts.ml_augmentation.neural_operator_surrogate import (
    NeuralOperatorSurrogate, GridPairDataset
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Operator Learning Benchmark")
    parser.add_argument("--samples", type=int, default=100,
                        help="Number of synthetic samples (default: 100 for demo)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs per model")
    return parser.parse_args()


def run_benchmark():
    args = parse_args()
    output_dir = Path("results/operator_learning")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # -------------------------------------------------------------------------
    # 1. Generate Data (Transonic Airfoils: Mach and AoA variations)
    # -------------------------------------------------------------------------
    logger.info(f"Generating {args.samples} transonic airfoil samples...")
    data = generate_transonic_airfoil_data(n_samples=args.samples, seed=42)
    
    # DeepONet format
    u_branch = data["input_functions"]  # (N, n_sensors)
    y_trunk = data["query_coords"]      # (n_query, 1)
    target_cp = data["target_fields"]   # (N, n_outputs, n_query)
    
    # FNO format: requires coarse & fine fields for super-resolution training.
    # To mimic this, we will pass the branch input as the "coarse" field
    # and the target as the "fine" field, along with flow parameters for FiLM.
    flow_params = data["flow_params"]   # (N, 2) [AoA, Mach]
    
    # Reshape u_branch for FNO input: (N, 1, n_sensors)
    U_in_fno = np.expand_dims(u_branch, axis=1)
    
    # Train/Val split: 80/20
    n_train = int(0.8 * args.samples)
    
    u_train, u_val = u_branch[:n_train], u_branch[n_train:]
    U_in_fno_train, U_in_fno_val = U_in_fno[:n_train], U_in_fno[n_train:]
    target_train, target_val = target_cp[:n_train], target_cp[n_train:]
    params_train, params_val = flow_params[:n_train], flow_params[n_train:]
    
    # -------------------------------------------------------------------------
    # 2. Train DeepONet
    # -------------------------------------------------------------------------
    logger.info(f"\nTraining DeepONet ({args.epochs} epochs)...")
    config_don = DeepONetConfig(
        branch_input_dim=u_branch.shape[1],
        trunk_input_dim=y_trunk.shape[1],
        n_epochs=args.epochs,
        batch_size=32,
        lr=5e-4
    )
    don_model = DeepONetSurrogate(config_don)
    don_history = don_model.fit(u_train, y_trunk, target_train)
    
    # Validate DeepONet
    don_pred = don_model.predict(u_val, y_trunk)
    don_err = relative_l2_loss(don_pred, target_val)
    logger.info(f"DeepONet Validation Error: {don_err:.4f}")

    # -------------------------------------------------------------------------
    # 3. Train FNO (Fourier Neural Operator)
    # -------------------------------------------------------------------------
    logger.info(f"\nTraining FNO2d ({args.epochs} epochs)...")
    fno_model = NeuralOperatorSurrogate(
        arch="fno",
        in_channels=1,
        out_channels=1,
        cond_dim=2,      # [AoA, Mach]
        n_modes=8,
        n_layers=3
    )
    fno_history = fno_model.fit(params_train, U_in_fno_train, target_train, n_epochs=args.epochs, lr=1e-3)
    
    # Validate FNO
    fno_pred, _ = fno_model.predict_at_resolution(params_val, U_in_fno_val, target_res=target_val.shape[-1])
    fno_err = relative_l2_loss(np.expand_dims(fno_pred, axis=1), target_val)
    logger.info(f"FNO Validation Error: {fno_err:.4f}")

    # -------------------------------------------------------------------------
    # 4. Train HUFNO (Hybrid U-Net + FNO)
    # -------------------------------------------------------------------------
    logger.info(f"\nTraining HUFNO ({args.epochs} epochs)...")
    hufno_model = NeuralOperatorSurrogate(
        arch="hufno",
        in_channels=1,
        out_channels=1,
        cond_dim=2,      # [AoA, Mach]
        n_modes=6,
        n_layers=2
    )
    hufno_history = hufno_model.fit(params_train, U_in_fno_train, target_train, n_epochs=args.epochs, lr=1e-3)
    
    # Validate HUFNO
    hufno_pred, _ = hufno_model.predict_at_resolution(params_val, U_in_fno_val, target_res=target_val.shape[-1])
    hufno_err = relative_l2_loss(np.expand_dims(hufno_pred, axis=1), target_val)
    logger.info(f"HUFNO Validation Error: {hufno_err:.4f}")

    # -------------------------------------------------------------------------
    # 5. Save & Print Results
    # -------------------------------------------------------------------------
    from tabulate import tabulate
    table_data = [
        ["DeepONet", don_err],
        ["FNO2d", fno_err],
        ["HUFNO", hufno_err]
    ]
    
    print("\n" + "="*50)
    print("      Operator Learning Validation Errors")
    print("="*50)
    print(tabulate(table_data, headers=["Model Architecture", "Relative L2 Error"], floatfmt=".4f"))
    print("="*50 + "\n")
    
    import json
    results = {
        "DeepONet": {"val_err": don_err, "history": don_history},
        "FNO2d": {"val_err": fno_err, "history": fno_history},
        "HUFNO": {"val_err": hufno_err, "history": hufno_history}
    }
    with open(output_dir / "operator_validation.json", "w") as f:
        json.dump(results, f, indent=2)


def relative_l2_loss(pred: np.ndarray, target: np.ndarray) -> float:
    """Relative L2 error over testing set."""
    diff = pred - target
    num = np.sqrt(np.mean(diff ** 2))
    den = np.sqrt(np.mean(target ** 2)) + 1e-8
    return float(num / den)


if __name__ == "__main__":
    run_benchmark()
