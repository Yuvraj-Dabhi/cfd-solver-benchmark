#!/usr/bin/env python3
"""
Conformal Prediction UQ Benchmark
==================================
Demonstrates distribution-free uncertainty quantification on
CFD surrogate model predictions using three conformal methods:

1. Split Conformal Prediction (SCP)
2. Conformalized Quantile Regression (CQR)
3. Conformal Jackknife+ (J+)

Also runs OOD (out-of-distribution) detection to flag extrapolation
regions where the surrogate is unreliable.

Validates the finite-sample coverage guarantee: P(Y ∈ C(X)) ≥ 1 − α
"""

import sys
import logging
import argparse
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.ml_augmentation.conformal_prediction import (
    SplitConformalPredictor,
    ConformalizedQuantileRegression,
    ConformalJackknifeplus,
    OODFlowDetector,
    AbsoluteResidualScore,
    NormalizedResidualScore,
    conformalize_surrogate,
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Conformal Prediction UQ Benchmark")
    parser.add_argument("--alpha", type=float, default=0.1,
                        help="Miscoverage rate (default: 0.1 → 90%% coverage)")
    parser.add_argument("--n-samples", type=int, default=500,
                        help="Total synthetic samples")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def generate_cfd_surrogate_data(n_samples: int, seed: int = 42):
    """
    Simulate a CFD surrogate that predicts C_L from (AoA, Re, Mach).
    
    Ground truth: C_L ≈ 2π·sin(α) · (1 + 0.1·log10(Re/1e6)) · β⁻¹
    where β = sqrt(1 - M²) (Prandtl-Glauert correction).
    
    The surrogate adds heteroscedastic noise (larger at high AoA).
    """
    rng = np.random.default_rng(seed)
    
    aoa_deg = rng.uniform(-2, 14, n_samples)
    Re = rng.uniform(1e6, 9e6, n_samples)
    Mach = rng.uniform(0.1, 0.7, n_samples)
    
    X = np.column_stack([aoa_deg, Re / 1e6, Mach])  # Normalize Re
    
    # True C_L (thin airfoil + compressibility + Re correction)
    aoa_rad = np.radians(aoa_deg)
    beta = np.sqrt(np.maximum(1 - Mach**2, 0.01))
    CL_true = 2 * np.pi * np.sin(aoa_rad) * (1 + 0.1 * np.log10(Re / 1e6)) / beta
    
    # Surrogate prediction = truth + heteroscedastic noise
    noise_scale = 0.02 + 0.01 * np.abs(aoa_deg)  # More noise at high AoA
    CL_pred = CL_true + rng.normal(0, noise_scale)
    
    return X, CL_true, CL_pred


def run_benchmark():
    args = parse_args()
    output_dir = Path("results/conformal_uq")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    alpha = args.alpha
    target_coverage = 1.0 - alpha
    
    logger.info(f"Generating {args.n_samples} surrogate samples (α={alpha}, target coverage={target_coverage:.0%})...")
    X, y_true, y_pred = generate_cfd_surrogate_data(args.n_samples, args.seed)
    
    # Split: 60% train, 20% calibration, 20% test
    n = len(X)
    n_train = int(0.6 * n)
    n_cal = int(0.2 * n)
    
    X_train, y_train, y_pred_train = X[:n_train], y_true[:n_train], y_pred[:n_train]
    X_cal, y_cal, y_pred_cal = X[n_train:n_train+n_cal], y_true[n_train:n_train+n_cal], y_pred[n_train:n_train+n_cal]
    X_test, y_test, y_pred_test = X[n_train+n_cal:], y_true[n_train+n_cal:], y_pred[n_train+n_cal:]
    
    results = {}
    
    # -------------------------------------------------------------------------
    # 1. Split Conformal Prediction (SCP)
    # -------------------------------------------------------------------------
    logger.info("\n[1/4] Split Conformal Prediction...")
    scp = SplitConformalPredictor(alpha=alpha, score_fn=AbsoluteResidualScore())
    scp.calibrate(y_pred_cal, y_cal)
    scp_interval = scp.predict_interval(y_pred_test)
    
    scp_cov = scp_interval.coverage(y_test)
    scp_width = scp_interval.mean_width()
    results["SCP"] = {"coverage": scp_cov, "mean_width": scp_width}
    logger.info(f"  Coverage: {scp_cov:.1%} (target ≥ {target_coverage:.0%})")
    logger.info(f"  Mean width: {scp_width:.4f}")
    
    # -------------------------------------------------------------------------
    # 2. Conformalized Quantile Regression (CQR)
    # -------------------------------------------------------------------------
    logger.info("\n[2/4] Conformalized Quantile Regression...")
    cqr = ConformalizedQuantileRegression(
        alpha=alpha, input_dim=X.shape[1], hidden_dim=32,
        n_epochs=100, seed=args.seed
    )
    cqr.fit(X_train, y_train)
    cqr.calibrate(X_cal, y_cal)
    cqr_interval = cqr.predict_interval(X_test, y_point=y_pred_test)
    
    cqr_cov = cqr_interval.coverage(y_test)
    cqr_width = cqr_interval.mean_width()
    results["CQR"] = {"coverage": cqr_cov, "mean_width": cqr_width}
    logger.info(f"  Coverage: {cqr_cov:.1%} (target ≥ {target_coverage:.0%})")
    logger.info(f"  Mean width: {cqr_width:.4f}")
    
    # -------------------------------------------------------------------------
    # 3. Conformal Jackknife+ (J+)
    # -------------------------------------------------------------------------
    logger.info("\n[3/4] Conformal Jackknife+...")
    # Use a small calibration set to demonstrate J+'s strength
    n_jp = min(50, n_cal)
    jp = ConformalJackknifeplus(alpha=alpha)
    jp.calibrate(X_cal[:n_jp], y_cal[:n_jp])
    jp_interval = jp.predict_interval(X_test)
    
    jp_cov = jp_interval.coverage(y_test)
    jp_width = jp_interval.mean_width()
    results["Jackknife+"] = {"coverage": jp_cov, "mean_width": jp_width}
    logger.info(f"  Coverage: {jp_cov:.1%} (target ≥ {target_coverage:.0%})")
    logger.info(f"  Mean width: {jp_width:.4f}")
    
    # -------------------------------------------------------------------------
    # 4. OOD Detection
    # -------------------------------------------------------------------------
    logger.info("\n[4/4] Out-of-Distribution Detection...")
    ood_detector = OODFlowDetector(scp, threshold_factor=2.0)
    ood_report = ood_detector.detect(y_pred_test)
    
    results["OOD"] = {
        "ood_fraction": float(ood_report.ood_fraction),
        "threshold": float(ood_report.threshold),
        "n_flagged": int(np.sum(ood_report.ood_flags)),
        "n_total": len(ood_report.ood_flags)
    }
    logger.info(f"  OOD fraction: {ood_report.ood_fraction:.1%}")
    logger.info(f"  Flagged {np.sum(ood_report.ood_flags)} / {len(ood_report.ood_flags)} points")
    
    # -------------------------------------------------------------------------
    # 5. Summary Table
    # -------------------------------------------------------------------------
    from tabulate import tabulate
    
    guarantee_check = lambda cov: "PASS" if cov >= target_coverage else "WARN"
    
    table_data = [
        ["Split CP", f"{scp_cov:.1%}", f"{scp_width:.4f}", guarantee_check(scp_cov)],
        ["CQR", f"{cqr_cov:.1%}", f"{cqr_width:.4f}", guarantee_check(cqr_cov)],
        ["Jackknife+", f"{jp_cov:.1%}", f"{jp_width:.4f}", guarantee_check(jp_cov)],
    ]
    
    print("\n" + "="*60)
    print(f"  Conformal Prediction Benchmark (alpha={alpha}, target >= {target_coverage:.0%})")
    print("="*60)
    print(tabulate(table_data, headers=["Method", "Coverage", "Mean Width", "Guarantee"]))
    print("="*60)
    print(f"  OOD Detection: {ood_report.ood_fraction:.1%} of test points flagged")
    print("="*60 + "\n")
    
    # Save results
    import json
    with open(output_dir / "conformal_results.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {output_dir / 'conformal_results.json'}")


if __name__ == "__main__":
    run_benchmark()
