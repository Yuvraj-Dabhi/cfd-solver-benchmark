#!/usr/bin/env python3
"""
Generative Super-Resolution for Flow Fields — Benchmark Harness
================================================================
Demonstrates the RANS-to-LES super-resolution pipeline using
physics-informed generation with spectral constraints and
TBNN-conditioned diffusion.

Pipeline:
  1. Synthesize a coarse RANS field (turbulent channel flow)
  2. Super-resolve to fine grid using PhysicsInformedGenerator
  3. Condition on TBNN anisotropy tensor
  4. Evaluate: spectral compliance, continuity residual, discriminator score
  5. Compare upscale factors: 2x, 4x, 8x

Usage:
    python simulations/run_diffusion_superres.py
    python simulations/run_diffusion_superres.py --coarse-res 16 --fine-res 64

Author: Yuvraj Singh
Date:   March 2026
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Synthetic RANS Field Generator
# =============================================================================
def generate_synthetic_rans_field(nx, ny, nz, seed=42):
    """
    Generate a synthetic RANS-like velocity + pressure field for testing.

    Simulates a turbulent channel flow with:
      - Parabolic mean streamwise velocity (u)
      - Small crossflow perturbations (v, w)
      - Pressure distribution with streamwise gradient

    Returns
    -------
    velocity : ndarray (3, nx, ny, nz)
    pressure : ndarray (nx, ny, nz)
    """
    rng = np.random.default_rng(seed)

    y = np.linspace(0, 2, ny)  # wall-normal, 0 = wall, 2 = channel center
    x = np.linspace(0, 4 * np.pi, nx)
    z = np.linspace(0, 2 * np.pi, nz)

    Y, X, Z = np.meshgrid(y, x, z, indexing="ij")
    # Rearrange to (nx, ny, nz)
    X = X.transpose(1, 0, 2)
    Y = Y.transpose(1, 0, 2)
    Z = Z.transpose(1, 0, 2)

    # Mean streamwise velocity (parabolic profile)
    u_mean = 1.5 * (1.0 - (Y / 2.0 - 0.5) ** 2)

    # Add turbulent-like fluctuations (small-scale)
    u = u_mean + 0.05 * rng.standard_normal((nx, ny, nz))
    v = 0.01 * np.sin(2 * np.pi * X / (4 * np.pi)) * np.cos(np.pi * Y / 2) \
        + 0.02 * rng.standard_normal((nx, ny, nz))
    w = 0.01 * np.cos(2 * np.pi * Z / (2 * np.pi)) \
        + 0.02 * rng.standard_normal((nx, ny, nz))

    velocity = np.stack([u, v, w], axis=0)  # (3, nx, ny, nz)

    # Pressure: streamwise gradient + fluctuations
    pressure = -0.1 * X / (4 * np.pi) + 0.01 * rng.standard_normal((nx, ny, nz))

    return velocity, pressure


# =============================================================================
# Anisotropy Tensor Generator
# =============================================================================
def generate_synthetic_anisotropy(nx, ny, nz, seed=42):
    """
    Generate synthetic Reynolds stress anisotropy tensor b_ij.

    Models typical channel flow anisotropy:
      - Near-wall: strong b_11 (streamwise dominance)
      - Center: nearly isotropic (b_ij -> 0)
    """
    rng = np.random.default_rng(seed)
    n_pts = nx * ny * nz
    b = np.zeros((n_pts, 3, 3))

    y = np.linspace(0, 2, ny)
    # Create structured y-coordinate for all points
    y_all = np.tile(np.repeat(y, nz), nx)

    # Near-wall anisotropy factor
    wall_factor = np.exp(-2 * np.minimum(y_all, 2 - y_all))

    # b_11 dominant near walls
    b[:, 0, 0] = 0.2 * wall_factor + 0.01 * rng.standard_normal(n_pts)
    b[:, 1, 1] = -0.1 * wall_factor + 0.005 * rng.standard_normal(n_pts)
    b[:, 2, 2] = -0.1 * wall_factor + 0.005 * rng.standard_normal(n_pts)
    b[:, 0, 1] = -0.08 * wall_factor + 0.005 * rng.standard_normal(n_pts)
    b[:, 1, 0] = b[:, 0, 1]  # Symmetric

    return b


# =============================================================================
# Benchmark Runner
# =============================================================================
def run_benchmark(coarse_res=16, fine_res=64):
    """Run the end-to-end generative super-resolution benchmark."""
    from scripts.ml_augmentation.generative_super_resolution import (
        SuperResolutionConfig,
        RANStoLESMapper,
        FlowFieldVoxelizer,
        TBNNConditionedDiffusion,
        SuperResolutionDiscriminator,
    )

    upscale = fine_res // coarse_res

    print("=" * 65)
    print("  Generative Super-Resolution for Flow Fields")
    print("=" * 65)
    print(f"  Coarse grid:    ({coarse_res}, {coarse_res}, {coarse_res})")
    print(f"  Fine grid:      ({fine_res}, {fine_res}, {fine_res})")
    print(f"  Upscale factor: {upscale}x")
    print()

    # -------------------------------------------------------------------------
    # 1. Generate synthetic coarse RANS field
    # -------------------------------------------------------------------------
    logger.info("[1/5] Generating synthetic RANS field...")
    velocity, pressure = generate_synthetic_rans_field(
        coarse_res, coarse_res, coarse_res, seed=42
    )
    logger.info(
        f"  Velocity range: [{velocity.min():.3f}, {velocity.max():.3f}]"
    )
    logger.info(
        f"  Pressure range: [{pressure.min():.3f}, {pressure.max():.3f}]"
    )

    # -------------------------------------------------------------------------
    # 2. Configure and run super-resolution pipeline
    # -------------------------------------------------------------------------
    logger.info("\n[2/5] Running RANS-to-LES super-resolution...")
    config = SuperResolutionConfig(
        coarse_shape=(coarse_res, coarse_res, coarse_res),
        fine_shape=(fine_res, fine_res, fine_res),
        n_channels=4,
        spectral_weight=1.0,
        continuity_weight=10.0,
        seed=42,
    )
    mapper = RANStoLESMapper(config)
    result = mapper.reconstruct(velocity, pressure, seed=42)

    logger.info(f"  HR velocity shape: {result.hr_velocity.shape}")
    logger.info(f"  HR pressure shape: {result.hr_pressure.shape}")
    logger.info(f"  Spectral error:    {result.spectral_error:.4f}")
    logger.info(f"  Continuity resid:  {result.continuity_residual:.6f}")

    # -------------------------------------------------------------------------
    # 3. TBNN-conditioned diffusion
    # -------------------------------------------------------------------------
    logger.info("\n[3/5] Conditioning on TBNN anisotropy tensor...")
    anisotropy = generate_synthetic_anisotropy(
        coarse_res, coarse_res, coarse_res, seed=42
    )
    tbnn_cond = TBNNConditionedDiffusion(seed=42)
    conditioning = tbnn_cond.encode_anisotropy(anisotropy)
    aniso_intensity = tbnn_cond.compute_anisotropy_intensity(anisotropy)

    logger.info(f"  Conditioning shape: {conditioning.shape}")
    logger.info(
        f"  Anisotropy intensity: "
        f"mean={aniso_intensity.mean():.4f}, max={aniso_intensity.max():.4f}"
    )

    # Reconstruct with anisotropy conditioning
    result_cond = mapper.reconstruct(velocity, pressure,
                                     anisotropy=anisotropy, seed=42)
    logger.info(f"  Conditioned spectral error:   {result_cond.spectral_error:.4f}")
    logger.info(f"  Conditioned continuity resid: {result_cond.continuity_residual:.6f}")

    # -------------------------------------------------------------------------
    # 4. Discriminator evaluation
    # -------------------------------------------------------------------------
    logger.info("\n[4/5] Running discriminator evaluation...")
    disc = SuperResolutionDiscriminator(n_channels=4, seed=42)

    # Evaluate coarse (upsampled naively) vs super-resolved
    from scipy.ndimage import zoom
    coarse_naive = np.zeros((4, fine_res, fine_res, fine_res))
    for c in range(3):
        coarse_naive[c] = zoom(velocity[c], upscale, order=1)
    coarse_naive[3] = zoom(pressure, upscale, order=1)

    fine_field = np.zeros((4,) + config.fine_shape)
    fine_field[:3] = result.hr_velocity
    fine_field[3] = result.hr_pressure

    score_naive = disc.discriminate(coarse_naive)
    score_sr = disc.discriminate(fine_field)
    logger.info(f"  Naive upscale score:   {score_naive:.4f} (0=fake, 1=real)")
    logger.info(f"  Super-resolved score:  {score_sr:.4f} (0=fake, 1=real)")

    # -------------------------------------------------------------------------
    # 5. Multi-scale comparison
    # -------------------------------------------------------------------------
    logger.info("\n[5/5] Multi-scale upscale comparison...")
    scales = []
    for uf in [2, 4]:
        fr = coarse_res * uf
        cfg = SuperResolutionConfig(
            coarse_shape=(coarse_res, coarse_res, coarse_res),
            fine_shape=(fr, fr, fr),
            seed=42,
        )
        m = RANStoLESMapper(cfg)
        r = m.reconstruct(velocity, pressure, seed=42)
        scales.append({
            "upscale": uf,
            "fine_res": fr,
            "spectral_error": round(float(r.spectral_error), 4),
            "continuity_residual": round(float(r.continuity_residual), 6),
        })
        logger.info(
            f"  {uf}x ({coarse_res}->{fr}): "
            f"spectral={r.spectral_error:.4f}, "
            f"continuity={r.continuity_residual:.6f}"
        )

    # -------------------------------------------------------------------------
    # Summary table
    # -------------------------------------------------------------------------
    from tabulate import tabulate

    table_data = [
        [
            f"{s['upscale']}x",
            f"{coarse_res} -> {s['fine_res']}",
            f"{s['spectral_error']:.4f}",
            f"{s['continuity_residual']:.6f}",
        ]
        for s in scales
    ]
    print("\n" + "=" * 65)
    print("  Generative Super-Resolution Results")
    print("=" * 65)
    print(tabulate(
        table_data,
        headers=["Upscale", "Resolution", "Spectral Err", "Continuity Res"]
    ))
    print(f"\n  Discriminator: naive={score_naive:.4f}, SR={score_sr:.4f}")
    print(f"  TBNN anisotropy intensity: {aniso_intensity.mean():.4f} (mean)")
    print("=" * 65)

    # -------------------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------------------
    out_dir = PROJECT_ROOT / "results" / "generative_sr"
    out_dir.mkdir(parents=True, exist_ok=True)

    results_json = {
        "config": {
            "coarse_res": coarse_res,
            "fine_res": fine_res,
            "upscale": upscale,
        },
        "unconditioned": {
            "spectral_error": round(float(result.spectral_error), 4),
            "continuity_residual": round(float(result.continuity_residual), 6),
        },
        "tbnn_conditioned": {
            "spectral_error": round(float(result_cond.spectral_error), 4),
            "continuity_residual": round(float(result_cond.continuity_residual), 6),
            "mean_anisotropy_intensity": round(float(aniso_intensity.mean()), 4),
        },
        "discriminator": {
            "naive_score": round(float(score_naive), 4),
            "sr_score": round(float(score_sr), 4),
        },
        "multiscale": scales,
    }

    with open(out_dir / "superres_results.json", "w") as f:
        json.dump(results_json, f, indent=2)
    logger.info(f"Results saved to {out_dir / 'superres_results.json'}")


# =============================================================================
# CLI
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generative Super-Resolution for Flow Fields"
    )
    parser.add_argument("--coarse-res", type=int, default=16,
                        help="Coarse grid resolution (default: 16)")
    parser.add_argument("--fine-res", type=int, default=64,
                        help="Fine grid resolution (default: 64)")
    args = parser.parse_args()

    run_benchmark(coarse_res=args.coarse_res, fine_res=args.fine_res)
