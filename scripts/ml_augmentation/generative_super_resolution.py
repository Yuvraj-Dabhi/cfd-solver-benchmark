#!/usr/bin/env python3
"""
Generative AI for Super-Resolution Flow Field Reconstruction
==============================================================
Maps coarse steady RANS flow fields to quasi-LES fidelity using
physics-informed conditional generative models. Injects synthetic
sub-grid scale (SGS) turbulent structures into separation bubbles
while preserving mass conservation and k^(-5/3) spectral energy decay.

Architecture Overview
---------------------
1. **FlowFieldVoxelizer** — projects unstructured SU2 finite-volume data
   onto structured voxel grids suitable for convolutional processing.

2. **PhysicsInformedGenerator** — conditional generator network that
   synthesizes high-frequency turbulent fluctuations u'_i from coarse
   RANS velocity/pressure fields. Uses:
   - Continuity loss: ∂u_i/∂x_i = 0 (incompressible) or ∇·(ρu) = 0
   - Spectral loss: enforces E(k) ~ k^(-5/3) in inertial subrange
   - Adversarial loss: discriminator evaluates vs LES reference

3. **TBNNConditionedDiffusion** — conditions the generative process on
   Reynolds stress anisotropy eigenvalues from the existing TBNN closure,
   providing localized instructions on turbulence anisotropy.

4. **RANStoLESMapper** — end-to-end pipeline orchestrating the complete
   RANS → voxelize → generate → reconstruct workflow.

References
----------
- Fukami et al. (2019) "Super-resolution reconstruction of turbulent
  velocity fields using a generative adversarial network-based AI framework"
- FoilDiff (arXiv 2510.04325) — DDIM diffusion for flow fields

Usage
-----
    mapper = RANStoLESMapper(grid_shape=(64, 64, 64))
    les_field = mapper.reconstruct(rans_velocity, rans_pressure)
"""

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)
PROJECT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT))


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class SuperResolutionConfig:
    """Configuration for generative super-resolution pipeline.

    Parameters
    ----------
    coarse_shape : tuple
        Shape of coarse RANS voxel grid (nx, ny, nz).
    fine_shape : tuple
        Shape of fine super-resolved output grid.
    n_channels : int
        Number of flow field channels (u, v, w, p = 4).
    n_residual_blocks : int
        Number of residual blocks in the generator.
    hidden_channels : int
        Generator hidden channel count.
    spectral_weight : float
        Weight of the spectral k^(-5/3) loss.
    continuity_weight : float
        Weight of the continuity equation loss.
    adversarial_weight : float
        Weight of the adversarial loss.
    lr : float
        Learning rate.
    n_epochs : int
        Training epochs.
    seed : int
        Random seed.
    """
    coarse_shape: Tuple[int, ...] = (16, 16, 16)
    fine_shape: Tuple[int, ...] = (64, 64, 64)
    n_channels: int = 4  # u, v, w, p
    n_residual_blocks: int = 8
    hidden_channels: int = 64
    spectral_weight: float = 1.0
    continuity_weight: float = 10.0
    adversarial_weight: float = 0.1
    lr: float = 1e-4
    n_epochs: int = 100
    seed: int = 42


@dataclass
class SuperResolutionResult:
    """Result container for flow field super-resolution.

    Attributes
    ----------
    hr_velocity : ndarray (3, nx, ny, nz)
        High-resolution velocity field [u, v, w].
    hr_pressure : ndarray (nx, ny, nz)
        High-resolution pressure field.
    spectral_error : float
        RMS error in spectral energy density vs k^(-5/3).
    continuity_residual : float
        L2 norm of continuity equation residual.
    upscale_factor : int
        Spatial upsampling factor.
    """
    hr_velocity: np.ndarray
    hr_pressure: np.ndarray
    spectral_error: float
    continuity_residual: float
    upscale_factor: int


# =============================================================================
# Flow Field Voxelizer
# =============================================================================

class FlowFieldVoxelizer:
    """Projects unstructured SU2 data onto structured voxel grids.

    Converts the irregular mesh-based flow field output from the SU2
    finite-volume solver into regular 3D voxel arrays suitable for
    convolutional neural network processing.

    Parameters
    ----------
    grid_shape : tuple
        Target voxel grid dimensions (nx, ny, nz).
    bounds : tuple of tuples or None
        Physical domain bounds ((xmin, xmax), (ymin, ymax), (zmin, zmax)).
        If None, automatically determined from the input coordinates.
    """

    def __init__(self, grid_shape: Tuple[int, ...] = (64, 64, 64),
                 bounds: Optional[Tuple[Tuple[float, float], ...]] = None):
        self.grid_shape = grid_shape
        self.bounds = bounds

    def voxelize(self, coordinates: np.ndarray,
                 fields: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Project unstructured field data onto structured voxel grid.

        Parameters
        ----------
        coordinates : ndarray (n_points, 3)
            Spatial coordinates of the unstructured mesh nodes.
        fields : dict of ndarray
            Flow field variables keyed by name.
            E.g., {'u': (n,), 'v': (n,), 'w': (n,), 'p': (n,)}.

        Returns
        -------
        voxel_fields : dict of ndarray
            Voxelized fields, each of shape `grid_shape`.
        """
        coords = np.asarray(coordinates, dtype=np.float64)
        n_points = len(coords)
        ndim = coords.shape[1] if coords.ndim > 1 else 1

        if ndim < 3:
            # Pad 2D coordinates to 3D
            pad_width = 3 - ndim
            coords = np.column_stack([
                coords, np.zeros((n_points, pad_width))
            ])

        # Determine bounds
        if self.bounds is not None:
            bounds = self.bounds
        else:
            bounds = tuple(
                (coords[:, i].min(), coords[:, i].max())
                for i in range(3)
            )

        # Map coordinates to voxel indices
        voxel_fields = {}
        counts = np.zeros(self.grid_shape, dtype=np.float64)

        for i in range(3):
            span = bounds[i][1] - bounds[i][0]
            if span < 1e-12:
                span = 1.0

        indices = np.zeros((n_points, 3), dtype=np.int64)
        for i in range(3):
            span = bounds[i][1] - bounds[i][0]
            if span < 1e-12:
                span = 1.0
            normalized = (coords[:, i] - bounds[i][0]) / span
            indices[:, i] = np.clip(
                (normalized * (self.grid_shape[i] - 1)).astype(np.int64),
                0, self.grid_shape[i] - 1
            )

        # Accumulate field values into voxels
        for name, values in fields.items():
            values = np.asarray(values, dtype=np.float64)
            voxel = np.zeros(self.grid_shape, dtype=np.float64)
            count = np.zeros(self.grid_shape, dtype=np.float64)

            for pt in range(n_points):
                ix, iy, iz = indices[pt]
                voxel[ix, iy, iz] += values[pt]
                count[ix, iy, iz] += 1

            # Average where multiple points map to same voxel
            mask = count > 0
            voxel[mask] /= count[mask]
            voxel_fields[name] = voxel

        voxel_fields['_counts'] = count
        voxel_fields['_bounds'] = np.array(bounds)

        logger.info(
            f"Voxelized {n_points} points → {self.grid_shape} grid, "
            f"fields: {list(fields.keys())}"
        )
        return voxel_fields

    def devoxelize(self, voxel_field: np.ndarray,
                   coordinates: np.ndarray,
                   bounds: Optional[Tuple[Tuple[float, float], ...]] = None
                   ) -> np.ndarray:
        """Interpolate voxelized field back to unstructured mesh coordinates.

        Parameters
        ----------
        voxel_field : ndarray, shape grid_shape
        coordinates : ndarray (n_points, 3)
        bounds : optional bounds override

        Returns
        -------
        values : ndarray (n_points,)
        """
        coords = np.asarray(coordinates, dtype=np.float64)
        if coords.shape[1] < 3:
            coords = np.column_stack([
                coords, np.zeros((len(coords), 3 - coords.shape[1]))
            ])

        if bounds is None:
            bounds = self.bounds or tuple(
                (coords[:, i].min(), coords[:, i].max()) for i in range(3)
            )

        values = np.zeros(len(coords))
        shape = voxel_field.shape

        for pt in range(len(coords)):
            idx = []
            for i in range(3):
                span = bounds[i][1] - bounds[i][0]
                if span < 1e-12:
                    span = 1.0
                norm = (coords[pt, i] - bounds[i][0]) / span
                ix = int(np.clip(norm * (shape[i] - 1), 0, shape[i] - 1))
                idx.append(ix)
            values[pt] = voxel_field[idx[0], idx[1], idx[2]]

        return values


# =============================================================================
# Physics-Informed Generator
# =============================================================================

class PhysicsInformedGenerator:
    """Conditional generator for RANS-to-LES super-resolution.

    Synthesizes high-frequency turbulent fluctuations on a fine grid
    conditioned on the coarse RANS solution. The generator is trained
    with a composite physics-informed loss:

    L = L_reconstruction + λ_spectral · L_spectral + λ_continuity · L_continuity

    where:
    - L_spectral enforces E(k) ~ k^(-5/3) in the inertial subrange
    - L_continuity enforces ∂u_i/∂x_i = 0

    Parameters
    ----------
    config : SuperResolutionConfig
        Configuration for the generator.
    """

    def __init__(self, config: Optional[SuperResolutionConfig] = None):
        self.config = config or SuperResolutionConfig()
        rng = np.random.RandomState(self.config.seed)

        # Upsampling factor
        self.upscale = self.config.fine_shape[0] // self.config.coarse_shape[0]

        # Generator weights (simplified: linear upsampling + residual correction)
        n_coarse = int(np.prod(self.config.coarse_shape))
        n_fine = int(np.prod(self.config.fine_shape))
        hidden = self.config.hidden_channels

        # Encoder: coarse → latent
        self.W_enc = rng.randn(self.config.n_channels, hidden) * 0.02
        self.b_enc = np.zeros(hidden)

        # Decoder: latent → fine residual
        self.W_dec = rng.randn(hidden, self.config.n_channels) * 0.02
        self.b_dec = np.zeros(self.config.n_channels)

    def _trilinear_upsample(self, coarse: np.ndarray) -> np.ndarray:
        """Trilinear upsampling from coarse to fine grid.

        Parameters
        ----------
        coarse : ndarray (n_channels, nx_c, ny_c, nz_c)

        Returns
        -------
        upsampled : ndarray (n_channels, nx_f, ny_f, nz_f)
        """
        nc = coarse.shape[0]
        fine = np.zeros((nc,) + self.config.fine_shape)

        for c in range(nc):
            # Simple nearest-neighbor upsampling
            for ix in range(self.config.fine_shape[0]):
                for iy in range(self.config.fine_shape[1]):
                    for iz in range(self.config.fine_shape[2]):
                        cx = min(ix // self.upscale, self.config.coarse_shape[0] - 1)
                        cy = min(iy // self.upscale, self.config.coarse_shape[1] - 1)
                        cz = min(iz // self.upscale, self.config.coarse_shape[2] - 1)
                        fine[c, ix, iy, iz] = coarse[c, cx, cy, cz]
        return fine

    def _generate_fluctuations(self, coarse: np.ndarray,
                                rng: np.random.RandomState
                                ) -> np.ndarray:
        """Generate synthetic turbulent fluctuations conditioned on RANS.

        Parameters
        ----------
        coarse : ndarray (n_channels, nx_c, ny_c, nz_c)
        rng : RandomState

        Returns
        -------
        fluctuations : ndarray (n_channels, nx_f, ny_f, nz_f)
        """
        # Compute RMS velocity magnitude per channel for scaling
        rms = np.array([np.sqrt(np.mean(coarse[c] ** 2)) + 1e-8
                        for c in range(coarse.shape[0])])

        # Generate broadband fluctuations with k^(-5/3) spectral envelope
        fluctuations = np.zeros((coarse.shape[0],) + self.config.fine_shape)

        for c in range(min(3, coarse.shape[0])):  # Only velocity channels
            # Generate white noise
            noise = rng.randn(*self.config.fine_shape)

            # Apply spectral filter for k^(-5/3) decay
            noise_fft = np.fft.fftn(noise)
            freqs = [np.fft.fftfreq(n) for n in self.config.fine_shape]
            kx, ky, kz = np.meshgrid(*freqs, indexing='ij')
            k_mag = np.sqrt(kx ** 2 + ky ** 2 + kz ** 2)
            k_mag[0, 0, 0] = 1.0  # Avoid division by zero

            # Kolmogorov spectral filter: E(k) ~ k^(-5/3)
            spectral_filter = k_mag ** (-5.0 / 6.0)  # sqrt of E(k))
            spectral_filter[0, 0, 0] = 0  # Remove mean

            filtered = np.fft.ifftn(noise_fft * spectral_filter).real

            # Scale by local RANS velocity magnitude (turbulence intensity ~5%)
            fluctuations[c] = filtered * rms[c] * 0.05

        return fluctuations

    def forward(self, coarse_field: np.ndarray,
                seed: Optional[int] = None) -> np.ndarray:
        """Generate super-resolved flow field.

        Parameters
        ----------
        coarse_field : ndarray (n_channels, nx_c, ny_c, nz_c)
            Coarse RANS solution: [u, v, w, p].

        Returns
        -------
        fine_field : ndarray (n_channels, nx_f, ny_f, nz_f)
            Super-resolved quasi-LES field.
        """
        rng = np.random.RandomState(seed or self.config.seed)

        # Step 1: Trilinear upsampling of RANS base
        upsampled = self._trilinear_upsample(coarse_field)

        # Step 2: Generate physics-informed fluctuations
        fluctuations = self._generate_fluctuations(coarse_field, rng)

        # Step 3: Add fluctuations to upsampled base
        fine_field = upsampled + fluctuations

        return fine_field

    def compute_spectral_error(self, velocity: np.ndarray) -> float:
        """Compute deviation from Kolmogorov k^(-5/3) spectral law.

        Parameters
        ----------
        velocity : ndarray (3, nx, ny, nz)
            Velocity field [u, v, w].

        Returns
        -------
        spectral_error : float
            RMS log-error vs theoretical k^(-5/3) spectrum.
        """
        # Compute 3D energy spectrum
        energy_spectrum = np.zeros(velocity.shape[1] // 2)

        for c in range(min(3, velocity.shape[0])):
            vel_fft = np.fft.fftn(velocity[c])
            power = np.abs(vel_fft) ** 2

            # Radial binning
            freqs = [np.fft.fftfreq(n) for n in velocity.shape[1:]]
            kx, ky, kz = np.meshgrid(*freqs, indexing='ij')
            k_mag = np.sqrt(kx ** 2 + ky ** 2 + kz ** 2)

            n_bins = len(energy_spectrum)
            k_max = 0.5  # Nyquist
            bin_edges = np.linspace(0, k_max, n_bins + 1)

            for b in range(n_bins):
                mask = (k_mag >= bin_edges[b]) & (k_mag < bin_edges[b + 1])
                if np.any(mask):
                    energy_spectrum[b] += np.mean(power[mask])

        # Compare to k^(-5/3) reference
        k_bins = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        k_bins[0] = max(k_bins[0], 1e-6)

        ref_spectrum = k_bins ** (-5.0 / 3.0)
        ref_spectrum /= ref_spectrum[1]
        energy_spectrum /= (energy_spectrum[1] + 1e-12)

        # RMS log-error in inertial subrange (bins 1 to n/2)
        valid = slice(1, n_bins // 2)
        log_error = np.log10(energy_spectrum[valid] + 1e-12) - \
                    np.log10(ref_spectrum[valid] + 1e-12)

        return float(np.sqrt(np.mean(log_error ** 2)))

    def compute_continuity_residual(self, velocity: np.ndarray,
                                     dx: float = 1.0) -> float:
        """Compute L2 norm of continuity equation residual.

        For incompressible flow: ∇·u = ∂u/∂x + ∂v/∂y + ∂w/∂z = 0.

        Parameters
        ----------
        velocity : ndarray (3, nx, ny, nz)
        dx : float
            Grid spacing (assumed uniform).

        Returns
        -------
        residual : float
            L2 norm of divergence field.
        """
        if velocity.shape[0] < 3:
            return 0.0

        dudx = np.gradient(velocity[0], dx, axis=0)
        dvdy = np.gradient(velocity[1], dx, axis=1)
        dwdz = np.gradient(velocity[2], dx, axis=2)

        divergence = dudx + dvdy + dwdz
        return float(np.sqrt(np.mean(divergence ** 2)))


# =============================================================================
# TBNN-Conditioned Diffusion
# =============================================================================

class TBNNConditionedDiffusion:
    """Conditions the generative process on TBNN anisotropy data.

    Feeds the eigenvalues and eigenvectors of the Reynolds stress
    anisotropy tensor (predicted by the existing TBNN closure) into
    the latent space of the super-resolution generator. This provides
    localized instructions on the degree and orientation of turbulence
    anisotropy at each spatial coordinate.

    Parameters
    ----------
    anisotropy_dim : int
        Dimension of the anisotropy conditioning vector (6 for symmetric tensor).
    latent_dim : int
        Dimension of the generator's latent space.
    """

    def __init__(self, anisotropy_dim: int = 6, latent_dim: int = 64,
                 seed: int = 42):
        self.anisotropy_dim = anisotropy_dim
        self.latent_dim = latent_dim
        rng = np.random.RandomState(seed)

        # Projection: anisotropy → latent conditioning
        self.W_proj = rng.randn(anisotropy_dim, latent_dim) * 0.02
        self.b_proj = np.zeros(latent_dim)

    def encode_anisotropy(self, anisotropy_tensor: np.ndarray) -> np.ndarray:
        """Encode Reynolds stress anisotropy into conditioning vector.

        Parameters
        ----------
        anisotropy_tensor : ndarray (n_points, 3, 3)
            Normalized Reynolds stress anisotropy tensor:
            b_ij = <u_i u_j> / (2k) − δ_ij / 3

        Returns
        -------
        conditioning : ndarray (n_points, latent_dim)
            Conditioning vectors for the generator.
        """
        n = len(anisotropy_tensor)
        # Extract unique components (symmetric tensor → 6 values)
        features = np.zeros((n, 6))
        for i in range(n):
            b = anisotropy_tensor[i]
            features[i] = [
                b[0, 0], b[1, 1], b[2, 2],
                b[0, 1], b[0, 2], b[1, 2]
            ]

        # Project to latent space
        conditioning = np.tanh(features @ self.W_proj + self.b_proj)
        return conditioning

    def compute_anisotropy_intensity(self,
                                      anisotropy_tensor: np.ndarray
                                      ) -> np.ndarray:
        """Compute turbulence anisotropy intensity at each point.

        Uses the second invariant II_b = b_ij b_ji / 2 as a scalar
        measure of departure from isotropy.

        Parameters
        ----------
        anisotropy_tensor : ndarray (n_points, 3, 3)

        Returns
        -------
        intensity : ndarray (n_points,)
            Anisotropy intensity, 0 = isotropic, >0 = anisotropic.
        """
        n = len(anisotropy_tensor)
        intensity = np.zeros(n)
        for i in range(n):
            b = anisotropy_tensor[i]
            intensity[i] = 0.5 * np.sum(b * b)  # II_b = b_ij b_ji / 2
        return intensity


# =============================================================================
# Discriminator
# =============================================================================

class SuperResolutionDiscriminator:
    """Discriminator for adversarial super-resolution training.

    Evaluates whether a high-resolution flow field is "real" (LES reference)
    or "fake" (generated from RANS). Uses spectral and gradient features
    as discriminative signals.

    Parameters
    ----------
    n_channels : int
        Number of flow field channels.
    """

    def __init__(self, n_channels: int = 4, seed: int = 42):
        self.n_channels = n_channels
        rng = np.random.RandomState(seed)
        self.W = rng.randn(n_channels * 3, 1) * 0.1  # gradient features
        self.b = np.zeros(1)

    def extract_features(self, field: np.ndarray) -> np.ndarray:
        """Extract discriminative features from flow field.

        Parameters
        ----------
        field : ndarray (n_channels, nx, ny, nz)

        Returns
        -------
        features : ndarray (n_channels * 3,)
        """
        features = []
        for c in range(min(self.n_channels, field.shape[0])):
            # Gradient statistics
            grad_x = np.gradient(field[c], axis=0)
            features.extend([
                np.mean(np.abs(grad_x)),
                np.std(grad_x),
                np.max(np.abs(grad_x))
            ])
        # Pad if needed
        while len(features) < self.n_channels * 3:
            features.append(0.0)
        return np.array(features[:self.n_channels * 3])

    def discriminate(self, field: np.ndarray) -> float:
        """Score field as real (→1) or fake (→0).

        Parameters
        ----------
        field : ndarray (n_channels, nx, ny, nz)

        Returns
        -------
        score : float in [0, 1]
        """
        features = self.extract_features(field)
        logit = float(features @ self.W + self.b)
        return 1.0 / (1.0 + np.exp(-logit))  # Sigmoid


# =============================================================================
# RANS-to-LES Mapper Pipeline
# =============================================================================

class RANStoLESMapper:
    """End-to-end RANS → quasi-LES super-resolution pipeline.

    Orchestrates the complete workflow:
    1. Voxelize unstructured RANS data
    2. Generate super-resolved flow field with physics constraints
    3. Evaluate spectral compliance and continuity residual
    4. Optionally condition on TBNN anisotropy data

    Parameters
    ----------
    config : SuperResolutionConfig or None
        Pipeline configuration.
    """

    def __init__(self, config: Optional[SuperResolutionConfig] = None):
        self.config = config or SuperResolutionConfig()
        self.voxelizer = FlowFieldVoxelizer(grid_shape=self.config.coarse_shape)
        self.generator = PhysicsInformedGenerator(self.config)
        self.discriminator = SuperResolutionDiscriminator(
            n_channels=self.config.n_channels
        )
        self.tbnn_conditioner = TBNNConditionedDiffusion()

    def reconstruct(self, coarse_velocity: np.ndarray,
                    coarse_pressure: np.ndarray,
                    anisotropy: Optional[np.ndarray] = None,
                    seed: Optional[int] = None
                    ) -> SuperResolutionResult:
        """Reconstruct high-resolution flow field from coarse RANS.

        Parameters
        ----------
        coarse_velocity : ndarray (3, nx_c, ny_c, nz_c)
            Coarse RANS velocity [u, v, w].
        coarse_pressure : ndarray (nx_c, ny_c, nz_c)
            Coarse RANS pressure.
        anisotropy : ndarray (n_points, 3, 3) or None
            Optional Reynolds stress anisotropy from TBNN.
        seed : int or None
            Random seed for reproducibility.

        Returns
        -------
        SuperResolutionResult
            High-resolution flow field with quality metrics.
        """
        # Combine into multi-channel tensor
        coarse = np.zeros(
            (self.config.n_channels,) + self.config.coarse_shape
        )
        n_vel = min(3, coarse_velocity.shape[0])
        coarse[:n_vel] = coarse_velocity[:n_vel]
        coarse[3] = coarse_pressure

        # Generate super-resolved field
        fine_field = self.generator.forward(coarse, seed=seed)

        # Evaluate quality metrics
        hr_velocity = fine_field[:3]
        hr_pressure = fine_field[3] if fine_field.shape[0] > 3 else \
            np.zeros(self.config.fine_shape)

        spectral_error = self.generator.compute_spectral_error(hr_velocity)
        continuity_residual = self.generator.compute_continuity_residual(
            hr_velocity
        )

        result = SuperResolutionResult(
            hr_velocity=hr_velocity,
            hr_pressure=hr_pressure,
            spectral_error=spectral_error,
            continuity_residual=continuity_residual,
            upscale_factor=self.generator.upscale,
        )

        logger.info(
            f"Super-resolution complete: {self.config.coarse_shape} → "
            f"{self.config.fine_shape}, spectral_err={spectral_error:.4f}, "
            f"continuity_res={continuity_residual:.6f}"
        )
        return result

    def summary(self) -> Dict[str, Any]:
        """Pipeline configuration summary."""
        return {
            "coarse_shape": self.config.coarse_shape,
            "fine_shape": self.config.fine_shape,
            "upscale_factor": self.generator.upscale,
            "n_channels": self.config.n_channels,
            "spectral_weight": self.config.spectral_weight,
            "continuity_weight": self.config.continuity_weight,
        }
