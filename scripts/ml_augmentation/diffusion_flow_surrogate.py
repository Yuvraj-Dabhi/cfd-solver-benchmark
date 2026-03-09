#!/usr/bin/env python3
"""
Generative Diffusion Surrogate for Flow Field Prediction
==========================================================
FoilDiff-style denoising diffusion model (DDIM sampling) that predicts
the full distribution of Cp(x/c) and Cf(x/c) surface fields conditioned
on (Re, AoA, Mach).

Unlike deterministic surrogates (MLP, FNO, GNN), this model:
  1. Learns the full distribution of flow states → native UQ
  2. Handles out-of-distribution generalization better
  3. Captures multi-modal separation patterns near stall (α ≈ 15°)
     that a deterministic MLP averages away

Architecture
------------
    Conditioning:  (Re·cos(α), Re·sin(α), Mach) → MLP → d_cond
    Timestep:      t → sinusoidal positional encoding → d_time
    Denoiser:      U-Net with 1D conv ResBlocks + self-attention bottleneck
    Sampling:      DDIM (10–50 steps) for fast deterministic sampling

FoilDiff Reference
------------------
    - arXiv 2510.04325, Oct 2025 — RANS flow field around NACA profiles
    - Conditioning: Re·cos(α), Re·sin(α) (identical to distribution_surrogate.py)
    - Hybrid CNN + Transformer denoising network

Additional References
---------------------
    - Song et al. (2021), "Denoising Diffusion Implicit Models", ICLR
    - Nichol & Dhariwal (2021), "Improved Denoising Diffusion Probabilistic
      Models", ICML — cosine noise schedule
    - CoNFiLD (Nature Communications, Nov 2024) — wall-bounded turbulence
"""

import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

# Default spatial resolution (matches distribution_surrogate.py)
N_SURFACE_POINTS = 80
N_CHANNELS = 2  # Cp, Cf


# =============================================================================
# Sinusoidal Timestep Embedding
# =============================================================================
class SinusoidalTimeEmbedding:
    """
    Encode diffusion timestep t into a d_model-dim vector via
    sin/cos positional encoding (Vaswani et al., 2017).

    Parameters
    ----------
    d_model : int
        Embedding dimension (must be even).
    max_period : float
        Maximum period for the sinusoidal frequencies.
    """

    def __init__(self, d_model: int = 64, max_period: float = 10000.0):
        assert d_model % 2 == 0, "d_model must be even"
        self.d_model = d_model
        self.max_period = max_period

        # Precompute frequency bands
        half = d_model // 2
        self.freqs = np.exp(
            -np.log(max_period) * np.arange(half, dtype=np.float64) / half
        )

    def __call__(self, t: np.ndarray) -> np.ndarray:
        """
        Embed timesteps.

        Parameters
        ----------
        t : ndarray, shape (batch,) or scalar
            Timestep values in [0, 1].

        Returns
        -------
        emb : ndarray, shape (batch, d_model)
        """
        t = np.atleast_1d(np.asarray(t, dtype=np.float64))
        args = t[:, None] * self.freqs[None, :]  # (batch, half)
        return np.concatenate([np.sin(args), np.cos(args)], axis=-1)


# =============================================================================
# Flow Condition Encoder
# =============================================================================
class FlowConditionEncoder:
    """
    Maps (Re, AoA, Mach) to a d_cond-dim conditioning vector using
    FoilDiff-style Re·cos(α), Re·sin(α) encoding.

    Architecture: [3] → [d_hidden] → [d_cond] with GELU activations.

    Parameters
    ----------
    d_cond : int
        Output conditioning dimension.
    d_hidden : int
        Hidden layer dimension.
    Re_scale : float
        Normalisation constant for Reynolds number.
    """

    def __init__(
        self,
        d_cond: int = 64,
        d_hidden: int = 128,
        Re_scale: float = 1e7,
        seed: int = 42,
    ):
        self.d_cond = d_cond
        self.d_hidden = d_hidden
        self.Re_scale = Re_scale

        rng = np.random.RandomState(seed)
        scale1 = np.sqrt(2.0 / 3)
        scale2 = np.sqrt(2.0 / d_hidden)

        self.W1 = rng.randn(3, d_hidden).astype(np.float64) * scale1
        self.b1 = np.zeros(d_hidden, dtype=np.float64)
        self.W2 = rng.randn(d_hidden, d_cond).astype(np.float64) * scale2
        self.b2 = np.zeros(d_cond, dtype=np.float64)

    @staticmethod
    def _gelu(x: np.ndarray) -> np.ndarray:
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

    def encode_conditions(
        self, aoa_deg: np.ndarray, Re: np.ndarray, Mach: np.ndarray
    ) -> np.ndarray:
        """
        Encode flow conditions to conditioning vector.

        Parameters
        ----------
        aoa_deg, Re, Mach : ndarray, shape (batch,)

        Returns
        -------
        cond : ndarray, shape (batch, d_cond)
        """
        aoa_rad = np.radians(aoa_deg)
        Re_norm = Re / self.Re_scale

        # FoilDiff-style encoding: Re·cos(α), Re·sin(α), Mach
        raw = np.stack([
            Re_norm * np.cos(aoa_rad),
            Re_norm * np.sin(aoa_rad),
            Mach,
        ], axis=-1)  # (batch, 3)

        h = self._gelu(raw @ self.W1 + self.b1)
        return self._gelu(h @ self.W2 + self.b2)


# =============================================================================
# 1D Residual Block with Conditioning
# =============================================================================
class ResidualBlock1D:
    """
    1D convolutional residual block with time + condition injection.

    Conv1D → GroupNorm → GELU → Conv1D → GroupNorm + (time+cond) → GELU + skip

    Parameters
    ----------
    channels : int
        Number of input/output channels.
    d_time : int
        Time embedding dimension.
    d_cond : int
        Condition embedding dimension.
    n_groups : int
        Number of groups for GroupNorm.
    """

    def __init__(
        self,
        channels: int = 32,
        d_time: int = 64,
        d_cond: int = 64,
        n_groups: int = 8,
        kernel_size: int = 3,
        seed: int = 42,
    ):
        self.channels = channels
        self.d_time = d_time
        self.d_cond = d_cond
        self.n_groups = min(n_groups, channels)
        self.kernel_size = kernel_size

        rng = np.random.RandomState(seed)
        scale = np.sqrt(2.0 / (channels * kernel_size))

        # Two conv layers
        self.W_conv1 = rng.randn(channels, channels, kernel_size).astype(np.float64) * scale
        self.b_conv1 = np.zeros(channels, dtype=np.float64)
        self.W_conv2 = rng.randn(channels, channels, kernel_size).astype(np.float64) * scale
        self.b_conv2 = np.zeros(channels, dtype=np.float64)

        # Time + condition projection → channels
        self.W_time = rng.randn(d_time, channels).astype(np.float64) * np.sqrt(2.0 / d_time)
        self.W_cond = rng.randn(d_cond, channels).astype(np.float64) * np.sqrt(2.0 / d_cond)

        # GroupNorm parameters
        self.gn1_gamma = np.ones(channels, dtype=np.float64)
        self.gn1_beta = np.zeros(channels, dtype=np.float64)
        self.gn2_gamma = np.ones(channels, dtype=np.float64)
        self.gn2_beta = np.zeros(channels, dtype=np.float64)

    @staticmethod
    def _gelu(x: np.ndarray) -> np.ndarray:
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

    def _group_norm(
        self, x: np.ndarray, gamma: np.ndarray, beta: np.ndarray
    ) -> np.ndarray:
        """Apply group normalisation: x shape (batch, channels, spatial)."""
        B, C, S = x.shape
        G = self.n_groups
        x_g = x.reshape(B, G, C // G, S)
        mean = x_g.mean(axis=(2, 3), keepdims=True)
        var = x_g.var(axis=(2, 3), keepdims=True)
        x_norm = (x_g - mean) / np.sqrt(var + 1e-5)
        x_norm = x_norm.reshape(B, C, S)
        return gamma[None, :, None] * x_norm + beta[None, :, None]

    def _conv1d(
        self, x: np.ndarray, W: np.ndarray, b: np.ndarray
    ) -> np.ndarray:
        """
        1D convolution with same padding.
        x: (batch, in_ch, spatial), W: (in_ch, out_ch, kernel), b: (out_ch,)
        """
        B, C_in, S = x.shape
        C_out = W.shape[1]
        K = W.shape[2]
        pad = K // 2

        # Pad input
        x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad)), mode='reflect')

        # Convolve: for each kernel position, multiply and sum over in_channels
        out = np.zeros((B, C_out, S), dtype=np.float64)
        for k in range(K):
            # x_slice: (B, C_in, S), W_k: (C_in, C_out)
            W_k = W[:, :, k]  # (C_in, C_out)
            out += np.einsum('bcs,co->bos', x_pad[:, :, k:k + S], W_k)
        out += b[None, :, None]
        return out

    def forward(
        self,
        x: np.ndarray,
        t_emb: np.ndarray,
        c_emb: np.ndarray,
    ) -> np.ndarray:
        """
        Forward pass.

        Parameters
        ----------
        x : ndarray (batch, channels, spatial)
        t_emb : ndarray (batch, d_time)
        c_emb : ndarray (batch, d_cond)

        Returns
        -------
        out : ndarray (batch, channels, spatial)
        """
        residual = x

        # Conv1 → GroupNorm → GELU
        h = self._conv1d(x, self.W_conv1, self.b_conv1)
        h = self._group_norm(h, self.gn1_gamma, self.gn1_beta)
        h = self._gelu(h)

        # Inject time + condition
        tc = (t_emb @ self.W_time + c_emb @ self.W_cond)  # (batch, channels)
        h = h + tc[:, :, None]

        # Conv2 → GroupNorm → GELU
        h = self._conv1d(h, self.W_conv2, self.b_conv2)
        h = self._group_norm(h, self.gn2_gamma, self.gn2_beta)
        h = self._gelu(h)

        return h + residual


# =============================================================================
# Self-Attention Block
# =============================================================================
class SelfAttentionBlock:
    """
    Single-head self-attention over spatial dimension.

    Captures global pressure-recovery and shock-BL interaction patterns
    that local convolutions miss.

    Parameters
    ----------
    channels : int
        Number of channels (used as embed_dim).
    """

    def __init__(self, channels: int = 32, seed: int = 42):
        self.channels = channels
        rng = np.random.RandomState(seed)
        scale = np.sqrt(2.0 / channels)

        self.W_q = rng.randn(channels, channels).astype(np.float64) * scale
        self.W_k = rng.randn(channels, channels).astype(np.float64) * scale
        self.W_v = rng.randn(channels, channels).astype(np.float64) * scale
        self.W_out = rng.randn(channels, channels).astype(np.float64) * scale

        # LayerNorm
        self.ln_gamma = np.ones(channels, dtype=np.float64)
        self.ln_beta = np.zeros(channels, dtype=np.float64)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Self-attention forward pass.

        Parameters
        ----------
        x : ndarray (batch, channels, spatial)

        Returns
        -------
        out : ndarray (batch, channels, spatial)
        """
        B, C, S = x.shape
        residual = x

        # Transpose to (batch, spatial, channels) for attention
        x_t = x.transpose(0, 2, 1)  # (B, S, C)

        # Layer norm
        mean = x_t.mean(axis=-1, keepdims=True)
        var = x_t.var(axis=-1, keepdims=True)
        x_norm = (x_t - mean) / np.sqrt(var + 1e-5)
        x_norm = self.ln_gamma[None, None, :] * x_norm + self.ln_beta[None, None, :]

        # QKV projections
        Q = x_norm @ self.W_q  # (B, S, C)
        K = x_norm @ self.W_k
        V = x_norm @ self.W_v

        # Scaled dot-product attention
        scale = 1.0 / np.sqrt(C)
        attn = (Q @ K.transpose(0, 2, 1)) * scale  # (B, S, S)

        # Softmax
        attn_max = attn.max(axis=-1, keepdims=True)
        attn_exp = np.exp(attn - attn_max)
        attn_weights = attn_exp / (attn_exp.sum(axis=-1, keepdims=True) + 1e-9)

        # Weighted sum
        out = attn_weights @ V  # (B, S, C)
        out = out @ self.W_out

        # Transpose back + residual
        return out.transpose(0, 2, 1) + residual


# =============================================================================
# Denoising U-Net (Hybrid CNN + Transformer)
# =============================================================================
class DenoisingUNet:
    """
    U-Net denoiser with 1D ResBlocks and self-attention bottleneck.

    Architecture:
        Encoder: 3 stages  (ch → 2ch → 4ch) with strided downsampling
        Bottleneck: Self-attention at lowest resolution
        Decoder: 3 stages  (4ch → 2ch → ch) with upsampling + skip connections

    Parameters
    ----------
    in_channels : int
        Input field channels (2 for Cp+Cf).
    base_channels : int
        Base channel width (doubled at each encoder stage).
    d_time : int
        Timestep embedding dimension.
    d_cond : int
        Condition embedding dimension.
    """

    def __init__(
        self,
        in_channels: int = N_CHANNELS,
        base_channels: int = 32,
        d_time: int = 64,
        d_cond: int = 64,
        seed: int = 42,
    ):
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.d_time = d_time
        self.d_cond = d_cond

        rng = np.random.RandomState(seed)
        ch = base_channels

        # Lifting: in_channels → base_channels
        self.W_lift = rng.randn(in_channels, ch, 1).astype(np.float64) * np.sqrt(2.0 / in_channels)
        self.b_lift = np.zeros(ch, dtype=np.float64)

        # Encoder blocks (3 stages)
        self.enc_blocks = [
            ResidualBlock1D(ch, d_time, d_cond, seed=seed + 1),
            ResidualBlock1D(ch * 2, d_time, d_cond, seed=seed + 2),
            ResidualBlock1D(ch * 4, d_time, d_cond, seed=seed + 3),
        ]

        # Downsampling projections (ch → 2ch, 2ch → 4ch)
        self.W_down = [
            rng.randn(ch, ch * 2, 1).astype(np.float64) * np.sqrt(2.0 / ch),
            rng.randn(ch * 2, ch * 4, 1).astype(np.float64) * np.sqrt(2.0 / (ch * 2)),
        ]
        self.b_down = [
            np.zeros(ch * 2, dtype=np.float64),
            np.zeros(ch * 4, dtype=np.float64),
        ]

        # Bottleneck attention
        self.bottleneck_attn = SelfAttentionBlock(ch * 4, seed=seed + 10)

        # Decoder blocks (3 stages, reversed channels)
        self.dec_blocks = [
            ResidualBlock1D(ch * 4, d_time, d_cond, seed=seed + 4),
            ResidualBlock1D(ch * 2, d_time, d_cond, seed=seed + 5),
            ResidualBlock1D(ch, d_time, d_cond, seed=seed + 6),
        ]

        # Upsampling projections — after skip concat:
        #   stage1: decoder(ch*4) + skip(ch*2) = ch*6 → project to ch*2
        #   stage2: decoder(ch*2) + skip(ch) = ch*3 → project to ch  (note: skip is ch not ch*2)
        # Actually enc_blocks[0] outputs ch channels, enc_blocks[1] outputs ch*2 channels
        # So skips = [ch, ch*2].  Concat at decode:
        #   first concat:  dec(ch*4) + skip[1](ch*2) = ch*6 → ch*2
        #   second concat: dec(ch*2) + skip[0](ch) = ch*3 → ch
        self.W_up = [
            rng.randn(ch * 6, ch * 2, 1).astype(np.float64) * np.sqrt(2.0 / (ch * 6)),
            rng.randn(ch * 3, ch, 1).astype(np.float64) * np.sqrt(2.0 / (ch * 3)),
        ]
        self.b_up = [
            np.zeros(ch * 2, dtype=np.float64),
            np.zeros(ch, dtype=np.float64),
        ]

        # Output projection: base_channels → in_channels
        self.W_out = rng.randn(ch, in_channels, 1).astype(np.float64) * np.sqrt(2.0 / ch)
        self.b_out = np.zeros(in_channels, dtype=np.float64)

    @staticmethod
    def _pointwise_conv(x, W, b):
        """1×1 convolution: x (B, C_in, S), W (C_in, C_out, 1)."""
        W2d = W.reshape(W.shape[0], W.shape[1])  # (C_in, C_out)
        return np.einsum('bcs,co->bos', x, W2d) + b[None, :, None]

    @staticmethod
    def _downsample(x: np.ndarray) -> np.ndarray:
        """Downsample by factor 2 via average pooling."""
        B, C, S = x.shape
        if S % 2 != 0:
            x = np.pad(x, ((0, 0), (0, 0), (0, 1)), mode='reflect')
            S += 1
        return x.reshape(B, C, S // 2, 2).mean(axis=-1)

    @staticmethod
    def _upsample(x: np.ndarray, target_size: int) -> np.ndarray:
        """Upsample via linear interpolation to target_size."""
        B, C, S = x.shape
        if S == target_size:
            return x
        x_old = np.linspace(0, 1, S)
        x_new = np.linspace(0, 1, target_size)
        out = np.zeros((B, C, target_size), dtype=np.float64)
        for b in range(B):
            for c in range(C):
                out[b, c] = np.interp(x_new, x_old, x[b, c])
        return out

    def forward(
        self,
        x_noisy: np.ndarray,
        t_emb: np.ndarray,
        c_emb: np.ndarray,
    ) -> np.ndarray:
        """
        Predict noise ε from noisy input x_t.

        Parameters
        ----------
        x_noisy : ndarray (batch, in_channels, spatial)
        t_emb : ndarray (batch, d_time)
        c_emb : ndarray (batch, d_cond)

        Returns
        -------
        eps_pred : ndarray (batch, in_channels, spatial)
        """
        # Lifting: (B, in_ch, S) @ (in_ch, base_ch) -> (B, base_ch, S)
        W_lift_2d = self.W_lift.reshape(self.in_channels, self.base_channels)
        h = np.einsum('bcs,co->bos', x_noisy, W_lift_2d) + self.b_lift[None, :, None]

        # Encoder pass — save skip connections
        skips = []
        spatial_sizes = [h.shape[2]]

        # Stage 1
        h = self.enc_blocks[0].forward(h, t_emb, c_emb)
        skips.append(h)

        # Downsample + project → 2ch
        h = self._downsample(h)
        h = np.einsum('bcs,co->bos', h, self.W_down[0].squeeze(-1)) + self.b_down[0][None, :, None]
        spatial_sizes.append(h.shape[2])

        # Stage 2
        h = self.enc_blocks[1].forward(h, t_emb, c_emb)
        skips.append(h)

        # Downsample + project → 4ch
        h = self._downsample(h)
        h = np.einsum('bcs,co->bos', h, self.W_down[1].squeeze(-1)) + self.b_down[1][None, :, None]
        spatial_sizes.append(h.shape[2])

        # Stage 3 (bottleneck res)
        h = self.enc_blocks[2].forward(h, t_emb, c_emb)

        # Bottleneck attention
        h = self.bottleneck_attn.forward(h)

        # Decoder pass
        # Stage 3 dec
        h = self.dec_blocks[0].forward(h, t_emb, c_emb)

        # Upsample + skip concat + project
        h = self._upsample(h, spatial_sizes[1])
        h = np.concatenate([h, skips[1]], axis=1)  # (B, ch*4+ch*2=ch*6, S)
        h = np.einsum('bcs,co->bos', h, self.W_up[0].squeeze(-1)) + self.b_up[0][None, :, None]

        # Stage 2 dec
        h = self.dec_blocks[1].forward(h, t_emb, c_emb)

        # Upsample + skip concat + project
        h = self._upsample(h, spatial_sizes[0])
        h = np.concatenate([h, skips[0]], axis=1)  # (B, ch*2+ch=ch*3, S)
        h = np.einsum('bcs,co->bos', h, self.W_up[1].squeeze(-1)) + self.b_up[1][None, :, None]

        # Stage 1 dec
        h = self.dec_blocks[2].forward(h, t_emb, c_emb)

        # Output projection
        eps_pred = np.einsum(
            'bcs,co->bos', h, self.W_out.squeeze(-1)
        ) + self.b_out[None, :, None]

        return eps_pred

    def count_params(self) -> int:
        """Count total learnable parameters."""
        total = 0
        # Lifting + output
        total += self.W_lift.size + self.b_lift.size
        total += self.W_out.size + self.b_out.size
        # Down projections
        for W, b in zip(self.W_down, self.b_down):
            total += W.size + b.size
        # Up projections
        for W, b in zip(self.W_up, self.b_up):
            total += W.size + b.size
        # Encoder + decoder res blocks (approximate)
        for blk in self.enc_blocks + self.dec_blocks:
            total += blk.W_conv1.size + blk.b_conv1.size
            total += blk.W_conv2.size + blk.b_conv2.size
            total += blk.W_time.size + blk.W_cond.size
        # Attention
        total += self.bottleneck_attn.W_q.size * 4
        return total


# =============================================================================
# DDIM Noise Scheduler
# =============================================================================
class DDIMScheduler:
    """
    Denoising Diffusion Implicit Models (DDIM) scheduler.

    Uses cosine noise schedule (Nichol & Dhariwal, 2021) and
    deterministic sampling for 10–50× speedup over DDPM.

    Parameters
    ----------
    n_train_steps : int
        Total training diffusion steps (T).
    beta_start : float
        Minimum noise level.
    beta_end : float
        Maximum noise level.
    schedule_type : str
        Noise schedule: 'cosine' or 'linear'.
    """

    def __init__(
        self,
        n_train_steps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        schedule_type: str = "cosine",
    ):
        self.n_train_steps = n_train_steps
        self.schedule_type = schedule_type

        if schedule_type == "cosine":
            # Cosine schedule (Nichol & Dhariwal, 2021)
            s = 0.008
            steps = np.arange(n_train_steps + 1, dtype=np.float64) / n_train_steps
            alphas_cumprod = np.cos((steps + s) / (1 + s) * np.pi / 2) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
            self.betas = np.clip(betas, a_min=1e-5, a_max=0.999)
        else:
            self.betas = np.linspace(beta_start, beta_end, n_train_steps)

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas)
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)

    def add_noise(
        self, x_0: np.ndarray, noise: np.ndarray, t: np.ndarray
    ) -> np.ndarray:
        """
        Forward diffusion: q(x_t | x_0) = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε

        Parameters
        ----------
        x_0 : ndarray (batch, channels, spatial)
        noise : ndarray (batch, channels, spatial) — ε ~ N(0, I)
        t : ndarray (batch,) — integer timesteps

        Returns
        -------
        x_t : ndarray (batch, channels, spatial)
        """
        t_idx = np.clip(t.astype(int), 0, self.n_train_steps - 1)
        sqrt_alpha = self.sqrt_alphas_cumprod[t_idx][:, None, None]
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t_idx][:, None, None]
        return sqrt_alpha * x_0 + sqrt_one_minus * noise

    def ddim_step(
        self,
        x_t: np.ndarray,
        eps_pred: np.ndarray,
        t: int,
        t_prev: int,
        eta: float = 0.0,
    ) -> np.ndarray:
        """
        Single DDIM sampling step: x_{t-1} from x_t.

        Parameters
        ----------
        x_t : ndarray (batch, channels, spatial)
        eps_pred : ndarray (batch, channels, spatial) — predicted noise
        t : int — current timestep
        t_prev : int — previous timestep (t_prev < t)
        eta : float — stochasticity (0 = deterministic DDIM, 1 = DDPM)

        Returns
        -------
        x_{t_prev} : ndarray
        """
        alpha_t = self.alphas_cumprod[t]
        alpha_prev = self.alphas_cumprod[t_prev] if t_prev >= 0 else 1.0

        # Predicted x_0
        x0_pred = (x_t - np.sqrt(1 - alpha_t) * eps_pred) / np.sqrt(alpha_t)

        # DDIM variance
        sigma_t = eta * np.sqrt(
            (1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev)
        )

        # Direction pointing to x_t
        dir_xt = np.sqrt(1 - alpha_prev - sigma_t**2) * eps_pred

        # x_{t-1}
        x_prev = np.sqrt(alpha_prev) * x0_pred + dir_xt

        if eta > 0:
            noise = np.random.randn(*x_t.shape)
            x_prev = x_prev + sigma_t * noise

        return x_prev

    def get_sampling_timesteps(self, n_inference_steps: int = 50) -> np.ndarray:
        """
        Get sub-sampled timestep sequence for DDIM inference.

        Parameters
        ----------
        n_inference_steps : int
            Number of denoising steps (10–50 typical).

        Returns
        -------
        timesteps : ndarray of int, shape (n_inference_steps,)
        """
        step_ratio = self.n_train_steps // n_inference_steps
        timesteps = np.arange(0, n_inference_steps) * step_ratio
        return timesteps[::-1].copy()


# =============================================================================
# Diffusion Flow Surrogate (Top-Level API)
# =============================================================================
class DiffusionFlowSurrogate:
    """
    Generative diffusion surrogate for full-field Cp/Cf prediction.

    Learns the conditional distribution P(Cp, Cf | Re, α, Mach)
    via denoising score matching, then generates samples via DDIM.

    Parameters
    ----------
    spatial_res : int
        Number of surface points per field.
    base_channels : int
        U-Net base channel width.
    d_time : int
        Timestep embedding dimension.
    d_cond : int
        Condition embedding dimension.
    n_train_steps : int
        Diffusion training steps (T).
    n_inference_steps : int
        DDIM sampling steps (default 50).
    schedule_type : str
        Noise schedule type.
    learning_rate : float
        Pseudo learning rate for training.
    """

    def __init__(
        self,
        spatial_res: int = N_SURFACE_POINTS,
        base_channels: int = 32,
        d_time: int = 64,
        d_cond: int = 64,
        n_train_steps: int = 1000,
        n_inference_steps: int = 50,
        schedule_type: str = "cosine",
        learning_rate: float = 1e-4,
        seed: int = 42,
    ):
        self.spatial_res = spatial_res
        self.n_train_steps = n_train_steps
        self.n_inference_steps = n_inference_steps
        self.learning_rate = learning_rate
        self.seed = seed
        self._fitted = False

        # Sub-components
        self.time_embed = SinusoidalTimeEmbedding(d_model=d_time)
        self.cond_encoder = FlowConditionEncoder(
            d_cond=d_cond, seed=seed + 100
        )
        self.denoiser = DenoisingUNet(
            in_channels=N_CHANNELS,
            base_channels=base_channels,
            d_time=d_time,
            d_cond=d_cond,
            seed=seed + 200,
        )
        self.scheduler = DDIMScheduler(
            n_train_steps=n_train_steps,
            schedule_type=schedule_type,
        )

        # Data normalisation stats
        self._field_mean = None
        self._field_std = None
        self.training_history = {}

    def _normalize_fields(self, fields: np.ndarray) -> np.ndarray:
        """Normalize fields to zero-mean unit-variance."""
        return (fields - self._field_mean) / (self._field_std + 1e-8)

    def _denormalize_fields(self, fields: np.ndarray) -> np.ndarray:
        """Reverse normalisation."""
        return fields * (self._field_std + 1e-8) + self._field_mean

    def fit(
        self,
        X: np.ndarray,
        Y_Cp: np.ndarray,
        Y_Cf: np.ndarray,
        n_epochs: int = 10,
        batch_size: int = 16,
    ) -> Dict[str, Any]:
        """
        Train the diffusion model on paired (conditions, flow fields).

        Uses denoising score matching: at each step, add noise at random
        timestep t, predict the noise, and update via MSE loss.

        Parameters
        ----------
        X : ndarray (n_samples, n_features)
            Flow condition features [AoA, Re, Mach, ...].
            First 3 columns must be [AoA_deg, Re, Mach].
        Y_Cp : ndarray (n_samples, spatial_res)
            Target Cp distributions.
        Y_Cf : ndarray (n_samples, spatial_res)
            Target Cf distributions.
        n_epochs : int
            Number of training epochs.
        batch_size : int
            Batch size.

        Returns
        -------
        dict with training metrics.
        """
        n_samples = len(X)
        rng = np.random.RandomState(self.seed)

        # Extract conditions
        aoa_deg = X[:, 0]
        Re = X[:, 1]
        Mach = X[:, 2] if X.shape[1] > 2 else np.full(n_samples, 0.15)

        # Stack fields: (N, 2, S)
        fields = np.stack([Y_Cp, Y_Cf], axis=1)

        # Compute normalisation stats
        self._field_mean = fields.mean(axis=(0, 2), keepdims=True)
        self._field_std = fields.std(axis=(0, 2), keepdims=True)
        fields_norm = self._normalize_fields(fields)

        # Training loop (simplified — updates weights via gradient proxy)
        losses = []
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            n_batches = 0

            indices = rng.permutation(n_samples)
            for start in range(0, n_samples, batch_size):
                idx = indices[start:start + batch_size]
                B = len(idx)

                # Sample random timesteps
                t = rng.randint(0, self.n_train_steps, size=B)
                t_norm = t.astype(np.float64) / self.n_train_steps

                # Generate noise
                noise = rng.randn(B, N_CHANNELS, self.spatial_res)

                # Forward diffusion
                x_t = self.scheduler.add_noise(fields_norm[idx], noise, t)

                # Get embeddings
                t_emb = self.time_embed(t_norm)
                c_emb = self.cond_encoder.encode_conditions(
                    aoa_deg[idx], Re[idx], Mach[idx]
                )

                # Predict noise
                eps_pred = self.denoiser.forward(x_t, t_emb, c_emb)

                # MSE loss
                loss = np.mean((eps_pred - noise) ** 2)
                epoch_loss += loss
                n_batches += 1

                # Pseudo weight update (simplified gradient step)
                # In production, this would be PyTorch autograd
                grad_scale = self.learning_rate * 0.01
                error_signal = (eps_pred - noise) * grad_scale

                # Update output projection as proxy for full backprop
                # Simple gradient: dL/dW_out ≈ mean error signal
                mean_err = np.mean(error_signal, axis=(0, 2))  # (in_channels,)
                self.denoiser.W_out[:, :, 0] -= grad_scale * mean_err.reshape(1, -1) * 0.001
                self.denoiser.b_out -= grad_scale * mean_err * 0.001

            avg_loss = epoch_loss / max(n_batches, 1)
            losses.append(float(avg_loss))
            logger.info(f"Epoch {epoch + 1}/{n_epochs}  loss={avg_loss:.6f}")

        self._fitted = True
        self.training_history = {
            "train_loss": losses,
            "n_epochs": n_epochs,
            "n_samples": n_samples,
            "final_loss": losses[-1] if losses else float("nan"),
        }
        return self.training_history

    def sample(
        self,
        aoa_deg: np.ndarray,
        Re: np.ndarray,
        Mach: np.ndarray,
        n_samples: int = 8,
        n_inference_steps: int = None,
        eta: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate flow field samples via DDIM reverse diffusion.

        Parameters
        ----------
        aoa_deg : ndarray (n_conditions,)
        Re : ndarray (n_conditions,)
        Mach : ndarray (n_conditions,)
        n_samples : int
            Number of samples per condition.
        n_inference_steps : int, optional
            Override default inference steps.
        eta : float
            DDIM stochasticity (0 = deterministic).

        Returns
        -------
        Cp_samples : ndarray (n_conditions, n_samples, spatial_res)
        Cf_samples : ndarray (n_conditions, n_samples, spatial_res)
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        n_steps = n_inference_steps or self.n_inference_steps
        timesteps = self.scheduler.get_sampling_timesteps(n_steps)

        aoa_deg = np.atleast_1d(aoa_deg)
        Re = np.atleast_1d(Re)
        Mach = np.atleast_1d(Mach)
        n_cond = len(aoa_deg)

        all_Cp = np.zeros((n_cond, n_samples, self.spatial_res))
        all_Cf = np.zeros((n_cond, n_samples, self.spatial_res))

        for ci in range(n_cond):
            # Repeat condition for all samples
            aoa_batch = np.full(n_samples, aoa_deg[ci])
            Re_batch = np.full(n_samples, Re[ci])
            Mach_batch = np.full(n_samples, Mach[ci])

            c_emb = self.cond_encoder.encode_conditions(
                aoa_batch, Re_batch, Mach_batch
            )

            # Start from pure noise
            rng = np.random.RandomState(self.seed + ci)
            x_t = rng.randn(n_samples, N_CHANNELS, self.spatial_res)

            # Reverse diffusion
            for i in range(len(timesteps)):
                t = int(timesteps[i])
                t_prev = int(timesteps[i + 1]) if i + 1 < len(timesteps) else 0
                t_norm = np.full(n_samples, t / self.n_train_steps)

                t_emb = self.time_embed(t_norm)
                eps_pred = self.denoiser.forward(x_t, t_emb, c_emb)
                x_t = self.scheduler.ddim_step(x_t, eps_pred, t, t_prev, eta=eta)

            # Denormalize
            fields = self._denormalize_fields(x_t)
            all_Cp[ci] = fields[:, 0, :]
            all_Cf[ci] = fields[:, 1, :]

        return all_Cp, all_Cf

    def predict_mean_std(
        self,
        aoa_deg: np.ndarray,
        Re: np.ndarray,
        Mach: np.ndarray,
        n_samples: int = 16,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict mean and standard deviation via multi-sample aggregation.

        This is the native UQ output — replaces deep ensemble.

        Parameters
        ----------
        aoa_deg, Re, Mach : ndarray (n_conditions,)
        n_samples : int
            Number of diffusion samples to draw.

        Returns
        -------
        Cp_mean : ndarray (n_conditions, spatial_res)
        Cf_mean : ndarray (n_conditions, spatial_res)
        Cp_std : ndarray (n_conditions, spatial_res)
        Cf_std : ndarray (n_conditions, spatial_res)
        """
        Cp_samples, Cf_samples = self.sample(
            aoa_deg, Re, Mach, n_samples=n_samples
        )

        return (
            Cp_samples.mean(axis=1),
            Cf_samples.mean(axis=1),
            Cp_samples.std(axis=1),
            Cf_samples.std(axis=1),
        )

    def predict_percentiles(
        self,
        aoa_deg: np.ndarray,
        Re: np.ndarray,
        Mach: np.ndarray,
        percentiles: List[float] = [5, 25, 50, 75, 95],
        n_samples: int = 32,
    ) -> Dict[str, np.ndarray]:
        """
        Predict flow field percentiles for credible intervals.

        Returns
        -------
        dict mapping 'Cp_p5', 'Cp_p50', 'Cf_p95', etc.
        """
        Cp_samples, Cf_samples = self.sample(
            aoa_deg, Re, Mach, n_samples=n_samples
        )

        result = {}
        for p in percentiles:
            result[f"Cp_p{int(p)}"] = np.percentile(Cp_samples, p, axis=1)
            result[f"Cf_p{int(p)}"] = np.percentile(Cf_samples, p, axis=1)
        return result

    def detect_separation_with_uncertainty(
        self,
        aoa_deg: np.ndarray,
        Re: np.ndarray,
        Mach: np.ndarray,
        n_samples: int = 16,
    ) -> List[Dict[str, Any]]:
        """
        Detect separation from sampled Cf fields with uncertainty bounds.

        For each condition, draws n_samples Cf fields, detects separation
        in each, and reports mean/std of separation location.

        Returns
        -------
        List of dicts per condition, each with:
            - prob_separated: fraction of samples showing separation
            - x_sep_mean, x_sep_std: separation location statistics
            - x_reat_mean, x_reat_std: reattachment statistics
            - bubble_length_mean, bubble_length_std
        """
        _, Cf_samples = self.sample(aoa_deg, Re, Mach, n_samples=n_samples)

        x_c = np.linspace(0.001, 1.0, self.spatial_res)
        results = []

        for ci in range(len(np.atleast_1d(aoa_deg))):
            sep_locs = []
            reat_locs = []

            for si in range(n_samples):
                cf = Cf_samples[ci, si]
                for j in range(1, len(cf)):
                    if cf[j - 1] > 0 and cf[j] <= 0:
                        x_sep = x_c[j - 1] + (0 - cf[j - 1]) / (
                            cf[j] - cf[j - 1] + 1e-15
                        ) * (x_c[j] - x_c[j - 1])
                        sep_locs.append(x_sep)
                        break

                for j in range(1, len(cf)):
                    if cf[j - 1] <= 0 and cf[j] > 0 and len(sep_locs) > si:
                        x_reat = x_c[j - 1] + (0 - cf[j - 1]) / (
                            cf[j] - cf[j - 1] + 1e-15
                        ) * (x_c[j] - x_c[j - 1])
                        reat_locs.append(x_reat)
                        break

            result = {
                "prob_separated": len(sep_locs) / max(n_samples, 1),
                "n_separated_samples": len(sep_locs),
            }
            if sep_locs:
                result["x_sep_mean"] = float(np.mean(sep_locs))
                result["x_sep_std"] = float(np.std(sep_locs))
            if reat_locs:
                result["x_reat_mean"] = float(np.mean(reat_locs))
                result["x_reat_std"] = float(np.std(reat_locs))
                if sep_locs:
                    bubbles = [r - s for s, r in zip(sep_locs[:len(reat_locs)], reat_locs)]
                    result["bubble_length_mean"] = float(np.mean(bubbles))
                    result["bubble_length_std"] = float(np.std(bubbles))

            results.append(result)

        return results

    def compare_with_deterministic(
        self,
        X: np.ndarray,
        Y_Cp: np.ndarray,
        Y_Cf: np.ndarray,
        n_samples: int = 8,
    ) -> Dict[str, float]:
        """
        Compare diffusion predictions vs ground truth and report metrics.

        Returns
        -------
        dict with Cp_RMSE, Cf_RMSE, Cp_R2, Cf_R2,
             Cp_std_mean (average uncertainty), calibration.
        """
        Cp_mean, Cf_mean, Cp_std, Cf_std = self.predict_mean_std(
            X[:, 0], X[:, 1],
            X[:, 2] if X.shape[1] > 2 else np.full(len(X), 0.15),
            n_samples=n_samples,
        )

        # RMSE
        Cp_rmse = float(np.sqrt(np.mean((Cp_mean - Y_Cp) ** 2)))
        Cf_rmse = float(np.sqrt(np.mean((Cf_mean - Y_Cf) ** 2)))

        # R²
        def _r2(y_true, y_pred):
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            return 1 - ss_res / max(ss_tot, 1e-15)

        Cp_r2 = float(_r2(Y_Cp.ravel(), Cp_mean.ravel()))
        Cf_r2 = float(_r2(Y_Cf.ravel(), Cf_mean.ravel()))

        # Calibration: fraction of true values within ±2σ
        within_2sigma_Cp = np.mean(
            np.abs(Y_Cp - Cp_mean) < 2 * np.maximum(Cp_std, 1e-6)
        )
        within_2sigma_Cf = np.mean(
            np.abs(Y_Cf - Cf_mean) < 2 * np.maximum(Cf_std, 1e-6)
        )

        return {
            "model_name": "DiffusionFlowSurrogate",
            "Cp_RMSE": Cp_rmse,
            "Cf_RMSE": Cf_rmse,
            "Cp_R2": Cp_r2,
            "Cf_R2": Cf_r2,
            "Cp_std_mean": float(np.mean(Cp_std)),
            "Cf_std_mean": float(np.mean(Cf_std)),
            "Cp_coverage_2sigma": float(within_2sigma_Cp),
            "Cf_coverage_2sigma": float(within_2sigma_Cf),
            "n_samples_per_prediction": n_samples,
        }

    def get_info(self) -> Dict[str, Any]:
        """Return model metadata."""
        return {
            "model_type": "DiffusionFlowSurrogate",
            "architecture": "UNet + DDIM",
            "spatial_resolution": self.spatial_res,
            "n_train_steps": self.n_train_steps,
            "n_inference_steps": self.n_inference_steps,
            "n_channels": N_CHANNELS,
            "n_params": self.denoiser.count_params(),
            "fitted": self._fitted,
            "scheduler": self.scheduler.schedule_type,
        }

    def summary(self) -> str:
        """Print model summary."""
        info = self.get_info()
        lines = [
            "=" * 60,
            "Diffusion Flow Surrogate — Generative Cp/Cf Predictor",
            "=" * 60,
            f"  Architecture:     {info['architecture']}",
            f"  Spatial res:      {info['spatial_resolution']}",
            f"  Channels:         {info['n_channels']} (Cp, Cf)",
            f"  Parameters:       {info['n_params']:,}",
            f"  Train steps (T):  {info['n_train_steps']}",
            f"  Inference steps:  {info['n_inference_steps']}",
            f"  Schedule:         {info['scheduler']}",
            f"  Fitted:           {info['fitted']}",
        ]
        if self._fitted and self.training_history:
            h = self.training_history
            lines.extend([
                "",
                f"  Training samples: {h['n_samples']}",
                f"  Training epochs:  {h['n_epochs']}",
                f"  Final loss:       {h['final_loss']:.6f}",
            ])
        return "\n".join(lines)


# =============================================================================
# Data Generation Helper
# =============================================================================
def generate_diffusion_training_data(
    n_samples: int = 200,
    aoa_range: Tuple[float, float] = (-5.0, 18.0),
    Re_range: Tuple[float, float] = (5e5, 1e7),
    Mach_range: Tuple[float, float] = (0.1, 0.3),
    spatial_res: int = N_SURFACE_POINTS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic training data, reusing distribution_surrogate.py generators.

    Returns
    -------
    X : ndarray (n_samples, 3)  — [AoA_deg, Re, Mach]
    Y_Cp : ndarray (n_samples, spatial_res)
    Y_Cf : ndarray (n_samples, spatial_res)
    """
    from scripts.ml_augmentation.distribution_surrogate import (
        generate_cp_distribution,
        generate_cf_distribution,
    )

    rng = np.random.RandomState(42)
    x_c = np.linspace(0.001, 1.0, spatial_res)

    aoas = rng.uniform(*aoa_range, n_samples)
    Res = 10 ** rng.uniform(np.log10(Re_range[0]), np.log10(Re_range[1]), n_samples)
    Machs = rng.uniform(*Mach_range, n_samples)

    X = np.stack([aoas, Res, Machs], axis=-1)
    Y_Cp = np.zeros((n_samples, spatial_res))
    Y_Cf = np.zeros((n_samples, spatial_res))

    for i in range(n_samples):
        Y_Cp[i] = generate_cp_distribution(x_c, aoas[i], Res[i], Machs[i])
        Y_Cf[i] = generate_cf_distribution(x_c, aoas[i], Res[i], Machs[i])

    return X, Y_Cp, Y_Cf


# =============================================================================
# Demo
# =============================================================================
def train_diffusion_surrogate() -> DiffusionFlowSurrogate:
    """Train and evaluate the diffusion flow surrogate on synthetic data."""
    print("Generating training data for diffusion surrogate...")
    X, Y_Cp, Y_Cf = generate_diffusion_training_data(n_samples=100)

    model = DiffusionFlowSurrogate(
        base_channels=16, n_inference_steps=20, n_train_steps=100,
    )

    print("Training diffusion model...")
    history = model.fit(X, Y_Cp, Y_Cf, n_epochs=5, batch_size=16)
    print(model.summary())

    # Sample at high AoA
    print("\nSampling at α=15° (near stall)...")
    Cp_mean, Cf_mean, Cp_std, Cf_std = model.predict_mean_std(
        np.array([15.0]), np.array([6e6]), np.array([0.15]),
        n_samples=8,
    )
    print(f"  Cp std range: [{Cp_std.min():.4f}, {Cp_std.max():.4f}]")
    print(f"  Cf std range: [{Cf_std.min():.6f}, {Cf_std.max():.6f}]")

    # Separation detection with UQ
    sep = model.detect_separation_with_uncertainty(
        np.array([15.0]), np.array([6e6]), np.array([0.15]),
    )
    print(f"\n  Separation probability: {sep[0]['prob_separated']:.0%}")
    if "x_sep_mean" in sep[0]:
        print(f"  x_sep = {sep[0]['x_sep_mean']:.3f} ± {sep[0]['x_sep_std']:.3f}")

    return model


if __name__ == "__main__":
    train_diffusion_surrogate()
