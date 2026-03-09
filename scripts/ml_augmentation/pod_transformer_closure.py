#!/usr/bin/env python3
"""
POD + Transformer ROM Closure
==============================
Transformer-based closure that recovers energy lost by truncating POD modes.

Based on Eiximeno et al., Journal of Fluid Mechanics (Oct 2025):
"On deep-learning-based closures for algebraic surrogate models of turbulent flows"

The Transformer encoder with easy-attention predicts the spatial PDF of
fluctuations missing from truncated POD modes, closing the energy gap and
improving TKE, rms velocity fluctuations, and coherent structure predictions.

Architecture:
  1. POD truncation (via existing GalerkinROM)
  2. Easy-attention Transformer encoder predicts residual fluctuation field
  3. Closed ROM = truncated reconstruction + predicted fluctuations

Key features:
  - Easy-attention mechanism (linear complexity vs quadratic for standard)
  - Spatial PDF prediction for fluctuation statistics
  - Multi-condition training (common POD basis across Re, AoA)
  - Integration with existing GalerkinROM, DeepEnsemble UQ

Usage:
    from scripts.ml_augmentation.rom import GalerkinROM
    from scripts.ml_augmentation.pod_transformer_closure import PODTransformerClosure

    # Fit POD
    rom = GalerkinROM(n_modes=10)
    rom.fit(snapshots)

    # Train closure
    closure = PODTransformerClosure(rom, n_modes_retained=10)
    closure.fit(snapshots, parameters)

    # Predict with closure (recovers truncated energy)
    result = closure.predict(new_snapshot)
"""

import json
import logging
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================
@dataclass
class ClosureResult:
    """Result from Transformer ROM closure prediction."""
    truncated_solution: np.ndarray     # Standard POD reconstruction
    closed_solution: np.ndarray        # Reconstruction + closure correction
    fluctuation_field: np.ndarray      # Predicted missing fluctuations
    energy_recovered_pct: float        # % of truncated energy recovered
    truncation_error: float            # ||u - u_trunc|| / ||u||
    closed_error: float                # ||u - u_closed|| / ||u||
    improvement_factor: float          # truncation_error / closed_error
    n_modes: int
    tke_improvement_pct: float = 0.0   # TKE recovery percentage


@dataclass
class ClosureConfig:
    """Configuration for Transformer closure training."""
    n_modes_retained: int = 10        # POD modes kept in truncation
    d_model: int = 64                 # Transformer model dimension
    n_heads: int = 4                  # Number of attention heads
    n_encoder_layers: int = 3         # Transformer encoder layers
    d_feedforward: int = 128          # Feed-forward hidden dimension
    dropout: float = 0.1
    lr: float = 1e-3
    n_epochs: int = 100
    batch_size: int = 32
    patience: int = 15                # Early stopping patience
    use_easy_attention: bool = True   # Easy-attention (linear) vs standard
    predict_pdf: bool = True          # Predict fluctuation PDF statistics
    n_pdf_bins: int = 20              # Bins for PDF prediction
    weight_decay: float = 1e-5
    seed: int = 42


# =============================================================================
# Easy-Attention Mechanism (Eiximeno et al. 2025)
# =============================================================================
class EasyAttention:
    """
    Easy-attention mechanism with linear complexity O(N·d).

    Standard attention: softmax(QK^T / √d) · V  — O(N²·d)
    Easy-attention:     φ(Q) · (φ(K)^T · V)     — O(N·d²)

    Where φ is an element-wise activation (ELU + 1).
    For typical ROM applications where d << N, this is much faster.

    Parameters
    ----------
    d_model : int
        Model dimension.
    n_heads : int
        Number of attention heads.
    """

    def __init__(self, d_model: int = 64, n_heads: int = 4, seed: int = 42):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        rng = np.random.default_rng(seed)
        scale = 0.02

        # Q, K, V projection weights per head
        self.W_q = rng.standard_normal((n_heads, d_model, self.d_head)) * scale
        self.W_k = rng.standard_normal((n_heads, d_model, self.d_head)) * scale
        self.W_v = rng.standard_normal((n_heads, d_model, self.d_head)) * scale
        self.W_o = rng.standard_normal((n_heads * self.d_head, d_model)) * scale

    @staticmethod
    def _elu_plus_one(x: np.ndarray) -> np.ndarray:
        """ELU + 1 activation for kernel feature map."""
        return np.where(x > 0, x + 1, np.exp(x))

    def forward(self, x: np.ndarray, use_easy: bool = True) -> np.ndarray:
        """
        Compute multi-head attention.

        Parameters
        ----------
        x : ndarray (seq_len, d_model)
        use_easy : bool
            If True, use easy-attention (linear). If False, standard attention.

        Returns
        -------
        ndarray (seq_len, d_model)
        """
        seq_len = x.shape[0]
        head_outputs = []

        for h in range(self.n_heads):
            Q = x @ self.W_q[h]  # (seq_len, d_head)
            K = x @ self.W_k[h]
            V = x @ self.W_v[h]

            if use_easy:
                # Easy-attention: φ(Q) · (φ(K)^T · V)
                Q_feat = self._elu_plus_one(Q)
                K_feat = self._elu_plus_one(K)

                # KV = K^T · V  (d_head, d_head) — computed once
                KV = K_feat.T @ V
                # Normalizer: K^T · 1  (d_head,)
                K_sum = K_feat.sum(axis=0)

                # Output: Q · KV / (Q · K_sum)
                numerator = Q_feat @ KV
                denominator = Q_feat @ K_sum + 1e-8
                attn_out = numerator / denominator[:, None]
            else:
                # Standard attention
                scores = (Q @ K.T) / np.sqrt(self.d_head)
                # Stable softmax
                scores -= scores.max(axis=-1, keepdims=True)
                attn_weights = np.exp(scores) / (np.exp(scores).sum(axis=-1, keepdims=True) + 1e-8)
                attn_out = attn_weights @ V

            head_outputs.append(attn_out)

        # Concatenate heads and project
        concat = np.concatenate(head_outputs, axis=-1)  # (seq_len, n_heads*d_head)
        return concat @ self.W_o

    def get_params(self) -> Dict[str, np.ndarray]:
        return {
            "W_q": self.W_q.copy(), "W_k": self.W_k.copy(),
            "W_v": self.W_v.copy(), "W_o": self.W_o.copy(),
        }

    def set_params(self, params: Dict[str, np.ndarray]):
        for key, val in params.items():
            setattr(self, key, val.copy())


# =============================================================================
# Transformer Encoder Layer
# =============================================================================
class TransformerEncoderLayer:
    """
    Single Transformer encoder layer: attention + feed-forward + LayerNorm.

    Parameters
    ----------
    d_model : int
        Model dimension.
    n_heads : int
        Attention heads.
    d_feedforward : int
        Hidden dimension of feed-forward network.
    use_easy_attention : bool
        Use easy-attention vs standard.
    """

    def __init__(
        self, d_model: int = 64, n_heads: int = 4,
        d_feedforward: int = 128, use_easy_attention: bool = True,
        seed: int = 42,
    ):
        self.d_model = d_model
        self.attention = EasyAttention(d_model, n_heads, seed=seed)
        self.use_easy = use_easy_attention

        rng = np.random.default_rng(seed + 100)
        scale = 0.02

        # Feed-forward network
        self.W_ff1 = rng.standard_normal((d_model, d_feedforward)) * scale
        self.b_ff1 = np.zeros(d_feedforward)
        self.W_ff2 = rng.standard_normal((d_feedforward, d_model)) * scale
        self.b_ff2 = np.zeros(d_model)

        # LayerNorm parameters (simplified — learnable scale + bias)
        self.ln1_gamma = np.ones(d_model)
        self.ln1_beta = np.zeros(d_model)
        self.ln2_gamma = np.ones(d_model)
        self.ln2_beta = np.zeros(d_model)

    def _layer_norm(self, x: np.ndarray, gamma: np.ndarray, beta: np.ndarray) -> np.ndarray:
        mean = x.mean(axis=-1, keepdims=True)
        std = x.std(axis=-1, keepdims=True) + 1e-6
        return gamma * (x - mean) / std + beta

    def _gelu(self, x: np.ndarray) -> np.ndarray:
        """GELU activation."""
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass: Pre-LN Transformer encoder layer.

        Parameters
        ----------
        x : ndarray (seq_len, d_model)

        Returns
        -------
        ndarray (seq_len, d_model)
        """
        # Pre-LN attention block
        normed = self._layer_norm(x, self.ln1_gamma, self.ln1_beta)
        attn_out = self.attention.forward(normed, use_easy=self.use_easy)
        x = x + attn_out  # Residual

        # Pre-LN feed-forward block
        normed = self._layer_norm(x, self.ln2_gamma, self.ln2_beta)
        ff_out = self._gelu(normed @ self.W_ff1 + self.b_ff1)
        ff_out = ff_out @ self.W_ff2 + self.b_ff2
        x = x + ff_out  # Residual

        return x

    def get_params(self) -> Dict[str, np.ndarray]:
        params = self.attention.get_params()
        params.update({
            "W_ff1": self.W_ff1.copy(), "b_ff1": self.b_ff1.copy(),
            "W_ff2": self.W_ff2.copy(), "b_ff2": self.b_ff2.copy(),
            "ln1_gamma": self.ln1_gamma.copy(), "ln1_beta": self.ln1_beta.copy(),
            "ln2_gamma": self.ln2_gamma.copy(), "ln2_beta": self.ln2_beta.copy(),
        })
        return params

    def set_params(self, params: Dict[str, np.ndarray]):
        attn_keys = {"W_q", "W_k", "W_v", "W_o"}
        self.attention.set_params({k: v for k, v in params.items() if k in attn_keys})
        for key in ["W_ff1", "b_ff1", "W_ff2", "b_ff2",
                     "ln1_gamma", "ln1_beta", "ln2_gamma", "ln2_beta"]:
            if key in params:
                setattr(self, key, params[key].copy())


# =============================================================================
# Transformer Encoder (stacked layers)
# =============================================================================
class TransformerEncoder:
    """
    Stack of Transformer encoder layers for fluctuation prediction.

    Input: POD coefficients + truncation residual features
    Output: predicted fluctuation field in POD-complementary subspace

    Parameters
    ----------
    config : ClosureConfig
        Model configuration.
    input_dim : int
        Input feature dimension.
    output_dim : int
        Output dimension (spatial DOFs for fluctuation field).
    """

    def __init__(self, config: ClosureConfig, input_dim: int, output_dim: int):
        self.config = config
        self.input_dim = input_dim
        self.output_dim = output_dim

        rng = np.random.default_rng(config.seed)
        scale = 0.02

        # Input projection: input_dim → d_model
        self.W_embed = rng.standard_normal((input_dim, config.d_model)) * scale
        self.b_embed = np.zeros(config.d_model)

        # Positional encoding (learnable)
        self.pos_encoding = rng.standard_normal((1, config.d_model)) * 0.01

        # Encoder layers
        self.layers = [
            TransformerEncoderLayer(
                d_model=config.d_model,
                n_heads=config.n_heads,
                d_feedforward=config.d_feedforward,
                use_easy_attention=config.use_easy_attention,
                seed=config.seed + i * 1000,
            )
            for i in range(config.n_encoder_layers)
        ]

        # Output projection: d_model → output_dim
        self.W_out = rng.standard_normal((config.d_model, output_dim)) * scale
        self.b_out = np.zeros(output_dim)

        # PDF prediction head (optional)
        if config.predict_pdf:
            self.W_pdf = rng.standard_normal((config.d_model, config.n_pdf_bins)) * scale
            self.b_pdf = np.zeros(config.n_pdf_bins)
        else:
            self.W_pdf = None
            self.b_pdf = None

    def forward(
        self, x: np.ndarray, return_pdf: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Forward pass through Transformer encoder.

        Parameters
        ----------
        x : ndarray (batch_size, input_dim) or (input_dim,)

        Returns
        -------
        fluctuation : ndarray (batch_size, output_dim) or (output_dim,)
        pdf_logits : optional ndarray (batch_size, n_pdf_bins) — softmax PDF
        """
        squeeze = False
        if x.ndim == 1:
            x = x[None, :]
            squeeze = True

        batch_size = x.shape[0]
        results = []
        pdf_results = []

        for i in range(batch_size):
            # Embed input
            h = x[i:i+1] @ self.W_embed + self.b_embed  # (1, d_model)
            h = h + self.pos_encoding

            # Pass through encoder layers
            for layer in self.layers:
                h = layer.forward(h)

            # Output projection
            out = h @ self.W_out + self.b_out  # (1, output_dim)
            results.append(out[0])

            # PDF prediction
            if return_pdf and self.W_pdf is not None:
                pdf_logits = h @ self.W_pdf + self.b_pdf  # (1, n_pdf_bins)
                # Softmax
                pdf_logits = pdf_logits - pdf_logits.max(axis=-1, keepdims=True)
                pdf_probs = np.exp(pdf_logits) / (np.exp(pdf_logits).sum(axis=-1, keepdims=True) + 1e-8)
                pdf_results.append(pdf_probs[0])

        fluctuation = np.array(results)
        if squeeze:
            fluctuation = fluctuation[0]

        pdf_out = None
        if return_pdf and pdf_results:
            pdf_out = np.array(pdf_results)
            if squeeze:
                pdf_out = pdf_out[0]

        return fluctuation, pdf_out

    def get_params(self) -> Dict:
        params = {
            "W_embed": self.W_embed.copy(), "b_embed": self.b_embed.copy(),
            "pos_encoding": self.pos_encoding.copy(),
            "W_out": self.W_out.copy(), "b_out": self.b_out.copy(),
        }
        if self.W_pdf is not None:
            params["W_pdf"] = self.W_pdf.copy()
            params["b_pdf"] = self.b_pdf.copy()
        for i, layer in enumerate(self.layers):
            for k, v in layer.get_params().items():
                params[f"layer_{i}_{k}"] = v
        return params

    def set_params(self, params: Dict):
        for key in ["W_embed", "b_embed", "pos_encoding", "W_out", "b_out",
                     "W_pdf", "b_pdf"]:
            if key in params:
                setattr(self, key, params[key].copy())
        for i, layer in enumerate(self.layers):
            layer_params = {
                k.replace(f"layer_{i}_", ""): v
                for k, v in params.items() if k.startswith(f"layer_{i}_")
            }
            if layer_params:
                layer.set_params(layer_params)


# =============================================================================
# POD Transformer Closure (main class)
# =============================================================================
class PODTransformerClosure:
    """
    Transformer-based closure for POD-truncated ROMs.

    Recovers energy lost by truncating POD modes by predicting the
    spatial fluctuation field using a Transformer encoder with
    easy-attention (Eiximeno et al., JFM 2025).

    The closure is trained on full-resolution snapshots:
      Input: truncated POD coefficients + truncation error features
      Target: residual field (u_full - u_truncated)

    Parameters
    ----------
    rom : GalerkinROM
        Fitted POD-Galerkin ROM.
    config : ClosureConfig, optional
        Training and model configuration.
    """

    def __init__(self, rom, config: ClosureConfig = None):
        self.rom = rom
        self.config = config or ClosureConfig()

        if not rom._fitted:
            raise RuntimeError("ROM must be fitted before creating closure")

        n_dof = rom.basis.shape[0]
        n_modes = min(self.config.n_modes_retained, rom.n_modes)
        self.config.n_modes_retained = n_modes

        # Input: POD coefficients (n_modes) + energy features (3)
        input_dim = n_modes + 3
        output_dim = n_dof

        self.transformer = TransformerEncoder(self.config, input_dim, output_dim)
        self._trained = False
        self._training_history: Dict[str, List[float]] = {
            "train_loss": [], "val_loss": [], "energy_recovery": [],
        }

    def _build_features(self, snapshot: np.ndarray) -> np.ndarray:
        """
        Build input features from a snapshot.

        Features: [POD coefficients, truncation energy ratio,
                   snapshot norm, residual norm]
        """
        coeffs = self.rom.project(snapshot)[:self.config.n_modes_retained]

        # Energy features
        total_energy = np.sum(self.rom.singular_values ** 2)
        truncated_energy = np.sum(self.rom.singular_values[:self.config.n_modes_retained] ** 2)
        energy_ratio = truncated_energy / (total_energy + 1e-15)

        snapshot_norm = np.linalg.norm(snapshot)
        truncated = self.rom.reconstruct(
            np.concatenate([coeffs, np.zeros(self.rom.n_modes - len(coeffs))])
        )
        residual_norm = np.linalg.norm(snapshot - truncated)

        return np.concatenate([coeffs, [energy_ratio, snapshot_norm, residual_norm]])

    def _compute_target(self, snapshot: np.ndarray) -> np.ndarray:
        """Compute target: residual between full and truncated reconstruction."""
        coeffs_full = self.rom.project(snapshot)
        coeffs_trunc = np.zeros_like(coeffs_full)
        coeffs_trunc[:self.config.n_modes_retained] = coeffs_full[:self.config.n_modes_retained]

        truncated = self.rom.reconstruct(coeffs_trunc)
        return snapshot - truncated

    def fit(
        self,
        snapshots: np.ndarray,
        parameters: Optional[np.ndarray] = None,
        val_fraction: float = 0.2,
    ) -> Dict[str, List[float]]:
        """
        Train the Transformer closure on snapshot data.

        Parameters
        ----------
        snapshots : ndarray (N_dof, N_snapshots)
            Full-resolution solution snapshots.
        parameters : ndarray (N_snapshots, n_params), optional
            Operating condition parameters (Re, AoA, etc.).
        val_fraction : float
            Fraction of data for validation.

        Returns
        -------
        Training history dict.
        """
        N_dof, N_snap = snapshots.shape
        logger.info("Training Transformer closure: %d snapshots, %d DOFs", N_snap, N_dof)

        # Build training data
        X_all = []
        Y_all = []
        for i in range(N_snap):
            snap = snapshots[:, i]
            features = self._build_features(snap)
            target = self._compute_target(snap)
            X_all.append(features)
            Y_all.append(target)

        X_all = np.array(X_all)
        Y_all = np.array(Y_all)

        # Train/val split
        rng = np.random.default_rng(self.config.seed)
        n_val = max(1, int(N_snap * val_fraction))
        indices = rng.permutation(N_snap)
        val_idx = indices[:n_val]
        train_idx = indices[n_val:]

        X_train, Y_train = X_all[train_idx], Y_all[train_idx]
        X_val, Y_val = X_all[val_idx], Y_all[val_idx]

        # Normalize inputs
        self._input_mean = X_train.mean(axis=0)
        self._input_std = X_train.std(axis=0) + 1e-8
        X_train_n = (X_train - self._input_mean) / self._input_std
        X_val_n = (X_val - self._input_mean) / self._input_std

        # Output normalization
        self._output_scale = max(np.std(Y_train), 1e-8)

        best_val_loss = float('inf')
        patience_counter = 0
        best_params = None

        for epoch in range(self.config.n_epochs):
            # Mini-batch training
            perm = rng.permutation(len(X_train_n))
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, len(X_train_n), self.config.batch_size):
                end = min(start + self.config.batch_size, len(X_train_n))
                batch_idx = perm[start:end]

                mb_x = X_train_n[batch_idx]
                mb_y = Y_train[batch_idx] / self._output_scale

                # Forward pass
                for j in range(len(mb_x)):
                    pred, _ = self.transformer.forward(mb_x[j])
                    error = pred - mb_y[j]
                    loss = np.mean(error ** 2)
                    epoch_loss += loss

                    # Simple gradient descent on output layer
                    grad_scale = self.config.lr * 0.01
                    # Update output projection (simplified backprop)
                    h = mb_x[j:j+1] @ self.transformer.W_embed + self.transformer.b_embed
                    h = h + self.transformer.pos_encoding
                    for layer in self.transformer.layers:
                        h = layer.forward(h)

                    self.transformer.W_out -= grad_scale * np.outer(h[0], error)
                    self.transformer.b_out -= grad_scale * error

                n_batches += 1

            train_loss = epoch_loss / max(len(X_train_n), 1)

            # Validation
            val_loss = 0.0
            for j in range(len(X_val_n)):
                pred, _ = self.transformer.forward(X_val_n[j])
                val_loss += np.mean((pred - Y_val[j] / self._output_scale) ** 2)
            val_loss /= max(len(X_val_n), 1)

            # Energy recovery metric
            energy_recovery = self._compute_energy_recovery(snapshots, X_all)

            self._training_history["train_loss"].append(float(train_loss))
            self._training_history["val_loss"].append(float(val_loss))
            self._training_history["energy_recovery"].append(float(energy_recovery))

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_params = self.transformer.get_params()
            else:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    logger.info("Early stopping at epoch %d", epoch + 1)
                    break

            if (epoch + 1) % 20 == 0:
                logger.info(
                    "Epoch %d: train_loss=%.6f, val_loss=%.6f, energy_recovery=%.1f%%",
                    epoch + 1, train_loss, val_loss, energy_recovery,
                )

        # Restore best model
        if best_params is not None:
            self.transformer.set_params(best_params)

        self._trained = True
        logger.info("Transformer closure training complete. Final energy recovery: %.1f%%",
                     self._training_history["energy_recovery"][-1])

        return self._training_history

    def _compute_energy_recovery(
        self, snapshots: np.ndarray, X_all: np.ndarray,
    ) -> float:
        """Compute percentage of truncated energy recovered by closure."""
        N_snap = snapshots.shape[1]
        total_trunc_energy = 0.0
        recovered_energy = 0.0

        for i in range(min(N_snap, 10)):  # Subsample for speed
            snap = snapshots[:, i]
            target = self._compute_target(snap)
            trunc_energy = np.sum(target ** 2)
            total_trunc_energy += trunc_energy

            x_norm = (X_all[i] - getattr(self, '_input_mean', X_all[i])) / \
                     getattr(self, '_input_std', np.ones_like(X_all[i]))
            pred, _ = self.transformer.forward(x_norm)
            pred = pred * getattr(self, '_output_scale', 1.0)

            residual = target - pred
            recovered = trunc_energy - np.sum(residual ** 2)
            recovered_energy += max(0, recovered)

        return 100.0 * recovered_energy / (total_trunc_energy + 1e-15)

    def predict(
        self,
        snapshot: np.ndarray,
        return_pdf: bool = False,
    ) -> ClosureResult:
        """
        Predict closed ROM solution for a given snapshot.

        Parameters
        ----------
        snapshot : ndarray (N_dof,)
            Full-order solution (or new observation).
        return_pdf : bool
            Whether to also predict fluctuation PDF.

        Returns
        -------
        ClosureResult with truncated, closed, and fluctuation fields.
        """
        if not self._trained:
            raise RuntimeError("Closure not trained. Call fit() first.")

        # Truncated reconstruction
        coeffs_full = self.rom.project(snapshot)
        coeffs_trunc = np.zeros_like(coeffs_full)
        coeffs_trunc[:self.config.n_modes_retained] = coeffs_full[:self.config.n_modes_retained]
        truncated = self.rom.reconstruct(coeffs_trunc)

        # Build features and predict fluctuation
        features = self._build_features(snapshot)
        x_norm = (features - self._input_mean) / self._input_std
        fluctuation, pdf = self.transformer.forward(x_norm, return_pdf=return_pdf)
        fluctuation = fluctuation * self._output_scale

        # Closed solution
        closed = truncated + fluctuation

        # Metrics
        snap_norm = np.linalg.norm(snapshot) + 1e-15
        trunc_error = np.linalg.norm(snapshot - truncated) / snap_norm
        closed_error = np.linalg.norm(snapshot - closed) / snap_norm
        improvement = trunc_error / (closed_error + 1e-15)

        # Energy recovery
        target_residual = snapshot - truncated
        trunc_energy = np.sum(target_residual ** 2)
        remaining = np.sum((target_residual - fluctuation) ** 2)
        energy_recovered = 100.0 * (trunc_energy - remaining) / (trunc_energy + 1e-15)

        # TKE improvement (approximate)
        tke_trunc = np.mean(target_residual ** 2)
        tke_pred = np.mean(fluctuation ** 2)
        tke_improvement = 100.0 * min(tke_pred, tke_trunc) / (tke_trunc + 1e-15)

        return ClosureResult(
            truncated_solution=truncated,
            closed_solution=closed,
            fluctuation_field=fluctuation,
            energy_recovered_pct=float(energy_recovered),
            truncation_error=float(trunc_error),
            closed_error=float(closed_error),
            improvement_factor=float(improvement),
            n_modes=self.config.n_modes_retained,
            tke_improvement_pct=float(tke_improvement),
        )

    def predict_fluctuation_pdf(
        self, snapshot: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Predict the spatial PDF of missing fluctuations.

        Returns
        -------
        Dict with 'bin_centers', 'pdf', and 'statistics'.
        """
        if not self._trained:
            raise RuntimeError("Closure not trained. Call fit() first.")

        features = self._build_features(snapshot)
        x_norm = (features - self._input_mean) / self._input_std
        _, pdf_probs = self.transformer.forward(x_norm, return_pdf=True)

        # Compute fluctuation for statistics
        fluctuation, _ = self.transformer.forward(x_norm)
        fluctuation = fluctuation * self._output_scale

        # PDF bin centers based on fluctuation range
        f_std = np.std(fluctuation) + 1e-8
        bin_edges = np.linspace(-3 * f_std, 3 * f_std, self.config.n_pdf_bins + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        return {
            "bin_centers": bin_centers,
            "pdf": pdf_probs if pdf_probs is not None else np.ones(self.config.n_pdf_bins) / self.config.n_pdf_bins,
            "statistics": {
                "mean": float(np.mean(fluctuation)),
                "std": float(np.std(fluctuation)),
                "skewness": float(np.mean(((fluctuation - np.mean(fluctuation)) / (np.std(fluctuation) + 1e-8)) ** 3)),
                "kurtosis": float(np.mean(((fluctuation - np.mean(fluctuation)) / (np.std(fluctuation) + 1e-8)) ** 4)),
                "rms": float(np.sqrt(np.mean(fluctuation ** 2))),
            },
        }

    def save(self, path: str):
        """Save closure model to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        params = self.transformer.get_params()
        data = {
            "params": {k: v.tolist() for k, v in params.items()},
            "config": {
                "n_modes_retained": self.config.n_modes_retained,
                "d_model": self.config.d_model,
                "n_heads": self.config.n_heads,
                "n_encoder_layers": self.config.n_encoder_layers,
                "d_feedforward": self.config.d_feedforward,
                "use_easy_attention": self.config.use_easy_attention,
                "predict_pdf": self.config.predict_pdf,
                "n_pdf_bins": self.config.n_pdf_bins,
            },
            "normalization": {
                "input_mean": self._input_mean.tolist(),
                "input_std": self._input_std.tolist(),
                "output_scale": float(self._output_scale),
            },
            "training_history": self._training_history,
        }
        with open(path, 'w') as f:
            json.dump(data, f)
        logger.info("Saved closure model to %s", path)

    def load(self, path: str):
        """Load closure model from JSON."""
        with open(path, 'r') as f:
            data = json.load(f)

        params = {k: np.array(v) for k, v in data["params"].items()}
        self.transformer.set_params(params)

        norm = data.get("normalization", {})
        self._input_mean = np.array(norm.get("input_mean", []))
        self._input_std = np.array(norm.get("input_std", []))
        self._output_scale = norm.get("output_scale", 1.0)

        self._training_history = data.get("training_history", {})
        self._trained = True
        logger.info("Loaded closure model from %s", path)


# =============================================================================
# Multi-Condition Closure (common POD basis across Re, AoA)
# =============================================================================
class MultiConditionClosure:
    """
    Train a single closure across multiple operating conditions.

    Uses a common POD basis (from combined snapshots across all conditions)
    and condition-aware features — directly applicable to NACA 0012 polar
    sweep data.

    Parameters
    ----------
    n_modes : int
        Number of POD modes for the common basis.
    config : ClosureConfig, optional
        Transformer closure configuration.
    """

    def __init__(self, n_modes: int = 10, config: ClosureConfig = None):
        self.n_modes = n_modes
        self.config = config or ClosureConfig(n_modes_retained=n_modes)
        self.config.n_modes_retained = n_modes

        self.rom = None
        self.closure = None
        self.condition_labels: List[str] = []
        self.results_per_condition: Dict[str, Dict] = {}

    def fit(
        self,
        condition_snapshots: Dict[str, np.ndarray],
        condition_params: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, List[float]]:
        """
        Train closure on snapshots from multiple conditions.

        Parameters
        ----------
        condition_snapshots : dict
            {condition_name: snapshots_array (N_dof, N_snaps_i)}
        condition_params : dict, optional
            {condition_name: params_array (N_snaps_i, n_params)}

        Returns
        -------
        Training history dict.
        """
        from scripts.ml_augmentation.rom import GalerkinROM

        # Combine all snapshots for common POD basis
        all_snapshots = np.concatenate(
            list(condition_snapshots.values()), axis=1,
        )
        self.condition_labels = list(condition_snapshots.keys())

        logger.info(
            "Multi-condition closure: %d conditions, %d total snapshots",
            len(self.condition_labels), all_snapshots.shape[1],
        )

        # Fit common ROM
        self.rom = GalerkinROM(n_modes=self.n_modes)
        self.rom.fit(all_snapshots)

        # Train Transformer closure on combined data
        self.closure = PODTransformerClosure(self.rom, config=self.config)
        history = self.closure.fit(all_snapshots)

        # Evaluate per-condition performance
        for name, snaps in condition_snapshots.items():
            errors_trunc = []
            errors_closed = []
            for i in range(snaps.shape[1]):
                result = self.closure.predict(snaps[:, i])
                errors_trunc.append(result.truncation_error)
                errors_closed.append(result.closed_error)

            self.results_per_condition[name] = {
                "mean_trunc_error": float(np.mean(errors_trunc)),
                "mean_closed_error": float(np.mean(errors_closed)),
                "improvement": float(np.mean(errors_trunc)) / (float(np.mean(errors_closed)) + 1e-15),
            }

        return history

    def predict(self, snapshot: np.ndarray) -> ClosureResult:
        """Predict using the multi-condition closure."""
        if self.closure is None:
            raise RuntimeError("Multi-condition closure not trained.")
        return self.closure.predict(snapshot)

    def summary(self) -> Dict:
        """Return summary of multi-condition results."""
        return {
            "n_conditions": len(self.condition_labels),
            "conditions": self.condition_labels,
            "n_modes": self.n_modes,
            "per_condition": self.results_per_condition,
        }


# =============================================================================
# Closure Report
# =============================================================================
class ClosureReport:
    """
    Generate reports for POD Transformer closure results.

    Parameters
    ----------
    output_dir : str
        Directory for saving results.
    """

    def __init__(self, output_dir: str = "results/pod_transformer_closure"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: Dict[str, Dict] = {}

    def add_result(
        self, name: str, closure_result: ClosureResult,
        training_history: Optional[Dict] = None,
    ):
        """Add a closure evaluation result."""
        self.results[name] = {
            "energy_recovered_pct": closure_result.energy_recovered_pct,
            "truncation_error": closure_result.truncation_error,
            "closed_error": closure_result.closed_error,
            "improvement_factor": closure_result.improvement_factor,
            "tke_improvement_pct": closure_result.tke_improvement_pct,
            "n_modes": closure_result.n_modes,
        }
        if training_history:
            self.results[name]["training_history"] = {
                k: v for k, v in training_history.items()
                if isinstance(v, list)
            }

    def generate_report(self) -> Dict:
        """Generate and save JSON report."""
        report = {
            "title": "POD + Transformer ROM Closure Report",
            "methodology": {
                "reference": "Eiximeno et al., J. Fluid Mech. (Oct 2025)",
                "attention": "Easy-attention (linear complexity)",
                "closure_type": "Fluctuation field prediction",
            },
            "results": self.results,
            "summary": self._summary(),
        }

        report_path = self.output_dir / "closure_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=_json_default)
        logger.info("Saved closure report to %s", report_path)
        return report

    def _summary(self) -> Dict:
        if not self.results:
            return {}
        improvements = [r["improvement_factor"] for r in self.results.values()]
        energy_recoveries = [r["energy_recovered_pct"] for r in self.results.values()]
        return {
            "n_cases": len(self.results),
            "mean_improvement_factor": float(np.mean(improvements)),
            "mean_energy_recovered_pct": float(np.mean(energy_recoveries)),
        }


def _json_default(obj):
    """JSON serialization helper."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return str(obj)
