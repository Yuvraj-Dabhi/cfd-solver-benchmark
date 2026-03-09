#!/usr/bin/env python3
"""
FIML — Neural Network Weight Embedding for SU2
================================================
Trains a neural network to map Galilean-invariant flow features
to the optimal beta correction field obtained from field inversion,
then exports the weights for direct embedding into SU2's turbulence
model source code.

Pipeline:
  1. Load optimal beta*(x) from field inversion (fiml_su2_adjoint.py)
  2. Extract Galilean-invariant features (q1-q5) at each mesh node
  3. Train NN: features(x) -> beta*(x)
  4. Export weights as JSON/C-header for SU2 runtime evaluation
  5. Validate: corrected SA with embedded NN improves separation

Feature Set (Galilean-Invariant):
  q1: Turbulence-to-mean-strain ratio (nu_t / (nu * |S| * d^2))
  q2: Wall-distance Reynolds number (min(sqrt(nu_t)*d / (50*nu), 2))
  q3: Strain-rotation ratio (S_ij O_ij / |S|^2)
  q4: Pressure gradient indicator (dp/dx * k / eps^2)
  q5: Turbulent viscosity ratio (nu_t / nu)

References:
  - Holland et al. (2019), JFM
  - Singh et al. (2017), "Machine-Learning-Augmented Predictive
    Modeling of Turbulent Separated Flows over Airfoils", AIAA J.
"""

import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)
PROJECT = Path(__file__).resolve().parent.parent.parent

try:
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, r2_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# =============================================================================
# NN Architecture for Beta Prediction
# =============================================================================
@dataclass
class EmbeddingConfig:
    """Configuration for the NN embedding."""
    n_features: int = 5
    hidden_layers: Tuple[int, ...] = (64, 64, 32)
    activation: str = "tanh"  # tanh for smooth beta field
    output_activation: str = "softplus"  # ensures beta > 0
    dropout: float = 0.0
    learning_rate: float = 1e-3
    max_epochs: int = 2000
    batch_size: int = 256
    early_stopping_patience: int = 50
    beta_min_clip: float = 0.1
    beta_max_clip: float = 5.0


class BetaCorrectionNN:
    """
    Neural network for mapping flow features to beta correction.

    Architecture: q1-q5 -> Dense(64,tanh) -> Dense(64,tanh) -> Dense(32,tanh) -> softplus -> beta

    The softplus output ensures beta > 0, corresponding to the
    physical constraint that the production term multiplier must
    be positive.
    """

    def __init__(self, config: EmbeddingConfig = None):
        self.config = config or EmbeddingConfig()
        self.scaler = None
        self.model = None
        self.training_history = {"train_loss": [], "val_loss": []}
        self.weights_exported = False

    def train(self, features: np.ndarray, beta_target: np.ndarray,
              val_split: float = 0.2) -> Dict:
        """
        Train the correction NN on (features, beta*) pairs.

        Parameters
        ----------
        features : array (N, 5)
            Galilean-invariant features at each mesh node.
        beta_target : array (N,)
            Optimal beta values from field inversion.
        val_split : float
            Fraction for validation.
        """
        n = len(features)
        n_val = int(n * val_split)
        perm = np.random.permutation(n)
        train_idx, val_idx = perm[n_val:], perm[:n_val]

        X_train, X_val = features[train_idx], features[val_idx]
        y_train, y_val = beta_target[train_idx], beta_target[val_idx]

        # Normalize inputs
        self.scaler = StandardScaler()
        X_train_norm = self.scaler.fit_transform(X_train)
        X_val_norm = self.scaler.transform(X_val)

        if HAS_TORCH:
            return self._train_torch(X_train_norm, y_train, X_val_norm, y_val)
        elif HAS_SKLEARN:
            return self._train_sklearn(X_train_norm, y_train, X_val_norm, y_val)
        else:
            raise RuntimeError("Requires PyTorch or scikit-learn")

    def _train_torch(self, X_train, y_train, X_val, y_val) -> Dict:
        """Train using PyTorch."""
        layers = []
        prev = self.config.n_features
        for h in self.config.hidden_layers:
            layers.append(nn.Linear(prev, h))
            if self.config.activation == "tanh":
                layers.append(nn.Tanh())
            else:
                layers.append(nn.ReLU())
            if self.config.dropout > 0:
                layers.append(nn.Dropout(self.config.dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        layers.append(nn.Softplus())  # Ensures beta > 0

        model = nn.Sequential(*layers)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=20, factor=0.5)

        X_t = torch.tensor(X_train, dtype=torch.float32)
        y_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        X_v = torch.tensor(X_val, dtype=torch.float32)
        y_v = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

        best_val = float('inf')
        patience_counter = 0

        for epoch in range(self.config.max_epochs):
            model.train()
            pred = model(X_t)
            loss = nn.MSELoss()(pred, y_t)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                val_pred = model(X_v)
                val_loss = nn.MSELoss()(val_pred, y_v).item()

            scheduler.step(val_loss)
            self.training_history["train_loss"].append(loss.item())
            self.training_history["val_loss"].append(val_loss)

            if val_loss < best_val - 1e-7:
                best_val = val_loss
                patience_counter = 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1

            if patience_counter >= self.config.early_stopping_patience:
                logger.info("Early stopping at epoch %d", epoch)
                break

        model.load_state_dict(best_state)
        self.model = model

        # Final metrics
        model.eval()
        with torch.no_grad():
            train_r2 = r2_score(y_train, model(X_t).numpy().flatten())
            val_r2 = r2_score(y_val, model(X_v).numpy().flatten())

        return {"train_r2": train_r2, "val_r2": val_r2,
                "best_val_loss": best_val, "epochs": epoch + 1}

    def _train_sklearn(self, X_train, y_train, X_val, y_val) -> Dict:
        """Train using sklearn MLPRegressor (fallback)."""
        self.model = MLPRegressor(
            hidden_layer_sizes=self.config.hidden_layers,
            activation='tanh',
            max_iter=self.config.max_epochs,
            learning_rate_init=self.config.learning_rate,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=42,
        )
        self.model.fit(X_train, y_train)

        train_r2 = r2_score(y_train, self.model.predict(X_train))
        val_r2 = r2_score(y_val, self.model.predict(X_val))

        return {"train_r2": train_r2, "val_r2": val_r2,
                "epochs": self.model.n_iter_}

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict beta correction for new flow features."""
        X_norm = self.scaler.transform(features)
        if HAS_TORCH and isinstance(self.model, nn.Module):
            self.model.eval()
            with torch.no_grad():
                pred = self.model(torch.tensor(X_norm, dtype=torch.float32))
                return np.clip(pred.numpy().flatten(),
                               self.config.beta_min_clip, self.config.beta_max_clip)
        else:
            pred = self.model.predict(X_norm)
            return np.clip(pred, self.config.beta_min_clip, self.config.beta_max_clip)

    # =========================================================================
    # Weight Export for SU2 Embedding
    # =========================================================================
    def export_weights_json(self, output_path: Path) -> Path:
        """
        Export NN weights as JSON for SU2 embedding.

        The JSON format is compatible with SU2's CReadNeuralNetwork
        class, which evaluates the NN at each mesh node during
        the flow iteration.

        Structure:
        {
            "n_layers": int,
            "layer_sizes": [n_in, h1, h2, ..., n_out],
            "activation": str,
            "scaler_mean": [...],
            "scaler_std": [...],
            "weights": [W0, W1, ...],
            "biases": [b0, b1, ...]
        }
        """
        export = {
            "n_features": self.config.n_features,
            "n_layers": len(self.config.hidden_layers) + 1,
            "layer_sizes": [self.config.n_features] +
                            list(self.config.hidden_layers) + [1],
            "activation": self.config.activation,
            "output_activation": self.config.output_activation,
            "scaler_mean": self.scaler.mean_.tolist(),
            "scaler_std": self.scaler.scale_.tolist(),
            "weights": [],
            "biases": [],
        }

        if HAS_TORCH and isinstance(self.model, nn.Module):
            for name, param in self.model.named_parameters():
                if 'weight' in name:
                    export["weights"].append(param.detach().numpy().tolist())
                elif 'bias' in name:
                    export["biases"].append(param.detach().numpy().tolist())
        elif HAS_SKLEARN:
            for i, (W, b) in enumerate(zip(self.model.coefs_, self.model.intercepts_)):
                export["weights"].append(W.tolist())
                export["biases"].append(b.tolist())

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(export, f, indent=2)

        self.weights_exported = True
        logger.info("Exported NN weights to %s", output_path)
        return output_path

    def export_weights_c_header(self, output_path: Path) -> Path:
        """
        Export NN weights as a C header file for direct compilation
        into SU2's turbulence model source code.

        This enables the NN to be evaluated at each mesh node during
        the SA production term computation without file I/O overhead.
        """
        lines = [
            "/* Auto-generated FIML correction NN weights */",
            "/* Trained on: field inversion beta correction */",
            f"/* Features: q1-q5 Galilean-invariant */",
            f"/* Architecture: {list(self.config.hidden_layers)} */",
            "",
            f"#define FIML_N_FEATURES {self.config.n_features}",
            f"#define FIML_N_LAYERS {len(self.config.hidden_layers) + 1}",
            "",
            "/* Input normalization */",
        ]

        # Scaler
        lines.append(f"static const double FIML_SCALER_MEAN[{self.config.n_features}] = {{")
        lines.append("  " + ", ".join(f"{v:.10e}" for v in self.scaler.mean_))
        lines.append("};")
        lines.append(f"static const double FIML_SCALER_STD[{self.config.n_features}] = {{")
        lines.append("  " + ", ".join(f"{v:.10e}" for v in self.scaler.scale_))
        lines.append("};")

        # Layer sizes
        sizes = [self.config.n_features] + list(self.config.hidden_layers) + [1]
        lines.append(f"\nstatic const int FIML_LAYER_SIZES[{len(sizes)}] = {{")
        lines.append("  " + ", ".join(str(s) for s in sizes))
        lines.append("};")

        # Weight matrices and bias vectors
        if HAS_TORCH and isinstance(self.model, nn.Module):
            layer_idx = 0
            for name, param in self.model.named_parameters():
                W = param.detach().numpy()
                if 'weight' in name:
                    rows, cols = W.shape
                    lines.append(f"\n/* Layer {layer_idx} weights [{rows}x{cols}] */")
                    lines.append(f"static const double FIML_W{layer_idx}[{rows}][{cols}] = {{")
                    for row in W:
                        lines.append("  {" + ", ".join(f"{v:.10e}" for v in row) + "},")
                    lines.append("};")
                elif 'bias' in name:
                    lines.append(f"static const double FIML_B{layer_idx}[{len(W)}] = {{")
                    lines.append("  " + ", ".join(f"{v:.10e}" for v in W))
                    lines.append("};")
                    layer_idx += 1

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(lines), encoding='utf-8')
        logger.info("Exported C header to %s", output_path)
        return output_path

    def export_torchscript(self, output_path: Path, batch_size: int = 1000) -> Path:
        """
        Export model as TorchScript for native C++ inference via libtorch.

        Eliminates Python GIL overhead by compiling the model graph into
        an intermediate representation executable by the C++ runtime.

        Parameters
        ----------
        output_path : Path
            Where to save the .pt TorchScript file.
        batch_size : int
            Sample batch size for tracing.

        Returns
        -------
        Path to the exported TorchScript file.
        """
        if not HAS_TORCH or not isinstance(self.model, nn.Module):
            raise RuntimeError("TorchScript export requires PyTorch model")

        from scripts.ml_augmentation.native_inference import TorchScriptExporter
        exporter = TorchScriptExporter(validate=True)
        result = exporter.export_correction_nn(
            self.model,
            n_features=self.config.n_features,
            output_path=output_path,
            batch_size=batch_size,
        )

        if not result.validation_passed:
            logger.warning("TorchScript validation failed: max_err=%.2e", result.max_abs_error)

        logger.info(
            "Exported TorchScript: %s (%d params, %.1f KB)",
            output_path, result.n_parameters, result.file_size_bytes / 1024,
        )
        return Path(result.output_path)

    def predict_batch_native(self, features: np.ndarray) -> np.ndarray:
        """
        High-performance batch prediction using torch.inference_mode().

        Faster than predict() because:
        1. Uses inference_mode (disables autograd + view tracking)
        2. Uses float32 tensor (no double conversion)
        3. Pre-normalizes in vectorized numpy

        Parameters
        ----------
        features : ndarray (N, n_features)

        Returns
        -------
        beta : ndarray (N,)
        """
        if not HAS_TORCH or not isinstance(self.model, nn.Module):
            return self.predict(features)

        X_norm = self.scaler.transform(features).astype(np.float32)
        input_t = torch.as_tensor(X_norm)  # Zero-copy if contiguous float32

        self.model.eval()
        with torch.inference_mode():
            pred = self.model(input_t)

        return np.clip(
            pred.numpy().flatten(),
            self.config.beta_min_clip,
            self.config.beta_max_clip,
        )


# =============================================================================
# Feature Extraction from SU2 Solution
# =============================================================================
def extract_fiml_features_from_field(
    nu_t: np.ndarray,
    nu: float,
    strain_magnitude: np.ndarray,
    wall_distance: np.ndarray,
    strain_rotation_ratio: np.ndarray,
    pressure_gradient_indicator: np.ndarray,
) -> np.ndarray:
    """
    Extract the 5 Galilean-invariant FIML features at each mesh node.

    Returns array (N, 5) of [q1, q2, q3, q4, q5].
    """
    d = np.maximum(wall_distance, 1e-15)
    S = np.maximum(strain_magnitude, 1e-15)
    nut = np.maximum(nu_t, 0.0)

    q1 = np.minimum(nut / (nu * S * d**2 + 1e-15), 10.0)
    q2 = np.minimum(np.sqrt(nut) * d / (50 * nu + 1e-15), 2.0)
    q3 = strain_rotation_ratio
    q4 = pressure_gradient_indicator
    q5 = np.minimum(nut / nu, 1e4)

    return np.column_stack([q1, q2, q3, q4, q5])


# =============================================================================
# Synthetic Demo
# =============================================================================
def run_embedding_demo():
    """
    Demonstrate the NN embedding pipeline with synthetic data.

    Creates synthetic features + beta, trains NN, exports weights,
    and validates prediction quality.
    """
    print("=" * 65)
    print("  FIML — NN Weight Embedding Demo")
    print("=" * 65)

    rng = np.random.default_rng(42)
    n = 2000

    # Synthetic features (q1-q5)
    x = np.linspace(0, 2, n)
    q1 = 0.5 * np.exp(-((x - 1.0)**2) / 0.1) + rng.normal(0, 0.02, n)
    q2 = np.clip(rng.uniform(0, 2, n), 0, 2)
    q3 = 0.2 * np.sin(2 * np.pi * x) + rng.normal(0, 0.05, n)
    q4 = -0.5 * np.exp(-((x - 0.7)**2) / 0.05) + rng.normal(0, 0.02, n)
    q5 = rng.lognormal(0, 1, n)

    features = np.column_stack([q1, q2, q3, q4, q5])

    # Synthetic beta (elevated in separation region)
    beta = np.ones(n)
    sep_mask = (x > 0.65) & (x < 1.1)
    beta[sep_mask] = 1.0 + 0.4 * np.sin(np.pi * (x[sep_mask] - 0.65) / 0.45)

    # Train
    print("\n  Training NN on q1-q5 -> beta mapping...")
    nn_model = BetaCorrectionNN(EmbeddingConfig(
        hidden_layers=(32, 32, 16), max_epochs=500))
    result = nn_model.train(features, beta)
    print(f"  Train R2: {result['train_r2']:.4f}")
    print(f"  Val R2:   {result['val_r2']:.4f}")
    print(f"  Epochs:   {result['epochs']}")

    # Predict
    beta_pred = nn_model.predict(features)
    rmse = np.sqrt(np.mean((beta_pred - beta)**2))
    print(f"\n  Prediction RMSE: {rmse:.6f}")
    print(f"  Beta range: [{beta_pred.min():.4f}, {beta_pred.max():.4f}]")

    # Export
    output_dir = PROJECT / "results" / "fiml_embedding_demo"
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = nn_model.export_weights_json(output_dir / "fiml_nn_weights.json")
    print(f"\n  JSON weights: {json_path}")

    c_path = nn_model.export_weights_c_header(output_dir / "fiml_nn_weights.h")
    print(f"  C header:     {c_path}")

    # Validate weight export
    with open(json_path) as f:
        weights = json.load(f)
    print(f"\n  Exported NN:")
    print(f"    Layers: {weights['layer_sizes']}")
    print(f"    Activation: {weights['activation']}")
    print(f"    Total parameters: {sum(np.prod(np.array(w).shape) for w in weights['weights']) + sum(len(b) for b in weights['biases'])}")

    print(f"\n  Demo complete!")
    return nn_model


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_embedding_demo()
