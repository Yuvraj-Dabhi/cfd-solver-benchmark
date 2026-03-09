"""
ML Training Pipeline
====================
End-to-end training for turbulence model correction using
invariant features. Supports multiple architectures via
config-driven approach.

Usage:
    python train.py --config train_config.json
    python train.py --model-type correction --epochs 200 --lr 1e-3
"""

import json
import logging
import time
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    model_type: str = "correction"  # "correction", "pinn", "tbnn"
    hidden_layers: List[int] = field(default_factory=lambda: [64, 64, 32])
    activation: str = "relu"
    learning_rate: float = 1e-3
    lr_decay: float = 0.95
    lr_decay_steps: int = 50
    epochs: int = 200
    batch_size: int = 256
    early_stopping_patience: int = 20
    weight_decay: float = 1e-4
    dropout_rate: float = 0.1
    validation_split: float = 0.15
    seed: int = 42
    # PINN-specific
    lambda_physics: float = 1.0
    lambda_data: float = 1.0
    n_collocation: int = 5000


@dataclass
class TrainingHistory:
    """Training metrics history."""
    epochs: List[int] = field(default_factory=list)
    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    train_metric: List[float] = field(default_factory=list)
    val_metric: List[float] = field(default_factory=list)
    lr: List[float] = field(default_factory=list)
    best_epoch: int = 0
    best_val_loss: float = float("inf")
    wall_time_s: float = 0.0


class MLTrainer:
    """
    Config-driven ML training pipeline.

    Handles data preparation, model instantiation, training loop,
    early stopping, learning rate scheduling, and checkpointing.
    """

    def __init__(self, config: TrainingConfig = None):
        self.config = config or TrainingConfig()
        self.history = TrainingHistory()
        self.model = None
        self.scaler = None

    def prepare_data(
        self,
        features: np.ndarray,
        targets: np.ndarray,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Prepare training and validation data with standardization.

        Returns (train_data, val_data) dicts with X, y keys.
        """
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split

        np.random.seed(self.config.seed)

        # Standardize features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(features)

        # Split
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, targets,
            test_size=self.config.validation_split,
            random_state=self.config.seed,
        )

        logger.info(f"Training data: {X_train.shape}, Validation: {X_val.shape}")

        return (
            {"X": X_train, "y": y_train},
            {"X": X_val, "y": y_val},
        )

    def train(
        self,
        features: np.ndarray,
        targets: np.ndarray,
    ) -> TrainingHistory:
        """
        Full training loop with early stopping and LR scheduling.

        Uses scikit-learn MLPRegressor as the default backend.
        Falls back gracefully if TensorFlow is not available.
        """
        train_data, val_data = self.prepare_data(features, targets)

        t0 = time.time()

        try:
            history = self._train_sklearn(train_data, val_data)
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise

        self.history.wall_time_s = time.time() - t0
        logger.info(f"Training completed in {self.history.wall_time_s:.1f}s")
        logger.info(f"Best validation loss: {self.history.best_val_loss:.6f} "
                    f"at epoch {self.history.best_epoch}")

        return self.history

    def _train_sklearn(
        self, train_data: Dict, val_data: Dict,
    ) -> TrainingHistory:
        """Train using scikit-learn MLPRegressor with manual epoch tracking."""
        from sklearn.neural_network import MLPRegressor
        from sklearn.metrics import mean_squared_error

        self.model = MLPRegressor(
            hidden_layer_sizes=tuple(self.config.hidden_layers),
            activation=self.config.activation,
            learning_rate="adaptive",
            learning_rate_init=self.config.learning_rate,
            max_iter=1,  # We control epochs manually
            warm_start=True,
            random_state=self.config.seed,
            alpha=self.config.weight_decay,
            batch_size=min(self.config.batch_size, len(train_data["X"])),
            early_stopping=False,  # We handle this ourselves
        )

        patience_counter = 0

        for epoch in range(self.config.epochs):
            self.model.fit(train_data["X"], train_data["y"])

            # Compute losses
            train_pred = self.model.predict(train_data["X"])
            val_pred = self.model.predict(val_data["X"])

            train_loss = mean_squared_error(train_data["y"], train_pred)
            val_loss = mean_squared_error(val_data["y"], val_pred)

            self.history.epochs.append(epoch)
            self.history.train_loss.append(train_loss)
            self.history.val_loss.append(val_loss)

            # Early stopping
            if val_loss < self.history.best_val_loss:
                self.history.best_val_loss = val_loss
                self.history.best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.config.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

            if epoch % 50 == 0:
                logger.info(f"Epoch {epoch}: train_loss={train_loss:.6f}, "
                           f"val_loss={val_loss:.6f}")

        return self.history

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Make predictions on new data."""
        if self.model is None:
            raise RuntimeError("Model not trained yet. Call train() first.")
        X_scaled = self.scaler.transform(features)
        return self.model.predict(X_scaled)

    def save_model(self, path: Path) -> None:
        """Save trained model and scaler."""
        import pickle
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        with open(path / "model.pkl", "wb") as f:
            pickle.dump(self.model, f)
        with open(path / "scaler.pkl", "wb") as f:
            pickle.dump(self.scaler, f)

        # Save config
        with open(path / "config.json", "w") as f:
            json.dump(vars(self.config), f, indent=2)

        # Save history
        hist = {
            "epochs": self.history.epochs,
            "train_loss": self.history.train_loss,
            "val_loss": self.history.val_loss,
            "best_epoch": self.history.best_epoch,
            "best_val_loss": self.history.best_val_loss,
            "wall_time_s": self.history.wall_time_s,
        }
        with open(path / "history.json", "w") as f:
            json.dump(hist, f, indent=2)

        logger.info(f"Model saved to {path}")

    def load_model(self, path: Path) -> None:
        """Load trained model and scaler."""
        import pickle
        path = Path(path)

        with open(path / "model.pkl", "rb") as f:
            self.model = pickle.load(f)
        with open(path / "scaler.pkl", "rb") as f:
            self.scaler = pickle.load(f)

        with open(path / "config.json") as f:
            config_dict = json.load(f)
            self.config = TrainingConfig(**{
                k: v for k, v in config_dict.items()
                if k in TrainingConfig.__dataclass_fields__
            })

        logger.info(f"Model loaded from {path}")


# =============================================================================
# CLI
# =============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ML Training Pipeline")
    parser.add_argument("--config", type=Path, default=None,
                       help="Path to training config JSON")
    parser.add_argument("--data-dir", type=Path, default=None,
                       help="Path to dataset directory (npz + metadata)")
    parser.add_argument("--output-dir", type=Path, default=Path("models/"),
                       help="Model output directory")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", nargs="+", type=int, default=[64, 64, 32])

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    # Build config
    if args.config:
        with open(args.config) as f:
            config = TrainingConfig(**json.load(f))
    else:
        config = TrainingConfig(
            epochs=args.epochs,
            learning_rate=args.lr,
            hidden_layers=args.hidden,
        )

    # Load or generate demo data
    if args.data_dir and (args.data_dir / "data.npz").exists():
        from scripts.ml_augmentation.dataset import DatasetBuilder
        dataset = DatasetBuilder.load(args.data_dir)
        features = dataset.features
        targets = dataset.targets
    else:
        logger.info("Using synthetic demo data")
        features = np.random.randn(2000, 14)
        targets = np.random.randn(2000, 3) * 0.01

    trainer = MLTrainer(config)
    history = trainer.train(features, targets)
    trainer.save_model(args.output_dir)

    print(f"\nTraining complete: best val_loss = {history.best_val_loss:.6f} "
          f"at epoch {history.best_epoch}")
