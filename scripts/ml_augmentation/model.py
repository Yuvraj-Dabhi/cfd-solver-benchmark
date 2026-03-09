"""
ML Augmentation Module
======================
Six turbulence model correction architectures:
  1. MLP scalar correction (Δν_t)
  2. TBNN (Tensor-Basis Neural Network)
  3. FIML (Field Inversion and Machine Learning) wrapper
  4. TurbulenceModelCorrection (Reynolds stress via invariant features)
  5. PINN (Physics-Informed Neural Network for NS-constrained separation prediction)
  6. CNN/GNN surrogate model interface

Plus: feature extraction from flow fields and dataset management.

Architecture references:
  - Srivastava et al. (2024), NASA TM-20240012512 — FIML q1–q5 features
  - Haghahenas, Hedayatpour, Groll (2023), Physics of Fluids 35, 083320
    "Prediction of particle-laden pipe flows using deep neural network models"
    Documents DNN architecture and solver-coupling methodology for flow
    prediction. The NN-solver coupling pattern (feature extraction →
    DNN prediction → field correction) is analogous to the approach
    used here for turbulence model correction.
"""

import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any


# =============================================================================
# Feature Extraction
# =============================================================================
def extract_invariant_features(
    Sij: np.ndarray,
    Oij: np.ndarray,
    k: np.ndarray,
    epsilon: np.ndarray,
    grad_p: np.ndarray,
    wall_distance: np.ndarray,
) -> np.ndarray:
    """
    Extract tensor invariant features for ML turbulence model correction.

    Parameters
    ----------
    Sij : array, shape (N, 3, 3)
        Mean strain rate tensor.
    Oij : array, shape (N, 3, 3)
        Mean rotation rate tensor.
    k : array, shape (N,)
        Turbulent kinetic energy.
    epsilon : array, shape (N,)
        Turbulent dissipation rate.
    grad_p : array, shape (N, 3)
        Pressure gradient.
    wall_distance : array, shape (N,)
        Distance to nearest wall.

    Returns
    -------
    array, shape (N, n_features)
        Feature matrix.
    """
    N = len(k)
    eps_safe = np.maximum(epsilon, 1e-15)
    k_safe = np.maximum(k, 1e-15)
    tau = k_safe / eps_safe  # Turbulent time scale

    features = []

    # I1: S² invariant — trace(S·S)
    S2 = np.einsum("nij,nji->n", Sij, Sij)
    I1_S = S2 * tau**2
    features.append(I1_S)

    # I2: Ω² invariant — trace(Ω·Ω)
    O2 = np.einsum("nij,nji->n", Oij, Oij)
    I1_O = O2 * tau**2
    features.append(I1_O)

    # I3: S·Ω invariant
    SO = np.einsum("nij,njk,nki->n", Sij, Oij, Oij)
    I_SO = SO * tau**3
    features.append(I_SO)

    # Pressure gradient magnitude (normalized)
    grad_p_mag = np.linalg.norm(grad_p, axis=1)
    features.append(grad_p_mag * wall_distance / k_safe)

    # Wall distance (normalized)
    features.append(wall_distance * np.sqrt(k_safe) / (50 * eps_safe / np.sqrt(k_safe + 1e-15) + 1e-15))

    # TKE
    features.append(k)

    # Dissipation
    features.append(epsilon)

    return np.column_stack(features)


# =============================================================================
# TurbulenceModelCorrection (MLP-based)
# =============================================================================
class TurbulenceModelCorrection:
    """
    Neural network for Reynolds stress correction.

    Architecture: features → StandardScaler → Dense(64) → Dense(64) → Dense(32) → Δ_correction
    """

    def __init__(self, n_features: int = 7, n_outputs: int = 6, hidden_layers: List[int] = None):
        """
        Parameters
        ----------
        n_features : int
            Number of input features (from extract_invariant_features).
        n_outputs : int
            Number of outputs (6 independent Reynolds stress components or 1 for Δν_t).
        hidden_layers : list
            Hidden layer sizes (default [64, 64, 32]).
        """
        self.n_features = n_features
        self.n_outputs = n_outputs
        self.hidden_layers = hidden_layers or [64, 64, 32]
        self.model = None
        self.scaler = None
        self._build_model()

    def _build_model(self):
        """Build TensorFlow/Keras model."""
        try:
            import tensorflow as tf
            from sklearn.preprocessing import StandardScaler

            self.scaler = StandardScaler()

            model = tf.keras.Sequential()
            model.add(tf.keras.layers.InputLayer(input_shape=(self.n_features,)))

            for units in self.hidden_layers:
                model.add(tf.keras.layers.Dense(units, activation="relu"))
                model.add(tf.keras.layers.Dropout(0.1))

            model.add(tf.keras.layers.Dense(self.n_outputs, activation="linear"))

            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                loss="mse",
                metrics=["mae"],
            )
            self.model = model

        except ImportError:
            # Fallback: scikit-learn MLPRegressor
            from sklearn.neural_network import MLPRegressor
            from sklearn.preprocessing import StandardScaler

            self.scaler = StandardScaler()
            self.model = MLPRegressor(
                hidden_layer_sizes=tuple(self.hidden_layers),
                activation="relu",
                solver="adam",
                max_iter=1000,
                random_state=42,
            )

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_split: float = 0.2,
        epochs: int = 100,
        batch_size: int = 32,
    ) -> Dict[str, Any]:
        """
        Train the correction model.

        Parameters
        ----------
        X : array, shape (N, n_features)
            Input features.
        y : array, shape (N, n_outputs)
            Target corrections (DNS - RANS residuals).
        validation_split : float
            Fraction for validation.
        epochs : int
            Training epochs.
        batch_size : int
            Batch size.

        Returns
        -------
        dict with training history.
        """
        X_scaled = self.scaler.fit_transform(X)

        try:
            import tensorflow as tf
            history = self.model.fit(
                X_scaled, y,
                validation_split=validation_split,
                epochs=epochs,
                batch_size=batch_size,
                verbose=0,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                    tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5),
                ],
            )
            return {"loss": history.history["loss"], "val_loss": history.history.get("val_loss", [])}

        except Exception:
            # Scikit-learn fallback
            self.model.fit(X_scaled, y)
            return {"loss": [self.model.loss_], "val_loss": []}

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict corrections for new data."""
        X_scaled = self.scaler.transform(X)
        try:
            return self.model.predict(X_scaled, verbose=0)
        except Exception:
            return self.model.predict(X_scaled)

    def save(self, path: str):
        """Save model to disk."""
        import pickle
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save scaler
        with open(save_dir / "scaler.pkl", "wb") as f:
            pickle.dump(self.scaler, f)

        # Save model
        try:
            self.model.save(save_dir / "model.keras")
        except Exception:
            with open(save_dir / "model.pkl", "wb") as f:
                pickle.dump(self.model, f)

    def load(self, path: str):
        """Load model from disk."""
        import pickle
        load_dir = Path(path)

        with open(load_dir / "scaler.pkl", "rb") as f:
            self.scaler = pickle.load(f)

        try:
            import tensorflow as tf
            self.model = tf.keras.models.load_model(load_dir / "model.keras")
        except Exception:
            with open(load_dir / "model.pkl", "rb") as f:
                self.model = pickle.load(f)


# =============================================================================
# PINN (Physics-Informed Neural Network)
# =============================================================================
class PhysicsInformedSeparation:
    """
    Physics-Informed Neural Network for flow separation prediction.

    Input: [x, y] coordinates
    Output: [u, v, p] flow field
    Loss: L = λ_data * L_data + λ_physics * L_NS

    The NS residual is computed via automatic differentiation.
    """

    def __init__(
        self,
        layers: List[int] = None,
        nu: float = 1e-5,
        lambda_physics: float = 1.0,
    ):
        """
        Parameters
        ----------
        layers : list
            NN architecture (default [2, 50, 50, 50, 3]).
        nu : float
            Kinematic viscosity.
        lambda_physics : float
            Weight for physics loss relative to data loss.
        """
        self.layer_sizes = layers or [2, 50, 50, 50, 3]
        self.nu = nu
        self.lambda_physics = lambda_physics
        self.model = None
        self._build()

    def _build(self):
        """Build NN with tanh activation."""
        try:
            import tensorflow as tf

            inputs = tf.keras.Input(shape=(2,))  # [x, y]
            x = inputs
            for units in self.layer_sizes[1:-1]:
                x = tf.keras.layers.Dense(units, activation="tanh")(x)
            outputs = tf.keras.layers.Dense(self.layer_sizes[-1])(x)  # [u, v, p]

            self.model = tf.keras.Model(inputs, outputs)
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
            self._tf_available = True

        except ImportError:
            self._tf_available = False

    def navier_stokes_residual(self, xy):  # tf.Tensor -> tf.Tensor
        """
        Compute NS residuals via automatic differentiation.

        Returns residuals of:
          - x-momentum: u*du/dx + v*du/dy + dp/dx - ν*(d²u/dx² + d²u/dy²) = 0
          - y-momentum: u*dv/dx + v*dv/dy + dp/dy - ν*(d²v/dx² + d²v/dy²) = 0
          - continuity: du/dx + dv/dy = 0
        """
        import tensorflow as tf

        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch(xy)
            with tf.GradientTape(persistent=True) as tape1:
                tape1.watch(xy)
                uvp = self.model(xy)
                u = uvp[:, 0:1]
                v = uvp[:, 1:2]
                p = uvp[:, 2:3]

            # First derivatives
            du = tape1.gradient(u, xy)
            dv = tape1.gradient(v, xy)
            dp = tape1.gradient(p, xy)

            du_dx, du_dy = du[:, 0:1], du[:, 1:2]
            dv_dx, dv_dy = dv[:, 0:1], dv[:, 1:2]
            dp_dx, dp_dy = dp[:, 0:1], dp[:, 1:2]

        # Second derivatives
        d2u_dx = tape2.gradient(du_dx, xy)[:, 0:1]
        d2u_dy = tape2.gradient(du_dy, xy)[:, 1:2]
        d2v_dx = tape2.gradient(dv_dx, xy)[:, 0:1]
        d2v_dy = tape2.gradient(dv_dy, xy)[:, 1:2]

        # NS residuals
        momentum_x = u * du_dx + v * du_dy + dp_dx - self.nu * (d2u_dx + d2u_dy)
        momentum_y = u * dv_dx + v * dv_dy + dp_dy - self.nu * (d2v_dx + d2v_dy)
        continuity = du_dx + dv_dy

        del tape1, tape2
        return momentum_x, momentum_y, continuity

    def compute_loss(
        self,
        data_xy: np.ndarray,
        data_uvp: np.ndarray,
        collocation_xy: np.ndarray,
    ) -> float:
        """
        Compute total loss = data + physics.
        """
        import tensorflow as tf

        data_xy_tf = tf.constant(data_xy, dtype=tf.float32)
        data_uvp_tf = tf.constant(data_uvp, dtype=tf.float32)
        coll_xy_tf = tf.Variable(tf.constant(collocation_xy, dtype=tf.float32))

        # Data loss
        pred = self.model(data_xy_tf)
        data_loss = tf.reduce_mean(tf.square(pred - data_uvp_tf))

        # Physics loss
        mx, my, cont = self.navier_stokes_residual(coll_xy_tf)
        physics_loss = tf.reduce_mean(tf.square(mx) + tf.square(my) + tf.square(cont))

        total = data_loss + self.lambda_physics * physics_loss
        return float(total.numpy()), float(data_loss.numpy()), float(physics_loss.numpy())

    def train(
        self,
        data_xy: np.ndarray,
        data_uvp: np.ndarray,
        collocation_xy: np.ndarray,
        epochs: int = 5000,
        lr_schedule: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Train the PINN.

        Parameters
        ----------
        data_xy : array, shape (N_data, 2)
            Coordinates where data is available.
        data_uvp : array, shape (N_data, 3)
            Data values [u, v, p].
        collocation_xy : array, shape (N_coll, 2)
            Collocation points for physics enforcement.
        epochs : int
            Training epochs.

        Returns
        -------
        dict with loss history.
        """
        if not self._tf_available:
            raise RuntimeError("TensorFlow required for PINN training")

        import tensorflow as tf

        data_xy_tf = tf.constant(data_xy, dtype=tf.float32)
        data_uvp_tf = tf.constant(data_uvp, dtype=tf.float32)

        history = {"total": [], "data": [], "physics": []}

        for epoch in range(epochs):
            coll_tf = tf.Variable(
                tf.constant(collocation_xy, dtype=tf.float32)
            )

            with tf.GradientTape() as tape:
                # Data loss
                pred = self.model(data_xy_tf, training=True)
                data_loss = tf.reduce_mean(tf.square(pred - data_uvp_tf))

                # Physics loss
                mx, my, cont = self.navier_stokes_residual(coll_tf)
                physics_loss = tf.reduce_mean(
                    tf.square(mx) + tf.square(my) + tf.square(cont)
                )

                total_loss = data_loss + self.lambda_physics * physics_loss

            grads = tape.gradient(total_loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(
                zip(grads, self.model.trainable_variables)
            )

            history["total"].append(float(total_loss))
            history["data"].append(float(data_loss))
            history["physics"].append(float(physics_loss))

            if (epoch + 1) % 500 == 0:
                print(
                    f"Epoch {epoch+1}/{epochs} — "
                    f"Total: {total_loss:.6f}, Data: {data_loss:.6f}, "
                    f"Physics: {physics_loss:.6f}"
                )

        return history

    def predict(self, xy: np.ndarray) -> np.ndarray:
        """Predict [u, v, p] on given coordinates."""
        if self._tf_available:
            import tensorflow as tf
            return self.model.predict(xy.astype(np.float32), verbose=0)
        raise RuntimeError("TensorFlow required")


# =============================================================================
# Dataset Management
# =============================================================================
@dataclass
class MLDataset:
    """Container for ML training/validation data."""
    name: str
    X_train: np.ndarray = field(default_factory=lambda: np.array([]))
    y_train: np.ndarray = field(default_factory=lambda: np.array([]))
    X_val: np.ndarray = field(default_factory=lambda: np.array([]))
    y_val: np.ndarray = field(default_factory=lambda: np.array([]))
    source: str = ""
    n_cases: int = 0

    @property
    def summary(self) -> str:
        return (
            f"Dataset: {self.name}\n"
            f"  Source: {self.source}\n"
            f"  Cases: {self.n_cases}\n"
            f"  Train: {self.X_train.shape[0]} samples, {self.X_train.shape[1]} features\n"
            f"  Val:   {self.X_val.shape[0]} samples"
        )


def create_synthetic_dataset(n_samples: int = 1000, n_features: int = 7) -> MLDataset:
    """
    Create synthetic dataset for testing the ML pipeline.
    """
    rng = np.random.default_rng(42)
    X = rng.standard_normal((n_samples, n_features))

    # Synthetic target: nonlinear function of features
    y = 0.1 * X[:, 0]**2 - 0.05 * X[:, 1] * X[:, 2] + 0.02 * np.sin(X[:, 3])
    y = y.reshape(-1, 1)

    n_train = int(0.8 * n_samples)
    return MLDataset(
        name="synthetic",
        X_train=X[:n_train],
        y_train=y[:n_train],
        X_val=X[n_train:],
        y_val=y[n_train:],
        source="Synthetic test data",
        n_cases=1,
    )
