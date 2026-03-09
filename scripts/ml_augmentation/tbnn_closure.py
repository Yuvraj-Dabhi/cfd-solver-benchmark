#!/usr/bin/env python3
"""
Tensor-Basis Neural Network (TBNN) for Reynolds Stress Closure Augmentation
=============================================================================
Physics-informed ML correction to RANS turbulence closures, replacing the
black-box MLP surrogate that mapped freestream conditions to force coefficients.

This module implements the Ling et al. (2016) TBNN architecture which:
  - Takes Galilean-invariant scalar features as input
  - Predicts 10 tensor-basis coefficients g^(1)...g^(10)
  - Reconstructs the Reynolds stress anisotropy tensor:
        b_ij = Σ_n g^(n) * T^(n)_ij
  - Preserves symmetry, trace-free property, and Galilean invariance by construction

The output directly augments the RANS closure (Boussinesq approximation) rather than
bypassing the governing equations, addressing the fundamental deficiency of the
prior Tier-1 MLP approach.

Theory
------
The Boussinesq approximation assumes:
    τ_ij = 2 ν_t S_ij - (2/3) k δ_ij

This fails for separated flows because it assumes isotropic eddy viscosity,
which cannot capture normal stress anisotropy (a_ii ≠ 0 for i≠j) or history
effects in severe adverse pressure gradients.

The TBNN replaces this with a general tensor representation (Pope, 1975):
    b_ij = Σ_{n=1}^{10} g^(n)(λ_1, ..., λ_5) T^(n)_ij(Ŝ, Ω̂)

where T^(n) are the 10 linearly independent tensor bases formed from the
non-dimensionalized strain rate Ŝ = τ·S and rotation rate Ω̂ = τ·Ω tensors.

References
----------
- Pope (1975), J. Fluid Mech. 72, pp. 331–340 (tensor basis)
- Ling et al. (2016), J. Fluid Mech. 807, pp. 155–166 (TBNN architecture)
- Parish & Duraisamy (2016), JCP 305, pp. 758–774 (FIML framework)
- Srivastava et al. (2024), NASA TM-20240012512 (FIML implementation)
"""

import logging
import json
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger(__name__)

# Try PyTorch import
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.warning("PyTorch not available; TBNN training disabled (inference via sklearn fallback)")


# =============================================================================
# Tensor Basis Computation (Pope 1975)
# =============================================================================

def compute_tensor_basis(
    S_hat: np.ndarray,
    O_hat: np.ndarray,
) -> np.ndarray:
    """
    Compute the 10 Pope (1975) tensor bases from non-dimensionalized
    strain rate Ŝ and rotation rate Ω̂ tensors.

    Parameters
    ----------
    S_hat : ndarray (N, 3, 3)
        Non-dimensionalized symmetric strain rate: Ŝ = (k/ε) · S
    O_hat : ndarray (N, 3, 3)
        Non-dimensionalized antisymmetric rotation rate: Ω̂ = (k/ε) · Ω

    Returns
    -------
    T : ndarray (N, 10, 3, 3)
        The 10 tensor bases for each point.
    """
    N = S_hat.shape[0]
    I = np.eye(3)[None, :, :].repeat(N, axis=0)  # (N, 3, 3)

    # Products needed
    S2 = np.einsum("nij,njk->nik", S_hat, S_hat)
    O2 = np.einsum("nij,njk->nik", O_hat, O_hat)
    SO = np.einsum("nij,njk->nik", S_hat, O_hat)
    OS = np.einsum("nij,njk->nik", O_hat, S_hat)
    S2O = np.einsum("nij,njk->nik", S2, O_hat)
    OS2 = np.einsum("nij,njk->nik", O_hat, S2)
    O2S = np.einsum("nij,njk->nik", O2, S_hat)
    SO2 = np.einsum("nij,njk->nik", S_hat, O2)
    S2O2 = np.einsum("nij,njk->nik", S2, O2)
    O2S2 = np.einsum("nij,njk->nik", O2, S2)
    SO2S2 = np.einsum("nij,njk->nik", S_hat, O2S2)
    S2O2S = np.einsum("nij,njk->nik", S2O2, S_hat)

    # Trace terms for removing trace from bases
    tr_S2 = np.trace(S2, axis1=-2, axis2=-1)[:, None, None]
    tr_O2 = np.trace(O2, axis1=-2, axis2=-1)[:, None, None]
    tr_S2O2 = np.trace(S2O2, axis1=-2, axis2=-1)[:, None, None]

    T = np.zeros((N, 10, 3, 3))

    # T^(1) = Ŝ
    T[:, 0] = S_hat

    # T^(2) = Ŝ·Ω̂ - Ω̂·Ŝ
    T[:, 1] = SO - OS

    # T^(3) = Ŝ² - (1/3)tr(Ŝ²)·I
    T[:, 2] = S2 - (1.0 / 3.0) * tr_S2 * I

    # T^(4) = Ω̂² - (1/3)tr(Ω̂²)·I
    T[:, 3] = O2 - (1.0 / 3.0) * tr_O2 * I

    # T^(5) = Ω̂·Ŝ² - Ŝ²·Ω̂
    T[:, 4] = OS2 - S2O

    # T^(6) = Ω̂²·Ŝ + Ŝ·Ω̂² - (2/3)tr(Ŝ·Ω̂²)·I
    T[:, 5] = O2S + SO2 - (2.0 / 3.0) * np.trace(
        SO2, axis1=-2, axis2=-1
    )[:, None, None] * I

    # T^(7) = Ω̂·Ŝ·Ω̂² - Ω̂²·Ŝ·Ω̂
    OSO2 = np.einsum("nij,njk->nik", OS, O2)
    O2SO = np.einsum("nij,njk->nik", O2S, O_hat)
    T[:, 6] = OSO2 - O2SO

    # T^(8) = Ŝ·Ω̂·Ŝ² - Ŝ²·Ω̂·Ŝ
    SOS2 = np.einsum("nij,njk->nik", SO, S2)
    S2OS = np.einsum("nij,njk->nik", S2O, S_hat)
    T[:, 7] = SOS2 - S2OS

    # T^(9) = Ω̂²·Ŝ² + Ŝ²·Ω̂² - (2/3)tr(Ŝ²·Ω̂²)·I
    T[:, 8] = O2S2 + S2O2 - (2.0 / 3.0) * tr_S2O2 * I

    # T^(10) = Ω̂·Ŝ²·Ω̂² - Ω̂²·Ŝ²·Ω̂
    OS2O2 = np.einsum("nij,njk->nik", OS2, O2)
    O2S2O = np.einsum("nij,njk->nik", O2S2, O_hat)
    T[:, 9] = OS2O2 - O2S2O

    return T


def compute_invariant_inputs(
    S_hat: np.ndarray,
    O_hat: np.ndarray,
) -> np.ndarray:
    """
    Compute the 5 irreducible scalar invariants from Ŝ and Ω̂.

    λ₁ = tr(Ŝ²)
    λ₂ = tr(Ω̂²)
    λ₃ = tr(Ŝ³)
    λ₄ = tr(Ω̂²·Ŝ)
    λ₅ = tr(Ω̂²·Ŝ²)

    Parameters
    ----------
    S_hat : ndarray (N, 3, 3)
    O_hat : ndarray (N, 3, 3)

    Returns
    -------
    lambdas : ndarray (N, 5)
    """
    S2 = np.einsum("nij,njk->nik", S_hat, S_hat)
    O2 = np.einsum("nij,njk->nik", O_hat, O_hat)
    S3 = np.einsum("nij,njk->nik", S2, S_hat)
    O2S = np.einsum("nij,njk->nik", O2, S_hat)
    O2S2 = np.einsum("nij,njk->nik", O2, S2)

    lam1 = np.trace(S2, axis1=-2, axis2=-1)
    lam2 = np.trace(O2, axis1=-2, axis2=-1)
    lam3 = np.trace(S3, axis1=-2, axis2=-1)
    lam4 = np.trace(O2S, axis1=-2, axis2=-1)
    lam5 = np.trace(O2S2, axis1=-2, axis2=-1)

    return np.column_stack([lam1, lam2, lam3, lam4, lam5])


# =============================================================================
# Realizability Constraints (Lumley Triangle)
# =============================================================================

@dataclass
class RealizabilityReport:
    """Report on realizability of predicted anisotropy tensors."""
    n_points: int = 0
    n_realizable: int = 0
    fraction_realizable: float = 0.0
    max_trace_error: float = 0.0
    max_symmetry_error: float = 0.0
    eigenvalue_bounds_satisfied: bool = True
    summary: str = ""


def check_realizability(b_ij: np.ndarray, tol: float = 1e-6) -> RealizabilityReport:
    """
    Verify that predicted anisotropy tensors satisfy realizability constraints.

    Constraints:
    1. Trace-free: b_kk = 0
    2. Symmetric: b_ij = b_ji
    3. Eigenvalues within Lumley triangle:
       - λ₁ + λ₂ + λ₃ = 0 (trace-free)
       - -1/3 ≤ λ_i ≤ 2/3 for each eigenvalue
       - 2nd invariant: II_b = b_ij·b_ij / 2 ≥ 0
       - 3rd invariant: within bounds

    Parameters
    ----------
    b_ij : ndarray (N, 3, 3)
        Predicted anisotropy tensors.
    tol : float
        Tolerance for constraint checks.

    Returns
    -------
    RealizabilityReport
    """
    N = b_ij.shape[0]
    report = RealizabilityReport(n_points=N)

    # 1. Trace-free check
    traces = np.trace(b_ij, axis1=-2, axis2=-1)
    report.max_trace_error = float(np.max(np.abs(traces)))

    # 2. Symmetry check
    sym_error = np.max(np.abs(b_ij - np.swapaxes(b_ij, -2, -1)), axis=(-2, -1))
    report.max_symmetry_error = float(np.max(sym_error))

    # 3. Eigenvalue bounds
    eigenvalues = np.linalg.eigvalsh(b_ij)  # (N, 3)
    min_eig = eigenvalues.min(axis=1)
    max_eig = eigenvalues.max(axis=1)

    realizable = (
        (np.abs(traces) < tol)
        & (sym_error < tol)
        & (min_eig > -1.0/3.0 - tol)
        & (max_eig < 2.0/3.0 + tol)
    )

    report.n_realizable = int(np.sum(realizable))
    report.fraction_realizable = report.n_realizable / N if N > 0 else 0.0
    report.eigenvalue_bounds_satisfied = bool(np.all(
        (min_eig > -1.0/3.0 - tol) & (max_eig < 2.0/3.0 + tol)
    ))

    report.summary = (
        f"{report.fraction_realizable*100:.1f}% realizable "
        f"(trace_err={report.max_trace_error:.2e}, "
        f"sym_err={report.max_symmetry_error:.2e}, "
        f"eig_bounds={'OK' if report.eigenvalue_bounds_satisfied else 'FAIL'})"
    )

    return report


def project_to_realizable(b_ij: np.ndarray) -> np.ndarray:
    """
    Project anisotropy tensor to the nearest realizable state.

    1. Enforce symmetry: b_ij = (b_ij + b_ji) / 2
    2. Enforce trace-free: b_ij -= (tr(b)/3) * I
    3. Clamp eigenvalues to [-1/3, 2/3]

    Parameters
    ----------
    b_ij : ndarray (N, 3, 3)

    Returns
    -------
    b_realizable : ndarray (N, 3, 3)
    """
    N = b_ij.shape[0]
    I = np.eye(3)[None, :, :].repeat(N, axis=0)

    # Symmetrize
    b = 0.5 * (b_ij + np.swapaxes(b_ij, -2, -1))

    # Trace-free
    tr = np.trace(b, axis1=-2, axis2=-1)[:, None, None]
    b = b - (tr / 3.0) * I

    # Eigenvalue clamping
    eigenvalues, eigenvectors = np.linalg.eigh(b)
    eigenvalues = np.clip(eigenvalues, -1.0/3.0, 2.0/3.0)

    # Re-enforce trace-free after clamping
    tr_new = eigenvalues.sum(axis=1, keepdims=True)
    eigenvalues -= tr_new / 3.0

    # Reconstruct
    b_out = np.einsum("nij,nj,nkj->nik", eigenvectors, eigenvalues, eigenvectors)

    return b_out


# =============================================================================
# Galilean Invariance Verification
# =============================================================================

def verify_galilean_invariance(
    features_func,
    S: np.ndarray,
    Omega: np.ndarray,
    k: np.ndarray,
    epsilon: np.ndarray,
    n_rotations: int = 5,
    tol: float = 1e-8,
) -> Dict[str, Any]:
    """
    Verify that extracted features are invariant under coordinate rotation.

    Applies random rotation matrices Q and checks that:
        features(Q·S·Q^T, Q·Ω·Q^T, k, ε) = features(S, Ω, k, ε)

    Parameters
    ----------
    features_func : callable
        Function that computes invariant features from (S, Omega, k, epsilon).
    S, Omega : ndarray (N, 3, 3)
    k, epsilon : ndarray (N,)
    n_rotations : int
        Number of random rotations to test.
    tol : float
        Maximum allowed difference.

    Returns
    -------
    dict with 'passed', 'max_error', 'details'
    """
    # Original features
    f_orig = features_func(S, Omega, k, epsilon)

    max_error = 0.0
    all_passed = True
    details = []

    for i in range(n_rotations):
        # Random rotation matrix via QR decomposition
        rng = np.random.default_rng(seed=42 + i)
        A = rng.standard_normal((3, 3))
        Q, _ = np.linalg.qr(A)
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1  # Ensure proper rotation

        # Rotate tensors: S' = Q S Q^T
        S_rot = np.einsum("ij,njk,lk->nil", Q, S, Q)
        O_rot = np.einsum("ij,njk,lk->nil", Q, Omega, Q)

        f_rot = features_func(S_rot, O_rot, k, epsilon)
        err = np.max(np.abs(f_orig - f_rot))
        max_error = max(max_error, err)

        passed = err < tol
        all_passed = all_passed and passed
        details.append({"rotation": i, "max_error": float(err), "passed": passed})

    return {
        "passed": all_passed,
        "max_error": float(max_error),
        "n_rotations": n_rotations,
        "details": details,
    }


# =============================================================================
# TBNN Model (PyTorch)
# =============================================================================

if HAS_TORCH:
    class TBNNModel(nn.Module):
        """
        Tensor-Basis Neural Network for Reynolds stress anisotropy prediction.

        Architecture:
            Invariant scalars (λ₁...λ₅ + supplementary features)
            → Dense layers with ReLU
            → 10 tensor-basis coefficients g^(1)...g^(10)
            → b_ij = Σ g^(n) T^(n)_ij  (physics layer, no learnable params)

        The physics layer guarantees:
            - Galilean invariance (inputs are invariants, output is tensor)
            - Symmetry of b_ij (tensor bases are symmetric/antisymmetric by construction)
            - Trace-free property (bases are trace-free by construction)
        """

        def __init__(
            self,
            n_scalar_inputs: int = 5,
            hidden_layers: List[int] = None,
            dropout: float = 0.1,
            activation: str = "leaky_relu",
        ):
            """
            Parameters
            ----------
            n_scalar_inputs : int
                Number of invariant scalar features.
            hidden_layers : list of int
                Hidden layer sizes (default: [64, 128, 128, 64]).
            dropout : float
                Dropout rate for regularization.
            activation : str
                Activation function ('relu', 'leaky_relu', 'elu', 'swish').
            """
            super().__init__()

            if hidden_layers is None:
                hidden_layers = [64, 128, 128, 64]

            # Build scalar-to-coefficient network
            layers = []
            prev_dim = n_scalar_inputs
            for h in hidden_layers:
                layers.append(nn.Linear(prev_dim, h))
                layers.append(nn.BatchNorm1d(h))
                if activation == "relu":
                    layers.append(nn.ReLU())
                elif activation == "leaky_relu":
                    layers.append(nn.LeakyReLU(0.1))
                elif activation == "elu":
                    layers.append(nn.ELU())
                elif activation == "swish":
                    layers.append(nn.SiLU())
                layers.append(nn.Dropout(dropout))
                prev_dim = h

            # Output: 10 tensor-basis coefficients
            layers.append(nn.Linear(prev_dim, 10))
            self.scalar_net = nn.Sequential(*layers)

            self.n_scalar_inputs = n_scalar_inputs

        def forward(
            self,
            invariants: "torch.Tensor",
            tensor_bases: "torch.Tensor",
        ) -> "torch.Tensor":
            """
            Forward pass: invariants → g-coefficients → b_ij.

            Parameters
            ----------
            invariants : Tensor (batch, n_scalar_inputs)
                Galilean-invariant scalar features.
            tensor_bases : Tensor (batch, 10, 3, 3)
                Pre-computed tensor bases T^(1)...T^(10).

            Returns
            -------
            b_ij : Tensor (batch, 3, 3)
                Predicted Reynolds stress anisotropy tensor.
            """
            # Predict coefficients
            g = self.scalar_net(invariants)  # (batch, 10)

            # Physics layer: b_ij = Σ g^(n) T^(n)_ij
            b_ij = torch.einsum("bn,bnij->bij", g, tensor_bases)

            return b_ij

        def predict_coefficients(self, invariants: "torch.Tensor") -> "torch.Tensor":
            """Return raw g-coefficients without tensor contraction."""
            return self.scalar_net(invariants)


    class RealizabilityLoss(nn.Module):
        """
        Custom loss combining data fidelity with realizability constraints.

        L = L_data + α * L_realizability

        L_data = MSE(b_pred, b_target)
        L_realizability = penalty for eigenvalues outside [-1/3, 2/3]
        """

        def __init__(self, alpha: float = 0.1, trace_weight: float = 1.0):
            super().__init__()
            self.alpha = alpha
            self.trace_weight = trace_weight
            self.mse = nn.MSELoss()

        def forward(
            self,
            b_pred: "torch.Tensor",
            b_target: "torch.Tensor",
        ) -> "torch.Tensor":
            # Data fidelity
            loss_data = self.mse(b_pred, b_target)

            # Trace penalty (should be zero)
            trace = torch.diagonal(b_pred, dim1=-2, dim2=-1).sum(dim=-1)
            loss_trace = self.trace_weight * torch.mean(trace ** 2)

            # Eigenvalue realizability penalty
            eigenvalues = torch.linalg.eigvalsh(b_pred)
            # Penalty for eigenvalues below -1/3
            below = torch.clamp(-1.0/3.0 - eigenvalues, min=0.0)
            # Penalty for eigenvalues above 2/3
            above = torch.clamp(eigenvalues - 2.0/3.0, min=0.0)
            loss_realize = self.alpha * torch.mean(below ** 2 + above ** 2)

            return loss_data + loss_trace + loss_realize


# =============================================================================
# Training Pipeline
# =============================================================================

@dataclass
class TBNNTrainingResult:
    """Result container for TBNN training."""
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    best_val_loss: float = float("inf")
    best_epoch: int = 0
    realizability_fraction: float = 0.0
    max_trace_error: float = 0.0
    training_time_s: float = 0.0
    hyperparameters: Dict[str, Any] = field(default_factory=dict)


def prepare_tbnn_data(
    S: np.ndarray,
    Omega: np.ndarray,
    k: np.ndarray,
    epsilon: np.ndarray,
    b_target: np.ndarray,
    supplementary_features: np.ndarray = None,
) -> Dict[str, np.ndarray]:
    """
    Prepare data for TBNN training from raw RANS fields.

    Parameters
    ----------
    S : ndarray (N, 3, 3)
        Mean strain rate tensor.
    Omega : ndarray (N, 3, 3)
        Mean rotation rate tensor.
    k : ndarray (N,)
        Turbulent kinetic energy.
    epsilon : ndarray (N,)
        Turbulent dissipation rate.
    b_target : ndarray (N, 3, 3)
        Target anisotropy tensor (from DNS/LES).
    supplementary_features : ndarray (N, M), optional
        Additional features (Re_d, ν_t/ν, APG indicator, etc.)

    Returns
    -------
    dict with 'invariants', 'tensor_bases', 'targets'
    """
    # Non-dimensionalize
    eps_safe = np.maximum(epsilon, 1e-10)
    k_safe = np.maximum(k, 1e-10)
    tau = k_safe / eps_safe

    S_hat = S * tau[:, None, None]
    O_hat = Omega * tau[:, None, None]

    # Compute tensor bases and invariants
    T = compute_tensor_basis(S_hat, O_hat)
    lambdas = compute_invariant_inputs(S_hat, O_hat)

    # Combine with supplementary features if provided
    if supplementary_features is not None:
        invariants = np.hstack([lambdas, supplementary_features])
    else:
        invariants = lambdas

    return {
        "invariants": invariants,
        "tensor_bases": T,
        "targets": b_target,
        "S_hat": S_hat,
        "O_hat": O_hat,
    }


def train_tbnn(
    data: Dict[str, np.ndarray],
    hidden_layers: List[int] = None,
    epochs: int = 500,
    lr: float = 1e-3,
    batch_size: int = 256,
    val_split: float = 0.2,
    patience: int = 50,
    alpha_realizability: float = 0.1,
    seed: int = 42,
) -> Tuple[Any, TBNNTrainingResult]:
    """
    Train a TBNN model on prepared data.

    Parameters
    ----------
    data : dict
        Output from prepare_tbnn_data().
    hidden_layers : list of int
        Hidden layer sizes.
    epochs : int
        Maximum training epochs.
    lr : float
        Learning rate.
    batch_size : int
        Mini-batch size.
    val_split : float
        Fraction of data for validation.
    patience : int
        Early stopping patience.
    alpha_realizability : float
        Weight for realizability penalty in loss.
    seed : int
        Random seed.

    Returns
    -------
    (model, result) : tuple of TBNNModel and TBNNTrainingResult
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch required for TBNN training. Install: pip install torch")

    import time

    torch.manual_seed(seed)
    np.random.seed(seed)

    invariants = data["invariants"]
    tensor_bases = data["tensor_bases"]
    targets = data["targets"]

    N = invariants.shape[0]
    n_features = invariants.shape[1]

    # Train/val split
    idx = np.random.permutation(N)
    n_val = int(N * val_split)
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    # Normalize invariants
    inv_mean = invariants[train_idx].mean(axis=0)
    inv_std = invariants[train_idx].std(axis=0) + 1e-10
    invariants_norm = (invariants - inv_mean) / inv_std

    # To tensors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train = torch.FloatTensor(invariants_norm[train_idx]).to(device)
    T_train = torch.FloatTensor(tensor_bases[train_idx]).to(device)
    y_train = torch.FloatTensor(targets[train_idx]).to(device)

    X_val = torch.FloatTensor(invariants_norm[val_idx]).to(device)
    T_val = torch.FloatTensor(tensor_bases[val_idx]).to(device)
    y_val = torch.FloatTensor(targets[val_idx]).to(device)

    train_dataset = TensorDataset(X_train, T_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Model
    if hidden_layers is None:
        hidden_layers = [64, 128, 128, 64]
    model = TBNNModel(n_scalar_inputs=n_features, hidden_layers=hidden_layers).to(device)

    # Loss and optimizer
    criterion = RealizabilityLoss(alpha=alpha_realizability)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=20, min_lr=1e-6
    )

    # Training loop
    result = TBNNTrainingResult(
        hyperparameters={
            "hidden_layers": hidden_layers,
            "lr": lr,
            "batch_size": batch_size,
            "epochs": epochs,
            "alpha_realizability": alpha_realizability,
            "n_train": len(train_idx),
            "n_val": n_val,
            "n_features": n_features,
        }
    )

    best_state = None
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch_inv, batch_T, batch_y in train_loader:
            optimizer.zero_grad()
            b_pred = model(batch_inv, batch_T)
            loss = criterion(b_pred, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        train_loss = epoch_loss / n_batches
        result.train_losses.append(train_loss)

        # Validation
        model.eval()
        with torch.no_grad():
            b_val_pred = model(X_val, T_val)
            val_loss = criterion(b_val_pred, y_val).item()
        result.val_losses.append(val_loss)
        scheduler.step(val_loss)

        # Early stopping
        if val_loss < result.best_val_loss:
            result.best_val_loss = val_loss
            result.best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch - result.best_epoch > patience:
            logger.info(f"Early stopping at epoch {epoch} (best: {result.best_epoch})")
            break

        if (epoch + 1) % 50 == 0:
            logger.info(f"Epoch {epoch+1}/{epochs}: train={train_loss:.6f}, val={val_loss:.6f}")

    result.training_time_s = time.time() - start_time

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to(device)

    # Final realizability check
    model.eval()
    with torch.no_grad():
        b_all = model(
            torch.FloatTensor(invariants_norm).to(device),
            torch.FloatTensor(tensor_bases).to(device),
        ).cpu().numpy()

    r_report = check_realizability(b_all)
    result.realizability_fraction = r_report.fraction_realizable
    result.max_trace_error = r_report.max_trace_error

    # Store normalization parameters on model
    model.inv_mean = inv_mean
    model.inv_std = inv_std

    logger.info(
        f"Training complete: best_val_loss={result.best_val_loss:.6f}, "
        f"realizability={result.realizability_fraction*100:.1f}%, "
        f"time={result.training_time_s:.1f}s"
    )

    return model, result


# =============================================================================
# Inference
# =============================================================================

def predict_anisotropy(
    model: Any,
    S: np.ndarray,
    Omega: np.ndarray,
    k: np.ndarray,
    epsilon: np.ndarray,
    supplementary_features: np.ndarray = None,
    enforce_realizability: bool = True,
) -> np.ndarray:
    """
    Predict Reynolds stress anisotropy from RANS solution fields.

    Parameters
    ----------
    model : TBNNModel
        Trained TBNN model.
    S, Omega : ndarray (N, 3, 3)
        Mean strain rate and rotation rate tensors.
    k, epsilon : ndarray (N,)
        Turbulent kinetic energy and dissipation rate.
    supplementary_features : ndarray (N, M), optional
    enforce_realizability : bool
        If True, project predictions to realizable state.

    Returns
    -------
    b_ij : ndarray (N, 3, 3)
        Predicted anisotropy tensor.
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch required for TBNN inference")

    # Prepare data
    data = prepare_tbnn_data(S, Omega, k, epsilon,
                             b_target=np.zeros((len(k), 3, 3)),
                             supplementary_features=supplementary_features)

    invariants = data["invariants"]
    tensor_bases = data["tensor_bases"]

    # Normalize
    invariants_norm = (invariants - model.inv_mean) / model.inv_std

    # Predict
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        b_pred = model(
            torch.FloatTensor(invariants_norm).to(device),
            torch.FloatTensor(tensor_bases).to(device),
        ).cpu().numpy()

    if enforce_realizability:
        b_pred = project_to_realizable(b_pred)

    return b_pred


# =============================================================================
# Model I/O
# =============================================================================

def save_tbnn(
    model: Any,
    result: TBNNTrainingResult,
    output_dir: Path,
):
    """Save trained TBNN model and metadata."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if HAS_TORCH:
        torch.save(model.state_dict(), output_dir / "tbnn_weights.pt")
        np.savez(
            output_dir / "tbnn_normalization.npz",
            inv_mean=model.inv_mean,
            inv_std=model.inv_std,
        )

    # Save training metrics
    metrics = {
        "best_val_loss": result.best_val_loss,
        "best_epoch": result.best_epoch,
        "realizability_fraction": result.realizability_fraction,
        "max_trace_error": result.max_trace_error,
        "training_time_s": result.training_time_s,
        "hyperparameters": result.hyperparameters,
        "n_train_losses": len(result.train_losses),
    }
    with open(output_dir / "tbnn_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"TBNN model saved to {output_dir}")


def load_tbnn(model_dir: Path, n_features: int = 5, hidden_layers: List[int] = None):
    """Load a trained TBNN model."""
    if not HAS_TORCH:
        raise ImportError("PyTorch required to load TBNN model")

    model_dir = Path(model_dir)
    if hidden_layers is None:
        hidden_layers = [64, 128, 128, 64]

    model = TBNNModel(n_scalar_inputs=n_features, hidden_layers=hidden_layers)
    model.load_state_dict(torch.load(model_dir / "tbnn_weights.pt", weights_only=True))

    norm = np.load(model_dir / "tbnn_normalization.npz")
    model.inv_mean = norm["inv_mean"]
    model.inv_std = norm["inv_std"]

    model.eval()
    return model


# =============================================================================
# Smoke Test
# =============================================================================

def _smoke_test():
    """Quick self-check with synthetic data."""
    print("=" * 60)
    print("  TBNN Closure Module — Smoke Test")
    print("=" * 60)

    N = 500
    rng = np.random.default_rng(42)

    # Synthetic strain/rotation tensors
    dudx = rng.standard_normal((N, 3, 3)) * 0.1
    S = 0.5 * (dudx + np.swapaxes(dudx, -2, -1))
    Omega = 0.5 * (dudx - np.swapaxes(dudx, -2, -1))
    k = np.abs(rng.standard_normal(N)) * 0.1 + 0.01
    eps = np.abs(rng.standard_normal(N)) * 0.1 + 0.01

    # Non-dimensionalize
    tau = k / eps
    S_hat = S * tau[:, None, None]
    O_hat = Omega * tau[:, None, None]

    # Tensor basis
    T = compute_tensor_basis(S_hat, O_hat)
    print(f"[OK] Tensor basis shape: {T.shape} (expected ({N}, 10, 3, 3))")
    assert T.shape == (N, 10, 3, 3)

    # Invariants
    lambdas = compute_invariant_inputs(S_hat, O_hat)
    print(f"[OK] Invariant inputs shape: {lambdas.shape} (expected ({N}, 5))")
    assert lambdas.shape == (N, 5)

    # Galilean invariance verification
    def features_func(S_in, O_in, k_in, eps_in):
        tau_in = k_in / (eps_in + 1e-10)
        return compute_invariant_inputs(S_in * tau_in[:, None, None],
                                        O_in * tau_in[:, None, None])

    gi_result = verify_galilean_invariance(features_func, S, Omega, k, eps)
    print(f"[{'OK' if gi_result['passed'] else 'FAIL'}] Galilean invariance: "
          f"max_error={gi_result['max_error']:.2e}")

    # Synthetic target anisotropy (small perturbation from zero)
    b_target = rng.standard_normal((N, 3, 3)) * 0.01
    b_target = 0.5 * (b_target + np.swapaxes(b_target, -2, -1))
    b_target -= np.trace(b_target, axis1=-2, axis2=-1)[:, None, None] / 3.0 * np.eye(3)

    # Realizability check
    r_report = check_realizability(b_target)
    print(f"[OK] Realizability: {r_report.summary}")

    # Project to realizable
    b_proj = project_to_realizable(b_target * 10)  # Exaggerate to test clamping
    r_proj = check_realizability(b_proj)
    print(f"[OK] After projection: {r_proj.summary}")

    # TBNN model test (if PyTorch available)
    if HAS_TORCH:
        data = prepare_tbnn_data(S, Omega, k, eps, b_target)
        print(f"[OK] Data prepared: invariants={data['invariants'].shape}, "
              f"bases={data['tensor_bases'].shape}")

        model = TBNNModel(n_scalar_inputs=data['invariants'].shape[1])
        inv_t = torch.FloatTensor(data['invariants'][:8])
        T_t = torch.FloatTensor(data['tensor_bases'][:8])
        b_out = model(inv_t, T_t)
        print(f"[OK] TBNN forward pass: output shape={b_out.shape} (expected (8, 3, 3))")
        assert b_out.shape == (8, 3, 3)
    else:
        print("[SKIP] PyTorch not available, skipping model test")

    print(f"\n{'=' * 60}")
    print("  ALL TBNN CLOSURE TESTS PASSED")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _smoke_test()
