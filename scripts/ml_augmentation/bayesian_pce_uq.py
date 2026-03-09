#!/usr/bin/env python3
"""
Bayesian PCE Uncertainty Quantification
==========================================
Polynomial Chaos Expansion surrogate-accelerated Bayesian inversion
for localized eigenspace perturbations in RANS turbulence models.

Key features:
  - PCESurrogate: orthogonal polynomial expansion of stochastic outputs
  - BayesianInverter: MCMC sampling with PCE-accelerated likelihood
  - LocalizedPerturbationPredictor: RF-based local perturbation magnitude
  - EigenspacePerturbation: Reynolds stress eigenvalue/eigenvector perturbation
  - BarycentricMapper: anisotropy mapping to barycentric triangle

Architecture reference:
  - Xiu & Karniadakis (2002): Polynomial Chaos for stochastic systems
  - Edeling et al. (2014): Bayesian RANS calibration
  - Emory et al. (2013): Eigenspace perturbation framework
  - Iaccarino et al. (2017): EQUiPS module for SU2

Usage:
    from scripts.ml_augmentation.bayesian_pce_uq import (
        BayesianPCEFramework, PCESurrogate, BarycentricMapper,
    )
    framework = BayesianPCEFramework(n_params=5)
    results = framework.run_calibration(model_func, obs_data)
"""

import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)
PROJECT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT))


# =============================================================================
# Polynomial Chaos Expansion
# =============================================================================
class PCESurrogate:
    """
    Polynomial Chaos Expansion surrogate model.

    Constructs orthogonal polynomial representation of stochastic
    CFD outputs as a function of uncertain input parameters.

    Uses Legendre polynomials for uniform distributions and
    non-intrusive regression for coefficient estimation.

    Parameters
    ----------
    n_params : int
        Number of uncertain parameters.
    max_order : int
        Maximum polynomial order.
    """

    def __init__(self, n_params: int = 5, max_order: int = 3):
        self.n_params = n_params
        self.max_order = max_order
        self.coefficients = None
        self.multi_indices = None
        self._fitted = False

        # Generate multi-indices for total-order expansion
        self.multi_indices = self._generate_multi_indices()
        self.n_terms = len(self.multi_indices)

    def _generate_multi_indices(self) -> np.ndarray:
        """Generate multi-indices for total-order PCE."""
        indices = []
        self._generate_indices_recursive(
            [], self.n_params, self.max_order, indices)
        return np.array(indices)

    def _generate_indices_recursive(self, current, n_remaining,
                                     order_remaining, result):
        if n_remaining == 0:
            result.append(current)
            return
        for i in range(order_remaining + 1):
            self._generate_indices_recursive(
                current + [i], n_remaining - 1,
                order_remaining - i, result)

    def _eval_legendre(self, x: np.ndarray, order: int) -> np.ndarray:
        """Evaluate Legendre polynomial of given order at x in [-1,1]."""
        if order == 0:
            return np.ones_like(x)
        elif order == 1:
            return x
        elif order == 2:
            return 0.5 * (3 * x ** 2 - 1)
        elif order == 3:
            return 0.5 * (5 * x ** 3 - 3 * x)
        else:
            # Recurrence relation
            p_prev = np.ones_like(x)
            p_curr = x.copy()
            for n in range(2, order + 1):
                p_next = ((2 * n - 1) * x * p_curr - (n - 1) * p_prev) / n
                p_prev = p_curr
                p_curr = p_next
            return p_curr

    def _eval_basis(self, samples: np.ndarray) -> np.ndarray:
        """
        Evaluate PCE basis functions at sample points.

        Parameters
        ----------
        samples : ndarray (n_samples, n_params)
            Parameter samples in [-1, 1].

        Returns
        -------
        Ψ : ndarray (n_samples, n_terms)
        """
        n_samples = len(samples)
        psi = np.ones((n_samples, self.n_terms))

        for j, multi_idx in enumerate(self.multi_indices):
            for k in range(self.n_params):
                if multi_idx[k] > 0:
                    psi[:, j] *= self._eval_legendre(
                        samples[:, k], multi_idx[k])
        return psi

    def fit(self, samples: np.ndarray, outputs: np.ndarray):
        """
        Fit PCE coefficients via least-squares regression.

        Parameters
        ----------
        samples : ndarray (n_samples, n_params)
            Parameter samples (normalized to [-1, 1]).
        outputs : ndarray (n_samples,) or (n_samples, n_qoi)
            Model outputs at sample points.
        """
        psi = self._eval_basis(samples)

        if outputs.ndim == 1:
            outputs = outputs[:, np.newaxis]

        # Least-squares: Ψ c = y
        self.coefficients, _, _, _ = np.linalg.lstsq(
            psi, outputs, rcond=None)
        self._fitted = True

    def predict(self, samples: np.ndarray) -> np.ndarray:
        """
        Predict outputs using PCE surrogate.

        Parameters
        ----------
        samples : ndarray (n_samples, n_params)

        Returns
        -------
        predictions : ndarray (n_samples, n_qoi)
        """
        if not self._fitted:
            raise RuntimeError("PCE not fitted")
        psi = self._eval_basis(samples)
        return psi @ self.coefficients

    def compute_statistics(self) -> Dict[str, np.ndarray]:
        """
        Compute mean and variance from PCE coefficients.

        The mean is the zeroth coefficient; variance is the sum
        of squared higher-order coefficients.
        """
        if not self._fitted:
            raise RuntimeError("PCE not fitted")

        mean = self.coefficients[0]  # Zeroth-order
        variance = np.sum(self.coefficients[1:] ** 2, axis=0)

        return {
            "mean": mean,
            "variance": variance,
            "std": np.sqrt(variance),
        }

    def sobol_indices(self) -> Dict[str, np.ndarray]:
        """
        Compute first-order Sobol sensitivity indices from PCE.

        Uses the analytical relationship between PCE coefficients
        and variance decomposition.
        """
        if not self._fitted:
            raise RuntimeError("PCE not fitted")

        total_var = np.sum(self.coefficients[1:] ** 2, axis=0)
        if np.any(total_var < 1e-15):
            total_var = np.maximum(total_var, 1e-15)

        s1 = np.zeros((self.n_params, self.coefficients.shape[1]))
        for j, multi_idx in enumerate(self.multi_indices):
            if j == 0:
                continue
            # Find which single parameter this index depends on
            active = np.where(np.array(multi_idx) > 0)[0]
            if len(active) == 1:
                s1[active[0]] += self.coefficients[j] ** 2 / total_var
        return {"S1": s1, "total_variance": total_var}


# =============================================================================
# Barycentric Mapper
# =============================================================================
class BarycentricMapper:
    """
    Maps Reynolds stress anisotropy to the barycentric triangle
    (Banerjee et al. 2007).

    The three vertices of the triangle represent:
      - x1c: one-component turbulence (1C)
      - x2c: two-component turbulence (2C)
      - x3c: three-component isotropic turbulence (3C)

    Parameters
    ----------
    None (static mapping utility).
    """

    # Triangle vertices
    X1C = np.array([1.0, 0.0])  # One-component
    X2C = np.array([0.0, 0.0])  # Two-component axisymmetric
    X3C = np.array([0.5, np.sqrt(3) / 2])  # Isotropic

    def anisotropy_to_barycentric(self, anisotropy: np.ndarray) -> np.ndarray:
        """
        Map anisotropy tensor eigenvalues to barycentric coordinates.

        Parameters
        ----------
        anisotropy : ndarray (n, 3, 3) or (3, 3)
            Normalized Reynolds stress anisotropy tensor:
            b_ij = <u_i u_j> / (2k) - δ_ij / 3

        Returns
        -------
        coords : ndarray (n, 2) or (2,)
            Barycentric triangle coordinates.
        """
        single = anisotropy.ndim == 2
        if single:
            anisotropy = anisotropy[np.newaxis, :]

        n = len(anisotropy)
        coords = np.zeros((n, 2))

        for i in range(n):
            eigvals = np.sort(np.linalg.eigvalsh(anisotropy[i]))[::-1]
            λ1, λ2, λ3 = eigvals

            # Barycentric weights
            c1c = λ1 - λ2  # One-component weight
            c2c = 2 * (λ2 - λ3)  # Two-component weight
            c3c = 3 * λ3 + 1  # Three-component weight

            # Normalize
            total = abs(c1c) + abs(c2c) + abs(c3c)
            if total > 1e-15:
                c1c /= total
                c2c /= total
                c3c /= total
            else:
                c3c = 1.0  # Default to isotropic

            coords[i] = c1c * self.X1C + c2c * self.X2C + c3c * self.X3C

        return coords[0] if single else coords

    def is_inside_triangle(self, coords: np.ndarray) -> np.ndarray:
        """Check if coordinates lie within the realizability triangle."""
        single = coords.ndim == 1
        if single:
            coords = coords[np.newaxis, :]

        # Barycentric test
        v0 = self.X3C - self.X2C
        v1 = self.X1C - self.X2C
        v2 = coords - self.X2C[np.newaxis, :]

        dot00 = np.dot(v0, v0)
        dot01 = np.dot(v0, v1)
        dot11 = np.dot(v1, v1)
        dot02 = v2 @ v0
        dot12 = v2 @ v1

        inv_denom = 1.0 / (dot00 * dot11 - dot01 ** 2 + 1e-15)
        u = (dot11 * dot02 - dot01 * dot12) * inv_denom
        v = (dot00 * dot12 - dot01 * dot02) * inv_denom

        inside = (u >= -1e-10) & (v >= -1e-10) & (u + v <= 1 + 1e-10)
        return inside[0] if single else inside


# =============================================================================
# Eigenspace Perturbation
# =============================================================================
class EigenspacePerturbation:
    """
    Reynolds stress eigenspace perturbation (Emory et al. 2013).

    Perturbs the Reynolds stress tensor by modifying:
      - Eigenvalues (shape/componentiality): shift toward 1C/2C/3C states
      - Eigenvectors (alignment): rotate toward mean strain eigenvectors
      - Magnitude (TKE): scale turbulent kinetic energy

    Used by the EQUiPS module in SU2 for epistemic UQ.

    Parameters
    ----------
    delta_b : float
        Eigenvalue perturbation magnitude [0, 1].
    target_state : str
        Target limiting state: '1c', '2c', '3c'.
    """

    def __init__(self, delta_b: float = 1.0, target_state: str = "3c"):
        self.delta_b = np.clip(delta_b, 0.0, 1.0)
        self.target_state = target_state

    def perturb(self, reynolds_stress: np.ndarray) -> np.ndarray:
        """
        Apply eigenspace perturbation to Reynolds stress tensor.

        Parameters
        ----------
        reynolds_stress : ndarray (3, 3) or (n, 3, 3)
            Reynolds stress tensor R_ij = <u_i u_j>.

        Returns
        -------
        perturbed : ndarray — same shape as input.
        """
        single = reynolds_stress.ndim == 2
        if single:
            reynolds_stress = reynolds_stress[np.newaxis, :]

        n = len(reynolds_stress)
        result = np.zeros_like(reynolds_stress)

        for i in range(n):
            R = reynolds_stress[i]
            k = 0.5 * np.trace(R)  # TKE
            if k < 1e-15:
                result[i] = R
                continue

            # Normalized anisotropy
            b = R / (2 * k) - np.eye(3) / 3

            # Eigendecomposition
            eigvals, eigvecs = np.linalg.eigh(b)
            # Sort descending and reorder eigenvectors to match
            sort_idx = np.argsort(eigvals)[::-1]
            eigvals = eigvals[sort_idx]
            eigvecs = eigvecs[:, sort_idx]

            # Target eigenvalues
            if self.target_state == "1c":
                target_eigvals = np.array([2 / 3, -1 / 3, -1 / 3])
            elif self.target_state == "2c":
                target_eigvals = np.array([1 / 6, 1 / 6, -1 / 3])
            else:  # 3c (isotropic)
                target_eigvals = np.array([0.0, 0.0, 0.0])

            # Interpolate eigenvalues
            perturbed_eigvals = eigvals + self.delta_b * (target_eigvals - eigvals)

            # Reconstruct
            b_perturbed = eigvecs @ np.diag(perturbed_eigvals) @ eigvecs.T
            result[i] = 2 * k * (b_perturbed + np.eye(3) / 3)

        return result[0] if single else result


# =============================================================================
# Localized Perturbation Predictor
# =============================================================================
class LocalizedPerturbationPredictor:
    """
    Random Forest-based predictor for spatially varying perturbation magnitudes.

    Replaces global uniform perturbations with locally adapted magnitudes
    based on flow topology indicators.

    Uses simple decision tree ensemble (numpy, no sklearn required).

    Parameters
    ----------
    n_features : int
        Number of flow topology features.
    n_trees : int
        Number of trees in the ensemble.
    max_depth : int
        Maximum tree depth.
    seed : int
    """

    def __init__(self, n_features: int = 6, n_trees: int = 10,
                 max_depth: int = 4, seed: int = 42):
        self.n_features = n_features
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.rng = np.random.default_rng(seed)
        self._fitted = False
        self._trees = []

    def fit(self, features: np.ndarray, perturbation_magnitudes: np.ndarray):
        """
        Train random forest on flow features → perturbation magnitudes.

        Parameters
        ----------
        features : ndarray (n_samples, n_features)
            Flow topology indicators (strain ratio, APG indicator, etc.)
        perturbation_magnitudes : ndarray (n_samples,)
            Target local perturbation magnitudes.
        """
        self._trees = []
        for t in range(self.n_trees):
            # Bootstrap sample
            idx = self.rng.choice(len(features), len(features), replace=True)
            tree = self._build_tree(features[idx], perturbation_magnitudes[idx], 0)
            self._trees.append(tree)
        self._fitted = True

    def _build_tree(self, X, y, depth):
        """Build a simple decision tree (regression)."""
        if depth >= self.max_depth or len(X) < 4:
            return {"leaf": True, "value": float(np.mean(y))}

        # Random feature subset
        n_try = max(1, self.n_features // 3)
        feat_subset = self.rng.choice(self.n_features, n_try, replace=False)

        best_feat = feat_subset[0]
        best_thresh = float(np.median(X[:, best_feat]))
        best_score = float("inf")

        for f in feat_subset:
            thresh = float(np.median(X[:, f]))
            left_mask = X[:, f] <= thresh
            right_mask = ~left_mask
            if np.sum(left_mask) < 2 or np.sum(right_mask) < 2:
                continue
            score = (np.var(y[left_mask]) * np.sum(left_mask) +
                     np.var(y[right_mask]) * np.sum(right_mask))
            if score < best_score:
                best_score = score
                best_feat = f
                best_thresh = thresh

        left_mask = X[:, best_feat] <= best_thresh
        if np.sum(left_mask) < 2 or np.sum(~left_mask) < 2:
            return {"leaf": True, "value": float(np.mean(y))}

        return {
            "leaf": False,
            "feature": best_feat,
            "threshold": best_thresh,
            "left": self._build_tree(X[left_mask], y[left_mask], depth + 1),
            "right": self._build_tree(X[~left_mask], y[~left_mask], depth + 1),
        }

    def _predict_tree(self, tree, x):
        if tree["leaf"]:
            return tree["value"]
        if x[tree["feature"]] <= tree["threshold"]:
            return self._predict_tree(tree["left"], x)
        return self._predict_tree(tree["right"], x)

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Predict local perturbation magnitudes.

        Parameters
        ----------
        features : ndarray (n_points, n_features)

        Returns
        -------
        magnitudes : ndarray (n_points,) in [0, 1].
        """
        if not self._fitted:
            raise RuntimeError("Predictor not fitted")

        n = len(features)
        preds = np.zeros(n)
        for i in range(n):
            tree_preds = [self._predict_tree(t, features[i]) for t in self._trees]
            preds[i] = np.mean(tree_preds)

        return np.clip(preds, 0.0, 1.0)


# =============================================================================
# Bayesian Inverter (MCMC)
# =============================================================================
class BayesianInverter:
    """
    MCMC-based Bayesian inversion with PCE-accelerated likelihood.

    Uses Metropolis-Hastings sampling with the PCE surrogate
    replacing expensive SU2 forward model evaluations.

    Parameters
    ----------
    pce : PCESurrogate
        Fitted PCE surrogate model.
    obs_data : ndarray
        Experimental observations.
    obs_noise_std : float
        Observation noise standard deviation.
    n_samples : int
        Number of MCMC samples.
    burn_in : int
        Burn-in samples to discard.
    seed : int
    """

    def __init__(self, pce: PCESurrogate, obs_data: np.ndarray,
                 obs_noise_std: float = 0.01, n_samples: int = 5000,
                 burn_in: int = 500, seed: int = 42):
        self.pce = pce
        self.obs_data = obs_data
        self.obs_noise_std = obs_noise_std
        self.n_samples = n_samples
        self.burn_in = burn_in
        self.rng = np.random.default_rng(seed)

    def log_likelihood(self, params: np.ndarray) -> float:
        """Compute log-likelihood using PCE prediction."""
        pred = self.pce.predict(params[np.newaxis, :]).flatten()
        residual = pred[:len(self.obs_data)] - self.obs_data
        return -0.5 * np.sum(residual ** 2) / (self.obs_noise_std ** 2)

    def log_prior(self, params: np.ndarray) -> float:
        """Uniform prior on [-1, 1] (PCE normalized domain)."""
        if np.all(np.abs(params) <= 1):
            return 0.0
        return -np.inf

    def run(self, proposal_std: float = 0.05) -> Dict[str, np.ndarray]:
        """
        Run MCMC sampling.

        Parameters
        ----------
        proposal_std : float
            Standard deviation of Gaussian proposal distribution.

        Returns
        -------
        Dict with:
            'samples': ndarray (n_samples - burn_in, n_params)
            'acceptance_rate': float
            'log_posteriors': ndarray
        """
        n_params = self.pce.n_params
        current = self.rng.uniform(-0.5, 0.5, n_params)
        current_log_post = self.log_likelihood(current) + self.log_prior(current)

        samples = np.zeros((self.n_samples, n_params))
        log_posteriors = np.zeros(self.n_samples)
        n_accepted = 0

        for i in range(self.n_samples):
            # Propose
            proposal = current + self.rng.standard_normal(n_params) * proposal_std
            proposal_log_post = self.log_likelihood(proposal) + self.log_prior(proposal)

            # Accept/reject
            log_alpha = proposal_log_post - current_log_post
            if np.log(self.rng.random()) < log_alpha:
                current = proposal
                current_log_post = proposal_log_post
                n_accepted += 1

            samples[i] = current
            log_posteriors[i] = current_log_post

        return {
            "samples": samples[self.burn_in:],
            "acceptance_rate": n_accepted / self.n_samples,
            "log_posteriors": log_posteriors[self.burn_in:],
        }


# =============================================================================
# High-Level Bayesian PCE Framework
# =============================================================================
@dataclass
class BayesianPCEConfig:
    """Configuration for Bayesian PCE UQ framework."""
    n_params: int = 5
    max_order: int = 3
    n_training_samples: int = 100
    n_mcmc_samples: int = 5000
    burn_in: int = 500
    obs_noise_std: float = 0.01
    proposal_std: float = 0.05
    seed: int = 42


class BayesianPCEFramework:
    """
    High-level Bayesian PCE uncertainty quantification framework.

    Workflow:
        1. Generate training samples (Latin hypercube or Sobol)
        2. Fit PCE surrogate from model evaluations
        3. Run MCMC Bayesian inversion
        4. Extract posterior statistics and credible intervals

    Parameters
    ----------
    config : BayesianPCEConfig
    """

    def __init__(self, config: BayesianPCEConfig = None, **kwargs):
        if config is None:
            config = BayesianPCEConfig(**{k: v for k, v in kwargs.items()
                                           if k in BayesianPCEConfig.__dataclass_fields__})
        self.config = config
        self.pce = PCESurrogate(config.n_params, config.max_order)
        self.barycentric = BarycentricMapper()
        self._results = None

    def generate_training_samples(self, n_samples: int = None,
                                   seed: int = None) -> np.ndarray:
        """Generate Latin hypercube samples in [-1, 1]."""
        n = n_samples or self.config.n_training_samples
        rng = np.random.default_rng(seed or self.config.seed)

        # Simple LHS
        samples = np.zeros((n, self.config.n_params))
        for j in range(self.config.n_params):
            perm = rng.permutation(n)
            samples[:, j] = (perm + rng.random(n)) / n * 2 - 1
        return samples

    def run_calibration(self, model_func: Callable,
                        obs_data: np.ndarray) -> Dict[str, Any]:
        """
        Run full Bayesian calibration pipeline.

        Parameters
        ----------
        model_func : callable (n_samples, n_params) → (n_samples, n_qoi)
        obs_data : ndarray (n_qoi,)

        Returns
        -------
        Dict with posterior statistics.
        """
        t0 = time.time()

        # 1. Generate training data
        samples = self.generate_training_samples()
        outputs = model_func(samples)

        # 2. Fit PCE
        self.pce.fit(samples, outputs)
        pce_stats = self.pce.compute_statistics()

        # 3. Bayesian inversion
        inverter = BayesianInverter(
            self.pce, obs_data,
            obs_noise_std=self.config.obs_noise_std,
            n_samples=self.config.n_mcmc_samples,
            burn_in=self.config.burn_in,
            seed=self.config.seed,
        )
        mcmc_results = inverter.run(self.config.proposal_std)

        # 4. Extract posterior statistics
        posterior_samples = mcmc_results["samples"]
        posterior_mean = np.mean(posterior_samples, axis=0)
        posterior_std = np.std(posterior_samples, axis=0)
        credible_lower = np.percentile(posterior_samples, 2.5, axis=0)
        credible_upper = np.percentile(posterior_samples, 97.5, axis=0)

        self._results = {
            "posterior_mean": posterior_mean,
            "posterior_std": posterior_std,
            "credible_interval_95": (credible_lower, credible_upper),
            "acceptance_rate": mcmc_results["acceptance_rate"],
            "pce_mean": pce_stats["mean"],
            "pce_std": pce_stats["std"],
            "sobol_indices": self.pce.sobol_indices(),
            "n_pce_terms": self.pce.n_terms,
            "calibration_time_s": time.time() - t0,
        }

        logger.info("Bayesian PCE calibration complete: acc_rate=%.2f",
                     mcmc_results["acceptance_rate"])
        return self._results

    def summary(self) -> str:
        """Generate human-readable summary."""
        if self._results is None:
            return "No calibration results available."

        r = self._results
        lines = [
            "Bayesian PCE UQ Summary",
            "=" * 40,
            f"Parameters: {self.config.n_params}",
            f"PCE order: {self.config.max_order}",
            f"PCE terms: {r['n_pce_terms']}",
            f"MCMC acceptance rate: {r['acceptance_rate']:.2%}",
            f"Posterior mean: {r['posterior_mean']}",
            f"Posterior std: {r['posterior_std']}",
            f"Calibration time: {r['calibration_time_s']:.1f}s",
        ]
        return "\n".join(lines)
