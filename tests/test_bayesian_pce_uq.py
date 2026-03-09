#!/usr/bin/env python3
"""
Tests for Bayesian PCE UQ Module
====================================
Validates PCE surrogate, Bayesian inversion, eigenspace perturbation,
barycentric mapping, localized perturbation prediction, and full framework.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ml_augmentation.bayesian_pce_uq import (
    PCESurrogate,
    BarycentricMapper,
    BayesianInverter,
    BayesianPCEConfig,
    BayesianPCEFramework,
    EigenspacePerturbation,
    LocalizedPerturbationPredictor,
)


# =========================================================================
# TestPCESurrogate
# =========================================================================
class TestPCESurrogate:
    """Tests for Polynomial Chaos Expansion surrogate."""

    def test_multi_index_generation(self):
        pce = PCESurrogate(n_params=2, max_order=2)
        # For 2 params, order 2: (0,0),(1,0),(0,1),(2,0),(1,1),(0,2) = 6
        assert pce.n_terms == 6

    def test_fit_and_predict(self):
        pce = PCESurrogate(n_params=2, max_order=2)
        rng = np.random.default_rng(42)
        samples = rng.uniform(-1, 1, (50, 2))
        outputs = 2 * samples[:, 0] + 3 * samples[:, 1] ** 2
        pce.fit(samples, outputs)
        pred = pce.predict(samples)
        rmse = np.sqrt(np.mean((pred.flatten() - outputs) ** 2))
        assert rmse < 0.5  # Should fit reasonably well

    def test_statistics(self):
        pce = PCESurrogate(n_params=1, max_order=3)
        samples = np.linspace(-1, 1, 30).reshape(-1, 1)
        outputs = samples[:, 0] ** 2  # Parabola
        pce.fit(samples, outputs)
        stats = pce.compute_statistics()
        assert "mean" in stats
        assert "variance" in stats
        assert "std" in stats

    def test_sobol_indices(self):
        pce = PCESurrogate(n_params=2, max_order=2)
        rng = np.random.default_rng(42)
        samples = rng.uniform(-1, 1, (100, 2))
        # Output dominated by first parameter
        outputs = 5 * samples[:, 0] + 0.1 * samples[:, 1]
        pce.fit(samples, outputs)
        sobol = pce.sobol_indices()
        assert "S1" in sobol
        assert sobol["S1"].shape[0] == 2
        # First parameter should dominate
        assert sobol["S1"][0, 0] > sobol["S1"][1, 0]

    def test_predict_before_fit_raises(self):
        pce = PCESurrogate(n_params=2)
        with pytest.raises(RuntimeError, match="not fitted"):
            pce.predict(np.zeros((5, 2)))

    def test_legendre_orthogonality(self):
        """Low-order Legendre polynomials should be approximately orthogonal."""
        pce = PCESurrogate(n_params=1, max_order=3)
        x = np.linspace(-1, 1, 1000)
        P0 = pce._eval_legendre(x, 0)
        P1 = pce._eval_legendre(x, 1)
        P2 = pce._eval_legendre(x, 2)
        # Orthogonality: ∫ Pm * Pn dx ≈ 0 for m ≠ n
        inner01 = np.trapezoid(P0 * P1, x)
        inner12 = np.trapezoid(P1 * P2, x)
        assert abs(inner01) < 0.01
        assert abs(inner12) < 0.01


# =========================================================================
# TestBarycentricMapper
# =========================================================================
class TestBarycentricMapper:
    """Tests for anisotropy-to-barycentric mapping."""

    def test_isotropic_maps_to_3c(self):
        mapper = BarycentricMapper()
        # Isotropic: b_ij = 0 → eigenvalues all zero
        b_iso = np.zeros((3, 3))
        coords = mapper.anisotropy_to_barycentric(b_iso)
        np.testing.assert_allclose(coords, mapper.X3C, atol=0.1)

    def test_1c_maps_to_vertex(self):
        mapper = BarycentricMapper()
        b_1c = np.diag([2 / 3, -1 / 3, -1 / 3])
        coords = mapper.anisotropy_to_barycentric(b_1c)
        np.testing.assert_allclose(coords, mapper.X1C, atol=0.1)

    def test_batch_mapping(self):
        mapper = BarycentricMapper()
        b_batch = np.zeros((5, 3, 3))
        for i in range(5):
            b_batch[i] = np.diag([0.1 * i, -0.05 * i, -0.05 * i])
        coords = mapper.anisotropy_to_barycentric(b_batch)
        assert coords.shape == (5, 2)

    def test_realizability_check(self):
        mapper = BarycentricMapper()
        # Point inside triangle
        inside = mapper.is_inside_triangle(np.array([0.5, 0.3]))
        assert inside
        # Point clearly outside
        outside = mapper.is_inside_triangle(np.array([2.0, 2.0]))
        assert not outside


# =========================================================================
# TestEigenspacePerturbation
# =========================================================================
class TestEigenspacePerturbation:
    """Tests for Reynolds stress eigenspace perturbation."""

    def test_identity_at_zero_delta(self):
        """Zero perturbation should approximately preserve original stress."""
        R = np.diag([0.5, 0.3, 0.2])  # Diagonal → no eigenvector misalignment
        perturber = EigenspacePerturbation(delta_b=0.0, target_state="3c")
        perturbed = perturber.perturb(R)
        np.testing.assert_allclose(perturbed, R, atol=1e-10)

    def test_full_perturbation_to_isotropic(self):
        """Full perturbation to 3C should yield isotropic stress."""
        R = np.array([[0.6, 0, 0], [0, 0.3, 0], [0, 0, 0.1]])
        k = 0.5 * np.trace(R)  # TKE
        perturber = EigenspacePerturbation(delta_b=1.0, target_state="3c")
        perturbed = perturber.perturb(R)
        # Should be (2k/3) * I
        expected = (2 * k / 3) * np.eye(3)
        np.testing.assert_allclose(perturbed, expected, atol=1e-10)

    def test_positive_semidefinite(self):
        """Perturbed Reynolds stress should be positive semi-definite."""
        R = np.array([[0.5, 0.1, 0.05], [0.1, 0.3, -0.02],
                       [0.05, -0.02, 0.2]])
        for state in ["1c", "2c", "3c"]:
            perturber = EigenspacePerturbation(delta_b=0.5, target_state=state)
            perturbed = perturber.perturb(R)
            eigvals = np.linalg.eigvalsh(perturbed)
            assert np.all(eigvals >= -1e-10)

    def test_batch_perturbation(self):
        R_batch = np.zeros((3, 3, 3))
        for i in range(3):
            R_batch[i] = np.diag([0.5 + 0.1 * i, 0.3, 0.2])
        perturber = EigenspacePerturbation(delta_b=0.5)
        perturbed = perturber.perturb(R_batch)
        assert perturbed.shape == (3, 3, 3)


# =========================================================================
# TestLocalizedPerturbationPredictor
# =========================================================================
class TestLocalizedPerturbationPredictor:
    """Tests for Random Forest perturbation magnitude predictor."""

    def test_fit_and_predict(self):
        rng = np.random.default_rng(42)
        features = rng.standard_normal((100, 6))
        magnitudes = np.clip(np.abs(features[:, 0]) * 0.5, 0, 1)
        predictor = LocalizedPerturbationPredictor(n_features=6, n_trees=5)
        predictor.fit(features, magnitudes)
        pred = predictor.predict(features[:10])
        assert pred.shape == (10,)
        assert np.all(pred >= 0)
        assert np.all(pred <= 1)

    def test_predict_before_fit_raises(self):
        predictor = LocalizedPerturbationPredictor()
        with pytest.raises(RuntimeError, match="not fitted"):
            predictor.predict(np.zeros((5, 6)))

    def test_bounded_output(self):
        rng = np.random.default_rng(42)
        features = rng.standard_normal((50, 4))
        mags = rng.uniform(0, 1, 50)
        predictor = LocalizedPerturbationPredictor(n_features=4, n_trees=3)
        predictor.fit(features, mags)
        pred = predictor.predict(rng.standard_normal((20, 4)))
        assert np.all(pred >= 0)
        assert np.all(pred <= 1)


# =========================================================================
# TestBayesianInverter
# =========================================================================
class TestBayesianInverter:
    """Tests for MCMC Bayesian inversion."""

    def test_mcmc_runs(self):
        pce = PCESurrogate(n_params=2, max_order=2)
        rng = np.random.default_rng(42)
        samples = rng.uniform(-1, 1, (50, 2))
        outputs = 2 * samples[:, 0] + samples[:, 1]
        pce.fit(samples, outputs)
        obs = np.array([1.0])
        inverter = BayesianInverter(pce, obs, n_samples=100, burn_in=10)
        results = inverter.run()
        assert results["samples"].shape[0] == 90  # 100 - 10
        assert 0 < results["acceptance_rate"] < 1

    def test_posterior_shape(self):
        pce = PCESurrogate(n_params=3, max_order=2)
        rng = np.random.default_rng(42)
        samples = rng.uniform(-1, 1, (80, 3))
        outputs = samples.sum(axis=1)
        pce.fit(samples, outputs)
        obs = np.array([0.5])
        inverter = BayesianInverter(pce, obs, n_samples=200, burn_in=20)
        results = inverter.run()
        assert results["samples"].shape == (180, 3)


# =========================================================================
# TestBayesianPCEFramework
# =========================================================================
class TestBayesianPCEFramework:
    """Tests for the high-level Bayesian PCE framework."""

    def test_training_samples_shape(self):
        framework = BayesianPCEFramework(n_params=3)
        samples = framework.generate_training_samples(n_samples=50)
        assert samples.shape == (50, 3)
        assert np.all(samples >= -1)
        assert np.all(samples <= 1)

    def test_full_calibration(self):
        config = BayesianPCEConfig(
            n_params=2, max_order=2,
            n_training_samples=50,
            n_mcmc_samples=100,
            burn_in=10,
        )
        framework = BayesianPCEFramework(config=config)

        def model_func(params):
            return params.sum(axis=1, keepdims=True)

        obs_data = np.array([0.5])
        results = framework.run_calibration(model_func, obs_data)

        assert "posterior_mean" in results
        assert "acceptance_rate" in results
        assert "sobol_indices" in results
        assert results["acceptance_rate"] > 0

    def test_summary(self):
        config = BayesianPCEConfig(
            n_params=2, n_training_samples=30,
            n_mcmc_samples=50, burn_in=5)
        framework = BayesianPCEFramework(config=config)

        def model(p):
            return p[:, :1]

        framework.run_calibration(model, np.array([0.2]))
        s = framework.summary()
        assert "Bayesian PCE" in s


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
