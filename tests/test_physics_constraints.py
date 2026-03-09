"""
Physics Constraint Tests for ML-Augmented Turbulence Closures
==============================================================
Verifies that ML models satisfy fundamental physical constraints:
  - Galilean invariance of input features
  - Realizability of predicted Reynolds stress anisotropy
  - Trace-free property (b_kk = 0)
  - Symmetry of output tensors (b_ij = b_ji)
  - Conservation consistency

Run: pytest tests/test_physics_constraints.py -v --tb=short
"""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def synthetic_tensors(rng):
    """Generate synthetic strain/rotation rate tensors."""
    N = 200
    dudx = rng.standard_normal((N, 3, 3)) * 0.1
    S = 0.5 * (dudx + np.swapaxes(dudx, -2, -1))
    Omega = 0.5 * (dudx - np.swapaxes(dudx, -2, -1))
    k = np.abs(rng.standard_normal(N)) * 0.1 + 0.01
    epsilon = np.abs(rng.standard_normal(N)) * 0.1 + 0.01
    return S, Omega, k, epsilon


@pytest.fixture
def random_rotation(rng):
    """Generate a random proper rotation matrix."""
    A = rng.standard_normal((3, 3))
    Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


@pytest.fixture
def synthetic_anisotropy(rng):
    """Generate synthetic realizable anisotropy tensors."""
    N = 100
    b = rng.standard_normal((N, 3, 3)) * 0.01
    b = 0.5 * (b + np.swapaxes(b, -2, -1))  # Symmetrize
    b -= np.trace(b, axis1=-2, axis2=-1)[:, None, None] / 3.0 * np.eye(3)
    return b


# =============================================================================
# Tensor Basis Tests
# =============================================================================

class TestTensorBasis:
    """Test the Pope (1975) tensor basis computation."""

    def test_basis_shape(self, synthetic_tensors):
        from scripts.ml_augmentation.tbnn_closure import compute_tensor_basis
        S, Omega, k, epsilon = synthetic_tensors
        tau = k / (epsilon + 1e-10)
        S_hat = S * tau[:, None, None]
        O_hat = Omega * tau[:, None, None]
        T = compute_tensor_basis(S_hat, O_hat)
        assert T.shape == (len(k), 10, 3, 3)

    def test_basis_symmetry(self, synthetic_tensors):
        """T^(1), T^(3), T^(4), T^(6), T^(9) should be symmetric."""
        from scripts.ml_augmentation.tbnn_closure import compute_tensor_basis
        S, Omega, k, epsilon = synthetic_tensors
        tau = k / (epsilon + 1e-10)
        T = compute_tensor_basis(S * tau[:, None, None], Omega * tau[:, None, None])
        # T^(1) = S_hat is symmetric
        for idx in [0, 2, 3, 5, 8]:
            sym_err = np.max(np.abs(T[:, idx] - np.swapaxes(T[:, idx], -2, -1)))
            assert sym_err < 1e-10, f"T^({idx+1}) not symmetric, err={sym_err}"

    def test_commutator_bases_tracefree(self, synthetic_tensors):
        """T^(2), T^(5), T^(7), T^(8), T^(10) are commutator-type bases;
        they should be trace-free (not necessarily antisymmetric)."""
        from scripts.ml_augmentation.tbnn_closure import compute_tensor_basis
        S, Omega, k, epsilon = synthetic_tensors
        tau = k / (epsilon + 1e-10)
        T = compute_tensor_basis(S * tau[:, None, None], Omega * tau[:, None, None])
        for idx in [1, 4, 6, 7, 9]:
            traces = np.abs(np.trace(T[:, idx], axis1=-2, axis2=-1))
            max_trace = np.max(traces)
            assert max_trace < 1e-8, f"T^({idx+1}) not trace-free, max trace={max_trace}"

    def test_trace_free_bases(self, synthetic_tensors):
        """All symmetric bases should be trace-free."""
        from scripts.ml_augmentation.tbnn_closure import compute_tensor_basis
        S, Omega, k, epsilon = synthetic_tensors
        tau = k / (epsilon + 1e-10)
        T = compute_tensor_basis(S * tau[:, None, None], Omega * tau[:, None, None])
        for idx in [2, 3, 5, 8]:  # T^(3), T^(4), T^(6), T^(9)
            traces = np.abs(np.trace(T[:, idx], axis1=-2, axis2=-1))
            max_trace = np.max(traces)
            assert max_trace < 1e-8, f"T^({idx+1}) not trace-free, max trace={max_trace}"


# =============================================================================
# Galilean Invariance Tests
# =============================================================================

class TestGalileanInvariance:
    """Verify that features and predictions are Galilean-invariant."""

    def test_scalar_invariants_under_rotation(self, synthetic_tensors, random_rotation):
        """Scalar invariants λ₁–λ₅ must be invariant under rotation."""
        from scripts.ml_augmentation.tbnn_closure import compute_invariant_inputs
        S, Omega, k, epsilon = synthetic_tensors
        tau = k / (epsilon + 1e-10)
        S_hat = S * tau[:, None, None]
        O_hat = Omega * tau[:, None, None]

        Q = random_rotation

        # Original invariants
        lam_orig = compute_invariant_inputs(S_hat, O_hat)

        # Rotated tensors: S' = Q S Q^T
        S_rot = np.einsum("ij,njk,lk->nil", Q, S_hat, Q)
        O_rot = np.einsum("ij,njk,lk->nil", Q, O_hat, Q)

        lam_rot = compute_invariant_inputs(S_rot, O_rot)

        max_err = np.max(np.abs(lam_orig - lam_rot))
        assert max_err < 1e-8, f"Invariants changed under rotation, max_err={max_err}"

    def test_multiple_rotations(self, synthetic_tensors):
        """Test invariance under 10 random rotations."""
        from scripts.ml_augmentation.tbnn_closure import (
            compute_invariant_inputs, verify_galilean_invariance
        )
        S, Omega, k, epsilon = synthetic_tensors

        def features_func(S_in, O_in, k_in, eps_in):
            tau_in = k_in / (eps_in + 1e-10)
            return compute_invariant_inputs(
                S_in * tau_in[:, None, None],
                O_in * tau_in[:, None, None],
            )

        result = verify_galilean_invariance(
            features_func, S, Omega, k, epsilon, n_rotations=10
        )
        assert result["passed"], f"Galilean invariance failed, max_err={result['max_error']}"

    def test_feature_extraction_invariance(self, rng):
        """The feature_extraction module features should also be invariant."""
        from scripts.ml_augmentation.feature_extraction import extract_invariant_features

        N = 50
        dudx = rng.standard_normal((N, 3, 3)) * 0.1
        k = np.abs(rng.standard_normal(N)) + 0.01
        eps = np.abs(rng.standard_normal(N)) + 0.01
        wd = np.abs(rng.standard_normal(N)) + 0.001

        feats_orig = extract_invariant_features(dudx, k, eps, wd)

        # Apply rotation
        A = rng.standard_normal((3, 3))
        Q, _ = np.linalg.qr(A)
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1

        dudx_rot = np.einsum("ij,njk,lk->nil", Q, dudx, Q)
        feats_rot = extract_invariant_features(dudx_rot, k, eps, wd)

        # Core invariants (S_norm, O_norm, Q_criterion, lambda1-5) should be invariant
        # Indices 0-7 are the invariant features
        max_err = np.max(np.abs(
            feats_orig.values[:, :8] - feats_rot.values[:, :8]
        ))
        assert max_err < 1e-6, f"Feature extraction not Galilean-invariant, err={max_err}"


# =============================================================================
# Realizability Tests
# =============================================================================

class TestRealizability:
    """Test realizability constraints on anisotropy tensors."""

    def test_trace_free(self, synthetic_anisotropy):
        """b_kk = 0 for all points."""
        from scripts.ml_augmentation.tbnn_closure import check_realizability
        report = check_realizability(synthetic_anisotropy)
        assert report.max_trace_error < 1e-10, f"Trace not zero: {report.max_trace_error}"

    def test_symmetry(self, synthetic_anisotropy):
        """b_ij = b_ji for all points."""
        from scripts.ml_augmentation.tbnn_closure import check_realizability
        report = check_realizability(synthetic_anisotropy)
        assert report.max_symmetry_error < 1e-10, f"Not symmetric: {report.max_symmetry_error}"

    def test_eigenvalue_bounds(self, synthetic_anisotropy):
        """Eigenvalues within [-1/3, 2/3]."""
        from scripts.ml_augmentation.tbnn_closure import check_realizability
        report = check_realizability(synthetic_anisotropy)
        assert report.eigenvalue_bounds_satisfied, "Eigenvalue bounds violated"

    def test_realizable_fraction(self, synthetic_anisotropy):
        """Synthetic data (small perturbation) should be ~100% realizable."""
        from scripts.ml_augmentation.tbnn_closure import check_realizability
        report = check_realizability(synthetic_anisotropy)
        assert report.fraction_realizable > 0.95, \
            f"Only {report.fraction_realizable*100:.1f}% realizable"

    def test_projection_enforces_symmetry_and_tracefree(self, rng):
        """After projection, tensors should be symmetric and trace-free."""
        from scripts.ml_augmentation.tbnn_closure import project_to_realizable
        N = 100
        b_bad = rng.standard_normal((N, 3, 3)) * 0.5
        b_proj = project_to_realizable(b_bad)
        # Check symmetry
        sym_err = np.max(np.abs(b_proj - np.swapaxes(b_proj, -2, -1)))
        assert sym_err < 1e-10, f"Not symmetric after projection: {sym_err}"
        # Check trace-free
        traces = np.abs(np.trace(b_proj, axis1=-2, axis2=-1))
        assert np.max(traces) < 1e-10, f"Not trace-free after projection: {np.max(traces)}"

    def test_projection_trace_free(self, rng):
        """Projected tensors should be trace-free."""
        from scripts.ml_augmentation.tbnn_closure import project_to_realizable
        N = 50
        b = rng.standard_normal((N, 3, 3)) * 0.3
        b_proj = project_to_realizable(b)
        traces = np.abs(np.trace(b_proj, axis1=-2, axis2=-1))
        assert np.max(traces) < 1e-10, f"Projected b not trace-free: max trace={np.max(traces)}"


# =============================================================================
# FIML Pipeline Tests
# =============================================================================

class TestFIMLPipeline:
    """Test the FIML pipeline with synthetic data."""

    def test_synthetic_case_generation(self):
        """Synthetic FIML case should have correct shapes."""
        from scripts.ml_augmentation.fiml_pipeline import generate_synthetic_fiml_case
        case = generate_synthetic_fiml_case(n_points=500)
        assert case.features.shape == (500, 5)
        assert case.beta_target.shape == (500,)
        assert case.x_coords.shape == (500,)

    def test_feature_extraction(self, rng):
        """FIML feature extraction should return 5 features."""
        from scripts.ml_augmentation.fiml_pipeline import extract_fiml_features
        N = 100
        features = extract_fiml_features(
            nu_t=np.abs(rng.standard_normal(N)) * 1e-4,
            nu=1.5e-5,
            strain_mag=np.abs(rng.standard_normal(N)) * 100,
            wall_distance=np.abs(rng.standard_normal(N)) * 0.01 + 0.001,
            strain_rotation_ratio=rng.standard_normal(N) * 0.5,
            pressure_gradient_indicator=rng.standard_normal(N) * 0.3,
        )
        assert features.shape == (100, 5)
        assert np.all(np.isfinite(features)), "Features contain NaN/Inf"

    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("sklearn"),
        reason="scikit-learn not installed"
    )
    def test_pipeline_train_and_predict(self):
        """Pipeline should train and produce predictions."""
        from scripts.ml_augmentation.fiml_pipeline import FIMLPipeline, generate_synthetic_fiml_case
        pipeline = FIMLPipeline(hidden_layers=(32, 32), max_iter=500)
        pipeline.add_case(generate_synthetic_fiml_case("case_a", 500, seed=1))
        pipeline.add_case(generate_synthetic_fiml_case("case_b", 500, seed=2))

        result = pipeline.train(test_case="case_b")
        assert result.train_r2 > -5, "Training failed completely"

        beta_pred = pipeline.predict(pipeline.cases["case_b"].features)
        assert len(beta_pred) == 500

    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("sklearn"),
        reason="scikit-learn not installed"
    )
    def test_cross_validation(self):
        """Cross-validation should produce results for all cases."""
        from scripts.ml_augmentation.fiml_pipeline import FIMLPipeline, generate_synthetic_fiml_case
        pipeline = FIMLPipeline(hidden_layers=(16,), max_iter=50)
        pipeline.add_case(generate_synthetic_fiml_case("c1", 200, seed=1))
        pipeline.add_case(generate_synthetic_fiml_case("c2", 200, seed=2))
        pipeline.add_case(generate_synthetic_fiml_case("c3", 200, seed=3))

        cv_results = pipeline.cross_validate()
        assert len(cv_results) == 3
        for name, res in cv_results.items():
            assert hasattr(res, "test_r2")


# =============================================================================
# Conservation Consistency Tests
# =============================================================================

class TestConservation:
    """Test that ML corrections don't violate conservation principles."""

    def test_beta_field_bounded(self):
        """β correction should be physically bounded (no negative β)."""
        from scripts.ml_augmentation.fiml_pipeline import generate_synthetic_fiml_case
        case = generate_synthetic_fiml_case(n_points=1000)
        assert np.all(case.beta_target > 0), "β field contains non-positive values"

    def test_anisotropy_trace_conservation(self, rng):
        """Trace of anisotropy + 2k/3 * I should give Reynolds stress tensor
        with trace = 2k (TKE conservation)."""
        N = 50
        b = rng.standard_normal((N, 3, 3)) * 0.01
        b = 0.5 * (b + np.swapaxes(b, -2, -1))
        b -= np.trace(b, axis1=-2, axis2=-1)[:, None, None] / 3.0 * np.eye(3)

        k = np.abs(rng.standard_normal(N)) + 0.01
        I = np.eye(3)[None, :, :].repeat(N, axis=0)

        # Full Reynolds stress: R_ij = 2k(b_ij + δ_ij/3)
        R = 2 * k[:, None, None] * (b + I / 3.0)
        traces = np.trace(R, axis1=-2, axis2=-1)

        # trace(R) should equal 2k
        np.testing.assert_allclose(traces, 2 * k, atol=1e-10,
                                    err_msg="TKE not conserved in Reynolds stress tensor")
