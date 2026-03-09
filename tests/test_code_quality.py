"""
Code Quality Tests
==================
Validates coding sophistication improvements:
- Vectorized vs loop equivalence
- Profiling utilities
- Custom exception hierarchy
- VTU reader imports
- NumPy broadcasting correctness

Run: pytest tests/test_code_quality.py -v
"""

import sys
import time
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =========================================================================
# Vectorization Equivalence Tests
# =========================================================================
class TestVectorizedPhysicsDiagnostics:
    """Verify vectorized implementations produce correct results."""

    def test_boussinesq_shapes(self):
        from scripts.postprocessing.physics_diagnostics import boussinesq_validity
        N = 50
        Sij = np.random.randn(N, 3, 3)
        tau_ij = np.random.randn(N, 3, 3)
        k = np.abs(np.random.randn(N)) + 0.1
        result = boussinesq_validity(Sij, tau_ij, k)
        assert result.values.shape == (N,)
        assert np.all(result.values >= 0)
        assert np.all(result.values <= 1)

    def test_boussinesq_trivial_no_turbulence(self):
        """Points with k=0 should have validity=1."""
        from scripts.postprocessing.physics_diagnostics import boussinesq_validity
        N = 10
        Sij = np.random.randn(N, 3, 3)
        tau_ij = np.random.randn(N, 3, 3)
        k = np.zeros(N)  # No turbulence
        result = boussinesq_validity(Sij, tau_ij, k)
        np.testing.assert_allclose(result.values, 1.0)

    def test_boussinesq_perfect_alignment(self):
        """When tau \u221d S, validity should be high."""
        from scripts.postprocessing.physics_diagnostics import boussinesq_validity
        N = 20
        Sij = np.random.randn(N, 3, 3)
        Sij = (Sij + Sij.transpose(0, 2, 1)) / 2  # Symmetric
        k = np.ones(N) * 10.0
        # Make tau proportional to S (Boussinesq-perfect)
        tau_ij = -2 * 0.1 * Sij + (2.0 / 3) * k[:, None, None] * np.eye(3)[None]
        result = boussinesq_validity(Sij, tau_ij, k)
        # Should be high (>0.8) for perfect alignment
        assert np.mean(result.values) > 0.7

    def test_production_dissipation_einsum(self):
        """Verify einsum tensor contraction matches manual loop."""
        from scripts.postprocessing.physics_diagnostics import production_dissipation_ratio
        N = 100
        Sij = np.random.randn(N, 3, 3)
        tau_ij = np.random.randn(N, 3, 3)
        epsilon = np.abs(np.random.randn(N)) + 0.01

        # Manual loop computation
        P_manual = np.zeros(N)
        for i in range(N):
            P_manual[i] = -np.sum(tau_ij[i] * Sij[i])
        expected = np.where(epsilon > 1e-15, P_manual / epsilon, 0.0)

        result = production_dissipation_ratio(Sij, tau_ij, epsilon)
        np.testing.assert_allclose(result.values, expected, atol=1e-12)

    def test_lumley_triangle_shapes(self):
        from scripts.postprocessing.physics_diagnostics import lumley_triangle_invariants
        N = 30
        uu = np.abs(np.random.randn(N)) + 0.1
        vv = np.abs(np.random.randn(N)) + 0.05
        ww = np.abs(np.random.randn(N)) + 0.05
        result = lumley_triangle_invariants(uu, vv, ww)
        assert result.values.shape == (N, 2)

    def test_lumley_isotropic(self):
        """Equal stresses -> isotropic -> eta near 0."""
        from scripts.postprocessing.physics_diagnostics import lumley_triangle_invariants
        N = 10
        stress = np.ones(N) * 5.0
        result = lumley_triangle_invariants(stress, stress, stress)
        eta = result.values[:, 1]
        np.testing.assert_allclose(eta, 0.0, atol=1e-12)

    def test_lumley_no_turbulence(self):
        """Zero stresses should give zero invariants."""
        from scripts.postprocessing.physics_diagnostics import lumley_triangle_invariants
        N = 10
        zeros = np.zeros(N)
        result = lumley_triangle_invariants(zeros, zeros, zeros)
        np.testing.assert_allclose(result.values, 0.0, atol=1e-15)


class TestVectorizedExtractProfiles:
    """Test vectorized separation/reattachment detection."""

    def test_separation_point_sin(self):
        from scripts.postprocessing.extract_profiles import find_separation_point
        x = np.linspace(0, 10, 1000)
        Cf = np.cos(x)  # Crosses zero at pi/2
        sep = find_separation_point(x, Cf)
        assert sep is not None
        assert abs(sep - np.pi / 2) < 0.02

    def test_no_separation(self):
        from scripts.postprocessing.extract_profiles import find_separation_point
        x = np.linspace(0, 1, 100)
        Cf = np.ones(100)  # Always positive
        assert find_separation_point(x, Cf) is None

    def test_reattachment_point(self):
        from scripts.postprocessing.extract_profiles import find_reattachment_point
        x = np.linspace(0, 10, 1000)
        Cf = np.sin(x) - 0.5  # Crosses zero multiple times
        reat = find_reattachment_point(x, Cf)
        assert reat is not None

    def test_max_contiguous_vectorized(self):
        from scripts.postprocessing.extract_profiles import _max_contiguous
        mask = np.array([False, True, True, True, False, True, True, False])
        assert _max_contiguous(mask) == 3

    def test_max_contiguous_empty(self):
        from scripts.postprocessing.extract_profiles import _max_contiguous
        assert _max_contiguous(np.array([])) == 0

    def test_max_contiguous_all_true(self):
        from scripts.postprocessing.extract_profiles import _max_contiguous
        assert _max_contiguous(np.ones(10, dtype=bool)) == 10

    def test_max_contiguous_all_false(self):
        from scripts.postprocessing.extract_profiles import _max_contiguous
        assert _max_contiguous(np.zeros(10, dtype=bool)) == 0


# =========================================================================
# Profiling Utility Tests
# =========================================================================
class TestProfilingUtilities:
    """Test profiling decorator and benchmark."""

    def test_profile_decorator(self):
        from scripts.utils.profiling import profile_function

        @profile_function
        def add(a, b):
            return a + b

        result = add(2, 3)
        assert result == 5
        assert add._last_profile is not None
        assert add._last_profile.function_name == "add"
        assert add._last_profile.wall_time_s >= 0

    def test_profile_decorator_with_args(self):
        from scripts.utils.profiling import profile_function

        @profile_function(top_n=5)
        def multiply(a, b):
            return a * b

        assert multiply(3, 4) == 12

    def test_benchmark_vectorization(self):
        from scripts.utils.profiling import benchmark_vectorization
        results = benchmark_vectorization(n_points=1000)
        assert "tensor_contraction" in results
        assert "batch_determinant" in results
        assert "zero_crossing" in results
        for key, r in results.items():
            assert r.results_match, f"Results mismatch in {key}"

    def test_timer_context(self):
        from scripts.utils.profiling import timer
        import io, logging
        with timer("test_block"):
            time.sleep(0.01)

    def test_memory_tracker(self):
        from scripts.utils.profiling import memory_tracker
        with memory_tracker("test"):
            x = np.zeros(10000)

    def test_benchmark_report(self):
        from scripts.utils.profiling import benchmark_vectorization, print_benchmark_report
        results = benchmark_vectorization(n_points=500)
        report = print_benchmark_report(results)
        assert "Speedup" in report
        assert "OK" in report


# =========================================================================
# Custom Exception Tests
# =========================================================================
class TestCustomExceptions:
    """Test exception hierarchy and messages."""

    def test_base_exception(self):
        from scripts.utils.exceptions import CFDBenchmarkError
        assert issubclass(CFDBenchmarkError, Exception)

    def test_solver_error_hierarchy(self):
        from scripts.utils.exceptions import (
            CFDBenchmarkError, SolverError, SolverNotFoundError,
            ConvergenceError, SolverCrashError,
        )
        assert issubclass(SolverError, CFDBenchmarkError)
        assert issubclass(SolverNotFoundError, SolverError)
        assert issubclass(ConvergenceError, SolverError)
        assert issubclass(SolverCrashError, SolverError)

    def test_mesh_error_hierarchy(self):
        from scripts.utils.exceptions import (
            CFDBenchmarkError, MeshError, MeshNotFoundError, MeshQualityError,
        )
        assert issubclass(MeshError, CFDBenchmarkError)
        assert issubclass(MeshNotFoundError, MeshError)
        assert issubclass(MeshQualityError, MeshError)

    def test_data_error_hierarchy(self):
        from scripts.utils.exceptions import (
            CFDBenchmarkError, DataError, DataNotFoundError, DataFormatError,
        )
        assert issubclass(DataError, CFDBenchmarkError)
        assert issubclass(DataNotFoundError, DataError)
        assert issubclass(DataFormatError, DataError)

    def test_convergence_error_message(self):
        from scripts.utils.exceptions import ConvergenceError
        err = ConvergenceError("SU2", 1e-10, 1e-5, 50000, "/tmp/case")
        assert "1e-10" in str(err) or "1e-05" in str(err)
        assert "50000" in str(err)

    def test_solver_not_found_message(self):
        from scripts.utils.exceptions import SolverNotFoundError
        err = SolverNotFoundError("SU2_CFD")
        assert "SU2_CFD" in str(err)
        assert "not found" in str(err).lower()

    def test_validation_error(self):
        from scripts.utils.exceptions import ValidationError
        err = ValidationError("CL", 1.09, 1.15, 0.02)
        assert "CL" in str(err)
        assert "deviation" in str(err).lower()

    def test_invalid_model_error(self):
        from scripts.utils.exceptions import InvalidModelError
        err = InvalidModelError("RSM", ["SA", "SST", "KE"])
        assert "RSM" in str(err)
        assert "SA" in str(err)

    def test_catch_specific_exception(self):
        """Verify specific exceptions can be caught without catching unrelated ones."""
        from scripts.utils.exceptions import SolverError, MeshError
        with pytest.raises(SolverError):
            raise SolverError("SU2", "test failure")
        # MeshError should NOT catch SolverError
        with pytest.raises(SolverError):
            try:
                raise SolverError("SU2", "test")
            except MeshError:
                pytest.fail("MeshError should not catch SolverError")


# =========================================================================
# VTU Reader Tests
# =========================================================================
class TestVTUReader:
    """Test VTU reader module (structure only - no actual VTU files needed)."""

    def test_import(self):
        from scripts.postprocessing.vtu_reader import VTUReader, VTUField
        assert VTUReader is not None
        assert VTUField is not None

    def test_has_pyvista_check(self):
        from scripts.postprocessing import vtu_reader
        # Should have these flags
        assert hasattr(vtu_reader, '_HAS_PYVISTA')
        assert hasattr(vtu_reader, '_HAS_MESHIO')

    def test_vtu_field_dataclass(self):
        from scripts.postprocessing.vtu_reader import VTUField
        f = VTUField(name="Velocity", n_components=3, field_type="point", dtype="float64")
        assert f.name == "Velocity"
        assert f.n_components == 3

    def test_read_su2_vtk_function_exists(self):
        from scripts.postprocessing.vtu_reader import read_su2_vtk
        assert callable(read_su2_vtk)

    def test_nonexistent_file_raises(self):
        """VTUReader should raise DataNotFoundError for missing files."""
        from scripts.postprocessing.vtu_reader import VTUReader
        from scripts.utils.exceptions import DataNotFoundError
        with pytest.raises(DataNotFoundError):
            VTUReader("/nonexistent/path/flow.vtu")
