"""
Tests for Transition Model Implementation
==========================================
Validates the ERCOFTAC T3 runner script: case parameters, analytical
correlations, SU2 config generation, and Cf parsing utilities.

Run: pytest tests/test_transition.py -v --tb=short
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Case Configuration Tests
# =============================================================================

class TestT3Cases:
    """Verify ERCOFTAC T3 case definitions are physically consistent."""

    def test_all_cases_defined(self):
        from run_transition_t3 import T3_CASES
        assert "T3A" in T3_CASES
        assert "T3B" in T3_CASES
        assert "T3A-" in T3_CASES

    def test_t3a_parameters(self):
        from run_transition_t3 import T3_CASES
        case = T3_CASES["T3A"]
        assert case.Tu_percent == pytest.approx(3.3)
        assert case.U_inf == pytest.approx(5.4)
        assert case.rho == pytest.approx(1.2)
        assert case.mu == pytest.approx(1.8e-5)
        assert case.transition_mechanism == "bypass"

    def test_t3b_higher_tu_than_t3a(self):
        from run_transition_t3 import T3_CASES
        assert T3_CASES["T3B"].Tu_percent > T3_CASES["T3A"].Tu_percent

    def test_t3a_minus_lower_tu_than_t3a(self):
        from run_transition_t3 import T3_CASES
        assert T3_CASES["T3A-"].Tu_percent < T3_CASES["T3A"].Tu_percent

    def test_reynolds_number_consistency(self):
        """Re/m = U * rho / mu should match computed property."""
        from run_transition_t3 import T3_CASES
        for name, case in T3_CASES.items():
            Re_per_m = case.U_inf * case.rho / case.mu
            assert Re_per_m == pytest.approx(case.Re_per_meter, rel=1e-10), \
                f"{name}: Re/m mismatch"

    def test_kinematic_viscosity(self):
        from run_transition_t3 import T3_CASES
        for name, case in T3_CASES.items():
            assert case.nu == pytest.approx(case.mu / case.rho)

    def test_mach_below_incompressible_limit(self):
        """All T3 cases should be M < 0.1 (incompressible)."""
        from run_transition_t3 import T3_CASES
        for name, case in T3_CASES.items():
            assert case.Mach < 0.1, f"{name}: M={case.Mach:.3f} not incompressible"

    def test_experimental_data_present(self):
        """Each case should have digitized experimental Cf(Rex) data."""
        from run_transition_t3 import T3_CASES
        for name, case in T3_CASES.items():
            assert len(case.exp_Rex) > 10, f"{name}: too few exp points"
            assert len(case.exp_Cf) == len(case.exp_Rex), f"{name}: Rex/Cf length mismatch"

    def test_experimental_cf_physical(self):
        """Cf values should be positive and in a physical range."""
        from run_transition_t3 import T3_CASES
        for name, case in T3_CASES.items():
            for cf in case.exp_Cf:
                assert 1e-4 < cf < 0.02, f"{name}: unphysical Cf={cf}"

    def test_higher_tu_earlier_transition(self):
        """Higher Tu should trigger earlier transition (lower Re_x)."""
        from run_transition_t3 import T3_CASES
        assert T3_CASES["T3B"].Re_x_transition_exp < T3_CASES["T3A"].Re_x_transition_exp
        assert T3_CASES["T3A"].Re_x_transition_exp < T3_CASES["T3A-"].Re_x_transition_exp


# =============================================================================
# Analytical Correlation Tests
# =============================================================================

class TestCorrelations:
    """Test analytical flat plate correlations."""

    def test_blasius_at_re_1e5(self):
        """Blasius Cf at Re_x = 1e5 should be ~0.0021."""
        from run_transition_t3 import cf_blasius
        cf = cf_blasius(np.array([1e5]))[0]
        assert cf == pytest.approx(0.664 / math.sqrt(1e5), rel=1e-10)
        assert 0.001 < cf < 0.003

    def test_blasius_decreases_with_rex(self):
        from run_transition_t3 import cf_blasius
        rex = np.array([1e4, 1e5, 1e6])
        cf = cf_blasius(rex)
        assert cf[0] > cf[1] > cf[2]

    def test_turbulent_higher_than_laminar(self):
        """At high Re_x, turbulent Cf > laminar Cf."""
        from run_transition_t3 import cf_blasius, cf_turbulent_power
        rex = np.array([1e6])
        assert cf_turbulent_power(rex)[0] > cf_blasius(rex)[0]

    def test_schlichting_vs_power_law_similar(self):
        """Both turbulent correlations should agree within ~30% at Re=1e6."""
        from run_transition_t3 import cf_turbulent_power, cf_turbulent_schlichting
        rex = np.array([1e6])
        cf_p = cf_turbulent_power(rex)[0]
        cf_s = cf_turbulent_schlichting(rex)[0]
        assert abs(cf_p - cf_s) / cf_p < 0.30

    def test_abu_ghannam_shaw_monotonic(self):
        """Higher Tu -> lower Re_theta,cr (earlier transition)."""
        from run_transition_t3 import re_theta_onset_abu_ghannam_shaw
        re_th_high_tu = re_theta_onset_abu_ghannam_shaw(6.0)
        re_th_low_tu = re_theta_onset_abu_ghannam_shaw(1.0)
        assert re_th_high_tu < re_th_low_tu

    def test_abu_ghannam_shaw_typical_value(self):
        """For Tu=3%, Re_theta,cr should be around 100-300."""
        from run_transition_t3 import re_theta_onset_abu_ghannam_shaw
        re_th = re_theta_onset_abu_ghannam_shaw(3.0)
        assert 50 < re_th < 500

    def test_rex_from_re_theta_blasius_relation(self):
        """Re_x = (Re_theta / 0.664)^2 from Blasius."""
        from run_transition_t3 import re_x_from_re_theta
        re_th = 100.0
        rex = re_x_from_re_theta(re_th)
        assert rex == pytest.approx((100.0 / 0.664) ** 2, rel=1e-6)


# =============================================================================
# SU2 Configuration Generation Tests
# =============================================================================

class TestConfigGeneration:
    """Test that SU2 config files are generated correctly."""

    def test_lm_config_generated(self, tmp_path):
        from run_transition_t3 import T3_CASES, generate_su2_config_lm
        case = T3_CASES["T3A"]
        config_path = generate_su2_config_lm(case, tmp_path)
        assert config_path.exists()
        content = config_path.read_text()
        assert "KIND_TRANS_MODEL = LM" in content
        assert "KIND_TURB_MODEL = SST" in content
        assert "INC_RANS" in content

    def test_lm_config_turbulence_intensity(self, tmp_path):
        from run_transition_t3 import T3_CASES, generate_su2_config_lm
        case = T3_CASES["T3A"]
        config_path = generate_su2_config_lm(case, tmp_path)
        content = config_path.read_text()
        assert f"FREESTREAM_TURBULENCEINTENSITY = {case.Tu_fraction}" in content

    def test_bcm_config_generated(self, tmp_path):
        from run_transition_t3 import T3_CASES, generate_su2_config_bcm
        case = T3_CASES["T3B"]
        config_path = generate_su2_config_bcm(case, tmp_path)
        assert config_path.exists()
        content = config_path.read_text()
        assert "SA_OPTIONS = BCM" in content
        assert "KIND_TURB_MODEL = SA" in content

    def test_fully_turbulent_config_no_transition(self, tmp_path):
        from run_transition_t3 import T3_CASES, generate_su2_config_fully_turbulent
        case = T3_CASES["T3A"]
        config_path = generate_su2_config_fully_turbulent(case, tmp_path)
        content = config_path.read_text()
        assert "KIND_TRANS_MODEL" not in content
        assert "BCM" not in content

    def test_config_uses_inc_rans(self, tmp_path):
        """All configs should use incompressible solver."""
        from run_transition_t3 import (
            T3_CASES, generate_su2_config_lm,
            generate_su2_config_bcm, generate_su2_config_fully_turbulent
        )
        case = T3_CASES["T3A"]
        for gen_fn in [generate_su2_config_lm, generate_su2_config_bcm,
                       generate_su2_config_fully_turbulent]:
            config_path = gen_fn(case, tmp_path / gen_fn.__name__)
            content = config_path.read_text()
            assert "INC_RANS" in content, f"{gen_fn.__name__} not using INC_RANS"

    def test_each_case_generates_different_velocity(self, tmp_path):
        from run_transition_t3 import T3_CASES, generate_su2_config_lm
        velocities = set()
        for name, case in T3_CASES.items():
            config_path = generate_su2_config_lm(case, tmp_path / name)
            content = config_path.read_text()
            assert f"{case.U_inf}" in content
            velocities.add(case.U_inf)
        assert len(velocities) == 3, "All three cases should have different velocities"


# =============================================================================
# Result Parsing Tests
# =============================================================================

class TestResultParsing:
    """Test the Cf extraction and Re_x computation utilities."""

    def test_compute_rex(self):
        from run_transition_t3 import compute_rex
        x = np.array([0.1, 0.5, 1.0])
        U = 5.4
        nu = 1.5e-5
        rex = compute_rex(x, U, nu)
        np.testing.assert_allclose(rex, U * x / nu)

    def test_find_transition_from_synthetic_cf(self):
        """Synthetic Cf with a clear laminar->turbulent transition."""
        from run_transition_t3 import find_transition_location
        x = np.linspace(0.01, 1.5, 500)
        nu = 1.5e-5
        U = 5.4
        # Laminar for x < 0.5, transition around x = 0.5, turbulent after
        rex = U * x / nu
        cf_lam = 0.664 / np.sqrt(rex)
        cf_turb = 0.0592 / rex**0.2
        # Smooth transition via sigmoid
        sigma = 50
        x_tr = 0.5
        blend = 1.0 / (1.0 + np.exp(-sigma * (x - x_tr)))
        cf = cf_lam * (1 - blend) + cf_turb * blend

        result = find_transition_location(x, cf, nu, U)
        assert result is not None
        # Transition should be near x = 0.5
        assert abs(result["x_tr"] - 0.5) < 0.15


# =============================================================================
# Integration Smoke Test
# =============================================================================

class TestDryRun:
    """Test the runner in dry-run mode (no SU2 required)."""

    def test_dry_run_t3a(self):
        from simulations.run_transition_t3 import run_single_case
        result = run_single_case("T3A", "LM", n_iter=100, dry_run=True)
        assert result.get("dry_run") is True
        assert "config_path" in result

    def test_dry_run_t3b_bcm(self):
        from simulations.run_transition_t3 import run_single_case
        result = run_single_case("T3B", "BCM", n_iter=100, dry_run=True)
        assert result.get("dry_run") is True

    def test_dry_run_unknown_case(self):
        from simulations.run_transition_t3 import run_single_case
        result = run_single_case("INVALID", "LM", n_iter=100, dry_run=True)
        assert "error" in result
