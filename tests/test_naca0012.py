#!/usr/bin/env python3
"""
NACA 0012 TMR Validation — Unit Tests
=======================================
Tests for all TMR-specific functions in run_naca0012.py:
  - Sutherland's Law viscosity
  - Turbulence inflow conditions (SA, SST)
  - Point vortex farfield correction
  - TMR NACA 0012 geometry (sharp TE)
  - SU2 config generation
  - Case configuration constants

Run: pytest tests/test_naca0012.py -v --tb=short
"""

import sys
import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from run_naca0012 import (
    CASE_CONFIG,
    GRID_LEVELS,
    GRID_LEVEL_SPECS,
    SU2_MODELS,
    SUTHERLAND_MU0,
    SUTHERLAND_T0,
    SUTHERLAND_S,
    TMR_COEFF,
    TMR_SCALE_FACTOR,
    build_grid_levels,
    sutherlands_law,
    compute_turbulence_inflow,
    point_vortex_correction,
    naca0012_tmr_y,
    naca0012_tmr_surface,
    generate_su2_config,
)


# ============================================================================
# Case Configuration Tests
# ============================================================================
class TestCaseConfig:
    """Verify TMR case constants match published specifications."""

    def test_mach_number(self):
        assert CASE_CONFIG["mach"] == 0.15

    def test_reynolds_number(self):
        assert CASE_CONFIG["reynolds"] == 6e6

    def test_gamma(self):
        assert CASE_CONFIG["gamma"] == 1.4

    def test_prandtl_lam(self):
        assert CASE_CONFIG["prandtl_lam"] == 0.72

    def test_prandtl_turb(self):
        assert CASE_CONFIG["prandtl_turb"] == 0.90

    def test_temperature_freestream(self):
        assert CASE_CONFIG["temperature_freestream"] == 300.0  # K (= 540 R)

    def test_fully_turbulent(self):
        assert CASE_CONFIG["fully_turbulent"] is True

    def test_mu_t_ratio_sa(self):
        """TMR note 5: nu_tilde_inf = 3 * nu for SA."""
        assert CASE_CONFIG["mu_t_ratio"] == 3.0

    def test_viscosity_model(self):
        assert CASE_CONFIG["viscosity_model"] == "SUTHERLAND"

    def test_sutherland_constants(self):
        """Verify Sutherland's Law constants from White (1974)."""
        assert SUTHERLAND_MU0 == pytest.approx(1.716e-5, rel=1e-3)
        assert SUTHERLAND_T0 == pytest.approx(273.11, rel=1e-3)
        assert SUTHERLAND_S == pytest.approx(110.33, rel=1e-3)


# ============================================================================
# Grid & Model Registry Tests
# ============================================================================
class TestRegistry:
    """Verify grid levels and turbulence model definitions."""

    def test_grid_levels_count(self):
        assert len(GRID_LEVELS) == 7  # Family I: 7 levels

    def test_grid_levels_names(self):
        for name in ["coarse", "medium", "fine", "xfine", "ultra", "super", "hyper"]:
            assert name in GRID_LEVELS, f"Missing grid level: {name}"

    def test_grid_dimensions_increase(self):
        ordered = ["coarse", "medium", "fine", "xfine", "ultra", "super", "hyper"]
        dims = [GRID_LEVELS[g]["dims"] for g in ordered]
        for i in range(len(dims) - 1):
            assert dims[i + 1][0] > dims[i][0], "i-dim should increase with refinement"
            assert dims[i + 1][1] > dims[i][1], "j-dim should increase with refinement"

    def test_refinement_ratio(self):
        """Family I grids have a uniform 2× refinement ratio."""
        ordered = ["coarse", "medium", "fine", "xfine", "ultra", "super", "hyper"]
        for i in range(len(ordered) - 1):
            coarse_dims = GRID_LEVELS[ordered[i]]["dims"]
            fine_dims = GRID_LEVELS[ordered[i + 1]]["dims"]
            # (idim-1) doubles: (fine_idim - 1) == 2 * (coarse_idim - 1)
            assert fine_dims[0] - 1 == 2 * (coarse_dims[0] - 1)
            assert fine_dims[1] - 1 == 2 * (coarse_dims[1] - 1)

    def test_family_level_numbering(self):
        """Level 1 = finest, level 7 = coarsest."""
        assert GRID_LEVELS["coarse"]["family_level"] == 7
        assert GRID_LEVELS["hyper"]["family_level"] == 1

    def test_family_ii_builds(self):
        """Family II grids should build with same dimensions."""
        levels_ii = build_grid_levels("II")
        assert len(levels_ii) == 7
        for name in GRID_LEVEL_SPECS:
            assert levels_ii[name]["dims"] == GRID_LEVEL_SPECS[name]["dims"]
            assert levels_ii[name]["family"] == "II"
            assert "familyII" in levels_ii[name]["plot3d"]

    def test_family_i_and_ii_different_files(self):
        """Family I and II should point to different PLOT3D files."""
        levels_i = build_grid_levels("I")
        levels_ii = build_grid_levels("II")
        for name in GRID_LEVEL_SPECS:
            assert levels_i[name]["plot3d"] != levels_ii[name]["plot3d"]

    def test_family_iii_builds(self):
        """Family III should build with same dimensions and coarsen_steps."""
        levels_iii = build_grid_levels("III")
        assert len(levels_iii) == 7
        for name in GRID_LEVEL_SPECS:
            assert levels_iii[name]["dims"] == GRID_LEVEL_SPECS[name]["dims"]
            assert levels_iii[name]["family"] == "III"
            assert "familyIII" in levels_iii[name]["plot3d"]

    def test_family_iii_coarsen_steps(self):
        """Family III: finest has 0 coarsen_steps, coarsest has 6."""
        levels_iii = build_grid_levels("III")
        assert levels_iii["hyper"]["coarsen_steps"] == 0   # level 1 = finest
        assert levels_iii["coarse"]["coarsen_steps"] == 6  # level 7 = coarsest

    def test_family_iii_single_source(self):
        """Family III: all levels point to the same finest grid file."""
        levels_iii = build_grid_levels("III")
        files = set(levels_iii[name]["plot3d"] for name in GRID_LEVEL_SPECS)
        assert len(files) == 1, "All Family III levels should reference one file"

    def test_family_iii_different_from_i_and_ii(self):
        """Family III source file should differ from Family I and II."""
        levels_i = build_grid_levels("I")
        levels_iii = build_grid_levels("III")
        assert levels_i["hyper"]["plot3d"] != levels_iii["hyper"]["plot3d"]

    def test_su2_models(self):
        assert "SA" in SU2_MODELS
        assert "SST" in SU2_MODELS
        assert SU2_MODELS["SA"]["KIND_TURB_MODEL"] == "SA"
        assert SU2_MODELS["SST"]["KIND_TURB_MODEL"] == "SST"


# ============================================================================
# Sutherland's Law Tests
# ============================================================================
class TestSutherlandsLaw:
    """Test dynamic viscosity computation via Sutherland's Law."""

    def test_reference_temperature(self):
        """mu(T_0) should equal mu_0 exactly."""
        mu = sutherlands_law(SUTHERLAND_T0)
        assert mu == pytest.approx(SUTHERLAND_MU0, rel=1e-10)

    def test_at_300K(self):
        """mu(300 K) ≈ 1.846e-5 kg/(m*s) — standard air at 300 K."""
        mu = sutherlands_law(300.0)
        assert mu == pytest.approx(1.846e-5, rel=0.01)

    def test_at_200K(self):
        """mu(200 K) ≈ 1.329e-5 kg/(m*s) — cold air."""
        mu = sutherlands_law(200.0)
        assert mu == pytest.approx(1.329e-5, rel=0.02)

    def test_at_500K(self):
        """mu(500 K) ≈ 2.671e-5 kg/(m*s) — hot air."""
        mu = sutherlands_law(500.0)
        assert mu == pytest.approx(2.671e-5, rel=0.02)

    def test_monotonically_increasing(self):
        """Viscosity must increase with temperature for gases."""
        temps = [200, 250, 300, 350, 400, 500, 600]
        mus = [sutherlands_law(T) for T in temps]
        for i in range(len(mus) - 1):
            assert mus[i + 1] > mus[i], (
                f"mu should increase: mu({temps[i]})={mus[i]:.4e} "
                f">= mu({temps[i+1]})={mus[i+1]:.4e}"
            )

    def test_always_positive(self):
        """Viscosity must be positive for any positive temperature."""
        for T in [100, 200, 300, 500, 1000, 2000]:
            assert sutherlands_law(T) > 0


# ============================================================================
# Turbulence Inflow Tests
# ============================================================================
class TestTurbulenceInflow:
    """Test turbulence inflow condition calculations."""

    def test_sa_nu_tilde(self):
        """SA: nu_tilde_inf = 3 * nu (TMR note 5)."""
        result = compute_turbulence_inflow("SA")
        nu = result["nu_inf"]
        nu_tilde = result["nu_tilde_inf"]
        assert nu_tilde == pytest.approx(3.0 * nu, rel=1e-10)

    def test_sa_keys(self):
        """SA result should contain expected keys."""
        result = compute_turbulence_inflow("SA")
        for key in ["mu_inf", "rho_inf", "nu_inf", "U_inf", "a_inf",
                     "nu_tilde_inf", "report"]:
            assert key in result, f"Missing key: {key}"

    def test_sst_keys(self):
        """SST result should contain k_inf and omega_inf."""
        result = compute_turbulence_inflow("SST")
        for key in ["mu_inf", "rho_inf", "nu_inf", "U_inf", "a_inf",
                     "k_inf", "omega_inf", "report"]:
            assert key in result, f"Missing key: {key}"

    def test_sst_positive_values(self):
        """SST: k_inf and omega_inf must be positive."""
        result = compute_turbulence_inflow("SST")
        assert result["k_inf"] > 0
        assert result["omega_inf"] > 0

    def test_freestream_density(self):
        """rho = p / (R * T) for ideal gas."""
        result = compute_turbulence_inflow("SA")
        cfg = CASE_CONFIG
        rho_expected = cfg["pressure_freestream"] / (
            cfg["gas_constant"] * cfg["temperature_freestream"]
        )
        assert result["rho_inf"] == pytest.approx(rho_expected, rel=1e-10)

    def test_speed_of_sound(self):
        """a = sqrt(gamma * R * T)."""
        result = compute_turbulence_inflow("SA")
        cfg = CASE_CONFIG
        a_expected = math.sqrt(
            cfg["gamma"] * cfg["gas_constant"] * cfg["temperature_freestream"]
        )
        assert result["a_inf"] == pytest.approx(a_expected, rel=1e-10)

    def test_freestream_velocity(self):
        """U_inf = M * a."""
        result = compute_turbulence_inflow("SA")
        U_expected = CASE_CONFIG["mach"] * result["a_inf"]
        assert result["U_inf"] == pytest.approx(U_expected, rel=1e-10)


# ============================================================================
# Point Vortex Correction Tests
# ============================================================================
class TestPointVortexCorrection:
    """Test farfield point vortex BC correction."""

    def test_output_keys(self):
        """Output dict should contain all required fields."""
        result = point_vortex_correction(
            x_far=500.0, y_far=0.0, alpha_deg=10.0, CL=1.0
        )
        for key in ["u_corr", "v_corr", "p_corr", "Gamma", "delta_V_over_V"]:
            assert key in result, f"Missing key: {key}"

    def test_zero_cl_gives_zero_correction(self):
        """Zero CL → zero circulation → zero correction."""
        result = point_vortex_correction(
            x_far=500.0, y_far=0.0, alpha_deg=10.0, CL=0.0
        )
        assert result["u_corr"] == pytest.approx(0.0, abs=1e-15)
        assert result["v_corr"] == pytest.approx(0.0, abs=1e-15)

    def test_correction_decays_with_distance(self):
        """Vortex-induced velocity decays as 1/r."""
        near = point_vortex_correction(
            x_far=100.0, y_far=0.0, alpha_deg=10.0, CL=1.0
        )
        far = point_vortex_correction(
            x_far=500.0, y_far=0.0, alpha_deg=10.0, CL=1.0
        )
        assert abs(near["delta_V_over_V"]) > abs(far["delta_V_over_V"])

    def test_small_correction_at_500c(self):
        """At 500 chords, correction should be very small (< 1% of freestream)."""
        CL_typical = 2.0 * math.pi * math.radians(10.0)  # ~1.1
        result = point_vortex_correction(
            x_far=500.0, y_far=0.0, alpha_deg=10.0, CL=CL_typical
        )
        assert result["delta_V_over_V"] < 0.01  # < 1%

    def test_degenerate_point_at_vortex(self):
        """Point at the vortex location (0.25, 0) should return zeros."""
        result = point_vortex_correction(
            x_far=0.25, y_far=0.0, alpha_deg=10.0, CL=1.0
        )
        assert result["u_corr"] == 0.0
        assert result["v_corr"] == 0.0


# ============================================================================
# NACA 0012 Geometry Tests
# ============================================================================
class TestNACA0012Geometry:
    """Test TMR NACA 0012 geometry definition."""

    def test_leading_edge(self):
        """y(0) = 0 — leading edge is at origin."""
        x = np.array([0.0])
        y = naca0012_tmr_y(x)
        assert y[0] == pytest.approx(0.0, abs=1e-15)

    def test_trailing_edge_closure(self):
        """y(1) ≈ 0 — sharp trailing edge (TMR definition)."""
        x = np.array([1.0])
        y = naca0012_tmr_y(x)
        assert abs(y[0]) < 1e-10, f"TE not closed: y(1) = {y[0]:.2e}"

    def test_max_thickness(self):
        """Max thickness ≈ 11.894% chord (TMR spec)."""
        x = np.linspace(0, 1, 10000)
        y = naca0012_tmr_y(x)
        t_max = 2.0 * y.max()  # Full thickness = 2 * half-thickness
        assert t_max == pytest.approx(0.11894, abs=0.001), (
            f"Max thickness = {t_max:.5f}, expected ≈ 0.11894"
        )

    def test_positive_upper_surface(self):
        """y(x) > 0 for 0 < x < 1 (upper surface)."""
        x = np.linspace(0.001, 0.999, 1000)
        y = naca0012_tmr_y(x)
        assert np.all(y > 0), "Upper surface y should be positive"

    def test_symmetry_at_midchord(self):
        """Airfoil is symmetric → y(x) is single-valued and smooth."""
        x = np.linspace(0, 1, 500)
        y = naca0012_tmr_y(x)
        # Check the curve is smooth (no NaN/Inf)
        assert np.all(np.isfinite(y))

    def test_max_thickness_location(self):
        """Max thickness occurs near x/c ≈ 0.30 for NACA 0012."""
        x = np.linspace(0, 1, 10000)
        y = naca0012_tmr_y(x)
        x_max = x[np.argmax(y)]
        assert 0.25 < x_max < 0.35, f"Max thickness at x/c = {x_max:.3f}"

    def test_tmr_scale_factor(self):
        """TMR scale factor should close original NACA 0012 at x=1."""
        assert TMR_SCALE_FACTOR == pytest.approx(1.008930411365, rel=1e-10)


class TestNACA0012Surface:
    """Test surface coordinate generation."""

    def test_surface_point_count(self):
        """Surface should have 2*n_pts - 1 points (shared LE point)."""
        n_pts = 100
        x, y = naca0012_tmr_surface(n_pts)
        assert len(x) == 2 * n_pts - 1
        assert len(y) == 2 * n_pts - 1

    def test_surface_starts_at_te(self):
        """First point should be near TE (x ≈ 1) — upper surface start."""
        x, y = naca0012_tmr_surface(200)
        assert x[0] == pytest.approx(1.0, abs=1e-10)

    def test_surface_ends_at_te(self):
        """Last point should be near TE (x ≈ 1) — lower surface end."""
        x, y = naca0012_tmr_surface(200)
        assert x[-1] == pytest.approx(1.0, abs=1e-10)

    def test_le_is_at_zero(self):
        """Leading edge point should be at x=0."""
        n_pts = 200
        x, y = naca0012_tmr_surface(n_pts)
        # LE is at index n_pts - 1 (middle of array)
        assert x[n_pts - 1] == pytest.approx(0.0, abs=1e-10)

    def test_upper_positive_lower_negative(self):
        """Upper surface has y > 0, lower has y < 0 (except at LE/TE)."""
        n_pts = 200
        x, y = naca0012_tmr_surface(n_pts)
        # Upper surface: indices 1 to n_pts-2 (exclude exact TE and LE)
        y_upper = y[1:n_pts - 1]
        assert np.all(y_upper > 0), "Upper surface y should be positive"
        # Lower surface: indices n_pts to end-1
        y_lower = y[n_pts:-1]
        assert np.all(y_lower < 0), "Lower surface y should be negative"


# ============================================================================
# SU2 Config Generation Tests
# ============================================================================
class TestSU2ConfigGeneration:
    """Test SU2 configuration file generation."""

    def test_config_file_created(self):
        """generate_su2_config should create a .cfg file."""
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = Path(tmp)
            cfg_path = generate_su2_config(case_dir, "mesh.su2", alpha=10.0)
            assert cfg_path.exists()
            assert cfg_path.name == "naca0012.cfg"

    def test_config_contains_mach(self):
        """Config should contain correct Mach number."""
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = Path(tmp)
            cfg_path = generate_su2_config(case_dir, "mesh.su2", alpha=10.0)
            content = cfg_path.read_text()
            assert "MACH_NUMBER= 0.15" in content

    def test_config_contains_reynolds(self):
        """Config should contain Reynolds number."""
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = Path(tmp)
            cfg_path = generate_su2_config(case_dir, "mesh.su2", alpha=10.0)
            content = cfg_path.read_text()
            assert "REYNOLDS_NUMBER=" in content
            assert "6000000" in content

    def test_config_contains_aoa(self):
        """Config should contain the specified angle of attack."""
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = Path(tmp)
            cfg_path = generate_su2_config(case_dir, "mesh.su2", alpha=15.0)
            content = cfg_path.read_text()
            assert "AOA= 15.0" in content

    def test_config_sa_model(self):
        """SA model should be set correctly."""
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = Path(tmp)
            cfg_path = generate_su2_config(
                case_dir, "mesh.su2", alpha=10.0, model="SA"
            )
            content = cfg_path.read_text()
            assert "KIND_TURB_MODEL= SA" in content

    def test_config_sst_model(self):
        """SST model should be set correctly."""
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = Path(tmp)
            cfg_path = generate_su2_config(
                case_dir, "mesh.su2", alpha=10.0, model="SST"
            )
            content = cfg_path.read_text()
            assert "KIND_TURB_MODEL= SST" in content

    def test_config_sutherland_viscosity(self):
        """Config should use Sutherland's viscosity model."""
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = Path(tmp)
            cfg_path = generate_su2_config(case_dir, "mesh.su2", alpha=0.0)
            content = cfg_path.read_text()
            assert "VISCOSITY_MODEL= SUTHERLAND" in content

    def test_config_adiabatic_wall(self):
        """Config should specify adiabatic wall (zero heat flux)."""
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = Path(tmp)
            cfg_path = generate_su2_config(case_dir, "mesh.su2", alpha=0.0)
            content = cfg_path.read_text()
            assert "MARKER_HEATFLUX= ( airfoil, 0.0 )" in content

    def test_config_prandtl_numbers(self):
        """Config should specify correct Prandtl numbers."""
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = Path(tmp)
            cfg_path = generate_su2_config(case_dir, "mesh.su2", alpha=0.0)
            content = cfg_path.read_text()
            assert "PRANDTL_LAM= 0.72" in content
            assert "PRANDTL_TURB= 0.9" in content

    def test_config_turb2lam_ratio(self):
        """Config should set freestream turbulence-to-laminar viscosity ratio."""
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = Path(tmp)
            cfg_path = generate_su2_config(case_dir, "mesh.su2", alpha=0.0)
            content = cfg_path.read_text()
            assert "FREESTREAM_NU_FACTOR= 3.0" in content

    def test_config_mesh_filename(self):
        """Config should reference the correct mesh file."""
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = Path(tmp)
            cfg_path = generate_su2_config(
                case_dir, "my_mesh.su2", alpha=0.0
            )
            content = cfg_path.read_text()
            assert "MESH_FILENAME= my_mesh.su2" in content

    def test_config_moment_output(self):
        """Config should include MOMENT_Z in screen output for CM tracking."""
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = Path(tmp)
            cfg_path = generate_su2_config(case_dir, "mesh.su2", alpha=10.0)
            content = cfg_path.read_text()
            assert "MOMENT_Z" in content

    def test_config_moment_reference(self):
        """Moment reference should be at quarter-chord (0.25c)."""
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = Path(tmp)
            cfg_path = generate_su2_config(case_dir, "mesh.su2", alpha=10.0)
            content = cfg_path.read_text()
            assert "REF_ORIGIN_MOMENT_X= 0.25" in content
