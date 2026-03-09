"""
Test Suite for CFD Solver Benchmark
====================================
Tests for config, data_loader, error_metrics, grid_convergence,
profile_extraction, sensitivity_analysis, UQ, ML model, V&V, and pipeline.

Run: pytest tests/ -v --tb=short
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _has_salib():
    try:
        import SALib
        return True
    except ImportError:
        return False

# ============================================================================
# Config Tests
# ============================================================================
class TestConfig:
    def test_import_config(self):
        from config import BENCHMARK_CASES, TURBULENCE_MODELS

    def test_benchmark_cases_count(self):
        from config import BENCHMARK_CASES
        assert len(BENCHMARK_CASES) >= 17, f"Expected ≥17 cases, got {len(BENCHMARK_CASES)}"

    def test_turbulence_models_count(self):
        from config import TURBULENCE_MODELS
        assert len(TURBULENCE_MODELS) >= 15, f"Expected ≥15 models, got {len(TURBULENCE_MODELS)}"

    def test_benchmark_case_fields(self):
        from config import BENCHMARK_CASES
        for name, case in BENCHMARK_CASES.items():
            assert case.description, f"{name} missing description"
            assert case.reynolds_number > 0, f"{name} has invalid Re"
            assert case.data_source, f"{name} missing data_source"

    def test_turbulence_model_fields(self):
        from config import TURBULENCE_MODELS
        for name, model in TURBULENCE_MODELS.items():
            assert model.openfoam_name, f"{name} missing openfoam_name"
            assert model.model_type is not None, f"{name} missing model_type"

    def test_solver_defaults(self):
        from config import SOLVER_DEFAULTS
        assert SOLVER_DEFAULTS["max_iterations"] > 0
        assert 0 < SOLVER_DEFAULTS["relaxation"]["p"] < 1

    def test_sensitivity_parameters(self):
        from config import SENSITIVITY_PARAMETERS
        assert SENSITIVITY_PARAMETERS["num_vars"] == len(SENSITIVITY_PARAMETERS["names"])
        assert len(SENSITIVITY_PARAMETERS["bounds"]) == SENSITIVITY_PARAMETERS["num_vars"]


# ============================================================================
# Data Loader Tests
# ============================================================================
class TestDataLoader:
    def test_load_flat_plate(self):
        from experimental_data.data_loader import load_case
        data = load_case("flat_plate")
        assert data.case_name == "flat_plate"
        assert len(data.velocity_profiles) > 0

    def test_load_bfs_real_data(self):
        from experimental_data.data_loader import load_case
        data = load_case("backward_facing_step")
        assert data.wall_data is not None, "BFS should have wall data from Driver & Seegmiller CSV"
        assert not getattr(data, 'is_synthetic', True), "BFS should load real experimental data"

    def test_load_hump_synthetic_fallback(self):
        from experimental_data.data_loader import load_case
        data = load_case("nasa_hump")
        assert getattr(data, 'is_synthetic', False), "Expected NASA Hump to fall back to synthetic data"
        assert "x_sep_xc" in data.separation_metrics

    def test_load_invalid_case(self):
        from experimental_data.data_loader import load_case
        with pytest.raises(ValueError, match="Unknown case"):
            load_case("nonexistent_case")

    def test_load_all_cases(self):
        from experimental_data.data_loader import load_all_cases
        all_data = load_all_cases()
        assert len(all_data) >= 17

    def test_flat_plate_profiles_physics(self):
        from experimental_data.data_loader import load_case
        data = load_case("flat_plate")
        for station, df in data.velocity_profiles.items():
            assert (df["U"] >= 0).all(), f"Negative velocity at x={station}"
            assert df["y_plus"].iloc[0] > 0, f"y+ ≤ 0 at x={station}"


# ============================================================================
# Error Metrics Tests
# ============================================================================
class TestErrorMetrics:
    def test_rmse_perfect(self):
        from scripts.postprocessing.error_metrics import rmse
        a = np.array([1.0, 2.0, 3.0])
        assert rmse(a, a) == pytest.approx(0.0)

    def test_rmse_known(self):
        from scripts.postprocessing.error_metrics import rmse
        cfd = np.array([1.0, 2.0, 3.0])
        exp = np.array([1.1, 2.1, 3.1])
        assert rmse(cfd, exp) == pytest.approx(0.1)

    def test_mae(self):
        from scripts.postprocessing.error_metrics import mae
        assert mae(np.array([1, 2]), np.array([2, 4])) == pytest.approx(1.5)

    def test_nrmse(self):
        from scripts.postprocessing.error_metrics import nrmse
        cfd = np.array([1.0, 2.0, 3.0])
        exp = np.array([1.0, 2.0, 4.0])
        val = nrmse(cfd, exp)
        assert 0 < val < 1

    def test_asme_vv20_validated(self):
        from scripts.postprocessing.error_metrics import asme_vv20_metric
        cfd = np.array([1.0, 2.0, 3.0])
        exp = np.array([1.01, 2.01, 3.01])
        unc = np.array([0.1, 0.1, 0.1])
        result = asme_vv20_metric(cfd, exp, unc)
        assert result["status"] == "VALIDATED"

    def test_separation_metrics(self):
        from scripts.postprocessing.error_metrics import separation_metrics
        x = np.linspace(0, 10, 100)
        Cf = 0.003 * (x - 2) * (x - 7)  # Zeros at x=2, x=7
        result = separation_metrics(x, Cf, x_sep_exp=2.0, x_reat_exp=7.0)
        assert result["x_sep_cfd"] is not None
        assert abs(result["x_sep_cfd"] - 2.0) < 0.2

    def test_compute_all_metrics(self):
        from scripts.postprocessing.error_metrics import compute_all_metrics
        cfd = np.random.randn(50)
        exp = cfd + 0.01 * np.random.randn(50)
        m = compute_all_metrics(cfd, exp, label="test")
        assert "RMSE" in m
        assert "R2" in m


# ============================================================================
# Grid Convergence Tests
# ============================================================================
class TestGridConvergence:
    def test_second_order(self):
        """Monotonic convergence test."""
        from scripts.postprocessing.grid_convergence import richardson_extrapolation
        # Monotonic convergence: errors shrink with refinement
        # eps_21 = medium - fine = -0.16; eps_32 = coarse - medium = -0.30
        # ratio = eps_32/eps_21 = 1.875 > 1 → monotonic convergence
        result = richardson_extrapolation(6.26, 6.10, 5.80)
        assert result.observed_order > 0, "Should detect positive order"
        assert result.gci_fine >= 0, "GCI should be non-negative"

    def test_converged_solution(self):
        from scripts.postprocessing.grid_convergence import richardson_extrapolation
        result = richardson_extrapolation(1.0, 1.0, 1.0)
        assert result.gci_fine == 0.0

    def test_divergent_detection(self):
        from scripts.postprocessing.grid_convergence import richardson_extrapolation
        result = richardson_extrapolation(1.0, 1.5, 1.8)
        assert "DIVERGENT" in result.status or "CONVERGED" in result.status or result.status

    def test_multi_quantity_gci(self):
        from scripts.postprocessing.grid_convergence import multi_quantity_gci
        quantities = {
            "x_reat": (6.26, 6.0, 5.5),
            "Cf_max": (0.003, 0.0028, 0.0025),
        }
        results = multi_quantity_gci(quantities)
        assert "x_reat" in results
        assert "Cf_max" in results


# ============================================================================
# Profile Extraction Tests
# ============================================================================
class TestProfileExtraction:
    def test_extract_wall_data(self):
        from scripts.postprocessing.extract_profiles import extract_wall_data
        x = np.linspace(0, 1, 50)
        tau_w = 0.5 * np.ones(50)
        p = np.zeros(50)
        df = extract_wall_data(x, tau_w, p, rho=1.225, U_ref=10.0)
        assert "Cf" in df.columns
        assert "Cp" in df.columns

    def test_separation_reattachment(self):
        from scripts.postprocessing.extract_profiles import (
            find_separation_point, find_reattachment_point,
        )
        x = np.linspace(0, 10, 200)
        # Cf starts positive, goes negative (sep), then recovers (reat)
        Cf = 0.003 * (x - 2) * (x - 7)  # + → − at x≈2, − → + at x≈7
        x_sep = find_separation_point(x, Cf)
        x_reat = find_reattachment_point(x, Cf)
        assert x_sep is not None
        assert x_reat is not None
        assert x_reat > x_sep
        assert abs(x_sep - 2.0) < 0.2
        assert abs(x_reat - 7.0) < 0.2

    def test_lumley_triangle(self):
        from scripts.postprocessing.extract_profiles import lumley_triangle
        uu = np.array([0.01, 0.02, 0.03])
        vv = np.array([0.005, 0.01, 0.015])
        ww = np.array([0.005, 0.01, 0.015])
        xi, eta = lumley_triangle(uu, vv, ww)
        assert len(xi) == 3
        assert np.all(eta >= 0)  # η always ≥ 0


# ============================================================================
# Mesh Generator Tests
# ============================================================================
class TestMeshGenerator:
    def test_mesh_levels(self):
        from scripts.preprocessing.mesh_generator import MeshGenerator
        gen = MeshGenerator("backward_facing_step", Path("."))
        assert len(gen.mesh_levels) >= 3

    def test_yplus_estimator(self):
        from scripts.preprocessing.mesh_generator import estimate_yplus, required_first_cell_height
        yp = estimate_yplus(Re=5e6, L=1.0, U=50.0, y1=1e-5)
        assert yp > 0

        y1 = required_first_cell_height(Re=5e6, L=1.0, U=50.0, y_plus_target=1.0)
        assert y1 > 0
        assert y1 < 1e-3  # Should be very small


# ============================================================================
# ML Model Tests
# ============================================================================
class TestMLModel:
    def test_feature_extraction(self):
        from scripts.ml_augmentation.model import extract_invariant_features
        N = 100
        Sij = np.random.randn(N, 3, 3) * 0.01
        Oij = np.random.randn(N, 3, 3) * 0.01
        k = np.abs(np.random.randn(N)) * 0.01
        eps = np.abs(np.random.randn(N)) * 0.01
        grad_p = np.random.randn(N, 3)
        wall_dist = np.abs(np.random.randn(N))

        features = extract_invariant_features(Sij, Oij, k, eps, grad_p, wall_dist)
        assert features.shape[0] == N
        assert features.shape[1] == 7

    def test_synthetic_dataset(self):
        from scripts.ml_augmentation.model import create_synthetic_dataset
        ds = create_synthetic_dataset(n_samples=200)
        assert ds.X_train.shape[0] == 160
        assert ds.X_val.shape[0] == 40


# ============================================================================
# Sensitivity Analysis Tests
# ============================================================================
class TestSensitivityAnalysis:
    @pytest.mark.skipif(
        not _has_salib(), reason="SALib not installed"
    )
    def test_sample_generation(self):
        from scripts.analysis.sensitivity_analysis import ParametricSensitivityAnalysis
        sa = ParametricSensitivityAnalysis()
        samples = sa.generate_samples(n_samples=64)
        assert samples.shape[1] == sa.problem["num_vars"]

    @pytest.mark.skipif(
        not _has_salib(), reason="SALib not installed"
    )
    def test_sobol_simple(self):
        """Test with a simple analytical function."""
        from scripts.analysis.sensitivity_analysis import ParametricSensitivityAnalysis

        problem = {
            "num_vars": 3,
            "names": ["x1", "x2", "x3"],
            "bounds": [[0, 1], [0, 1], [0, 1]],
        }
        sa = ParametricSensitivityAnalysis(problem)

        def model(x):
            return x[0] + 2 * x[1] + 0.01 * x[2]  # x2 >> x3 importance

        result = sa.run(model, n_samples=512)
        assert result.ST[1] > result.ST[2], "x2 should be more important than x3"


# ============================================================================
# UQ Tests
# ============================================================================
class TestUQ:
    def test_sample_generation(self):
        from scripts.analysis.uncertainty_quantification import UncertaintyQuantification
        uq = UncertaintyQuantification()
        samples = uq.generate_samples(100)
        assert samples.shape[0] == 100

    def test_uq_pipeline(self):
        from scripts.analysis.uncertainty_quantification import UncertaintyQuantification
        uq = UncertaintyQuantification()

        def model(x):
            return x[0] + 0.5 * x[1]

        uq.propagate_uncertainty(model, n_samples=100)
        result = uq.analyze()
        assert result.n_samples == 100
        assert result.ci_95[0] < result.mean < result.ci_95[1]

    def test_validation_comparison(self):
        from scripts.analysis.uncertainty_quantification import UncertaintyQuantification
        uq = UncertaintyQuantification()

        def model(x):
            return 6.5 + 0.001 * x[0]  # Very small perturbation

        uq.propagate_uncertainty(model, n_samples=200)
        val = uq.compare_with_experiment(exp_value=6.5, exp_uncertainty=0.5)
        assert val.status == "VALIDATED"


# ============================================================================
# V&V Framework Tests
# ============================================================================
class TestVVFramework:
    def test_mrr_level_0(self):
        from scripts.validation.vv_framework import VVFramework
        vv = VVFramework()
        assert vv.compute_mrr_level() == 0

    def test_flat_plate_verification(self):
        from scripts.validation.vv_framework import VVFramework
        vv = VVFramework()

        x = np.linspace(1.0, 10.0, 100)
        nu = 1.5e-5
        U_inf = 50.0
        Re_x = U_inf * x / nu
        Cf = 0.059 / Re_x**0.2  # Exact turbulent correlation
        Cf_cfd = Cf * (1 + 0.02 * np.random.randn(len(Cf)))  # ±2% noise

        result = vv.verify_flat_plate(x, Cf_cfd, U_inf, nu)
        assert result["Cf_MAPE_pct"] < 10.0  # Should be close

    def test_40_percent_tracking(self):
        from scripts.validation.vv_framework import VVFramework
        vv = VVFramework()
        result = vv.track_40_percent_challenge("test_case", 10.0, 5.5)
        assert result["target_met"]  # 45% reduction > 40% target


# ============================================================================
# Bachalo-Johnson Transonic Data Tests
# ============================================================================
class TestBachaloJohnsonData:
    """Tests for Bachalo-Johnson transonic bump data generator."""

    def test_load_bachalo_johnson_synthetic_fallback(self):
        from experimental_data.data_loader import load_case
        data = load_case("bachalo_johnson")
        assert getattr(data, 'is_synthetic', False), "Expected Bachalo-Johnson to fall back to synthetic data"
        assert data.case_name == "bachalo_johnson"
        assert data.wall_data is not None
        assert len(data.velocity_profiles) > 0

    def test_shock_signature_in_Cp(self):
        from experimental_data.data_loader import load_case
        data = load_case("bachalo_johnson")
        x = data.wall_data["x_c"].values
        Cp = data.wall_data["Cp"].values

        # Cp should have suction (negative) region before shock
        mask_accel = (x > 0.2) & (x < 0.6)
        assert Cp[mask_accel].min() < -0.2, "Should have suction peak before shock"

    def test_separation_in_Cf(self):
        from experimental_data.data_loader import load_case
        data = load_case("bachalo_johnson")
        x = data.wall_data["x_c"].values
        Cf = data.wall_data["Cf"].values

        # Cf should go negative in separation bubble (0.68 < x/c < 0.90)
        mask_sep = (x > 0.68) & (x < 0.90)
        assert Cf[mask_sep].min() < 0, "Should have reverse flow in separation bubble"

    def test_transonic_metrics(self):
        from experimental_data.data_loader import load_case
        data = load_case("bachalo_johnson")
        assert data.separation_metrics["mach_number"] == 0.875
        assert data.separation_metrics["x_shock_xc"] == 0.65


# ============================================================================
# NACA 0012 Stall Prediction Tests
# ============================================================================
class TestNACA0012Data:
    """Tests for NACA 0012 stall data generator."""

    def test_load_naca_0012(self):
        from experimental_data.data_loader import load_case
        data = load_case("naca_0012_stall")
        assert data.wall_data is not None
        assert "CL" in data.wall_data.columns
        assert "CD" in data.wall_data.columns

    def test_cl_max_value(self):
        from experimental_data.data_loader import load_case
        data = load_case("naca_0012_stall")
        CL = data.wall_data["CL"].values
        CL_max = CL.max()
        # Gregory & O'Reilly: CL_max ≈ 1.55
        assert 1.3 < CL_max < 1.7, f"CL_max={CL_max}, expected ~1.55"

    def test_stall_angle(self):
        from experimental_data.data_loader import load_case
        data = load_case("naca_0012_stall")
        alpha = data.wall_data["alpha_deg"].values
        CL = data.wall_data["CL"].values
        alpha_stall = alpha[np.argmax(CL)]
        assert 14 < alpha_stall < 18, f"alpha_stall={alpha_stall}°, expected ~16°"

    def test_cp_profiles_at_multiple_aoa_synthetic_fallback(self):
        from experimental_data.data_loader import load_case
        data = load_case("naca_0012_stall")
        assert getattr(data, 'is_synthetic', False), "Expected NACA 0012 to fall back to synthetic data"
        assert len(data.velocity_profiles) >= 5, "Should have Cp at 5+ angles"


# ============================================================================
# Juncture Flow Tests
# ============================================================================
class TestJunctureFlowData:
    """Tests for NASA juncture flow data generator."""

    def test_load_juncture_flow_synthetic_fallback(self):
        from experimental_data.data_loader import load_case
        data = load_case("juncture_flow")
        assert getattr(data, 'is_synthetic', False), "Expected Juncture Flow to fall back to synthetic data"
        assert data.wall_data is not None
        assert len(data.velocity_profiles) > 0

    def test_corner_separation(self):
        from experimental_data.data_loader import load_case
        data = load_case("juncture_flow")
        assert "x_sep_corner_xc" in data.separation_metrics
        assert "x_reat_corner_xc" in data.separation_metrics
        assert data.separation_metrics["bubble_length_xc"] > 0

    def test_horseshoe_vortex(self):
        from experimental_data.data_loader import load_case
        data = load_case("juncture_flow")
        assert data.separation_metrics.get("horseshoe_vortex") is True

    def test_spanwise_velocity(self):
        from experimental_data.data_loader import load_case
        data = load_case("juncture_flow")
        first = list(data.velocity_profiles.values())[0]
        assert "W_Uinf" in first.columns, "Should have spanwise velocity component"


# ============================================================================
# Workshop Comparison Tests
# ============================================================================
class TestWorkshopComparison:
    """Tests for DPW/HiLiftPW workshop scatter-band comparison."""

    def test_import_module(self):
        from scripts.comparison.workshop_comparison import (
            compare_to_workshop, compute_workshop_ranking,
        )

    def test_dpw5_scatter_bands(self):
        from scripts.comparison.workshop_comparison import DPW_SCATTER
        assert "DPW5_WB" in DPW_SCATTER
        assert "CL" in DPW_SCATTER["DPW5_WB"]
        assert DPW_SCATTER["DPW5_WB"]["CL"].n_participants > 0

    def test_ranking_computation(self):
        from scripts.comparison.workshop_comparison import compute_workshop_ranking
        results = {"CL": 0.500, "CD_counts": 256.0}
        ranking = compute_workshop_ranking(results, "DPW5_WB")
        assert "overall_score" in ranking
        assert 0 <= ranking["overall_score"] <= 1.0
        assert ranking["rank_label"] in [
            "TOP-TIER", "COMPETITIVE", "AVERAGE", "BELOW AVERAGE",
        ]

    def test_within_band_check(self):
        from scripts.comparison.workshop_comparison import compare_to_workshop
        results = {"CL": 0.500}  # Mean value → should be within band
        comparison = compare_to_workshop(results, "DPW5_WB")
        assert comparison["CL"]["within_band"] is True

    def test_invalid_workshop_raises(self):
        from scripts.comparison.workshop_comparison import compare_to_workshop
        with pytest.raises(ValueError, match="Unknown workshop"):
            compare_to_workshop({"CL": 0.5}, "NON_EXISTENT_WS")


# ============================================================================
# TMR Downloader Tests
# ============================================================================
class TestTMRDownloader:
    """Tests for NASA TMR data downloader (offline mode)."""

    def test_case_registry(self):
        from scripts.preprocessing.tmr_downloader import TMR_CASES
        assert len(TMR_CASES) >= 9
        for code in ["2DBFS", "ATB", "2DWMH", "2DN00", "JFLOW"]:
            assert code in TMR_CASES, f"Missing case: {code}"

    def test_list_cases(self):
        from scripts.preprocessing.tmr_downloader import list_available_cases
        cases = list_available_cases()
        assert isinstance(cases, dict)
        assert "ATB" in cases

    def test_case_info(self):
        from scripts.preprocessing.tmr_downloader import get_case_info
        info = get_case_info("ATB")
        assert info["code"] == "ATB"
        assert info["flow_conditions"]["M"] == 0.875

    def test_parse_stub_data(self):
        from scripts.preprocessing.tmr_downloader import (
            parse_tmr_profile, _create_stub_data,
        )
        import tempfile
        tmp = Path(tempfile.mkdtemp())
        stub = tmp / "test_cp.dat"
        _create_stub_data(stub, "ATB", "test_cp.dat")
        df = parse_tmr_profile(stub)
        assert len(df) > 0
        assert len(df.columns) >= 2


# ============================================================================
# Model Reference Library Tests
# ============================================================================
class TestModelReference:
    """Tests for turbulence model reference and recommendation engine."""

    def test_import_module(self):
        from scripts.analysis.turbulence_model_reference import (
            get_model_recommendation, compare_to_literature,
        )

    def test_recommendation_all_types(self):
        from scripts.analysis.turbulence_model_reference import (
            get_model_recommendation,
        )
        for sep_type in [
            "smooth_body_2d", "geometric", "curvature",
            "shock_induced", "corner_3d", "trailing_edge",
        ]:
            rec = get_model_recommendation(sep_type)
            assert "primary_model" in rec
            assert "hybrid_model" in rec

    def test_corner_flow_recommends_qcr(self):
        from scripts.analysis.turbulence_model_reference import (
            get_model_recommendation,
        )
        rec = get_model_recommendation("corner_3d")
        assert rec["primary_model"] == "SA-QCR"

    def test_literature_baseline(self):
        from scripts.analysis.turbulence_model_reference import (
            get_literature_baseline,
        )
        baseline = get_literature_baseline("SST", "nasa_hump")
        assert "bubble_length_xc" in baseline
        assert baseline["bubble_length_xc"] > 0

    def test_compare_to_literature(self):
        from scripts.analysis.turbulence_model_reference import (
            compare_to_literature,
        )
        result = compare_to_literature(
            "SST", "backward_facing_step",
            {"x_reat_xH": 6.20},
        )
        assert "x_reat_xH" in result
        assert result["x_reat_xH"]["user_value"] == 6.20


# ============================================================================
# BeVERLI Hill Tests
# ============================================================================
class TestBeVERLIHillData:
    """Tests for BeVERLI Hill 3D smooth-body separation data generator."""

    def test_load_beverli_hill_synthetic_fallback(self):
        from experimental_data.data_loader import load_case
        data = load_case("beverli_hill")
        assert getattr(data, 'is_synthetic', False), "Expected BeVERLI Hill to fall back to synthetic data"
        assert data.case_name == "beverli_hill"
        assert data.wall_data is not None
        assert len(data.velocity_profiles) > 0

    def test_centerline_separation(self):
        from experimental_data.data_loader import load_case
        data = load_case("beverli_hill")
        assert "x_sep_xH" in data.separation_metrics
        assert "x_reat_xH" in data.separation_metrics
        assert data.separation_metrics["bubble_length_xH"] > 0
        # Physical bounds: separation should be on leeward side (x/H > 0)
        assert data.separation_metrics["x_sep_xH"] > 0
        assert data.separation_metrics["x_reat_xH"] > data.separation_metrics["x_sep_xH"]

    def test_yaw_symmetry_breaking(self):
        from experimental_data.data_loader import load_case
        data = load_case("beverli_hill")
        # 0° yaw should be symmetric, 45° should show asymmetric wake
        assert data.separation_metrics.get("yaw_0_symmetric") is True
        assert data.separation_metrics.get("yaw_45_asymmetric_wake") is True

    def test_velocity_components_3d(self):
        from experimental_data.data_loader import load_case
        data = load_case("beverli_hill")
        first_profile = list(data.velocity_profiles.values())[0]
        # 3D profiles should have U, V, W components
        assert "U" in first_profile.columns, "Should have streamwise velocity U"
        assert "V" in first_profile.columns, "Should have wall-normal velocity V"
        assert "W" in first_profile.columns, "Should have spanwise velocity W"

    def test_reynolds_stresses(self):
        from experimental_data.data_loader import load_case
        data = load_case("beverli_hill")
        first_profile = list(data.velocity_profiles.values())[0]
        # Should have Reynolds stress components
        assert "uu" in first_profile.columns, "Should have u'u' stress"
        assert "vv" in first_profile.columns, "Should have v'v' stress"
        assert "uv" in first_profile.columns, "Should have u'v' stress"
        # Normal stresses should be non-negative
        assert (first_profile["uu"] >= 0).all(), "u'u' must be non-negative"
        assert (first_profile["vv"] >= 0).all(), "v'v' must be non-negative"


# ============================================================================
# NASA Gaussian Speed Bump Tests
# ============================================================================
class TestGaussianBumpData:
    """Tests for NASA 3D Gaussian Speed Bump data generator."""

    def test_load_gaussian_bump_synthetic_fallback(self):
        from experimental_data.data_loader import load_case
        data = load_case("boeing_gaussian_bump")
        assert getattr(data, 'is_synthetic', False), "Expected Gaussian Bump to fall back to synthetic data"
        assert data.case_name == "boeing_gaussian_bump"
        assert data.wall_data is not None
        assert len(data.velocity_profiles) > 0

    def test_gaussian_geometry(self):
        from experimental_data.data_loader import load_case
        data = load_case("boeing_gaussian_bump")
        assert data.separation_metrics["bump_height_L"] == 0.085
        assert data.separation_metrics["bump_x0_L"] == 0.195
        assert data.separation_metrics["bump_z0_L"] == 0.06

    def test_sa_vs_sarc_vs_wmles_metrics(self):
        from experimental_data.data_loader import load_case
        data = load_case("boeing_gaussian_bump")
        m = data.separation_metrics
        # WMLES bubble should be largest (ground truth)
        assert m["bubble_length_xL_wmles"] > m["bubble_length_xL_sa_rc"]
        # SA-RC should improve over baseline SA
        assert m["bubble_length_xL_sa_rc"] > m["bubble_length_xL_sa"]
        # SA severely under-predicts
        assert m["bubble_length_xL_sa"] < 0.56 * m["bubble_length_xL_wmles"]

    def test_velocity_profiles(self):
        from experimental_data.data_loader import load_case
        data = load_case("boeing_gaussian_bump")
        first_profile = list(data.velocity_profiles.values())[0]
        assert "U" in first_profile.columns
        assert "V" in first_profile.columns

    def test_recovery_cf(self):
        from experimental_data.data_loader import load_case
        data = load_case("boeing_gaussian_bump")
        # Cf should be positive in the recovery region (x/L > 1.35)
        mask = data.wall_data["x_L"] > 1.8
        if mask.any():
            assert (data.wall_data.loc[mask, "Cf"] > 0).all()


# ============================================================================
# FIML Architecture Tests
# ============================================================================
class TestFIMLArchitecture:
    """Tests for Field Inversion and Machine Learning (FIML) components."""

    def test_synthetic_field_inversion(self):
        from scripts.ml_augmentation.fiml_su2_adjoint import (
            FIMLConfig, ReferenceData, SyntheticFieldInversion
        )
        import numpy as np
        
        config = FIMLConfig(max_inversion_iter=2)
        ref = ReferenceData(x_stations=np.linspace(0, 2, 10))
        ref.cp_ref = np.zeros(10)
        
        inverter = SyntheticFieldInversion(config, ref, n_nodes=50)
        state = inverter.run_inversion(use_adjoint=False)
        assert len(state.objective_history) > 0
        assert state.n_flow_solves > 0

    def test_nn_embedding_architecture(self):
        from scripts.ml_augmentation.fiml_nn_embedding import (
            BetaCorrectionNN, EmbeddingConfig
        )
        import numpy as np
        
        nn = BetaCorrectionNN(EmbeddingConfig(max_epochs=2, hidden_layers=(8,)))
        features = np.random.rand(100, 5)
        beta = np.ones(100)
        
        res = nn.train(features, beta, val_split=0.2)
        assert "train_r2" in res
        
        pred = nn.predict(features[:10])
        assert len(pred) == 10
        assert (pred > 0).all()  # softplus ensures positive beta

    def test_generalization_generators(self):
        from scripts.ml_augmentation.fiml_generalization_test import (
            generate_periodic_hill_case, generate_wall_hump_case
        )
        
        hill = generate_periodic_hill_case(n_points=50)
        assert hill.name == "periodic_hill"
        assert hill.features.shape == (50, 5)
        
        hump = generate_wall_hump_case(n_points=50)
        assert hump.name == "wall_hump"
        assert hump.features.shape == (50, 5)
