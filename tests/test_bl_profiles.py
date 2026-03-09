"""
Tests for Boundary-Layer Profile Analysis and Data Sources
===========================================================
Validates:
- BL profile analyzer integral quantities (δ*, θ, H)
- Log-law deviation detection
- TKE budget computation
- Experimental profile registry
- ML pipeline cross-validation
- Expanded DATA_SOURCES and EXPERIMENTAL_REFERENCES

Run: pytest tests/test_bl_profiles.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestBLProfileAnalyzer:
    """Tests for boundary-layer profile analysis."""

    def _make_bl_profile(self, n=100, U_inf=10.0, delta=0.01, nu=1.5e-5):
        """Create a synthetic turbulent BL profile."""
        y = np.linspace(1e-6, 5 * delta, n)
        # Power-law profile: U/U_inf = (y/delta)^(1/7)
        U = U_inf * np.minimum((y / delta) ** (1.0 / 7), 1.0)
        return y, U

    def test_import(self):
        from scripts.analysis.bl_profile_analyzer import BLProfileAnalyzer
        analyzer = BLProfileAnalyzer()
        assert analyzer.nu > 0

    def test_shape_factor_attached(self):
        from scripts.analysis.bl_profile_analyzer import BLProfileAnalyzer
        analyzer = BLProfileAnalyzer(nu=1.5e-5, U_inf=10.0)
        y, U = self._make_bl_profile()
        report = analyzer.analyze_station(y, U)
        # For 1/7th power law: H ≈ 1.28-1.3
        assert 1.0 < report.shape_factor_H < 2.0, f"H={report.shape_factor_H}"
        assert not report.is_separated

    def test_separation_detection(self):
        from scripts.analysis.bl_profile_analyzer import BLProfileAnalyzer
        analyzer = BLProfileAnalyzer(nu=1.5e-5, U_inf=10.0)
        y = np.linspace(1e-6, 0.01, 100)
        # Clear reversed flow near wall: velocity decreasing from wall
        U = np.zeros_like(y)
        U[:30] = -1.0  # Reversed flow near wall
        U[30:] = 10.0 * ((y[30:] - y[30]) / (y[-1] - y[30])) ** (1.0 / 7)
        report = analyzer.analyze_station(y, U)
        # Shape factor should detect separation even if Cf estimation is ambiguous
        assert "SEPARATED" in report.separation_status

    def test_clauser_beta_zpg(self):
        from scripts.analysis.bl_profile_analyzer import BLProfileAnalyzer
        analyzer = BLProfileAnalyzer(nu=1.5e-5, U_inf=10.0)
        y, U = self._make_bl_profile()
        report = analyzer.analyze_station(y, U, dpdx=0.0)
        assert report.clauser_beta == 0.0

    def test_clauser_beta_apg(self):
        from scripts.analysis.bl_profile_analyzer import BLProfileAnalyzer
        analyzer = BLProfileAnalyzer(nu=1.5e-5, U_inf=10.0)
        y, U = self._make_bl_profile()
        report = analyzer.analyze_station(y, U, dpdx=100.0)
        assert report.clauser_beta > 0  # APG → β > 0

    def test_inner_scaling(self):
        from scripts.analysis.bl_profile_analyzer import BLProfileAnalyzer
        analyzer = BLProfileAnalyzer(nu=1.5e-5, U_inf=10.0)
        y, U = self._make_bl_profile()
        report = analyzer.analyze_station(y, U)
        assert report.inner_scaled is not None
        assert "y_plus" in report.inner_scaled.columns
        assert "U_plus" in report.inner_scaled.columns

    def test_multi_station_analysis(self):
        from scripts.analysis.bl_profile_analyzer import BLProfileAnalyzer
        analyzer = BLProfileAnalyzer(nu=1.5e-5, U_inf=10.0)
        y, U = self._make_bl_profile()
        stations = {
            "x/c=0.5": {"y": y, "U": U},
            "x/c=1.0": {"y": y, "U": U * 0.8},
        }
        report = analyzer.analyze_case(stations, case_name="test")
        assert len(report.stations) == 2
        assert "FULLY ATTACHED" in report.overall_status

    def test_tke_budget(self):
        from scripts.analysis.bl_profile_analyzer import BLProfileAnalyzer
        analyzer = BLProfileAnalyzer(nu=1.5e-5, U_inf=10.0)
        y, U = self._make_bl_profile()
        k = 0.01 * U ** 2
        omega = 100 * np.ones_like(y)
        report = analyzer.analyze_station(y, U, k=k, omega=omega)
        assert report.tke_budget is not None
        assert "P_k" in report.tke_budget.columns
        assert "epsilon" in report.tke_budget.columns


class TestComputeHelpers:
    """Test standalone helper functions."""

    def test_shape_factor(self):
        from scripts.analysis.bl_profile_analyzer import compute_shape_factor
        y = np.linspace(0, 0.01, 100)
        U = 10.0 * (y / 0.01) ** (1.0 / 7)
        H = compute_shape_factor(y, U, U_inf=10.0)
        assert 1.0 < H < 2.0

    def test_clauser_beta(self):
        from scripts.analysis.bl_profile_analyzer import compute_clauser_beta
        beta = compute_clauser_beta(delta_star=0.001, tau_w=5.0, dpdx=1000)
        assert beta > 0


class TestExperimentalProfiles:
    """Tests for experimental profile registry."""

    def test_import(self):
        from scripts.preprocessing.experimental_profiles import PROFILE_REGISTRY
        assert len(PROFILE_REGISTRY) >= 4

    def test_nasa_hump_stations(self):
        from scripts.preprocessing.experimental_profiles import get_profile_stations
        reg = get_profile_stations("nasa_hump")
        assert len(reg.profiles) >= 7
        assert reg.separation_metrics["x_sep_xc"] == 0.665

    def test_bfs_stations(self):
        from scripts.preprocessing.experimental_profiles import get_profile_stations
        reg = get_profile_stations("backward_facing_step")
        assert reg.separation_metrics["x_reat_xH"] == 6.26

    def test_get_specific_station(self):
        from scripts.preprocessing.experimental_profiles import get_reference_data
        profile = get_reference_data("backward_facing_step", "x/H = 6")
        assert profile is not None
        assert "U/U_ref" in profile.quantities

    def test_list_all(self):
        from scripts.preprocessing.experimental_profiles import list_all_profiles
        result = list_all_profiles()
        assert "nasa_hump" in result
        assert "flat_plate" in result

    def test_invalid_case(self):
        from scripts.preprocessing.experimental_profiles import get_profile_stations
        with pytest.raises(ValueError):
            get_profile_stations("nonexistent")


class TestExpandedDataSources:
    """Verify expanded DATA_SOURCES and EXPERIMENTAL_REFERENCES."""

    def test_data_sources_expanded(self):
        from config import DATA_SOURCES
        assert "nparc_wind_us" in DATA_SOURCES
        assert "ercoftac_dns_data" in DATA_SOURCES
        assert "tmr_experiments" in DATA_SOURCES

    def test_jhtdb_has_datasets(self):
        from config import DATA_SOURCES
        assert "datasets" in DATA_SOURCES["jhtdb"]

    def test_experimental_references(self):
        from config import EXPERIMENTAL_REFERENCES
        assert "nasa_hump" in EXPERIMENTAL_REFERENCES
        assert "backward_facing_step" in EXPERIMENTAL_REFERENCES
        assert "naca_0012" in EXPERIMENTAL_REFERENCES
        assert "flat_plate" in EXPERIMENTAL_REFERENCES

    def test_hump_stations(self):
        from config import EXPERIMENTAL_REFERENCES
        hump = EXPERIMENTAL_REFERENCES["nasa_hump"]
        assert len(hump["profile_stations_xc"]) >= 7
        assert 0.65 in hump["profile_stations_xc"]

    def test_bfs_ref_has_reat(self):
        from config import EXPERIMENTAL_REFERENCES
        bfs = EXPERIMENTAL_REFERENCES["backward_facing_step"]
        assert bfs["key_metrics"]["x_reat_xH"] == 6.26


class TestMLPipeline:
    """Tests for ML training-validation pipeline."""

    def test_import(self):
        from scripts.ml_augmentation.train_validate_pipeline import MLTrainValidatePipeline
        pipeline = MLTrainValidatePipeline()
        assert pipeline.n_features == 7

    def test_generate_data(self):
        from scripts.ml_augmentation.train_validate_pipeline import MLTrainValidatePipeline
        pipeline = MLTrainValidatePipeline()
        data = pipeline.generate_case_data("flat_plate", n_samples=50)
        assert data.features.shape == (50, 7)
        assert data.targets.shape == (50, 1)

    def test_cross_validation(self):
        from scripts.ml_augmentation.train_validate_pipeline import MLTrainValidatePipeline
        pipeline = MLTrainValidatePipeline()
        results = pipeline.run_cross_validation(
            case_names=["flat_plate", "backward_facing_step", "nasa_hump"],
            n_samples_per_case=50,
        )
        assert len(results.fold_results) == 3
        assert results.overall_mape >= 0

    def test_train_all(self):
        from scripts.ml_augmentation.train_validate_pipeline import MLTrainValidatePipeline
        pipeline = MLTrainValidatePipeline()
        metrics = pipeline.train_all(
            case_names=["flat_plate", "backward_facing_step"],
            n_samples_per_case=50,
        )
        assert "r2" in metrics
        assert "rmse" in metrics


class TestExpandedTMRDownloader:
    """Verify new TMR cases in downloader."""

    def test_bump_case_exists(self):
        from scripts.preprocessing.tmr_downloader import TMR_CASES
        assert "BUMP" in TMR_CASES

    def test_hjet_case_exists(self):
        from scripts.preprocessing.tmr_downloader import TMR_CASES
        assert "HJET" in TMR_CASES

    def test_aswbli_case_exists(self):
        from scripts.preprocessing.tmr_downloader import TMR_CASES
        assert "ASWBLI" in TMR_CASES
        assert TMR_CASES["ASWBLI"].flow_conditions["M"] == 2.85

    def test_total_cases_12_plus(self):
        from scripts.preprocessing.tmr_downloader import TMR_CASES
        assert len(TMR_CASES) >= 12
