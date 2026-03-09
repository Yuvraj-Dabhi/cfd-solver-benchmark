#!/usr/bin/env python3
"""
Tests for Additional Validation Cases
=========================================
Tests new benchmark case entries, data loaders, runner utilities,
and cross-dataset BFS validation.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =========================================================================
# Config Entry Tests
# =========================================================================
class TestNewBenchmarkCases:
    """Tests for new BENCHMARK_CASES entries."""

    def test_axisymmetric_jet_in_config(self):
        """axisymmetric_jet should be a T2 free-shear case."""
        from config import BENCHMARK_CASES, CaseTier, SeparationCategory
        case = BENCHMARK_CASES["axisymmetric_jet"]
        assert case.tier == CaseTier.T2_INTERMEDIATE
        assert case.category == SeparationCategory.FREE_SHEAR
        assert case.mach_number == 0.5

    def test_bump_3d_channel_in_config(self):
        """bump_3d_channel should be a T3 smooth-body 3D case."""
        from config import BENCHMARK_CASES, CaseTier, SeparationCategory
        case = BENCHMARK_CASES["bump_3d_channel"]
        assert case.tier == CaseTier.T3_COMPLEX_3D
        assert case.category == SeparationCategory.SMOOTH_BODY_3D
        assert case.mach_number == 0.2

    def test_nasa_crm_in_config(self):
        """nasa_crm should be a T3 smooth-body 3D case at M=0.85."""
        from config import BENCHMARK_CASES, CaseTier
        case = BENCHMARK_CASES["nasa_crm"]
        assert case.tier == CaseTier.T3_COMPLEX_3D
        assert case.mach_number == 0.85
        assert case.reynolds_number == 5e6

    def test_free_shear_category_exists(self):
        """FREE_SHEAR category should exist in SeparationCategory."""
        from config import SeparationCategory
        assert hasattr(SeparationCategory, "FREE_SHEAR")

    def test_benchmark_cases_count_at_least_20(self):
        """Should have at least 20 benchmark cases."""
        from config import BENCHMARK_CASES
        assert len(BENCHMARK_CASES) >= 20

    def test_crm_has_dpw_grid_family(self):
        """CRM should define DPW grid levels."""
        from config import BENCHMARK_CASES
        case = BENCHMARK_CASES["nasa_crm"]
        assert "L1_finest" in case.mesh_levels
        assert case.mesh_levels["L1_finest"] > 100_000_000

    def test_jet_has_reference_papers(self):
        """Jet case should cite Bridges & Wernet and Witze."""
        from config import BENCHMARK_CASES
        case = BENCHMARK_CASES["axisymmetric_jet"]
        papers = " ".join(case.reference_papers)
        assert "Bridges" in papers
        assert "Witze" in papers


# =========================================================================
# Data Loader Tests
# =========================================================================
class TestJetDataLoader:
    """Tests for axisymmetric jet data loader."""

    def test_load_jet_data(self):
        """Jet data should load with correct structure."""
        from experimental_data.axisymmetric_jet.jet_data_loader import load_jet_data
        data = load_jet_data()
        assert data.case_name == "axisymmetric_jet"
        assert data.wall_data is not None
        assert "x_D" in data.wall_data.columns
        assert "Uc_Uj" in data.wall_data.columns

    def test_jet_radial_profiles(self):
        """Jet should have radial profiles at multiple stations."""
        from experimental_data.axisymmetric_jet.jet_data_loader import load_jet_data
        data = load_jet_data()
        assert len(data.velocity_profiles) == 5
        assert 8.0 in data.velocity_profiles
        profile = data.velocity_profiles[8.0]
        assert "r_D" in profile.columns
        assert "U_Uj" in profile.columns

    def test_jet_potential_core(self):
        """Potential core should be ~6 x/D."""
        from experimental_data.axisymmetric_jet.jet_data_loader import load_jet_data
        data = load_jet_data()
        assert 4 <= data.separation_metrics["potential_core_xD"] <= 8

    def test_jet_spreading_rate(self):
        """Spreading rate should be ~0.094 (consensus value)."""
        from experimental_data.axisymmetric_jet.jet_data_loader import load_jet_data
        data = load_jet_data()
        assert 0.08 <= data.separation_metrics["spreading_rate"] <= 0.11


class TestCRMDataLoader:
    """Tests for NASA CRM DPW data loader."""

    def test_load_crm_data(self):
        """CRM data should load with drag polar."""
        from experimental_data.nasa_crm.crm_data_loader import load_crm_data
        data = load_crm_data()
        assert data.case_name == "nasa_crm"
        assert data.wall_data is not None
        assert "CL" in data.wall_data.columns
        assert "CD_counts" in data.wall_data.columns
        assert len(data.wall_data) == 10  # 10 alpha values

    def test_crm_cp_sections(self):
        """CRM should have Cp sections at 6 spanwise stations."""
        from experimental_data.nasa_crm.crm_data_loader import load_crm_data
        data = load_crm_data()
        assert len(data.velocity_profiles) == 6
        assert 0.283 in data.velocity_profiles
        cp_section = data.velocity_profiles[0.283]
        assert "Cp_upper" in cp_section.columns

    def test_crm_design_cl(self):
        """Design CL should be 0.500."""
        from experimental_data.nasa_crm.crm_data_loader import load_crm_data
        data = load_crm_data()
        assert abs(data.separation_metrics["CL_design"] - 0.500) < 0.001

    def test_crm_dpw_scatter(self):
        """DPW scatter should be defined."""
        from experimental_data.nasa_crm.crm_data_loader import load_crm_data
        data = load_crm_data()
        assert data.separation_metrics["dpw5_CD_scatter_counts"] == 4.0


# =========================================================================
# Runner Utility Tests
# =========================================================================
class TestBFSCrossDataset:
    """Tests for BFS cross-dataset runner."""

    def test_bfs_datasets_defined(self):
        """Both BFS datasets should be defined."""
        from simulations.run_bfs_validation import DATASETS
        assert "driver_seegmiller" in DATASETS
        assert "kim_et_al" in DATASETS

    def test_bfs_cross_comparison(self):
        """Cross-dataset comparison should return results for all models."""
        from simulations.run_bfs_validation import run_full_comparison
        results = run_full_comparison()
        assert "SA" in results
        assert "SST" in results
        assert "KE" in results
        # Each model should have results for both datasets
        for model_results in results.values():
            assert "driver_seegmiller" in model_results
            assert "kim_et_al" in model_results

    def test_bfs_sst_within_uncertainty(self):
        """SST should predict BFS reattachment within 5% of experiment."""
        from simulations.run_bfs_validation import compare_datasets_for_model
        results = compare_datasets_for_model("SST")
        assert abs(results["driver_seegmiller"]["error_pct"]) < 5.0

    def test_bfs_ke_underpredicts(self):
        """k-ε should systematically underpredict x_R."""
        from simulations.run_bfs_validation import compare_datasets_for_model
        results = compare_datasets_for_model("KE")
        assert results["driver_seegmiller"]["error_pct"] < 0  # Underprediction


class TestJetRunner:
    """Tests for axisymmetric jet runner utilities."""

    def test_jet_config_generation(self):
        """SU2 config should generate for jet case."""
        from simulations.run_axisymmetric_jet import generate_su2_config
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = generate_su2_config(Path(tmpdir))
            assert cfg.exists()
            content = cfg.read_text()
            assert "RANS" in content
            assert "SA" in content

    def test_jet_centerline_decay_error(self):
        """Centerline decay error should compute valid metrics."""
        from simulations.run_axisymmetric_jet import (
            generate_synthetic_prediction,
            compute_centerline_decay_error,
        )
        pred = generate_synthetic_prediction("SA")
        metrics = compute_centerline_decay_error(pred["x_D"], pred["Uc_Uj"])
        assert metrics["rmse_Uc"] > 0
        assert metrics["potential_core_xD"] > 0

    def test_jet_ke_round_jet_anomaly(self):
        """k-ε should show the round-jet anomaly (short core)."""
        from simulations.run_axisymmetric_jet import (
            generate_synthetic_prediction,
            compute_centerline_decay_error,
        )
        sa_pred = generate_synthetic_prediction("SA")
        ke_pred = generate_synthetic_prediction("KE")
        sa_metrics = compute_centerline_decay_error(sa_pred["x_D"], sa_pred["Uc_Uj"])
        ke_metrics = compute_centerline_decay_error(ke_pred["x_D"], ke_pred["Uc_Uj"])
        # k-ε should have shorter core (more negative error)
        assert ke_metrics["core_length_error"] < sa_metrics["core_length_error"]


class TestBump3DRunner:
    """Tests for 3D bump-in-channel runner."""

    def test_bump3d_config_generation(self):
        """SU2 config with SA-RC should include RC option."""
        from simulations.run_bump_3d_channel import generate_su2_config
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = generate_su2_config(Path(tmpdir), model="SA-RC")
            content = cfg.read_text()
            assert "SA_OPTIONS= RC" in content

    def test_bump3d_model_comparison(self):
        """Model comparison should show SA-RC improvement over SA."""
        from simulations.run_bump_3d_channel import compare_models
        results = compare_models()
        assert results["SA-RC"]["bubble_len"] > results["SA"]["bubble_len"]
        assert abs(results["SA-RC"]["bubble_error_pct"]) < abs(results["SA"]["bubble_error_pct"])


class TestCRMRunner:
    """Tests for NASA CRM runner scaffolding."""

    def test_crm_config_generation(self):
        """CRM config should generate with correct Mach and Re."""
        from simulations.run_nasa_crm import generate_su2_config
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = generate_su2_config(Path(tmpdir))
            content = cfg.read_text()
            assert "MACH_NUMBER= 0.85" in content
            assert "REYNOLDS_NUMBER= 5000000" in content

    def test_crm_slurm_generation(self):
        """SLURM script should be generated."""
        from simulations.run_nasa_crm import generate_slurm_script
        with tempfile.TemporaryDirectory() as tmpdir:
            script = generate_slurm_script(Path(tmpdir), "crm_SA.cfg", n_procs=256)
            content = script.read_text()
            assert "256" in content
            assert "SU2_CFD" in content

    def test_crm_dpw_scatter_check(self):
        """DPW scatter check should identify within-band results."""
        from simulations.run_nasa_crm import check_within_dpw_scatter
        result = check_within_dpw_scatter(CL=0.505, CD_counts=254.0)
        assert result["CL_within"]
        assert result["CD_within"]

    def test_crm_dpw_scatter_out_of_band(self):
        """Should flag results outside DPW scatter band."""
        from simulations.run_nasa_crm import check_within_dpw_scatter
        result = check_within_dpw_scatter(CL=0.55, CD_counts=270.0)
        assert not result["CL_within"]
        assert not result["CD_within"]
