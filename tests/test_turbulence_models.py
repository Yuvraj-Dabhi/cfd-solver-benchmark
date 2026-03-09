#!/usr/bin/env python3
"""
Tests for Turbulence Model Expansion (Component 1)
=====================================================
Tests SA-BCM config entry, transition model config generation,
RSM config generation, and hybrid RANS-LES resolution estimation.
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestSABCMConfig:
    """Tests for SA-BCM model entry in config.py."""

    def test_sa_bcm_in_config(self):
        """SA-BCM should be present in TURBULENCE_MODELS."""
        from config import TURBULENCE_MODELS
        assert "SA-BCM" in TURBULENCE_MODELS

    def test_sa_bcm_is_transition_type(self):
        """SA-BCM should be classified as a transition model."""
        from config import TURBULENCE_MODELS, ModelType
        model = TURBULENCE_MODELS["SA-BCM"]
        assert model.model_type == ModelType.TRANSITION

    def test_sa_bcm_has_su2_name(self):
        """SA-BCM should have SU2 solver name."""
        from config import TURBULENCE_MODELS
        model = TURBULENCE_MODELS["SA-BCM"]
        assert model.su2_name == "SA_BCM"

    def test_sa_bcm_has_expected_performance(self):
        """SA-BCM should have expected performance for T3A and NACA0012."""
        from config import TURBULENCE_MODELS
        model = TURBULENCE_MODELS["SA-BCM"]
        assert "T3A" in model.expected_performance
        assert "NACA0012_alpha0" in model.expected_performance

    def test_model_count_at_least_15(self):
        """Should have at least 15 turbulence models after adding SA-BCM."""
        from config import TURBULENCE_MODELS
        assert len(TURBULENCE_MODELS) >= 15


class TestSATransitionConfigGeneration:
    """Tests for SA transition model config generators."""

    def test_bcm_config_generation(self):
        """BCM config should be generated with correct solver settings."""
        from scripts.models.sa_transition import (
            TransitionCaseConfig,
            generate_su2_config_bcm,
        )

        case = TransitionCaseConfig(name="test", U_inf=50.0, Tu_percent=3.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = generate_su2_config_bcm(case, Path(tmpdir))
            assert config_path.exists()

            content = config_path.read_text()
            assert "SA" in content
            assert "BCM" in content
            assert "INC_RANS" in content

    def test_sa_aft_config_generation(self):
        """SA-AFT config should be generated correctly."""
        from scripts.models.sa_transition import (
            TransitionCaseConfig,
            generate_su2_config_sa_aft,
        )

        case = TransitionCaseConfig(name="test_aft", U_inf=50.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = generate_su2_config_sa_aft(case, Path(tmpdir))
            assert config_path.exists()

            content = config_path.read_text()
            assert "SA" in content
            assert "CROSSFLOW" in content

    def test_fully_turbulent_baseline(self):
        """Fully turbulent baseline should not have transition keywords."""
        from scripts.models.sa_transition import (
            TransitionCaseConfig,
            generate_su2_config_fully_turbulent,
        )

        case = TransitionCaseConfig(name="test_ft", U_inf=50.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = generate_su2_config_fully_turbulent(case, Path(tmpdir))
            content = config_path.read_text()
            assert "KIND_TURB_MODEL= SA" in content
            assert "BCM" not in content
            assert "TRANS_MODEL" not in content

    def test_compare_transition_models(self):
        """compare_transition_models should generate configs for all models."""
        from scripts.models.sa_transition import (
            TransitionCaseConfig,
            compare_transition_models,
        )

        case = TransitionCaseConfig(name="test_compare", U_inf=50.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            configs = compare_transition_models(case, Path(tmpdir))
            assert len(configs) == 3
            assert "BCM" in configs
            assert "SA_AFT" in configs
            assert "fully_turbulent" in configs
            for path in configs.values():
                assert path.exists()


class TestRSMConfigGeneration:
    """Tests for RSM OpenFOAM config generation."""

    def test_lrr_config_generation(self):
        """LRR RSM config should generate all required files."""
        from scripts.models.rsm_runner import generate_openfoam_rsm

        with tempfile.TemporaryDirectory() as tmpdir:
            files = generate_openfoam_rsm(
                "nasa_hump", "medium", "LRR", output_dir=Path(tmpdir)
            )
            assert "momentumTransport" in files
            assert "fvSchemes" in files
            assert "fvSolution" in files
            assert "controlDict" in files

            # Check LRR specifics
            content = Path(files["momentumTransport"]).read_text()
            assert "LRR" in content
            assert "RAS" in content

    def test_ssg_config_generation(self):
        """SSG RSM config should generate all required files."""
        from scripts.models.rsm_runner import generate_openfoam_rsm

        with tempfile.TemporaryDirectory() as tmpdir:
            files = generate_openfoam_rsm(
                "nasa_hump", "medium", "SSG", output_dir=Path(tmpdir)
            )
            content = Path(files["momentumTransport"]).read_text()
            assert "SSG" in content

    def test_invalid_case_raises(self):
        """Unknown case name should raise ValueError."""
        from scripts.models.rsm_runner import generate_openfoam_rsm
        with pytest.raises(ValueError, match="Unknown case"):
            generate_openfoam_rsm("nonexistent_case")

    def test_invalid_variant_raises(self):
        """Unknown RSM variant should raise ValueError."""
        from scripts.models.rsm_runner import generate_openfoam_rsm
        with pytest.raises(ValueError, match="Unknown RSM variant"):
            generate_openfoam_rsm("nasa_hump", rsm_variant="INVALID")


class TestHybridRANSLES:
    """Tests for hybrid RANS-LES helpers."""

    def test_les_requirements_estimation(self):
        """estimate_les_requirements should return physically reasonable values."""
        from scripts.models.hybrid_rans_les import estimate_les_requirements

        reqs = estimate_les_requirements(Re=936_000, L_ref=0.420, U_ref=34.6)
        assert reqs["kolmogorov_eta"] > 0
        assert reqs["delta_nu"] > 0
        assert reqs["dx_plus_50"] > 0
        assert reqs["n_cells_estimate"] > 0
        assert reqs["dt_cfl1"] > 0

    def test_les_resolution_check_adequate(self):
        """Very fine mesh should be marked as adequate."""
        from scripts.models.hybrid_rans_les import check_les_resolution

        res = check_les_resolution(
            dx=0.0001, dy_wall=1e-6, dz=0.00005,
            Re=100_000, U_ref=10.0, L_ref=1.0,
        )
        # With such a fine mesh, should be adequate
        assert isinstance(res.adequate, bool)
        assert res.delta_plus_x > 0
        assert res.cfl_estimate > 0

    def test_ddes_config_generation(self):
        """DDES config should generate all required OpenFOAM files."""
        from scripts.models.hybrid_rans_les import generate_ddes_config

        with tempfile.TemporaryDirectory() as tmpdir:
            files = generate_ddes_config(
                "nasa_hump", base_model="SA", output_dir=Path(tmpdir)
            )
            assert "momentumTransport" in files
            content = Path(files["momentumTransport"]).read_text()
            assert "DDES" in content
            assert "LES" in content

    def test_sas_config_generation(self):
        """SAS config should generate momentumTransport with kOmegaSSTSAS."""
        from scripts.models.hybrid_rans_les import generate_sas_config

        with tempfile.TemporaryDirectory() as tmpdir:
            files = generate_sas_config("nasa_hump", output_dir=Path(tmpdir))
            content = Path(files["momentumTransport"]).read_text()
            assert "kOmegaSSTSAS" in content
