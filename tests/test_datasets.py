#!/usr/bin/env python3
"""
Tests for Additional Dataset Integration (Component 3)
=========================================================
Tests Kim BFS data, NACA 4412 data, Bradshaw archive catalog,
and DATA_SOURCES registry.
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestKimBFSData:
    """Tests for Kim et al. (1998) backward-facing step data."""

    def test_load_kim_bfs(self):
        """Kim BFS data should load with correct structure."""
        from experimental_data.bfs_kim.bfs_kim_loader import load_kim_bfs_data

        data = load_kim_bfs_data()
        assert data.case_name == "bfs_kim"
        assert data.wall_data is not None
        assert len(data.wall_data) == 19
        assert "x_H" in data.wall_data.columns
        assert "Cp" in data.wall_data.columns
        assert "Cf" in data.wall_data.columns

    def test_kim_bfs_velocity_profiles(self):
        """Kim BFS should have velocity profiles at 5 stations."""
        from experimental_data.bfs_kim.bfs_kim_loader import load_kim_bfs_data

        data = load_kim_bfs_data()
        assert len(data.velocity_profiles) == 5
        # Check station 4 profile
        assert 4.0 in data.velocity_profiles
        profile = data.velocity_profiles[4.0]
        assert "y_H" in profile.columns
        assert "U_Uref" in profile.columns

    def test_kim_bfs_reattachment(self):
        """Reattachment should be at x/H ≈ 7.0 (within literature range)."""
        from experimental_data.bfs_kim.bfs_kim_loader import load_kim_bfs_data

        data = load_kim_bfs_data()
        x_reat = data.separation_metrics["x_reat_xH"]
        assert 5.5 <= x_reat <= 8.5  # Literature range for BFS

    def test_kim_bfs_reynolds_number(self):
        """Reynolds number should be 132,000."""
        from experimental_data.bfs_kim.bfs_kim_loader import load_kim_bfs_data

        data = load_kim_bfs_data()
        assert data.separation_metrics["Re_H"] == 132_000


class TestNACA4412Data:
    """Tests for NACA 4412 Wadcock/Coles data."""

    def test_load_naca4412(self):
        """NACA 4412 data should load correctly."""
        from experimental_data.naca4412_wadcock.naca4412_wadcock_loader import (
            load_naca4412_data,
        )

        data = load_naca4412_data()
        assert data.case_name == "naca4412_wadcock"
        assert data.wall_data is not None
        assert "x_c" in data.wall_data.columns
        assert "Cp" in data.wall_data.columns

    def test_naca4412_has_upper_lower_surfaces(self):
        """Wall data should include both upper and lower surface."""
        from experimental_data.naca4412_wadcock.naca4412_wadcock_loader import (
            load_naca4412_data,
        )

        data = load_naca4412_data()
        surfaces = data.wall_data["surface"].unique()
        assert "upper" in surfaces
        assert "lower" in surfaces

    def test_naca4412_te_separation(self):
        """Trailing-edge separation should onset near x/c ≈ 0.75."""
        from experimental_data.naca4412_wadcock.naca4412_wadcock_loader import (
            load_naca4412_data,
        )

        data = load_naca4412_data()
        x_sep = data.separation_metrics["x_sep_te_xc"]
        assert 0.6 <= x_sep <= 0.9

    def test_naca4412_velocity_profiles(self):
        """Should have velocity profiles at 6 stations."""
        from experimental_data.naca4412_wadcock.naca4412_wadcock_loader import (
            load_naca4412_data,
        )

        data = load_naca4412_data()
        assert len(data.velocity_profiles) == 6

    def test_naca4412_alpha(self):
        """Angle of attack should be ~13.87°."""
        from experimental_data.naca4412_wadcock.naca4412_wadcock_loader import (
            load_naca4412_data,
        )

        data = load_naca4412_data()
        assert abs(data.separation_metrics["alpha_deg"] - 13.87) < 0.1


class TestBradshawArchive:
    """Tests for the Bradshaw archive catalog and loader."""

    def test_bradshaw_catalog_not_empty(self):
        """Catalog should contain cases."""
        from scripts.validation.bradshaw_archive_loader import list_available_cases

        cases = list_available_cases()
        assert len(cases) > 0

    def test_bradshaw_has_kim_bfs(self):
        """Catalog should include Kim BFS."""
        from scripts.validation.bradshaw_archive_loader import list_available_cases

        cases = list_available_cases()
        assert "KIM_BFS" in cases

    def test_bradshaw_has_naca4412(self):
        """Catalog should include NACA 4412 TE."""
        from scripts.validation.bradshaw_archive_loader import list_available_cases

        cases = list_available_cases()
        assert "NACA_4412_TE" in cases

    def test_bradshaw_case_info(self):
        """get_case_info should return complete metadata."""
        from scripts.validation.bradshaw_archive_loader import get_case_info

        info = get_case_info("KIM_BFS")
        assert info["code"] == "KIM_BFS"
        assert info["Re"] == 132_000
        assert len(info["available_quantities"]) > 0

    def test_bradshaw_load_kim_bfs(self):
        """load_bradshaw_case should delegate to Kim BFS loader."""
        from scripts.validation.bradshaw_archive_loader import load_bradshaw_case

        data = load_bradshaw_case("KIM_BFS")
        assert data.case_name == "bfs_kim"

    def test_bradshaw_load_naca4412(self):
        """load_bradshaw_case should delegate to NACA 4412 loader."""
        from scripts.validation.bradshaw_archive_loader import load_bradshaw_case

        data = load_bradshaw_case("NACA_4412_TE")
        assert data.case_name == "naca4412_wadcock"

    def test_bradshaw_unknown_case_raises(self):
        """Unknown case should raise ValueError."""
        from scripts.validation.bradshaw_archive_loader import load_bradshaw_case

        with pytest.raises(ValueError):
            load_bradshaw_case("NONEXISTENT")


class TestDataSourcesRegistry:
    """Tests for DATA_SOURCES config entries."""

    def test_bradshaw_archive_in_data_sources(self):
        """bradshaw_archive should be registered in DATA_SOURCES."""
        from config import DATA_SOURCES

        assert "bradshaw_archive" in DATA_SOURCES
        assert "url" in DATA_SOURCES["bradshaw_archive"]

    def test_jhtdb_in_data_sources(self):
        """JHTDB should be registered in DATA_SOURCES."""
        from config import DATA_SOURCES

        assert "jhtdb" in DATA_SOURCES
        assert "Johns Hopkins" in DATA_SOURCES["jhtdb"]["name"]
