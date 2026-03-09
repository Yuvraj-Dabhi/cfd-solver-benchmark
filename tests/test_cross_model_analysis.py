"""
Tests for Cross-Model Wall-Hump Comparison & SWBLI Enrichment
===============================================================
Validates separation metrics extraction, report generation,
and analysis pipeline for both extensions.

Run: pytest tests/test_cross_model_analysis.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =========================================================================
# Wall-Hump Cross-Model Tests
# =========================================================================
class TestWallHumpSeparation:
    """Test separation detection functions."""

    def test_find_separation_from_cf(self):
        from scripts.analysis.wall_hump_cross_model import find_separation_from_cf
        x = np.linspace(0, 2, 1000)
        Cf = np.cos(np.pi * (x - 0.665) / 0.5) * 0.003
        # Cf crosses zero near 0.665 + 0.25 = ~0.915?
        # Actually cos(0) = 1, cos(pi) = -1, so zero at x=0.665+0.25=0.915
        # Let's use a simpler test
        Cf2 = 0.003 - 0.006 * (x - 0.5)  # crosses zero at x=1.0
        sep = find_separation_from_cf(x, Cf2, x_min=0.5, x_max=1.5)
        assert sep is not None
        assert abs(sep - 1.0) < 0.01

    def test_find_separation_no_crossing(self):
        from scripts.analysis.wall_hump_cross_model import find_separation_from_cf
        x = np.linspace(0, 2, 100)
        Cf = np.ones(100) * 0.003  # Always positive
        sep = find_separation_from_cf(x, Cf)
        assert sep is None

    def test_find_reattachment(self):
        from scripts.analysis.wall_hump_cross_model import find_reattachment_from_cf
        x = np.linspace(0, 2, 1000)
        Cf = -0.001 + 0.002 * (x - 0.9)  # negative→positive near x=1.4
        reat = find_reattachment_from_cf(x, Cf, x_min=0.9, x_max=1.8)
        assert reat is not None
        assert abs(reat - 1.4) < 0.02

    def test_compute_region_rmse(self):
        from scripts.analysis.wall_hump_cross_model import compute_region_rmse
        x = np.linspace(0, 2, 100)
        y1 = np.sin(x)
        y2 = np.sin(x) + 0.01  # Small offset
        rmse = compute_region_rmse(x, y1, x, y2, 0.6, 1.3)
        assert rmse < 0.02
        assert rmse > 0.005


class TestWallHumpSyntheticData:
    """Test synthetic data generation."""

    def test_sa_data_shapes(self):
        from scripts.analysis.wall_hump_cross_model import generate_synthetic_hump_data
        x, Cp, Cf = generate_synthetic_hump_data("SA")
        assert x.shape == Cp.shape == Cf.shape
        assert len(x) == 500

    def test_sa_separation_location(self):
        from scripts.analysis.wall_hump_cross_model import (
            generate_synthetic_hump_data, find_separation_from_cf,
        )
        x, _, Cf = generate_synthetic_hump_data("SA")
        sep = find_separation_from_cf(x, Cf)
        assert sep is not None
        assert abs(sep - 0.665) < 0.02

    def test_sst_has_larger_bubble(self):
        from scripts.analysis.wall_hump_cross_model import generate_synthetic_hump_data
        _, _, Cf_sa = generate_synthetic_hump_data("SA")
        _, _, Cf_sst = generate_synthetic_hump_data("SST")
        # SST should have more negative Cf (larger separation)
        assert np.min(Cf_sst) < np.min(Cf_sa)

    def test_keps_data_valid(self):
        from scripts.analysis.wall_hump_cross_model import generate_synthetic_hump_data
        x, Cp, Cf = generate_synthetic_hump_data("kEpsilon")
        assert not np.any(np.isnan(Cp))
        assert not np.any(np.isnan(Cf))


class TestWallHumpMetrics:
    """Test metrics extraction."""

    def test_extract_metrics_sa(self):
        from scripts.analysis.wall_hump_cross_model import (
            generate_synthetic_hump_data, extract_metrics_from_surface_data,
        )
        x, Cp, Cf = generate_synthetic_hump_data("SA")
        met = extract_metrics_from_surface_data(x, Cp, Cf, "SA")
        assert not np.isnan(met.x_sep)
        assert not np.isnan(met.x_reatt)
        assert met.bubble_length > 0
        assert met.Cf_min < 0

    def test_sa_close_to_tmr(self):
        from scripts.analysis.wall_hump_cross_model import (
            generate_synthetic_hump_data, extract_metrics_from_surface_data,
        )
        x, Cp, Cf = generate_synthetic_hump_data("SA")
        met = extract_metrics_from_surface_data(x, Cp, Cf, "SA")
        # SA should be close to TMR CFL3D (0.664)
        assert abs(met.x_sep - 0.665) < 0.02

    def test_rmse_with_reference(self):
        from scripts.analysis.wall_hump_cross_model import (
            generate_synthetic_hump_data, generate_experimental_reference,
            extract_metrics_from_surface_data,
        )
        x, Cp, Cf = generate_synthetic_hump_data("SST")
        x_ref, cp_ref, cf_ref = generate_experimental_reference()
        met = extract_metrics_from_surface_data(
            x, Cp, Cf, "SST",
            x_ref_cp=x_ref, cp_ref=cp_ref,
            x_ref_cf=x_ref, cf_ref=cf_ref,
        )
        assert not np.isnan(met.Cp_rmse_separation)
        assert met.Cp_rmse_separation >= 0


class TestWallHumpReport:
    """Test report generation."""

    def test_build_report(self):
        from scripts.analysis.wall_hump_cross_model import build_comparison_report
        report = build_comparison_report(["SA", "SST", "kEpsilon"])
        assert "SA" in report.metrics
        assert "SST" in report.metrics
        assert "kEpsilon" in report.metrics

    def test_format_table(self):
        from scripts.analysis.wall_hump_cross_model import (
            build_comparison_report, format_metrics_table,
        )
        report = build_comparison_report()
        table = format_metrics_table(report)
        assert "x_sep" in table
        assert "x_reatt" in table
        assert "Cf_min" in table

    def test_markdown_report(self):
        from scripts.analysis.wall_hump_cross_model import (
            build_comparison_report, generate_markdown_report,
        )
        report = build_comparison_report()
        md = generate_markdown_report(report)
        assert "# Wall-Hump" in md
        assert "SA" in md
        assert "SST" in md


# =========================================================================
# SWBLI Enrichment Tests
# =========================================================================
class TestSWBLISyntheticData:
    """Test SWBLI synthetic data generation."""

    def test_sa_surface_data(self):
        from scripts.analysis.swbli_enrichment import generate_swbli_surface_data
        data = generate_swbli_surface_data("SA")
        assert "x_mm" in data
        assert "Cf" in data
        assert "p_wall" in data
        assert "St" in data
        assert len(data["x_mm"]) == 600

    def test_sa_has_separation(self):
        from scripts.analysis.swbli_enrichment import generate_swbli_surface_data
        data = generate_swbli_surface_data("SA")
        # Cf should go negative in separation
        assert np.min(data["Cf"]) < 0

    def test_sst_larger_separation(self):
        from scripts.analysis.swbli_enrichment import generate_swbli_surface_data
        sa = generate_swbli_surface_data("SA")
        sst = generate_swbli_surface_data("SST")
        # SST should have more negative Cf
        assert np.min(sst["Cf"]) < np.min(sa["Cf"])

    def test_pressure_rises_through_shock(self):
        from scripts.analysis.swbli_enrichment import generate_swbli_surface_data
        data = generate_swbli_surface_data("SA")
        # Pressure should be ~1 upstream and >1 at shock
        assert data["p_wall"][0] == pytest.approx(1.0)
        assert np.max(data["p_wall"]) > 3.0

    def test_stanton_peaks_at_reattachment(self):
        from scripts.analysis.swbli_enrichment import generate_swbli_surface_data
        data = generate_swbli_surface_data("SST")
        peak_idx = np.argmax(data["St"])
        peak_x = data["x_mm"][peak_idx]
        # Peak should be near reattachment (~20mm for SST)
        assert 10 < peak_x < 30


class TestSWBLIMetrics:
    """Test SWBLI metrics extraction."""

    def test_extract_sa_metrics(self):
        from scripts.analysis.swbli_enrichment import extract_swbli_metrics
        met = extract_swbli_metrics("SA")
        assert met.x_sep_mm < 0  # upstream of shock
        assert met.x_reatt_mm > 0  # downstream of shock
        assert met.L_sep_mm > 0
        assert met.Cf_plateau < 0

    def test_sa_separation_error(self):
        from scripts.analysis.swbli_enrichment import extract_swbli_metrics
        met = extract_swbli_metrics("SA")
        assert met.x_sep_error_pct < 20  # Within 20% of experiment

    def test_sst_separation_error(self):
        from scripts.analysis.swbli_enrichment import extract_swbli_metrics
        met = extract_swbli_metrics("SST")
        assert met.L_sep_error_pct < 20

    def test_stanton_ordering(self):
        from scripts.analysis.swbli_enrichment import extract_swbli_metrics
        met = extract_swbli_metrics("SST")
        assert met.St_peak > met.St_upstream


class TestSWBLIReport:
    """Test SWBLI report generation."""

    def test_build_report(self):
        from scripts.analysis.swbli_enrichment import build_swbli_report
        report = build_swbli_report()
        assert "SA" in report.model_metrics
        assert "SST" in report.model_metrics
        assert len(report.discussion) > 100

    def test_format_table(self):
        from scripts.analysis.swbli_enrichment import (
            build_swbli_report, format_swbli_table,
        )
        report = build_swbli_report()
        table = format_swbli_table(report)
        assert "x_sep" in table
        assert "L_sep" in table
        assert "Schulein" in table

    def test_markdown_report(self):
        from scripts.analysis.swbli_enrichment import (
            build_swbli_report, generate_markdown_report,
        )
        report = build_swbli_report()
        md = generate_markdown_report(report)
        assert "Schulein" in md
        assert "Hypersonic" in md

    def test_discussion_content(self):
        from scripts.analysis.swbli_enrichment import generate_discussion
        disc = generate_discussion()
        assert "SA" in disc
        assert "SST" in disc
        assert "heat" in disc.lower()
        assert "separation" in disc.lower()
