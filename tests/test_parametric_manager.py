"""
Tests for Parametric Case Manager
===================================
Validates YAML/JSON parsing, case expansion, metadata tracking,
pre-built sweep templates, and the ParametricManager workflow.

Run: pytest tests/test_parametric_manager.py -v
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =========================================================================
# CaseDefinition Tests
# =========================================================================
class TestCaseDefinition:
    """Test CaseDefinition expansion."""

    def test_single_case(self):
        from scripts.parametric.parametric_manager import CaseDefinition
        defn = CaseDefinition(geometry="NACA0012", alpha=[10], model=["SA"])
        cases = defn.expand()
        assert len(cases) == 1
        assert cases[0]["geometry"] == "NACA0012"
        assert cases[0]["alpha"] == 10
        assert cases[0]["model"] == "SA"

    def test_cartesian_expansion(self):
        from scripts.parametric.parametric_manager import CaseDefinition
        defn = CaseDefinition(
            geometry="NACA0012",
            alpha=[0, 5, 10],
            model=["SA", "SST"],
            grid=["medium", "fine"],
        )
        cases = defn.expand()
        assert len(cases) == 3 * 2 * 2  # 12 cases

    def test_expansion_preserves_geometry(self):
        from scripts.parametric.parametric_manager import CaseDefinition
        defn = CaseDefinition(geometry="wall_hump", model=["SA", "SST", "kEpsilon"])
        cases = defn.expand()
        assert all(c["geometry"] == "wall_hump" for c in cases)
        assert len(cases) == 3

    def test_extra_fields_preserved(self):
        from scripts.parametric.parametric_manager import CaseDefinition
        defn = CaseDefinition(
            geometry="NACA0012",
            extra={"turbulence_intensity": 0.03},
        )
        cases = defn.expand()
        assert cases[0]["turbulence_intensity"] == 0.03


# =========================================================================
# YAML / JSON Parser Tests
# =========================================================================
class TestCaseParser:
    """Test YAML/JSON case file parsing."""

    def test_parse_json(self, tmp_path):
        from scripts.parametric.parametric_manager import parse_case_file
        case_file = tmp_path / "sweep.json"
        data = {
            "cases": [
                {
                    "geometry": "NACA0012",
                    "alpha": [0, 5, 10],
                    "mach": 0.15,
                    "model": ["SA", "SST"],
                    "grid": "fine",
                },
            ]
        }
        case_file.write_text(json.dumps(data))
        definitions = parse_case_file(case_file)
        assert len(definitions) == 1
        assert definitions[0].geometry == "NACA0012"
        assert len(definitions[0].alpha) == 3

    def test_parse_multiple_cases(self, tmp_path):
        from scripts.parametric.parametric_manager import parse_case_file
        case_file = tmp_path / "multi.json"
        data = {
            "cases": [
                {"geometry": "NACA0012", "alpha": [0, 5], "model": ["SA"]},
                {"geometry": "wall_hump", "model": ["SA", "SST"]},
            ]
        }
        case_file.write_text(json.dumps(data))
        definitions = parse_case_file(case_file)
        assert len(definitions) == 2

    def test_parse_dict(self):
        from scripts.parametric.parametric_manager import parse_case_dict
        data = {"cases": [{"geometry": "swbli", "mach": 5.0, "model": "SA"}]}
        definitions = parse_case_dict(data)
        assert len(definitions) == 1
        assert definitions[0].geometry == "swbli"

    def test_missing_cases_key_raises(self, tmp_path):
        from scripts.parametric.parametric_manager import parse_case_file
        case_file = tmp_path / "bad.json"
        case_file.write_text('{"runs": []}')
        with pytest.raises(ValueError, match="cases"):
            parse_case_file(case_file)


# =========================================================================
# Pre-built Template Tests
# =========================================================================
class TestSweepTemplates:
    """Test pre-built sweep templates."""

    def test_naca_airfoil_sweep(self):
        from scripts.parametric.parametric_manager import naca_airfoil_sweep
        defn = naca_airfoil_sweep()
        cases = defn.expand()
        assert len(cases) == 8 * 2  # 8 alphas × 2 models = 16
        assert all(c["geometry"] == "NACA0012" for c in cases)

    def test_naca_4digit_dataset(self):
        from scripts.parametric.parametric_manager import naca_4digit_dataset
        defs = naca_4digit_dataset()
        assert len(defs) == 6  # 6 airfoils
        total_cases = sum(len(d.expand()) for d in defs)
        assert total_cases == 6 * 8 * 2  # 96 cases

    def test_wall_hump_sweep(self):
        from scripts.parametric.parametric_manager import wall_hump_sweep
        defn = wall_hump_sweep()
        cases = defn.expand()
        assert len(cases) == 3 * 3  # 3 models × 3 grids = 9
        assert all(c["mach"] == 0.1 for c in cases)

    def test_swbli_sweep(self):
        from scripts.parametric.parametric_manager import swbli_sweep
        defn = swbli_sweep()
        cases = defn.expand()
        assert len(cases) == 2  # 2 models
        assert all(c["mach"] == 5.0 for c in cases)

    def test_custom_naca_sweep(self):
        from scripts.parametric.parametric_manager import naca_airfoil_sweep
        defn = naca_airfoil_sweep(
            digits="2412", alphas=[0, 5], models=["SA"], grids=["coarse"]
        )
        cases = defn.expand()
        assert len(cases) == 2
        assert cases[0]["geometry"] == "NACA2412"


# =========================================================================
# ParametricManager Tests
# =========================================================================
class TestParametricManager:
    """Test the ParametricManager workflow."""

    def test_from_dict(self):
        from scripts.parametric.parametric_manager import ParametricManager
        data = {
            "cases": [
                {"geometry": "NACA0012", "alpha": [0, 5], "model": ["SA"]},
            ]
        }
        mgr = ParametricManager.from_dict(data)
        assert len(mgr.definitions) == 1

    def test_expand_all(self):
        from scripts.parametric.parametric_manager import (
            ParametricManager, naca_airfoil_sweep,
        )
        mgr = ParametricManager()
        mgr.add_case(naca_airfoil_sweep(alphas=[0, 5], models=["SA"]))
        cases = mgr.expand_all()
        assert len(cases) == 2

    def test_generate_metadata(self):
        from scripts.parametric.parametric_manager import (
            ParametricManager, naca_airfoil_sweep,
        )
        mgr = ParametricManager()
        mgr.add_case(naca_airfoil_sweep(alphas=[0], models=["SA"]))
        mgr.expand_all()
        metadata = mgr.generate_all()
        assert len(metadata) == 1
        assert metadata[0].run_id
        assert metadata[0].mesh_id
        assert metadata[0].config_hash
        assert metadata[0].date

    def test_summary(self):
        from scripts.parametric.parametric_manager import (
            ParametricManager, naca_airfoil_sweep, wall_hump_sweep,
        )
        mgr = ParametricManager()
        mgr.add_case(naca_airfoil_sweep(alphas=[0, 5], models=["SA"]))
        mgr.add_case(wall_hump_sweep(models=["SA"], grids=["medium"]))
        mgr.expand_all()
        mgr.generate_all()
        summary = mgr.summary()
        assert "NACA0012" in summary
        assert "wall_hump" in summary

    def test_dry_run(self):
        from scripts.parametric.parametric_manager import (
            ParametricManager, naca_airfoil_sweep,
        )
        mgr = ParametricManager()
        mgr.add_case(naca_airfoil_sweep(alphas=[0], models=["SA"]))
        mgr.expand_all()
        mgr.generate_all()
        mgr.run_all(dry_run=True)  # Should not raise

    def test_save_manifest(self, tmp_path):
        from scripts.parametric.parametric_manager import (
            ParametricManager, naca_airfoil_sweep,
        )
        mgr = ParametricManager(output_dir=tmp_path)
        mgr.add_case(naca_airfoil_sweep(alphas=[0, 5], models=["SA"]))
        mgr.expand_all()
        mgr.generate_all()
        path = mgr.save_manifest()
        assert path.exists()
        with open(path) as f:
            manifest = json.load(f)
        assert manifest["n_cases"] == 2

    def test_add_multiple_cases(self):
        from scripts.parametric.parametric_manager import (
            ParametricManager, naca_4digit_dataset,
        )
        mgr = ParametricManager()
        mgr.add_cases(naca_4digit_dataset(digits_list=["0012", "2412"]))
        cases = mgr.expand_all()
        assert len(cases) == 2 * 8 * 2  # 32 cases

    def test_from_json_file(self, tmp_path):
        from scripts.parametric.parametric_manager import ParametricManager
        case_file = tmp_path / "test.json"
        case_file.write_text(json.dumps({
            "cases": [
                {"geometry": "NACA0012", "alpha": [0, 10], "model": "SA"},
            ]
        }))
        mgr = ParametricManager.from_yaml(case_file)
        cases = mgr.expand_all()
        assert len(cases) == 2
