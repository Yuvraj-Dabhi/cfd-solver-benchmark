"""
Tests for TMR Heated Jet and Extended SWBLI setup.
===================================================
Run: pytest tests/test_heated_jet.py -v --tb=short
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

class TestHeatedJetGen:
    
    def test_su2_config_heated_jet(self, tmp_path):
        from simulations.run_heated_jet import generate_su2_config
        
        cfg_path = generate_su2_config(model="SST", case_dir=tmp_path)
        content = cfg_path.read_text()
        
        assert "NASA TMR AJM163H Heated Jet (M_j = 1.63)" in content
        assert "KIND_TURB_MODEL = SST" in content
        
        # Check inlet boundaries (T0=813, P0=454950)
        assert "MARKER_INLET" in content
        assert "813" in content
        assert "454950" in content
        assert "AXISYMMETRIC = YES" in content
        assert "REF_LENGTH = 0.0508" in content

class TestSWBLIGen:

    def test_swbli_su2_config(self, tmp_path):
        from simulations.run_swbli import generate_su2_config
        
        # Test M=5 2D case
        cfg_m5 = generate_su2_config("M5_2D", tmp_path / "M5", "swbli_L1_coarse.su2", model="SA", n_iter=100)
        content_m5 = cfg_m5.read_text()
        assert "MACH_NUMBER= 5.0" in content_m5
        assert "AXISYMMETRIC= NO" in content_m5
        
        # Test M=7 Axisymmetric case
        cfg_m7 = generate_su2_config("M7_AXI", tmp_path / "M7", "aswbli_M7_mesh.su2", model="SST", n_iter=100)
        content_m7 = cfg_m7.read_text()
        assert "MACH_NUMBER= 7.0" in content_m7
        assert "AXISYMMETRIC= YES" in content_m7
        
    def test_swbli_openfoam_config(self, tmp_path):
        try:
            from simulations.run_swbli import generate_openfoam_config
        except ImportError:
            pytest.skip("openfoam_utils not available")
            
        case_dir = tmp_path / "M7_OF"
        # The generator modifies the directory and returns the case directory Path
        generate_openfoam_config("M7_AXI", case_dir)
        
        # Verify structure
        assert (case_dir / "system" / "controlDict").exists()
        assert (case_dir / "system" / "fvSolution").exists()
        assert (case_dir / "constant" / "transportProperties").exists()
        
        # Ensure it's using rhoCentralFoam
        control_content = (case_dir / "system" / "controlDict").read_text()
        assert "application     rhoCentralFoam;" in control_content
        
        # Verify thermophysics uses perfectGas
        thermo_content = (case_dir / "constant" / "transportProperties").read_text()
        assert "equationOfState perfectGas;" in thermo_content
