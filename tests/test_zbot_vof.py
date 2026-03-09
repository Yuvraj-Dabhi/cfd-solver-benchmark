"""
Tests for ZBOT VOF implementation.
===================================
Run: pytest tests/test_zbot_vof.py -v --tb=short
"""

import math
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestZBOTCaseGen:
    
    def test_case_parameters(self):
        from simulations.run_zbot_vof import CASES
        
        std = CASES["standard"]
        assert std.inlet_velocity == 0.5
        assert std.g_initial == 9.81
        assert std.g_micro == 1e-6
        assert std.H_tank == 0.3
        
        hf = CASES["high_flow"]
        assert hf.inlet_velocity == 1.5
        
        lg = CASES["low_g"]
        assert lg.g_micro == 1e-8
        
    def test_inlet_flow_rate(self):
        from simulations.run_zbot_vof import CASES
        
        std = CASES["standard"]
        expected_v_in = np.pi * (0.005)**2 * 0.5
        assert std.V_inlet == pytest.approx(expected_v_in)
        
    def test_case_directory_generation(self, tmp_path):
        from simulations.run_zbot_vof import CASES, FoamCaseGenerator, write_blockMeshDict, write_setFieldsDict, write_fvOptions, generate_fvSchemes, generate_fvSolution, generate_transportProperties, write_0_fields
        
        case = CASES["standard"]
        test_dir = tmp_path / "zbot_test"
        
        generator = FoamCaseGenerator(test_dir)
        generator.setup_directories()
        
        generator.write_controlDict(application="interFoam")
        generator.write_fvSchemes(generate_fvSchemes())
        generator.write_fvSolution(generate_fvSolution())
        write_blockMeshDict(case, test_dir)
        write_setFieldsDict(case, test_dir)
        write_fvOptions(case, test_dir)
        generator.write_transportProperties(generate_transportProperties(case.sigma))
        generator.write_momentumTransport("laminar")
        generator.write_g((0, 0, 0))
        write_0_fields(case, test_dir)
        
        # Verify structure
        assert (test_dir / "system" / "controlDict").exists()
        assert (test_dir / "system" / "blockMeshDict").exists()
        assert (test_dir / "system" / "fvOptions").exists()
        assert (test_dir / "constant" / "transportProperties").exists()
        assert (test_dir / "constant" / "g").exists()
        assert (test_dir / "0" / "alpha.water").exists()
        assert (test_dir / "0" / "U").exists()
        assert (test_dir / "0" / "p_rgh").exists()
        
        # Verify blockMesh content
        blockMesh_content = (test_dir / "system" / "blockMeshDict").read_text()
        assert "arc 1 2" in blockMesh_content
        assert "front" in blockMesh_content
        assert "type wedge;" in blockMesh_content

    def test_fvOptions_ramp(self, tmp_path):
        from simulations.run_zbot_vof import CASES, write_fvOptions
        case = CASES["standard"]
        test_dir = tmp_path / "fvOpt_test"
        (test_dir / "system").mkdir(parents=True)
        write_fvOptions(case, test_dir)
        
        content = (test_dir / "system" / "fvOptions").read_text()
        assert "type            coded;" in content
        assert "scalar g_initial = -9.81;" in content
        assert "scalar g_micro = -1e-06;" in content

class TestFoamLogParser:

    def test_parser_init(self, tmp_path):
        from scripts.openfoam_utils.foam_log_parser import FoamLogParser
        
        # Create a mock openfoam log
        log_content = """Time = 0.05
Solving for alpha.water, Initial residual = 0.001, Final residual = 1e-6, No Iterations 3
Courant Number mean: 0.1 max: 0.5
time step continuity errors : sum local = 1e-10, global = 1e-11, cumulative = 1e-9
ExecutionTime = 10.5 s

Time = 0.10
Solving for alpha.water, Initial residual = 0.0005, Final residual = 1e-7, No Iterations 2
Courant Number mean: 0.15 max: 0.55
time step continuity errors : sum local = 2e-10, global = 2e-11, cumulative = 2e-9
ExecutionTime = 20.1 s
"""
        log_file = tmp_path / "mock.log"
        log_file.write_text(log_content)
        
        parser = FoamLogParser(log_file)
        success = parser.parse()
        
        assert success is True
        assert len(parser.time) == 2
        assert parser.time == [0.05, 0.10]
        assert "alpha.water" in parser.residuals
        assert parser.residuals["alpha.water"] == [0.001, 0.0005]
        
        assert parser.courant_numbers == [0.5, 0.55]
        assert parser.continuity_errors == [1e-10, 2e-10]
        assert parser.execution_times == [10.5, 20.1]
        
        summary = parser.get_summary()
        assert summary["total_timesteps"] == 2
        assert summary["final_time"] == 0.10
        assert summary["max_courant"] == 0.55
