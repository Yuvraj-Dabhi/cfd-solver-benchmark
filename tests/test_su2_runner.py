"""
Tests for SU2 Runner Configuration Generation
===============================================
Validates that SU2Runner produces correct config files with:
- Second-order discretization (MUSCL_TURB=YES)
- Tighter convergence targets (-10)
- SA-RC / SA-QCR variant support
- Compressible vs incompressible solver selection
- Dual-time stepping for unsteady flows

No SU2 installation is required — tests only validate config file content.

Run: pytest tests/test_su2_runner.py -v
"""

import sys
import tempfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestSU2ConfigGeneration:
    """Tests for SU2 config.cfg generation logic."""

    def _generate_config(self, model="SA", case_name="backward_facing_step", **kwargs):
        """Helper: generate config and return its text content."""
        from scripts.solvers.su2_runner import SU2Runner
        tmpdir = Path(tempfile.mkdtemp())
        runner = SU2Runner(
            case_dir=tmpdir,
            case_name=case_name,
            model=model,
            **kwargs,
        )
        runner.setup_case()
        config_text = (tmpdir / "config.cfg").read_text()
        return config_text

    def test_muscl_turb_enabled(self):
        """NASA TMR: first-order turbulence is inadequate for verification."""
        config = self._generate_config()
        assert "MUSCL_TURB= YES" in config, (
            "MUSCL_TURB must be YES for second-order turbulence discretization"
        )

    def test_muscl_flow_enabled(self):
        """Flow variables must also use second-order."""
        config = self._generate_config()
        assert "MUSCL_FLOW= YES" in config

    def test_convergence_target_default(self):
        """Default convergence should be -10 (stricter than -8)."""
        config = self._generate_config()
        assert "CONV_RESIDUAL_MINVAL= -10" in config

    def test_convergence_target_custom(self):
        """Custom convergence target should be respected."""
        config = self._generate_config(conv_residual_minval=-12)
        assert "CONV_RESIDUAL_MINVAL= -12" in config

    def test_sa_rc_options(self):
        """SA-RC must emit SA_OPTIONS= RC."""
        config = self._generate_config(model="SA-RC")
        assert "SA_OPTIONS= RC" in config
        assert "KIND_TURB_MODEL= SA" in config

    def test_sa_qcr_options(self):
        """SA-QCR must emit SA_OPTIONS= QCR2000."""
        config = self._generate_config(model="SA-QCR")
        assert "SA_OPTIONS= QCR2000" in config
        assert "KIND_TURB_MODEL= SA" in config

    def test_sa_no_extra_options(self):
        """Plain SA should not have SA_OPTIONS."""
        config = self._generate_config(model="SA")
        assert "SA_OPTIONS" not in config

    def test_sst_model(self):
        """SST model should be correctly mapped."""
        config = self._generate_config(model="SST")
        assert "KIND_TURB_MODEL= SST" in config

    def test_incompressible_solver(self):
        """Low-Mach cases should use INC_RANS solver."""
        config = self._generate_config(case_name="backward_facing_step")
        # BFS is low-Mach → INC_RANS
        assert "SOLVER= INC_RANS" in config

    def test_venkatakrishnan_limiter_turb(self):
        """Turbulence limiter should be Venkatakrishnan."""
        config = self._generate_config()
        assert "SLOPE_LIMITER_TURB= VENKATAKRISHNAN" in config

    def test_dual_time_disabled_by_default(self):
        """Dual-time stepping should not appear by default."""
        config = self._generate_config()
        assert "TIME_MARCHING" not in config

    def test_dual_time_enabled(self):
        """When enabled, dual-time stepping config should be present."""
        config = self._generate_config(enable_dual_time=True)
        assert "TIME_MARCHING= DUAL_TIME_STEPPING-2ND_ORDER" in config
        assert "INNER_ITER= 50" in config


class TestSU2RunResult:
    """Tests for the SU2RunResult dataclass."""

    def test_result_defaults(self):
        from scripts.solvers.su2_runner import SU2RunResult
        result = SU2RunResult(case_dir="/tmp/test", model="SA", mesh_level="medium")
        assert result.converged is False
        assert result.iterations == 0
        assert result.cl == 0.0
        assert result.cd == 0.0
        assert result.error is None
