"""
Tests for Advanced Turbulence Model Support
=============================================
Validates:
- γ-Reθ transition model config generation
- SA-neg option
- Model comparison matrix
- RSM reference results
- Multi-model comparison runner
- NACA 0012 transition study

Run: pytest tests/test_model_comparison.py -v
"""

import sys
import tempfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestGammaReThetaConfig:
    """Tests for γ-Reθ SST transition model in SU2 config."""

    def _generate_config(self, model="gammaReTheta", case_name="naca_0012_stall"):
        from scripts.solvers.su2_runner import SU2Runner
        tmpdir = Path(tempfile.mkdtemp())
        runner = SU2Runner(case_dir=tmpdir, case_name=case_name, model=model)
        runner.setup_case()
        return (tmpdir / "config.cfg").read_text()

    def test_transition_model_present(self):
        config = self._generate_config()
        assert "TRANS_MODEL= LM" in config

    def test_sst_base_model(self):
        config = self._generate_config()
        assert "KIND_TURB_MODEL= SST" in config

    def test_intermittency_set(self):
        config = self._generate_config()
        assert "FREESTREAM_INTERMITTENCY= 1.0" in config

    def test_low_turbulence_intensity(self):
        """Transition requires low freestream Tu for correlation."""
        config = self._generate_config()
        assert "FREESTREAM_TURBULENCEINTENSITY= 0.01" in config

    def test_no_transition_for_sa(self):
        config = self._generate_config(model="SA")
        assert "TRANS_MODEL" not in config

    def test_no_transition_for_sst(self):
        config = self._generate_config(model="SST")
        assert "TRANS_MODEL" not in config


class TestSANegConfig:
    """Tests for SA-negative variant."""

    def test_sa_neg_options(self):
        from scripts.solvers.su2_runner import SU2Runner
        tmpdir = Path(tempfile.mkdtemp())
        runner = SU2Runner(case_dir=tmpdir, case_name="backward_facing_step", model="SA-neg")
        runner.setup_case()
        config = (tmpdir / "config.cfg").read_text()
        assert "SA_OPTIONS= NEGATIVE" in config
        assert "KIND_TURB_MODEL= SA" in config


class TestModelComparisonMatrix:
    """Tests for MODEL_COMPARISON_MATRIX and RSM_REFERENCE_RESULTS."""

    def test_matrix_exists(self):
        from config import MODEL_COMPARISON_MATRIX
        assert isinstance(MODEL_COMPARISON_MATRIX, dict)
        assert len(MODEL_COMPARISON_MATRIX) >= 5

    def test_core_cases_present(self):
        from config import MODEL_COMPARISON_MATRIX
        for case in ["backward_facing_step", "nasa_hump", "naca_0012_stall"]:
            assert case in MODEL_COMPARISON_MATRIX

    def test_naca_has_transition(self):
        from config import MODEL_COMPARISON_MATRIX
        assert "gammaReTheta" in MODEL_COMPARISON_MATRIX["naca_0012_stall"]

    def test_hump_has_sa_rc(self):
        from config import MODEL_COMPARISON_MATRIX
        assert "SA-RC" in MODEL_COMPARISON_MATRIX["nasa_hump"]

    def test_juncture_has_qcr(self):
        from config import MODEL_COMPARISON_MATRIX
        assert "SA-QCR" in MODEL_COMPARISON_MATRIX["juncture_flow"]

    def test_rsm_results_exist(self):
        from config import RSM_REFERENCE_RESULTS
        assert "nasa_hump" in RSM_REFERENCE_RESULTS
        assert "LRR" in RSM_REFERENCE_RESULTS["nasa_hump"]

    def test_rsm_hump_reattachment(self):
        from config import RSM_REFERENCE_RESULTS
        lrr = RSM_REFERENCE_RESULTS["nasa_hump"]["LRR"]
        assert lrr["x_reat"] < 1.28  # RSM should reattach earlier than SA

    def test_rsm_periodic_hill(self):
        from config import RSM_REFERENCE_RESULTS
        assert "periodic_hill" in RSM_REFERENCE_RESULTS
        lrr = RSM_REFERENCE_RESULTS["periodic_hill"]["LRR"]
        assert lrr["error_pct"] < 5.0


class TestModelComparisonRunner:
    """Tests for ModelComparisonRunner."""

    def test_import(self):
        from scripts.analysis.model_comparison_runner import ModelComparisonRunner
        runner = ModelComparisonRunner("nasa_hump")
        assert len(runner.model_list) >= 3

    def test_invalid_case(self):
        from scripts.analysis.model_comparison_runner import ModelComparisonRunner
        with pytest.raises(ValueError):
            ModelComparisonRunner("nonexistent_case")

    def test_generate_configs(self):
        from scripts.analysis.model_comparison_runner import ModelComparisonRunner
        runner = ModelComparisonRunner("backward_facing_step")
        tmpdir = Path(tempfile.mkdtemp())
        configs = runner.generate_all_configs(tmpdir)
        assert len(configs) == 3  # SA, SST, kEpsilon
        for model, path in configs.items():
            assert path.exists()

    def test_comparison_table(self):
        from scripts.analysis.model_comparison_runner import ModelComparisonRunner
        runner = ModelComparisonRunner("nasa_hump")
        table = runner.build_comparison_table(
            su2_results={"SA": {"CL": 0, "CD": 0, "x_reat": 1.28}}
        )
        md = table.to_markdown()
        assert "SA" in md
        assert "RSM (LRR)" in md  # Should include RSM reference

    def test_list_cases(self):
        from scripts.analysis.model_comparison_runner import list_comparison_cases
        cases = list_comparison_cases()
        assert len(cases) >= 5

    def test_summary_report(self):
        from scripts.analysis.model_comparison_runner import ModelComparisonRunner
        runner = ModelComparisonRunner("nasa_hump")
        report = runner.generate_summary_report()
        assert "Model Comparison" in report


class TestNACA0012TransitionStudy:
    """Tests for NACA 0012 transition study."""

    def test_import(self):
        from scripts.analysis.naca0012_transition_study import NACA0012TransitionStudy
        study = NACA0012TransitionStudy()
        assert len(study.matrix) == 6

    def test_study_summary(self):
        from scripts.analysis.naca0012_transition_study import NACA0012TransitionStudy
        study = NACA0012TransitionStudy()
        summary = study.get_study_summary()
        assert "γ-Reθ" in summary
        assert "Fully Turb" in summary

    def test_generate_configs(self):
        from scripts.analysis.naca0012_transition_study import NACA0012TransitionStudy
        study = NACA0012TransitionStudy()
        tmpdir = Path(tempfile.mkdtemp())
        configs = study.generate_all_configs(tmpdir)
        assert len(configs) == 6

    def test_aoa_patched(self):
        from scripts.analysis.naca0012_transition_study import NACA0012TransitionStudy
        study = NACA0012TransitionStudy()
        tmpdir = Path(tempfile.mkdtemp())
        configs = study.generate_all_configs(tmpdir)
        # Check α=10° config has correct AOA
        for label, path in configs.items():
            if "a10.0" in label:
                text = path.read_text()
                assert "AOA= 10.0" in text

    def test_comparison_table(self):
        from scripts.analysis.naca0012_transition_study import NACA0012TransitionStudy
        study = NACA0012TransitionStudy()
        table = study.build_comparison_table(su2_results={
            "NACA0012_a0.0_SA_fully_turb": {"CL": 0.0, "CD": 0.00825},
        })
        assert "Transition Study" in table
        assert "TMR Reference" in table
