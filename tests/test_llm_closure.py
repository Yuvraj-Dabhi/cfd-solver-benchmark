"""Tests for LLM-Driven Turbulence Closure and Physics Foundation Model."""
import numpy as np
import pytest
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))


class TestResidualAnalyzer:
    def test_analyze(self):
        from scripts.ml_augmentation.llm_turbulence_closure import ResidualAnalyzer
        analyzer = ResidualAnalyzer()
        x = np.linspace(0, 1, 50)
        su2_cp = -0.5 * np.sin(2 * np.pi * x)
        exp_cp = -0.5 * np.sin(2 * np.pi * x) + 0.02
        su2_cf = 0.003 * np.ones(50)
        exp_cf = 0.003 * np.ones(50) + 0.001
        report = analyzer.analyze(
            "wall_hump", su2_cp, su2_cf, exp_cp, exp_cf, x,
            model_name="SA", exp_x_sep=0.65, su2_x_sep=0.64,
            exp_x_reat=1.1, su2_x_reat=1.15
        )
        assert report.case_name == "wall_hump"
        assert report.model_name == "SA"
        assert isinstance(report.cp_rmse, float)
        assert isinstance(report.cf_rmse, float)
        assert report.cp_rmse > 0

    def test_mechanism_classification(self):
        from scripts.ml_augmentation.llm_turbulence_closure import ResidualAnalyzer
        analyzer = ResidualAnalyzer()
        x = np.linspace(0, 1, 20)
        # Large separation error → APG mechanism
        report = analyzer.analyze(
            "test", np.zeros(20), np.zeros(20), np.zeros(20), np.zeros(20), x,
            exp_x_sep=0.5, su2_x_sep=0.6
        )
        assert report.error_type == "separation_onset"


class TestLLMClosurePrompter:
    def test_build_prompt(self):
        from scripts.ml_augmentation.llm_turbulence_closure import (
            LLMClosurePrompter, ResidualReport
        )
        prompter = LLMClosurePrompter(model_type="sa")
        report = ResidualReport(
            case_name="wall_hump", model_name="SA",
            separation_error=-0.01, reattachment_error=0.05,
            cp_rmse=0.03, cf_rmse=0.001,
            peak_error_location=0.8,
            error_type="reattachment",
            physical_mechanism="Reattachment overshoot"
        )
        prompt = prompter.build_prompt(report)
        assert "<think>" in prompt
        assert "SA" in prompt
        assert "wall_hump" in prompt
        assert "galilean_invariance" in prompt

    def test_sst_prompt(self):
        from scripts.ml_augmentation.llm_turbulence_closure import (
            LLMClosurePrompter, ResidualReport
        )
        prompter = LLMClosurePrompter(model_type="sst")
        report = ResidualReport(
            case_name="swbli", model_name="SST",
            separation_error=0.085, reattachment_error=0.1,
            cp_rmse=0.05, cf_rmse=0.002,
            peak_error_location=0.3,
        )
        prompt = prompter.build_prompt(report)
        assert "SST" in prompt
        assert "P_k" in prompt


class TestAlgebraicCorrectionParser:
    def test_parse_python_block(self):
        from scripts.ml_augmentation.llm_turbulence_closure import (
            AlgebraicCorrectionParser
        )
        parser = AlgebraicCorrectionParser()
        llm_output = """
Here is the correction:

```python
correction = 0.1 * S * exp(-d / 10)
```
"""
        corrections = parser.parse_response(llm_output, target_variable="nu_t")
        assert len(corrections) >= 1
        assert corrections[0].corrected_variable == "nu_t"
        assert "S" in corrections[0].python_expr

    def test_parse_inline(self):
        from scripts.ml_augmentation.llm_turbulence_closure import (
            AlgebraicCorrectionParser
        )
        parser = AlgebraicCorrectionParser()
        llm_output = "correction = 0.5 * k / omega"
        corrections = parser.parse_response(llm_output)
        assert len(corrections) >= 1

    def test_constraint_checking(self):
        from scripts.ml_augmentation.llm_turbulence_closure import (
            AlgebraicCorrectionParser
        )
        parser = AlgebraicCorrectionParser()
        constraints = parser._check_constraints("abs(S * k / epsilon)")
        assert "galilean_invariant" in constraints
        assert "positive_definite" in constraints


class TestSU2SourceInjector:
    def test_generate_code(self):
        from scripts.ml_augmentation.llm_turbulence_closure import (
            SU2SourceInjector, AlgebraicCorrection
        )
        injector = SU2SourceInjector()
        correction = AlgebraicCorrection(
            formula="0.1 * S * exp(-d/10)",
            python_expr="0.1 * S * np.exp(-d / 10)",
            cpp_expr="0.1 * S * exp(-d / 10.0)",
            corrected_variable="nu_t",
            constraints_satisfied=["galilean_invariant"],
            confidence=0.7,
        )
        code = injector.generate_correction_code(correction, model="sa")
        assert "nu_t_original" in code
        assert "correction" in code
        assert "0.1 * S * exp(-d / 10.0)" in code

    def test_header_comment(self):
        from scripts.ml_augmentation.llm_turbulence_closure import (
            SU2SourceInjector, AlgebraicCorrection
        )
        injector = SU2SourceInjector()
        corrections = [
            AlgebraicCorrection("f1", "e1", "c1", "nu_t"),
            AlgebraicCorrection("f2", "e2", "c2", "omega"),
        ]
        header = injector.generate_header_comment(corrections)
        assert "LLM-Derived" in header
        assert "nu_t" in header
        assert "omega" in header


class TestClosureValidationLoop:
    def test_run_analysis(self):
        from scripts.ml_augmentation.llm_turbulence_closure import (
            ClosureValidationLoop
        )
        loop = ClosureValidationLoop(max_iterations=3)
        x = np.linspace(0, 1, 30)
        result = loop.run_analysis(
            "wall_hump",
            su2_cp=np.sin(x), su2_cf=0.003 * np.ones(30),
            exp_cp=np.sin(x) + 0.01, exp_cf=0.003 * np.ones(30) + 0.001,
            x_coords=x, model_name="SA",
        )
        assert "report" in result
        assert "prompt" in result
        assert "cpp_code" in result
        assert result["iteration"] == 0

    def test_summary(self):
        from scripts.ml_augmentation.llm_turbulence_closure import (
            ClosureValidationLoop
        )
        loop = ClosureValidationLoop()
        s = loop.summary()
        assert s["n_iterations"] == 0


class TestGPhyTEncoder:
    def test_encode(self):
        from scripts.ml_augmentation.physics_foundation_model import (
            GPhyTEncoder, FoundationModelConfig
        )
        cfg = FoundationModelConfig(d_model=32, n_heads=4, n_layers=2)
        encoder = GPhyTEncoder(cfg)
        positions = np.random.randn(20, 2)
        solution = np.random.randn(20, 3)
        embeddings = encoder.encode(positions, solution,
                                     conditions={"Re": 6e6, "Mach": 0.15})
        assert embeddings.shape == (20, 32)

    def test_encode_case(self):
        from scripts.ml_augmentation.physics_foundation_model import (
            GPhyTEncoder, FoundationModelConfig
        )
        cfg = FoundationModelConfig(d_model=16, n_layers=1)
        encoder = GPhyTEncoder(cfg)
        emb = encoder.encode_case(
            "naca0012",
            np.random.randn(10, 2),
            np.random.randn(10, 4),
            {"Re": 6e6}
        )
        assert emb.case_name == "naca0012"
        assert emb.embedding.shape == (10, 16)


class TestCrossAttentionFuser:
    def test_fuse_and_decode(self):
        from scripts.ml_augmentation.physics_foundation_model import (
            CrossAttentionFuser, CaseEmbedding
        )
        fuser = CrossAttentionFuser(d_model=32, n_output_fields=4)
        cases = [
            CaseEmbedding("case1", np.random.randn(10, 32), {"Re": 1e6}, 10),
            CaseEmbedding("case2", np.random.randn(15, 32), {"Re": 2e6}, 15),
        ]
        fused = fuser.fuse(cases)
        assert fused.shape == (25, 32)

        pred = fuser.decode(fused)
        assert pred.shape == (25, 4)


class TestFoundationModelBenchmark:
    def test_zero_shot(self):
        from scripts.ml_augmentation.physics_foundation_model import (
            FoundationModelBenchmark, FoundationModelConfig
        )
        cfg = FoundationModelConfig(d_model=16, n_layers=1, n_output_fields=2)
        bench = FoundationModelBenchmark()

        train_cases = [
            {"name": "flat_plate", "positions": np.random.randn(20, 2),
             "solution": np.random.randn(20, 2), "conditions": {"Re": 5e6}},
        ]
        test_case = {
            "name": "naca0012", "positions": np.random.randn(15, 2),
            "solution": np.random.randn(15, 2), "conditions": {"Re": 6e6},
        }
        metrics = bench.zero_shot_evaluate(train_cases, test_case)
        assert "rmse" in metrics
        assert "mae" in metrics
        assert metrics["n_train_cases"] == 1

    def test_summary(self):
        from scripts.ml_augmentation.physics_foundation_model import (
            FoundationModelBenchmark
        )
        bench = FoundationModelBenchmark()
        s = bench.summary()
        assert s["n_evaluations"] == 0
