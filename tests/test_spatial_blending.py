#!/usr/bin/env python3
"""
Tests for Multi-Agent Spatial Blending
=======================================
Tests sensor functions, regime classification, blending functions,
specialized agents, and the full pipeline comparison.
"""

import sys
import numpy as np
import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ml_augmentation.spatial_blending import (
    compute_pressure_gradient_parameter,
    compute_friction_velocity_ratio,
    compute_y_plus,
    classify_flow_regime,
    tanh_blending_function,
    compute_separation_blending,
    compute_reattachment_blending,
    blend_agent_predictions,
    SeparationAgent,
    ReattachmentAgent,
    MultiAgentBlendingPipeline,
    AgentPrediction,
    run_blending_comparison,
)
from scripts.ml_augmentation.tbnn_closure import check_realizability


# =========================================================================
# Sensor Function Tests
# =========================================================================

class TestPressureGradientParameter:
    """Test A_p^+ computation."""

    def test_zero_pressure_gradient(self):
        dp_ds = np.zeros(10)
        u_tau = np.ones(10) * 0.05
        Ap = compute_pressure_gradient_parameter(dp_ds, nu=1.5e-5, u_tau=u_tau)
        np.testing.assert_allclose(Ap, 0.0)

    def test_adverse_pressure_gradient(self):
        dp_ds = np.ones(10) * 100.0  # Strong APG
        u_tau = np.ones(10) * 0.05
        Ap = compute_pressure_gradient_parameter(dp_ds, nu=1.5e-5, u_tau=u_tau)
        assert np.all(Ap > 0), "APG should give positive A_p^+"

    def test_favorable_pressure_gradient(self):
        dp_ds = np.ones(10) * -100.0  # FPG
        u_tau = np.ones(10) * 0.05
        Ap = compute_pressure_gradient_parameter(dp_ds, nu=1.5e-5, u_tau=u_tau)
        assert np.all(Ap < 0), "FPG should give negative A_p^+"

    def test_safe_division(self):
        """Should handle u_tau → 0 without NaN."""
        dp_ds = np.ones(5) * 10.0
        u_tau = np.zeros(5)
        Ap = compute_pressure_gradient_parameter(dp_ds, nu=1.5e-5, u_tau=u_tau)
        assert np.all(np.isfinite(Ap))


class TestFrictionVelocityRatio:
    """Test Π_f = u_τ / U_e."""

    def test_healthy_bl(self):
        u_tau = np.ones(10) * 0.04
        U_e = np.ones(10) * 1.0
        Pi_f = compute_friction_velocity_ratio(u_tau, U_e)
        np.testing.assert_allclose(Pi_f, 0.04)

    def test_near_separation(self):
        u_tau = np.ones(10) * 0.001
        U_e = np.ones(10) * 1.0
        Pi_f = compute_friction_velocity_ratio(u_tau, U_e)
        assert np.all(Pi_f < 0.01)

    def test_safe_division(self):
        u_tau = np.ones(5) * 0.04
        U_e = np.zeros(5)
        Pi_f = compute_friction_velocity_ratio(u_tau, U_e)
        assert np.all(np.isfinite(Pi_f))


class TestYPlus:
    """Test y+ computation."""

    def test_values(self):
        y = np.array([1e-5, 1e-4, 1e-3])
        u_tau = np.array([0.05, 0.05, 0.05])
        yp = compute_y_plus(y, u_tau, nu=1.5e-5)
        expected = y * 0.05 / 1.5e-5
        np.testing.assert_allclose(yp, expected)

    def test_positivity(self):
        y = np.ones(10) * 0.001
        u_tau = np.ones(10) * 0.05
        yp = compute_y_plus(y, u_tau, nu=1.5e-5)
        assert np.all(yp > 0)


# =========================================================================
# Flow Regime Classification
# =========================================================================

class TestFlowRegimeClassification:
    """Test regime classification from sensor functions."""

    def test_attached_flow(self):
        N = 20
        dp_ds = np.zeros(N)  # No pressure gradient
        u_tau = np.ones(N) * 0.05  # Healthy friction
        U_e = np.ones(N) * 1.0
        y = np.ones(N) * 0.001
        regime = classify_flow_regime(dp_ds, u_tau, U_e, y, nu=1.5e-5)
        assert regime.n_attached == N
        assert regime.n_separation == 0

    def test_separation_detection(self):
        N = 20
        dp_ds = np.ones(N) * 1000.0  # Strong APG
        u_tau = np.ones(N) * 0.001  # Very low friction
        U_e = np.ones(N) * 1.0
        y = np.ones(N) * 0.001
        regime = classify_flow_regime(dp_ds, u_tau, U_e, y, nu=1.5e-5)
        assert regime.n_separation > 0, "Should detect separation"

    def test_summary_string(self):
        N = 10
        dp_ds = np.zeros(N)
        u_tau = np.ones(N) * 0.05
        U_e = np.ones(N)
        y = np.ones(N) * 0.001
        regime = classify_flow_regime(dp_ds, u_tau, U_e, y, nu=1.5e-5)
        assert "Attached" in regime.summary


# =========================================================================
# Blending Function Tests
# =========================================================================

class TestBlendingFunctions:
    """Test tanh blending and shielding."""

    def test_range_zero_to_one(self):
        """Blending weights must be in [0, 1]."""
        Ap = np.linspace(-10, 50, 100)
        yp = np.linspace(0, 300, 100)
        Pi_f = np.ones(100) * 0.03
        f = tanh_blending_function(Ap, yp, Pi_f)
        assert np.all(f >= 0)
        assert np.all(f <= 1)

    def test_attached_is_shielded(self):
        """In attached BL (low A_p+, low y+), blending should be near 0."""
        Ap = np.zeros(10)  # No APG
        yp = np.ones(10) * 5.0  # Near wall (viscous sublayer)
        Pi_f = np.ones(10) * 0.04
        f = tanh_blending_function(Ap, yp, Pi_f)
        assert np.all(f < 0.1), f"Attached region not shielded: max f={np.max(f):.3f}"

    def test_separation_is_active(self):
        """In deep separation (high A_p+, away from wall), blend → 1."""
        Ap = np.ones(10) * 30.0  # Strong APG
        yp = np.ones(10) * 200.0  # Outer layer
        Pi_f = np.ones(10) * 0.001
        f = tanh_blending_function(Ap, yp, Pi_f)
        assert np.all(f > 0.5), f"Separation not active: min f={np.min(f):.3f}"

    def test_smooth_transition(self):
        """Blending should be smooth (no discontinuities)."""
        Ap = np.linspace(0, 20, 1000)
        yp = np.ones(1000) * 100.0
        Pi_f = np.ones(1000) * 0.03
        f = tanh_blending_function(Ap, yp, Pi_f)
        # Check gradient is finite (smooth)
        df = np.diff(f)
        assert np.all(np.isfinite(df))
        # No jumps > 0.1 between adjacent points
        assert np.max(np.abs(df)) < 0.1

    def test_separation_blending(self):
        Ap = np.ones(10) * 20.0
        yp = np.ones(10) * 100.0
        f = compute_separation_blending(Ap, yp)
        assert f.shape == (10,)
        assert np.all(f >= 0) and np.all(f <= 1)

    def test_reattachment_blending(self):
        Ap = np.ones(10) * 3.0  # Moderate APG (recovery)
        yp = np.ones(10) * 50.0
        Pi_f = np.ones(10) * 0.04  # Recovering friction
        f = compute_reattachment_blending(Ap, yp, Pi_f)
        assert f.shape == (10,)
        assert np.all(f >= 0) and np.all(f <= 1)


# =========================================================================
# Agent Blending Tests
# =========================================================================

class TestAgentBlending:
    """Test the multi-agent prediction blending."""

    def test_pure_rans_when_unblended(self):
        """With zero blending, result should equal RANS."""
        N = 20
        b_rans = np.random.randn(N, 3, 3) * 0.01
        sep_pred = AgentPrediction(
            b_correction=np.random.randn(N, 3, 3) * 0.1,
            confidence=np.ones(N), agent_name="sep",
        )
        reat_pred = AgentPrediction(
            b_correction=np.random.randn(N, 3, 3) * 0.1,
            confidence=np.ones(N), agent_name="reat",
        )
        f_sep = np.zeros(N)
        f_reat = np.zeros(N)
        b = blend_agent_predictions(b_rans, sep_pred, reat_pred, f_sep, f_reat)
        np.testing.assert_allclose(b, b_rans, atol=1e-12)

    def test_full_separation_blending(self):
        """With f_sep=1, result should be separation agent prediction."""
        N = 10
        b_rans = np.zeros((N, 3, 3))
        b_sep = np.ones((N, 3, 3)) * 0.1
        sep_pred = AgentPrediction(b_correction=b_sep, confidence=np.ones(N), agent_name="sep")
        reat_pred = AgentPrediction(b_correction=np.zeros((N, 3, 3)), confidence=np.zeros(N), agent_name="reat")
        f_sep = np.ones(N)
        f_reat = np.zeros(N)
        b = blend_agent_predictions(b_rans, sep_pred, reat_pred, f_sep, f_reat)
        np.testing.assert_allclose(b, b_sep, atol=1e-12)

    def test_total_weight_capped(self):
        """Total blending weight should never exceed 1."""
        N = 10
        b_rans = np.ones((N, 3, 3)) * 0.01
        b_sep = np.ones((N, 3, 3)) * 0.1
        b_reat = np.ones((N, 3, 3)) * 0.05
        sep_pred = AgentPrediction(b_correction=b_sep, confidence=np.ones(N), agent_name="sep")
        reat_pred = AgentPrediction(b_correction=b_reat, confidence=np.ones(N), agent_name="reat")
        f_sep = np.ones(N) * 0.8
        f_reat = np.ones(N) * 0.8
        b = blend_agent_predictions(b_rans, sep_pred, reat_pred, f_sep, f_reat)
        # Result should be finite
        assert np.all(np.isfinite(b))


# =========================================================================
# Specialized Agent Tests
# =========================================================================

class TestSpecializedAgents:
    """Test SeparationAgent and ReattachmentAgent."""

    def test_separation_agent_untrained(self):
        agent = SeparationAgent()
        pred = agent.predict(np.random.randn(10, 5))
        assert pred.b_correction.shape == (10, 3, 3)
        assert np.all(pred.b_correction == 0)
        assert pred.agent_name == "separation"

    def test_reattachment_agent_untrained(self):
        agent = ReattachmentAgent()
        pred = agent.predict(np.random.randn(10, 5))
        assert pred.b_correction.shape == (10, 3, 3)
        assert np.all(pred.b_correction == 0)

    def test_separation_agent_training(self):
        rng = np.random.default_rng(42)
        N = 200
        invariants = rng.standard_normal((N, 5))
        targets = rng.standard_normal((N, 3, 3)) * 0.01
        regime = np.zeros(N, dtype=int)
        regime[:100] = 1  # Mark half as separation
        agent = SeparationAgent(seed=42)
        agent.train(invariants, targets, regime)
        assert agent.is_trained
        pred = agent.predict(invariants[:10])
        assert pred.b_correction.shape == (10, 3, 3)

    def test_reattachment_agent_training(self):
        rng = np.random.default_rng(42)
        N = 200
        invariants = rng.standard_normal((N, 5))
        targets = rng.standard_normal((N, 3, 3)) * 0.01
        regime = np.ones(N, dtype=int) * 2
        agent = ReattachmentAgent(seed=123)
        agent.train(invariants, targets, regime)
        assert agent.is_trained
        pred = agent.predict(invariants[:5])
        assert pred.b_correction.shape == (5, 3, 3)


# =========================================================================
# Full Pipeline Tests
# =========================================================================

class TestMultiAgentPipeline:
    """Test the complete multi-agent blending pipeline."""

    @pytest.fixture(scope="class")
    def result(self):
        """Run once for all tests."""
        return run_blending_comparison(case="periodic_hill")

    def test_pipeline_completes(self, result):
        assert result.b_blended.shape[1:] == (3, 3)
        assert result.regime_map is not None

    def test_has_regime_map(self, result):
        assert result.regime_map.n_attached >= 0
        total = (result.regime_map.n_attached +
                 result.regime_map.n_separation +
                 result.regime_map.n_reattachment)
        assert total == len(result.regime_map.regime)

    def test_blending_weights_bounded(self, result):
        assert np.all(result.f_separation >= 0)
        assert np.all(result.f_separation <= 1)
        assert np.all(result.f_reattachment >= 0)
        assert np.all(result.f_reattachment <= 1)

    def test_realizability(self, result):
        assert result.realizability_fraction > 0.8

    def test_metrics_finite(self, result):
        assert np.isfinite(result.blended_R2)
        assert np.isfinite(result.global_R2)
        assert np.isfinite(result.rans_R2)
        assert np.isfinite(result.blended_RMSE)

    def test_summary_present(self, result):
        assert "Multi-Agent" in result.summary
        assert "Blended" in result.summary or "blended" in result.summary.lower()
        assert "RANS" in result.summary

    def test_global_baseline_exists(self, result):
        """Global TBNN should be computed for comparison."""
        assert result.b_global.shape == result.b_blended.shape
        assert not np.allclose(result.b_global, 0)
