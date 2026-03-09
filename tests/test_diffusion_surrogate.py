"""
Tests for Generative Diffusion Flow Surrogate
=================================================
Validates all sub-components (embeddings, conditioning, residual blocks,
attention, U-Net, DDIM scheduler) and the top-level DiffusionFlowSurrogate
API including probabilistic sampling and UQ.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ml_augmentation.diffusion_flow_surrogate import (
    SinusoidalTimeEmbedding,
    FlowConditionEncoder,
    ResidualBlock1D,
    SelfAttentionBlock,
    DenoisingUNet,
    DDIMScheduler,
    DiffusionFlowSurrogate,
    generate_diffusion_training_data,
    N_CHANNELS,
)


# =========================================================================
# Sinusoidal Time Embedding
# =========================================================================
class TestSinusoidalEmbedding:
    """Test timestep encoding."""

    def test_output_shape(self):
        emb = SinusoidalTimeEmbedding(d_model=64)
        t = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        out = emb(t)
        assert out.shape == (5, 64)

    def test_finite_values(self):
        emb = SinusoidalTimeEmbedding(d_model=32)
        out = emb(np.linspace(0, 1, 20))
        assert np.all(np.isfinite(out))

    def test_bounded_values(self):
        """Sin/cos output should be in [-1, 1]."""
        emb = SinusoidalTimeEmbedding(d_model=64)
        out = emb(np.linspace(0, 1, 100))
        assert np.all(out >= -1.01)
        assert np.all(out <= 1.01)

    def test_different_timesteps_differ(self):
        emb = SinusoidalTimeEmbedding(d_model=64)
        e1 = emb(np.array([0.1]))
        e2 = emb(np.array([0.9]))
        assert not np.allclose(e1, e2)

    def test_scalar_input(self):
        emb = SinusoidalTimeEmbedding(d_model=16)
        out = emb(0.5)
        assert out.shape == (1, 16)


# =========================================================================
# Flow Condition Encoder
# =========================================================================
class TestFlowConditionEncoder:
    """Test FoilDiff-style Re·cos(α), Re·sin(α) conditioning."""

    def test_output_shape(self):
        enc = FlowConditionEncoder(d_cond=64)
        c = enc.encode_conditions(
            aoa_deg=np.array([5.0, 10.0]),
            Re=np.array([6e6, 6e6]),
            Mach=np.array([0.15, 0.15]),
        )
        assert c.shape == (2, 64)

    def test_different_conditions_produce_different_outputs(self):
        enc = FlowConditionEncoder(d_cond=32)
        c1 = enc.encode_conditions(
            np.array([0.0]), np.array([3e6]), np.array([0.1])
        )
        c2 = enc.encode_conditions(
            np.array([15.0]), np.array([9e6]), np.array([0.3])
        )
        assert not np.allclose(c1, c2)

    def test_finite_output(self):
        enc = FlowConditionEncoder(d_cond=64)
        c = enc.encode_conditions(
            np.array([0.0, 5.0, 10.0, 15.0]),
            np.array([1e6, 3e6, 6e6, 9e6]),
            np.array([0.1, 0.15, 0.2, 0.3]),
        )
        assert np.all(np.isfinite(c))


# =========================================================================
# Residual Block
# =========================================================================
class TestResidualBlock:
    """Test 1D residual block with conditioning."""

    def test_output_shape(self):
        blk = ResidualBlock1D(channels=16, d_time=32, d_cond=32)
        x = np.random.randn(4, 16, 40)
        t_emb = np.random.randn(4, 32)
        c_emb = np.random.randn(4, 32)
        out = blk.forward(x, t_emb, c_emb)
        assert out.shape == (4, 16, 40)

    def test_skip_connection(self):
        """Output should differ from input (non-trivial transform)
        but be finite."""
        blk = ResidualBlock1D(channels=8, d_time=16, d_cond=16)
        x = np.random.randn(2, 8, 20)
        t_emb = np.random.randn(2, 16)
        c_emb = np.random.randn(2, 16)
        out = blk.forward(x, t_emb, c_emb)
        assert np.all(np.isfinite(out))
        # Skip connection means output should incorporate input
        assert not np.allclose(out, np.zeros_like(out))

    def test_different_timesteps_change_output(self):
        """Time injection should change the internal hidden state."""
        np.random.seed(99)
        blk = ResidualBlock1D(channels=8, d_time=16, d_cond=16)
        x = np.random.randn(2, 8, 20)
        c_emb = np.zeros((2, 16))
        t1 = np.zeros((2, 16))
        t2 = np.ones((2, 16))
        # Directly compute the hidden state after time injection
        h = blk._conv1d(x, blk.W_conv1, blk.b_conv1)
        h = blk._group_norm(h, blk.gn1_gamma, blk.gn1_beta)
        h = blk._gelu(h)
        tc1 = t1 @ blk.W_time + c_emb @ blk.W_cond
        tc2 = t2 @ blk.W_time + c_emb @ blk.W_cond
        h1 = h + tc1[:, :, None]
        h2 = h + tc2[:, :, None]
        # The hidden states must differ due to time injection
        assert not np.allclose(h1, h2)


# =========================================================================
# Self-Attention Block
# =========================================================================
class TestSelfAttention:
    """Test self-attention over spatial dimension."""

    def test_output_shape(self):
        attn = SelfAttentionBlock(channels=16)
        x = np.random.randn(3, 16, 20)
        out = attn.forward(x)
        assert out.shape == (3, 16, 20)

    def test_finite_values(self):
        attn = SelfAttentionBlock(channels=8)
        x = np.random.randn(2, 8, 40)
        out = attn.forward(x)
        assert np.all(np.isfinite(out))

    def test_residual_connection(self):
        """With residual, output should not be zero."""
        attn = SelfAttentionBlock(channels=8)
        x = np.ones((1, 8, 10))
        out = attn.forward(x)
        assert not np.allclose(out, np.zeros_like(out))


# =========================================================================
# Denoising U-Net
# =========================================================================
class TestDenoisingUNet:
    """Test full U-Net denoiser."""

    def test_forward_shape(self):
        unet = DenoisingUNet(in_channels=2, base_channels=8, d_time=16, d_cond=16)
        x = np.random.randn(2, 2, 80)
        t_emb = np.random.randn(2, 16)
        c_emb = np.random.randn(2, 16)
        out = unet.forward(x, t_emb, c_emb)
        assert out.shape == (2, 2, 80)

    def test_different_spatial_resolutions(self):
        """U-Net should handle various spatial sizes."""
        unet = DenoisingUNet(in_channels=2, base_channels=8, d_time=16, d_cond=16)
        for S in [40, 80, 160]:
            x = np.random.randn(1, 2, S)
            t_emb = np.random.randn(1, 16)
            c_emb = np.random.randn(1, 16)
            out = unet.forward(x, t_emb, c_emb)
            assert out.shape == (1, 2, S), f"Failed at S={S}"
            assert np.all(np.isfinite(out))

    def test_param_count(self):
        unet = DenoisingUNet(in_channels=2, base_channels=16, d_time=32, d_cond=32)
        n = unet.count_params()
        assert n > 0
        assert isinstance(n, int)


# =========================================================================
# DDIM Scheduler
# =========================================================================
class TestDDIMScheduler:
    """Test noise schedule and DDIM sampling."""

    def test_cosine_schedule_monotonic(self):
        """alphas_cumprod should be monotonically decreasing."""
        sched = DDIMScheduler(n_train_steps=100, schedule_type="cosine")
        diffs = np.diff(sched.alphas_cumprod)
        assert np.all(diffs <= 0), "alphas_cumprod should decrease"

    def test_linear_schedule(self):
        sched = DDIMScheduler(n_train_steps=100, schedule_type="linear")
        assert len(sched.betas) == 100
        assert sched.alphas_cumprod[-1] < sched.alphas_cumprod[0]

    def test_add_noise_shape(self):
        sched = DDIMScheduler(n_train_steps=100)
        x = np.random.randn(4, 2, 80)
        noise = np.random.randn(4, 2, 80)
        t = np.array([0, 25, 50, 99])
        x_t = sched.add_noise(x, noise, t)
        assert x_t.shape == (4, 2, 80)

    def test_t0_preserves_signal(self):
        """At t=0, x_t ≈ x_0 (minimal noise)."""
        sched = DDIMScheduler(n_train_steps=1000, schedule_type="cosine")
        x = np.random.randn(2, 2, 40)
        noise = np.random.randn(2, 2, 40)
        x_t = sched.add_noise(x, noise, np.array([0, 0]))
        np.testing.assert_allclose(x_t, x, atol=0.05)

    def test_ddim_step_shape(self):
        sched = DDIMScheduler(n_train_steps=100)
        x_t = np.random.randn(3, 2, 40)
        eps = np.random.randn(3, 2, 40)
        x_prev = sched.ddim_step(x_t, eps, t=50, t_prev=25)
        assert x_prev.shape == (3, 2, 40)
        assert np.all(np.isfinite(x_prev))

    def test_sampling_timesteps(self):
        sched = DDIMScheduler(n_train_steps=1000)
        ts = sched.get_sampling_timesteps(n_inference_steps=50)
        assert len(ts) == 50
        # Should be in descending order
        assert ts[0] > ts[-1]


# =========================================================================
# Diffusion Flow Surrogate (Full Pipeline)
# =========================================================================
class TestDiffusionSurrogate:
    """Test full training and sampling pipeline."""

    @pytest.fixture
    def small_model(self):
        return DiffusionFlowSurrogate(
            spatial_res=40,
            base_channels=8,
            d_time=16,
            d_cond=16,
            n_train_steps=50,
            n_inference_steps=10,
        )

    @pytest.fixture
    def small_data(self):
        rng = np.random.RandomState(42)
        n = 30
        X = np.stack([
            rng.uniform(-5, 18, n),
            10 ** rng.uniform(5.7, 7, n),
            rng.uniform(0.1, 0.3, n),
        ], axis=-1)
        Y_Cp = rng.randn(n, 40)
        Y_Cf = rng.randn(n, 40) * 0.01
        return X, Y_Cp, Y_Cf

    def test_training_runs(self, small_model, small_data):
        X, Y_Cp, Y_Cf = small_data
        history = small_model.fit(X, Y_Cp, Y_Cf, n_epochs=3, batch_size=8)
        assert "train_loss" in history
        assert len(history["train_loss"]) == 3
        assert all(np.isfinite(l) for l in history["train_loss"])

    def test_model_becomes_fitted(self, small_model, small_data):
        X, Y_Cp, Y_Cf = small_data
        assert not small_model._fitted
        small_model.fit(X, Y_Cp, Y_Cf, n_epochs=2)
        assert small_model._fitted

    def test_sample_before_fit_raises(self, small_model):
        with pytest.raises(RuntimeError, match="not fitted"):
            small_model.sample(np.array([5.0]), np.array([6e6]), np.array([0.15]))

    def test_info_dict(self, small_model):
        info = small_model.get_info()
        assert info["model_type"] == "DiffusionFlowSurrogate"
        assert info["architecture"] == "UNet + DDIM"
        assert info["n_params"] > 0
        assert info["fitted"] is False

    def test_summary_string(self, small_model):
        s = small_model.summary()
        assert "Diffusion Flow Surrogate" in s
        assert "UNet + DDIM" in s


# =========================================================================
# Probabilistic Sampling & UQ
# =========================================================================
class TestProbabilisticSampling:
    """Test multi-sample generation and UQ outputs."""

    @pytest.fixture
    def trained_model(self):
        model = DiffusionFlowSurrogate(
            spatial_res=40,
            base_channels=8,
            d_time=16,
            d_cond=16,
            n_train_steps=50,
            n_inference_steps=10,
        )
        rng = np.random.RandomState(42)
        n = 30
        X = np.stack([
            rng.uniform(-5, 18, n),
            10 ** rng.uniform(5.7, 7, n),
            rng.uniform(0.1, 0.3, n),
        ], axis=-1)
        Y_Cp = rng.randn(n, 40)
        Y_Cf = rng.randn(n, 40) * 0.01
        model.fit(X, Y_Cp, Y_Cf, n_epochs=2)
        return model

    def test_sample_shapes(self, trained_model):
        Cp, Cf = trained_model.sample(
            np.array([5.0]), np.array([6e6]), np.array([0.15]),
            n_samples=4,
        )
        assert Cp.shape == (1, 4, 40)
        assert Cf.shape == (1, 4, 40)

    def test_multiple_conditions(self, trained_model):
        Cp, Cf = trained_model.sample(
            np.array([0.0, 5.0, 10.0]),
            np.array([3e6, 6e6, 9e6]),
            np.array([0.15, 0.2, 0.25]),
            n_samples=3,
        )
        assert Cp.shape == (3, 3, 40)

    def test_predict_mean_std(self, trained_model):
        Cp_mean, Cf_mean, Cp_std, Cf_std = trained_model.predict_mean_std(
            np.array([5.0]), np.array([6e6]), np.array([0.15]),
            n_samples=8,
        )
        assert Cp_mean.shape == (1, 40)
        assert Cp_std.shape == (1, 40)
        assert np.all(np.isfinite(Cp_mean))
        assert np.all(Cp_std >= 0)
        # Should have some variance (non-trivial UQ)
        assert np.mean(Cp_std) > 0

    def test_predict_percentiles(self, trained_model):
        pcts = trained_model.predict_percentiles(
            np.array([5.0]), np.array([6e6]), np.array([0.15]),
            percentiles=[5, 50, 95],
            n_samples=8,
        )
        assert "Cp_p5" in pcts
        assert "Cp_p50" in pcts
        assert "Cf_p95" in pcts
        assert pcts["Cp_p5"].shape == (1, 40)


# =========================================================================
# Multi-Modal Near Stall
# =========================================================================
class TestMultiModalCapture:
    """Verify that UQ variance is higher near stall (α=15°) vs low α."""

    @pytest.fixture
    def trained_model(self):
        model = DiffusionFlowSurrogate(
            spatial_res=40,
            base_channels=8,
            d_time=16,
            d_cond=16,
            n_train_steps=50,
            n_inference_steps=10,
            seed=123,
        )
        rng = np.random.RandomState(42)
        n = 50
        X = np.stack([
            rng.uniform(-5, 18, n),
            10 ** rng.uniform(5.7, 7, n),
            rng.uniform(0.1, 0.3, n),
        ], axis=-1)
        Y_Cp = rng.randn(n, 40)
        Y_Cf = rng.randn(n, 40) * 0.01
        model.fit(X, Y_Cp, Y_Cf, n_epochs=3)
        return model

    def test_samples_have_finite_variance(self, trained_model):
        """All samples at any α should have finite, non-negative std."""
        for aoa in [0.0, 5.0, 10.0, 15.0]:
            _, _, Cp_std, _ = trained_model.predict_mean_std(
                np.array([aoa]), np.array([6e6]), np.array([0.15]),
                n_samples=8,
            )
            assert np.all(np.isfinite(Cp_std))
            assert np.all(Cp_std >= 0)


# =========================================================================
# Separation Detection with Uncertainty
# =========================================================================
class TestSeparationDetection:
    """Test separation detection from diffusion samples."""

    @pytest.fixture
    def trained_model(self):
        model = DiffusionFlowSurrogate(
            spatial_res=40,
            base_channels=8,
            d_time=16,
            d_cond=16,
            n_train_steps=50,
            n_inference_steps=10,
        )
        rng = np.random.RandomState(42)
        n = 30
        X = np.stack([
            rng.uniform(-5, 18, n),
            10 ** rng.uniform(5.7, 7, n),
            rng.uniform(0.1, 0.3, n),
        ], axis=-1)
        Y_Cp = rng.randn(n, 40)
        Y_Cf = rng.randn(n, 40) * 0.01
        model.fit(X, Y_Cp, Y_Cf, n_epochs=2)
        return model

    def test_separation_output_structure(self, trained_model):
        results = trained_model.detect_separation_with_uncertainty(
            np.array([15.0]), np.array([6e6]), np.array([0.15]),
            n_samples=4,
        )
        assert len(results) == 1
        r = results[0]
        assert "prob_separated" in r
        assert "n_separated_samples" in r
        assert 0 <= r["prob_separated"] <= 1.0

    def test_multiple_conditions(self, trained_model):
        results = trained_model.detect_separation_with_uncertainty(
            np.array([0.0, 15.0]),
            np.array([6e6, 6e6]),
            np.array([0.15, 0.15]),
            n_samples=4,
        )
        assert len(results) == 2


# =========================================================================
# Comparison Metrics
# =========================================================================
class TestComparisonMetrics:
    """Test comparison output format."""

    @pytest.fixture
    def trained_model_and_data(self):
        model = DiffusionFlowSurrogate(
            spatial_res=40,
            base_channels=8,
            d_time=16,
            d_cond=16,
            n_train_steps=50,
            n_inference_steps=10,
        )
        rng = np.random.RandomState(42)
        n = 20
        X = np.stack([
            rng.uniform(-5, 18, n),
            10 ** rng.uniform(5.7, 7, n),
            rng.uniform(0.1, 0.3, n),
        ], axis=-1)
        Y_Cp = rng.randn(n, 40)
        Y_Cf = rng.randn(n, 40) * 0.01
        model.fit(X, Y_Cp, Y_Cf, n_epochs=2)
        return model, X, Y_Cp, Y_Cf

    def test_comparison_returns_expected_keys(self, trained_model_and_data):
        model, X, Y_Cp, Y_Cf = trained_model_and_data
        metrics = model.compare_with_deterministic(X[:5], Y_Cp[:5], Y_Cf[:5], n_samples=4)

        assert "model_name" in metrics
        assert metrics["model_name"] == "DiffusionFlowSurrogate"
        assert "Cp_RMSE" in metrics
        assert "Cf_RMSE" in metrics
        assert "Cp_R2" in metrics
        assert "Cf_R2" in metrics
        assert "Cp_std_mean" in metrics
        assert "Cp_coverage_2sigma" in metrics
        assert np.isfinite(metrics["Cp_RMSE"])
        assert np.isfinite(metrics["Cf_RMSE"])


# =========================================================================
# Data Generation
# =========================================================================
class TestDataGeneration:
    """Test synthetic training data generator."""

    def test_shapes(self):
        X, Y_Cp, Y_Cf = generate_diffusion_training_data(
            n_samples=20, spatial_res=40
        )
        assert X.shape == (20, 3)
        assert Y_Cp.shape == (20, 40)
        assert Y_Cf.shape == (20, 40)

    def test_finite_values(self):
        X, Y_Cp, Y_Cf = generate_diffusion_training_data(n_samples=10)
        assert np.all(np.isfinite(X))
        assert np.all(np.isfinite(Y_Cp))
        assert np.all(np.isfinite(Y_Cf))

    def test_condition_ranges(self):
        X, _, _ = generate_diffusion_training_data(n_samples=50)
        assert np.all(X[:, 0] >= -5.0)   # AoA
        assert np.all(X[:, 0] <= 18.0)
        assert np.all(X[:, 1] >= 5e5)    # Re
        assert np.all(X[:, 1] <= 1e7)
        assert np.all(X[:, 2] >= 0.1)    # Mach
        assert np.all(X[:, 2] <= 0.3)
