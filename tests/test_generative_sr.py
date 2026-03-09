"""Tests for Generative Super-Resolution flow field reconstruction."""
import numpy as np
import pytest
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))


class TestFlowFieldVoxelizer:
    def test_voxelize_basic(self):
        from scripts.ml_augmentation.generative_super_resolution import FlowFieldVoxelizer
        vox = FlowFieldVoxelizer(grid_shape=(8, 8, 8))
        rng = np.random.RandomState(42)
        coords = rng.rand(100, 3)
        fields = {'u': rng.randn(100), 'v': rng.randn(100)}
        result = vox.voxelize(coords, fields)
        assert result['u'].shape == (8, 8, 8)
        assert result['v'].shape == (8, 8, 8)
        assert '_counts' in result

    def test_voxelize_2d(self):
        from scripts.ml_augmentation.generative_super_resolution import FlowFieldVoxelizer
        vox = FlowFieldVoxelizer(grid_shape=(8, 8, 8))
        coords = np.random.rand(50, 2)
        fields = {'p': np.random.randn(50)}
        result = vox.voxelize(coords, fields)
        assert result['p'].shape == (8, 8, 8)

    def test_devoxelize(self):
        from scripts.ml_augmentation.generative_super_resolution import FlowFieldVoxelizer
        vox = FlowFieldVoxelizer(grid_shape=(4, 4, 4))
        field = np.ones((4, 4, 4)) * 3.0
        coords = np.array([[0.5, 0.5, 0.5], [0.1, 0.1, 0.1]])
        bounds = ((0, 1), (0, 1), (0, 1))
        vals = vox.devoxelize(field, coords, bounds)
        np.testing.assert_allclose(vals, [3.0, 3.0])


class TestPhysicsInformedGenerator:
    def test_forward_shape(self):
        from scripts.ml_augmentation.generative_super_resolution import (
            PhysicsInformedGenerator, SuperResolutionConfig
        )
        cfg = SuperResolutionConfig(
            coarse_shape=(4, 4, 4), fine_shape=(16, 16, 16), n_channels=4
        )
        gen = PhysicsInformedGenerator(cfg)
        coarse = np.random.randn(4, 4, 4, 4)
        fine = gen.forward(coarse)
        assert fine.shape == (4, 16, 16, 16)

    def test_spectral_compliance(self):
        from scripts.ml_augmentation.generative_super_resolution import (
            PhysicsInformedGenerator, SuperResolutionConfig
        )
        cfg = SuperResolutionConfig(
            coarse_shape=(4, 4, 4), fine_shape=(16, 16, 16)
        )
        gen = PhysicsInformedGenerator(cfg)
        coarse = np.random.randn(4, 4, 4, 4) * 10
        fine = gen.forward(coarse)
        err = gen.compute_spectral_error(fine[:3])
        assert isinstance(err, float)
        assert err >= 0

    def test_continuity_residual(self):
        from scripts.ml_augmentation.generative_super_resolution import (
            PhysicsInformedGenerator, SuperResolutionConfig
        )
        cfg = SuperResolutionConfig(
            coarse_shape=(4, 4, 4), fine_shape=(16, 16, 16)
        )
        gen = PhysicsInformedGenerator(cfg)
        velocity = np.random.randn(3, 16, 16, 16) * 0.01
        res = gen.compute_continuity_residual(velocity)
        assert isinstance(res, float)
        assert res >= 0


class TestTBNNConditionedDiffusion:
    def test_encode_anisotropy(self):
        from scripts.ml_augmentation.generative_super_resolution import (
            TBNNConditionedDiffusion
        )
        cond = TBNNConditionedDiffusion(anisotropy_dim=6, latent_dim=32)
        # Create anisotropy tensors (symmetric, traceless)
        b = np.zeros((10, 3, 3))
        for i in range(10):
            b[i] = np.eye(3) / 3 * (i * 0.1 - 0.5)  # Varying anisotropy
        result = cond.encode_anisotropy(b)
        assert result.shape == (10, 32)
        assert np.all(np.abs(result) <= 1)  # tanh output

    def test_anisotropy_intensity(self):
        from scripts.ml_augmentation.generative_super_resolution import (
            TBNNConditionedDiffusion
        )
        cond = TBNNConditionedDiffusion()
        # Isotropic case: b_ij = 0
        iso = np.zeros((5, 3, 3))
        intensity_iso = cond.compute_anisotropy_intensity(iso)
        np.testing.assert_allclose(intensity_iso, 0)

        # Anisotropic case
        aniso = np.zeros((5, 3, 3))
        aniso[:, 0, 0] = 0.3
        intensity_aniso = cond.compute_anisotropy_intensity(aniso)
        assert np.all(intensity_aniso > 0)


class TestRANStoLESMapper:
    def test_reconstruct(self):
        from scripts.ml_augmentation.generative_super_resolution import (
            RANStoLESMapper, SuperResolutionConfig
        )
        cfg = SuperResolutionConfig(
            coarse_shape=(4, 4, 4), fine_shape=(16, 16, 16)
        )
        mapper = RANStoLESMapper(cfg)
        vel = np.random.randn(3, 4, 4, 4)
        pres = np.random.randn(4, 4, 4)
        result = mapper.reconstruct(vel, pres)
        assert result.hr_velocity.shape == (3, 16, 16, 16)
        assert result.upscale_factor == 4
        assert isinstance(result.spectral_error, float)
        assert isinstance(result.continuity_residual, float)

    def test_summary(self):
        from scripts.ml_augmentation.generative_super_resolution import (
            RANStoLESMapper, SuperResolutionConfig
        )
        cfg = SuperResolutionConfig(
            coarse_shape=(8, 8, 8), fine_shape=(32, 32, 32)
        )
        mapper = RANStoLESMapper(cfg)
        s = mapper.summary()
        assert s["upscale_factor"] == 4
        assert s["n_channels"] == 4


class TestDiscriminator:
    def test_discriminate(self):
        from scripts.ml_augmentation.generative_super_resolution import (
            SuperResolutionDiscriminator
        )
        disc = SuperResolutionDiscriminator(n_channels=4)
        field = np.random.randn(4, 8, 8, 8)
        score = disc.discriminate(field)
        assert 0 <= score <= 1
