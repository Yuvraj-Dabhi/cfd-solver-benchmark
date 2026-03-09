"""Tests for SmartSim In-Situ Orchestration module."""
import numpy as np
import pytest
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))


class TestLocalTensorStore:
    def test_put_get(self):
        from scripts.ml_augmentation.smartsim_orchestrator import LocalTensorStore
        store = LocalTensorStore()
        data = np.array([1.0, 2.0, 3.0])
        store.put_tensor("test", data)
        retrieved = store.get_tensor("test")
        np.testing.assert_array_equal(retrieved, data)

    def test_tensor_exists(self):
        from scripts.ml_augmentation.smartsim_orchestrator import LocalTensorStore
        store = LocalTensorStore()
        assert not store.tensor_exists("missing")
        store.put_tensor("exists", np.zeros(5))
        assert store.tensor_exists("exists")

    def test_delete(self):
        from scripts.ml_augmentation.smartsim_orchestrator import LocalTensorStore
        store = LocalTensorStore()
        store.put_tensor("del_me", np.ones(3))
        store.delete_tensor("del_me")
        assert not store.tensor_exists("del_me")

    def test_list_and_clear(self):
        from scripts.ml_augmentation.smartsim_orchestrator import LocalTensorStore
        store = LocalTensorStore()
        store.put_tensor("a", np.zeros(1))
        store.put_tensor("b", np.ones(2))
        assert set(store.list_tensors()) == {"a", "b"}
        store.clear()
        assert len(store.list_tensors()) == 0

    def test_metadata(self):
        from scripts.ml_augmentation.smartsim_orchestrator import LocalTensorStore
        store = LocalTensorStore()
        data = np.ones((3, 4))
        store.put_tensor("meta_test", data)
        meta = store.get_metadata("meta_test")
        assert meta["shape"] == (3, 4)
        assert meta["size_bytes"] == data.nbytes

    def test_memory_usage(self):
        from scripts.ml_augmentation.smartsim_orchestrator import LocalTensorStore
        store = LocalTensorStore()
        store.put_tensor("big", np.zeros(1000))
        assert store.total_memory_bytes() == 1000 * 8


class TestSU2SmartRedisClient:
    def test_local_fallback(self):
        from scripts.ml_augmentation.smartsim_orchestrator import (
            SU2SmartRedisClient, SmartSimConfig
        )
        client = SU2SmartRedisClient(SmartSimConfig())
        assert client.backend in ("local", "smartredis")

    def test_push_flow_state(self):
        from scripts.ml_augmentation.smartsim_orchestrator import (
            SU2SmartRedisClient
        )
        client = SU2SmartRedisClient()
        vel = np.random.randn(100, 3)
        pres = np.random.randn(100)
        client.push_flow_state(0, vel, pres)
        assert client.client.tensor_exists("su2_iter_0_velocity")
        assert client.client.tensor_exists("su2_iter_0_pressure")

    def test_push_with_tke_nut(self):
        from scripts.ml_augmentation.smartsim_orchestrator import SU2SmartRedisClient
        client = SU2SmartRedisClient()
        vel = np.random.randn(50, 3)
        pres = np.random.randn(50)
        tke = np.random.randn(50)
        nu_t = np.random.randn(50)
        client.push_flow_state(1, vel, pres, tke=tke, nu_t=nu_t)
        assert client.client.tensor_exists("su2_iter_1_tke")
        assert client.client.tensor_exists("su2_iter_1_nu_t")


class TestOnlineInferenceManager:
    def test_register_and_run(self):
        from scripts.ml_augmentation.smartsim_orchestrator import (
            SU2SmartRedisClient, OnlineInferenceManager
        )

        class MockModel:
            def forward(self, velocity=None, pressure=None):
                return np.ones(10) * 0.5

        client = SU2SmartRedisClient()
        mgr = OnlineInferenceManager(client)
        mgr.register_model("test_model", MockModel())

        # Push data first
        client.push_flow_state(0, np.random.randn(10, 3), np.random.randn(10))
        result = mgr.run_inference(0, "test_model")
        assert result is not None
        np.testing.assert_allclose(result, 0.5)


class TestMLInLoopPipeline:
    def test_step(self):
        from scripts.ml_augmentation.smartsim_orchestrator import MLInLoopPipeline

        class SimpleModel:
            def forward(self, velocity=None, pressure=None):
                return np.zeros(5)

        pipeline = MLInLoopPipeline()
        pipeline.register_model("corrector", SimpleModel())
        results = pipeline.step(
            velocity=np.random.randn(10, 3),
            pressure=np.random.randn(10),
            model_name="corrector"
        )
        assert "corrector" in results

    def test_summary(self):
        from scripts.ml_augmentation.smartsim_orchestrator import MLInLoopPipeline
        pipeline = MLInLoopPipeline()
        s = pipeline.summary()
        assert "backend" in s
        assert "registered_models" in s
        assert "smartsim_available" in s


class TestStreamingTBNNUpdater:
    def test_online_update(self):
        from scripts.ml_augmentation.smartsim_orchestrator import (
            SU2SmartRedisClient, StreamingTBNNUpdater
        )
        client = SU2SmartRedisClient()
        updater = StreamingTBNNUpdater(client, lr=0.01)

        weights = {"W1": np.ones((3, 3)), "b1": np.zeros(3)}
        predicted = np.array([1.0, 2.0, 3.0])
        target = np.array([1.1, 1.9, 3.2])

        updated = updater.compute_online_update(predicted, target, weights)
        assert "W1" in updated
        assert "b1" in updated
        assert updater.update_count == 1
        # Weights should have changed
        assert not np.array_equal(updated["W1"], weights["W1"])
