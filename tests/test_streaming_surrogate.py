"""
Tests for Online/Streaming Surrogate API (Gap 3)
==================================================
Tests the ModelRegistry, batch prediction, and FastAPI endpoints
using synthetic models (no server required).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ml_augmentation.streaming_surrogate import (
    ModelRegistry,
    PredictionInput,
    PredictionResult,
    DistributionResult,
    predict_batch,
)


class TestModelRegistry:
    """Test model registry and prediction."""

    @pytest.fixture
    def registry(self):
        reg = ModelRegistry()
        reg.initialize()
        return reg

    def test_registry_initializes(self, registry):
        assert registry.is_ready
        health = registry.get_health()
        assert health["status"] == "ready"

    def test_predict_scalar(self, registry):
        inputs = [
            PredictionInput(aoa_deg=5.0, Re=6e6, Mach=0.15),
        ]
        results = registry.predict_scalar(inputs)
        assert len(results) == 1
        r = results[0]
        assert isinstance(r, PredictionResult)
        assert np.isfinite(r.CL)
        assert np.isfinite(r.CD)
        assert r.latency_ms >= 0

    def test_predict_scalar_batch(self, registry):
        inputs = [
            PredictionInput(aoa_deg=a, Re=6e6, Mach=0.15)
            for a in [0, 5, 10, 15]
        ]
        results = registry.predict_scalar(inputs)
        assert len(results) == 4
        # CL should increase with AoA for attached flow
        CLs = [r.CL for r in results]
        assert CLs[1] > CLs[0]

    def test_predict_distribution(self, registry):
        inputs = [
            PredictionInput(aoa_deg=5.0, Re=6e6, Mach=0.15),
        ]
        results = registry.predict_distribution(inputs)
        assert len(results) == 1
        r = results[0]
        assert isinstance(r, DistributionResult)
        assert len(r.Cp) == 80
        assert len(r.Cf) == 80
        assert len(r.x_c) == 80

    def test_predict_distribution_separation(self, registry):
        """High AoA should trigger separation detection."""
        inputs = [
            PredictionInput(aoa_deg=15.0, Re=6e6, Mach=0.15),
        ]
        results = registry.predict_distribution(inputs)
        sep = results[0].separation_info
        assert sep["separated"] is True
        assert sep["x_sep"] is not None

    def test_health_endpoint(self, registry):
        health = registry.get_health()
        assert "status" in health
        assert "version" in health
        assert "input_features" in health
        assert health["input_features"] == ["aoa_deg", "Re", "Mach"]


class TestBatchPrediction:
    """Test batch prediction without FastAPI."""

    def test_predict_endpoint(self):
        """Verify batch prediction returns CL/CD keys."""
        results = predict_batch([
            {"aoa_deg": 0, "Re": 6e6, "Mach": 0.15},
            {"aoa_deg": 5, "Re": 6e6, "Mach": 0.15},
        ])
        assert len(results) == 2
        for r in results:
            assert "CL" in r
            assert "CD" in r
            assert "CM" in r

    def test_predict_distribution_shape(self):
        """Verify distribution output has 80-point arrays."""
        registry = ModelRegistry()
        registry.initialize()
        inputs = [PredictionInput(aoa_deg=5, Re=6e6)]
        results = registry.predict_distribution(inputs)
        assert len(results[0].Cp) == 80
        assert len(results[0].Cf) == 80


class TestPredictionResult:
    """Test data structures."""

    def test_result_to_dict(self):
        r = PredictionResult(CL=0.5, CD=0.01, CM=-0.05)
        d = r.to_dict()
        assert isinstance(d, dict)
        assert d["CL"] == 0.5
        # Should be JSON-serializable
        json_str = json.dumps(d)
        assert len(json_str) > 0

    def test_invalid_input_handled(self):
        """Non-initialized registry should raise."""
        registry = ModelRegistry()
        with pytest.raises(RuntimeError, match="not initialized"):
            registry.predict_scalar([
                PredictionInput(aoa_deg=0, Re=6e6)
            ])


# Optional FastAPI tests (only run if fastapi is installed)
try:
    from fastapi.testclient import TestClient
    from scripts.ml_augmentation.streaming_surrogate import create_app
    _HAS_FASTAPI_TEST = True
except ImportError:
    _HAS_FASTAPI_TEST = False


@pytest.mark.skipif(not _HAS_FASTAPI_TEST,
                    reason="FastAPI not installed")
class TestFastAPIEndpoints:
    """Test FastAPI endpoints using TestClient."""

    @pytest.fixture
    def client(self):
        app = create_app()
        return TestClient(app)

    def test_health_endpoint(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ready"

    def test_predict_endpoint(self, client):
        resp = client.post("/predict", json={
            "aoa_deg": 5.0, "Re": 6e6, "Mach": 0.15,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "CL" in data
        assert "CD" in data

    def test_predict_distribution_endpoint(self, client):
        resp = client.post("/predict_distribution", json={
            "aoa_deg": 5.0, "Re": 6e6, "Mach": 0.15,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["Cp"]) == 80
        assert len(data["Cf"]) == 80

    def test_websocket_round_trip(self, client):
        with client.websocket_connect("/ws/design_loop") as ws:
            for aoa in [0, 5, 10]:
                ws.send_text(json.dumps({
                    "aoa_deg": aoa, "Re": 6e6, "Mach": 0.15,
                    "request_id": f"test_{aoa}",
                }))
                data = json.loads(ws.receive_text())
                assert "CL" in data

    def test_invalid_input_422(self, client):
        """Malformed input should return 422."""
        resp = client.post("/predict", json={"bad_key": 5.0})
        assert resp.status_code == 422
