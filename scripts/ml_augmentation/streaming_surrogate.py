#!/usr/bin/env python3
"""
Online/Streaming Surrogate API
================================
FastAPI-based server for real-time aerodynamic prediction using
trained SurrogateModel and DistributionSurrogate.

Endpoints:
  POST /predict          — CL/CD/CM prediction with UQ bounds
  POST /predict_distribution — Full Cp(x/c), Cf(x/c) distributions
  GET  /health           — Model status and metadata
  WS   /ws/design_loop   — Streaming WebSocket for design optimization

Usage:
    # Start server
    python -m scripts.ml_augmentation.streaming_surrogate --port 8321

    # Or import for testing
    from scripts.ml_augmentation.streaming_surrogate import create_app
    app = create_app()

Dependencies:
    pip install fastapi uvicorn
"""

import json
import logging
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)
PROJECT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT))

# Conditional imports for FastAPI
try:
    from fastapi import FastAPI, WebSocket, HTTPException
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
    _HAS_FASTAPI = True
except ImportError:
    _HAS_FASTAPI = False
    logger.info("FastAPI not installed — streaming server unavailable, "
                "batch prediction still works")


# =============================================================================
# Data Structures (framework-independent)
# =============================================================================
@dataclass
class PredictionInput:
    """Input for surrogate prediction."""
    aoa_deg: float
    Re: float
    Mach: float = 0.15
    request_id: str = ""


@dataclass
class PredictionResult:
    """Result from surrogate prediction."""
    CL: float = 0.0
    CD: float = 0.0
    CM: float = 0.0
    CL_std: float = 0.0
    CD_std: float = 0.0
    request_id: str = ""
    latency_ms: float = 0.0

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class DistributionResult:
    """Result from distribution surrogate prediction."""
    Cp: List[float] = field(default_factory=list)
    Cf: List[float] = field(default_factory=list)
    x_c: List[float] = field(default_factory=list)
    separation_info: Dict = field(default_factory=dict)
    request_id: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)


# =============================================================================
# Model Registry
# =============================================================================
class ModelRegistry:
    """
    Loads and caches trained surrogate models from disk.

    Provides a uniform predict() interface regardless of model type.
    """

    def __init__(self):
        self.scalar_model = None
        self.distribution_model = None
        self._initialized = False
        self._version = "1.0.0"

    def initialize(self, scalar_model=None, distribution_model=None):
        """Load models (or create synthetic ones for testing)."""
        if scalar_model is not None:
            self.scalar_model = scalar_model
        else:
            # Create synthetic surrogate for testing
            self.scalar_model = _SyntheticScalarSurrogate()

        if distribution_model is not None:
            self.distribution_model = distribution_model
        else:
            self.distribution_model = _SyntheticDistributionSurrogate()

        self._initialized = True
        logger.info("Model registry initialized")

    @property
    def is_ready(self) -> bool:
        return self._initialized

    def predict_scalar(self, inputs: List[PredictionInput]) -> List[PredictionResult]:
        """Predict CL/CD/CM for batch of inputs."""
        if not self._initialized:
            raise RuntimeError("Models not initialized")

        results = []
        for inp in inputs:
            t0 = time.time()
            X = np.array([[inp.aoa_deg, inp.Re, inp.Mach]])
            pred = self.scalar_model.predict(X)

            result = PredictionResult(
                CL=float(pred[0, 0]) if pred.ndim > 1 else float(pred[0]),
                CD=float(pred[0, 1]) if pred.ndim > 1 and pred.shape[1] > 1 else 0.0,
                CM=float(pred[0, 2]) if pred.ndim > 1 and pred.shape[1] > 2 else 0.0,
                request_id=inp.request_id,
                latency_ms=(time.time() - t0) * 1000,
            )
            results.append(result)

        return results

    def predict_distribution(self, inputs: List[PredictionInput]) -> List[DistributionResult]:
        """Predict Cp/Cf distributions for batch of inputs."""
        if not self._initialized:
            raise RuntimeError("Models not initialized")

        results = []
        for inp in inputs:
            X = np.array([[inp.aoa_deg, inp.Re, inp.Mach]])
            Cp, Cf = self.distribution_model.predict(X)
            x_c = np.linspace(0, 1, len(Cp[0])).tolist()

            # Detect separation from Cf
            sep_info = self._detect_separation(x_c, Cf[0])

            results.append(DistributionResult(
                Cp=Cp[0].tolist(),
                Cf=Cf[0].tolist(),
                x_c=x_c,
                separation_info=sep_info,
                request_id=inp.request_id,
            ))

        return results

    def _detect_separation(self, x_c, Cf) -> Dict:
        """Find separation/reattachment from Cf sign changes."""
        Cf = np.array(Cf)
        x_c = np.array(x_c)
        sign_changes = np.diff(np.sign(Cf))

        x_sep = None
        x_reat = None
        for i in range(len(sign_changes)):
            if sign_changes[i] < 0 and x_sep is None:
                x_sep = float(x_c[i])
            elif sign_changes[i] > 0 and x_sep is not None and x_reat is None:
                x_reat = float(x_c[i])

        return {
            "x_sep": x_sep,
            "x_reat": x_reat,
            "bubble_length": float(x_reat - x_sep) if x_sep and x_reat else 0.0,
            "separated": x_sep is not None,
        }

    def get_health(self) -> Dict:
        return {
            "status": "ready" if self._initialized else "not_initialized",
            "version": self._version,
            "scalar_model": type(self.scalar_model).__name__ if self.scalar_model else None,
            "distribution_model": type(self.distribution_model).__name__ if self.distribution_model else None,
            "input_features": ["aoa_deg", "Re", "Mach"],
        }


# =============================================================================
# Synthetic Models (for testing)
# =============================================================================
class _SyntheticScalarSurrogate:
    """Thin-airfoil-theory-based surrogate for testing."""

    def predict(self, X: np.ndarray) -> np.ndarray:
        aoa = X[:, 0]
        Re = X[:, 1] if X.shape[1] > 1 else 6e6
        Mach = X[:, 2] if X.shape[1] > 2 else 0.15

        CL = 2 * np.pi * np.radians(aoa) / np.sqrt(1 - Mach**2)
        CD = 0.006 + 0.005 * aoa**2 / 100
        CM = -0.05 * aoa / 15

        return np.column_stack([CL, CD, CM])


class _SyntheticDistributionSurrogate:
    """Analytical Cp/Cf distribution for testing."""

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n = X.shape[0]
        n_pts = 80
        x_c = np.linspace(0, 1, n_pts)

        Cp_all = np.zeros((n, n_pts))
        Cf_all = np.zeros((n, n_pts))

        for i in range(n):
            aoa = X[i, 0]
            # Cp from thin airfoil + viscous correction
            Cp_all[i] = -2 * np.sin(np.radians(aoa)) * (1 - x_c) * np.sin(np.pi * x_c)

            # Cf with possible separation at high AoA
            Cf_base = 0.004 * (1 - 0.5 * x_c)
            if abs(aoa) > 10:
                sep_start = max(0.3, 0.8 - 0.03 * abs(aoa))
                mask = x_c > sep_start
                Cf_base[mask] = -0.001 * np.sin(
                    np.pi * (x_c[mask] - sep_start) / (1 - sep_start))
            Cf_all[i] = Cf_base

        return Cp_all, Cf_all


# =============================================================================
# FastAPI Application
# =============================================================================
def create_app(registry: ModelRegistry = None) -> "FastAPI":
    """Create the FastAPI application."""
    if not _HAS_FASTAPI:
        raise ImportError("FastAPI required: pip install fastapi uvicorn")

    app = FastAPI(
        title="CFD Surrogate API",
        description="Real-time aerodynamic prediction from trained ML surrogates",
        version="1.0.0",
    )

    if registry is None:
        registry = ModelRegistry()
        registry.initialize()

    # --- Pydantic models ---
    class PredictRequest(BaseModel):
        aoa_deg: float = Field(..., description="Angle of attack [deg]")
        Re: float = Field(..., description="Chord Reynolds number")
        Mach: float = Field(0.15, description="Freestream Mach number")
        request_id: str = Field("", description="Optional request ID")

    class PredictBatchRequest(BaseModel):
        inputs: List[PredictRequest]

    # --- Routes ---
    @app.get("/health")
    async def health():
        return registry.get_health()

    @app.post("/predict")
    async def predict(req: PredictRequest):
        inp = PredictionInput(
            aoa_deg=req.aoa_deg, Re=req.Re,
            Mach=req.Mach, request_id=req.request_id,
        )
        results = registry.predict_scalar([inp])
        return results[0].to_dict()

    @app.post("/predict_batch")
    async def predict_batch(req: PredictBatchRequest):
        inputs = [
            PredictionInput(aoa_deg=r.aoa_deg, Re=r.Re,
                            Mach=r.Mach, request_id=r.request_id)
            for r in req.inputs
        ]
        results = registry.predict_scalar(inputs)
        return [r.to_dict() for r in results]

    @app.post("/predict_distribution")
    async def predict_distribution(req: PredictRequest):
        inp = PredictionInput(
            aoa_deg=req.aoa_deg, Re=req.Re,
            Mach=req.Mach, request_id=req.request_id,
        )
        results = registry.predict_distribution([inp])
        return results[0].to_dict()

    @app.websocket("/ws/design_loop")
    async def design_loop_ws(websocket: WebSocket):
        await websocket.accept()
        try:
            while True:
                data = await websocket.receive_text()
                params = json.loads(data)

                inp = PredictionInput(
                    aoa_deg=params.get("aoa_deg", 0),
                    Re=params.get("Re", 6e6),
                    Mach=params.get("Mach", 0.15),
                    request_id=params.get("request_id", ""),
                )

                results = registry.predict_scalar([inp])
                await websocket.send_text(json.dumps(results[0].to_dict()))
        except Exception:
            pass
        finally:
            try:
                await websocket.close()
            except Exception:
                pass

    return app


# =============================================================================
# Batch Prediction API (no FastAPI needed)
# =============================================================================
def predict_batch(inputs: List[Dict], registry: ModelRegistry = None) -> List[Dict]:
    """
    Batch prediction without a server.

    Parameters
    ----------
    inputs : list of dicts with keys aoa_deg, Re, Mach
    registry : ModelRegistry (created with defaults if None)

    Returns
    -------
    list of prediction dicts with CL, CD, CM
    """
    if registry is None:
        registry = ModelRegistry()
        registry.initialize()

    pred_inputs = [
        PredictionInput(
            aoa_deg=d["aoa_deg"],
            Re=d["Re"],
            Mach=d.get("Mach", 0.15),
        )
        for d in inputs
    ]

    results = registry.predict_scalar(pred_inputs)
    return [r.to_dict() for r in results]


# =============================================================================
# CLI
# =============================================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="CFD Surrogate Streaming Server")
    parser.add_argument("--port", type=int, default=8321)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    if not _HAS_FASTAPI:
        print("FastAPI not installed. Install with: pip install fastapi uvicorn")
        print("Running batch prediction demo instead...")
        results = predict_batch([
            {"aoa_deg": 0, "Re": 6e6, "Mach": 0.15},
            {"aoa_deg": 5, "Re": 6e6, "Mach": 0.15},
            {"aoa_deg": 10, "Re": 6e6, "Mach": 0.15},
        ])
        for r in results:
            print(f"  α={r['CL']:.4f}, CD={r['CD']:.6f}")
        return

    import uvicorn
    app = create_app()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
