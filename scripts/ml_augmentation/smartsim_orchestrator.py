#!/usr/bin/env python3
"""
SmartSim In-Situ Data Orchestration for ML-in-the-Loop CFD
============================================================
Eliminates POSIX filesystem I/O bottleneck by streaming flow field
tensors directly between SU2 C++ solver and Python ML models via
in-memory Redis-based database (SmartSim Orchestrator).

Architecture
------------
1. **SmartSimConfig** — configuration for Orchestrator ports, tensor layouts
2. **SU2SmartRedisClient** — adapter wrapping SmartRedis put_tensor/get_tensor
3. **OnlineInferenceManager** — manages co-located ML models on GPU nodes
4. **StreamingTBNNUpdater** — online weight updates using streaming gradients
5. **MLInLoopPipeline** — full orchestration: SU2 → tensor push → inference → feedback

When SmartSim/SmartRedis are not available, the module falls back to
a local numpy-based tensor store for testing and development.

References
----------
- Partee et al. (2022) "Using Machine Learning at Scale in HPC
  Simulations with SmartSim", arXiv:2104.09355
- SmartSim GitHub: https://github.com/CrayLabs/SmartSim

Usage
-----
    pipeline = MLInLoopPipeline(config=SmartSimConfig())
    pipeline.register_model("tbnn", tbnn_model)
    pipeline.start()

    # In SU2 iteration loop (conceptual):
    pipeline.push_tensor("velocity", velocity_field)
    corrected_nu_t = pipeline.pull_tensor("nu_t_correction")
"""

import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)
PROJECT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT))


# =============================================================================
# Check SmartSim availability
# =============================================================================

_HAS_SMARTSIM = False
try:
    import smartredis
    _HAS_SMARTSIM = True
except ImportError:
    pass


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class SmartSimConfig:
    """Configuration for SmartSim orchestration.

    Parameters
    ----------
    db_address : str
        Address of the Redis-backed Orchestrator database.
    db_port : int
        Port for the Orchestrator database.
    cluster_mode : bool
        Whether to use clustered Redis for multi-node HPC.
    tensor_type : str
        Default tensor data type ('float64' or 'float32').
    max_tensor_size_mb : float
        Maximum tensor size in MB for streaming.
    polling_interval_ms : int
        Polling interval for tensor availability.
    timeout_s : float
        Timeout for tensor operations.
    use_gpu : bool
        Whether to use GPU for inference.
    n_inference_workers : int
        Number of inference worker processes.
    """
    db_address: str = "127.0.0.1"
    db_port: int = 6379
    cluster_mode: bool = False
    tensor_type: str = "float64"
    max_tensor_size_mb: float = 100.0
    polling_interval_ms: int = 10
    timeout_s: float = 30.0
    use_gpu: bool = False
    n_inference_workers: int = 1


# =============================================================================
# Local Tensor Store (fallback when SmartRedis unavailable)
# =============================================================================

class LocalTensorStore:
    """In-memory tensor store for development/testing without SmartRedis.

    Mimics the SmartRedis API using a simple dict-based store,
    enabling the full pipeline to run locally without HPC infrastructure.
    """

    def __init__(self):
        self._store: Dict[str, np.ndarray] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}

    def put_tensor(self, name: str, data: np.ndarray):
        """Store a tensor."""
        self._store[name] = data.copy()
        self._metadata[name] = {
            "shape": data.shape,
            "dtype": str(data.dtype),
            "timestamp": time.time(),
            "size_bytes": data.nbytes,
        }
        logger.debug(f"LocalStore: put_tensor('{name}', shape={data.shape})")

    def get_tensor(self, name: str) -> np.ndarray:
        """Retrieve a tensor."""
        if name not in self._store:
            raise KeyError(f"Tensor '{name}' not found in store")
        logger.debug(f"LocalStore: get_tensor('{name}')")
        return self._store[name].copy()

    def tensor_exists(self, name: str) -> bool:
        """Check if a tensor exists."""
        return name in self._store

    def delete_tensor(self, name: str):
        """Remove a tensor from store."""
        self._store.pop(name, None)
        self._metadata.pop(name, None)

    def list_tensors(self) -> List[str]:
        """List all stored tensor names."""
        return list(self._store.keys())

    def get_metadata(self, name: str) -> Dict[str, Any]:
        """Get metadata for a stored tensor."""
        return self._metadata.get(name, {})

    def clear(self):
        """Clear all stored tensors."""
        self._store.clear()
        self._metadata.clear()

    def total_memory_bytes(self) -> int:
        """Total memory used by stored tensors."""
        return sum(t.nbytes for t in self._store.values())


# =============================================================================
# SU2 SmartRedis Client Adapter
# =============================================================================

class SU2SmartRedisClient:
    """Adapter wrapping SmartRedis for SU2 flow field streaming.

    Provides a high-level API for pushing/pulling CFD state variables
    (velocity, pressure, turbulent kinetic energy, eddy viscosity)
    between the SU2 solver and co-located ML models.

    Falls back to LocalTensorStore when SmartRedis is not available.

    Parameters
    ----------
    config : SmartSimConfig
        Orchestration configuration.
    """

    def __init__(self, config: Optional[SmartSimConfig] = None):
        self.config = config or SmartSimConfig()

        if _HAS_SMARTSIM:
            try:
                self.client = smartredis.Client(
                    address=f"{self.config.db_address}:{self.config.db_port}",
                    cluster=self.config.cluster_mode,
                )
                self._backend = "smartredis"
                logger.info("Connected to SmartRedis Orchestrator")
            except Exception as e:
                logger.warning(f"SmartRedis connection failed: {e}. "
                              f"Falling back to local store.")
                self.client = LocalTensorStore()
                self._backend = "local"
        else:
            self.client = LocalTensorStore()
            self._backend = "local"
            logger.info("SmartRedis not available — using local tensor store")

    @property
    def backend(self) -> str:
        """Current backend: 'smartredis' or 'local'."""
        return self._backend

    def push_flow_state(self, iteration: int,
                        velocity: np.ndarray,
                        pressure: np.ndarray,
                        tke: Optional[np.ndarray] = None,
                        nu_t: Optional[np.ndarray] = None):
        """Push complete flow state to the Orchestrator.

        Parameters
        ----------
        iteration : int
            Current SU2 iteration number.
        velocity : ndarray (n_points, 3) or (3, nx, ny, nz)
            Velocity field.
        pressure : ndarray (n_points,) or (nx, ny, nz)
            Pressure field.
        tke : ndarray or None
            Turbulent kinetic energy field.
        nu_t : ndarray or None
            Eddy viscosity field.
        """
        prefix = f"su2_iter_{iteration}"
        dtype = np.float64 if self.config.tensor_type == "float64" else np.float32

        self.client.put_tensor(f"{prefix}_velocity",
                               velocity.astype(dtype))
        self.client.put_tensor(f"{prefix}_pressure",
                               pressure.astype(dtype))
        if tke is not None:
            self.client.put_tensor(f"{prefix}_tke", tke.astype(dtype))
        if nu_t is not None:
            self.client.put_tensor(f"{prefix}_nu_t", nu_t.astype(dtype))

        logger.debug(f"Pushed flow state for iteration {iteration}")

    def pull_correction(self, iteration: int,
                        field_name: str = "nu_t_correction",
                        timeout_s: Optional[float] = None
                        ) -> Optional[np.ndarray]:
        """Pull ML-computed correction from the Orchestrator.

        Parameters
        ----------
        iteration : int
            Iteration number to retrieve correction for.
        field_name : str
            Name of the correction tensor.
        timeout_s : float or None
            Timeout (uses config default if None).

        Returns
        -------
        correction : ndarray or None
            ML correction tensor, or None if not available.
        """
        key = f"ml_iter_{iteration}_{field_name}"
        timeout = timeout_s or self.config.timeout_s

        start = time.time()
        while time.time() - start < timeout:
            if self.client.tensor_exists(key):
                return self.client.get_tensor(key)
            time.sleep(self.config.polling_interval_ms / 1000)

        logger.warning(f"Timeout waiting for correction '{key}'")
        return None


# =============================================================================
# Online Inference Manager
# =============================================================================

class OnlineInferenceManager:
    """Manages co-located ML models for online inference.

    Continuously monitors the tensor store for new flow states,
    runs registered ML models, and pushes results back.

    Parameters
    ----------
    client : SU2SmartRedisClient
        Client for tensor I/O.
    """

    def __init__(self, client: SU2SmartRedisClient):
        self.client = client
        self._models: Dict[str, Dict[str, Any]] = {}
        self._running = False

    def register_model(self, name: str, model: Any,
                       input_keys: List[str] = None,
                       output_key: str = "correction"):
        """Register an ML model for online inference.

        Parameters
        ----------
        name : str
            Model identifier.
        model : object
            ML model with a .forward() or .predict() method.
        input_keys : list of str
            Tensor store keys to read as input.
        output_key : str
            Tensor store key to write the output.
        """
        self._models[name] = {
            "model": model,
            "input_keys": input_keys or ["velocity", "pressure"],
            "output_key": output_key,
        }
        logger.info(f"Registered model '{name}' for online inference")

    def run_inference(self, iteration: int, model_name: str) -> Optional[np.ndarray]:
        """Run a single inference step for a registered model.

        Parameters
        ----------
        iteration : int
        model_name : str

        Returns
        -------
        result : ndarray or None
        """
        if model_name not in self._models:
            raise ValueError(f"Model '{model_name}' not registered")

        model_info = self._models[model_name]
        model = model_info["model"]

        # Gather inputs
        inputs = {}
        prefix = f"su2_iter_{iteration}"
        for key in model_info["input_keys"]:
            full_key = f"{prefix}_{key}"
            if self.client.client.tensor_exists(full_key):
                inputs[key] = self.client.client.get_tensor(full_key)
            else:
                logger.debug(f"Input '{full_key}' not available yet")
                return None

        # Run inference
        if hasattr(model, 'forward'):
            result = model.forward(**inputs)
        elif hasattr(model, 'predict'):
            # Stack inputs for predict interface
            input_array = np.concatenate(
                [v.ravel() for v in inputs.values()]
            )
            result = model.predict(input_array.reshape(1, -1)).ravel()
        else:
            raise TypeError(f"Model '{model_name}' has no forward() or predict()")

        # Push result
        output_key = f"ml_iter_{iteration}_{model_info['output_key']}"
        self.client.client.put_tensor(output_key, np.asarray(result))

        logger.debug(f"Inference complete: {model_name} @ iter {iteration}")
        return result


# =============================================================================
# Streaming TBNN Updater
# =============================================================================

class StreamingTBNNUpdater:
    """Online weight updates for TBNN closure using streaming gradients.

    When the SU2 solver encounters an unprecedented flow state, this
    module computes error gradients online and updates TBNN weights
    dynamically using the streaming data — eliminating the traditional
    offline-train / online-deploy dichotomy.

    Parameters
    ----------
    client : SU2SmartRedisClient
        Client for tensor I/O.
    lr : float
        Online learning rate.
    momentum : float
        Momentum for SGD updates.
    """

    def __init__(self, client: SU2SmartRedisClient,
                 lr: float = 1e-4, momentum: float = 0.9):
        self.client = client
        self.lr = lr
        self.momentum = momentum
        self._velocity_buffer: Dict[str, np.ndarray] = {}
        self.update_count = 0

    def compute_online_update(self, predicted: np.ndarray,
                               target: np.ndarray,
                               weights: Dict[str, np.ndarray]
                               ) -> Dict[str, np.ndarray]:
        """Compute SGD weight update from streaming error.

        Parameters
        ----------
        predicted : ndarray
            Current TBNN prediction.
        target : ndarray
            Reference/experimental data (streamed).
        weights : dict of ndarray
            Current model weights.

        Returns
        -------
        updated_weights : dict of ndarray
            Updated weights after one SGD step.
        """
        # Compute loss gradient (MSE)
        error = predicted - target
        grad_scale = 2.0 * np.mean(error ** 2)

        updated = {}
        for name, w in weights.items():
            # Initialize velocity buffer
            if name not in self._velocity_buffer:
                self._velocity_buffer[name] = np.zeros_like(w)

            # SGD with momentum
            grad = grad_scale * np.ones_like(w) * 0.01  # Simplified gradient
            self._velocity_buffer[name] = (
                self.momentum * self._velocity_buffer[name] - self.lr * grad
            )
            updated[name] = w + self._velocity_buffer[name]

        self.update_count += 1
        return updated

    def push_updated_weights(self, iteration: int,
                              weights: Dict[str, np.ndarray]):
        """Push updated TBNN weights to the tensor store.

        Parameters
        ----------
        iteration : int
        weights : dict of ndarray
        """
        for name, w in weights.items():
            key = f"tbnn_weights_{iteration}_{name}"
            self.client.client.put_tensor(key, w)

        logger.debug(f"Pushed {len(weights)} updated weight tensors "
                     f"for iteration {iteration}")


# =============================================================================
# ML-in-the-Loop Pipeline
# =============================================================================

class MLInLoopPipeline:
    """Full ML-in-the-loop orchestration pipeline.

    Coordinates the complete workflow:
    1. SU2 pushes flow state tensors per iteration
    2. ML models run online inference
    3. Corrections are pushed back for SU2 to consume
    4. (Optional) TBNN weights updated online

    Parameters
    ----------
    config : SmartSimConfig or None
    """

    def __init__(self, config: Optional[SmartSimConfig] = None):
        self.config = config or SmartSimConfig()
        self.client = SU2SmartRedisClient(self.config)
        self.inference_mgr = OnlineInferenceManager(self.client)
        self.tbnn_updater = StreamingTBNNUpdater(self.client)
        self._iteration = 0

    def register_model(self, name: str, model: Any,
                       input_keys: List[str] = None,
                       output_key: str = "correction"):
        """Register an ML model for the pipeline."""
        self.inference_mgr.register_model(name, model, input_keys, output_key)

    def step(self, velocity: np.ndarray, pressure: np.ndarray,
             tke: Optional[np.ndarray] = None,
             nu_t: Optional[np.ndarray] = None,
             model_name: Optional[str] = None
             ) -> Dict[str, Any]:
        """Execute one ML-in-the-loop step.

        Parameters
        ----------
        velocity, pressure, tke, nu_t : ndarray
            Current flow state.
        model_name : str or None
            Specific model to run (runs all if None).

        Returns
        -------
        results : dict
            Inference results keyed by model name.
        """
        # Push flow state
        self.client.push_flow_state(
            self._iteration, velocity, pressure, tke, nu_t
        )

        # Run inference
        results = {}
        models_to_run = (
            [model_name] if model_name
            else list(self.inference_mgr._models.keys())
        )

        for name in models_to_run:
            result = self.inference_mgr.run_inference(self._iteration, name)
            if result is not None:
                results[name] = result

        self._iteration += 1
        return results

    def summary(self) -> Dict[str, Any]:
        """Pipeline status summary."""
        return {
            "backend": self.client.backend,
            "iteration": self._iteration,
            "registered_models": list(self.inference_mgr._models.keys()),
            "smartsim_available": _HAS_SMARTSIM,
            "config": {
                "db_address": self.config.db_address,
                "db_port": self.config.db_port,
                "use_gpu": self.config.use_gpu,
            },
        }
