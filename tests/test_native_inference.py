"""
Tests for Native Inference Bridge & High-Performance Coupling
==============================================================
Tests TorchScript export, zero-copy buffers, C++ generation,
coupling benchmark, and inference equivalence.

Run: pytest tests/test_native_inference.py -v
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Check torch availability
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _make_simple_model(n_features=5, hidden=32):
    """Build a simple NN for testing."""
    return nn.Sequential(
        nn.Linear(n_features, hidden),
        nn.Tanh(),
        nn.Linear(hidden, hidden),
        nn.Tanh(),
        nn.Linear(hidden, 1),
        nn.Softplus(),
    )


# =========================================================================
# TorchScript Export Tests
# =========================================================================
@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
class TestTorchScriptExporter:
    """Test TorchScript JIT tracing and validation."""

    def test_export_module(self, tmp_path):
        from scripts.ml_augmentation.native_inference import TorchScriptExporter

        model = _make_simple_model()
        model.eval()
        exporter = TorchScriptExporter(validate=True, atol=1e-5)

        sample = torch.randn(100, 5)
        result = exporter.export_module(
            model, sample, tmp_path / "test_model.pt", "TestModel"
        )

        assert result.validation_passed
        assert result.max_abs_error < 1e-5
        assert result.n_parameters > 0
        assert result.file_size_bytes > 0
        assert (tmp_path / "test_model.pt").exists()

    def test_export_correction_nn(self, tmp_path):
        from scripts.ml_augmentation.native_inference import TorchScriptExporter

        model = _make_simple_model()
        exporter = TorchScriptExporter()
        result = exporter.export_correction_nn(
            model, n_features=5, output_path=tmp_path / "beta.pt"
        )
        assert result.validation_passed

    def test_export_ensemble(self, tmp_path):
        from scripts.ml_augmentation.native_inference import TorchScriptExporter

        models = [_make_simple_model() for _ in range(3)]
        exporter = TorchScriptExporter()
        results = exporter.export_ensemble(
            models, output_dir=tmp_path / "ensemble"
        )
        assert len(results) == 3
        assert all(r.validation_passed for r in results)

    def test_round_trip_equivalence(self, tmp_path):
        """Python output == TorchScript loaded output."""
        from scripts.ml_augmentation.native_inference import TorchScriptExporter

        model = _make_simple_model()
        model.eval()
        exporter = TorchScriptExporter()

        path = tmp_path / "roundtrip.pt"
        exporter.export_correction_nn(model, output_path=path)

        # Load and compare
        loaded = torch.jit.load(str(path))
        x = torch.randn(50, 5)

        with torch.no_grad():
            py_out = model(x)
            ts_out = loaded(x)

        np.testing.assert_allclose(
            py_out.numpy(), ts_out.numpy(), atol=1e-5
        )


# =========================================================================
# Zero-Copy Buffer Tests
# =========================================================================
@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
class TestSharedMemoryBuffer:
    """Test pre-allocated zero-copy buffers."""

    def test_creation(self):
        from scripts.ml_augmentation.native_inference import SharedMemoryBuffer
        buf = SharedMemoryBuffer(10000, 5)
        assert buf.allocated_bytes > 0

    def test_load_features(self):
        from scripts.ml_augmentation.native_inference import SharedMemoryBuffer
        buf = SharedMemoryBuffer(1000, 5)
        features = np.random.randn(500, 5).astype(np.float32)
        tensor = buf.load_features(features)
        assert tensor.shape == (500, 5)
        np.testing.assert_allclose(
            tensor.numpy(), features, atol=1e-7
        )

    def test_overflow_raises(self):
        from scripts.ml_augmentation.native_inference import SharedMemoryBuffer
        buf = SharedMemoryBuffer(100, 5)
        with pytest.raises(ValueError, match="exceeds buffer capacity"):
            buf.load_features(np.zeros((200, 5), dtype=np.float32))

    def test_output_buffer(self):
        from scripts.ml_augmentation.native_inference import SharedMemoryBuffer
        buf = SharedMemoryBuffer(1000, 5)
        out = buf.get_output_buffer(500)
        assert out.shape == (500,)

    def test_zero_copy_shares_memory(self):
        """Verify that torch.from_numpy shares memory (no copy)."""
        from scripts.ml_augmentation.native_inference import SharedMemoryBuffer
        buf = SharedMemoryBuffer(100, 5)
        features = np.ones((50, 5), dtype=np.float32) * 42.0
        tensor = buf.load_features(features)
        # Modifying the numpy buffer should reflect in tensor
        buf._input_buffer[0, 0] = 999.0
        assert tensor[0, 0].item() == 999.0


# =========================================================================
# Native Inference Bridge Tests
# =========================================================================
@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
class TestNativeInferenceBridge:
    """Test the high-performance inference pipeline."""

    def test_from_model(self):
        from scripts.ml_augmentation.native_inference import NativeInferenceBridge
        model = _make_simple_model()
        bridge = NativeInferenceBridge(model=model, max_points=1000)
        features = np.random.randn(100, 5).astype(np.float32)
        result = bridge.batch_predict(features)
        assert result.shape == (100,)
        assert np.all(result > 0)  # Softplus output

    def test_from_torchscript(self, tmp_path):
        from scripts.ml_augmentation.native_inference import (
            TorchScriptExporter, NativeInferenceBridge,
        )
        model = _make_simple_model()
        exporter = TorchScriptExporter()
        path = tmp_path / "bridge_test.pt"
        exporter.export_correction_nn(model, output_path=path)

        bridge = NativeInferenceBridge(model_path=path, max_points=1000)
        features = np.random.randn(100, 5).astype(np.float32)
        result = bridge.batch_predict(features)
        assert result.shape == (100,)

    def test_warmup(self):
        from scripts.ml_augmentation.native_inference import NativeInferenceBridge
        model = _make_simple_model()
        bridge = NativeInferenceBridge(model=model, max_points=100)
        warmup_time = bridge.warmup(n_warmup=2)
        assert warmup_time > 0
        assert bridge._warmed_up

    def test_benchmark(self):
        from scripts.ml_augmentation.native_inference import NativeInferenceBridge
        model = _make_simple_model()
        bridge = NativeInferenceBridge(model=model, max_points=1000)
        metrics = bridge.benchmark_single_call(n_points=500, n_repeats=5)
        assert metrics.mean_latency_us > 0
        assert metrics.throughput_pts_per_s > 0
        assert metrics.n_points == 500


# =========================================================================
# C++ Generator Tests
# =========================================================================
class TestCppInferenceGenerator:
    """Test C++ source code generation."""

    def test_generate_files(self, tmp_path):
        from scripts.ml_augmentation.native_inference import CppInferenceGenerator
        gen = CppInferenceGenerator()

        # Create a dummy .pt file
        dummy_model = tmp_path / "dummy.pt"
        dummy_model.write_text("placeholder")

        files = gen.generate(
            dummy_model, tmp_path / "cpp_out",
            class_name="TestInference", n_features=5,
        )

        assert "header" in files
        assert "implementation" in files
        assert "cmake" in files
        assert files["header"].exists()
        assert files["implementation"].exists()
        assert files["cmake"].exists()

    def test_header_content(self, tmp_path):
        from scripts.ml_augmentation.native_inference import CppInferenceGenerator
        gen = CppInferenceGenerator()
        dummy = tmp_path / "dummy.pt"
        dummy.write_text("x")
        files = gen.generate(dummy, tmp_path / "cpp", class_name="MyNN")

        header = files["header"].read_text()
        assert "#pragma once" in header
        assert "#include <torch/script.h>" in header
        assert "class MyNN" in header
        assert "void predict" in header
        assert "void warmup" in header

    def test_implementation_content(self, tmp_path):
        from scripts.ml_augmentation.native_inference import CppInferenceGenerator
        gen = CppInferenceGenerator()
        dummy = tmp_path / "dummy.pt"
        dummy.write_text("x")
        files = gen.generate(dummy, tmp_path / "cpp", class_name="Beta")

        impl = files["implementation"].read_text()
        assert "torch::from_blob" in impl  # Zero-copy
        assert "torch::NoGradGuard" in impl  # No autograd
        assert "torch::InferenceMode" in impl  # inference mode
        assert "std::memcpy" in impl  # Copy output back

    def test_cmake_content(self, tmp_path):
        from scripts.ml_augmentation.native_inference import CppInferenceGenerator
        gen = CppInferenceGenerator()
        dummy = tmp_path / "dummy.pt"
        dummy.write_text("x")
        files = gen.generate(dummy, tmp_path / "cpp")

        cmake = files["cmake"].read_text()
        assert "find_package(Torch REQUIRED)" in cmake
        assert "TORCH_LIBRARIES" in cmake


# =========================================================================
# Coupling Benchmark Tests
# =========================================================================
class TestCouplingBenchmark:
    """Test the coupling benchmark runner."""

    def test_benchmark_runs(self):
        from scripts.utils.coupling_benchmark import run_coupling_benchmark
        results = run_coupling_benchmark(n_points=100, n_repeats=3)
        assert len(results) > 0
        # At minimum, NumPy is always available
        numpy_results = [r for r in results if "NumPy" in r.method]
        assert len(numpy_results) >= 1

    def test_format_table(self):
        from scripts.utils.coupling_benchmark import (
            run_coupling_benchmark, format_benchmark_table,
        )
        results = run_coupling_benchmark(n_points=100, n_repeats=3)
        table = format_benchmark_table(results)
        assert "Latency" in table
        assert "Method" in table

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_torch_methods_present(self):
        from scripts.utils.coupling_benchmark import run_coupling_benchmark
        results = run_coupling_benchmark(n_points=100, n_repeats=3)
        methods = [r.method for r in results]
        assert any("TorchScript" in m for m in methods)
        assert any("zero-copy" in m for m in methods)


# =========================================================================
# Integration Tests
# =========================================================================
@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
class TestInferenceComparison:
    """Test the inference method comparison utility."""

    def test_compare_methods(self):
        from scripts.ml_augmentation.native_inference import compare_inference_methods
        model = _make_simple_model()
        results = compare_inference_methods(
            model, n_points=500, n_repeats=3
        )
        assert len(results) >= 3
        for r in results:
            assert r.mean_latency_us > 0
            assert r.throughput_pts_per_s > 0

    def test_convenience_export(self, tmp_path):
        from scripts.ml_augmentation.native_inference import (
            export_model_for_native_inference,
        )
        model = _make_simple_model()
        result = export_model_for_native_inference(
            model,
            output_path=tmp_path / "export.pt",
            generate_cpp=True,
            cpp_output_dir=tmp_path / "cpp",
        )
        assert result["export_result"].validation_passed
        assert "cpp_files" in result
        assert result["bridge"] is not None

    def test_dataclass_imports(self):
        from scripts.ml_augmentation.native_inference import (
            ExportResult, InferenceMetrics, CouplingBenchmarkResult,
        )
        assert ExportResult is not None
        assert InferenceMetrics is not None
        assert CouplingBenchmarkResult is not None
