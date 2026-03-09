#!/usr/bin/env python3
"""
Tests for GNN Mesh Adaptation Module
========================================
Validates Hessian metric prediction, mesh quality assessment,
adaptivity criteria, and the full adaptation pipeline.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ml_augmentation.gnn_mesh_adaptation import (
    MeshAdaptationConfig,
    MeshAdaptationPipeline,
    HessianMetricPredictor,
    AdaptivityCriterion,
    MeshQualityAssessor,
)


# =========================================================================
# TestMeshQualityAssessor
# =========================================================================
class TestMeshQualityAssessor:
    """Tests for mesh quality metric computation."""

    def _make_equilateral_triangle(self):
        vertices = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, np.sqrt(3) / 2],
        ])
        elements = np.array([[0, 1, 2]])
        return vertices, elements

    def test_equilateral_triangle_quality(self):
        vertices, elements = self._make_equilateral_triangle()
        assessor = MeshQualityAssessor()
        metrics = assessor.compute_element_quality(vertices, elements)
        # Equilateral: aspect ratio ≈ 1, skewness ≈ 0
        assert metrics["aspect_ratio"][0] == pytest.approx(1.0, abs=0.1)
        assert metrics["skewness"][0] == pytest.approx(0.0, abs=0.01)
        assert metrics["area"][0] > 0

    def test_degenerate_triangle(self):
        """Very thin triangle should have high aspect ratio."""
        vertices = np.array([
            [0.0, 0.0],
            [10.0, 0.0],
            [5.0, 0.01],
        ])
        elements = np.array([[0, 1, 2]])
        assessor = MeshQualityAssessor()
        metrics = assessor.compute_element_quality(vertices, elements)
        assert metrics["aspect_ratio"][0] > 1.5
        assert metrics["skewness"][0] > 0.1

    def test_multiple_elements(self):
        """Should handle multiple elements."""
        pipeline = MeshAdaptationPipeline()
        vertices, elements = pipeline.generate_synthetic_mesh(nx=5, ny=5)
        assessor = MeshQualityAssessor()
        metrics = assessor.compute_element_quality(vertices, elements)
        assert len(metrics["aspect_ratio"]) == len(elements)
        assert np.all(metrics["area"] > 0)

    def test_overall_quality_score(self):
        pipeline = MeshAdaptationPipeline()
        vertices, elements = pipeline.generate_synthetic_mesh(nx=10, ny=5)
        assessor = MeshQualityAssessor()
        assessor.compute_element_quality(vertices, elements)
        score = assessor.overall_quality_score()
        assert 0.0 <= score <= 1.0

    def test_summary(self):
        pipeline = MeshAdaptationPipeline()
        vertices, elements = pipeline.generate_synthetic_mesh(nx=5, ny=5)
        assessor = MeshQualityAssessor()
        assessor.compute_element_quality(vertices, elements)
        summary = assessor.summary()
        assert "mean_aspect_ratio" in summary
        assert "quality_score" in summary
        assert summary["n_elements"] == len(elements)


# =========================================================================
# TestAdaptivityCriterion
# =========================================================================
class TestAdaptivityCriterion:
    """Tests for adaptivity error indicators."""

    def test_gradient_indicator_shape(self):
        pipeline = MeshAdaptationPipeline()
        vertices, elements = pipeline.generate_synthetic_mesh(nx=10, ny=5)
        solution = np.sin(2 * np.pi * vertices[:, 0])
        criterion = AdaptivityCriterion(method="gradient")
        indicator = criterion.compute_indicator(vertices, solution, elements)
        assert indicator.shape == (len(elements),)

    def test_indicator_normalized(self):
        """Indicator should be in [0, 1]."""
        pipeline = MeshAdaptationPipeline()
        vertices, elements = pipeline.generate_synthetic_mesh(nx=8, ny=4)
        solution = np.exp(-vertices[:, 0])  # Varying gradient
        criterion = AdaptivityCriterion(method="gradient")
        indicator = criterion.compute_indicator(vertices, solution, elements)
        assert np.all(indicator >= 0)
        assert np.all(indicator <= 1.0 + 1e-10)

    def test_refinement_flags(self):
        criterion = AdaptivityCriterion(threshold=0.5)
        indicator = np.array([0.1, 0.3, 0.6, 0.8, 0.01])
        flags = criterion.mark_for_refinement(indicator)
        assert flags[2] == 1   # 0.6 > 0.5, refine
        assert flags[3] == 1   # 0.8 > 0.5, refine
        assert flags[4] == -1  # 0.01 < 0.05, coarsen

    def test_hessian_method(self):
        pipeline = MeshAdaptationPipeline()
        vertices, elements = pipeline.generate_synthetic_mesh(nx=6, ny=4)
        solution = vertices[:, 0] ** 2
        criterion = AdaptivityCriterion(method="hessian")
        indicator = criterion.compute_indicator(vertices, solution, elements)
        assert len(indicator) == len(elements)

    def test_constant_solution_low_indicator(self):
        """Constant solution should have near-zero gradient indicator."""
        pipeline = MeshAdaptationPipeline()
        vertices, elements = pipeline.generate_synthetic_mesh(nx=6, ny=4)
        solution = np.ones(len(vertices))
        criterion = AdaptivityCriterion(method="gradient")
        indicator = criterion.compute_indicator(vertices, solution, elements)
        # All zeros for constant field
        assert np.all(indicator <= 1e-10)


# =========================================================================
# TestHessianMetricPredictor
# =========================================================================
class TestHessianMetricPredictor:
    """Tests for GNN Hessian metric predictor."""

    def _make_simple_graph(self):
        n_nodes = 20
        config = MeshAdaptationConfig(node_feature_dim=6, edge_feature_dim=4)
        node_features = np.random.randn(n_nodes, 6)
        # Simple chain graph
        src = list(range(n_nodes - 1))
        tgt = list(range(1, n_nodes))
        edge_index = np.array([src + tgt, tgt + src])
        n_edges = edge_index.shape[1]
        edge_features = np.random.randn(n_edges, 4)
        return config, node_features, edge_index, edge_features

    def test_output_shape(self):
        config, nf, ei, ef = self._make_simple_graph()
        predictor = HessianMetricPredictor(config)
        metrics = predictor.forward(nf, ei, ef)
        assert metrics.shape == (len(nf), config.metric_output_dim)

    def test_metric_to_tensor_shape(self):
        config, nf, ei, ef = self._make_simple_graph()
        predictor = HessianMetricPredictor(config)
        comp = predictor.forward(nf, ei, ef)
        tensors = predictor.metric_to_tensor(comp)
        assert tensors.shape == (len(nf), 2, 2)

    def test_metric_positive_definite(self):
        """Predicted metric tensors should be positive definite."""
        config, nf, ei, ef = self._make_simple_graph()
        predictor = HessianMetricPredictor(config)
        comp = predictor.forward(nf, ei, ef)
        tensors = predictor.metric_to_tensor(comp)
        for i in range(len(tensors)):
            eigvals = np.linalg.eigvalsh(tensors[i])
            assert np.all(eigvals > 0), f"Node {i}: eigvals={eigvals}"

    def test_metric_symmetric(self):
        """Metric tensors should be symmetric."""
        config, nf, ei, ef = self._make_simple_graph()
        predictor = HessianMetricPredictor(config)
        comp = predictor.forward(nf, ei, ef)
        tensors = predictor.metric_to_tensor(comp)
        for i in range(len(tensors)):
            np.testing.assert_allclose(tensors[i], tensors[i].T, atol=1e-10)

    def test_param_count(self):
        config = MeshAdaptationConfig()
        predictor = HessianMetricPredictor(config)
        assert predictor.count_params() > 0


# =========================================================================
# TestMeshAdaptationPipeline
# =========================================================================
class TestMeshAdaptationPipeline:
    """Tests for the end-to-end adaptation pipeline."""

    def test_synthetic_mesh_generation(self):
        pipeline = MeshAdaptationPipeline()
        vertices, elements = pipeline.generate_synthetic_mesh(nx=10, ny=5)
        assert vertices.shape[1] == 2
        assert elements.shape[1] == 3
        assert len(vertices) == 50
        assert len(elements) == (9 * 4 * 2)  # (nx-1) * (ny-1) * 2

    def test_feature_extraction(self):
        pipeline = MeshAdaptationPipeline()
        vertices, elements = pipeline.generate_synthetic_mesh(nx=8, ny=4)
        feats = pipeline.extract_features(vertices, elements)
        assert feats["node_features"].shape[0] == len(vertices)
        assert feats["edge_index"].shape[0] == 2
        assert feats["edge_features"].shape[1] == 4

    def test_feature_extraction_with_solution(self):
        pipeline = MeshAdaptationPipeline()
        vertices, elements = pipeline.generate_synthetic_mesh(nx=6, ny=4)
        n = len(vertices)
        solution = {
            "Cp": np.sin(vertices[:, 0]),
            "Cf": np.ones(n) * 0.003,
            "Mach": np.ones(n) * 0.8,
        }
        feats = pipeline.extract_features(vertices, elements, solution)
        # Cp should be in feature column 3
        assert np.allclose(feats["node_features"][:, 3], solution["Cp"])

    def test_full_pipeline(self):
        pipeline = MeshAdaptationPipeline()
        vertices, elements = pipeline.generate_synthetic_mesh(nx=8, ny=4)
        solution = {"Cp": np.sin(2 * np.pi * vertices[:, 0])}
        results = pipeline.identify_refinement_regions(
            vertices, elements, solution)
        assert "metric_components" in results
        assert "refinement_flags" in results
        assert "mesh_quality" in results
        assert results["n_refine"] >= 0
        assert results["n_coarsen"] >= 0

    def test_report_generation(self):
        pipeline = MeshAdaptationPipeline()
        vertices, elements = pipeline.generate_synthetic_mesh(nx=6, ny=4)
        results = pipeline.identify_refinement_regions(vertices, elements)
        report = pipeline.report(results)
        assert "Mesh Adaptation Report" in report
        assert "refine" in report.lower()


# =========================================================================
# New Tests — Adaptnet Extensions (Extension III)
# =========================================================================

class TestAdaptnetExtensions:
    """Tests for MeshnetCAD, GraphnetHessian, and AdaptnetPipeline."""

    def test_meshnet_cad(self):
        from scripts.ml_augmentation.gnn_mesh_adaptation import MeshnetCAD
        meshnet = MeshnetCAD(hidden_dim=32)
        features = np.random.randn(10, 5)
        density = meshnet.predict_density(features)
        assert density.shape == (10,)
        assert np.all(density > 0)

    def test_graphnet_hessian(self):
        from scripts.ml_augmentation.gnn_mesh_adaptation import GraphnetHessian, MeshAdaptationConfig
        config = MeshAdaptationConfig(node_feature_dim=6, edge_feature_dim=4, hidden_dim=16)
        model = GraphnetHessian(config)
        
        nf = np.random.randn(10, 6)
        ef = np.random.randn(15, 4)
        ei = np.vstack([np.random.randint(0, 10, 15), np.random.randint(0, 10, 15)])
        
        metric = model.forward(nf, ei, ef)
        assert metric.shape == (10, 3)

    def test_adaptnet_pipeline(self):
        from scripts.ml_augmentation.gnn_mesh_adaptation import AdaptnetPipeline, MeshAdaptationConfig
        config = MeshAdaptationConfig(hidden_dim=16)
        pipeline = AdaptnetPipeline(config)
        
        cad = np.random.randn(20, 3)
        density = pipeline.generate_initial_mesh_density(cad)
        assert density.shape == (20,)
        assert hasattr(pipeline, "graphnet")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
