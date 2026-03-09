import pytest
import numpy as np

from scripts.ml_augmentation.foundation_model_alignment import (
    generate_generic_pretraining_data,
    generate_separation_domain_data,
    FoundationModelAlignmentStudy,
    FMAlignmentResult
)


class TestDataDiversity:
    """Test that the data generators produce distinctly different datasets."""
    
    def test_pretraining_vs_domain_fidelity(self):
        """Ensure domain data contains negative Cf (separation) while pretrain data does not."""
        X_pt, Cp_pt, Cf_pt = generate_generic_pretraining_data(n_samples=50, seed=42)
        X_dom, Cp_dom, Cf_dom = generate_separation_domain_data(n_samples=50, seed=42)
        
        # Pretraining data should generally be attached (positive Cf)
        # Small noise might cause tiny negative values, so we check the min
        assert np.min(Cf_pt) > -1e-4, "Pretraining data should not have massive separation"
        
        # Domain data MUST contain significant negative Cf
        assert np.min(Cf_dom) < -5e-4, "Domain data must contain clear separation (negative Cf)"

    def test_data_shapes(self):
        """Verify the data generators return correct shapes."""
        X, Cp, Cf = generate_generic_pretraining_data(n_samples=10, spatial_res=40)
        assert X.shape == (10, 3)
        assert Cp.shape == (10, 40)
        assert Cf.shape == (10, 40)


class TestAlignmentStudy:
    """End-to-end tests for the Foundation Model Alignment execution."""

    def test_run_alignment_study(self):
        """Test the study runner completes all three phases."""
        study = FoundationModelAlignmentStudy(seed=0, spatial_res=20)
        # Very small number of samples for fast testing
        results = study.run(
            n_pretrain_samples=20, 
            n_finetune_samples=5, 
            n_custom_samples=10, 
            n_test_samples=5
        )
        
        assert len(results) == 3
        types = [r.model_type for r in results]
        assert "Zero-Shot FM" in types
        assert "Fine-Tuned FM" in types
        assert "Custom Domain Model" in types
        
        # Check that metrics are populated
        for r in results:
            assert isinstance(r, FMAlignmentResult)
            assert r.Cf_RMSE > 0
            assert r.Cp_RMSE > 0
            assert r.train_time_s > 0

    def test_generate_report(self):
        """Test that the report generator produces the expected markdown."""
        study = FoundationModelAlignmentStudy(seed=0, spatial_res=20)
        report = study.generate_report()
        assert "No results generated yet" in report
        
        # Run minimal study
        study.run(n_pretrain_samples=2, n_finetune_samples=2, n_custom_samples=2, n_test_samples=2)
        report = study.generate_report()
        
        assert "CFD Foundation Model Alignment Study" in report
        assert "Zero-Shot FM" in report
        assert "Fine-Tuned FM" in report
        assert "Custom Domain Model" in report
