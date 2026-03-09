import pytest
import numpy as np
from pathlib import Path

from run_nasa_crm import (
    generate_synthetic_crm_results,
    CRMPostprocessor,
    CRMAlphaSweep,
    CRMMLIntegration,
    CRMGCIStudy,
    WingSeparationAnalyzer
)
from scripts.analysis.uq_summary_report import (
    MODEL_UNCERTAINTY,
    ML_STOCHASTIC_UNCERTAINTY,
    DOMINO_OOD_UNCERTAINTY
)

def test_synthetic_generation(tmp_path):
    results = generate_synthetic_crm_results(tmp_path)
    assert Path(results["history_path"]).exists()
    assert Path(results["surface_path"]).exists()
    assert "surface_data" in results
    assert "forces_by_alpha" in results

def test_crm_ml_integration_extracts_sections():
    integration = CRMMLIntegration()
    # Create simple dummy surface data
    b_semi = 29.3860
    n_pts = 1000
    y = np.linspace(0.1*b_semi, 0.9*b_semi, n_pts)
    x = np.random.uniform(0, 10, n_pts)
    
    surface_data = {
        "x": x,
        "y": y,
        "z": np.zeros_like(x),
        "Cp": np.random.uniform(-1, 1, n_pts),
        "Cf": np.random.uniform(0, 0.005, n_pts)
    }
    
    sections = integration.extract_wing_sections(surface_data, eta_stations=[0.5])
    assert "eta_0.50" in sections
    assert sections["eta_0.50"]["n_points"] > 0

def test_crm_ml_integration_evaluates_transfer():
    integration = CRMMLIntegration()
    # Create better dummy surface data that will cluster around eta=0.15 (one of default stations)
    b_semi = 29.3860
    n_pts = 100
    y = np.full(n_pts, 0.15 * b_semi)
    x = np.linspace(0, 7, n_pts)
    
    surface_data = {
        "x": x,
        "y": y,
        "z": np.zeros_like(x),
        "Cp": np.random.uniform(-1, 1, n_pts),
        "Cf": np.random.uniform(0, 0.005, n_pts)
    }
    
    results = integration.evaluate_transfer(surface_data)
    assert "eta_0.15" in results
    assert "beta_rmse" in results["eta_0.15"]
    assert "summary" in results

def test_nasa_crm_in_uq_report():
    assert "nasa_crm" in MODEL_UNCERTAINTY
    assert "nasa_crm" in ML_STOCHASTIC_UNCERTAINTY
    assert "nasa_crm" in DOMINO_OOD_UNCERTAINTY
