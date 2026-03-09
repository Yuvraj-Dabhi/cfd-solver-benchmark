"""
Turbulence Models Module
========================
Contains implementations and reference data for turbulence models
used in the CFD Solver Benchmark for Flow Separation Prediction.

Models:
  - spalart_allmaras: Complete SA model from NASA TMR
"""

from .spalart_allmaras import (
    SA_CONSTANTS,
    SA_NOFT2_CONSTANTS,
    SAConstants,
    SABoundaryConditions,
    compute_chi,
    compute_fv1,
    compute_fv2,
    compute_ft2,
    compute_S_tilde,
    compute_r,
    compute_g,
    compute_fw,
    compute_nu_t,
    production_term,
    destruction_term,
    sutherland_viscosity,
    verify_constants,
)
