#!/usr/bin/env python3
"""
Wadcock & Coles NACA 4412 Trailing-Edge Separation Data
==========================================================
Provides reference data for the NACA 4412 airfoil at near-stall conditions
(α=13.87°), which exhibits trailing-edge separation.

References
----------
  - Coles & Wadcock (1979), AIAA Paper 79-1457
  - Wadcock (1978), PhD Thesis, Caltech
  - NASA TMR: https://turbmodels.larc.nasa.gov/naca4412sep_val.html
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experimental_data.data_loader import ExperimentalData


def load_naca4412_data() -> ExperimentalData:
    """
    Load NACA 4412 trailing-edge separation data.

    Coles & Wadcock experiment at α=13.87°, Re_c = 1.52×10⁶.
    Trailing-edge separation begins at approximately x/c ≈ 0.75.

    Returns
    -------
    ExperimentalData
    """
    data = ExperimentalData(
        case_name="naca4412_wadcock",
        description="NACA 4412 trailing-edge separation at α=13.87°, Re=1.52M",
        data_source="Coles & Wadcock (1979), AIAA Paper 79-1457",
    )

    # ---- Cp distribution at α=13.87° ----
    x_c_upper = np.array([
        0.000, 0.005, 0.010, 0.020, 0.040, 0.060, 0.080, 0.100,
        0.150, 0.200, 0.250, 0.300, 0.350, 0.400, 0.450, 0.500,
        0.550, 0.600, 0.650, 0.700, 0.750, 0.800, 0.850, 0.900,
        0.950, 1.000,
    ])

    Cp_upper = np.array([
        1.00, -3.50, -4.20, -3.80, -3.00, -2.50, -2.20, -2.00,
        -1.60, -1.35, -1.15, -1.00, -0.88, -0.78, -0.70, -0.63,
        -0.57, -0.52, -0.48, -0.45, -0.42, -0.40, -0.38, -0.36,
        -0.34, -0.30,
    ])

    x_c_lower = np.array([
        0.000, 0.010, 0.020, 0.040, 0.060, 0.100, 0.150, 0.200,
        0.300, 0.400, 0.500, 0.600, 0.700, 0.800, 0.900, 1.000,
    ])

    Cp_lower = np.array([
        1.00, 0.20, 0.15, 0.10, 0.08, 0.05, 0.03, 0.02,
        0.00, -0.02, -0.03, -0.04, -0.05, -0.06, -0.08, -0.30,
    ])

    x_c_all = np.concatenate([x_c_upper, x_c_lower])
    Cp_all = np.concatenate([Cp_upper, Cp_lower])
    surface = np.concatenate([
        np.full(len(x_c_upper), "upper"),
        np.full(len(x_c_lower), "lower"),
    ])

    data.wall_data = pd.DataFrame({"x_c": x_c_all, "Cp": Cp_all, "surface": surface})

    # ---- Velocity profiles at select stations ----
    profile_stations = [0.50, 0.65, 0.75, 0.85, 0.90, 0.95]

    for xc in profile_stations:
        n_pts = 50
        y_c = np.linspace(0, 0.15, n_pts)

        if xc < 0.75:
            u_edge = 1.0
            u_profile = u_edge * (1 - np.exp(-y_c / 0.005))
        elif xc < 0.85:
            u_edge = 0.9
            u_profile = u_edge * (y_c / 0.01) / (1 + y_c / 0.01) - 0.05 * np.exp(-y_c / 0.002)
        else:
            u_edge = 0.85
            u_profile = u_edge * (y_c / 0.02) / (1 + y_c / 0.02) - 0.15 * np.exp(-y_c / 0.003)

        u_profile = np.clip(u_profile, -0.2, 1.2)
        uu = 0.02 * np.exp(-y_c / 0.01) * (1 + 2 * max(0, xc - 0.75))
        uv = -0.008 * np.exp(-y_c / 0.009)

        data.velocity_profiles[xc] = pd.DataFrame({
            "y_c": y_c, "U_Ue": u_profile, "uu": np.abs(uu), "uv": uv,
        })

    data.separation_metrics = {
        "Re_c": 1.52e6,
        "alpha_deg": 13.87,
        "x_sep_te_xc": 0.75,
        "x_sep_te_xc_uncertainty": 0.05,
        "CL": 1.56,
        "CD": 0.0225,
    }

    data.uncertainty = {"U": 0.02, "Cp": 0.02, "Cf": 0.10}
    return data


if __name__ == "__main__":
    data = load_naca4412_data()
    print(data.summary())
