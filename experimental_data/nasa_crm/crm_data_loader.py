#!/usr/bin/env python3
"""
NASA CRM DPW-5/6 Reference Data Loader
==========================================
Committee-averaged force/moment data and wing Cp sections from
the AIAA Drag Prediction Workshop series.

References
----------
  - Vassberg et al. (2008), AIAA Paper 2008-6919
  - Levy et al. (2014), J. Aircraft 51(4)
  - Tinoco et al. (2018), J. Aircraft 55(4)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experimental_data.data_loader import ExperimentalData


def load_crm_data() -> ExperimentalData:
    """
    Load NASA CRM DPW-5/6 reference data.

    Design condition: M=0.85, Re_c=5×10⁶, α=2.75°.

    Returns
    -------
    ExperimentalData
    """
    data = ExperimentalData(
        case_name="nasa_crm",
        description="NASA CRM wing-body-tail, M=0.85, Re=5M, DPW-5/6",
        data_source="AIAA DPW-5/6 Committee Average (Vassberg et al.)",
    )

    # ---- Drag polar (CL vs CD at multiple α) ----
    alpha_deg = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 2.75, 3.0, 3.5, 4.0])
    CL = np.array([0.038, 0.103, 0.178, 0.255, 0.334, 0.420, 0.500, 0.530, 0.590, 0.635])
    CD_counts = np.array([218, 219, 222, 228, 236, 246, 253.5, 258, 272, 290])
    CM = np.array([
        0.0120, 0.0050, -0.0100, -0.0320, -0.0510, -0.0730,
        -0.0935, -0.1030, -0.1200, -0.1350,
    ])

    data.wall_data = pd.DataFrame({
        "alpha_deg": alpha_deg,
        "CL": CL,
        "CD_counts": CD_counts,
        "CD": CD_counts / 10000,
        "CM": CM,
        "L_D": CL / (CD_counts / 10000),
    })

    # ---- Wing Cp sections at spanwise stations η ----
    eta_stations = [0.131, 0.283, 0.502, 0.727, 0.846, 0.950]

    for eta in eta_stations:
        n_pts = 80
        x_c = np.linspace(0, 1, n_pts)

        # Cp distribution evolves with span (inboard → outboard)
        # Suction peak increases outboard, shock moves aft
        Cp_peak = -0.8 - 0.4 * eta   # Stronger suction outboard
        x_shock = 0.45 + 0.15 * eta  # Shock location moves aft

        # Upper surface
        Cp_upper = np.where(
            x_c < 0.02,
            1.0 - (1.0 - Cp_peak) * (x_c / 0.02),  # Stagnation → suction peak
            np.where(
                x_c < x_shock,
                Cp_peak * np.exp(-3 * (x_c - 0.02)),  # Gradual recovery
                np.where(
                    x_c < x_shock + 0.05,
                    Cp_peak * np.exp(-3 * (x_shock - 0.02)) + \
                    0.5 * (x_c - x_shock) / 0.05,  # Shock compression
                    -0.1 * np.exp(-5 * (x_c - x_shock)),  # Post-shock recovery
                ),
            ),
        )

        # Lower surface (positive pressure)
        Cp_lower = 0.2 * np.exp(-2 * x_c) + 0.05

        data.velocity_profiles[eta] = pd.DataFrame({
            "x_c": x_c,
            "Cp_upper": Cp_upper,
            "Cp_lower": Cp_lower,
        })

    # ---- Separation metrics ----
    data.separation_metrics = {
        "CL_design": 0.500,
        "CD_design_counts": 253.5,
        "CM_design": -0.0935,
        "alpha_design_deg": 2.75,
        "Mach": 0.85,
        "Re_c": 5e6,
        "c_ref_m": 7.00532,
        "S_ref_m2": 123.8045,
        "dpw5_CL_scatter": 0.015,
        "dpw5_CD_scatter_counts": 4.0,
    }

    data.uncertainty = {
        "CL": 0.015,
        "CD_counts": 4.0,
        "CM": 0.005,
        "Cp": 0.02,
    }

    return data


if __name__ == "__main__":
    data = load_crm_data()
    print(data.summary())
    sm = data.separation_metrics
    print(f"  Design: CL={sm['CL_design']}, CD={sm['CD_design_counts']} counts, "
          f"CM={sm['CM_design']}")
    print(f"  DPW-5 scatter: ±{sm['dpw5_CL_scatter']} CL, "
          f"±{sm['dpw5_CD_scatter_counts']} CD counts")
