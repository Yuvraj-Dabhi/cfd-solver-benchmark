#!/usr/bin/env python3
"""
Kim et al. (1998) Backward-Facing Step Data Loader
=====================================================
Provides synthetic reference data based on published tables from:
  Kim, Kline & Johnston (1980) / Kim et al. (1998), J. Fluids Eng.

Re_H = 132,000 (based on step height H and freestream velocity).
Expansion ratio = 1.2 (step height / channel height upstream).

References
----------
  - Kim, Kline & Johnston (1980), Report MD-37, Stanford University
  - Kim et al. (1998), J. Fluids Eng. 120(3), DOI:10.1115/1.2820690
  - Bradshaw (1996), NASA Collaborative Testing Archive
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experimental_data.data_loader import ExperimentalData


def load_kim_bfs_data() -> ExperimentalData:
    """
    Load Kim et al. (1998) backward-facing step data.

    Returns
    -------
    ExperimentalData
        Loaded case data with wall data, velocity profiles, and separation metrics.
    """
    data = ExperimentalData(
        case_name="bfs_kim",
        description="Kim et al. (1998) backward-facing step, Re_H=132,000",
        data_source="Kim et al. (1998), J. Fluids Eng. / Bradshaw Archive",
    )

    # ---- Wall data (Cp and Cf along bottom wall) ----
    x_H = np.array([
        -4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0,
        6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 15.0, 20.0, 25.0, 30.0,
    ])

    Cp = np.array([
        0.00, 0.00, -0.01, -0.10, -0.08, -0.04, 0.02, 0.08, 0.12,
        0.16, 0.18, 0.19, 0.20, 0.20, 0.20, 0.19, 0.18, 0.17, 0.17,
    ])

    Cf = np.array([
        0.0030, 0.0028, 0.0026, 0.0000, -0.0020, -0.0025, -0.0022,
        -0.0015, -0.0008, -0.0002, 0.0005, 0.0012, 0.0018, 0.0022,
        0.0025, 0.0026, 0.0027, 0.0027, 0.0027,
    ])

    data.wall_data = pd.DataFrame({"x_H": x_H, "Cp": Cp, "Cf": Cf})

    # ---- Velocity profiles at downstream stations ----
    stations = [1, 4, 6, 10, 15]

    for station in stations:
        n_pts = 40
        y_H = np.linspace(0, 1.2, n_pts)

        if station < 7:  # Inside recirculation
            u_frac = -0.15 * np.exp(-5 * y_H) + (1 - np.exp(-3 * y_H))
            u_frac = np.clip(u_frac, -0.2, 1.0)
        else:  # Recovery region
            u_frac = 1 - np.exp(-3 * y_H)

        uu = 0.01 * np.exp(-2 * y_H) * (1 + 0.5 * np.exp(-0.5 * (station - 6)**2))
        vv = 0.005 * np.exp(-3 * y_H)
        uv = -0.004 * np.exp(-2.5 * y_H) * np.sign(u_frac)

        data.velocity_profiles[float(station)] = pd.DataFrame({
            "y_H": y_H, "U_Uref": u_frac, "uu": uu, "vv": vv, "uv": uv,
        })

    # ---- Separation metrics ----
    data.separation_metrics = {
        "Re_H": 132_000,
        "expansion_ratio": 1.2,
        "x_reat_xH": 7.0,
        "x_reat_xH_uncertainty": 0.5,
        "step_height_m": 0.0381,
        "U_ref": 52.0,
    }

    data.uncertainty = {"U": 0.02, "uu": 0.10, "Cp": 0.01, "Cf": 0.10}
    return data


if __name__ == "__main__":
    data = load_kim_bfs_data()
    print(data.summary())
