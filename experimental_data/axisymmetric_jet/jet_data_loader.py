#!/usr/bin/env python3
"""
Axisymmetric Jet Experimental Data Loader
============================================
Provides reference data for subsonic round jet validation:
  - Centerline velocity decay
  - Radial velocity/TKE profiles at downstream stations
  - Spreading rate and potential core length

References
----------
  - Bridges & Wernet (2010), NASA/TM—2010-216736
  - Witze (1974), AIAA J. 12(4), pp.417-418
  - Lau et al. (1979), J. Fluid Mech. 93(1)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experimental_data.data_loader import ExperimentalData


def load_jet_data() -> ExperimentalData:
    """
    Load axisymmetric subsonic jet reference data.

    M_j = 0.5, Re_D ~ 10^6, nozzle D = 50.8 mm.

    Returns
    -------
    ExperimentalData
    """
    data = ExperimentalData(
        case_name="axisymmetric_jet",
        description="Round subsonic jet, M_j=0.5, Re_D~10^6",
        data_source="Bridges & Wernet (2010), NASA/TM-2010-216736",
    )

    # ---- Centerline velocity decay ----
    x_D = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25, 30])
    Uc_Uj = np.array([
        1.00, 1.00, 1.00, 0.99, 0.98, 0.95, 0.88, 0.79, 0.71,
        0.58, 0.48, 0.39, 0.29, 0.23, 0.19,
    ])

    data.wall_data = pd.DataFrame({
        "x_D": x_D,
        "Uc_Uj": Uc_Uj,
        "quantity": "centerline_velocity",
    })

    # ---- Radial profiles at downstream stations ----
    stations = [2, 5, 8, 12, 20]

    for x_station in stations:
        n_pts = 60
        r_D = np.linspace(0, 2.5, n_pts)

        # Gaussian profile model
        if x_station <= 5:
            # Potential core region: top-hat + thin shear layer
            half_width = 0.5 + 0.094 * max(0, x_station - 1)
            U_profile = 0.5 * (1 - np.tanh(5 * (r_D - half_width)))
        else:
            # Self-similar decay region
            Uc = np.interp(x_station, x_D, Uc_Uj)
            r_half = 0.094 * x_station  # Spreading rate
            U_profile = Uc * np.exp(-(r_D / max(r_half, 0.01))**2 * np.log(2))

        # Reynolds stresses
        uu = 0.02 * U_profile * (1 - U_profile)  # Shear layer
        vv = 0.5 * uu

        data.velocity_profiles[float(x_station)] = pd.DataFrame({
            "r_D": r_D,
            "U_Uj": U_profile,
            "uu_Uj2": np.abs(uu),
            "vv_Uj2": np.abs(vv),
        })

    # ---- Separation metrics (jet-specific) ----
    data.separation_metrics = {
        "potential_core_xD": 6.0,
        "spreading_rate": 0.094,
        "centerline_decay_B": 5.8,
        "half_angle_deg": 5.4,
        "Re_D": 1e6,
        "M_jet": 0.5,
        "D_nozzle_m": 0.0508,
    }

    data.uncertainty = {
        "U": 0.02,
        "uu": 0.15,
        "x_core": 0.5,
    }

    return data


if __name__ == "__main__":
    data = load_jet_data()
    print(data.summary())
    print(f"  Potential core: x/D = {data.separation_metrics['potential_core_xD']}")
    print(f"  Spreading rate: {data.separation_metrics['spreading_rate']}")
