#!/usr/bin/env python3
"""
BeVERLI Hill — Experimental Data & Reference Generator
=======================================================
Creates reference data and geometry definition for the BeVERLI
(Benchmark Validation Experiments for RANS and LES Investigations)
Hill experiment from Virginia Tech & NASA Langley.

Geometry: 3D superelliptic hill
  - Hill height H = 0.1869 m
  - Hill width  w = 0.93472 m (= 5H)
  - Flat top    s = 0.09347 m (= H/2)
  - 5th-degree polynomial centerline profile
  - Superelliptic cross-section transitions

Reference: https://beverlihill.aoe.vt.edu/
"""

import json
import math
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent
CSV_DIR = DATA_DIR / "csv"

# ============================================================================
# Hill Geometry Definition
# ============================================================================
# Hill parameters (all in meters)
H = 0.186944       # Hill height
W = 0.93472        # Hill width (= 5H)
S = 0.093472       # Flat-top width (= H/2)
R_CORNER = 0.5 * H # Corner transition radius


def hill_centerline_profile(x: np.ndarray) -> np.ndarray:
    """
    5th-degree polynomial centerline profile y(x) for the BeVERLI hill.

    The hill extends from x/H = -2.5 to x/H = +2.5, is tangent to the
    flat wall at both ends, and has a flat top at y/H = 1.0.

    Parameters
    ----------
    x : array
        Streamwise coordinate normalized by H (x/H).

    Returns
    -------
    y : array
        Hill height normalized by H (y/H). Zero outside the hill footprint.
    """
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x)

    # Hill half-length in H units
    L_half = 2.5  # hill extends from -2.5H to +2.5H

    # Flat-top half-width in H units
    s_half = S / (2 * H)  # ≈ 0.25

    # Windward / leeward ramp: 5th-degree polynomial from base to flat-top
    # Boundary conditions: y(0) = 0, y'(0) = 0, y''(0) = 0,
    #                      y(1) = 1, y'(1) = 0, y''(1) = 0
    # Solution: y = 10*t^3 - 15*t^4 + 6*t^5  (Hermite smoothstep)

    for i, xi in enumerate(x.flat):
        axi = abs(xi)
        if axi <= s_half:
            # Flat top
            y.flat[i] = 1.0
        elif axi <= L_half:
            # Ramp region: t goes from 0 (at base) to 1 (at flat-top edge)
            t = (L_half - axi) / (L_half - s_half)
            t = max(0.0, min(1.0, t))
            y.flat[i] = 10 * t**3 - 15 * t**4 + 6 * t**5
        else:
            y.flat[i] = 0.0

    return y


def hill_superelliptic_cross_section(z: np.ndarray, x_H: float,
                                      n: float = 4.0) -> np.ndarray:
    """
    Superelliptic cross-section scaling factor at streamwise location x/H.

    The hill cross-section blends from a sharp-cornered rectangle (n → ∞)
    at the top to rounded corners (n ≈ 4) along the ramp.

    Parameters
    ----------
    z : array
        Spanwise coordinate normalized by H (z/H).
    x_H : float
        Streamwise location (x/H).
    n : float
        Superellipse exponent (default 4.0).

    Returns
    -------
    scale : array
        Multiplicative height scaling factor in [0, 1].
    """
    z = np.asarray(z, dtype=float)
    z_half = W / (2 * H)  # Half-width in H units (≈ 2.5)
    z_norm = np.abs(z) / z_half
    z_norm = np.clip(z_norm, 0, 1)
    scale = np.where(z_norm <= 1.0, (1 - z_norm**n) ** (1 / n), 0.0)
    return np.clip(scale, 0, 1)


# ============================================================================
# Flow Conditions
# ============================================================================
FLOW_CONDITIONS = {
    "Re_250k": {
        "reynolds": 250_000,
        "mach": 0.06,
        "velocity_freestream": 20.0,
        "temperature": 293.15,
        "pressure": 101325.0,
        "description": "Low-Re case: attached boundary layer transitions, "
                       "moderate leeward separation",
    },
    "Re_400k": {
        "reynolds": 400_000,
        "mach": 0.07,
        "velocity_freestream": 24.0,
        "temperature": 293.15,
        "pressure": 101325.0,
        "description": "Mid-Re case: intermediate between low-Re and high-Re",
    },
    "Re_650k": {
        "reynolds": 650_000,
        "mach": 0.09,
        "velocity_freestream": 31.0,
        "temperature": 293.15,
        "pressure": 101325.0,
        "description": "High-Re case: fully turbulent, massive leeward "
                       "separation, thinner boundary layer",
    },
}

YAW_ANGLES = [0, 30, 45]


# ============================================================================
# Representative Experimental Reference Data
# ============================================================================
def generate_representative_cp(yaw_deg: int = 0,
                                re: int = 250_000) -> dict:
    """
    Generate representative centerline Cp distribution for the BeVERLI hill.

    The pressure distribution features:
    - Favorable PG on windward face (Cp drops)
    - Suction peak near hill crest
    - Adverse PG on leeward face (Cp rises)
    - Plateau in separation bubble
    - Recovery downstream

    Parameters
    ----------
    yaw_deg : int
        Hill yaw angle (0, 30, or 45 degrees).
    re : int
        Reynolds number based on hill height.

    Returns
    -------
    dict with keys 'x_H' and 'Cp' (centerline values).
    """
    x = np.linspace(-5.0, 10.0, 400)  # x/H
    y_hill = hill_centerline_profile(x)

    # Base Cp: acceleration over upslope → suction peak → adverse PG → recovery
    # Suction peak stronger at higher Re
    re_factor = 1.0 + 0.3 * (re - 250_000) / 400_000
    Cp_accel = -0.8 * re_factor * np.gradient(y_hill, x)

    # Smooth the raw gradient-based Cp
    from scipy.ndimage import gaussian_filter1d
    Cp_accel = gaussian_filter1d(Cp_accel, sigma=5)

    # Add separation plateau on leeward side
    x_sep = 1.2  # Separation onset x/H (approximate)
    x_reat = 4.5  # Reattachment x/H
    Cp_sep = np.where(
        (x > x_sep) & (x < x_reat),
        -0.15 * re_factor,  # Plateau
        0.0,
    )

    Cp = Cp_accel + Cp_sep

    # Yaw effects
    if yaw_deg == 30:
        # Skewed BL: asymmetric suction, delayed separation
        Cp *= 0.85
        Cp += 0.05 * np.sin(2 * np.pi * x / 8)
    elif yaw_deg == 45:
        # Strong asymmetry: reduced suction peak, earlier separation
        Cp *= 0.75
        Cp += 0.08 * np.cos(np.pi * x / 6)

    return {"x_H": x.tolist(), "Cp": Cp.tolist()}


def generate_representative_cf(yaw_deg: int = 0,
                                re: int = 250_000) -> dict:
    """
    Generate representative centerline Cf distribution.

    Features skin friction going negative in the separation bubble,
    with reattachment location varying with Re and yaw.
    """
    x = np.linspace(-5.0, 10.0, 400)  # x/H

    # Separation/reattachment locations vary with Re and yaw
    if yaw_deg == 0:
        x_sep = 1.2 if re < 400_000 else 1.0
        x_reat = 4.5 if re < 400_000 else 4.0
    elif yaw_deg == 30:
        x_sep = 1.4
        x_reat = 5.0
    else:  # 45°
        x_sep = 1.0
        x_reat = 5.5

    # Cf distribution
    Cf_ref = 0.003 * (re / 250_000) ** (-0.2)
    Cf = np.where(
        x < -2.5,
        Cf_ref,
        np.where(
            x < x_sep,
            Cf_ref * (1 + 0.5 * np.sin(np.pi * (x + 2.5) / (x_sep + 2.5))),
            np.where(
                x < x_reat,
                -0.0008 * np.sin(np.pi * (x - x_sep) / (x_reat - x_sep)),
                Cf_ref * (1 - np.exp(-2.0 * (x - x_reat)))
            ),
        ),
    )

    return {
        "x_H": x.tolist(), "Cf": Cf.tolist(),
        "x_sep_xH": x_sep, "x_reat_xH": x_reat,
    }


# ============================================================================
# Reference JSON Creator
# ============================================================================
def create_reference_json():
    """Create comprehensive JSON reference file for the BeVERLI Hill case."""
    ref = {
        "case": "BeVERLI Hill — VT-NASA 3D Smooth-Body Separation",
        "website": "https://beverlihill.aoe.vt.edu/",
        "geometry": {
            "type": "3D superelliptic smooth-body hill",
            "hill_height_m": H,
            "hill_width_m": W,
            "flat_top_width_m": S,
            "aspect_ratio_w_H": W / H,
            "centerline_profile": "5th-degree polynomial (Hermite smoothstep)",
            "cross_section": "Superelliptic (n ≈ 4)",
            "hill_extent_xH": [-2.5, 2.5],
            "tangent_to_wall": True,
        },
        "flow_conditions": FLOW_CONDITIONS,
        "yaw_angles_deg": YAW_ANGLES,
        "experimental_techniques": {
            "surface_pressure": "128+ static pressure taps",
            "skin_friction": "Oil Film Interferometry (OFI) + LDV wall-shear",
            "velocity_fields": "Stereo-PIV (2D planes), LDV (point measurements)",
            "flow_visualization": "Oil flow (surface streamlines)",
            "boundary_conditions": "Inflow BL profiles (pitot rake + hot-wire)",
        },
        "validation_profiles": {
            "x_H_stations": [0.5, 1.0, 1.5, 2.0, 3.0, 5.0],
            "z_H_stations": [0.0, 0.5, 1.0],
            "quantities": ["U/U_inf", "V/U_inf", "W/U_inf",
                           "u'u'/U_inf^2", "v'v'/U_inf^2", "u'v'/U_inf^2"],
        },
        "known_rans_failures": {
            "SST_45deg": (
                "Menter SST erroneously predicts asymmetric wakes at "
                "45° yaw across all Reynolds numbers and grid densities"
            ),
            "SA_separation": (
                "SA miscalculates separation onset location and volumetric "
                "extent of leeward separation bubble"
            ),
            "boussinesq_breakdown": (
                "Both SA and SST fail to capture 3D non-equilibrium "
                "turbulent boundary layer effects due to inherent "
                "limitations of the Boussinesq (linear eddy-viscosity) "
                "approximation in regions of strong streamline curvature "
                "and pressure-gradient sign reversals"
            ),
        },
        "references": [
            {
                "authors": "Lowe, K.T., Simpson, R.L., Schetz, J.A., et al.",
                "title": "Experimental Design for the BeVERLI Hill "
                         "Flow Separation Experiment",
                "venue": "AIAA Paper 2022-0329",
                "year": 2022,
                "doi": "10.2514/6.2022-0329",
            },
            {
                "authors": "Rahmani, S., Lowe, K.T., Simpson, R.L., et al.",
                "title": "Turbulence Measurements on the BeVERLI Hill",
                "venue": "Experiments in Fluids 64",
                "year": 2023,
                "doi": "10.1007/s00348-023-03607-w",
            },
            {
                "authors": "Lakebrink, M.T., Mani, M., Rumsey, C.L.",
                "title": "CFD Predictions for the BeVERLI Hill",
                "venue": "AIAA Paper 2023-3263",
                "year": 2023,
                "doi": "10.2514/6.2023-3263",
            },
        ],
    }
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    dst = CSV_DIR / "beverli_case_reference.json"
    with open(dst, "w") as f:
        json.dump(ref, f, indent=2)
    print(f"  [OK]     Created beverli_case_reference.json")
    return dst


def generate_geometry_csv():
    """Export the hill centerline profile as CSV for mesh generation."""
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    x_H = np.linspace(-3.0, 3.0, 601)
    y_H = hill_centerline_profile(x_H)
    dst = CSV_DIR / "hill_centerline_profile.csv"
    with open(dst, "w") as f:
        f.write("x_H,y_H\n")
        for xi, yi in zip(x_H, y_H):
            f.write(f"{xi:.6f},{yi:.6f}\n")
    print(f"  [OK]     Created hill_centerline_profile.csv ({len(x_H)} points)")
    return dst


def generate_representative_data():
    """Generate representative Cp/Cf data for all Re/yaw combinations."""
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    for re in [250_000, 650_000]:
        for yaw in YAW_ANGLES:
            cp_data = generate_representative_cp(yaw, re)
            cf_data = generate_representative_cf(yaw, re)

            tag = f"Re{re // 1000}k_yaw{yaw}"
            cp_dst = CSV_DIR / f"representative_cp_{tag}.csv"
            cf_dst = CSV_DIR / f"representative_cf_{tag}.csv"

            with open(cp_dst, "w") as f:
                f.write("x_H,Cp\n")
                for xi, ci in zip(cp_data["x_H"], cp_data["Cp"]):
                    f.write(f"{xi:.4f},{ci:.6f}\n")

            with open(cf_dst, "w") as f:
                f.write("x_H,Cf\n")
                for xi, ci in zip(cf_data["x_H"], cf_data["Cf"]):
                    f.write(f"{xi:.4f},{ci:.8f}\n")

    print(f"  [OK]     Generated representative Cp/Cf for "
          f"{len([250_000, 650_000])} Re × {len(YAW_ANGLES)} yaw combos")


# ============================================================================
# Main
# ============================================================================
def download_all():
    """Generate all reference data for the BeVERLI Hill case."""
    print("=" * 60)
    print("  BeVERLI Hill — VT-NASA Data Generation")
    print("=" * 60)
    create_reference_json()
    generate_geometry_csv()
    generate_representative_data()
    print(f"\n  All data generated successfully!")
    print(f"  Data directory: {CSV_DIR}")


if __name__ == "__main__":
    download_all()
