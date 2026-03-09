"""
Unified Experimental Data Loader
================================
Loads reference data for all 17 benchmark cases from local files or generates
representative profiles for initial development. Provides a single API:

    data = load_case("backward_facing_step")
    data.velocity_profiles   # {station: DataFrame}
    data.wall_data           # DataFrame with x, Cp, Cf
    data.separation_metrics  # dict with x_sep, x_reat, etc.
"""

import os
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Resolve project root
_THIS_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = _THIS_DIR.parent

# Add project root so we can import config
sys.path.insert(0, str(PROJECT_ROOT))
from config import BENCHMARK_CASES, BenchmarkCase


# =============================================================================
# Data Container
# =============================================================================
@dataclass
class ExperimentalData:
    """Container for one benchmark case's reference data."""
    case_name: str
    description: str = ""
    data_source: str = ""
    is_synthetic: bool = False

    # Velocity/Reynolds-stress profiles at discrete stations
    velocity_profiles: Dict[float, pd.DataFrame] = field(default_factory=dict)

    # Wall data: x, Cp, Cf (possibly with uncertainty)
    wall_data: Optional[pd.DataFrame] = None

    # Key separation metrics
    separation_metrics: Dict[str, float] = field(default_factory=dict)

    # Experimental uncertainty (±)
    uncertainty: Dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        n_profiles = len(self.velocity_profiles)
        wall_pts = len(self.wall_data) if self.wall_data is not None else 0
        synthetic_warning = (
            "\n  ⚠️  WARNING: DATA IS SYNTHETIC (generated from analytical formulas, "
            "NOT real experimental data)" if self.is_synthetic else ""
        )
        return (
            f"Case: {self.case_name}\n"
            f"  Source: {self.data_source}\n"
            f"  Profiles: {n_profiles} stations\n"
            f"  Wall data points: {wall_pts}\n"
            f"  Separation metrics: {self.separation_metrics}"
            f"{synthetic_warning}"
        )


# =============================================================================
# Loader Dispatch
# =============================================================================
def load_case(
    case_name: str,
    data_dir: Optional[Path] = None,
    allow_synthetic: bool = True,
) -> ExperimentalData:
    """
    Load reference data for a benchmark case.

    Parameters
    ----------
    case_name : str
        Key from config.BENCHMARK_CASES (e.g., "backward_facing_step").
    data_dir : Path, optional
        Override data directory (default: experimental_data/{case_name}).
    allow_synthetic : bool
        If True (default), fall back to synthetic representative data when
        no real CSV/DAT files are found. If False, raise FileNotFoundError.
        **Callers performing validation should set this to False.**

    Returns
    -------
    ExperimentalData
        Container with profiles, wall data, and metrics.
    """
    if case_name not in BENCHMARK_CASES:
        raise ValueError(
            f"Unknown case '{case_name}'. Available: {list(BENCHMARK_CASES.keys())}"
        )

    case_cfg = BENCHMARK_CASES[case_name]
    
    # Override paths for specific experimental datasets
    if case_name == "backward_facing_step":
        base_dir = data_dir or (_THIS_DIR / "bfs_driver_seegmiller" / "csv")
    elif case_name == "periodic_hill":
        base_dir = data_dir or (_THIS_DIR / "periodic_hill" / "csv")
    elif case_name == "axi_swbli":
        base_dir = data_dir or (_THIS_DIR / "swbli_mach5" / "csv")
    else:
        base_dir = data_dir or (_THIS_DIR / case_name)

    # Try to load from files first; fall back to representative data
    # Check for actual data files (CSV, DAT) or profiles directory,
    # not just any files (e.g. .py scripts) in the case directory
    if base_dir.exists() and _has_data_files(base_dir):
        return _load_from_files(case_name, case_cfg, base_dir)
    else:
        if not allow_synthetic:
            raise FileNotFoundError(
                f"No experimental data files found for '{case_name}' in "
                f"{base_dir}. Set allow_synthetic=True to use synthetic "
                f"placeholder data (NOT suitable for validation claims)."
            )
        warnings.warn(
            f"No experimental data files found for '{case_name}' in "
            f"{base_dir}. Falling back to SYNTHETIC data generated from "
            f"analytical formulas (np.tanh/np.sin). This data is NOT real "
            f"experimental data and MUST NOT be used for validation claims.",
            UserWarning,
            stacklevel=2,
        )
        return _generate_representative(case_name, case_cfg)


def _has_data_files(data_dir: Path) -> bool:
    """Check if a directory contains actual experimental data files."""
    if (data_dir / "profiles").exists():
        return True
    if (data_dir / "wall_data.csv").exists():
        return True
    if (data_dir / "metrics.csv").exists():
        return True
    # Check for any CSV or DAT files directly in the directory
    for ext in ("*.csv", "*.dat"):
        if any(data_dir.glob(ext)):
            return True
    return False


def load_all_cases(
    allow_synthetic: bool = True,
) -> Dict[str, ExperimentalData]:
    """Load reference data for every registered benchmark case."""
    return {
        name: load_case(name, allow_synthetic=allow_synthetic)
        for name in BENCHMARK_CASES
    }


# =============================================================================
# File-Based Loading
# =============================================================================
def _load_from_files(
    case_name: str, case_cfg: BenchmarkCase, data_dir: Path
) -> ExperimentalData:
    """Load experimental data from CSV/DAT files in the case directory."""
    data = ExperimentalData(
        case_name=case_name,
        description=case_cfg.description,
        data_source=case_cfg.data_source,
    )

    # Load velocity profiles
    profiles_dir = data_dir / "profiles"
    if profiles_dir.exists():
        for f in sorted(profiles_dir.glob("*.csv")):
            station = _parse_station(f.stem)
            df = pd.read_csv(f)
            data.velocity_profiles[station] = df

    # Load wall data (Cp, Cf) - handle unified or split files
    wall_file = data_dir / "wall_data.csv"
    if wall_file.exists():
        data.wall_data = pd.read_csv(wall_file)
    else:
        # Check for individual _cp.csv and _cf.csv files
        cp_file = list(data_dir.glob("*_cp.csv"))
        cf_file = list(data_dir.glob("*_cf.csv"))
        
        df_wall = None
        if cp_file:
            df_cp = pd.read_csv(cp_file[0])
            # Assuming first column is x-coordinate
            x_col = df_cp.columns[0]
            df_wall = df_cp
        
        if cf_file:
            df_cf = pd.read_csv(cf_file[0])
            x_col = df_cf.columns[0]
            if df_wall is not None:
                # Merge on x-coordinate
                df_wall = pd.merge(df_wall, df_cf, on=x_col, how='outer').sort_values(x_col)
            else:
                df_wall = df_cf
                
        if df_wall is not None:
            # Rename the x-coordinate column to simply 'x' for standardization if it isn't already
            x_col = df_wall.columns[0]
            if x_col.lower() != 'x':
                df_wall = df_wall.rename(columns={x_col: 'x'})
            data.wall_data = df_wall

    # Load separation metrics
    metrics_file = data_dir / "metrics.csv"
    if metrics_file.exists():
        df = pd.read_csv(metrics_file)
        data.separation_metrics = df.to_dict(orient="records")[0]
    
    # Calculate some basic separation metrics on the fly if Cf data exists and metrics aren't provided
    if not data.separation_metrics and data.wall_data is not None and 'Cf' in data.wall_data.columns:
        cf_data = data.wall_data.dropna(subset=['Cf'])
        if not cf_data.empty:
            # Naive separation/reattachment detection based on zero crossings
            zero_crossings = np.where(np.diff(np.sign(cf_data['Cf'])))[0]
            if len(zero_crossings) > 0:
                coords = cf_data['x'].iloc[zero_crossings].values
                if len(coords) >= 1:
                    data.separation_metrics['x_sep'] = coords[0]
                if len(coords) >= 2:
                    data.separation_metrics['x_reat'] = coords[1]

    # Uncertainty
    data.uncertainty = {
        "U": 0.02,       # ±2% velocity
        "uu": 0.10,      # ±10% Reynolds stress
        "Cp": 0.01,      # ±0.01 Cp
        "Cf": 0.10,      # ±10% Cf
    }

    return data


def _parse_station(stem: str) -> float:
    """Extract station value from filename like 'xH_4.0' → 4.0."""
    parts = stem.split("_")
    try:
        return float(parts[-1])
    except (ValueError, IndexError):
        return 0.0


# =============================================================================
# Representative Data Generators (for development & testing)
# =============================================================================
def _generate_representative(
    case_name: str, case_cfg: BenchmarkCase
) -> ExperimentalData:
    """Generate SYNTHETIC data from analytical formulas for development only.

    WARNING: This data is fabricated using np.tanh/np.sin/np.exp formulas.
    It is NOT real experimental or simulation data and MUST NOT be presented
    as validation evidence.
    """
    generators = {
        "flat_plate": _gen_flat_plate,
        "backward_facing_step": _gen_bfs,
        "nasa_hump": _gen_hump,
        "periodic_hill": _gen_periodic_hill,
        "bachalo_johnson": _gen_bachalo_johnson,
        "naca_0012_stall": _gen_naca_0012_stall,
        "naca_4412_te": _gen_naca_4412_te,
        "juncture_flow": _gen_juncture_flow,
        "axi_swbli": _gen_axi_swbli,
        "obi_diffuser": _gen_obi_diffuser,
        "beverli_hill": _gen_beverli_hill,
        "boeing_gaussian_bump": _gen_gaussian_bump,
    }
    gen_func = generators.get(case_name, _gen_generic)
    result = gen_func(case_name, case_cfg)
    # Always tag synthetic data clearly
    result.is_synthetic = True
    if not result.data_source.startswith("SYNTHETIC"):
        result.data_source = f"SYNTHETIC — {result.data_source}"
    return result


# ---- Flat Plate ----
def _gen_flat_plate(name: str, cfg: BenchmarkCase) -> ExperimentalData:
    """Analytical Blasius/turbulent BL for flat plate verification."""
    data = ExperimentalData(
        case_name=name,
        description=cfg.description,
        data_source="Analytical (Blasius + log-law)",
    )

    nu = cfg.reference_velocity * cfg.reference_length / cfg.reynolds_number

    # Wall data: Cf along plate
    x = np.linspace(0.1, cfg.reference_length, 200)
    Re_x = cfg.reference_velocity * x / nu
    Cf_lam = 0.664 / np.sqrt(Re_x)
    Cf_turb = 0.059 / Re_x**0.2
    # Transition around Re_x ~ 5e5
    transition = 1.0 / (1.0 + np.exp(-(Re_x - 5e5) / 1e5))
    Cf = (1 - transition) * Cf_lam + transition * Cf_turb

    data.wall_data = pd.DataFrame({"x": x, "Cf": Cf})

    # Velocity profiles at selected stations
    for x_loc in [1.0, 3.0, 5.0, 8.0]:
        Re_xl = cfg.reference_velocity * x_loc / nu
        delta = 0.37 * x_loc / Re_xl**0.2  # Turbulent BL thickness

        y = np.linspace(1e-6, 2 * delta, 100)
        u_tau = cfg.reference_velocity * np.sqrt(Cf_turb[np.argmin(np.abs(x - x_loc))] / 2)
        yplus = y * u_tau / nu

        # Log-law velocity profile
        U_plus = np.where(
            yplus < 5,
            yplus,  # Viscous sublayer
            np.where(
                yplus < 30,
                5.0 * np.log(yplus) - 3.05,  # Buffer layer
                (1 / 0.41) * np.log(yplus) + 5.0,  # Log layer
            ),
        )
        U = U_plus * u_tau
        U = np.minimum(U, cfg.reference_velocity)

        data.velocity_profiles[x_loc] = pd.DataFrame({
            "y": y, "y_plus": yplus, "U": U, "U_plus": U_plus,
        })

    data.separation_metrics = {}  # No separation
    data.uncertainty = {"U": 0.0, "Cf": 0.0}  # Analytical = exact
    return data


# ---- Backward-Facing Step ----
def _gen_bfs(name: str, cfg: BenchmarkCase) -> ExperimentalData:
    """Representative BFS profiles based on Driver & Seegmiller data."""
    data = ExperimentalData(
        case_name=name,
        description=cfg.description,
        data_source="Representative (Driver & Seegmiller 1985)",
    )

    H = cfg.reference_length
    U_inf = cfg.reference_velocity
    x_R = cfg.separation_metrics.get("x_reat_xH", 6.26)

    # Wall data: Cf
    x_wall = np.linspace(0, 20, 200)
    x_norm = x_wall / x_R
    Cf = np.where(
        x_norm < 1.0,
        -0.002 * np.sin(np.pi * x_norm),  # Reversed flow in bubble
        0.003 * (1 - np.exp(-0.5 * (x_norm - 1))),  # Recovery
    )
    Cp = np.where(
        x_norm < 0.5,
        0.0,
        0.2 * (1 - np.exp(-2 * (x_norm - 0.5))),  # Pressure recovery
    )
    data.wall_data = pd.DataFrame({"x_H": x_wall, "Cf": Cf, "Cp": Cp})

    # Velocity profiles at stations x/H = 1, 4, 6, 10
    for xH in [1.0, 4.0, 6.0, 10.0]:
        y = np.linspace(0, 3, 100)  # y/H
        fraction_recovered = min(xH / x_R, 1.0)

        if xH < x_R:
            # Inside recirculation
            U = U_inf * (
                -0.2 * (1 - fraction_recovered) * np.exp(-2 * y)
                + (1 - np.exp(-2 * y))
            )
        else:
            # Downstream of reattachment
            U = U_inf * (1 - np.exp(-2.5 * y))

        uu = 0.01 * U_inf**2 * np.exp(-1.5 * (y - 0.5)**2)
        uv = -0.003 * U_inf**2 * np.exp(-2 * (y - 0.5)**2)

        data.velocity_profiles[xH] = pd.DataFrame({
            "y_H": y, "U": U, "uu": uu, "uv": uv,
        })

    data.separation_metrics = {
        "x_sep_xH": 0.0,
        "x_reat_xH": x_R,
        "bubble_length_xH": x_R,
    }
    data.uncertainty = {"U": 0.02, "uu": 0.10, "Cf": 0.10}
    return data


# ---- NASA Hump ----
def _gen_hump(name: str, cfg: BenchmarkCase) -> ExperimentalData:
    """Representative hump profiles based on Greenblatt et al. PIV data."""
    data = ExperimentalData(
        case_name=name,
        description=cfg.description,
        data_source="Representative (Greenblatt et al. 2006)",
    )

    x_sep = cfg.separation_metrics.get("x_sep_xc", 0.665)
    x_reat = cfg.separation_metrics.get("x_reat_xc", 1.11)
    U_ref = cfg.reference_velocity

    # Wall data
    x = np.linspace(-0.5, 2.0, 300)
    # Hump surface: Glauert-Goldschmied shape
    hump_height = 0.128 * np.exp(-18 * (x - 0.5)**2) * (x > 0) * (x < 1)
    # Cp distribution
    Cp = np.where(x < 0.3, 0.0,
         np.where(x < x_sep, -0.8 * np.sin(np.pi * (x - 0.3) / 0.4),
         np.where(x < x_reat, -0.3 + 0.1 * (x - x_sep),
                  -0.1 * np.exp(-3 * (x - x_reat)))))
    # Cf distribution — ensure Cf is clearly positive upstream of separation
    # Using np.abs instead of np.maximum to avoid zero-derivative regions that
    # confuse the find_reattachment_point crossing detection algorithms.
    Cf = np.where(x < x_sep, 0.0005 + 0.003 * np.abs(1 + 2 * x),
         np.where(x < x_reat, -0.001 * np.sin(np.pi * (x - x_sep) / (x_reat - x_sep)),
                  0.002 * (1 - np.exp(-5 * (x - x_reat)))))

    data.wall_data = pd.DataFrame({
        "x_c": x, "Cp": Cp, "Cf": Cf, "hump_surface": hump_height,
    })

    # Velocity profiles at 7 stations
    for xc in [0.65, 0.80, 0.90, 1.00, 1.10, 1.20, 1.30]:
        y = np.linspace(0, 0.15, 80)  # y in meters

        if xc < x_sep:
            U = U_ref * np.tanh(15 * y)
        elif xc < x_reat:
            frac = (xc - x_sep) / (x_reat - x_sep)
            reverse_strength = 0.15 * np.sin(np.pi * frac)
            U = U_ref * (-reverse_strength * np.exp(-50 * y) + np.tanh(12 * y))
        else:
            recovery = min((xc - x_reat) / 0.3, 1.0)
            U = U_ref * (1 - (1 - recovery) * 0.3) * np.tanh(10 * y)

        uu = 0.008 * U_ref**2 * np.exp(-30 * (y - 0.02)**2)
        uv = -0.003 * U_ref**2 * np.exp(-40 * (y - 0.015)**2)

        data.velocity_profiles[xc] = pd.DataFrame({
            "y": y, "U": U, "uu": uu, "uv": uv,
        })

    data.separation_metrics = {
        "x_sep_xc": x_sep,
        "x_reat_xc": x_reat,
        "bubble_length_xc": x_reat - x_sep,
    }
    data.uncertainty = {"U": 0.02, "uu": 0.10, "Cp": 0.01, "Cf": 0.10}
    return data


# ---- Periodic Hill ----
def _gen_periodic_hill(name: str, cfg: BenchmarkCase) -> ExperimentalData:
    """Representative periodic hill profiles based on Breuer et al. DNS."""
    data = ExperimentalData(
        case_name=name,
        description=cfg.description,
        data_source="Representative (Breuer et al. DNS Re=10595)",
    )

    x_sep = cfg.separation_metrics.get("x_sep_xh", 0.22)
    x_reat = cfg.separation_metrics.get("x_reat_xh", 4.72)
    U_b = cfg.reference_velocity

    # Velocity profiles at stations x/h = 0.5, 2, 4, 6, 8
    for xh in [0.5, 2.0, 4.0, 6.0, 8.0]:
        y = np.linspace(0, 3.036, 100)

        if xh < x_reat:
            frac = (xh - x_sep) / (x_reat - x_sep)
            reverse = 0.1 * np.sin(np.pi * frac)
            U = U_b * (-reverse * np.exp(-3 * y) + 1.5 * y / 3.036 * (2 - y / 3.036))
        else:
            U = U_b * 1.5 * y / 3.036 * (2 - y / 3.036)

        data.velocity_profiles[xh] = pd.DataFrame({"y_h": y, "U": U})

    data.separation_metrics = {
        "x_sep_xh": x_sep,
        "x_reat_xh": x_reat,
        "bubble_length_xh": x_reat - x_sep,
    }
    data.uncertainty = {"U": 0.01}  # DNS = very low
    return data


# ---- Bachalo-Johnson Transonic Bump ----
def _gen_bachalo_johnson(name: str, cfg: BenchmarkCase) -> ExperimentalData:
    """
    Representative data for Bachalo-Johnson axisymmetric transonic bump.

    Based on Bachalo & Johnson (1986), AIAA J. 24(3), pp. 437-443.
    M=0.875, Re≈2.7×10⁶. Circular-arc bump on cylinder.
    Shock at x/c ≈ 0.65 → separation → reattachment at x/c ≈ 0.90.
    NASA TMR case: ATB (https://turbmodels.larc.nasa.gov/axibump_val.html)
    """
    data = ExperimentalData(
        case_name=name,
        description=cfg.description,
        data_source="Representative (Bachalo & Johnson 1986, AIAA J. 24(3))",
    )

    M_inf = 0.875
    x_shock = 0.65   # Shock location x/c
    x_sep = 0.68     # Separation onset (just downstream of shock foot)
    x_reat = 0.90    # Reattachment

    # Wall Cp distribution: supersonic acceleration → shock → separation
    x = np.linspace(-0.5, 1.5, 400)
    # Pre-bump: Cp ≈ 0
    Cp_pre = np.zeros_like(x)
    # Acceleration over bump (0 < x/c < 0.65): Cp drops to ~-0.6
    Cp_accel = -0.6 * np.sin(np.pi * np.clip(x, 0, x_shock) / x_shock)
    # Shock compression: rapid Cp rise
    shock_width = 0.03
    Cp_shock = 0.35 * (1 + np.tanh((x - x_shock) / shock_width))
    # Separation plateau + recovery
    Cp_sep = np.where(
        x < x_sep, 0.0,
        np.where(x < x_reat,
                 0.05 * (x - x_sep) / (x_reat - x_sep),  # Weak plateau
                 0.15 * np.exp(-3 * (x - x_reat)))        # Recovery
    )
    Cp = np.where(x < 0, 0.0,
         np.where(x < x_shock, Cp_accel,
                  Cp_shock + Cp_sep - 0.3))

    # Cf: attached turbulent → zero at separation → negative in bubble → recovery
    Cf = np.where(x < 0, 0.003,
         np.where(x < x_sep,
                  0.003 * (1 - 0.5 * np.clip(x, 0, 1)),
         np.where(x < x_reat,
                  -0.0008 * np.sin(np.pi * (x - x_sep) / (x_reat - x_sep)),
                  0.002 * (1 - np.exp(-8 * (x - x_reat))))))

    data.wall_data = pd.DataFrame({"x_c": x, "Cp": Cp, "Cf": Cf})

    # Velocity profiles at key stations
    for xc in [0.50, 0.65, 0.72, 0.80, 0.90, 1.00, 1.20]:
        y = np.linspace(0, 0.05, 80)  # y in meters (BL thickness ~few mm)

        if xc < x_sep:
            # Attached turbulent BL with acceleration
            U_edge = 1.0 + 0.15 * np.sin(np.pi * xc / x_shock)  # Accelerated
            U = U_edge * np.tanh(80 * y)
        elif xc < x_reat:
            # Separated region: reverse flow near wall
            frac = (xc - x_sep) / (x_reat - x_sep)
            reverse = 0.12 * np.sin(np.pi * frac)
            U = 1.0 * (-reverse * np.exp(-200 * y) + np.tanh(60 * y))
        else:
            # Recovery region
            recovery = min((xc - x_reat) / 0.3, 1.0)
            U = (0.85 + 0.15 * recovery) * np.tanh(50 * y)

        # Turbulence intensity peaks in shear layer
        uu = 0.02 * np.exp(-100 * (y - 0.008)**2)
        uv = -0.008 * np.exp(-120 * (y - 0.006)**2)

        data.velocity_profiles[xc] = pd.DataFrame({
            "y": y, "U_Uinf": U, "uu_Uinf2": uu, "uv_Uinf2": uv,
        })

    data.separation_metrics = {
        "x_shock_xc": x_shock,
        "x_sep_xc": x_sep,
        "x_reat_xc": x_reat,
        "bubble_length_xc": x_reat - x_sep,
        "mach_number": M_inf,
    }
    data.uncertainty = {"U": 0.03, "uu": 0.15, "Cp": 0.02, "Cf": 0.15}
    return data


# ---- NACA 0012 Stall ----
def _gen_naca_0012_stall(name: str, cfg: BenchmarkCase) -> ExperimentalData:
    """
    Representative data for NACA 0012 at stall conditions.

    Based on Gregory & O'Reilly (1970), NPL Aero Report 1308.
    NASA TMR: https://turbmodels.larc.nasa.gov/naca0012_val.html
    Re = 6×10⁶. CL_max ≈ 1.55 at α ≈ 16°. Trailing-edge stall type.
    """
    data = ExperimentalData(
        case_name=name,
        description=cfg.description,
        data_source="Representative (Gregory & O'Reilly 1970 / NASA TMR 2DN00)",
    )

    # CL vs alpha curve (key output for stall prediction)
    alpha_deg = np.linspace(0, 22, 45)
    alpha_rad = np.radians(alpha_deg)
    # Thin-airfoil theory slope ≈ 2π, but viscous effects reduce to ~5.7/rad
    CL_linear = 5.7 * alpha_rad
    # Stall model: CL_max ≈ 1.55 at α ≈ 16°
    alpha_stall = 16.0
    CL_max = 1.55
    stall_factor = 1.0 / (1.0 + np.exp(2.0 * (alpha_deg - alpha_stall)))
    CL_post_stall = CL_max * (1 - 0.4 * ((alpha_deg - alpha_stall) / 6)**2)
    CL = np.where(
        alpha_deg < alpha_stall - 2,
        np.minimum(CL_linear, CL_max),
        stall_factor * np.minimum(CL_linear, CL_max) + (1 - stall_factor) * CL_post_stall,
    )
    CL = np.clip(CL, 0.0, CL_max * 1.02)

    # CD: parasitic + induced + separation drag
    CD_p = 0.008  # Zero-lift parasitic
    CD_i = CL**2 / (np.pi * 6.0)  # AR ≈ 6 for typical test
    CD_sep = np.where(alpha_deg > 12, 0.002 * np.maximum(alpha_deg - 12, 0)**1.5, 0.0)
    CD = CD_p + CD_i + CD_sep

    # Cp distribution at selected angles of attack
    x_c = np.linspace(0, 1, 200)  # Chord-normalized
    # NACA 0012 thickness distribution
    t = 0.12  # thickness ratio
    y_t = 5 * t * (0.2969 * np.sqrt(x_c) - 0.1260 * x_c
                    - 0.3516 * x_c**2 + 0.2843 * x_c**3 - 0.1015 * x_c**4)

    profiles = {}
    for alpha in [0.0, 4.0, 8.0, 12.0, 15.0, 16.0, 18.0]:
        alpha_r = np.radians(alpha)
        # Simplified Cp from thin-airfoil + correction
        # Upper surface: suction peak then adverse PG
        Cp_upper = np.where(
            x_c < 0.05,
            -2 * alpha_r * 10 * (0.05 - x_c) / 0.05,  # Suction peak
            -2 * alpha_r * np.exp(-2 * x_c) + 0.1 * x_c,  # Recovery
        )
        # Stall: separation from trailing edge moving forward
        if alpha > 12:
            sep_frac = min((alpha - 12) / 8, 0.6)  # Separation covers 0-60% from TE
            x_sep_upper = 1.0 - sep_frac
            Cp_upper = np.where(
                x_c > x_sep_upper,
                Cp_upper[np.argmin(np.abs(x_c - x_sep_upper))],  # Cp plateau
                Cp_upper,
            )

        Cp_lower = 2 * alpha_r * np.exp(-3 * x_c)  # Positive pressure on lower

        profiles[alpha] = pd.DataFrame({
            "x_c": x_c,
            "Cp_upper": Cp_upper,
            "Cp_lower": Cp_lower,
            "y_upper": y_t,
        })

    data.velocity_profiles = profiles  # Using profiles dict for Cp at different α

    data.wall_data = pd.DataFrame({
        "alpha_deg": alpha_deg,
        "CL": CL,
        "CD": CD,
        "CL_CD": CL / np.maximum(CD, 1e-6),
    })

    data.separation_metrics = {
        "alpha_stall_deg": alpha_stall,
        "CL_max": CL_max,
        "CD_at_CLmax": float(np.interp(alpha_stall, alpha_deg, CD)),
        "reynolds_number": cfg.reynolds_number,
    }
    data.uncertainty = {"CL": 0.02, "CD": 0.001, "Cp": 0.02}
    return data


# ---- NACA 4412 Trailing-Edge Separation ----
def _gen_naca_4412_te(name: str, cfg: BenchmarkCase) -> ExperimentalData:
    """
    Representative data for NACA 4412 trailing-edge separation.

    Based on Coles & Wadcock (1979). NASA TMR 2DN44 case.
    Re = 1.5×10⁶, α = 13.87°. Trailing-edge separation from ~x/c = 0.75.
    """
    data = ExperimentalData(
        case_name=name,
        description=cfg.description,
        data_source="Representative (Coles & Wadcock 1979 / NASA TMR 2DN44)",
    )

    x_sep = 0.75  # Separation onset at high AoA (13.87°)

    # Wall Cp distribution
    x = np.linspace(0, 1, 250)
    # Upper surface: strong suction peak at LE, then adverse PG
    Cp_upper = np.where(
        x < 0.05,
        -6.0 * (1 - x / 0.05),  # Suction peak ≈ -6.0
        np.where(x < x_sep,
                 -6.0 * np.exp(-4 * x) + 0.3 * x,  # Pressure recovery
                 -0.4 + 0.3 * (x - x_sep))  # Plateau in separated region
    )
    Cp_lower = 0.3 * np.exp(-3 * x)

    # Cf on upper surface
    Cf = np.where(x < x_sep,
         0.004 * (1 - 0.8 * x / x_sep),  # Decreasing Cf toward separation
         np.where(x < 0.95,
                  -0.0005 * np.sin(np.pi * (x - x_sep) / (1 - x_sep)),
                  0.0))

    data.wall_data = pd.DataFrame({
        "x_c": x, "Cp_upper": Cp_upper, "Cp_lower": Cp_lower, "Cf": Cf,
    })

    # Velocity profiles at key stations (boundary layer on upper surface)
    for xc in [0.40, 0.60, 0.75, 0.85, 0.95, 1.00]:
        y = np.linspace(0, 0.10, 80)  # y in chord units

        if xc < x_sep:
            # Attached adverse PG boundary layer — thickening
            delta = 0.01 + 0.05 * xc
            U = np.tanh(y / delta * 3)
        else:
            # Separated: reverse flow near wall
            frac = (xc - x_sep) / (1 - x_sep)
            reverse = 0.08 * np.sin(np.pi * frac)
            delta = 0.04 + 0.06 * frac
            U = -reverse * np.exp(-y / 0.005) + np.tanh(y / delta * 2.5)

        uu = 0.015 * np.exp(-(y - 0.02)**2 / 0.001)

        data.velocity_profiles[xc] = pd.DataFrame({
            "y_c": y, "U_Uinf": U, "uu_Uinf2": uu,
        })

    data.separation_metrics = {
        "x_sep_xc": x_sep,
        "alpha_deg": 13.87,
        "Cp_suction_peak": -6.0,
    }
    data.uncertainty = {"U": 0.02, "Cp": 0.02, "Cf": 0.10}
    return data


# ---- NASA Juncture Flow ----
def _gen_juncture_flow(name: str, cfg: BenchmarkCase) -> ExperimentalData:
    """
    Representative data for NASA Juncture Flow experiment.

    Based on Rumsey et al. (2018), AIAA Paper 2018-3319, and
    Rumsey et al. (2020), AIAA J. 58 (QCR2020).
    Wing-body junction with horseshoe vortex and corner separation bubble.
    Re = 2.4×10⁶ based on wing mean aerodynamic chord.
    """
    data = ExperimentalData(
        case_name=name,
        description=cfg.description,
        data_source="Representative (Rumsey et al. 2018, AIAA-2018-3319)",
    )

    # Surface Cp along wing-body juncture line (spanwise cut at junction)
    x = np.linspace(0, 1, 200)  # x/c along wing chord at junction

    # Leading-edge suction, adverse PG, corner separation
    x_sep_corner = 0.70  # Corner separation onset
    x_reat_corner = 0.92  # Corner reattachment

    Cp_junction = np.where(
        x < 0.05,
        -1.5 * (1 - x / 0.05),  # LE suction peak
        np.where(x < x_sep_corner,
                 -1.5 * np.exp(-3 * x) + 0.2,  # Recovery
        np.where(x < x_reat_corner,
                 -0.15 - 0.05 * np.sin(np.pi * (x - x_sep_corner) /
                                        (x_reat_corner - x_sep_corner)),
                 -0.1 * np.exp(-5 * (x - x_reat_corner))))
    )

    # Cf along junction: drops to zero/negative in corner bubble
    Cf_junction = np.where(
        x < x_sep_corner,
        0.003 * (1 - 0.6 * x),
        np.where(x < x_reat_corner,
                 -0.0003 * np.sin(np.pi * (x - x_sep_corner) /
                                   (x_reat_corner - x_sep_corner)),
                 0.002 * (1 - np.exp(-10 * (x - x_reat_corner))))
    )

    data.wall_data = pd.DataFrame({
        "x_c": x, "Cp_junction": Cp_junction, "Cf_junction": Cf_junction,
    })

    # Velocity profiles at stations along the juncture
    for xc in [0.30, 0.50, 0.70, 0.80, 0.90, 1.00]:
        # y = distance from fuselage wall (spanwise), z = wall-normal
        z = np.linspace(0, 0.08, 60)

        if xc < x_sep_corner:
            U = np.tanh(z / 0.008)
            # Horseshoe vortex: spanwise velocity component
            W = 0.05 * np.sin(np.pi * z / 0.04) * np.exp(-z / 0.02)
        elif xc < x_reat_corner:
            frac = (xc - x_sep_corner) / (x_reat_corner - x_sep_corner)
            reverse = 0.06 * np.sin(np.pi * frac)
            U = -reverse * np.exp(-z / 0.002) + np.tanh(z / 0.010)
            W = 0.08 * np.sin(np.pi * z / 0.03) * np.exp(-z / 0.015)
        else:
            U = 0.95 * np.tanh(z / 0.009)
            W = 0.02 * np.exp(-z / 0.02)

        data.velocity_profiles[xc] = pd.DataFrame({
            "z": z, "U_Uinf": U, "W_Uinf": W,
        })

    data.separation_metrics = {
        "x_sep_corner_xc": x_sep_corner,
        "x_reat_corner_xc": x_reat_corner,
        "bubble_length_xc": x_reat_corner - x_sep_corner,
        "horseshoe_vortex": True,
    }
    data.uncertainty = {"U": 0.02, "W": 0.03, "Cp": 0.01, "Cf": 0.15}
    return data


# ---- Axisymmetric SWBLI ----
def _gen_axi_swbli(name: str, cfg: BenchmarkCase) -> ExperimentalData:
    """
    Representative data for axisymmetric shock-wave/boundary-layer interaction.

    NASA 40% Challenge Case #5. M = 2.85, Re ≈ 10⁶.
    Incident oblique shock impinging on turbulent boundary layer.
    """
    data = ExperimentalData(
        case_name=name,
        description=cfg.description,
        data_source="Representative (NASA 40% Challenge Case #5, M=2.85)",
    )

    M_inf = 2.85
    x_imp = 0.0    # Shock impingement (normalized to x=0)
    x_sep = -0.02  # Separation upstream of impingement
    x_reat = 0.04  # Reattachment downstream

    # Wall Cp: upstream constant → rise at interaction → peak → recovery
    x = np.linspace(-0.15, 0.15, 300)
    Cp_undisturbed = 0.0
    # Inviscid shock Cp rise (oblique shock theory)
    Cp_rise = 0.35  # Typical for M=2.85 interaction
    Cp = np.where(
        x < x_sep,
        Cp_undisturbed,
        np.where(x < x_imp,
                 Cp_rise * 0.5 * (1 + np.tanh(50 * (x - x_sep))),
                 np.where(x < x_reat,
                          Cp_rise * (0.8 + 0.2 * np.sin(np.pi * (x - x_imp) /
                                                         (x_reat - x_imp))),
                          Cp_rise * 1.0 * np.exp(-5 * (x - x_reat))))
    )

    # Cf: drops through interaction, negative in separation bubble
    Cf = np.where(
        x < x_sep,
        0.0015,  # Supersonic TBL Cf
        np.where(x < x_reat,
                 -0.0004 * np.sin(np.pi * (x - x_sep) / (x_reat - x_sep)),
                 0.0012 * (1 - np.exp(-20 * (x - x_reat))))
    )

    data.wall_data = pd.DataFrame({"x_norm": x, "Cp": Cp, "Cf": Cf})

    # Velocity profiles across the interaction
    for x_loc in [-0.10, -0.02, 0.0, 0.02, 0.04, 0.08]:
        y = np.linspace(0, 0.02, 80)  # BL is thin at supersonic speeds

        if x_loc < x_sep:
            # Undisturbed supersonic TBL
            U = np.tanh(y / 0.003)
        elif x_loc < x_reat:
            # Interaction region: deceleration, possible reverse
            frac = (x_loc - x_sep) / (x_reat - x_sep)
            decel = 0.3 * np.sin(np.pi * frac)
            U = (1 - decel) * np.tanh(y / 0.004) - 0.05 * frac * np.exp(-y / 0.001)
        else:
            # Recovery
            U = 0.85 * np.tanh(y / 0.005)

        data.velocity_profiles[x_loc] = pd.DataFrame({
            "y": y, "U_Uinf": U,
        })

    data.separation_metrics = {
        "x_sep": x_sep,
        "x_reat": x_reat,
        "bubble_length": x_reat - x_sep,
        "mach_number": M_inf,
        "shock_impingement": x_imp,
    }
    data.uncertainty = {"U": 0.03, "Cp": 0.02, "Cf": 0.20}
    return data


# ---- Obi Diffuser ----
def _gen_obi_diffuser(name: str, cfg: BenchmarkCase) -> ExperimentalData:
    """
    Representative data for Obi asymmetric diffuser.

    Based on Obi et al. (1993), ERCOFTAC SIG15 Workshop #8.
    Re = 20,000, 10° opening angle, expansion ratio 4.7.
    Separation on inclined wall, reattachment downstream.
    """
    data = ExperimentalData(
        case_name=name,
        description=cfg.description,
        data_source="Representative (Obi et al. 1993, ERCOFTAC SIG15 #8)",
    )

    U_b = cfg.reference_velocity if cfg.reference_velocity > 0 else 1.0
    # Diffuser geometry
    x_inlet = 0.0
    x_sep = 5.0     # Separation at x/H ≈ 5 (on inclined wall)
    x_reat = 22.0   # Reattachment at x/H ≈ 22
    x_outlet = 30.0

    # Wall Cp: gradually increasing through diffuser
    x = np.linspace(0, x_outlet, 250)
    Cp_ideal = 1 - (1 / (1 + 0.04 * x))**2  # Ideal diffuser recovery
    Cp = np.where(
        x < x_sep,
        0.8 * Cp_ideal,  # Slightly below ideal before separation
        np.where(x < x_reat,
                 Cp_ideal[np.argmin(np.abs(x - x_sep))]  # Cp plateaus
                 + 0.02 * (x - x_sep) / (x_reat - x_sep),
                 Cp_ideal * 0.75)  # Loss in recovery after reattachment
    )

    # Cf on inclined wall
    Cf = np.where(
        x < x_sep,
        0.005 * (1 - 0.8 * x / x_sep),
        np.where(x < x_reat,
                 -0.001 * np.sin(np.pi * (x - x_sep) / (x_reat - x_sep)),
                 0.003 * (1 - np.exp(-0.5 * (x - x_reat))))
    )

    data.wall_data = pd.DataFrame({"x_H": x, "Cp": Cp, "Cf": Cf})

    # Velocity profiles across diffuser section
    for xH in [2.0, 5.0, 10.0, 15.0, 22.0, 28.0]:
        # Channel half-height expands with x
        h_local = 1.0 + 0.087 * xH  # tan(10° / 2) ≈ 0.087
        y = np.linspace(0, h_local, 80)

        if xH < x_sep:
            # Fully attached channel flow (Poiseuille-like)
            U = U_b * 1.5 * (y / h_local) * (2 - y / h_local)
        elif xH < x_reat:
            # Separation on inclined wall: asymmetric profile
            frac = (xH - x_sep) / (x_reat - x_sep)
            reverse = 0.15 * np.sin(np.pi * frac)
            U = U_b * (
                -reverse * np.exp(-5 * y / h_local)
                + 1.2 * (y / h_local) * (2 - y / h_local)
            )
        else:
            # Recovery
            U = U_b * 0.8 * (y / h_local) * (2 - y / h_local)

        data.velocity_profiles[xH] = pd.DataFrame({
            "y_H": y / h_local, "U_Ub": U / U_b,
        })

    data.separation_metrics = {
        "x_sep_xH": x_sep,
        "x_reat_xH": x_reat,
        "bubble_length_xH": x_reat - x_sep,
        "opening_angle_deg": 10.0,
        "expansion_ratio": 4.7,
    }
    data.uncertainty = {"U": 0.02, "Cp": 0.02, "Cf": 0.15}
    return data


# ---- NASA 3D Gaussian Speed Bump ----
def _gen_gaussian_bump(name: str, cfg: BenchmarkCase) -> ExperimentalData:
    """
    Representative data for the NASA 3D Gaussian Speed Bump.

    Based on Iyer & Malik (2020), NASA TM 2020-220469 (WMLES reference)
    and Gray et al. (2021), AIAA J. 59(10) (RANS assessment).
    M = 0.176, Re_L = 2×10⁶.
    h(x,z) = h₀·exp(-(x/x₀)²-(z/z₀)²), h₀=0.085L, x₀=0.195L, z₀=0.06L.
    """
    data = ExperimentalData(
        case_name=name,
        description=cfg.description,
        data_source="Representative (Iyer & Malik 2020 WMLES, NASA TMR)",
    )

    L = 1.0  # Reference length
    H0, X0, Z0 = 0.085, 0.195, 0.06
    U_inf = cfg.reference_velocity if cfg.reference_velocity > 0 else 60.0

    # Centerline wall data (WMLES reference baseline)
    x = np.linspace(-0.5, 2.5, 400)  # x/L
    h = H0 * np.exp(-(x / X0)**2)  # Centerline bump height

    # WMLES reference separation
    x_sep = 0.75
    x_reat = 1.35

    # Cp distribution
    Cp = np.where(
        x < 0.0,
        -0.3 * np.exp(-((x + 0.1) / 0.15)**2),  # Mild suction upstream
        np.where(
            x < x_sep,
            -0.45 * np.exp(-((x - 0.3) / 0.25)**2),  # Suction peak over bump
            np.where(
                x < x_reat,
                -0.12 + 0.01 * (x - x_sep),  # Separation plateau
                -0.05 * np.exp(-2.0 * (x - x_reat)),  # Recovery
            ),
        ),
    )

    # Cf distribution
    Cf = np.where(
        x < x_sep,
        0.003 * (1 + 0.5 * h / H0),  # Attached, enhanced over bump
        np.where(
            x < x_reat,
            -0.0006 * np.sin(np.pi * (x - x_sep) / (x_reat - x_sep)),
            0.0025 * (1 - np.exp(-3.0 * (x - x_reat))),  # Recovery
        ),
    )

    data.wall_data = pd.DataFrame({
        "x_L": x, "Cp": Cp, "Cf": Cf,
        "bump_surface": h,
    })

    # Velocity profiles at 6 stations
    for xL in [0.6, 0.8, 1.0, 1.2, 1.4, 1.8]:
        y = np.linspace(0, 0.3, 80)  # y/L
        h_local = H0 * np.exp(-(xL / X0)**2)

        if xL < x_sep:
            U = U_inf * np.tanh((y - h_local) / 0.02)
            U = np.maximum(U, 0)
            V = U_inf * 0.03 * h_local / H0 * np.exp(-y / 0.1)
        elif xL < x_reat:
            frac = (xL - x_sep) / (x_reat - x_sep)
            reverse = 0.12 * np.sin(np.pi * frac)
            U = U_inf * (-reverse * np.exp(-y / 0.008) + np.tanh(y / 0.03))
            V = U_inf * (-0.02 * np.sin(np.pi * frac) * np.exp(-y / 0.05))
        else:
            recovery = min((xL - x_reat) / 0.5, 1.0)
            U = U_inf * (0.88 + 0.12 * recovery) * np.tanh(y / 0.04)
            V = U_inf * 0.005 * np.exp(-y / 0.1)

        W = np.zeros_like(y)  # Centerline symmetry
        uu = 0.012 * U_inf**2 * np.exp(-((y - 0.02)**2) / 0.002)
        vv = 0.006 * U_inf**2 * np.exp(-((y - 0.02)**2) / 0.003)
        uv = -0.004 * U_inf**2 * np.exp(-((y - 0.015)**2) / 0.002)

        data.velocity_profiles[xL] = pd.DataFrame({
            "y_L": y, "U": U, "V": V, "W": W,
            "uu": uu, "vv": vv, "uv": uv,
        })

    data.separation_metrics = {
        # WMLES reference
        "x_sep_xL_wmles": x_sep,
        "x_reat_xL_wmles": x_reat,
        "bubble_length_xL_wmles": x_reat - x_sep,
        # SA baseline (known failure)
        "x_sep_xL_sa": 0.82,
        "x_reat_xL_sa": 1.15,
        "bubble_length_xL_sa": 0.33,
        # SA-RC corrected
        "x_sep_xL_sa_rc": 0.78,
        "x_reat_xL_sa_rc": 1.25,
        "bubble_length_xL_sa_rc": 0.47,
        # Geometry
        "bump_height_L": H0,
        "bump_x0_L": X0,
        "bump_z0_L": Z0,
        "mach_number": 0.176,
        "reynolds_number": 2_000_000,
    }
    data.uncertainty = {"U": 0.02, "V": 0.03, "Cp": 0.01, "Cf": 0.10}
    return data


# ---- BeVERLI Hill ----
def _gen_beverli_hill(name: str, cfg: BenchmarkCase) -> ExperimentalData:
    """
    Representative data for the BeVERLI Hill 3D smooth-body separation.

    Based on Lowe et al. (2022), AIAA Paper 2022-0329, and
    Rahmani et al. (2023), Exp. Fluids 64.
    Virginia Tech / NASA Langley collaboration.
    Superelliptic 3D hill, H = 0.1869 m, Re_H = 250k–650k.
    Multiple yaw angles: 0°, 30°, 45°.
    """
    data = ExperimentalData(
        case_name=name,
        description=cfg.description,
        data_source="Representative (Lowe et al. 2022, VT-NASA BeVERLI)",
    )

    H = 0.1869  # Hill height [m]
    U_inf = cfg.reference_velocity if cfg.reference_velocity > 0 else 20.0

    # --- Centerline wall data (0° yaw, Re = 250k) ---
    x = np.linspace(-5.0, 10.0, 400)  # x/H

    # Hill profile (Hermite smoothstep)
    x_hill = np.clip(x, -2.5, 2.5)
    t = np.where(
        np.abs(x_hill) <= 0.25,
        np.ones_like(x_hill),
        np.where(
            np.abs(x_hill) <= 2.5,
            _hermite_smoothstep((2.5 - np.abs(x_hill)) / 2.25),
            np.zeros_like(x_hill),
        ),
    )
    y_hill = np.where(np.abs(x) <= 2.5, t, 0.0)

    # Cp: acceleration (windward) → suction peak → adverse PG → separation → recovery
    Cp = np.where(
        x < -2.5,
        0.0,
        np.where(
            x < 0.0,
            -0.6 * np.sin(np.pi * (x + 2.5) / 2.5),  # Windward acceleration
            np.where(
                x < 1.2,
                -0.6 * np.exp(-2 * x) + 0.1,  # Leeward deceleration
                np.where(
                    x < 4.5,
                    -0.15 + 0.02 * (x - 1.2),  # Separation plateau
                    -0.08 * np.exp(-0.5 * (x - 4.5)),  # Recovery
                ),
            ),
        ),
    )

    # Cf: attached → zero → negative (sep) → recovery
    x_sep_0 = 1.2  # Separation onset at 0° yaw
    x_reat_0 = 4.5  # Reattachment
    Cf = np.where(
        x < x_sep_0,
        0.003 * (1 + 0.3 * np.maximum(y_hill, 0)),  # Attached flow over hill
        np.where(
            x < x_reat_0,
            -0.0008 * np.sin(np.pi * (x - x_sep_0) / (x_reat_0 - x_sep_0)),
            0.002 * (1 - np.exp(-1.5 * (x - x_reat_0))),
        ),
    )

    data.wall_data = pd.DataFrame({
        "x_H": x, "Cp": Cp, "Cf": Cf,
        "hill_surface": y_hill,
        "yaw_deg": 0,
    })

    # --- Velocity profiles at 6 PIV stations ---
    for xH in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
        y = np.linspace(0, 2.0 * H, 80)  # Wall-normal [m]
        y_H = y / H

        if xH < x_sep_0:
            # Attached flow over windward/crest: accelerated BL
            U = U_inf * np.tanh(y_H / 0.15)
            V = U_inf * 0.05 * y_hill[np.argmin(np.abs(x - xH))] * np.exp(-y_H / 0.3)
            W = np.zeros_like(y)  # Symmetric at 0° yaw
        elif xH < x_reat_0:
            # Leeward separation: reverse flow near wall
            frac = (xH - x_sep_0) / (x_reat_0 - x_sep_0)
            reverse = 0.15 * np.sin(np.pi * frac)
            U = U_inf * (-reverse * np.exp(-y_H / 0.05) + np.tanh(y_H / 0.2))
            V = U_inf * (-0.03 * np.sin(np.pi * frac) * np.exp(-y_H / 0.1))
            W = np.zeros_like(y)
        else:
            # Wake recovery
            recovery = min((xH - x_reat_0) / 2.0, 1.0)
            U = U_inf * (0.85 + 0.15 * recovery) * np.tanh(y_H / 0.25)
            V = U_inf * 0.01 * np.exp(-y_H / 0.4)
            W = np.zeros_like(y)

        # Reynolds stresses: peak in shear layer
        uu = 0.015 * U_inf**2 * np.exp(-((y_H - 0.1)**2) / 0.01)
        vv = 0.008 * U_inf**2 * np.exp(-((y_H - 0.1)**2) / 0.012)
        ww = 0.006 * U_inf**2 * np.exp(-((y_H - 0.12)**2) / 0.015)
        uv = -0.005 * U_inf**2 * np.exp(-((y_H - 0.09)**2) / 0.008)

        data.velocity_profiles[xH] = pd.DataFrame({
            "y": y, "y_H": y_H,
            "U": U, "V": V, "W": W,
            "uu": uu, "vv": vv, "ww": ww, "uv": uv,
        })

    data.separation_metrics = {
        "x_sep_xH": x_sep_0,
        "x_reat_xH": x_reat_0,
        "bubble_length_xH": x_reat_0 - x_sep_0,
        "yaw_deg": 0,
        "reynolds_number": 250_000,
        "hill_height_m": H,
        "yaw_0_symmetric": True,
        "yaw_45_asymmetric_wake": True,
    }
    data.uncertainty = {"U": 0.02, "V": 0.03, "W": 0.03,
                        "uu": 0.10, "Cp": 0.01, "Cf": 0.12}
    return data


def _hermite_smoothstep(t: np.ndarray) -> np.ndarray:
    """Hermite smoothstep: 10*t^3 - 15*t^4 + 6*t^5, clamped to [0, 1]."""
    t = np.clip(t, 0, 1)
    return 10 * t**3 - 15 * t**4 + 6 * t**5


# ---- Generic Fallback ----
def _gen_generic(name: str, cfg: BenchmarkCase) -> ExperimentalData:
    """Minimal placeholder for cases without a specific generator yet."""
    return ExperimentalData(
        case_name=name,
        description=cfg.description,
        data_source=f"{cfg.data_source} (placeholder — replace with real data)",
        separation_metrics=cfg.separation_metrics,
    )


# =============================================================================
# CLI
# =============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load benchmark case reference data")
    parser.add_argument("case", nargs="?", default=None,
                        help="Case name from config.BENCHMARK_CASES")
    parser.add_argument("--list", action="store_true", help="List all cases")
    args = parser.parse_args()

    if args.list or args.case is None:
        print("Available benchmark cases:")
        for key, case in BENCHMARK_CASES.items():
            print(f"  {key:<30} {case.tier.name:<20} {case.data_source}")
    else:
        exp_data = load_case(args.case)
        print(exp_data.summary())
