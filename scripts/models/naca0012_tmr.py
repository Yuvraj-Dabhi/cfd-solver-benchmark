"""
NACA 0012 TMR Numerics — Corrected Airfoil & Grid Reference
=============================================================
Implements the corrected NACA 0012 definition from the NASA TMR 
Numerical Analysis page, along with grid family specifications
and reference result data.

Source: https://turbmodels.larc.nasa.gov/naca0012numerics_val.html
Grids:  https://turbmodels.larc.nasa.gov/naca0012numerics_grids.html

Primary Reference:
  Rumsey (2016), AIAA J. 54(9), pp.2563-2588, DOI:10.2514/1.J054555
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Corrected NACA 0012 Airfoil Definition (Sharp TE)
# =============================================================================

# Original NACA 0012 coefficients
NACA0012_ORIGINAL_COEFFS = {
    "a0": 0.2969,
    "a1": -0.1260,
    "a2": -0.3516,
    "a3": 0.2843,
    "a4": -0.1015,
    "half_thickness_scale": 0.6,  # gives 12% max thickness
}

# Corrected (scaled) NACA 0012 for sharp TE closure
# Airfoil is scaled down by factor 1.008930411365 so that
# the sharp TE falls exactly at x = 1.0
NACA0012_SCALE_FACTOR = 1.008930411365

NACA0012_CORRECTED_COEFFS = {
    "a0": 0.298222773,
    "a1": -0.127125232,
    "a2": -0.357907906,
    "a3": 0.291984971,
    "a4": -0.105174606,   # NOTE: was 0.105174696 (typo) before 6/23/2014
    "half_thickness_scale": 0.594689181,
}


def naca0012_original(x: np.ndarray) -> np.ndarray:
    """
    Original NACA 0012 half-thickness (unclosed, blunt TE).
    
    y = 0.6 * [0.2969*sqrt(x) - 0.1260*x - 0.3516*x^2 
               + 0.2843*x^3 - 0.1015*x^4]
    
    Valid for 0 <= x <= 1.0 (blunt TE at x=1).
    Max thickness = 12% at ~30% chord.
    """
    c = NACA0012_ORIGINAL_COEFFS
    return c["half_thickness_scale"] * (
        c["a0"] * np.sqrt(x) + c["a1"] * x + c["a2"] * x**2
        + c["a3"] * x**3 + c["a4"] * x**4
    )


def naca0012_corrected(x: np.ndarray) -> np.ndarray:
    """
    TMR corrected NACA 0012 half-thickness (sharp TE, closed).
    
    y = 0.594689181 * [0.298222773*sqrt(x) - 0.127125232*x 
                       - 0.357907906*x^2 + 0.291984971*x^3 
                       - 0.105174606*x^4]
    
    Valid for 0 <= x <= 1.0 (sharp TE at x=1, y=0).
    Max thickness = 11.894% chord (same shape, just scaled to close).
    
    Source: https://turbmodels.larc.nasa.gov/naca0012numerics_val.html
    Updated 6/23/2014 (corrected typo in a4 coefficient).
    """
    c = NACA0012_CORRECTED_COEFFS
    return c["half_thickness_scale"] * (
        c["a0"] * np.sqrt(x) + c["a1"] * x + c["a2"] * x**2
        + c["a3"] * x**3 + c["a4"] * x**4
    )


def generate_naca0012_coordinates(
    n_points: int = 201,
    corrected: bool = True,
    cosine_spacing: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate NACA 0012 airfoil coordinates (upper + lower surface).
    
    Parameters
    ----------
    n_points : int
        Number of points on each surface (default 201).
    corrected : bool
        If True, use the TMR corrected (sharp TE) definition.
        If False, use the original (blunt TE) definition.
    cosine_spacing : bool
        If True, use cosine spacing (finer at LE and TE).
    
    Returns
    -------
    x_coords : ndarray, shape (2*n_points - 1,)
        x/c coordinates (TE lower -> LE -> TE upper).
    y_coords : ndarray, shape (2*n_points - 1,)
        y/c coordinates.
    """
    if cosine_spacing:
        beta = np.linspace(0, np.pi, n_points)
        x = 0.5 * (1.0 - np.cos(beta))
    else:
        x = np.linspace(0, 1, n_points)
    
    thickness_func = naca0012_corrected if corrected else naca0012_original
    yt = thickness_func(x)
    
    # Upper surface: LE to TE
    x_upper = x
    y_upper = yt
    
    # Lower surface: TE to LE (reversed)
    x_lower = x[::-1]
    y_lower = -yt[::-1]
    
    # Combine: lower TE -> LE -> upper TE
    x_coords = np.concatenate([x_lower, x_upper[1:]])
    y_coords = np.concatenate([y_lower, y_upper[1:]])
    
    return x_coords, y_coords


def verify_airfoil_closure(corrected: bool = True) -> Dict[str, float]:
    """
    Verify that the airfoil closes at the trailing edge.
    
    Returns dict with TE y-value and max thickness info.
    """
    x = np.array([0.0, 0.3, 1.0])
    
    func = naca0012_corrected if corrected else naca0012_original
    y = func(x)
    
    # Find max thickness location
    x_fine = np.linspace(0, 1, 10001)
    y_fine = func(x_fine)
    max_idx = np.argmax(y_fine)
    
    return {
        "variant": "corrected (sharp TE)" if corrected else "original (blunt TE)",
        "y_at_TE": float(y[2]),
        "TE_closed": abs(y[2]) < 1e-10 if corrected else False,
        "max_thickness_pct": float(2 * y_fine[max_idx] * 100),
        "max_thickness_x": float(x_fine[max_idx]),
    }


# =============================================================================
# TMR Grid Family Specifications
# =============================================================================

@dataclass
class TMRGridLevel:
    """Single grid refinement level."""
    level: int              # 1 (finest) to 7 (coarsest)
    ni: int                 # Streamwise points
    nj: int                 # Normal points
    n_airfoil: int          # Points on airfoil surface
    n_wake: int             # Points in wake (TE to outflow)
    total_cells: int        # ni * nj (approximate)
    h_ratio: float          # h/h_finest


@dataclass
class TMRGridFamily:
    """TMR NACA 0012 grid family specification."""
    family_id: str          # "I", "II", or "III"
    te_spacing: float       # Trailing edge spacing / chord
    te_relative: str        # Relative to Family I
    le_spacing: float       # Leading edge spacing / chord
    wall_spacing: float     # Min wall-normal spacing / chord
    stretch_rate: float     # Average near-wall stretching
    farfield_extent: float  # Farfield distance / chord
    n_levels: int           # Number of available levels
    recommended: bool       # TMR recommended for convergence?
    levels: List[TMRGridLevel]
    base_url_2d: str        # Download URL template
    base_url_3d: str = ""
    base_url_cgns: str = ""


# Grid levels (same for all families)
_GRID_LEVELS = [
    TMRGridLevel(1, 7169, 2049, 4097, 1537, 7169*2049, 1.0),
    TMRGridLevel(2, 3585, 1025, 2049, 769,  3585*1025, 2.0),
    TMRGridLevel(3, 1793, 513,  1025, 385,  1793*513,  4.0),
    TMRGridLevel(4, 897,  257,  513,  193,  897*257,   8.0),
    TMRGridLevel(5, 449,  129,  257,  97,   449*129,   16.0),
    TMRGridLevel(6, 225,  65,   129,  49,   225*65,    32.0),
    TMRGridLevel(7, 113,  33,   65,   25,   113*33,    64.0),
]

TMR_GRID_BASE = "https://turbmodels.larc.nasa.gov/NACA0012numerics_grids"

TMR_GRID_FAMILIES = {
    "I": TMRGridFamily(
        family_id="I",
        te_spacing=0.000125,
        te_relative="1x (baseline)",
        le_spacing=0.0000125,
        wall_spacing=1e-7,
        stretch_rate=1.02,
        farfield_extent=500.0,
        n_levels=7,
        recommended=False,
        levels=_GRID_LEVELS,
        base_url_2d=f"{TMR_GRID_BASE}/n0012familyI.{{level}}.p2dfmt.gz",
        base_url_3d=f"{TMR_GRID_BASE}/n0012familyI.{{level}}.p3dfmt.gz",
        base_url_cgns=f"{TMR_GRID_BASE}/n0012familyI.{{level}}.hex.cgns.gz",
    ),
    "II": TMRGridFamily(
        family_id="II",
        te_spacing=0.0000125,
        te_relative="10x finer than Family I",
        le_spacing=0.0000125,
        wall_spacing=1e-7,
        stretch_rate=1.02,
        farfield_extent=500.0,
        n_levels=7,
        recommended=True,
        levels=_GRID_LEVELS,
        base_url_2d=f"{TMR_GRID_BASE}/n0012familyII.{{level}}.p2dfmt.gz",
        base_url_3d=f"{TMR_GRID_BASE}/n0012familyII.{{level}}.p3dfmt.gz",
        base_url_cgns=f"{TMR_GRID_BASE}/n0012familyII.{{level}}.hex.cgns.gz",
    ),
    "III": TMRGridFamily(
        family_id="III",
        te_spacing=0.0000375,
        te_relative="3.33x finer than Family I",
        le_spacing=0.0000125,
        wall_spacing=1e-7,
        stretch_rate=1.02,
        farfield_extent=500.0,
        n_levels=1,  # Only finest level provided
        recommended=False,
        levels=[_GRID_LEVELS[0]],  # Only level 1
        base_url_2d=f"{TMR_GRID_BASE}/n0012familyIII.{{level}}.p2dfmt.gz",
    ),
}


def get_grid_download_url(
    family: str = "II", 
    level: int = 5, 
    format: str = "2d",
) -> str:
    """
    Get the download URL for a specific TMR NACA 0012 grid.
    
    Parameters
    ----------
    family : str
        Grid family: "I", "II", or "III".
    level : int
        Grid refinement level: 1 (finest) to 7 (coarsest).
    format : str
        "2d" for 2D PLOT3D, "3d" for 3D PLOT3D, "cgns" for unstructured.
    
    Returns
    -------
    url : str
        Direct download URL (gzipped file).
    """
    fam = TMR_GRID_FAMILIES.get(family)
    if fam is None:
        raise ValueError(f"Unknown family '{family}'. Use 'I', 'II', or 'III'.")
    if level < 1 or level > fam.n_levels:
        raise ValueError(f"Family {family} has levels 1-{fam.n_levels}, got {level}.")
    
    if format == "2d":
        return fam.base_url_2d.format(level=level)
    elif format == "3d":
        if not fam.base_url_3d:
            raise ValueError(f"No 3D grids for Family {family}.")
        return fam.base_url_3d.format(level=level)
    elif format == "cgns":
        if not fam.base_url_cgns:
            raise ValueError(f"No CGNS grids for Family {family}.")
        return fam.base_url_cgns.format(level=level)
    else:
        raise ValueError(f"Unknown format '{format}'. Use '2d', '3d', or 'cgns'.")


# =============================================================================
# Grid-Converged Reference Results
# =============================================================================

@dataclass
class NACA0012ReferenceResult:
    """Grid-converged SA reference results from TMR."""
    alpha_deg: float
    mach: float = 0.15
    reynolds: float = 6e6
    tref_rankine: float = 540.0
    
    # Force/moment coefficients (grid-converged, with PV correction)
    CL: Optional[float] = None
    CD: Optional[float] = None
    CDp: Optional[float] = None
    CDv: Optional[float] = None
    CM: Optional[float] = None
    
    # Uncertainty range
    CL_range: Optional[Tuple[float, float]] = None
    CD_range: Optional[Tuple[float, float]] = None
    CM_range: Optional[Tuple[float, float]] = None
    
    # Source
    model: str = "SA"
    grid_family: str = "II"
    codes: str = "FUN3D + CFL3D"
    pv_correction: bool = True


# Grid-converged reference values from TMR (with point vortex correction)
TMR_REFERENCE_RESULTS = {
    10: NACA0012ReferenceResult(
        alpha_deg=10.0,
        CL=1.09125,
        CD=0.012215,
        CDp=0.006015,
        CDv=0.006205,
        CM=0.00678,
        CL_range=(1.0912, 1.0913),
        CD_range=(0.01221, 0.01222),
        CM_range=(0.00677, 0.00679),
    ),
}

# Data file download URLs
TMR_DATA_FILES = {
    "fun3d_forces_withpv": (
        "https://turbmodels.larc.nasa.gov/NACA0012numerics_val/"
        "fun3d_results_sa_withN.dat"
    ),
    "cfl3d_forces_withpv": (
        "https://turbmodels.larc.nasa.gov/NACA0012numerics_val/"
        "cfl3d_results_sa_withN.dat"
    ),
    "fun3d_cp": (
        "https://turbmodels.larc.nasa.gov/NACA0012numerics_val/"
        "fun3d_cp_sa.dat"
    ),
    "cfl3d_cp": (
        "https://turbmodels.larc.nasa.gov/NACA0012numerics_val/"
        "cfl3d_cp_sa.dat"
    ),
    "fun3d_cf": (
        "https://turbmodels.larc.nasa.gov/NACA0012numerics_val/"
        "fun3d_cf_sa.dat"
    ),
    "cfl3d_cf": (
        "https://turbmodels.larc.nasa.gov/NACA0012numerics_val/"
        "cfl3d_cf_sa.dat"
    ),
}


# =============================================================================
# Case Setup Specification
# =============================================================================

NACA0012_CASE_SETUP = {
    "flow_conditions": {
        "mach": 0.15,
        "reynolds_chord": 6e6,
        "alpha_deg": [0, 10, 15],
        "tref_rankine": 540.0,
        "tref_kelvin": 300.0,
        "gamma": 1.4,
    },
    "fluid_properties": {
        "Pr": 0.72,
        "Pr_t": 0.90,
        "viscosity_law": "Sutherland",
        "sutherland_T_ref_K": 273.15,
        "sutherland_mu_ref": 1.716e-5,
        "sutherland_S_K": 110.4,
    },
    "boundary_conditions": {
        "farfield": "inviscid characteristic + point vortex correction",
        "wall": "adiabatic no-slip",
        "SA_inflow": "nu_hat = 3*nu (standard TMR)",
    },
    "turbulence_model": {
        "model": "SA (standard) or SA-neg (recommended)",
        "S_tilde_clipping": "method (c) [ICCFD7-1902] recommended",
        "turbulence_advection": "2nd order recommended (FUN3D default)",
    },
    "grid_recommendation": {
        "family": "II (TE spacing = 0.0000125c)",
        "rationale": (
            "Family I converges to wrong CL/CM. "
            "Family II yields correct grid-converged results."
        ),
        "levels_for_gci": [7, 6, 5, 4],
        "finest_practical": 3,  # 1793 x 513
    },
    "quantities_of_interest": [
        "CL (lift coefficient)",
        "CD (drag coefficient: total, pressure, viscous)",
        "CM (moment coefficient about 0.25c)",
        "Cp(x/c) (surface pressure coefficient)",
        "Cf(x/c) (surface skin friction coefficient)",
    ],
}


# =============================================================================
# CLI
# =============================================================================

def print_summary():
    """Print a summary of the NACA 0012 TMR numerics case."""
    print("=" * 70)
    print("NACA 0012 TMR NUMERICS CASE REFERENCE")
    print("=" * 70)
    
    # Airfoil verification
    print("\n--- Airfoil Closure Verification ---")
    for variant in [True, False]:
        info = verify_airfoil_closure(variant)
        label = info["variant"]
        print(f"  {label}:")
        print(f"    y at TE:          {info['y_at_TE']:.2e}")
        print(f"    TE closed:        {info['TE_closed']}")
        print(f"    Max thickness:    {info['max_thickness_pct']:.3f}%")
        print(f"    At x/c:           {info['max_thickness_x']:.4f}")
    
    # Flow conditions
    setup = NACA0012_CASE_SETUP
    fc = setup["flow_conditions"]
    print(f"\n--- Flow Conditions ---")
    print(f"  Mach:    {fc['mach']}")
    print(f"  Re/c:    {fc['reynolds_chord']:.0e}")
    print(f"  Alpha:   {fc['alpha_deg']} deg")
    print(f"  T_ref:   {fc['tref_rankine']} R ({fc['tref_kelvin']} K)")
    print(f"  Gamma:   {fc['gamma']}")
    
    # Grid families
    print(f"\n--- TMR Grid Families ---")
    for fid, fam in TMR_GRID_FAMILIES.items():
        tag = " ** RECOMMENDED **" if fam.recommended else ""
        print(f"  Family {fid}: TE={fam.te_spacing}c "
              f"({fam.te_relative}){tag}")
        print(f"    Levels: {fam.n_levels}, LE={fam.le_spacing}c, "
              f"wall={fam.wall_spacing}c")
    
    # Grid level table
    print(f"\n--- Grid Levels (all families) ---")
    print(f"  {'Level':<8} {'Size':<16} {'Airfoil pts':<14} {'Cells':<12} {'h/h_f'}")
    for gl in _GRID_LEVELS:
        print(f"  {gl.level:<8} {gl.ni}x{gl.nj:<10} {gl.n_airfoil:<14} "
              f"{gl.total_cells:<12,} {gl.h_ratio}")
    
    # Reference results
    print(f"\n--- Grid-Converged SA Results (alpha=10, with PV) ---")
    ref = TMR_REFERENCE_RESULTS[10]
    print(f"  CL  = {ref.CL_range[0]} - {ref.CL_range[1]}")
    print(f"  CD  = {ref.CD_range[0]} - {ref.CD_range[1]}")
    print(f"  CDp = {ref.CDp}")
    print(f"  CDv = {ref.CDv}")
    print(f"  CM  = {ref.CM_range[0]} - {ref.CM_range[1]}")
    print(f"  Codes: {ref.codes}")
    
    # Download URLs
    print(f"\n--- Example Download URLs (Family II) ---")
    for level in [5, 4, 3]:
        url = get_grid_download_url("II", level, "2d")
        gl = _GRID_LEVELS[level - 1]
        print(f"  Level {level} ({gl.ni}x{gl.nj}): {url}")
    
    print("=" * 70)


if __name__ == "__main__":
    print_summary()
