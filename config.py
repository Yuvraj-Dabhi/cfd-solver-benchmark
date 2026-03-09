"""
CFD Solver Benchmark for Flow Separation Prediction
====================================================
Central configuration module defining all benchmark cases, turbulence models,
solver parameters, numerical schemes, data sources, and validation criteria.

Synthesizes findings from six research files plus citation deep-dives (NASA TMR,
ERCOFTAC, DPW/HLPW, WMLES workshops, Nature Comms DRL, ASME V&V 20-2009).
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any
from pathlib import Path
import numpy as np

# =============================================================================
# Project Paths
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.resolve()
DATA_DIR = PROJECT_ROOT / "experimental_data"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
RESULTS_DIR = PROJECT_ROOT / "results"
DOCS_DIR = PROJECT_ROOT / "docs"
TESTS_DIR = PROJECT_ROOT / "tests"


# =============================================================================
# Enums & Data Classes
# =============================================================================
class CaseTier(Enum):
    """Benchmark case complexity tiers."""
    T0_VERIFICATION = 0    # Code verification (analytical solutions)
    T1_CANONICAL = 1       # Canonical 2D separated flows
    T2_INTERMEDIATE = 2    # Intermediate complexity
    T3_COMPLEX_3D = 3      # Complex 3D flows


class SeparationCategory(Enum):
    """Flow separation mechanism categories."""
    GEOMETRIC = "geometric"            # Step, ramp
    SMOOTH_BODY_2D = "smooth_body_2d"  # Pressure-gradient induced
    SMOOTH_BODY_3D = "smooth_body_3d"  # 3D smooth-body
    CURVATURE = "curvature"            # Curvature-driven
    SHOCK_INDUCED = "shock_induced"    # SBLI
    CORNER_JUNCTURE = "corner_juncture"
    VERIFICATION = "verification"      # Analytical solutions available
    DIFFUSER = "diffuser"              # Diffuser-type separation
    FREE_SHEAR = "free_shear"          # Free-shear (jets, wakes, mixing layers)


class ModelType(Enum):
    """Turbulence modeling approach."""
    RANS_1EQ = "1-equation RANS"
    RANS_2EQ = "2-equation RANS"
    RANS_4EQ = "4-equation RANS"
    RSM = "Reynolds Stress Model"
    EARSM = "Explicit Algebraic RSM"
    TRANSITION = "Transition model"
    HYBRID = "Hybrid RANS-LES"
    WMLES = "Wall-Modeled LES"


@dataclass
class ExperimentalStation:
    """Profile extraction station from experiment."""
    location: float           # x/H, x/c, or x/L
    location_label: str       # e.g., "x/H", "x/c"
    quantities: List[str]     # Available: U, V, uu, vv, uv, Cp, Cf


@dataclass
class BenchmarkCase:
    """Complete benchmark case specification."""
    name: str
    tier: CaseTier
    category: SeparationCategory
    description: str

    # Flow conditions
    reynolds_number: float
    mach_number: float = 0.0
    reference_velocity: float = 0.0
    temperature: float = 300.0

    # Geometry
    reference_length: float = 1.0
    reference_length_name: str = "H"
    domain_description: str = ""

    # Mesh requirements
    mesh_levels: Dict[str, int] = field(default_factory=dict)
    yplus_target: float = 1.0
    delta_x_plus: Dict[str, str] = field(default_factory=dict)

    # Experimental data
    data_source: str = ""
    data_url: str = ""
    profile_stations: List[ExperimentalStation] = field(default_factory=list)
    separation_metrics: Dict[str, Any] = field(default_factory=dict)

    # Reference papers (DOI or citation)
    reference_papers: List[str] = field(default_factory=list)

    # Expected RANS errors
    rans_error_baseline: Dict[str, str] = field(default_factory=dict)

    # Analytical solutions (for T0)
    analytical_solutions: Dict[str, str] = field(default_factory=dict)


@dataclass
class TurbulenceModel:
    """Turbulence model specification."""
    name: str
    abbreviation: str
    model_type: ModelType
    openfoam_name: str
    su2_name: str = ""
    n_equations: int = 0
    description: str = ""
    key_paper: str = ""  # DOI or citation of founding paper
    expected_performance: Dict[str, str] = field(default_factory=dict)
    openfoam_settings: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Benchmark Cases (17 total: T0=1, T1=5, T2=7, T3=4)
# =============================================================================
BENCHMARK_CASES: Dict[str, BenchmarkCase] = {

    # ---- Tier 0: Code Verification ----
    # Case 1: Zero Pressure Gradient Flat Plate 
    # (Re = 1 Million, Analytical geometry, Baseline boundary layer validation)
    "flat_plate": BenchmarkCase(
        name="Zero Pressure Gradient Flat Plate",
        tier=CaseTier.T0_VERIFICATION,
        category=SeparationCategory.VERIFICATION,
        description="Code verification case with analytical Blasius/log-law solutions",
        reynolds_number=1e6,
        reference_velocity=10.0,
        reference_length=10.0,
        reference_length_name="L",
        domain_description="Length=10m, Height=1m, 2D",
        mesh_levels={"coarse": 20_000, "medium": 50_000, "fine": 100_000},
        yplus_target=1.0,
        data_source="Analytical",
        data_url="https://turbmodels.larc.nasa.gov/flatplate.html",
        separation_metrics={},
        rans_error_baseline={"SA": "<1%", "SST": "<1%"},
        analytical_solutions={
            "delta_99": "0.37 * x / Re_x**0.2",
            "Cf_turbulent": "0.059 / Re_x**0.2",
            "log_law": "U+ = (1/0.41) * ln(y+) + 5.0",
            "viscous_sublayer": "U+ = y+",
        },
    ),

    # ---- Tier 1: Canonical 2D Separated Flows ----
    # Case 2: NASA Wall-Mounted Hump 
    # (Re = 936k, Glauert-Goldschmied body geometry, NASA TMR Reference Data)
    "nasa_hump": BenchmarkCase(
        name="NASA Wall-Mounted Hump",
        tier=CaseTier.T1_CANONICAL,
        category=SeparationCategory.SMOOTH_BODY_2D,
        description="Glauert-Goldschmied body; smooth-body separation with extensive PIV data",
        reynolds_number=936_000,
        mach_number=0.1,
        reference_velocity=34.6,
        reference_length=0.420,
        reference_length_name="c",
        domain_description="Inlet x/c=-2.14, Outlet x/c=4.0, Tunnel height 0.9144m",
        mesh_levels={"coarse": 150_000, "medium": 350_000, "fine": 700_000},
        yplus_target=0.5,
        data_source="NASA TMR 2DWMH",
        data_url="https://turbmodels.larc.nasa.gov/nasahump_val.html",
        profile_stations=[
            ExperimentalStation(loc, "x/c", ["U", "V", "uu", "vv", "uv"])
            for loc in [0.65, 0.80, 0.90, 1.00, 1.10, 1.20, 1.30]
        ],
        separation_metrics={
            "x_sep_xc": 0.665,
            "x_reat_xc": 1.11,
            "bubble_length_xc": 0.445,
        },
        rans_error_baseline={
            "SA": "~35% bubble overpredict",
            "SST": "Best RANS; still ~20% bubble error",
            "k-epsilon": "May fail to predict separation",
            "v2f": "Improved vs SST (E3S benchmarks)",
        },
    ),

    # Case 3: Periodic Hill
    # (Re = 10.6k, Curved-wall periodic domain, ERCOFTAC DNS Database)
    "periodic_hill": BenchmarkCase(
        name="Periodic Hill (ERCOFTAC)",
        tier=CaseTier.T1_CANONICAL,
        category=SeparationCategory.CURVATURE,
        description="DNS-resolved curved-wall separation; ERCOFTAC case #081",
        reynolds_number=10_595,
        reference_velocity=1.0,
        reference_length=1.0,
        reference_length_name="h",
        domain_description="9h × 3.036h × 4.5h periodic domain",
        mesh_levels={"coarse": 50_000, "medium": 150_000, "fine": 400_000},
        yplus_target=0.5,
        data_source="ERCOFTAC Classic Database #081 / Breuer et al. DNS",
        data_url="http://cfd.mace.manchester.ac.uk/ercoftac/doku.php?id=cases:case081",
        reference_papers=[
            "Breuer et al. (2009), Computers & Fluids 38(2), pp.433-457, DOI:10.1016/j.compfluid.2008.05.002",
        ],
        separation_metrics={
            "x_sep_xh": 0.22,
            "x_reat_xh": 4.72,
            "dns_grid": "~5M nodes",
        },
        rans_error_baseline={
            "SA": "Overpredicts separation extent",
            "SST": "Better but still >15% error",
        },
    ),

    # Case 4: 2D Backward-Facing Step
    # (Re = 36k, Expansion ratio 1.125, Driver & Seegmiller 1985 Reference)
    "backward_facing_step": BenchmarkCase(
        name="2D Backward-Facing Step (Driver & Seegmiller)",
        tier=CaseTier.T1_CANONICAL,
        category=SeparationCategory.GEOMETRIC,
        description="Canonical geometry-forced separation; extensively validated",
        reynolds_number=36_000,
        reference_velocity=44.2,
        reference_length=0.0127,
        reference_length_name="H",
        domain_description="50H upstream, 200H downstream; step height 0.0127m; expansion ratio 1.125",
        mesh_levels={
            "coarse": 40_000, "medium": 90_000,
            "fine": 200_000, "xfine": 450_000,
        },
        delta_x_plus={
            "coarse": "50-100", "medium": "25-50",
            "fine": "10-25", "xfine": "5-10",
        },
        yplus_target=1.0,
        data_source="NASA TMR 2DBFS / ERCOFTAC C.30",
        data_url="https://turbmodels.larc.nasa.gov/backstep_val.html",
        reference_papers=[
            "Driver & Seegmiller (1985), AIAA J. 23(2), pp.163-171, DOI:10.2514/3.8890",
        ],
        profile_stations=[
            ExperimentalStation(loc, "x/H", ["U", "uu", "vv", "uv"])
            for loc in [1, 4, 6, 10]
        ],
        separation_metrics={
            "x_reat_xH": 6.26,
            "x_reat_xH_uncertainty": 0.10,
        },
        rans_error_baseline={
            "SA": "<5% on x_R",
            "SST": "<5% on x_R",
            "k-epsilon": "-20% on x_R",
        },
    ),

    "bachalo_johnson": BenchmarkCase(
        name="Bachalo-Johnson Axisymmetric Transonic Bump",
        tier=CaseTier.T1_CANONICAL,
        category=SeparationCategory.SHOCK_INDUCED,
        description="Shock-induced separation on circular-arc bump; M=0.875",
        reynolds_number=2.7e6,
        mach_number=0.875,
        reference_length=1.0,
        reference_length_name="c",
        data_source="NASA TMR ATB",
        data_url="https://turbmodels.larc.nasa.gov/axibump_val.html",
        reference_papers=[
            "Bachalo & Johnson (1986), AIAA J. 24(3), pp.437-443, DOI:10.2514/3.9307",
        ],
        separation_metrics={"bubble_overpredict": "20-30%"},
        rans_error_baseline={
            "SA": "20-30% bubble overpredict",
            "SST": "Similar to SA",
        },
    ),

    "axi_swbli": BenchmarkCase(
        name="Axisymmetric SWBLI M=2.85 (40% Challenge)",
        tier=CaseTier.T1_CANONICAL,
        category=SeparationCategory.SHOCK_INDUCED,
        description="NASA 40% Challenge primary case #5; shock-boundary layer interaction",
        reynolds_number=1e6,
        mach_number=2.85,
        data_source="NASA 40% Challenge",
        data_url="https://turbmodels.larc.nasa.gov/nasa40percent.html",
        rans_error_baseline={"SA": "Separation extent error"},
    ),

    # ---- Tier 2: Intermediate Complexity ----
    "simpsons_diffuser": BenchmarkCase(
        name="Simpson's Turbulent Boundary Layer Diffuser",
        tier=CaseTier.T2_INTERMEDIATE,
        category=SeparationCategory.SMOOTH_BODY_2D,
        description="Incipient detachment (1% backflow); wall-function sensitivity test",
        reynolds_number=1.5e6,
        data_source="Simpson 1981",
        rans_error_baseline={"wall_funcs": "Fail to predict incipient sep"},
    ),

    "buice_eaton_diffuser": BenchmarkCase(
        name="Buice-Eaton Asymmetric Diffuser",
        tier=CaseTier.T2_INTERMEDIATE,
        category=SeparationCategory.DIFFUSER,
        description="Asymmetric channel diffuser with NASA GRC validation data",
        reynolds_number=20_000,
        data_source="NPARC Alliance / NASA GRC",
        data_url="https://www.grc.nasa.gov/www/wind/valid/buice/buice.html",
        rans_error_baseline={"SA": "Model-dependent", "v2f": "~6% bubble"},
    ),

    "obi_diffuser": BenchmarkCase(
        name="Obi Asymmetric Diffuser",
        tier=CaseTier.T2_INTERMEDIATE,
        category=SeparationCategory.DIFFUSER,
        description="10° opening, expansion ratio 4.7; LDV data; ERCOFTAC SIG15 #8",
        reynolds_number=20_000,
        data_source="ERCOFTAC SIG15 Workshop #8",
        separation_metrics={"opening_angle": 10, "expansion_ratio": 4.7},
        rans_error_baseline={
            "k-epsilon": "Fails to predict recirculation",
            "v2f": "6% bubble error (Apsley & Leschziner)",
        },
    ),

    "naca_4412_te": BenchmarkCase(
        name="NACA 4412 Trailing-Edge Separation",
        tier=CaseTier.T2_INTERMEDIATE,
        category=SeparationCategory.SMOOTH_BODY_2D,
        description="Coles & Wadcock trailing-edge separation",
        reynolds_number=1.5e6,
        data_source="NASA TMR",
        data_url="https://turbmodels.larc.nasa.gov/naca4412sep_val.html",
    ),

    "naca_0012_stall": BenchmarkCase(
        name="NACA 0012 Airfoil (TMR Numerics)",
        tier=CaseTier.T2_INTERMEDIATE,
        category=SeparationCategory.SMOOTH_BODY_2D,
        description=(
            "NACA 0012 with corrected sharp-TE definition; TMR primary numerics "
            "case used in DPW-5/6 verification. Corrected airfoil equation: "
            "y = +/- 0.594689181*[0.298222773*sqrt(x) - 0.127125232*x "
            "- 0.357907906*x^2 + 0.291984971*x^3 - 0.105174606*x^4]. "
            "Max thickness ~11.894% chord."
        ),
        reynolds_number=6e6,
        mach_number=0.15,
        temperature=300.0,  # 540 R = 300 K
        reference_length=1.0,
        reference_length_name="c",
        domain_description=(
            "C-grid topology; farfield ~500c; sharp TE (corrected equation). "
            "Adiabatic wall BC; point vortex farfield correction recommended. "
            "Grid Family II (TE spacing=0.0000125c) recommended for convergence."
        ),
        mesh_levels={
            "L7_coarsest": 3_729,       # 113 x 33
            "L6": 14_625,               # 225 x 65
            "L5": 57_921,               # 449 x 129
            "L4": 230_529,              # 897 x 257
            "L3": 919_809,              # 1793 x 513
            "L2": 3_674_625,            # 3585 x 1025
            "L1_finest": 14_693_281,    # 7169 x 2049
        },
        yplus_target=0.5,
        data_source="NASA TMR Numerics Analysis (MRR Level 4)",
        data_url="https://turbmodels.larc.nasa.gov/naca0012numerics_val.html",
        reference_papers=[
            "Rumsey (2016), AIAA J. 54(9), pp.2563-2588, DOI:10.2514/1.J054555",
            "Thomas & Salas (1986), AIAA J. 24(7), pp.1074-1080, DOI:10.2514/3.9394",
        ],
        separation_metrics={
            "alpha_deg": [0, 10, 15],
            "grid_converged_SA_alpha10": {
                "CL": 1.0912,
                "CD": 0.01222,
                "CDp": 0.00601,
                "CDv": 0.006205,
                "CM": 0.00678,
            },
        },
        rans_error_baseline={
            "SA": "Grid-converged on Family II (CL=1.0912, CD=0.01222 at alpha=10)",
            "note": "Family I TE spacing yields wrong CL/CM convergence path",
        },
    ),


    "oat15a_buffet": BenchmarkCase(
        name="OAT15A Transonic Buffet",
        tier=CaseTier.T2_INTERMEDIATE,
        category=SeparationCategory.SHOCK_INDUCED,
        description="Supercritical airfoil buffet onset; requires unsteady RANS/DDES",
        reynolds_number=3e6,
        mach_number=0.73,
        data_source="ONERA",
        rans_error_baseline={"steady_rans": "Fails to predict buffet"},
    ),

    "axi_transonic_bump": BenchmarkCase(
        name="Axisymmetric Transonic Bump (ATB, 40% Challenge)",
        tier=CaseTier.T2_INTERMEDIATE,
        category=SeparationCategory.SHOCK_INDUCED,
        description="NASA 40% Challenge primary case #2; shock-separation interaction",
        reynolds_number=2.7e6,
        mach_number=0.875,
        data_source="NASA TMR ATB",
        data_url="https://turbmodels.larc.nasa.gov/axibump_val.html",
    ),

    # ---- Tier 3: Complex 3D ----
    "beverli_hill": BenchmarkCase(
        name="BeVERLI Hill (VT-NASA)",
        tier=CaseTier.T3_COMPLEX_3D,
        category=SeparationCategory.SMOOTH_BODY_3D,
        description=(
            "3D superelliptic smooth-body hill with massive leeward separation. "
            "Rapid pressure-gradient sign changes, 3D surface curvature, and "
            "non-equilibrium TBL development. Re_H = 250k–650k at 0°/30°/45° yaw. "
            "Designed by Virginia Tech & NASA Langley to expose Boussinesq "
            "approximation failures in standard RANS closures."
        ),
        reynolds_number=250_000,
        mach_number=0.08,
        reference_velocity=20.0,
        temperature=293.15,
        reference_length=0.1869,
        reference_length_name="H",
        domain_description=(
            "Wind-tunnel mounted 3D hill; H = 0.1869 m, width w = 0.93472 m, "
            "square flat top s = 0.09347 m. 5th-degree polynomial centerline "
            "profile with superelliptic corner transitions. Hill center at "
            "X = Z = 0 m; base Y = 0 flush with tunnel floor."
        ),
        mesh_levels={
            "coarse": 500_000, "medium": 2_000_000,
            "fine": 8_000_000, "xfine": 25_000_000,
        },
        yplus_target=1.0,
        data_source="VT-NASA BeVERLI Validation Challenge",
        data_url="https://beverlihill.aoe.vt.edu/",
        profile_stations=[
            ExperimentalStation(loc, "x/H", ["U", "V", "W", "uu", "vv", "ww", "uv"])
            for loc in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
        ],
        separation_metrics={
            "yaw_angles_deg": [0, 30, 45],
            "reynolds_numbers": [250_000, 650_000],
            "hill_height_m": 0.1869,
            "hill_width_m": 0.93472,
            "flat_top_m": 0.09347,
            "yaw_0_symmetric": True,
            "yaw_45_asymmetric_wake": True,
            "yaw_30_skewed_tbl": True,
        },
        reference_papers=[
            "Lowe et al. (2022), AIAA Paper 2022-0329, DOI:10.2514/6.2022-0329",
            "Rahmani et al. (2023), Exp. Fluids 64, DOI:10.1007/s00348-023-03607-w",
            "Lakebrink et al. (2023), AIAA Paper 2023-3263, DOI:10.2514/6.2023-3263",
        ],
        rans_error_baseline={
            "SA": (
                "Miscalculates separation onset location and volumetric extent "
                "of the leeward separation bubble across all yaw angles"
            ),
            "SST": (
                "Erroneously predicts asymmetric wakes on 45° yaw hill across "
                "all Re and grid densities; overpredicts separation extent"
            ),
            "SA_vs_SST": (
                "Inter-model variability exposes Boussinesq approximation "
                "breakdown in 3D non-equilibrium boundary layers"
            ),
        },
    ),

    "boeing_gaussian_bump": BenchmarkCase(
        name="NASA 3D Gaussian Speed Bump (SBSE)",
        tier=CaseTier.T3_COMPLEX_3D,
        category=SeparationCategory.SMOOTH_BODY_3D,
        description=(
            "3D Gaussian bump on flat plate: prolonged adverse pressure gradient "
            "leading to smooth-body 3D separation. h(x,z) = h₀·exp(-(x/x₀)²-(z/z₀)²). "
            "M = 0.176, Re_L = 2×10⁶. Standard SA under-predicts separation bubble; "
            "SA-RC rotation/curvature correction improves prediction. "
            "WMLES reference from NASA TMR."
        ),
        reynolds_number=2_000_000,
        mach_number=0.176,
        reference_velocity=60.0,
        temperature=300.0,
        reference_length=1.0,
        reference_length_name="L",
        domain_description=(
            "Gaussian bump geometry: h(x,z) = h₀·exp(-(x/x₀)²-(z/z₀)²) where "
            "h₀ = 0.085L (bump height), x₀ = 0.195L (streamwise half-width), "
            "z₀ = 0.06L (spanwise half-width). L = bump reference length. "
            "Bump mounted on wind tunnel floor. Error-function shoulders "
            "ensure smooth transition to flat surface."
        ),
        mesh_levels={
            "coarse": 500_000, "medium": 2_000_000,
            "fine": 8_000_000, "xfine": 20_000_000,
        },
        yplus_target=1.0,
        data_source="NASA TMR / WMLES Reference",
        data_url="https://turbmodels.larc.nasa.gov/Other_LES_Data/GaussianBump.html",
        profile_stations=[
            ExperimentalStation(loc, "x/L", ["U", "V", "W", "uu", "vv", "uv", "Cp", "Cf"])
            for loc in [0.6, 0.8, 1.0, 1.2, 1.4, 1.8]
        ],
        separation_metrics={
            "bump_height_L": 0.085,
            "bump_x0_L": 0.195,
            "bump_z0_L": 0.06,
            "x_sep_xL_wmles": 0.75,
            "x_reat_xL_wmles": 1.35,
            "bubble_length_xL_wmles": 0.60,
            "x_sep_xL_sa": 0.82,
            "x_reat_xL_sa": 1.15,
            "bubble_length_xL_sa": 0.33,
        },
        reference_papers=[
            "Williams et al. (2020), AIAA Paper 2020-1087, DOI:10.2514/6.2020-1087",
            "Iyer & Malik (2020), NASA TM 2020-220469",
            "Gray et al. (2021), AIAA J. 59(10), DOI:10.2514/1.J060234",
        ],
        rans_error_baseline={
            "SA": (
                "Standard SA severely under-predicts spatial extent of the "
                "separation bubble without rotation/curvature corrections; "
                "bubble length is ~45% shorter than WMLES reference"
            ),
            "SA-RC": (
                "SA with rotation/curvature correction (SA_OPTIONS= RC in SU2) "
                "improves separation bubble prediction by 8-14% over baseline SA; "
                "still under-predicts but captures correct topology"
            ),
            "SST": (
                "SST also under-predicts separation extent; "
                "comparable to SA in underestimating bubble volume"
            ),
        },
    ),

    "stanford_3d_diffuser": BenchmarkCase(
        name="Stanford 3D Diffuser",
        tier=CaseTier.T3_COMPLEX_3D,
        category=SeparationCategory.DIFFUSER,
        description="Two configs: 11.3°/2.56° vs 9°/4° wall angles; high geometry sensitivity",
        reynolds_number=10_000,
        data_source="Cherry et al. 2008 / ERCOFTAC UFR 4-16",
        separation_metrics={
            "config1": {"upper_wall": "11.3°", "side_wall": "2.56°"},
            "config2": {"upper_wall": "9°", "side_wall": "4°"},
        },
        rans_error_baseline={"all": "Highly geometry-sensitive 3D separation"},
    ),

    "juncture_flow": BenchmarkCase(
        name="NASA Juncture Flow (JF)",
        tier=CaseTier.T3_COMPLEX_3D,
        category=SeparationCategory.CORNER_JUNCTURE,
        description="Wing-body junction; horseshoe vortex and corner separation bubble",
        reynolds_number=2.4e6,
        data_source="NASA TMR / HLPW-4",
        data_url="https://turbmodels.larc.nasa.gov/junctureflow_val.html",
        reference_papers=[
            "Rumsey et al. (2018), AIAA Paper 2018-3319, DOI:10.2514/6.2018-3319",
            "Rumsey et al. (2020), AIAA J. 58, DOI:10.2514/1.J058725",
        ],
        rans_error_baseline={
            "SA": "Misses corner bubble",
            "SA-QCR": "Captures corner bubble",
            "RSM": "Best for horseshoe vortex",
        },
    ),

    # ---- Tier 2: Axisymmetric Jet ----
    "axisymmetric_jet": BenchmarkCase(
        name="Axisymmetric Subsonic/Supersonic Jet",
        tier=CaseTier.T2_INTERMEDIATE,
        category=SeparationCategory.FREE_SHEAR,
        description=(
            "Round jet exhausting into quiescent air; tests free-shear layer "
            "spreading, centerline velocity decay, and turbulent mixing. "
            "Subsonic: M_j=0.5, Re_D~10⁶; supersonic: M_j=1.4 for noise studies."
        ),
        reynolds_number=1e6,
        mach_number=0.5,
        reference_velocity=170.0,
        reference_length=0.0508,
        reference_length_name="D",
        domain_description=(
            "Axisymmetric nozzle, D=50.8mm, domain 60D downstream × 15D radial. "
            "Nozzle-exit BL profile specified from Bridges & Wernet."
        ),
        mesh_levels={
            "coarse": 80_000, "medium": 250_000,
            "fine": 700_000, "xfine": 2_000_000,
        },
        yplus_target=1.0,
        data_source="NASA SHJAR / Bridges & Wernet (2010)",
        data_url="https://turbmodels.larc.nasa.gov/jetsubsonic_val.html",
        reference_papers=[
            "Bridges & Wernet (2010), NASA/TM—2010-216736",
            "Witze (1974), AIAA J. 12(4), pp.417-418",
        ],
        separation_metrics={
            "potential_core_xD": 6.0,
            "spreading_rate": 0.094,
            "centerline_decay_coeff": 5.8,
        },
        rans_error_baseline={
            "SA": "Overpredicts spreading rate by ~15%, short potential core",
            "SST": "Better spreading rate but still ~10% core length error",
            "k-epsilon": "Classic round-jet anomaly: ~30% overprediction",
        },
    ),

    # ---- Tier 3: 3D Bump-in-Channel ----
    "bump_3d_channel": BenchmarkCase(
        name="3D Gaussian Bump in Channel (TMR Verification)",
        tier=CaseTier.T3_COMPLEX_3D,
        category=SeparationCategory.SMOOTH_BODY_3D,
        description=(
            "3D Gaussian bump mounted on channel floor. Smooth-body separation "
            "driven by adverse pressure gradient over bump crest. Extends 2D "
            "bump-in-channel verification case to 3D with spanwise confinement. "
            "WMLES reference data from NASA TMR."
        ),
        reynolds_number=3e6,
        mach_number=0.2,
        reference_velocity=69.44,
        temperature=300.0,
        reference_length=1.0,
        reference_length_name="L",
        domain_description=(
            "Channel with Gaussian bump: h(x,z) = h₀·exp(-(x/x₀)²-(z/z₀)²). "
            "h₀ = 0.05L, x₀ = 0.2L, z₀ = 0.1L. Channel height = 0.5L. "
            "Inlet at x/L = -3.0, outlet at x/L = 5.0."
        ),
        mesh_levels={
            "coarse": 500_000, "medium": 2_000_000,
            "fine": 6_000_000, "xfine": 18_000_000,
        },
        yplus_target=1.0,
        data_source="NASA TMR / WMLES Reference",
        data_url="https://turbmodels.larc.nasa.gov/Other_LES_Data/3Dbump.html",
        profile_stations=[
            ExperimentalStation(loc, "x/L", ["U", "V", "W", "Cp", "Cf"])
            for loc in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
        ],
        separation_metrics={
            "bump_height_L": 0.05,
            "bump_x0_L": 0.2,
            "bump_z0_L": 0.1,
            "x_sep_xL_sa": 0.85,
            "x_reat_xL_sa": 1.20,
            "x_sep_xL_wmles": 0.78,
            "x_reat_xL_wmles": 1.40,
        },
        reference_papers=[
            "Uzun & Malik (2018), AIAA Paper 2018-3713, DOI:10.2514/6.2018-3713",
        ],
        rans_error_baseline={
            "SA": "Under-predicts separation extent by ~40% vs WMLES",
            "SST": "Similar to SA; slight improvement with RC correction",
            "SA-RC": "10-15% improvement over baseline SA",
        },
    ),

    # ---- Tier 3: NASA Common Research Model ----
    "nasa_crm": BenchmarkCase(
        name="NASA Common Research Model (DPW-5/6)",
        tier=CaseTier.T3_COMPLEX_3D,
        category=SeparationCategory.SMOOTH_BODY_3D,
        description=(
            "Transonic transport wing-body-tail configuration from the AIAA Drag "
            "Prediction Workshop series (DPW-5/DPW-6). M=0.85, Re_c=5×10⁶, "
            "α=2.75° (design condition). Primary metrics: total drag count, "
            "drag polar, wing Cp sections. Requires HPC resources."
        ),
        reynolds_number=5e6,
        mach_number=0.85,
        temperature=310.93,
        reference_length=7.00532,
        reference_length_name="c_ref",
        domain_description=(
            "Wing-body-tail; semi-span with symmetry BC. Reference area = 191,844 in². "
            "Moment reference at x = 1325.9 in, z = 177.95 in. "
            "Farfield ~100 chord lengths from body."
        ),
        mesh_levels={
            "L5_coarsest": 638_976,
            "L4": 5_111_808,
            "L3": 14_370_048,
            "L2": 40_894_464,
            "L1_finest": 115_246_080,
        },
        yplus_target=1.0,
        data_source="AIAA DPW-5/6 / NASA CRM (Vassberg et al.)",
        data_url="https://aiaa-dpw.larc.nasa.gov/",
        reference_papers=[
            "Vassberg et al. (2008), AIAA Paper 2008-6919, DOI:10.2514/6.2008-6919",
            "Levy et al. (2014), J. Aircraft 51(4), DOI:10.2514/1.C032389",
            "Tinoco et al. (2018), J. Aircraft 55(4), DOI:10.2514/1.C034409",
        ],
        profile_stations=[
            ExperimentalStation(loc, "eta", ["Cp"])
            for loc in [0.131, 0.283, 0.502, 0.727, 0.846, 0.950]
        ],
        separation_metrics={
            "CL_design": 0.500,
            "CD_design_counts": 253.5,
            "CM_design": -0.0935,
            "alpha_design_deg": 2.75,
            "dpw5_committee_avg": {
                "CL": 0.500, "CD_counts": 253.5, "CM": -0.0935,
            },
        },
        rans_error_baseline={
            "SA": "Within DPW-5 scatter band (~4 counts spread)",
            "SST": "Similar accuracy; slightly different drag split",
            "note": "Grid sensitivity > model sensitivity at design condition",
        },
    ),
}


# =============================================================================
# Turbulence Models (15 total)
# =============================================================================
TURBULENCE_MODELS: Dict[str, TurbulenceModel] = {

    "SA": TurbulenceModel(
        name="Spalart-Allmaras",
        abbreviation="SA",
        model_type=ModelType.RANS_1EQ,
        openfoam_name="SpalartAllmaras",
        su2_name="SA",
        n_equations=1,
        description="Robust 1-equation model; production workhorse",
        key_paper="Spalart & Allmaras (1992), AIAA Paper 92-0439, DOI:10.2514/6.1992-439",
        expected_performance={
            "BFS": "<5% x_R error",
            "Hump": "~35% bubble overpredict",
            "Juncture": "Misses corner bubble",
        },
        openfoam_settings={"simulationType": "RAS", "RASModel": "SpalartAllmaras"},
    ),

    "SA-QCR": TurbulenceModel(
        name="Spalart-Allmaras with QCR",
        abbreviation="SA-QCR",
        model_type=ModelType.RANS_1EQ,
        openfoam_name="SpalartAllmaras",
        n_equations=1,
        description="Quadratic constitutive relation for corner flows; DPW-mandated",
        expected_performance={"Juncture": "Captures corner bubble"},
        openfoam_settings={
            "simulationType": "RAS",
            "RASModel": "SpalartAllmaras",
            "QCR": True,
        },
    ),

    "SA-RC": TurbulenceModel(
        name="Spalart-Allmaras with Rotation Correction",
        abbreviation="SA-RC",
        model_type=ModelType.RANS_1EQ,
        openfoam_name="SpalartAllmaras",
        n_equations=1,
        description="Rotation/curvature correction; NASA TMR MRR-3",
        expected_performance={"Hump": "Improved for curved flows"},
        openfoam_settings={
            "simulationType": "RAS",
            "RASModel": "SpalartAllmaras",
            "rotationCorrection": True,
        },
    ),

    "SST": TurbulenceModel(
        name="k-omega SST",
        abbreviation="SST",
        model_type=ModelType.RANS_2EQ,
        openfoam_name="kOmegaSST",
        su2_name="SST",
        n_equations=2,
        description="Best general-purpose RANS for APG; blends k-ω and k-ε",
        key_paper="Menter (1994), AIAA J. 32(8), pp.1598-1605, DOI:10.2514/3.12149",
        expected_performance={
            "BFS": "<5% x_R error",
            "Hump": "Best RANS (~20% bubble error, MAPE ~8.3%)",
            "Diffuser": "Good with EASM",
        },
        openfoam_settings={"simulationType": "RAS", "RASModel": "kOmegaSST"},
    ),

    "kEpsilon": TurbulenceModel(
        name="k-epsilon Realizable",
        abbreviation="k-ε",
        model_type=ModelType.RANS_2EQ,
        openfoam_name="realizableKE",
        su2_name="KE",
        n_equations=2,
        description="Standard 2-eq; historically poor for separation",
        expected_performance={
            "BFS": "-20% x_R error",
            "Diffuser": "Fails to predict recirculation",
        },
        openfoam_settings={"simulationType": "RAS", "RASModel": "realizableKE"},
    ),

    "v2f": TurbulenceModel(
        name="v²-f",
        abbreviation="v²-f",
        model_type=ModelType.RANS_4EQ,
        openfoam_name="v2f",
        n_equations=4,
        description="4-equation model with improved near-wall anisotropy; best for diffusers",
        expected_performance={
            "Obi diffuser": "6% bubble error (Apsley & Leschziner)",
            "Hump": "Improved vs SST",
        },
        openfoam_settings={"simulationType": "RAS", "RASModel": "v2f"},
    ),

    "EASM": TurbulenceModel(
        name="Explicit Algebraic Stress Model",
        abbreviation="EASM",
        model_type=ModelType.EARSM,
        openfoam_name="kOmegaSSTLM",  # Approximate; may need custom
        n_equations=2,
        description="Explicit algebraic RSM; best with SST for diffuser",
        expected_performance={
            "Diffuser": "Best with SST",
            "High-lift": "Exaggerates separation (HLPW-2)",
        },
    ),

    "RSM": TurbulenceModel(
        name="SSG/LRR Reynolds Stress Model",
        abbreviation="RSM",
        model_type=ModelType.RSM,
        openfoam_name="LRR",
        n_equations=7,
        description="Full stress transport; captures anisotropy and secondary flows",
        expected_performance={
            "Juncture": "Captures horseshoe vortex",
            "Hump": "TMR MRR-3",
        },
        openfoam_settings={"simulationType": "RAS", "RASModel": "LRR"},
    ),

    "gammaReTheta": TurbulenceModel(
        name="γ-Reθ SST Transition Model",
        abbreviation="γ-Reθ",
        model_type=ModelType.TRANSITION,
        openfoam_name="kOmegaSSTLM",
        n_equations=4,
        description="Transition model for untripped cases",
        openfoam_settings={"simulationType": "RAS", "RASModel": "kOmegaSSTLM"},
    ),

    "SA-BCM": TurbulenceModel(
        name="SA with BCM Algebraic Transition",
        abbreviation="SA-BCM",
        model_type=ModelType.TRANSITION,
        openfoam_name="SpalartAllmaras",
        su2_name="SA_BCM",
        n_equations=1,
        description=(
            "Bas-Cakmakcioglu algebraic transition model coupled with SA; "
            "no extra transport equations, uses local flow quantities to "
            "trigger transition. Suitable for low-Tu bypass transition."
        ),
        key_paper="Cakmakcioglu et al. (2018), AIAA J. 56(9), DOI:10.2514/1.J056467",
        expected_performance={
            "T3A": "Captures transition onset within 10% of experiment",
            "NACA0012_alpha0": "Resolves laminar bubble on upper surface",
        },
        openfoam_settings={
            "simulationType": "RAS",
            "RASModel": "SpalartAllmaras",
            "BCM": True,
        },
    ),

    "DDES": TurbulenceModel(
        name="Delayed Detached Eddy Simulation",
        abbreviation="DDES",
        model_type=ModelType.HYBRID,
        openfoam_name="SpalartAllmarasDDES",
        n_equations=1,
        description="Hybrid RANS-LES; grid/CFL sensitive",
        expected_performance={"Hump": "Best accuracy but 10-100× cost"},
        openfoam_settings={
            "simulationType": "LES",
            "LESModel": "SpalartAllmarasDDES",
            "delta": "cubeRootVol",
        },
    ),

    "SBES": TurbulenceModel(
        name="Stress-Blended Eddy Simulation",
        abbreviation="SBES",
        model_type=ModelType.HYBRID,
        openfoam_name="kOmegaSSTSBES",  # Custom or Fluent
        n_equations=2,
        description="Improved shielding vs DDES; smoother RANS→LES transition",
    ),

    "SAS": TurbulenceModel(
        name="Scale-Adaptive Simulation",
        abbreviation="SAS",
        model_type=ModelType.HYBRID,
        openfoam_name="kOmegaSSTSAS",
        n_equations=2,
        description="Auto length-scale adjustment; between URANS and LES",
        openfoam_settings={"simulationType": "RAS", "RASModel": "kOmegaSSTSAS"},
    ),

    "WMLES-EQWM": TurbulenceModel(
        name="Wall-Modeled LES (Equilibrium)",
        abbreviation="EQWM",
        model_type=ModelType.WMLES,
        openfoam_name="WALE",
        description="Equilibrium wall model; HLPW-5: 2-3% consistency",
        expected_performance={"Hump": "Underpredicts Cf attached; overpredicts in sep"},
        openfoam_settings={
            "simulationType": "LES",
            "LESModel": "WALE",
            "delta": "cubeRootVol",
        },
    ),

    "WMLES-NEQWM": TurbulenceModel(
        name="Wall-Modeled LES (Non-Equilibrium)",
        abbreviation="NEQWM",
        model_type=ModelType.WMLES,
        openfoam_name="WALE",
        description="Non-equilibrium wall model; outperforms EQWM in sep bubble (Park & Larsson)",
        expected_performance={
            "Hump": "Better Cf in sep bubble than EQWM; pressure+advection key",
        },
    ),
}


# =============================================================================
# Model Comparison Matrix (which models to compare per case)
# =============================================================================
MODEL_COMPARISON_MATRIX: Dict[str, List[str]] = {
    "backward_facing_step": ["SA", "SST", "kEpsilon"],
    "nasa_hump":            ["SA", "SST", "SA-RC", "kEpsilon"],
    "naca_0012_stall":      ["SA", "SST", "gammaReTheta"],
    "bachalo_johnson":      ["SA", "SST", "SA-RC"],
    "flat_plate":           ["SA", "SST", "kEpsilon"],
    "periodic_hill":        ["SA", "SST", "kEpsilon"],
    "juncture_flow":        ["SA", "SA-QCR", "SST"],
    "boeing_gaussian_bump": ["SA", "SA-RC", "SST"],
}


# =============================================================================
# RSM Reference Results (from TMR CFL3D/FUN3D — SU2 lacks native RSM)
# =============================================================================
RSM_REFERENCE_RESULTS: Dict[str, Dict[str, Any]] = {
    "nasa_hump": {
        "LRR": {
            "x_reat": 1.05, "bubble_length": 0.385,
            "source": "TMR MRR-3: RSM outperforms all eddy-viscosity models",
            "improvement_over_SA": "Reattachment 17% closer to experiment",
        },
        "SSG": {
            "x_reat": 1.08,
            "source": "TMR: SSG slightly worse than LRR on hump",
        },
    },
    "backward_facing_step": {
        "LRR": {
            "x_reat_xH": 6.10,
            "source": "TMR: RSM captures near-wall anisotropy better",
        },
    },
    "juncture_flow": {
        "LRR": {
            "captures_corner_bubble": True,
            "horseshoe_vortex": "Correctly predicts primary and secondary vortices",
            "source": "Rumsey et al. (2020), AIAA J.",
        },
    },
    "periodic_hill": {
        "LRR": {
            "x_reat_xH": 4.60,
            "source": "Breuer et al. 2009: RSM best RANS for periodic hill",
            "dns_reference": 4.72,
            "error_pct": 2.5,
        },
    },
}


# =============================================================================
# Solver & Numerical Settings
# =============================================================================
SOLVER_DEFAULTS = {
    "convergence_residual": 1e-5,
    "max_iterations": 5000,
    "relaxation": {
        "U": 0.7,
        "p": 0.3,
        "nuTilda": 0.7,
        "k": 0.7,
        "omega": 0.7,
        "epsilon": 0.7,
    },
    "linear_solvers": {
        "p": {"solver": "GAMG", "smoother": "GaussSeidel", "tolerance": 1e-6, "relTol": 0.1},
        "U": {"solver": "smoothSolver", "smoother": "GaussSeidel", "tolerance": 1e-8, "relTol": 0.1},
    },
    "pressure_velocity_coupling": "SIMPLE",
    "nNonOrthogonalCorrectors": 0,
    "first_to_second_order": True,
    "strict_convergence_residual": 1e-12,
    "strict_max_iterations": 50000,
}


# =============================================================================
# Convergence Criteria (NASA/AIAA guidelines)
# =============================================================================
CONVERGENCE_CRITERIA = {
    "residual_target": 1e-12,           # TMR-recommended residual level
    "force_stability_window": 500,       # Last N iterations for CL/CD stability
    "force_stability_tolerance": 0.001,  # 0.1% relative change threshold
    "gci_threshold": 0.05,              # GCI < 5% for grid independence
    "min_grid_levels": 3,               # Minimum 3 levels for GCI
    "asymptotic_range_tolerance": 0.1,   # |R - 1| < 0.1 for asymptotic range
}

NUMERICAL_SCHEMES = {
    "gradient": {
        "default": "Gauss linear",
        "limited": "cellLimited Gauss linear 1",
    },
    "convection": {
        "upwind": "bounded Gauss upwind",
        "linearUpwind": "bounded Gauss linearUpwind grad(U)",
        "LUST": "bounded Gauss LUST grad(U)",
    },
    "time": {
        "steady": "steadyState",
        "backward": "backward",
        "CrankNicolson": "CrankNicolson 0.9",
    },
}

SCHEME_SENSITIVITY_MATRIX = [
    {"gradient": "default", "convection": "upwind", "label": "1st-order baseline"},
    {"gradient": "default", "convection": "linearUpwind", "label": "2nd-order standard"},
    {"gradient": "limited", "convection": "linearUpwind", "label": "2nd-order limited"},
    {"gradient": "limited", "convection": "LUST", "label": "Blended LUST"},
]


# =============================================================================
# SU2 Numerical Defaults (Second-Order Everywhere - NASA TMR Compliant)
# =============================================================================
SU2_NUMERICAL_DEFAULTS: Dict[str, Any] = {
    "spatial_order": 2,
    "muscl_flow": True,
    "muscl_turb": True,        # NASA TMR: first-order is "inadequate for verification"
    "limiter_flow": "VENKATAKRISHNAN",
    "limiter_turb": "VENKATAKRISHNAN",
    "conv_method_flow": "ROE",
    "conv_method_turb": "SCALAR_UPWIND",
    "conv_residual_minval": -10,  # Tighter than -8; target -12 for strict V&V
    "cfl_number": 10.0,
    "cfl_adapt": True,
    "cfl_adapt_param": (0.5, 1.5, 1.0, 100.0),
    "gradient_method": "GREEN_GAUSS",
    "notes": {
        "second_order_rationale": (
            "NASA TMR and ASME V&V 20-2009 require second-order spatial accuracy "
            "for both flow and turbulence variables. First-order upwind introduces "
            "excessive numerical diffusion that masks modeling errors."
        ),
        "limiter_choice": (
            "Venkatakrishnan limiter preserves second-order accuracy in smooth "
            "regions while preventing oscillations near discontinuities (shocks). "
            "For SWBLI, VENKATAKRISHNAN_WANG may be needed for sharper shock capture."
        ),
        "line_implicit_reference": (
            "For stiff hypersonic SWBLI, NASA Wind-US uses line-implicit solving "
            "(Ref: NPARC Alliance). SU2 equivalent: enable ILU preconditioner with "
            "MUSCL + ROE. Consider under-relaxation ~0.7 for stability."
        ),
    },
}


# =============================================================================
# TMR Published Code Results (for Cross-Solver Comparison)
# =============================================================================
TMR_CODE_RESULTS: Dict[str, Dict[str, Any]] = {
    "flat_plate": {
        "CFL3D": {
            "Cf_x5": 0.002967,
            "source": "https://turbmodels.larc.nasa.gov/flatplate.html",
            "model": "SA",
        },
        "FUN3D": {
            "Cf_x5": 0.002969,
            "source": "https://turbmodels.larc.nasa.gov/flatplate.html",
            "model": "SA",
        },
    },
    "naca_0012_stall": {
        "CFL3D_SA_alpha0": {
            "CL": 0.0, "CD": 0.00819, "CM": 0.0,
            "source": "https://turbmodels.larc.nasa.gov/naca0012_val.html",
            "model": "SA", "grid": "finest",
        },
        "CFL3D_SA_alpha10": {
            "CL": 1.0909, "CD": 0.01231, "CDp": 0.00613, "CDv": 0.00618,
            "source": "https://turbmodels.larc.nasa.gov/naca0012_val.html",
            "model": "SA", "grid": "L1_finest",
        },
        "FUN3D_SA_alpha10": {
            "CL": 1.0912, "CD": 0.01222, "CDp": 0.00601, "CDv": 0.006205,
            "source": "https://turbmodels.larc.nasa.gov/naca0012_val.html",
            "model": "SA", "grid": "L1_finest",
        },
        "CFL3D_SA_alpha15": {
            "CL": 1.5461, "CD": 0.02124,
            "source": "https://turbmodels.larc.nasa.gov/naca0012_val.html",
            "model": "SA", "grid": "L1_finest",
        },
    },
    "nasa_hump": {
        "CFL3D_SA": {
            "x_sep": 0.665, "x_reat": 1.27, "bubble_length": 0.605,
            "source": "https://turbmodels.larc.nasa.gov/nasahump_val.html",
            "model": "SA",
        },
        "CFL3D_SST": {
            "x_sep": 0.665, "x_reat": 1.17, "bubble_length": 0.505,
            "source": "https://turbmodels.larc.nasa.gov/nasahump_val.html",
            "model": "SST",
        },
    },
    "backward_facing_step": {
        "CFL3D_SA": {
            "x_reat": 6.28,
            "source": "https://turbmodels.larc.nasa.gov/backstep_val.html",
            "model": "SA",
        },
        "CFL3D_SST": {
            "x_reat": 6.40,
            "source": "https://turbmodels.larc.nasa.gov/backstep_val.html",
            "model": "SST",
        },
    },
    "bump_channel": {
        "CFL3D_SA": {
            "Cf_peak": 0.00509,
            "source": "https://tmbwg.github.io/turbmodels/bump.html",
            "model": "SA",
        },
    },
}


# =============================================================================
# Sensitivity Analysis Parameters (Sobol)
# =============================================================================
SENSITIVITY_PARAMETERS = {
    "num_vars": 6,
    "names": [
        "inlet_velocity",
        "turbulence_intensity",
        "inlet_length_scale",
        "wall_roughness",
        "relaxation_factor_U",
        "relaxation_factor_p",
    ],
    "bounds": [
        [40.0, 50.0],       # ±10% around 44.2 m/s
        [0.01, 0.10],       # 1-10%
        [0.001, 0.01],      # Length scale
        [0.0, 1e-5],        # Roughness height
        [0.5, 0.9],         # Under-relaxation U
        [0.2, 0.5],         # Under-relaxation p
    ],
}


# =============================================================================
# UQ Input Distributions
# =============================================================================
UQ_INPUT_DISTRIBUTIONS = {
    "U_inlet": {"distribution": "normal", "mean": 44.2, "std": 0.5},
    "turbulence_intensity": {"distribution": "uniform", "lower": 0.03, "upper": 0.07},
    "step_height": {"distribution": "normal", "mean": 1.0, "std": 0.01},
}


# =============================================================================
# NASA 40% Challenge Cases
# =============================================================================
NASA_40_PERCENT_CASES = [
    {"name": "2D NASA Hump", "tmr_code": "2DWMH",
     "url": "https://turbmodels.larc.nasa.gov/nasahump_val.html"},
    {"name": "Axisymmetric Transonic Bump", "tmr_code": "ATB",
     "url": "https://turbmodels.larc.nasa.gov/axibump_val.html"},
    {"name": "2D Compressible Mixing Layer", "tmr_code": None,
     "data": "https://turbmodels.larc.nasa.gov/CompressMixingLayer/Goebel_data.tar.gz"},
    {"name": "Round Jet (set points 3/23/7)", "tmr_code": "ASJ/AHSJ/ANSJ",
     "url": "https://turbmodels.larc.nasa.gov/jetsubsonic_val.html"},
    {"name": "Axi SWBLI M=2.85", "tmr_code": None,
     "data": "https://turbmodels.larc.nasa.gov/SWBLI_Brown/axi_swbli_data.tar.gz"},
]


# =============================================================================
# Data Source Registry
# =============================================================================
DATA_SOURCES = {
    "nasa_tmr": {
        "name": "NASA Turbulence Modeling Resource",
        "url": "https://turbmodels.larc.nasa.gov/",
        "contents": "All RANS cases + grids + experimental data",
    },
    "nasa_40_percent": {
        "name": "NASA 40% Challenge",
        "url": "https://turbmodels.larc.nasa.gov/nasa40percent.html",
        "contents": "Standard test cases document v6",
    },
    "ercoftac_classic": {
        "name": "ERCOFTAC Classic Collection",
        "url": "http://cfd.mace.manchester.ac.uk/ercoftac/",
        "contents": "BFS, periodic hill, diffusers",
    },
    "ercoftac_ml": {
        "name": "ERCOFTAC ML Workshop Dataset",
        "url": "https://www.ercoftac.org/",
        "contents": "87 flat-plate separation cases (Re 15k-80k, Tu 1.5-3.5%)",
    },
    "vt_nasa_beverli": {
        "name": "VT-NASA BeVERLI Hill",
        "url": "https://beverlihill.aoe.vt.edu/",
        "contents": "3D hill experimental data",
    },
    "nparc": {
        "name": "NPARC Alliance Validation Archive",
        "url": "https://www.grc.nasa.gov/www/wind/valid/",
        "contents": "Buice diffuser and other cases",
    },
    "jhu_turbulence": {
        "name": "JHU Turbulence Database",
        "url": "https://turbulence.pha.jhu.edu/",
        "contents": "DNS/LES datasets",
    },
    "aiaa_dpw": {
        "name": "AIAA Drag Prediction Workshop (DPW)",
        "url": "https://aiaa-dpw.larc.nasa.gov/",
        "contents": "Multi-solver benchmark: CRM grids, exp data, DPW-1 to DPW-7 results",
        "key_papers": [
            "Tinoco et al. (2018), J. Aircraft 55(4), DOI:10.2514/1.C034409",
            "Levy et al. (2014), J. Aircraft 51(4), DOI:10.2514/1.C032389",
        ],
    },
    "aiaa_hiliftpw": {
        "name": "AIAA High-Lift Prediction Workshop (HiLiftPW)",
        "url": "https://hiliftpw.larc.nasa.gov/",
        "contents": "High-lift stall benchmark: Trap Wing, CRM-HL, grids + exp data",
        "key_papers": [
            "Rumsey et al. (2011), J. Aircraft 48(6), pp.2068-2079, DOI:10.2514/1.C031447",
        ],
    },
    "nasa_crm": {
        "name": "NASA Common Research Model (CRM)",
        "url": "https://commonresearchmodel.larc.nasa.gov/",
        "contents": "Standard geometry for CFD validation: STEP files, grids, exp data",
        "key_papers": [
            "Rivers & Dittberner (2014), AIAA-2014-3129, DOI:10.2514/6.2014-3129",
        ],
    },
    "nasa_juncture_flow": {
        "name": "NASA Juncture Flow Experiment",
        "url": "https://turbmodels.larc.nasa.gov/junctureflow_val.html",
        "contents": "Wing-body junction LDV/PIV data for corner separation validation",
        "key_papers": [
            "Rumsey et al. (2018), AIAA-2018-3319, DOI:10.2514/6.2018-3319",
        ],
    },
    "arxiv_hills_dns": {
        "name": "Parameterized Periodic Hills DNS",
        "url": "https://arxiv.org/abs/1910.01264",
        "contents": "Family of slopes for ML training",
    },
    "bradshaw_archive": {
        "name": "Bradshaw 1996 Collaborative Testing Archive",
        "url": "https://turbmodels.larc.nasa.gov/bradshaw.html",
        "contents": "Kim BFS, NACA 4412 TE sep, duct flows, mixing layers",
        "key_papers": [
            "Kim et al. (1998), J. Fluids Eng. 120(3), DOI:10.1115/1.2820690",
            "Coles & Wadcock (1979), AIAA Paper 79-1457",
        ],
    },
    "jhtdb": {
        "name": "Johns Hopkins Turbulence Database",
        "url": "https://turbulence.pha.jhu.edu/",
        "contents": "DNS/LES datasets for channel, isotropic turbulence, MHD, boundary layers",
        "datasets": {
            "channel_flow": "Re_tau=1000 channel flow DNS (Graham et al. 2016)",
            "isotropic_turbulence": "Forced isotropic turbulence at Re_lambda=433",
            "boundary_layer": "Turbulent BL at Re_theta=1000-4000",
        },
        "access": "REST API and Python client (pyJHTDB)",
    },
    "nparc_wind_us": {
        "name": "NPARC Alliance Wind-US Validation Archive",
        "url": "https://www.grc.nasa.gov/www/wind/valid/",
        "contents": "Structured CFD results for SWBLI, bump, diffuser, jet cases",
        "cases": {
            "swbli_m5": "https://www.grc.nasa.gov/www/wind/valid/swbli/swbli.html",
            "bump_channel": "https://www.grc.nasa.gov/www/wind/valid/bump/bump.html",
            "buice_diffuser": "https://www.grc.nasa.gov/www/wind/valid/buice/buice.html",
        },
        "notes": "Wind-US uses line-implicit solving for stiff hypersonic cases",
    },
    "ercoftac_dns_data": {
        "name": "ERCOFTAC DNS/LES Databases",
        "url": "http://cfd.mace.manchester.ac.uk/ercoftac/",
        "contents": "High-fidelity reference data for turbulence model assessment",
        "cases": {
            "periodic_hill_081": {
                "url": "http://cfd.mace.manchester.ac.uk/ercoftac/doku.php?id=cases:case081",
                "ref": "Breuer et al. (2009), Computers & Fluids 38(2)",
                "Re_H": 10595,
            },
            "backward_step_030": {
                "url": "http://cfd.mace.manchester.ac.uk/ercoftac/doku.php?id=cases:case030",
                "ref": "Le et al. (1997) DNS, Re_h=5100",
                "Re_h": 5100,
            },
            "diffuser_UFR_4_16": {
                "url": "http://qnet-ercoftac.cfms.org.uk/w/index.php/UFR_4-16",
                "ref": "Cherry et al. (2008), 3D diffuser DNS",
            },
        },
    },
    "tmr_experiments": {
        "name": "TMR Data from Experiments Section",
        "url": "https://turbmodels.larc.nasa.gov/Other_Coverage/data_from_exp.html",
        "contents": "Experimental datasets referenced by TMR for direct download",
    },
}


# =============================================================================
# Experimental Reference Papers Registry
# =============================================================================
EXPERIMENTAL_REFERENCES: Dict[str, Dict[str, Any]] = {
    "nasa_hump": {
        "experiment": "Greenblatt et al. (2006)",
        "title": "Experimental Investigation of Separation Control",
        "journal": "AIAA J. 44(1), pp.51-68, DOI:10.2514/1.9377",
        "measurements": ["Cp", "Cf", "PIV velocity profiles"],
        "profile_stations_xc": [0.65, 0.80, 0.90, 1.00, 1.10, 1.20, 1.30],
        "key_metrics": {
            "x_sep_xc": 0.665,
            "x_reat_xc": 1.10,
            "cf_min": -0.00146,
        },
    },
    "backward_facing_step": {
        "experiment": "Driver & Seegmiller (1985)",
        "title": "Features of a Reattaching Turbulent Shear Layer",
        "journal": "AIAA J. 23(2), pp.163-171, DOI:10.2514/3.8890",
        "measurements": ["Cf", "U profiles", "u'v' Reynolds stress"],
        "profile_stations_xH": [1, 4, 6, 10],
        "key_metrics": {
            "x_reat_xH": 6.26,
            "x_reat_uncertainty": 0.10,
        },
    },
    "naca_0012": {
        "experiment": "Gregory & O'Reilly (1970) / Ladson (1988)",
        "title": "Low-Speed Aerodynamic Characteristics of NACA 0012",
        "journal": "ARC R&M 3726 / NASA TM-4074",
        "measurements": ["CL(alpha)", "CD(alpha)", "Cp distributions"],
        "alpha_deg_available": [0, 2, 4, 6, 8, 10, 12, 14, 15, 16, 18, 20],
        "key_metrics": {
            "CL_max": 1.55,
            "alpha_stall_deg": 16,
        },
    },
    "swbli_schulein": {
        "experiment": "Schulein (2006)",
        "title": "Skin Friction and Heat Transfer Measurements in SWBLI",
        "journal": "AIAA J. 44(8), DOI:10.2514/1.18029",
        "measurements": ["Cf upstream/downstream of shock", "wall pressure"],
        "key_metrics": {
            "M_freestream": 5.0,
            "shock_angle_deg": 14.0,
        },
    },
    "flat_plate": {
        "experiment": "Wieghardt & Tillmann (1951) / Coles (1962)",
        "title": "ZPG turbulent boundary layer",
        "journal": "NACA TM-1314",
        "measurements": ["Cf(x)", "U+(y+) log-law"],
        "key_metrics": {
            "kappa": 0.41,
            "B": 5.0,
        },
    },
}


# =============================================================================
# Validation Criteria
# =============================================================================
VALIDATION_CRITERIA = {
    "gci_threshold": 0.05,               # GCI < 5% for grid independence
    "mape_target": 15.0,                  # < 15% MAPE vs experiment
    "asme_vv20_threshold": 1.0,           # E/U_exp < 1.0 (validated)
    "combined_confidence": 0.95,          # 95% CI
    "mrr_levels": {
        1: "Verified on unit problems",
        2: "Validated on simple canonical cases",
        3: "Validated on multiple benchmark cases",
        4: "Applied to complex 3D configuration",
    },
}


# =============================================================================
# Target Journals & Conferences
# =============================================================================
TARGET_PUBLICATIONS = {
    "journals": [
        {"name": "Computers & Fluids", "publisher": "Elsevier", "IF": 3.0},
        {"name": "JVVUQ (ASME)", "publisher": "ASME", "IF": None},
        {"name": "Flow, Turbulence and Combustion", "publisher": "Springer", "IF": 2.0},
        {"name": "Int. J. Numer. Methods Fluids", "publisher": "Wiley", "IF": 1.7},
    ],
    "conferences": [
        "AIAA SciTech Forum",
        "ECCOMAS",
        "ICCFD",
        "DLRK (German Aerospace Congress)",
    ],
}


# =============================================================================
# Utility Functions
# =============================================================================
def get_tier_cases(tier: CaseTier) -> Dict[str, BenchmarkCase]:
    """Return all benchmark cases for a given tier."""
    return {k: v for k, v in BENCHMARK_CASES.items() if v.tier == tier}


def get_category_cases(category: SeparationCategory) -> Dict[str, BenchmarkCase]:
    """Return all benchmark cases for a given separation category."""
    return {k: v for k, v in BENCHMARK_CASES.items() if v.category == category}


def get_model(abbreviation: str) -> Optional[TurbulenceModel]:
    """Look up a turbulence model by abbreviation."""
    return TURBULENCE_MODELS.get(abbreviation)


def list_cases_summary() -> str:
    """Print a summary table of all benchmark cases."""
    lines = [f"{'Tier':<5} {'Key':<25} {'Name':<45} {'Re':<12} {'Data Source'}"]
    lines.append("-" * 110)
    for key, case in BENCHMARK_CASES.items():
        lines.append(
            f"{case.tier.name:<5} {key:<25} {case.name:<45} "
            f"{case.reynolds_number:<12.0f} {case.data_source}"
        )
    return "\n".join(lines)


def list_models_summary() -> str:
    """Print a summary table of all turbulence models."""
    lines = [f"{'Type':<10} {'Abbreviation':<15} {'Name'}"]
    lines.append("-" * 70)
    for key, model in TURBULENCE_MODELS.items():
        lines.append(f"{model.model_type.name:<10} {key:<15} {model.name}")
    return "\n".join(lines)

# =============================================================================
# ML Augmentation Modules Registry
# =============================================================================
ML_MODULES: Dict[str, Dict[str, str]] = {
    "conformal_prediction": {
        "description": "Distribution-free uncertainty quantification (Split CP, CQR, Jackknife+)",
        "path": "scripts/ml_augmentation/conformal_prediction.py",
    },
    "generative_super_resolution": {
        "description": "Physics-informed generative models (TBNN-conditioned diffusion) for RANS-to-LES mapping",
        "path": "scripts/ml_augmentation/generative_super_resolution.py",
    },
    "gnn_mesh_adaptation": {
        "description": "Adaptnet extensions: MeshnetCAD and GraphnetHessian metric tensor prediction",
        "path": "scripts/ml_augmentation/gnn_mesh_adaptation.py",
    },
    "drl_flow_control": {
        "description": "MARL extensions for active flow control with ZNMF actuators and zero-shot transfer",
        "path": "scripts/ml_augmentation/drl_flow_control.py",
    },
    "smartsim_orchestrator": {
        "description": "In-situ data orchestration and online inference utilizing Redis/SmartSim",
        "path": "scripts/ml_augmentation/smartsim_orchestrator.py",
    },
    "llm_turbulence_closure": {
        "description": "LLM-driven algebraic turbulence closure discovery and code injection",
        "path": "scripts/ml_augmentation/llm_turbulence_closure.py",
    },
    "physics_foundation_model": {
        "description": "General Physics Transformer (GPhyT) architecture for zero-shot aerodynamic prediction",
        "path": "scripts/ml_augmentation/physics_foundation_model.py",
    },
}

def get_ml_module_count() -> int:
    ml_dir = PROJECT_ROOT / "scripts" / "ml_augmentation"
    if ml_dir.exists():
        # Exclude __init__.py and similar infra scripts
        files = [f for f in ml_dir.glob("*.py") if not f.name.startswith("__")]
        return len(files)
    return len(ML_MODULES)

if __name__ == "__main__":
    print("=" * 60)
    print("CFD Solver Benchmark Configuration")
    print("=" * 60)
    print(f"\nBenchmark Cases: {len(BENCHMARK_CASES)}")
    print(f"Turbulence Models: {len(TURBULENCE_MODELS)}")
    print(f"ML Modules (Actual Files): {get_ml_module_count()}")
    print(f"ML Modules (Core Registry): {len(ML_MODULES)}")
    print(f"\n{list_cases_summary()}")
    print(f"\n{list_models_summary()}")
