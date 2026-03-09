"""
Experimental Profile Reference Registry
========================================
Station-by-station reference data for canonical benchmark cases.
Maps each case to its experimental profile stations, data sources,
and key reference values for direct comparison.

Usage:
    from scripts.preprocessing.experimental_profiles import (
        get_profile_stations, get_reference_data
    )
    stations = get_profile_stations("nasa_hump")
    refdata = get_reference_data("backward_facing_step", "x/H=6")
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class ExperimentalProfile:
    """One experimental profile station."""
    station_label: str         # e.g., "x/c = 0.65"
    location_value: float      # e.g., 0.65
    location_type: str         # e.g., "x/c", "x/H"
    quantities: List[str]      # e.g., ["U/U_inf", "u'v'/U_inf^2"]
    reference: str             # Citation
    notes: str = ""


@dataclass
class CaseProfileRegistry:
    """All experimental profiles for one case."""
    case_name: str
    experiment: str
    profiles: List[ExperimentalProfile]
    flow_conditions: Dict[str, Any] = field(default_factory=dict)
    separation_metrics: Dict[str, float] = field(default_factory=dict)


# =============================================================================
# Profile Station Registry
# =============================================================================
PROFILE_REGISTRY: Dict[str, CaseProfileRegistry] = {
    "nasa_hump": CaseProfileRegistry(
        case_name="nasa_hump",
        experiment="Greenblatt et al. (2006), AIAA J. 44(1)",
        flow_conditions={"Re_c": 936000, "M": 0.1, "U_ref": 34.6},
        separation_metrics={"x_sep_xc": 0.665, "x_reat_xc": 1.11},
        profiles=[
            ExperimentalProfile(
                station_label=f"x/c = {loc:.2f}",
                location_value=loc,
                location_type="x/c",
                quantities=["U/U_inf", "V/U_inf", "u'u'/U_inf^2", "v'v'/U_inf^2", "u'v'/U_inf^2"],
                reference="Greenblatt et al. (2006), PIV measurements",
                notes="Inside separation bubble" if 0.665 < loc < 1.11 else "Recovery region" if loc > 1.11 else "Pre-separation",
            )
            for loc in [0.65, 0.80, 0.90, 1.00, 1.10, 1.20, 1.30]
        ],
    ),

    "backward_facing_step": CaseProfileRegistry(
        case_name="backward_facing_step",
        experiment="Driver & Seegmiller (1985), AIAA J. 23(2)",
        flow_conditions={"Re_h": 37400, "U_ref": 44.2, "step_height": 0.0127},
        separation_metrics={"x_reat_xH": 6.26, "uncertainty": 0.10},
        profiles=[
            ExperimentalProfile(
                station_label=f"x/H = {loc}",
                location_value=loc,
                location_type="x/H",
                quantities=["U/U_ref", "u'u'/U_ref^2", "v'v'/U_ref^2", "u'v'/U_ref^2"],
                reference="Driver & Seegmiller (1985), LDV measurements",
                notes="In recirculation zone" if loc < 6.26 else "Recovery region",
            )
            for loc in [1, 4, 6, 10]
        ],
    ),

    "naca_0012": CaseProfileRegistry(
        case_name="naca_0012",
        experiment="Gregory & O'Reilly (1970) / Ladson (1988)",
        flow_conditions={"Re": 6e6, "M": 0.15},
        separation_metrics={"CL_max": 1.55, "alpha_stall": 16},
        profiles=[
            ExperimentalProfile(
                station_label=f"alpha = {alpha}°",
                location_value=alpha,
                location_type="alpha_deg",
                quantities=["Cp(x/c)"],
                reference="Gregory & O'Reilly (1970), pressure taps",
                notes="Near stall" if alpha > 12 else "Linear CL range",
            )
            for alpha in [0, 4, 8, 10, 12, 15]
        ],
    ),

    "swbli_schulein": CaseProfileRegistry(
        case_name="swbli_schulein",
        experiment="Schulein (2006), AIAA J. 44(8)",
        flow_conditions={"M": 5.0, "Re_per_m": 3.7e7, "shock_angle": 14.0},
        separation_metrics={"x_sep_interaction_lengths": 3.5},
        profiles=[
            ExperimentalProfile(
                station_label="upstream of shock",
                location_value=-3.0,
                location_type="x/delta",
                quantities=["Cf", "p_w/p_inf"],
                reference="Schulein (2006), oil-film + pressure taps",
                notes="Undisturbed boundary layer",
            ),
            ExperimentalProfile(
                station_label="separation region",
                location_value=0.0,
                location_type="x/delta",
                quantities=["Cf", "p_w/p_inf"],
                reference="Schulein (2006)",
                notes="Interaction zone — RANS models differ most here",
            ),
            ExperimentalProfile(
                station_label="downstream recovery",
                location_value=5.0,
                location_type="x/delta",
                quantities=["Cf", "p_w/p_inf"],
                reference="Schulein (2006)",
                notes="Post-reattachment recovery",
            ),
        ],
    ),

    "flat_plate": CaseProfileRegistry(
        case_name="flat_plate",
        experiment="Wieghardt & Tillmann (1951) / Coles (1962)",
        flow_conditions={"Re_L": 1e7},
        separation_metrics={},
        profiles=[
            ExperimentalProfile(
                station_label=f"Re_x = {rex:.0e}",
                location_value=rex,
                location_type="Re_x",
                quantities=["U+(y+)", "Cf"],
                reference="Coles (1962), NACA TM-1314",
                notes="Log law: U⁺ = (1/κ)ln(y⁺) + B, κ=0.41, B=5.0",
            )
            for rex in [1e5, 5e5, 1e6, 5e6]
        ],
    ),
}


def get_profile_stations(case_name: str) -> CaseProfileRegistry:
    """Get experimental profile stations for a given case."""
    if case_name not in PROFILE_REGISTRY:
        available = list(PROFILE_REGISTRY.keys())
        raise ValueError(f"Unknown case: {case_name}. Available: {available}")
    return PROFILE_REGISTRY[case_name]


def get_reference_data(case_name: str, station_label: str) -> Optional[ExperimentalProfile]:
    """Get specific experimental profile by station label."""
    registry = get_profile_stations(case_name)
    for profile in registry.profiles:
        if profile.station_label == station_label:
            return profile
    return None


def list_all_profiles() -> Dict[str, List[str]]:
    """List all cases and their profile stations."""
    result = {}
    for case_name, registry in PROFILE_REGISTRY.items():
        result[case_name] = [p.station_label for p in registry.profiles]
    return result
