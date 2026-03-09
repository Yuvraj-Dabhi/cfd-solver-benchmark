#!/usr/bin/env python3
"""
Bradshaw 1996 NASA Collaborative Testing Archive Loader
=========================================================
Generic loader for the Bradshaw (1996) NASA Collaborative Testing
of Turbulence Models archive. Provides a catalog of available cases
and parsers for the archived experimental data.

References
----------
  - Bradshaw (1996), NASA CR-198237
  - https://turbmodels.larc.nasa.gov/bradshaw.html
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class BradshawCase:
    """A case from the Bradshaw archive."""
    code: str
    name: str
    description: str
    flow_type: str
    Re: float = 0
    data_url: str = ""
    available_quantities: List[str] = field(default_factory=list)
    reference: str = ""


# Complete catalog of Bradshaw archive cases
BRADSHAW_CATALOG: Dict[str, BradshawCase] = {
    "KIM_BFS": BradshawCase(
        code="KIM_BFS",
        name="Kim et al. Backward-Facing Step",
        description="Backward-facing step at Re_H=132,000, expansion ratio 1.2",
        flow_type="geometric_separation",
        Re=132_000,
        available_quantities=["Cp", "Cf", "U", "uu", "vv", "uv"],
        reference="Kim, Kline & Johnston (1980), Report MD-37",
    ),
    "DRIVER_BFS": BradshawCase(
        code="DRIVER_BFS",
        name="Driver & Seegmiller Backward-Facing Step",
        description="BFS at Re_H=36,000, exp ratio 1.125, 0° and 6° divergence",
        flow_type="geometric_separation",
        Re=36_000,
        available_quantities=["Cp", "Cf", "U", "uu", "vv", "uv"],
        reference="Driver & Seegmiller (1985), AIAA J. 23(2)",
    ),
    "NACA_4412_TE": BradshawCase(
        code="NACA_4412_TE",
        name="NACA 4412 Trailing-Edge Separation",
        description="Coles & Wadcock, alpha=13.87°, Re_c=1.52M, TE separation",
        flow_type="smooth_body_separation",
        Re=1.52e6,
        available_quantities=["Cp", "U", "uu", "vv", "uv"],
        reference="Coles & Wadcock (1979), AIAA Paper 79-1457",
    ),
    "SAMUEL_JOUBERT": BradshawCase(
        code="SAMUEL_JOUBERT",
        name="Samuel & Joubert APG Boundary Layer",
        description="Adverse pressure gradient TBL, Re_theta~2000-5000",
        flow_type="apg_boundary_layer",
        Re=5000,
        available_quantities=["U", "uu", "vv", "uv", "Cf"],
        reference="Samuel & Joubert (1974), J. Fluid Mech. 66(3)",
    ),
    "SPALART_1340": BradshawCase(
        code="SPALART_1340",
        name="Spalart DNS ZPG Boundary Layer",
        description="Direct numerical simulation at Re_theta=1410",
        flow_type="zpg_boundary_layer",
        Re=1410,
        available_quantities=["U", "uu", "vv", "ww", "uv", "Cf", "budget"],
        reference="Spalart (1988), J. Fluid Mech. 187",
    ),
    "DUCT_FLOW": BradshawCase(
        code="DUCT_FLOW",
        name="Fully Developed Duct Flow",
        description="Square duct flow, secondary flow driven by Reynolds stress anisotropy",
        flow_type="duct_flow",
        Re=83_000,
        available_quantities=["U", "W", "uu", "vv", "ww", "uv", "uw"],
        reference="Brundrett & Baines (1964), J. Fluid Mech. 19",
    ),
    "MIXING_LAYER": BradshawCase(
        code="MIXING_LAYER",
        name="Plane Mixing Layer",
        description="Two-stream mixing layer, velocity ratio 0.6",
        flow_type="free_shear",
        Re=0,
        available_quantities=["U", "uu", "vv", "uv"],
        reference="Bell & Mehta (1990), AIAA J. 28(12)",
    ),
    "CURVED_CHANNEL": BradshawCase(
        code="CURVED_CHANNEL",
        name="Curved Channel Flow",
        description="U-bend channel with curvature-driven secondary flow",
        flow_type="curvature",
        Re=50_000,
        available_quantities=["U", "V", "uu", "vv", "uv", "Cp"],
        reference="Hunt & Joubert (1979), J. Fluid Mech. 91",
    ),
    "PIPE_EXPANSION": BradshawCase(
        code="PIPE_EXPANSION",
        name="Axisymmetric Sudden Expansion",
        description="Pipe expansion at Re_D=45,000",
        flow_type="geometric_separation",
        Re=45_000,
        available_quantities=["U", "Cp", "Cf", "uu", "vv"],
        reference="Durrett et al. (1988), AIAA J. 26(8)",
    ),
}


def list_available_cases() -> Dict[str, str]:
    """
    List all available cases in the Bradshaw archive.

    Returns
    -------
    dict mapping case code → description.
    """
    return {code: case.name for code, case in BRADSHAW_CATALOG.items()}


def get_case_info(case_code: str) -> Dict:
    """
    Get detailed information about a specific case.

    Parameters
    ----------
    case_code : str
        Case identifier (e.g., 'KIM_BFS', 'NACA_4412_TE').

    Returns
    -------
    dict with case metadata.
    """
    if case_code not in BRADSHAW_CATALOG:
        raise ValueError(
            f"Unknown case: {case_code}. Available: {list(BRADSHAW_CATALOG.keys())}"
        )

    case = BRADSHAW_CATALOG[case_code]
    return {
        "code": case.code,
        "name": case.name,
        "description": case.description,
        "flow_type": case.flow_type,
        "Re": case.Re,
        "available_quantities": case.available_quantities,
        "reference": case.reference,
    }


def load_bradshaw_case(case_code: str):
    """
    Load data for a Bradshaw archive case.

    Delegates to specialized loaders for cases that have
    full data implementations (Kim BFS, NACA 4412 TE).

    Parameters
    ----------
    case_code : str
        Case identifier.

    Returns
    -------
    CaseData or dict with available data.
    """
    if case_code not in BRADSHAW_CATALOG:
        raise ValueError(
            f"Unknown case: {case_code}. Available: {list(BRADSHAW_CATALOG.keys())}"
        )

    # Delegate to specialized loaders when available
    if case_code == "KIM_BFS":
        from experimental_data.bfs_kim.bfs_kim_loader import load_kim_bfs_data
        return load_kim_bfs_data()

    elif case_code == "NACA_4412_TE":
        from experimental_data.naca4412_wadcock.naca4412_wadcock_loader import (
            load_naca4412_data,
        )
        return load_naca4412_data()

    else:
        # Return metadata for cases without full data implementation yet
        return get_case_info(case_code)


if __name__ == "__main__":
    print("=" * 60)
    print("Bradshaw 1996 NASA Collaborative Testing Archive")
    print("=" * 60)

    cases = list_available_cases()
    for code, name in cases.items():
        info = get_case_info(code)
        print(f"\n  [{code}] {name}")
        print(f"    Flow type: {info['flow_type']}")
        print(f"    Re: {info['Re']}")
        print(f"    Reference: {info['reference']}")
        print(f"    Quantities: {', '.join(info['available_quantities'])}")
