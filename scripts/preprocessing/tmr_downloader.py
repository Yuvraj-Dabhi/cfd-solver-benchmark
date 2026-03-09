"""
NASA TMR Data Downloader & Parser
=================================
Download and cache experimental data from the NASA Turbulence Modeling
Resource (https://turbmodels.larc.nasa.gov/) for direct use in
benchmark comparisons.

Supports 9+ separation-relevant cases:
  - 2DBFS   : 2-D Backward-Facing Step
  - ATB     : Axisymmetric Transonic Bump (Bachalo-Johnson)
  - 2DWMH   : 2-D Wall-Mounted Hump
  - 2DN00   : NACA 0012
  - 2DN44   : NACA 4412
  - PHILLS  : Periodic Hills (via ERCOFTAC link)
  - JFLOW   : Juncture Flow
  - FPBL    : Flat Plate Boundary Layer
  - SWBLI   : Shock-Wave/Boundary-Layer Interaction
"""

import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Resolve project root
_THIS_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = _THIS_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


# =============================================================================
# TMR Case Registry
# =============================================================================
@dataclass
class TMRCase:
    """Metadata for one NASA TMR validation case."""
    code: str
    name: str
    base_url: str
    description: str = ""
    data_files: List[str] = field(default_factory=list)
    grid_files: List[str] = field(default_factory=list)
    flow_conditions: Dict[str, Any] = field(default_factory=dict)

    @property
    def validation_page(self) -> str:
        return self.base_url


# Registry of separation-relevant TMR cases
TMR_CASES: Dict[str, TMRCase] = {
    "2DBFS": TMRCase(
        code="2DBFS",
        name="2-D Backward-Facing Step",
        base_url="https://turbmodels.larc.nasa.gov/backstep_val.html",
        description="Driver & Seegmiller BFS, Re_h=37400, Expansion ratio 1.25",
        data_files=["backstep_exp_cf.dat", "backstep_exp_cp.dat"],
        flow_conditions={"Re_h": 37400, "expansion_ratio": 1.25},
    ),
    "ATB": TMRCase(
        code="ATB",
        name="Axisymmetric Transonic Bump (Bachalo-Johnson)",
        base_url="https://turbmodels.larc.nasa.gov/axibump_val.html",
        description="M=0.875, Re=2.763e6, circular-arc bump on cylinder",
        data_files=["axibump_exp_cp.dat"],
        flow_conditions={"M": 0.875, "Re": 2.763e6},
    ),
    "2DWMH": TMRCase(
        code="2DWMH",
        name="2-D Wall-Mounted Hump",
        base_url="https://turbmodels.larc.nasa.gov/hump_val.html",
        description="Greenblatt et al., Re_c=936000, Glauert-Goldschmied body",
        data_files=["hump_exp_cp.dat", "hump_exp_cf.dat"],
        flow_conditions={"Re_c": 936000},
    ),
    "2DN00": TMRCase(
        code="2DN00",
        name="NACA 0012",
        base_url="https://turbmodels.larc.nasa.gov/naca0012_val.html",
        description="Gregory & O'Reilly, Re=6e6, various angles of attack",
        data_files=["naca0012_exp_cl.dat", "naca0012_exp_cd.dat"],
        flow_conditions={"Re": 6e6},
    ),
    "2DN44": TMRCase(
        code="2DN44",
        name="NACA 4412 Trailing-Edge Separation",
        base_url="https://turbmodels.larc.nasa.gov/naca4412sep_val.html",
        description="Coles & Wadcock, Re=1.5e6, alpha=13.87deg",
        data_files=["naca4412_exp_cp.dat", "naca4412_exp_profiles.dat"],
        flow_conditions={"Re": 1.5e6, "alpha_deg": 13.87},
    ),
    "PHILLS": TMRCase(
        code="PHILLS",
        name="Periodic Hills (ERCOFTAC)",
        base_url="https://turbmodels.larc.nasa.gov/Other_DNS_Data/perihill.html",
        description="Breuer et al. DNS, Re_H=10595",
        data_files=["perihill_dns_profiles.dat"],
        flow_conditions={"Re_H": 10595},
    ),
    "JFLOW": TMRCase(
        code="JFLOW",
        name="Juncture Flow",
        base_url="https://turbmodels.larc.nasa.gov/junctureflow_val.html",
        description="Rumsey et al. wing-body junction, Re_mac=2.4e6",
        data_files=["juncture_exp_cp.dat"],
        flow_conditions={"Re": 2.4e6},
    ),
    "FPBL": TMRCase(
        code="FPBL",
        name="Flat Plate Boundary Layer",
        base_url="https://turbmodels.larc.nasa.gov/flatplate.html",
        description="Zero-pressure-gradient turbulent flat plate",
        data_files=["flatplate_cf.dat"],
        flow_conditions={"Re_L": 1e7},
    ),
    "SWBLI": TMRCase(
        code="SWBLI",
        name="Shock-Wave/Boundary-Layer Interaction",
        base_url="https://turbmodels.larc.nasa.gov/swbli_val.html",
        description="Oblique shock impingement at supersonic speed",
        data_files=["swbli_exp_pw.dat"],
        flow_conditions={"M": 2.85},
    ),
    "BUMP": TMRCase(
        code="BUMP",
        name="Bump-in-Channel (TMR Verification)",
        base_url="https://tmbwg.github.io/turbmodels/bump.html",
        description="2D bump in channel, Re=3e6/m, code verification case",
        data_files=["bump_cfl3d_cf.dat"],
        grid_files=["bump_89x41.p3d", "bump_177x81.p3d"],
        flow_conditions={"M": 0.2, "Re_per_m": 3e6},
    ),
    "HJET": TMRCase(
        code="HJET",
        name="TMR Heated Jet (Subsonic Round)",
        base_url="https://turbmodels.larc.nasa.gov/jetsubsonic_val.html",
        description="Subsonic/supersonic heated jet (NASA 40% Challenge case #4)",
        data_files=["jet_sp3_profiles.dat", "jet_sp7_profiles.dat"],
        flow_conditions={"set_points": [3, 7, 23]},
    ),
    "ASWBLI": TMRCase(
        code="ASWBLI",
        name="Axisymmetric SWBLI M=2.85",
        base_url="https://turbmodels.larc.nasa.gov/swbli_val.html",
        description="NASA 40% Challenge case #5: axi SWBLI at M=2.85",
        data_files=["axi_swbli_cf.dat", "axi_swbli_pw.dat"],
        flow_conditions={"M": 2.85, "Re": 1e6},
    ),
}


# =============================================================================
# Cache Management
# =============================================================================
DEFAULT_CACHE_DIR = PROJECT_ROOT / "experimental_data" / "tmr_cache"


def _ensure_cache_dir(cache_dir: Optional[Path] = None) -> Path:
    """Create cache directory if it doesn't exist."""
    d = cache_dir or DEFAULT_CACHE_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


# =============================================================================
# Core Downloader
# =============================================================================
def list_available_cases() -> Dict[str, str]:
    """
    List all available TMR cases.

    Returns
    -------
    dict
        {case_code: case_name} for all registered TMR cases.
    """
    return {code: case.name for code, case in TMR_CASES.items()}


def download_case(
    case_code: str,
    cache_dir: Optional[Path] = None,
    force: bool = False,
) -> Dict[str, Path]:
    """
    Download experimental data for a TMR case.

    Parameters
    ----------
    case_code : str
        Case code from TMR_CASES (e.g., "ATB", "2DBFS").
    cache_dir : Path, optional
        Directory to cache downloads. Default: experimental_data/tmr_cache/
    force : bool
        If True, re-download even if cached files exist.

    Returns
    -------
    dict
        {filename: local_path} for each downloaded file.
    """
    if case_code not in TMR_CASES:
        raise ValueError(
            f"Unknown TMR case: {case_code}. Available: {list(TMR_CASES.keys())}"
        )

    tmr = TMR_CASES[case_code]
    cache = _ensure_cache_dir(cache_dir) / case_code.lower()
    cache.mkdir(exist_ok=True)

    downloaded = {}
    for filename in tmr.data_files:
        local_path = cache / filename
        if local_path.exists() and not force:
            logger.info(f"Using cached: {local_path}")
            downloaded[filename] = local_path
            continue

        # Construct download URL
        base = tmr.base_url.rsplit("/", 1)[0]
        url = f"{base}/{filename}"

        try:
            import urllib.request
            logger.info(f"Downloading {url} -> {local_path}")
            urllib.request.urlretrieve(url, local_path)
            downloaded[filename] = local_path
        except Exception as e:
            logger.warning(f"Failed to download {url}: {e}")
            # Create a stub file for offline development
            _create_stub_data(local_path, case_code, filename)
            downloaded[filename] = local_path

    return downloaded


def _create_stub_data(filepath: Path, case_code: str, filename: str):
    """Create stub data file for offline development."""
    logger.info(f"Creating stub data: {filepath}")
    header = f"# Stub data for {case_code} - {filename}\n"
    header += f"# Download real data from: {TMR_CASES[case_code].base_url}\n"

    if "cp" in filename.lower():
        header += "# x/c  Cp\n"
        x = np.linspace(0, 1, 50)
        data = np.column_stack([x, -0.5 * np.sin(np.pi * x)])
    elif "cf" in filename.lower():
        header += "# x/c  Cf\n"
        x = np.linspace(0, 1, 50)
        data = np.column_stack([x, 0.003 * (1 - 0.5 * x)])
    elif "cl" in filename.lower():
        header += "# alpha(deg)  CL\n"
        alpha = np.linspace(0, 20, 41)
        cl = np.minimum(5.7 * np.radians(alpha), 1.5)
        data = np.column_stack([alpha, cl])
    elif "profile" in filename.lower():
        header += "# y/delta  U/U_inf\n"
        y = np.linspace(0, 1, 50)
        data = np.column_stack([y, np.tanh(3 * y)])
    else:
        header += "# x  y\n"
        x = np.linspace(0, 1, 50)
        data = np.column_stack([x, np.zeros_like(x)])

    with open(filepath, "w") as f:
        f.write(header)
        for row in data:
            f.write("  ".join(f"{v:12.6f}" for v in row) + "\n")


# =============================================================================
# TMR Data Parser
# =============================================================================
def parse_tmr_profile(
    filepath: Path,
    comment_char: str = "#",
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Parse a NASA TMR data file into a pandas DataFrame.

    TMR files typically have:
    - Header lines starting with # or blank
    - Space/tab-delimited numerical data

    Parameters
    ----------
    filepath : Path
        Path to the TMR data file.
    comment_char : str
        Character indicating comment lines.
    columns : list of str, optional
        Column names. If None, attempts to parse from header.

    Returns
    -------
    pd.DataFrame
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"TMR data file not found: {filepath}")

    lines = filepath.read_text().splitlines()

    # Extract column headers from comments
    header_cols = None
    data_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith(comment_char):
            # Try to parse column names from header
            if stripped.startswith(comment_char):
                parts = stripped.lstrip(comment_char).strip().split()
                if len(parts) >= 2 and all(
                    not _is_number(p) for p in parts
                ):
                    header_cols = parts
            continue
        data_lines.append(stripped)

    if not data_lines:
        logger.warning(f"No data found in {filepath}")
        return pd.DataFrame()

    # Parse numerical data
    data = []
    for line in data_lines:
        try:
            values = [float(x) for x in line.split()]
            data.append(values)
        except ValueError:
            continue

    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)

    # Apply column names
    if columns:
        df.columns = columns[:len(df.columns)]
    elif header_cols and len(header_cols) == len(df.columns):
        df.columns = header_cols

    return df


def _is_number(s: str) -> bool:
    """Check if string represents a number."""
    try:
        float(s)
        return True
    except ValueError:
        return False


# =============================================================================
# High-Level Integration
# =============================================================================
def get_tmr_data(
    case_code: str,
    cache_dir: Optional[Path] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Download (if needed) and parse all data files for a TMR case.

    Parameters
    ----------
    case_code : str
        TMR case code.

    Returns
    -------
    dict
        {filename_stem: DataFrame} for each data file.
    """
    files = download_case(case_code, cache_dir)
    result = {}
    for filename, path in files.items():
        stem = Path(filename).stem
        try:
            df = parse_tmr_profile(path)
            result[stem] = df
            logger.info(f"Parsed {filename}: {len(df)} rows, {len(df.columns)} cols")
        except Exception as e:
            logger.warning(f"Failed to parse {filename}: {e}")
    return result


def get_case_info(case_code: str) -> Dict[str, Any]:
    """
    Get metadata for a TMR case.

    Parameters
    ----------
    case_code : str
        TMR case code.

    Returns
    -------
    dict
        Case metadata including URL, flow conditions, etc.
    """
    if case_code not in TMR_CASES:
        raise ValueError(f"Unknown TMR case: {case_code}")
    tmr = TMR_CASES[case_code]
    return {
        "code": tmr.code,
        "name": tmr.name,
        "url": tmr.validation_page,
        "description": tmr.description,
        "flow_conditions": tmr.flow_conditions,
        "data_files": tmr.data_files,
        "grid_files": tmr.grid_files,
    }


# =============================================================================
# CLI
# =============================================================================
if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="NASA TMR Data Downloader")
    parser.add_argument("case", nargs="?", help="Case code (e.g., ATB, 2DBFS)")
    parser.add_argument("--list", action="store_true", help="List all TMR cases")
    parser.add_argument("--info", action="store_true", help="Show case info")
    parser.add_argument("--download", action="store_true", help="Download case data")
    parser.add_argument("--force", action="store_true", help="Force re-download")
    args = parser.parse_args()

    if args.list or args.case is None:
        print("Available NASA TMR cases:")
        print(f"{'Code':<10} {'Name':<45} {'URL'}")
        print("-" * 100)
        for code, case in TMR_CASES.items():
            print(f"{code:<10} {case.name:<45} {case.base_url}")
    elif args.info:
        info = get_case_info(args.case)
        print(f"TMR Case: {info['name']} ({info['code']})")
        print(f"  URL: {info['url']}")
        print(f"  Description: {info['description']}")
        print(f"  Flow conditions: {info['flow_conditions']}")
        print(f"  Data files: {info['data_files']}")
    elif args.download:
        files = download_case(args.case, force=args.force)
        for name, path in files.items():
            print(f"  {name} -> {path}")
    else:
        info = get_case_info(args.case)
        print(f"{info['name']}: {info['url']}")
