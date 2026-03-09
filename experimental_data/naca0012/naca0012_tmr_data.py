#!/usr/bin/env python3
"""
NACA 0012 TMR Data Downloader & Parser
=======================================
Downloads all experimental and CFD reference data from NASA Langley's
Turbulence Modeling Resource (TMR) for the 2D NACA 0012 validation case.

Source: https://turbmodels.larc.nasa.gov/naca0012_val.html

Case conditions:
    Mach  = 0.15 (essentially incompressible)
    Re    = 6 × 10^6 (per chord)
    Fully turbulent, sharp trailing edge

Experimental sources:
    - Ladson (1988): Tripped, Re=6M, CL & CD (best for fully-turbulent CFD)
    - Gregory & O'Reilly (1970): Tripped, Re=3M, CL & Cp
    - Abbott & von Doenhoff (1959): Un-tripped, Re=6M, CL & CD
    - McCroskey (1987): CL-alpha best fit line

CFD reference:
    - CFL3D SA model on 897×257 grid (7-code consensus within 1% CL, 4% CD)

Usage:
    python naca0012_tmr_data.py              # Download all data
    python naca0012_tmr_data.py --info       # Show data summary
"""

import sys
import os
import urllib.request
import urllib.error
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Force UTF-8 output
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ============================================================================
# TMR Data File Registry
# ============================================================================

TMR_BASE = "https://turbmodels.larc.nasa.gov/NACA0012_validation"

TMR_FILES = {
    # Experimental force data
    "ladson_forces": {
        "url": f"{TMR_BASE}/CLCD_Ladson_expdata.dat",
        "description": "Ladson tripped CL & CD vs alpha (Re=6M) — PRIMARY validation",
        "type": "experimental",
    },
    "abbott_cl": {
        "url": f"{TMR_BASE}/0012.abbottdata.cl.dat",
        "description": "Abbott & von Doenhoff CL vs alpha (un-tripped, Re=6M)",
        "type": "experimental",
    },
    "abbott_cd": {
        "url": f"{TMR_BASE}/0012.abbottdata.cd.dat",
        "description": "Abbott & von Doenhoff CD vs CL (un-tripped, Re=6M)",
        "type": "experimental",
    },
    "gregory_cl": {
        "url": f"{TMR_BASE}/CL_Gregory_expdata.dat",
        "description": "Gregory & O'Reilly CL vs alpha (tripped, Re=3M)",
        "type": "experimental",
    },
    "mccroskey_cl": {
        "url": f"{TMR_BASE}/0012.mccroskeydata.cl.dat",
        "description": "McCroskey CL-alpha best fit line",
        "type": "experimental",
    },
    # Experimental surface data
    "gregory_cp": {
        "url": f"{TMR_BASE}/CP_Gregory_expdata.dat",
        "description": "Gregory & O'Reilly Cp vs x/c (Re=3M) — best resolved Cp",
        "type": "experimental",
    },
    "ladson_cp": {
        "url": f"{TMR_BASE}/CP_Ladson.dat",
        "description": "Ladson et al Cp vs x/c (Re=6M) — LE peak less resolved",
        "type": "experimental",
    },
    # CFL3D SA reference (7-code consensus)
    "cfl3d_sa_forces": {
        "url": f"{TMR_BASE}/n0012clcd_cfl3d_sa.dat",
        "description": "CFL3D SA: CL & CD vs alpha (897x257 grid)",
        "type": "cfd_reference",
    },
    "cfl3d_sa_cp": {
        "url": f"{TMR_BASE}/n0012cp_cfl3d_sa.dat",
        "description": "CFL3D SA: Cp vs x/c at multiple alpha",
        "type": "cfd_reference",
    },
    "cfl3d_sa_cf": {
        "url": f"{TMR_BASE}/n0012cf_cfl3d_sa.dat",
        "description": "CFL3D SA: Cf vs x/c at multiple alpha",
        "type": "cfd_reference",
    },
}

# TMR grid files
TMR_GRID_BASE = "https://turbmodels.larc.nasa.gov/NACA0012_grids"

TMR_GRIDS = {
    "coarse_113x33": {
        "url": f"{TMR_GRID_BASE}/n0012_113-33.p2dfmt",
        "dims": (113, 33),
        "airfoil_pts": 65,
        "compressed": False,
    },
    "medium_225x65": {
        "url": f"{TMR_GRID_BASE}/n0012_225-65.p2dfmt",
        "dims": (225, 65),
        "airfoil_pts": 129,
        "compressed": False,
    },
    "fine_449x129": {
        "url": f"{TMR_GRID_BASE}/n0012_449-129.p2dfmt.gz",
        "dims": (449, 129),
        "airfoil_pts": 257,
        "compressed": True,
    },
    "xfine_897x257": {
        "url": f"{TMR_GRID_BASE}/n0012_897-257.p2dfmt.gz",
        "dims": (897, 257),
        "airfoil_pts": 513,
        "compressed": True,
    },
}

# Airfoil surface points file
TMR_AIRFOIL_POINTS = {
    "url": f"{TMR_GRID_BASE}/n0012points_superbig_clust_fix.dat",
    "description": "NACA 0012 surface coordinates (sharp TE, scaled)",
}

# ============================================================================
# TMR Reference Values (SA model, 7-code consensus)
# ============================================================================

TMR_SA_REFERENCE = {
    "description": "SA model results from 7 independent CFD codes on 897x257 grid",
    "codes": ["CFL3D", "FUN3D", "NTS", "JOE", "SUMB", "TURNS", "GGNS"],
    "conditions": {"Mach": 0.15, "Re": 6e6, "fully_turbulent": True},
    "alpha_0": {"CL": 0.0, "CD": 0.00819},
    "alpha_10": {"CL": 1.0909, "CD": 0.01231},
    "alpha_15": {"CL": 1.5461, "CD": 0.02124},
    "tolerances": {"CL_percent": 1.0, "CD_percent": 4.0},
    # Complete 7-code comparison table from TMR SA results page
    "per_code_results": {
        "CFL3D":  {"CL_0": 0.0, "CL_10": 1.0909, "CL_15": 1.5461,
                   "CD_0": 0.00819, "CD_10": 0.01231, "CD_15": 0.02124},
        "FUN3D":  {"CL_0": 0.0, "CL_10": 1.0983, "CL_15": 1.5547,
                   "CD_0": 0.00812, "CD_10": 0.01242, "CD_15": 0.02159},
        "NTS":    {"CL_0": 0.0, "CL_10": 1.0891, "CL_15": 1.5461,
                   "CD_0": 0.00813, "CD_10": 0.01243, "CD_15": 0.02105},
        "JOE":    {"CL_0": 0.0, "CL_10": 1.0918, "CL_15": 1.5490,
                   "CD_0": 0.00812, "CD_10": 0.01245, "CD_15": 0.02148},
        "SUMB":   {"CL_0": 0.0, "CL_10": 1.0904, "CL_15": 1.5446,
                   "CD_0": 0.00813, "CD_10": 0.01233, "CD_15": 0.02141},
        "TURNS":  {"CL_0": 0.0, "CL_10": 1.1000, "CL_15": 1.5642,
                   "CD_0": 0.00830, "CD_10": 0.01230, "CD_15": 0.02140},
        "GGNS":   {"CL_0": 0.0, "CL_10": 1.0941, "CL_15": 1.5576,
                   "CD_0": 0.00817, "CD_10": 0.01225, "CD_15": 0.02073},
    },
}


# ============================================================================
# Download Functions
# ============================================================================

def download_file(url: str, dest: Path, force: bool = False) -> bool:
    """Download a file from URL to local path."""
    if dest.exists() and not force:
        print(f"  [EXISTS] {dest.name}")
        return True

    try:
        print(f"  [GET]    {url.split('/')[-1]} ... ", end="", flush=True)
        req = urllib.request.Request(url, headers={
            'User-Agent': 'CFD-Benchmark/1.0 (Academic Research)'
        })
        with urllib.request.urlopen(req, timeout=30) as response:
            data = response.read()

        # Handle gzipped files
        if url.endswith('.gz'):
            import gzip
            data = gzip.decompress(data)
            # Save without .gz extension
            dest = dest.with_suffix('') if dest.suffix == '.gz' else dest

        dest.write_bytes(data)
        size_kb = len(data) / 1024
        print(f"{size_kb:.1f} KB")
        return True

    except urllib.error.URLError as e:
        print(f"FAILED ({e.reason})")
        return False
    except Exception as e:
        print(f"FAILED ({e})")
        return False


def download_all_data(data_dir: Path, force: bool = False) -> Dict[str, bool]:
    """Download all TMR experimental and reference data files."""
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    print("\n--- Experimental Data ---")
    for key, info in TMR_FILES.items():
        filename = info["url"].split("/")[-1]
        dest = raw_dir / filename
        results[key] = download_file(info["url"], dest, force)

    print("\n--- Airfoil Surface Points ---")
    filename = TMR_AIRFOIL_POINTS["url"].split("/")[-1]
    dest = raw_dir / filename
    results["airfoil_points"] = download_file(TMR_AIRFOIL_POINTS["url"], dest, force)

    return results


def download_grids(data_dir: Path, levels: Optional[List[str]] = None,
                   force: bool = False) -> Dict[str, bool]:
    """Download TMR PLOT3D grids."""
    grid_dir = data_dir / "grids"
    grid_dir.mkdir(parents=True, exist_ok=True)

    if levels is None:
        levels = ["medium_225x65", "fine_449x129"]

    results = {}
    print("\n--- TMR C-Grids ---")
    for level in levels:
        if level not in TMR_GRIDS:
            print(f"  [SKIP]   Unknown level: {level}")
            continue
        info = TMR_GRIDS[level]
        filename = info["url"].split("/")[-1]
        if info["compressed"]:
            filename = filename.replace('.gz', '')
        dest = grid_dir / filename
        results[level] = download_file(info["url"], dest, force)

    return results


# ============================================================================
# Data Parsing Functions
# ============================================================================

def parse_tmr_dat(filepath: Path) -> List[List[float]]:
    """
    Parse a TMR .dat file (whitespace-separated, with possible header/comment lines).
    
    TMR files typically have:
    - Title line(s)
    - Optional column headers
    - Numeric data rows
    """
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            # Try to parse as numbers
            try:
                values = [float(x) for x in line.split()]
                if values:  # Skip empty parses
                    data.append(values)
            except ValueError:
                continue  # Skip header/text lines
    return data


def parse_tmr_zones(filepath: Path) -> Dict[str, Dict[str, list]]:
    """
    Parse a zone-separated TMR .dat file into a dict keyed by zone title.

    TMR surface data files use 'zone, t="..."' headers to separate data
    by angle of attack or surface (upper/lower).

    Returns
    -------
    dict
        Keys are zone titles (e.g. 'alpha=0', 'alpha=10, upper surface').
        Values are dicts with column-name keys mapping to float lists.
    """
    zones = {}
    current_zone = None
    col_names = None

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            # Detect column header line: variables="x","cp" etc.
            if line.lower().startswith('variables='):
                # Parse column names from variables="x","cp" format
                parts = line.split('=', 1)[1]
                col_names = [c.strip().strip('"').strip("'") for c in parts.split(',')]
                continue

            # Detect zone header: zone, t="alpha=0"
            if line.lower().startswith('zone'):
                # Extract title from t="..."
                title = line
                if 't=' in line or 'T=' in line:
                    import re
                    m = re.search(r'[tT]=["\']([^"\']+)["\']', line)
                    if m:
                        title = m.group(1)
                current_zone = title
                zones[current_zone] = {c: [] for c in (col_names or ['col0', 'col1'])}
                continue

            # Parse numeric data
            try:
                values = [float(x) for x in line.split()]
                if not values:
                    continue
            except ValueError:
                continue

            # If no zone header seen yet, create a default zone
            if current_zone is None:
                current_zone = 'default'
                zones[current_zone] = {c: [] for c in (col_names or ['col0', 'col1'])}

            # Store values into named columns
            names = list(zones[current_zone].keys())
            for i, v in enumerate(values):
                if i < len(names):
                    zones[current_zone][names[i]].append(v)

    return zones


def parse_ladson_forces(data_dir: Path) -> Optional[Dict]:
    """Parse Ladson tripped force data (CL, CD vs alpha)."""
    filepath = data_dir / "raw" / "CLCD_Ladson_expdata.dat"
    if not filepath.exists():
        return None

    data = parse_tmr_dat(filepath)
    if not data:
        return None

    # Ladson data format: alpha, CL, CD (or similar)
    result = {"alpha": [], "CL": [], "CD": [], "source": "Ladson (1988), tripped, Re=6M"}

    for row in data:
        if len(row) >= 3:
            result["alpha"].append(row[0])
            result["CL"].append(row[1])
            result["CD"].append(row[2])
        elif len(row) == 2:
            result["alpha"].append(row[0])
            result["CL"].append(row[1])

    return result


def parse_abbott_cl(data_dir: Path) -> Optional[Dict]:
    """Parse Abbott & von Doenhoff CL data."""
    filepath = data_dir / "raw" / "0012.abbottdata.cl.dat"
    if not filepath.exists():
        return None

    data = parse_tmr_dat(filepath)
    result = {"alpha": [], "CL": [], "source": "Abbott & von Doenhoff (1959), un-tripped"}

    for row in data:
        if len(row) >= 2:
            result["alpha"].append(row[0])
            result["CL"].append(row[1])

    return result


def parse_abbott_cd(data_dir: Path) -> Optional[Dict]:
    """Parse Abbott & von Doenhoff CD data (CD vs CL)."""
    filepath = data_dir / "raw" / "0012.abbottdata.cd.dat"
    if not filepath.exists():
        return None

    data = parse_tmr_dat(filepath)
    result = {"CL": [], "CD": [], "source": "Abbott & von Doenhoff (1959), un-tripped"}

    for row in data:
        if len(row) >= 2:
            result["CL"].append(row[0])
            result["CD"].append(row[1])

    return result


def parse_gregory_cl(data_dir: Path) -> Optional[Dict]:
    """Parse Gregory & O'Reilly CL data."""
    filepath = data_dir / "raw" / "CL_Gregory_expdata.dat"
    if not filepath.exists():
        return None

    data = parse_tmr_dat(filepath)
    result = {"alpha": [], "CL": [], "source": "Gregory & O'Reilly (1970), tripped, Re=3M"}

    for row in data:
        if len(row) >= 2:
            result["alpha"].append(row[0])
            result["CL"].append(row[1])

    return result


def parse_mccroskey_cl(data_dir: Path) -> Optional[Dict]:
    """Parse McCroskey CL best fit line."""
    filepath = data_dir / "raw" / "0012.mccroskeydata.cl.dat"
    if not filepath.exists():
        return None

    data = parse_tmr_dat(filepath)
    result = {"alpha": [], "CL": [], "source": "McCroskey (1987), best fit"}

    for row in data:
        if len(row) >= 2:
            result["alpha"].append(row[0])
            result["CL"].append(row[1])

    return result


def parse_gregory_cp(data_dir: Path) -> Optional[Dict]:
    """
    Parse Gregory & O'Reilly Cp data (upper surface, multiple alphas).

    Returns dict with zones keyed by alpha label, each containing
    'x/c' and 'cp' arrays.
    """
    filepath = data_dir / "raw" / "CP_Gregory_expdata.dat"
    if not filepath.exists():
        return None

    zones = parse_tmr_zones(filepath)
    return {
        "zones": zones,
        "source": "Gregory & O'Reilly (1970), Re=2.88M — best resolved Cp",
        "note": "Upper surface only. Better LE resolution than Ladson.",
    }


def parse_ladson_cp(data_dir: Path) -> Optional[Dict]:
    """
    Parse Ladson et al Cp data (multiple Re and transition conditions).

    Zones include Re=6M free transition, Re=3M/9M fixed transition.
    Each zone has 'x/c' and 'cp' arrays covering upper and lower surfaces.
    """
    filepath = data_dir / "raw" / "CP_Ladson.dat"
    if not filepath.exists():
        return None

    zones = parse_tmr_zones(filepath)
    return {
        "zones": zones,
        "source": "Ladson et al (1987), NASA TM 100526",
        "note": "LE peak less well resolved. Model aspect ratio only 1.333.",
    }


def parse_cfl3d_forces(data_dir: Path) -> Optional[Dict]:
    """Parse CFL3D SA force data (CL, CD vs alpha)."""
    filepath = data_dir / "raw" / "n0012clcd_cfl3d_sa.dat"
    if not filepath.exists():
        return None

    data = parse_tmr_dat(filepath)
    result = {
        "alpha": [], "CL": [], "CD": [],
        "source": "CFL3D SA model, 897x257 grid",
    }

    for row in data:
        if len(row) >= 3:
            result["alpha"].append(row[0])
            result["CL"].append(row[1])
            result["CD"].append(row[2])

    return result


def parse_cfl3d_cp(data_dir: Path) -> Optional[Dict]:
    """
    Parse CFL3D SA Cp data at multiple angles of attack.

    Zones are keyed by alpha label (e.g. 'alpha=0', 'alpha=10', 'alpha=15').
    Each zone contains 'x' and 'cp' arrays for the full surface.
    """
    filepath = data_dir / "raw" / "n0012cp_cfl3d_sa.dat"
    if not filepath.exists():
        return None

    zones = parse_tmr_zones(filepath)
    return {
        "zones": zones,
        "source": "CFL3D SA model, 897x257 grid, nu_tilde/nu=3",
    }


def parse_cfl3d_cf(data_dir: Path) -> Optional[Dict]:
    """
    Parse CFL3D SA Cf data at multiple angles of attack.

    Zones include upper and lower surface at each alpha.
    Each zone contains 'x' and 'cf' arrays.
    """
    filepath = data_dir / "raw" / "n0012cf_cfl3d_sa.dat"
    if not filepath.exists():
        return None

    zones = parse_tmr_zones(filepath)
    return {
        "zones": zones,
        "source": "CFL3D SA model, 897x257 grid, nu_tilde/nu=3",
        "note": "No experimental Cf data available for validation.",
    }


def parse_airfoil_points(data_dir: Path) -> Optional[Dict]:
    """Parse NACA 0012 surface points (sharp TE definition)."""
    filepath = data_dir / "raw" / "n0012points_superbig_clust_fix.dat"
    if not filepath.exists():
        return None

    data = parse_tmr_dat(filepath)
    x_coords = [row[0] for row in data if len(row) >= 2]
    y_coords = [row[1] for row in data if len(row) >= 2]

    return {
        "x": x_coords,
        "y": y_coords,
        "n_points": len(x_coords),
        "source": "TMR sharp TE definition (scaled to close at c=1)",
    }


# ============================================================================
# Data Export and Summary
# ============================================================================

def save_parsed_csv(data: Dict, filepath: Path, columns: List[str]) -> bool:
    """Save parsed data as CSV."""
    try:
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(columns)
            n_rows = len(data[columns[0]])
            for i in range(n_rows):
                row = [data[col][i] for col in columns if col in data]
                writer.writerow(row)
        return True
    except Exception as e:
        print(f"  [ERROR]  Failed to write {filepath.name}: {e}")
        return False


def export_all_csv(data_dir: Path) -> None:
    """Export all parsed data as clean CSV files."""
    csv_dir = data_dir / "csv"
    csv_dir.mkdir(exist_ok=True)

    print("\n--- Exporting Parsed CSV Files ---")

    # Ladson forces
    ladson = parse_ladson_forces(data_dir)
    if ladson and ladson["alpha"]:
        cols = ["alpha", "CL"] + (["CD"] if ladson.get("CD") else [])
        if save_parsed_csv(ladson, csv_dir / "ladson_forces.csv", cols):
            print(f"  [OK]     ladson_forces.csv ({len(ladson['alpha'])} rows)")

    # Abbott CL
    abbott_cl = parse_abbott_cl(data_dir)
    if abbott_cl and abbott_cl["alpha"]:
        if save_parsed_csv(abbott_cl, csv_dir / "abbott_cl.csv", ["alpha", "CL"]):
            print(f"  [OK]     abbott_cl.csv ({len(abbott_cl['alpha'])} rows)")

    # Abbott CD
    abbott_cd = parse_abbott_cd(data_dir)
    if abbott_cd and abbott_cd["CL"]:
        if save_parsed_csv(abbott_cd, csv_dir / "abbott_cd.csv", ["CL", "CD"]):
            print(f"  [OK]     abbott_cd.csv ({len(abbott_cd['CL'])} rows)")

    # Gregory CL
    gregory_cl = parse_gregory_cl(data_dir)
    if gregory_cl and gregory_cl["alpha"]:
        if save_parsed_csv(gregory_cl, csv_dir / "gregory_cl.csv", ["alpha", "CL"]):
            print(f"  [OK]     gregory_cl.csv ({len(gregory_cl['alpha'])} rows)")

    # McCroskey CL
    mccroskey = parse_mccroskey_cl(data_dir)
    if mccroskey and mccroskey["alpha"]:
        if save_parsed_csv(mccroskey, csv_dir / "mccroskey_cl.csv", ["alpha", "CL"]):
            print(f"  [OK]     mccroskey_cl.csv ({len(mccroskey['alpha'])} rows)")

    # CFL3D SA forces
    cfl3d_f = parse_cfl3d_forces(data_dir)
    if cfl3d_f and cfl3d_f["alpha"]:
        if save_parsed_csv(cfl3d_f, csv_dir / "cfl3d_sa_forces.csv",
                          ["alpha", "CL", "CD"]):
            print(f"  [OK]     cfl3d_sa_forces.csv ({len(cfl3d_f['alpha'])} rows)")

    # Airfoil points
    pts = parse_airfoil_points(data_dir)
    if pts and pts["x"]:
        if save_parsed_csv(pts, csv_dir / "naca0012_surface.csv", ["x", "y"]):
            print(f"  [OK]     naca0012_surface.csv ({pts['n_points']} points)")

    # --- Cp/Cf zone-based exports ---
    def _export_zones(parsed, prefix, cols, label):
        """Export zone-separated data as one CSV per zone."""
        if not parsed or "zones" not in parsed:
            return
        for zone_name, zone_data in parsed["zones"].items():
            # Sanitize zone name for filename
            safe = zone_name.replace(' ', '_').replace(',', '').replace('=', '')
            safe = safe.replace('/', '_').replace('__', '_').lower()
            fname = f"{prefix}_{safe}.csv"
            n_rows = len(zone_data[cols[0]]) if cols[0] in zone_data else 0
            if n_rows == 0:
                continue
            data_dict = {c: zone_data.get(c, []) for c in cols}
            if save_parsed_csv(data_dict, csv_dir / fname, cols):
                print(f"  [OK]     {fname} ({n_rows} rows)")

    # Gregory Cp (upper surface, 3 alphas)
    gregory_cp = parse_gregory_cp(data_dir)
    _export_zones(gregory_cp, "gregory_cp", ["x/c", "cp"], "Gregory Cp")

    # Ladson Cp (multiple Re/conditions)
    ladson_cp = parse_ladson_cp(data_dir)
    _export_zones(ladson_cp, "ladson_cp", ["x/c", "cp"], "Ladson Cp")

    # CFL3D SA Cp (3 alphas)
    cfl3d_cp = parse_cfl3d_cp(data_dir)
    _export_zones(cfl3d_cp, "cfl3d_sa_cp", ["x", "cp"], "CFL3D Cp")

    # CFL3D SA Cf (upper+lower at 3 alphas)
    cfl3d_cf = parse_cfl3d_cf(data_dir)
    _export_zones(cfl3d_cf, "cfl3d_sa_cf", ["x", "cf"], "CFL3D Cf")

    # Save TMR reference values as JSON
    ref_file = csv_dir / "tmr_sa_reference.json"
    with open(ref_file, 'w') as f:
        json.dump(TMR_SA_REFERENCE, f, indent=2)
    print(f"  [OK]     tmr_sa_reference.json (SA model targets + 7-code table)")


def print_summary(data_dir: Path) -> None:
    """Print a summary of available data."""
    print("\n" + "=" * 70)
    print("  NACA 0012 TMR DATA SUMMARY")
    print("=" * 70)

    print(f"\n  Case:  M=0.15, Re=6M, fully turbulent, sharp TE")
    print(f"  Source: https://turbmodels.larc.nasa.gov/naca0012_val.html")

    print("\n  TMR SA Reference Values (7-code consensus):")
    print("  " + "-" * 50)
    print(f"  {'alpha':>6s}  {'CL':>8s}  {'CD':>10s}")
    print("  " + "-" * 50)
    for key in ["alpha_0", "alpha_10", "alpha_15"]:
        alpha = key.replace("alpha_", "")
        ref = TMR_SA_REFERENCE[key]
        print(f"  {alpha + chr(176):>6s}  {ref['CL']:>8.4f}  {ref['CD']:>10.5f}")
    print(f"\n  Tolerances: CL within {TMR_SA_REFERENCE['tolerances']['CL_percent']}%, "
          f"CD within {TMR_SA_REFERENCE['tolerances']['CD_percent']}%")

    # Check what data files exist
    raw_dir = data_dir / "raw"
    csv_dir = data_dir / "csv"

    if raw_dir.exists():
        raw_files = list(raw_dir.glob("*"))
        print(f"\n  Raw data files:  {len(raw_files)} in {raw_dir}")
    else:
        print(f"\n  Raw data: NOT DOWNLOADED (run without --info)")

    if csv_dir.exists():
        csv_files = list(csv_dir.glob("*.csv"))
        print(f"  Parsed CSV files: {len(csv_files)} in {csv_dir}")

    grid_dir = data_dir / "grids"
    if grid_dir.exists():
        grid_files = list(grid_dir.glob("*"))
        print(f"  Grid files:      {len(grid_files)} in {grid_dir}")
    else:
        print(f"  Grids: NOT DOWNLOADED (use --grids flag)")


# ============================================================================
# Main
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Download NACA 0012 TMR validation data from NASA Langley"
    )
    parser.add_argument("--info", action="store_true",
                        help="Show data summary without downloading")
    parser.add_argument("--grids", action="store_true",
                        help="Also download TMR C-grids")
    parser.add_argument("--grid-levels", nargs="+",
                        default=["medium_225x65", "fine_449x129"],
                        help="Grid levels to download")
    parser.add_argument("--force", action="store_true",
                        help="Re-download existing files")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory (default: script directory)")
    args = parser.parse_args()

    data_dir = args.output_dir or Path(__file__).parent.resolve()

    print("=" * 70)
    print("  NACA 0012 TMR DATA DOWNLOADER")
    print("  NASA Langley Turbulence Modeling Resource")
    print("=" * 70)

    if args.info:
        print_summary(data_dir)
        return

    # Download experimental and reference data
    data_results = download_all_data(data_dir, force=args.force)

    # Download grids if requested
    grid_results = {}
    if args.grids:
        grid_results = download_grids(data_dir, args.grid_levels, force=args.force)

    # Parse and export CSV
    export_all_csv(data_dir)

    # Print summary
    print_summary(data_dir)

    # Final status
    n_ok = sum(1 for v in data_results.values() if v)
    n_total = len(data_results)
    print(f"\n  Download status: {n_ok}/{n_total} data files")
    if grid_results:
        g_ok = sum(1 for v in grid_results.values() if v)
        print(f"  Grid status:     {g_ok}/{len(grid_results)} grids")

    if n_ok == n_total:
        print(f"\n  [OK] All data downloaded. Ready for simulation setup.")
        print(f"  Next: python run_naca0012.py --help")
    else:
        print(f"\n  [!!] Some downloads failed. Re-run with --force to retry.")


if __name__ == "__main__":
    main()
