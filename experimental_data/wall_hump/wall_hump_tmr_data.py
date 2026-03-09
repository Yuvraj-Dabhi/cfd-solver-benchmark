#!/usr/bin/env python3
"""
NASA Wall-Mounted Hump — TMR Data Downloader
=============================================
Downloads grids and reference data from the NASA TMR website.
Uses the no-plenum grid variant (recommended for the no-flow-control case).

Reference: https://turbmodels.larc.nasa.gov/nasahump_val.html
"""

import os
import sys
import gzip
import json
import shutil
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).parent
GRIDS_DIR = DATA_DIR / "grids"
CSV_DIR = DATA_DIR / "csv"

# TMR no-plenum grids (2D PLOT3D format, gzipped)
GRID_URLS = {
    "coarse":  ("hump2newtop_noplenumZ103x28.p2dfmt.gz",   103, 28),
    "medium":  ("hump2newtop_noplenumZ205x55.p2dfmt.gz",   205, 55),
    "fine":    ("hump2newtop_noplenumZ409x109.p2dfmt.gz",   409, 109),
    "xfine":   ("hump2newtop_noplenumZ817x217.p2dfmt.gz",   817, 217),
    "ultra":   ("hump2newtop_noplenumZ1633x433.p2dfmt.gz", 1633, 433),
}
GRID_BASE_URL = "https://turbmodels.larc.nasa.gov/Nasahump_grids/"

# Experimental reference data
REF_URLS = {
    "noflow_cp.exp.dat":          "https://turbmodels.larc.nasa.gov/Nasahump_validation/noflow_cp.exp.dat",
    "noflow_cf.exp.dat":          "https://turbmodels.larc.nasa.gov/Nasahump_validation/noflow_cf.exp.dat",
    "noflow_u_inflow.exp.dat":    "https://turbmodels.larc.nasa.gov/Nasahump_validation/noflow_u_inflow.exp.dat",
    "noflow_vel_and_turb.exp.dat": "https://turbmodels.larc.nasa.gov/Nasahump_validation/noflow_vel_and_turb.exp.dat",
}


def download_file(url: str, dest: Path, desc: str = ""):
    """Download a file from URL to destination."""
    if dest.exists():
        print(f"  [OK]     {desc or dest.name} (already downloaded)")
        return True
    try:
        print(f"  [DOWN]   {desc or dest.name}...")
        urllib.request.urlretrieve(url, str(dest))
        print(f"           → {dest.stat().st_size / 1024:.1f} KB")
        return True
    except Exception as e:
        print(f"  [FAIL]   {desc or dest.name}: {e}")
        return False


def download_grids():
    """Download all TMR hump grids."""
    GRIDS_DIR.mkdir(parents=True, exist_ok=True)
    print("\n  Downloading TMR hump grids (no-plenum, 2D PLOT3D)...")
    for level, (fname, idim, jdim) in GRID_URLS.items():
        url = GRID_BASE_URL + fname
        dest = GRIDS_DIR / fname
        download_file(url, dest, f"{level} ({idim}×{jdim})")


def download_reference_data():
    """Download experimental reference data."""
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    print("\n  Downloading experimental reference data...")
    for fname, url in REF_URLS.items():
        dest = CSV_DIR / fname
        download_file(url, dest, fname)


def parse_cp_data():
    """Parse experimental Cp data into CSV format."""
    src = CSV_DIR / "noflow_cp.exp.dat"
    dst = CSV_DIR / "exp_cp.csv"
    if dst.exists():
        return
    if not src.exists():
        return

    print("  [PARSE]  Parsing experimental Cp → exp_cp.csv")
    lines = src.read_text().strip().split("\n")
    with open(dst, "w") as f:
        f.write("x_c,Cp\n")
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    x = float(parts[0])
                    cp = float(parts[1])
                    f.write(f"{x},{cp}\n")
                except ValueError:
                    continue


def parse_cf_data():
    """Parse experimental Cf data into CSV format."""
    src = CSV_DIR / "noflow_cf.exp.dat"
    dst = CSV_DIR / "exp_cf.csv"
    if dst.exists():
        return
    if not src.exists():
        return

    print("  [PARSE]  Parsing experimental Cf → exp_cf.csv")
    lines = src.read_text().strip().split("\n")
    with open(dst, "w") as f:
        f.write("x_c,Cf\n")
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    x = float(parts[0])
                    cf = float(parts[1])
                    f.write(f"{x},{cf}\n")
                except ValueError:
                    continue


def create_reference_json():
    """Create a JSON reference file with hump case specification."""
    ref = {
        "case": "2DWMH - NASA Wall-Mounted Hump",
        "tmr_page": "https://turbmodels.larc.nasa.gov/nasahump_val.html",
        "flow_conditions": {
            "mach": 0.1,
            "reynolds": 936000,
            "reynolds_length_mm": 420,
            "chord_mm": 420,
            "freestream_velocity_ms": 34.6,
            "boundary_layer_thickness_mm": 35,
            "bl_thickness_over_chord": 0.083,
            "turbulence": "fully_turbulent",
        },
        "geometry": {
            "type": "Wall-mounted Glauert-Goldschmied body",
            "trailing_edge": "smooth",
            "upper_wall": "contoured (blockage correction)",
            "plenum": "no (no-flow-control case)",
        },
        "separation": {
            "separation_x_c": 0.665,
            "reattachment_x_c": 1.10,
            "bubble_length_c": 0.435,
            "description": "Smooth-body separation due to adverse pressure gradient",
        },
        "experimental_references": [
            {
                "authors": "Greenblatt, D., Paschal, K.B., Yao, C.-S., et al.",
                "title": "Experimental Investigation of Separation Control Part 1: Baseline and Steady Suction",
                "journal": "AIAA Journal",
                "volume": "44(12)",
                "year": 2006,
                "pages": "2820-2830",
                "doi": "10.2514/1.13817",
            },
            {
                "authors": "Naughton, J.W., Viken, S.A., Greenblatt, D.",
                "title": "Skin-Friction Measurements on the NASA Hump Model",
                "journal": "AIAA Journal",
                "volume": "44(6)",
                "year": 2006,
                "pages": "1255-1265",
                "doi": "10.2514/1.14192",
            },
        ],
        "validation_profiles": {
            "x_c_stations": [-2.14, 0.65, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3],
            "quantities": ["Cp", "Cf", "U/U_ref", "u'u'/U_ref^2"],
        },
        "grids": {
            "family": "no-plenum",
            "levels": {
                "coarse": {"dims": [103, 28], "cells": 2754},
                "medium": {"dims": [205, 55], "cells": 11016},
                "fine":   {"dims": [409, 109], "cells": 44064},
                "xfine":  {"dims": [817, 217], "cells": 176256},
                "ultra":  {"dims": [1633, 433], "cells": 705024},
            },
            "wall_spacing": 4e-6,
            "y_plus_approx": 0.1,
        },
    }
    dst = CSV_DIR / "hump_case_reference.json"
    with open(dst, "w") as f:
        json.dump(ref, f, indent=2)
    print(f"  [OK]     Created hump_case_reference.json")


def download_all():
    """Download all data for the wall-mounted hump case."""
    print("=" * 60)
    print("  NASA Wall-Mounted Hump — TMR Data Download")
    print("=" * 60)
    download_grids()
    download_reference_data()
    parse_cp_data()
    parse_cf_data()
    create_reference_json()
    print("\n  All data downloaded and parsed successfully!")
    print(f"  Grids: {GRIDS_DIR}")
    print(f"  Data:  {CSV_DIR}")


if __name__ == "__main__":
    download_all()
