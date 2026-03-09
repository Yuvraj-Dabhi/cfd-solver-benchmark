#!/usr/bin/env python3
"""
NACA 0012 Validation & Plotting
================================
Post-processes CFD results and compares against TMR experimental data.

Generates publication-quality plots:
  1. CL vs alpha polar (with Ladson, Gregory, Abbott, McCroskey, CFL3D)
  2. CD vs CL drag polar
  3. Cp distributions at alpha = 0, 10, 15 (vs Gregory data)
  4. Cf distributions at alpha = 0, 10, 15 (vs CFL3D SA reference)

Usage:
    python naca0012_validation.py --results-dir runs/naca0012
    python naca0012_validation.py --demo   # Plot experimental data only (no CFD)
"""

import sys
import json
import csv
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ============================================================================
# Style Configuration
# ============================================================================

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'figure.facecolor': 'white',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 1.5,
})

# Color palette
COLORS = {
    'ladson': '#D62728',      # Red — primary experimental
    'gregory': '#2CA02C',     # Green
    'abbott': '#FF7F0E',      # Orange
    'mccroskey': '#9467BD',   # Purple
    'cfl3d': '#1F77B4',       # Blue — CFD reference
    'user_cfd': '#E377C2',    # Pink — user's CFD results
}


# ============================================================================
# Data Loading
# ============================================================================

def load_csv(filepath: Path) -> Optional[Dict[str, List[float]]]:
    """Load a CSV file with headers into dict of lists."""
    if not filepath.exists():
        return None
    data = {}
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for col in reader.fieldnames:
            data[col] = []
        for row in reader:
            for col in reader.fieldnames:
                try:
                    data[col].append(float(row[col]))
                except (ValueError, KeyError):
                    pass
    return data


def load_tmr_raw(filepath: Path) -> List[List[float]]:
    """Load raw TMR .dat file (whitespace-separated numerics)."""
    data = []
    if not filepath.exists():
        return data
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            try:
                values = [float(x) for x in line.split()]
                if values:
                    data.append(values)
            except ValueError:
                continue
    return data


def load_all_data(data_dir: Path) -> Dict:
    """Load all experimental and reference data."""
    csv_dir = data_dir / "csv"
    raw_dir = data_dir / "raw"

    exp = {}

    # Parsed CSV files
    exp['ladson'] = load_csv(csv_dir / "ladson_forces.csv")
    exp['abbott_cl'] = load_csv(csv_dir / "abbott_cl.csv")
    exp['abbott_cd'] = load_csv(csv_dir / "abbott_cd.csv")
    exp['gregory_cl'] = load_csv(csv_dir / "gregory_cl.csv")
    exp['mccroskey_cl'] = load_csv(csv_dir / "mccroskey_cl.csv")
    exp['cfl3d_forces'] = load_csv(csv_dir / "cfl3d_sa_forces.csv")

    # Raw TMR files for Cp and Cf
    exp['gregory_cp_raw'] = load_tmr_raw(raw_dir / "CP_Gregory_expdata.dat")
    exp['cfl3d_cp_raw'] = load_tmr_raw(raw_dir / "n0012cp_cfl3d_sa.dat")
    exp['cfl3d_cf_raw'] = load_tmr_raw(raw_dir / "n0012cf_cfl3d_sa.dat")

    # TMR reference values
    ref_file = csv_dir / "tmr_sa_reference.json"
    if ref_file.exists():
        with open(ref_file) as f:
            exp['tmr_ref'] = json.load(f)

    return exp


# ============================================================================
# Plot: CL vs Alpha
# ============================================================================

def plot_cl_alpha(exp: Dict, cfd_results: Optional[Dict] = None,
                   output_dir: Path = Path(".")) -> Path:
    """Plot lift coefficient vs angle of attack."""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Ladson tripped (PRIMARY)
    if exp.get('ladson') and 'alpha' in exp['ladson']:
        ax.plot(exp['ladson']['alpha'], exp['ladson']['CL'],
                'o', color=COLORS['ladson'], markersize=4, alpha=0.7,
                label='Ladson (tripped, Re=6M)')

    # Gregory
    if exp.get('gregory_cl') and 'alpha' in exp['gregory_cl']:
        ax.plot(exp['gregory_cl']['alpha'], exp['gregory_cl']['CL'],
                's', color=COLORS['gregory'], markersize=4, alpha=0.7,
                label='Gregory (tripped, Re=3M)')

    # Abbott
    if exp.get('abbott_cl') and 'alpha' in exp['abbott_cl']:
        ax.plot(exp['abbott_cl']['alpha'], exp['abbott_cl']['CL'],
                '^', color=COLORS['abbott'], markersize=4, alpha=0.7,
                label='Abbott (un-tripped, Re=6M)')

    # McCroskey best fit
    if exp.get('mccroskey_cl') and 'alpha' in exp['mccroskey_cl']:
        ax.plot(exp['mccroskey_cl']['alpha'], exp['mccroskey_cl']['CL'],
                '--', color=COLORS['mccroskey'], linewidth=1.0,
                label='McCroskey (best fit)')

    # CFL3D SA reference
    if exp.get('cfl3d_forces') and 'alpha' in exp['cfl3d_forces']:
        ax.plot(exp['cfl3d_forces']['alpha'], exp['cfl3d_forces']['CL'],
                'D-', color=COLORS['cfl3d'], markersize=5,
                label='CFL3D SA (897x257)')

    # User's CFD results
    if cfd_results and 'alphas' in cfd_results:
        ax.plot(cfd_results['alphas'], cfd_results['CL'],
                '*-', color=COLORS['user_cfd'], markersize=8, linewidth=2,
                label='This Study', zorder=10)

    ax.set_xlabel(r'Angle of Attack, $\alpha$ [deg]')
    ax.set_ylabel(r'Lift Coefficient, $C_L$')
    ax.set_title('NACA 0012 — CL vs Alpha (M=0.15, Re=6M)')
    ax.legend(loc='lower right')
    ax.set_xlim(-2, 20)

    filepath = output_dir / "naca0012_cl_alpha.png"
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    return filepath


# ============================================================================
# Plot: CD vs CL (Drag Polar)
# ============================================================================

def plot_cd_cl(exp: Dict, cfd_results: Optional[Dict] = None,
                output_dir: Path = Path(".")) -> Path:
    """Plot drag polar (CD vs CL)."""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Ladson tripped
    if exp.get('ladson') and 'CD' in exp['ladson'] and exp['ladson']['CD']:
        ax.plot(exp['ladson']['CL'], exp['ladson']['CD'],
                'o', color=COLORS['ladson'], markersize=4, alpha=0.7,
                label='Ladson (tripped, Re=6M)')

    # Abbott
    if exp.get('abbott_cd') and 'CL' in exp['abbott_cd']:
        ax.plot(exp['abbott_cd']['CL'], exp['abbott_cd']['CD'],
                '^', color=COLORS['abbott'], markersize=4, alpha=0.7,
                label='Abbott (un-tripped, Re=6M)')

    # CFL3D SA
    if exp.get('cfl3d_forces') and 'CD' in exp['cfl3d_forces']:
        ax.plot(exp['cfl3d_forces']['CL'], exp['cfl3d_forces']['CD'],
                'D-', color=COLORS['cfl3d'], markersize=5,
                label='CFL3D SA (897x257)')

    # User's CFD
    if cfd_results and 'CD' in cfd_results:
        ax.plot(cfd_results['CL'], cfd_results['CD'],
                '*-', color=COLORS['user_cfd'], markersize=8, linewidth=2,
                label='This Study', zorder=10)

    ax.set_xlabel(r'Lift Coefficient, $C_L$')
    ax.set_ylabel(r'Drag Coefficient, $C_D$')
    ax.set_title('NACA 0012 — Drag Polar (M=0.15, Re=6M)')
    ax.legend()

    filepath = output_dir / "naca0012_cd_cl.png"
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    return filepath


# ============================================================================
# Plot: CM vs Alpha (Pitching Moment)
# ============================================================================

def plot_cm_alpha(exp: Dict, cfd_results: Optional[Dict] = None,
                   output_dir: Path = Path(".")) -> Path:
    """
    Plot pitching moment coefficient vs angle of attack.

    Note: No experimental CM data available from TMR. Comparison is
    against CFL3D SA reference only. Moment reference point at 0.25c.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # CFL3D SA — if force data includes CM
    if exp.get('cfl3d_forces') and 'CM' in exp['cfl3d_forces']:
        ax.plot(exp['cfl3d_forces']['alpha'], exp['cfl3d_forces']['CM'],
                'D-', color=COLORS['cfl3d'], markersize=5,
                label='CFL3D SA (897×257)')

    # User's CFD
    if cfd_results and 'CM' in cfd_results and any(
            cm is not None for cm in cfd_results.get('CM', [])):
        alphas = cfd_results['alphas']
        cms = cfd_results['CM']
        # Filter out None values
        valid = [(a, cm) for a, cm in zip(alphas, cms) if cm is not None]
        if valid:
            a_vals, cm_vals = zip(*valid)
            ax.plot(a_vals, cm_vals,
                    '*-', color=COLORS['user_cfd'], markersize=8, linewidth=2,
                    label='This Study', zorder=10)

    ax.set_xlabel(r'Angle of Attack, $\alpha$ (deg)')
    ax.set_ylabel(r'Pitching Moment Coefficient, $C_M$ (about 0.25c)')
    ax.set_title('NACA 0012 — Pitching Moment (M=0.15, Re=6M)')
    ax.legend()
    ax.axhline(y=0, color='k', linewidth=0.5, alpha=0.3)

    filepath = output_dir / "naca0012_cm_alpha.png"
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    return filepath


# ============================================================================
# Plot: Cp Distribution
# ============================================================================

def plot_cp_distribution(exp: Dict, alpha: float = 0.0,
                          cfd_cp: Optional[Dict] = None,
                          output_dir: Path = Path(".")) -> Path:
    """Plot surface pressure coefficient distribution."""
    fig, ax = plt.subplots(figsize=(8, 5))

    # Gregory Cp data (try to find data for this alpha)
    if exp.get('gregory_cp_raw'):
        raw = exp['gregory_cp_raw']
        # Gregory data is typically: x/c, Cp at various alphas
        if raw:
            x_cp = [row[0] for row in raw if len(row) >= 2]
            y_cp = [row[1] for row in raw if len(row) >= 2]
            if x_cp:
                ax.plot(x_cp, y_cp, 'o', color=COLORS['gregory'], markersize=3,
                        alpha=0.7, label=f'Gregory (Re=3M)')

    # CFL3D Cp
    if exp.get('cfl3d_cp_raw'):
        raw = exp['cfl3d_cp_raw']
        if raw:
            x_cp = [row[0] for row in raw if len(row) >= 2]
            y_cp = [row[1] for row in raw if len(row) >= 2]
            if x_cp:
                ax.plot(x_cp, y_cp, '-', color=COLORS['cfl3d'], linewidth=1.0,
                        label='CFL3D SA')

    # User's CFD Cp
    if cfd_cp:
        ax.plot(cfd_cp['x'], cfd_cp['Cp'], '-', color=COLORS['user_cfd'],
                linewidth=2, label='This Study', zorder=10)

    ax.invert_yaxis()
    ax.set_xlabel('x/c')
    ax.set_ylabel(r'$C_p$')
    ax.set_title(f'NACA 0012 — Cp Distribution (alpha={alpha:.0f} deg)')
    ax.legend()

    filepath = output_dir / f"naca0012_cp_alpha{alpha:.0f}.png"
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    return filepath


# ============================================================================
# Plot: Cf Distribution (Skin Friction)
# ============================================================================

def plot_cf_distribution(exp: Dict, alpha: float = 0.0,
                          cfd_cf: Optional[Dict] = None,
                          output_dir: Path = Path(".")) -> Path:
    """Plot skin friction coefficient distribution.

    Compares user CFD results against CFL3D SA reference.

    Parameters
    ----------
    exp : dict
        Experimental/reference data from load_all_data().
    alpha : float
        Angle of attack [degrees].
    cfd_cf : dict, optional
        User's CFD Cf data with keys 'x' and 'Cf'.
    output_dir : Path
        Directory for output plot.

    Returns
    -------
    Path
        Path to saved plot.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    # CFL3D SA reference Cf
    if exp.get('cfl3d_cf_raw'):
        raw = exp['cfl3d_cf_raw']
        if raw:
            x_cf = [row[0] for row in raw if len(row) >= 2]
            y_cf = [row[1] for row in raw if len(row) >= 2]
            if x_cf:
                ax.plot(x_cf, y_cf, '-', color=COLORS['cfl3d'], linewidth=1.0,
                        label='CFL3D SA (897×257)')

    # User's CFD Cf
    if cfd_cf:
        ax.plot(cfd_cf['x'], cfd_cf['Cf'], '-', color=COLORS['user_cfd'],
                linewidth=2, label='This Study', zorder=10)

    # Zero line for separation indication
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.6)

    ax.set_xlabel('x/c')
    ax.set_ylabel(r'$C_f$')
    ax.set_title(f'NACA 0012 — Skin Friction (α={alpha:.0f}°, M=0.15, Re=6M)')
    ax.legend()
    ax.set_xlim(-0.05, 1.05)

    filepath = output_dir / f"naca0012_cf_alpha{alpha:.0f}.png"
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    return filepath


# ============================================================================
# Plot: Airfoil Geometry
# ============================================================================

def plot_airfoil(data_dir: Path, output_dir: Path = Path(".")) -> Path:
    """Plot NACA 0012 airfoil shape using TMR surface points."""
    surface = load_csv(data_dir / "csv" / "naca0012_surface.csv")
    if not surface:
        print("  [SKIP]   No airfoil surface data")
        return None

    fig, ax = plt.subplots(figsize=(10, 3))

    x = np.array(surface['x'])
    y = np.array(surface['y'])

    ax.plot(x, y, 'k-', linewidth=0.8)
    ax.fill(x, y, alpha=0.1, color='steelblue')
    ax.set_aspect('equal')
    ax.set_xlabel('x/c')
    ax.set_ylabel('y/c')
    ax.set_title('NACA 0012 (TMR Sharp TE Definition)')
    ax.set_xlim(-0.05, 1.05)

    filepath = output_dir / "naca0012_airfoil.png"
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    return filepath


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="NACA 0012 TMR Validation Post-Processing"
    )
    parser.add_argument("--results-dir", type=Path, default=None,
                        help="Directory with CFD results")
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="TMR data directory")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory for plots")
    parser.add_argument("--demo", action="store_true",
                        help="Plot experimental data only (no CFD results)")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent.parent.resolve()
    data_dir = args.data_dir or project_root / "experimental_data" / "naca0012"
    output_dir = args.output_dir or project_root / "validation_results" / "naca0012"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  NACA 0012 TMR Validation Plots")
    print("=" * 60)

    # Load experimental data
    print("\n  Loading TMR experimental data...")
    exp = load_all_data(data_dir)

    n_sources = sum(1 for v in exp.values() if v is not None)
    print(f"  Loaded {n_sources} data sources")

    # Load CFD results if available
    cfd = None
    if args.results_dir and not args.demo:
        summary = args.results_dir / "results_summary.json"
        if summary.exists():
            with open(summary) as f:
                cfd_data = json.load(f)
            # Extract alpha, CL, CD, CM from summary
            cfd = {"alphas": [], "CL": [], "CD": [], "CM": []}
            for key, val in cfd_data.items():
                if key.startswith("alpha_") and isinstance(val, dict):
                    parts = key.split("_")
                    try:
                        alpha = float(parts[1])
                        cfd["alphas"].append(alpha)
                        cfd["CL"].append(val.get("CL", 0))
                        cfd["CD"].append(val.get("CD", 0))
                        cfd["CM"].append(val.get("CM"))
                    except (ValueError, IndexError):
                        pass

    # Generate plots — TMR QoI
    print("\n  Generating plots (TMR Quantities of Interest)...")
    print(f"  QoI: CL vs alpha, CD vs CL, CM vs alpha, Cp(0,10,15), Cf(0,10,15)")

    # Airfoil shape
    p = plot_airfoil(data_dir, output_dir)
    if p:
        print(f"  [OK]     {p.name}")

    # CL vs alpha
    p = plot_cl_alpha(exp, cfd, output_dir)
    print(f"  [OK]     {p.name}")

    # CD vs CL
    p = plot_cd_cl(exp, cfd, output_dir)
    print(f"  [OK]     {p.name}")

    # CM vs alpha
    p = plot_cm_alpha(exp, cfd, output_dir)
    print(f"  [OK]     {p.name}")

    # Cp distributions at alpha = 0, 10, 15
    for alpha in [0, 10, 15]:
        p = plot_cp_distribution(exp, alpha, output_dir=output_dir)
        print(f"  [OK]     {p.name}")

    # Cf distributions at alpha = 0, 10, 15
    for alpha in [0, 10, 15]:
        p = plot_cf_distribution(exp, alpha, output_dir=output_dir)
        print(f"  [OK]     {p.name}")

    print(f"\n  All plots saved to: {output_dir}")
    print(f"  [OK]     Validation complete.")


if __name__ == "__main__":
    main()
