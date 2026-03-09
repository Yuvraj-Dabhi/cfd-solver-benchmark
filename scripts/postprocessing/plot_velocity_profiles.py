#!/usr/bin/env python3
"""
plot_velocity_profiles.py — Wall Hump Velocity Profile Validation
===================================================================
Extracts u(y) profiles from SU2 volume VTU at multiple streamwise stations
and compares against Greenblatt PIV data (CFDVAL2004 style).

Stations: x/c = 0.65, 0.80, 1.00, 1.20 (spanning the separation bubble)
"""
import re, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

try:
    import pyvista as pv
except ImportError:
    print("PyVista not available. Install: pip install pyvista"); sys.exit(1)

PROJECT = Path(__file__).resolve().parent
RUNS_DIR = PROJECT / "runs" / "wall_hump"
EXP_FILE = PROJECT / "experimental_data" / "wall_hump" / "csv" / "noflow_vel_and_turb.exp.dat"
OUT_DIR = PROJECT / "plots" / "wall_hump"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Profile stations (x/c)
STATIONS = [0.65, 0.80, 1.00, 1.20]

# Hump geometry: y_wall at each station (approximate from TMR geometry)
# The hump height function: h(x) = 0.0 for x < 0 and x > 1, with max h ≈ 0.0537 at x/c ≈ 0.5
# For profile extraction, we sample from y_wall up to y = 0.15
Y_MAX = 0.15
N_SAMPLE = 300

COLORS_SIM = {
    "SA_coarse": ("#ef9a9a", "--"),
    "SA_medium": ("#e53935", "-"),
    "SA_fine":   ("#b71c1c", "-."),
    "SST_medium": ("#1e88e5", "-"),
}
COLOR_EXP = "#2e7d32"

# --- Load experimental PIV data ---
def load_piv_data(filepath):
    """Parse Tecplot-format PIV data with zone headers."""
    profiles = {}
    with open(filepath, "r") as f:
        lines = f.readlines()

    current_xc = None
    data_lines = []

    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("variables"):
            continue
        if line.startswith("zone"):
            # Save previous zone
            if current_xc is not None and data_lines:
                arr = np.array(data_lines)
                profiles[current_xc] = {
                    "x/c": arr[:, 0], "y/c": arr[:, 1],
                    "u/Uinf": arr[:, 2], "v/Uinf": arr[:, 3],
                }
            # Parse new zone
            m = re.search(r"x/c=([\d.]+)", line)
            if m:
                current_xc = float(m.group(1))
            data_lines = []
        else:
            vals = [float(v) for v in line.split()]
            if len(vals) >= 4:
                data_lines.append(vals[:4])

    # Last zone
    if current_xc is not None and data_lines:
        arr = np.array(data_lines)
        profiles[current_xc] = {
            "x/c": arr[:, 0], "y/c": arr[:, 1],
            "u/Uinf": arr[:, 2], "v/Uinf": arr[:, 3],
        }

    return profiles


def extract_profiles(vtu_path, stations, y_max=0.15, n_pts=300):
    """Extract velocity profiles at given x/c stations from volume VTU."""
    mesh = pv.read(str(vtu_path))

    # Get freestream velocity for normalization
    gamma = 1.4; R = 287.058; T = 300.0; M = 0.1
    a_inf = (gamma * R * T) ** 0.5
    U_inf = M * a_inf  # ~34.72 m/s

    # Estimate cell size for interpolation radius
    bounds = mesh.bounds
    # Use a small radius for near-wall accuracy
    cell_size = (bounds[1] - bounds[0]) / (mesh.n_cells ** 0.5)
    radius = max(cell_size * 3, 0.01)

    profiles = {}
    for xc in stations:
        # Create probe points along wall-normal line
        pts = np.zeros((n_pts, 3))
        pts[:, 0] = xc
        pts[:, 1] = np.linspace(0.0, y_max, n_pts)
        pts[:, 2] = 0.0

        probe = pv.PolyData(pts)
        result = probe.interpolate(mesh, radius=radius, sharpness=4)

        if result.n_points == 0 or "Velocity" not in result.point_data:
            print(f"    WARNING: No velocity data at x/c={xc}")
            continue

        y = result.points[:, 1]
        vel = result["Velocity"]
        u = vel[:, 0]
        v = vel[:, 1]

        profiles[xc] = {
            "y": y,
            "u/Uinf": u / U_inf,
            "v/Uinf": v / U_inf,
            "U_inf": U_inf,
        }

    return profiles


def main():
    print("=" * 60)
    print("  WALL HUMP — VELOCITY PROFILE EXTRACTION")
    print("=" * 60)
    print(f"  Stations: x/c = {STATIONS}")

    # Load experimental PIV data
    piv = {}
    if EXP_FILE.exists():
        piv = load_piv_data(EXP_FILE)
        print(f"  PIV data loaded: {len(piv)} stations")
        print(f"    Stations: {sorted(piv.keys())}")
    else:
        print("  [WARN] No PIV data file found")

    # Extract profiles from each available simulation
    sim_profiles = {}
    for model in ["SA", "SST"]:
        for grid in ["coarse", "medium", "fine"]:
            label = f"{model}_{grid}"
            vtu = RUNS_DIR / f"hump_{model}_{grid}" / "flow.vtu"
            if not vtu.exists():
                continue
            print(f"\n  Extracting {label}...")
            try:
                profs = extract_profiles(vtu, STATIONS, Y_MAX, N_SAMPLE)
                sim_profiles[label] = profs
                for xc, p in profs.items():
                    u_max = p["u/Uinf"].max()
                    print(f"    x/c={xc}: {len(p['y'])} pts, u/Uinf_max={u_max:.3f}")
            except Exception as e:
                print(f"    [ERROR] {e}")

    if not sim_profiles:
        print("\n  No simulation data extracted!"); return

    # ─── Plot: u/Uinf profiles at each station ───────────────────────
    n_stations = len(STATIONS)
    fig, axes = plt.subplots(1, n_stations, figsize=(4 * n_stations, 7),
                              sharey=True)
    if n_stations == 1:
        axes = [axes]

    for i, xc in enumerate(STATIONS):
        ax = axes[i]

        # Experimental PIV
        if xc in piv:
            p = piv[xc]
            ax.plot(p["u/Uinf"], p["y/c"], "o", color=COLOR_EXP,
                    ms=2.5, alpha=0.6, label="PIV (Greenblatt)", zorder=5)

        # Simulations
        for label, profs in sim_profiles.items():
            if xc not in profs:
                continue
            p = profs[xc]
            color, ls = COLORS_SIM.get(label, ("#333", "-"))
            ax.plot(p["u/Uinf"], p["y"], ls, color=color,
                    lw=1.5, label=f"SU2 {label}", zorder=4)

        ax.axvline(0, color="gray", ls="-", lw=0.5, alpha=0.4)
        ax.set_xlabel("$u / U_\\infty$")
        ax.set_title(f"$x/c = {xc}$", fontsize=13, fontweight="bold")
        ax.set_xlim(-0.2, 1.4)
        ax.set_ylim(0, Y_MAX)
        ax.grid(True, alpha=0.2)

    axes[0].set_ylabel("$y / c$")
    # Single legend for all panels
    handles, labels = axes[0].get_legend_handles_labels()
    # Collect all unique labels
    for ax in axes[1:]:
        h, l = ax.get_legend_handles_labels()
        for hi, li in zip(h, l):
            if li not in labels:
                handles.append(hi)
                labels.append(li)
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 5),
               framealpha=0.9, fontsize=10,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("Wall-Mounted Hump — Velocity Profiles", fontsize=15, y=1.01)
    plt.tight_layout()

    fname = OUT_DIR / "wall_hump_velocity_profiles.png"
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: {fname.name}")

    # ─── Plot 2: v/Uinf (wall-normal velocity) ───────────────────────
    fig, axes = plt.subplots(1, n_stations, figsize=(4 * n_stations, 7),
                              sharey=True)
    if n_stations == 1:
        axes = [axes]

    for i, xc in enumerate(STATIONS):
        ax = axes[i]
        if xc in piv:
            p = piv[xc]
            ax.plot(p["v/Uinf"], p["y/c"], "o", color=COLOR_EXP,
                    ms=2.5, alpha=0.6, label="PIV", zorder=5)
        for label, profs in sim_profiles.items():
            if xc not in profs:
                continue
            p = profs[xc]
            color, ls = COLORS_SIM.get(label, ("#333", "-"))
            ax.plot(p["v/Uinf"], p["y"], ls, color=color,
                    lw=1.5, label=f"SU2 {label}", zorder=4)
        ax.axvline(0, color="gray", ls="-", lw=0.5, alpha=0.4)
        ax.set_xlabel("$v / U_\\infty$")
        ax.set_title(f"$x/c = {xc}$", fontsize=13, fontweight="bold")
        ax.set_xlim(-0.3, 0.15)
        ax.set_ylim(0, Y_MAX)
        ax.grid(True, alpha=0.2)

    axes[0].set_ylabel("$y / c$")
    fig.suptitle("Wall-Mounted Hump — Wall-Normal Velocity Profiles",
                 fontsize=15, y=1.01)
    plt.tight_layout()
    fname2 = OUT_DIR / "wall_hump_vnormal_profiles.png"
    fig.savefig(fname2, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname2.name}")

    print(f"\n  Output: {OUT_DIR}")
    print("  Done!")


if __name__ == "__main__":
    main()
