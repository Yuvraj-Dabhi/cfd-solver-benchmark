#!/usr/bin/env python3
"""
Grid Visualization for NACA 0012 SU2 Meshes
=============================================

Reads SU2 mesh files and generates publication-quality grid plots showing:
  1. Full domain view (farfield extent)
  2. Airfoil close-up with mesh cells
  3. Leading edge detail
  4. Trailing edge detail

Usage:
    python scripts/postprocessing/visualize_grid.py                        # default: Family I medium
    python scripts/postprocessing/visualize_grid.py --family I --grid fine  # specific family/level
    python scripts/postprocessing/visualize_grid.py --all                   # compare all 3 families
    python scripts/postprocessing/visualize_grid.py --file path/to/mesh.su2
"""

import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from typing import Optional, Tuple, Dict, List

# --------------------------------------------------------------------------
# SU2 Mesh Reader
# --------------------------------------------------------------------------

def read_su2_mesh(filepath: Path) -> Dict:
    """
    Read an SU2 mesh file and extract nodes, elements, and markers.

    Returns dict with keys:
        nodes:    (N, 2) array of node coordinates
        quads:    (M, 4) array of quad element connectivity
        markers:  dict of marker_name -> list of edge segments
        ni, nj:   structured grid dimensions (if detectable)
    """
    nodes = []
    quads = []
    markers = {}

    with open(filepath, 'r') as f:
        lines = f.readlines()

    i = 0
    n_lines = len(lines)

    while i < n_lines:
        line = lines[i].strip()

        # Parse elements
        if line.startswith("NELEM="):
            n_elem = int(line.split("=")[1])
            for j in range(n_elem):
                i += 1
                parts = lines[i].strip().split()
                elem_type = int(parts[0])
                if elem_type == 9:  # Quad
                    quads.append([int(parts[1]), int(parts[2]),
                                  int(parts[3]), int(parts[4])])

        # Parse nodes
        elif line.startswith("NPOIN="):
            n_points = int(line.split("=")[1].split()[0])
            for j in range(n_points):
                i += 1
                parts = lines[i].strip().split()
                nodes.append([float(parts[0]), float(parts[1])])

        # Parse markers
        elif line.startswith("MARKER_TAG="):
            marker_name = line.split("=")[1].strip()
            i += 1
            n_elems = int(lines[i].strip().split("=")[1])
            edges = []
            for j in range(n_elems):
                i += 1
                parts = lines[i].strip().split()
                edges.append([int(parts[1]), int(parts[2])])
            markers[marker_name] = edges

        i += 1

    nodes_arr = np.array(nodes)
    quads_arr = np.array(quads) if quads else np.zeros((0, 4), dtype=int)

    # Try to detect structured dimensions from filename
    ni, nj = None, None
    stem = filepath.stem
    parts = stem.split("_")
    for p in parts:
        if "x" in p:
            try:
                dims = p.split("x")
                ni, nj = int(dims[0]), int(dims[1])
            except (ValueError, IndexError):
                pass

    return {
        "nodes": nodes_arr,
        "quads": quads_arr,
        "markers": markers,
        "ni": ni,
        "nj": nj,
        "filepath": filepath,
    }


def get_structured_lines(nodes: np.ndarray, ni: int, nj: int
                         ) -> Tuple[List, List]:
    """
    Extract i-lines and j-lines from a structured grid stored in flat order.

    Node ordering assumed: node[i + j*ni] for i in [0..ni-1], j in [0..nj-1]
    """
    i_lines = []  # Lines of constant j
    j_lines = []  # Lines of constant i

    # j-lines (constant i)
    for i in range(ni):
        indices = [i + j * ni for j in range(nj)]
        j_lines.append(nodes[indices])

    # i-lines (constant j)
    for j in range(nj):
        indices = [i + j * ni for i in range(ni)]
        i_lines.append(nodes[indices])

    return i_lines, j_lines


# --------------------------------------------------------------------------
# Plotting Functions
# --------------------------------------------------------------------------

COLORS = {
    "grid": "#336699",
    "airfoil": "#000000",
    "farfield": "#999999",
    "wake": "#cc4400",
    "bg": "#ffffff",
}


def plot_grid_views(mesh: Dict, output_dir: Path, label: str = "",
                    dark_mode: bool = False) -> List[Path]:
    """
    Generate a 4-panel grid visualization:
      Top-left:     Full domain
      Top-right:    Airfoil close-up
      Bottom-left:  Leading edge detail
      Bottom-right: Trailing edge detail

    Returns list of saved file paths.
    """
    nodes = mesh["nodes"]
    ni, nj = mesh.get("ni"), mesh.get("nj")

    # Style setup — white background
    plt.style.use('default')
    bg_color = "#ffffff"
    grid_color = "#2266aa"
    grid_lw = 0.2
    airfoil_color = "#000000"
    text_color = "#1a1a1a"
    wake_color = COLORS["wake"]
    farfield_color = COLORS["farfield"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.patch.set_facecolor(bg_color)

    for ax in axes.flat:
        ax.set_facecolor(bg_color)
        ax.set_aspect('equal')
        ax.tick_params(colors=text_color, labelsize=8)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_color("#cccccc")
            spine.set_linewidth(0.5)

    # --- Draw grid lines ---
    if ni and nj and ni * nj == len(nodes):
        i_lines, j_lines = get_structured_lines(nodes, ni, nj)

        for ax in axes.flat:
            for line_pts in i_lines:
                ax.plot(line_pts[:, 0], line_pts[:, 1],
                        color=grid_color, linewidth=grid_lw, zorder=1)
            for line_pts in j_lines:
                ax.plot(line_pts[:, 0], line_pts[:, 1],
                        color=grid_color, linewidth=grid_lw, zorder=1)
    else:
        # Unstructured: draw quad edges
        quads = mesh["quads"]
        for ax in axes.flat:
            for q in quads:
                corners = nodes[q]
                # Close the quad
                poly = np.vstack([corners, corners[0:1]])
                ax.plot(poly[:, 0], poly[:, 1],
                        color=grid_color, linewidth=grid_lw, zorder=1)

    # --- Draw markers ---
    for ax in axes.flat:
        for marker_name, edges in mesh["markers"].items():
            if not edges:
                continue
            if marker_name == "airfoil":
                color = airfoil_color
                lw = 2.0
            elif marker_name == "wake":
                color = wake_color
                lw = 1.0
            elif marker_name == "farfield":
                color = farfield_color
                lw = 0.8
            else:
                color = "#888888"
                lw = 0.8

            for e in edges:
                p0, p1 = nodes[e[0]], nodes[e[1]]
                ax.plot([p0[0], p1[0]], [p0[1], p1[1]],
                        color=color, linewidth=lw, zorder=5)

    # --- Set view limits ---
    # Panel 1: Full domain
    ax1 = axes[0, 0]
    x_range = nodes[:, 0].max() - nodes[:, 0].min()
    y_range = nodes[:, 1].max() - nodes[:, 1].min()
    margin = 0.05
    ax1.set_xlim(nodes[:, 0].min() - margin * x_range,
                 nodes[:, 0].max() + margin * x_range)
    ax1.set_ylim(nodes[:, 1].min() - margin * y_range,
                 nodes[:, 1].max() + margin * y_range)
    ax1.set_title("Full Domain (~500c extent)",
                   fontsize=11, fontweight='bold', color=text_color)
    ax1.set_xlabel("x", fontsize=9)
    ax1.set_ylabel("y", fontsize=9)

    # Panel 2: Airfoil close-up
    ax2 = axes[0, 1]
    ax2.set_xlim(-0.15, 1.25)
    ax2.set_ylim(-0.4, 0.4)
    ax2.set_title("Airfoil Close-Up",
                   fontsize=11, fontweight='bold', color=text_color)
    ax2.set_xlabel("x/c", fontsize=9)
    ax2.set_ylabel("y/c", fontsize=9)

    # Panel 3: Leading edge
    ax3 = axes[1, 0]
    ax3.set_xlim(-0.02, 0.08)
    ax3.set_ylim(-0.04, 0.04)
    ax3.set_title("Leading Edge Detail",
                   fontsize=11, fontweight='bold', color=text_color)
    ax3.set_xlabel("x/c", fontsize=9)
    ax3.set_ylabel("y/c", fontsize=9)

    # Panel 4: Trailing edge
    ax4 = axes[1, 1]
    ax4.set_xlim(0.92, 1.05)
    ax4.set_ylim(-0.03, 0.03)
    ax4.set_title("Trailing Edge Detail",
                   fontsize=11, fontweight='bold', color=text_color)
    ax4.set_xlabel("x/c", fontsize=9)
    ax4.set_ylabel("y/c", fontsize=9)

    # --- Title ---
    dims_str = f"{ni}x{nj}" if ni and nj else f"{len(nodes)} nodes"
    title = f"NACA 0012 Grid: {dims_str}"
    if label:
        title += f" -- {label}"
    fig.suptitle(title, fontsize=14, fontweight='bold', color=text_color, y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save
    safe_label = label.replace(" ", "_").lower() if label else dims_str
    filepath = output_dir / f"grid_{safe_label}.png"
    plt.savefig(filepath, dpi=200, bbox_inches='tight', facecolor=bg_color)
    plt.close()
    print(f"  [OK] Saved: {filepath.name} ({filepath.stat().st_size / 1024:.0f} KB)")

    return [filepath]


def plot_family_comparison(meshes: List[Dict], labels: List[str],
                           output_dir: Path) -> Path:
    """
    Compare LE and TE detail across multiple grid families side by side.
    """
    n = len(meshes)
    fig, axes = plt.subplots(2, n, figsize=(6 * n, 8))
    if n == 1:
        axes = axes.reshape(2, 1)

    plt.style.use('default')
    bg_color = "#ffffff"
    text_color = "#1a1a1a"
    fig.patch.set_facecolor(bg_color)

    for col, (mesh, label) in enumerate(zip(meshes, labels)):
        nodes = mesh["nodes"]
        ni, nj = mesh.get("ni"), mesh.get("nj")

        for row in range(2):
            ax = axes[row, col]
            ax.set_facecolor(bg_color)
            ax.set_aspect('equal')
            ax.tick_params(colors=text_color, labelsize=7)
            ax.grid(False)
            for spine in ax.spines.values():
                spine.set_color("#cccccc")
                spine.set_linewidth(0.5)

            # Draw grid
            grid_color = "#2266aa"
            if ni and nj and ni * nj == len(nodes):
                i_lines, j_lines = get_structured_lines(nodes, ni, nj)
                for lpts in i_lines:
                    ax.plot(lpts[:, 0], lpts[:, 1],
                            color=grid_color, linewidth=0.25)
                for lpts in j_lines:
                    ax.plot(lpts[:, 0], lpts[:, 1],
                            color=grid_color, linewidth=0.25)

            # Draw airfoil
            for marker_name, edges in mesh["markers"].items():
                if marker_name == "airfoil":
                    for e in edges:
                        p0, p1 = nodes[e[0]], nodes[e[1]]
                        ax.plot([p0[0], p1[0]], [p0[1], p1[1]],
                                color="#000000", linewidth=1.8, zorder=5)

            # Set view
            if row == 0:  # LE
                ax.set_xlim(-0.015, 0.06)
                ax.set_ylim(-0.03, 0.03)
                ax.set_title(f"{label} -- LE",
                             fontsize=10, fontweight='bold', color=text_color)
            else:  # TE
                ax.set_xlim(0.93, 1.04)
                ax.set_ylim(-0.025, 0.025)
                ax.set_title(f"{label} -- TE",
                             fontsize=10, fontweight='bold', color=text_color)

    dims = f"{meshes[0].get('ni', '?')}x{meshes[0].get('nj', '?')}"
    fig.suptitle(f"Grid Family Comparison ({dims})",
                 fontsize=13, fontweight='bold', color=text_color, y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    filepath = output_dir / f"grid_family_comparison_{dims}.png"
    plt.savefig(filepath, dpi=200, bbox_inches='tight', facecolor=bg_color)
    plt.close()
    print(f"  [OK] Saved: {filepath.name}")
    return filepath


def plot_grid_convergence(grids_dir: Path, family: str,
                          output_dir: Path) -> Path:
    """
    Show grid refinement progression for a single family (coarse -> fine).
    """
    from run_naca0012 import build_grid_levels, GRID_LEVEL_SPECS

    levels_to_show = ["coarse", "medium", "fine", "xfine"]
    levels = build_grid_levels(family)

    meshes = []
    labels = []
    for level_name in levels_to_show:
        if level_name not in levels:
            continue
        cfg = levels[level_name]
        ni, nj = cfg["dims"]
        mesh_file = grids_dir / f"naca0012_fam{family}_{ni}x{nj}.su2"
        if mesh_file.exists():
            print(f"  Reading {mesh_file.name}...")
            meshes.append(read_su2_mesh(mesh_file))
            labels.append(f"{level_name} ({ni}x{nj})")

    if not meshes:
        print("  [SKIP] No mesh files found")
        return None

    n = len(meshes)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    plt.style.use('default')
    bg_color = "#ffffff"
    text_color = "#1a1a1a"
    fig.patch.set_facecolor(bg_color)

    for idx, (mesh, label) in enumerate(zip(meshes, labels)):
        ax = axes[idx]
        ax.set_facecolor(bg_color)
        ax.set_aspect('equal')
        ax.tick_params(colors=text_color, labelsize=7)
        ax.grid(False)

        nodes = mesh["nodes"]
        ni, nj = mesh.get("ni"), mesh.get("nj")

        if ni and nj and ni * nj == len(nodes):
            i_lines, j_lines = get_structured_lines(nodes, ni, nj)
            for lpts in i_lines:
                ax.plot(lpts[:, 0], lpts[:, 1],
                        color="#2266aa", linewidth=0.3)
            for lpts in j_lines:
                ax.plot(lpts[:, 0], lpts[:, 1],
                        color="#2266aa", linewidth=0.3)

        # Airfoil
        for marker_name, edges in mesh["markers"].items():
            if marker_name == "airfoil":
                for e in edges:
                    p0, p1 = nodes[e[0]], nodes[e[1]]
                    ax.plot([p0[0], p1[0]], [p0[1], p1[1]],
                            color="#000000", linewidth=1.5, zorder=5)

        ax.set_xlim(-0.1, 1.15)
        ax.set_ylim(-0.35, 0.35)
        ax.set_title(label, fontsize=10, fontweight='bold', color=text_color)

    fig.suptitle(f"Grid Refinement -- Family {family}",
                 fontsize=13, fontweight='bold', color=text_color, y=1.02)
    plt.tight_layout()

    filepath = output_dir / f"grid_refinement_fam{family}.png"
    plt.savefig(filepath, dpi=200, bbox_inches='tight', facecolor=bg_color)
    plt.close()
    print(f"  [OK] Saved: {filepath.name}")
    return filepath


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Visualize NACA 0012 SU2 grids",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python scripts/postprocessing/visualize_grid.py                         # Family I medium
  python scripts/postprocessing/visualize_grid.py --family I --grid fine   # Family I fine
  python scripts/postprocessing/visualize_grid.py --compare               # Compare all 3 families
  python scripts/postprocessing/visualize_grid.py --convergence           # Grid refinement series
  python scripts/postprocessing/visualize_grid.py --file mesh.su2         # Any SU2 file
"""
    )
    parser.add_argument("--family", default="I", choices=["I", "II", "III"],
                        help="Grid family (default: I)")
    parser.add_argument("--grid", default="medium",
                        help="Grid level (default: medium)")
    parser.add_argument("--compare", action="store_true",
                        help="Compare all 3 families at same grid level")
    parser.add_argument("--convergence", action="store_true",
                        help="Show grid refinement series for one family")
    parser.add_argument("--file", type=Path, default=None,
                        help="Directly specify an SU2 mesh file")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output directory for plots")
    args = parser.parse_args()

    # Project paths
    project_root = Path(__file__).parent.parent.parent.resolve()
    grids_dir = project_root / "experimental_data" / "naca0012" / "grids"
    output_dir = args.output or (project_root / "results" / "grid_plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(project_root))

    print("=" * 60)
    print("  NACA 0012 Grid Visualization")
    print("=" * 60)

    if args.file:
        # Direct file mode
        print(f"\n  Reading: {args.file}")
        mesh = read_su2_mesh(args.file)
        print(f"  Nodes: {len(mesh['nodes']):,}, "
              f"Quads: {len(mesh['quads']):,}")
        plot_grid_views(mesh, output_dir, label=args.file.stem)

    elif args.compare:
        # Family comparison
        from run_naca0012 import build_grid_levels
        level_info = build_grid_levels("I")[args.grid]
        ni, nj = level_info["dims"]

        meshes = []
        labels = []
        for fam in ["I", "II", "III"]:
            mesh_file = grids_dir / f"naca0012_fam{fam}_{ni}x{nj}.su2"
            if mesh_file.exists():
                print(f"\n  Reading Family {fam}: {mesh_file.name}")
                m = read_su2_mesh(mesh_file)
                meshes.append(m)
                labels.append(f"Family {fam}")
                print(f"  Nodes: {len(m['nodes']):,}")
            else:
                print(f"  [SKIP] {mesh_file.name} not found")

        if len(meshes) >= 2:
            plot_family_comparison(meshes, labels, output_dir)

        # Also generate individual 4-panel views
        for m, lbl in zip(meshes, labels):
            plot_grid_views(m, output_dir, label=f"{lbl} {ni}x{nj}")

    elif args.convergence:
        # Grid refinement series
        plot_grid_convergence(grids_dir, args.family, output_dir)

    else:
        # Single grid
        from run_naca0012 import build_grid_levels
        levels = build_grid_levels(args.family)
        if args.grid not in levels:
            print(f"  [ERROR] Grid level '{args.grid}' not found")
            return
        ni, nj = levels[args.grid]["dims"]
        mesh_file = grids_dir / f"naca0012_fam{args.family}_{ni}x{nj}.su2"

        if not mesh_file.exists():
            print(f"  [ERROR] Mesh file not found: {mesh_file}")
            print(f"  Run: python run_naca0012.py --grid-family {args.family} "
                  f"--grid {args.grid} --dry-run")
            return

        print(f"\n  Reading: {mesh_file.name}")
        mesh = read_su2_mesh(mesh_file)
        print(f"  Nodes: {len(mesh['nodes']):,}, "
              f"Quads: {len(mesh['quads']):,}")
        print(f"  Markers: {', '.join(mesh['markers'].keys())}")
        plot_grid_views(mesh, output_dir,
                        label=f"Family {args.family} {args.grid}")

    print(f"\n  Output: {output_dir}")
    print("  Done!")


if __name__ == "__main__":
    main()
