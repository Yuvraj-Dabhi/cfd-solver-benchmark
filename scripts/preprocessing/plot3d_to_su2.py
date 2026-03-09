#!/usr/bin/env python3
"""
PLOT3D to SU2 Grid Converter
=============================
Converts NASA TMR structured PLOT3D 2D C-grids to SU2 native mesh format.

The TMR NACA 0012 grids are structured C-grids:
  - Grid wraps around airfoil from downstream, around lower surface,
    to upper surface, back to downstream
  - 1-to-1 connectivity in the wake
  - Farfield boundary ~500 chords away

PLOT3D format (2D, formatted, multi-grid):
  Line 1: nbl (number of blocks, typically 1)
  Line 2: idim, jdim
  Data:   x(i,j) for all i,j, then y(i,j) for all i,j

Usage:
    python plot3d_to_su2.py --input grid.p2dfmt --output mesh.su2
    python plot3d_to_su2.py --input grid.p2dfmt --output mesh.su2 --plot
"""

import sys
import gzip
import argparse
import numpy as np
from pathlib import Path
from typing import Tuple, Optional

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


# ============================================================================
# PLOT3D Reader
# ============================================================================

def read_plot3d_2d(filepath: Path) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """
    Read a 2D structured PLOT3D grid file.

    Supports both plain text (.p2dfmt) and gzip-compressed (.p2dfmt.gz)
    files. The TMR Family I grids are distributed as gzip archives.

    Format (TMR spec, formatted, MG, 2D with nbl=1):
        read(2,*) nbl
        read(2,*) (idim(n),jdim(n),n=1,nbl)
        do n=1,nbl
          read(2,*) ((x(i,j,n),i=1,idim(n)),j=1,jdim(n)),
                    ((y(i,j,n),i=1,idim(n)),j=1,jdim(n))
        enddo

    Note: double precision values are used in the files.

    Parameters
    ----------
    filepath : Path
        Path to the PLOT3D formatted file (.p2dfmt or .p2dfmt.gz).

    Returns
    -------
    x : ndarray, shape (idim, jdim)
        X-coordinates.
    y : ndarray, shape (idim, jdim)
        Y-coordinates.
    idim : int
        Number of points in i-direction (wrap-around + wake).
    jdim : int
        Number of points in j-direction (wall-normal).
    """
    filepath = Path(filepath)
    print(f"  Reading PLOT3D grid: {filepath.name}")

    # Choose opener: gzip for .gz files, plain text otherwise
    is_gzip = filepath.suffix == '.gz' or filepath.name.endswith('.gz')
    opener = gzip.open if is_gzip else open

    # Read all numeric values from the file
    values = []
    with opener(filepath, 'rt') as f:
        for line in f:
            for token in line.split():
                try:
                    values.append(float(token))
                except ValueError:
                    continue

    # Parse header
    nbl = int(values[0])
    idim = int(values[1])
    jdim = int(values[2])

    print(f"  Blocks: {nbl}, Dimensions: {idim} x {jdim}")
    print(f"  Total points: {idim * jdim}")

    # Parse coordinates
    n_pts = idim * jdim
    offset = 3  # After nbl, idim, jdim

    x_flat = np.array(values[offset:offset + n_pts])
    y_flat = np.array(values[offset + n_pts:offset + 2 * n_pts])

    # Reshape to 2D (Fortran ordering: i varies fastest)
    x = x_flat.reshape((jdim, idim)).T  # Now shape (idim, jdim)
    y = y_flat.reshape((jdim, idim)).T

    print(f"  X range: [{x.min():.6f}, {x.max():.6f}]")
    print(f"  Y range: [{y.min():.6f}, {y.max():.6f}]")

    return x, y, idim, jdim


def coarsen_grid(x: np.ndarray, y: np.ndarray,
                 n_coarsen: int = 1) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """
    Extract a coarser grid level by taking every other point.

    This is the TMR-standard method for generating coarser Family III
    levels: "simply every other grid point in each coordinate direction
    for each successively-coarser level."

    Parameters
    ----------
    x : ndarray, shape (idim, jdim)
        Fine grid X-coordinates.
    y : ndarray, shape (idim, jdim)
        Fine grid Y-coordinates.
    n_coarsen : int
        Number of coarsening steps (1 = every other point, 2 = every 4th, etc.)

    Returns
    -------
    x_c : ndarray
        Coarsened X-coordinates.
    y_c : ndarray
        Coarsened Y-coordinates.
    idim_c : int
        Coarsened i-dimension.
    jdim_c : int
        Coarsened j-dimension.
    """
    stride = 2 ** n_coarsen
    idim, jdim = x.shape

    # Verify the grid supports this coarsening level
    # For a valid coarsening, (idim - 1) and (jdim - 1) must be
    # divisible by stride
    if (idim - 1) % stride != 0 or (jdim - 1) % stride != 0:
        raise ValueError(
            f"Cannot coarsen {idim}x{jdim} grid by {n_coarsen} levels "
            f"(stride={stride}): ({idim}-1)={idim-1} and ({jdim}-1)={jdim-1} "
            f"must both be divisible by {stride}"
        )

    x_c = x[::stride, ::stride].copy()
    y_c = y[::stride, ::stride].copy()
    idim_c, jdim_c = x_c.shape

    print(f"  Coarsened: {idim}x{jdim} → {idim_c}x{jdim_c} "
          f"(stride={stride}, level +{n_coarsen})")

    return x_c, y_c, idim_c, jdim_c


def identify_boundaries(x: np.ndarray, y: np.ndarray, idim: int, jdim: int,
                         n_airfoil_pts: Optional[int] = None) -> dict:
    """
    Identify boundary markers in a C-grid topology.

    C-grid structure (i-direction wraps around):
      i=0 to i=n_wake-1          : wake (downstream to TE, lower)
      i=n_wake to i=n_wake+n_af  : airfoil surface (lower TE -> LE -> upper TE)
      i=n_wake+n_af to i=idim-1  : wake (TE, upper to downstream)

    j=0         : wall / wake cut (inner boundary)
    j=jdim-1    : farfield (outer boundary)

    Returns
    -------
    dict with boundary info including airfoil indices and farfield indices.
    """
    # The airfoil surface is on j=0, where the grid wraps around the body
    # In a C-grid, the first/last points in i at j=0 are in the wake
    # The airfoil is the contiguous section where points are ON the airfoil

    # Detect airfoil region: points on j=0 that are between x=0 and x=1
    # (airfoil chord is 0 to 1)
    x_wall = x[:, 0]
    y_wall = y[:, 0]

    # Find points close to the airfoil (within chord length + small tolerance)
    on_airfoil = (x_wall >= -0.01) & (x_wall <= 1.01)

    # Find contiguous airfoil region
    airfoil_indices = np.where(on_airfoil)[0]

    if n_airfoil_pts is not None:
        # Use known count from TMR spec
        # Center the airfoil region
        center_i = idim // 2
        half = n_airfoil_pts // 2
        i_start = center_i - half
        i_end = i_start + n_airfoil_pts - 1
    else:
        # Auto-detect
        i_start = airfoil_indices[0]
        i_end = airfoil_indices[-1]
        n_airfoil_pts = i_end - i_start + 1

    n_wake_pts = idim - n_airfoil_pts

    print(f"  Airfoil: i=[{i_start}, {i_end}] ({n_airfoil_pts} points)")
    print(f"  Wake:    {n_wake_pts} points (split at i=0..{i_start-1} and i={i_end+1}..{idim-1})")
    print(f"  Farfield: j={jdim-1} ({idim} points)")

    return {
        "airfoil_i_start": i_start,
        "airfoil_i_end": i_end,
        "n_airfoil_pts": n_airfoil_pts,
        "n_wake_pts": n_wake_pts,
        "jdim": jdim,
        "idim": idim,
    }


# ============================================================================
# SU2 Mesh Writer
# ============================================================================

def convert_to_su2(x: np.ndarray, y: np.ndarray, idim: int, jdim: int,
                    output_path: Path, boundaries: dict) -> None:
    """
    Convert structured PLOT3D C-grid to SU2 unstructured mesh format.

    C-grid wake handling: In a C-grid, the wake cut at j=0 has duplicate
    nodes where the lower (i < i_start) and upper (i > i_end) wake lines
    coincide. At j=0 the coordinates are identical; at j>0 they diverge
    (lower side goes to -y, upper side to +y).

    We merge the DUPLICATE nodes at j=0 only: for each upper-wake index
    i_upper = idim - 1 - k (k = 0..n_wake-1), the j=0 node is remapped
    to the corresponding lower-wake node at i_lower = k, j=0. This
    eliminates the duplicate cut-line nodes. No wake boundary marker is
    written, matching the standard SU2 C-grid mesh convention.

    SU2 element types:
      - Line (3): 2 nodes (boundary marker)
      - Quadrilateral (9): 4 nodes

    Reference: SU2 tutorial mesh n0012_897-257.su2 uses identical approach.
    """
    print(f"\n  Converting to SU2 format...")

    i_start = boundaries["airfoil_i_start"]
    i_end = boundaries["airfoil_i_end"]
    n_wake = i_start  # number of wake points per side

    # Build node ID mapping: old (i,j) -> new sequential ID
    # Merge upper-wake j=0 nodes onto lower-wake j=0 nodes
    old_to_new = {}
    new_coords = []
    new_id = 0

    for j in range(jdim):
        for i in range(idim):
            if j == 0 and i > i_end:
                # Upper-wake j=0 node maps to lower-wake j=0 node
                i_lower = idim - 1 - i
                old_to_new[(i, j)] = old_to_new[(i_lower, 0)]
            else:
                old_to_new[(i, j)] = new_id
                new_coords.append((x[i, j], y[i, j]))
                new_id += 1

    n_nodes = new_id
    n_merged = idim * jdim - n_nodes
    n_elems = (idim - 1) * (jdim - 1)

    print(f"  Wake nodes merged at j=0: {n_merged}")

    with open(output_path, 'w') as f:
        # Dimension
        f.write("% Problem dimension\n")
        f.write("NDIME= 2\n")

        # Elements (quadrilaterals)
        f.write(f"% Inner element connectivity\nNELEM= {n_elems}\n")
        elem_idx = 0
        for j in range(jdim - 1):
            for i in range(idim - 1):
                n0 = old_to_new[(i, j)]
                n1 = old_to_new[(i + 1, j)]
                n2 = old_to_new[(i + 1, j + 1)]
                n3 = old_to_new[(i, j + 1)]
                f.write(f"9\t{n0}\t{n1}\t{n2}\t{n3}\t{elem_idx}\n")
                elem_idx += 1

        # Node coordinates
        f.write(f"% Node coordinates\nNPOIN= {n_nodes}\n")
        for nid, (px, py) in enumerate(new_coords):
            f.write(f"{px:.15e}\t{py:.15e}\t{nid}\n")

        # Boundary markers: only airfoil and farfield (no wake boundary)
        n_airfoil_segs = i_end - i_start
        # Farfield includes: j=jdim-1 row + i=0 column + i=idim-1 column
        n_farfield_top = idim - 1          # j=jdim-1 row
        n_farfield_left = jdim - 1         # i=0 column
        n_farfield_right = jdim - 1        # i=idim-1 column
        n_farfield_segs = n_farfield_top + n_farfield_left + n_farfield_right

        f.write(f"% Boundary elements\nNMARK= 2\n")

        # --- Airfoil wall ---
        f.write("MARKER_TAG= airfoil\n")
        f.write(f"MARKER_ELEMS= {n_airfoil_segs}\n")
        for i in range(i_start, i_end):
            n0 = old_to_new[(i, 0)]
            n1 = old_to_new[(i + 1, 0)]
            f.write(f"3\t{n0}\t{n1}\n")

        # --- Farfield (top row + left column + right column) ---
        f.write("MARKER_TAG= farfield\n")
        f.write(f"MARKER_ELEMS= {n_farfield_segs}\n")
        # Top row: j=jdim-1
        for i in range(idim - 1):
            n0 = old_to_new[(i, jdim - 1)]
            n1 = old_to_new[(i + 1, jdim - 1)]
            f.write(f"3\t{n0}\t{n1}\n")
        # Left column: i=0, j=0..jdim-2
        for j in range(jdim - 1):
            n0 = old_to_new[(0, j)]
            n1 = old_to_new[(0, j + 1)]
            f.write(f"3\t{n0}\t{n1}\n")
        # Right column: i=idim-1, j=0..jdim-2
        for j in range(jdim - 1):
            n0 = old_to_new[(idim - 1, j)]
            n1 = old_to_new[(idim - 1, j + 1)]
            f.write(f"3\t{n0}\t{n1}\n")

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  Written: {output_path.name} ({size_mb:.1f} MB)")
    print(f"  Nodes:    {n_nodes} (from {idim * jdim}, {n_merged} merged)")
    print(f"  Elements: {n_elems} (quads)")
    print(f"  Markers:  airfoil ({n_airfoil_segs} segs), "
          f"farfield ({n_farfield_segs} segs = {n_farfield_top} top + "
          f"{n_farfield_left} left + {n_farfield_right} right)")
    print(f"  Wake:     {n_merged} j=0 nodes merged (no boundary marker)")


# ============================================================================
# Visualization
# ============================================================================

def plot_grid(x: np.ndarray, y: np.ndarray, boundaries: dict,
              output_path: Optional[Path] = None) -> None:
    """Plot the grid to verify conversion."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [SKIP]   matplotlib not available for plotting")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Full grid view
    ax = axes[0]
    ax.set_title("Full Grid (every 4th line)")
    step = max(1, x.shape[0] // 50)
    for i in range(0, x.shape[0], step):
        ax.plot(x[i, :], y[i, :], 'b-', linewidth=0.3, alpha=0.5)
    for j in range(0, x.shape[1], step):
        ax.plot(x[:, j], y[:, j], 'b-', linewidth=0.3, alpha=0.5)
    ax.set_aspect('equal')
    ax.set_xlim(-2, 3)
    ax.set_ylim(-2, 2)
    ax.set_xlabel('x/c')
    ax.set_ylabel('y/c')

    # Close-up of airfoil
    ax = axes[1]
    ax.set_title("Airfoil Close-up")
    i_s = boundaries["airfoil_i_start"]
    i_e = boundaries["airfoil_i_end"]
    n_show_j = min(20, x.shape[1])

    for i in range(i_s, i_e, max(1, (i_e - i_s) // 40)):
        ax.plot(x[i, :n_show_j], y[i, :n_show_j], 'b-', linewidth=0.3)
    for j in range(n_show_j):
        ax.plot(x[i_s:i_e, j], y[i_s:i_e, j], 'b-', linewidth=0.3)

    # Highlight airfoil surface
    ax.plot(x[i_s:i_e+1, 0], y[i_s:i_e+1, 0], 'r-', linewidth=1.5,
            label='Airfoil surface')
    ax.set_aspect('equal')
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.15, 0.15)
    ax.set_xlabel('x/c')
    ax.set_ylabel('y/c')
    ax.legend(fontsize=8)

    plt.tight_layout()
    save_path = output_path or Path("grid_preview.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Grid preview saved: {save_path.name}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Convert PLOT3D 2D C-grid to SU2 mesh format"
    )
    parser.add_argument("--input", "-i", type=Path, required=True,
                        help="Input PLOT3D grid file")
    parser.add_argument("--output", "-o", type=Path, default=None,
                        help="Output SU2 mesh file (default: same name .su2)")
    parser.add_argument("--airfoil-pts", type=int, default=None,
                        help="Number of airfoil surface points (from TMR spec)")
    parser.add_argument("--plot", action="store_true",
                        help="Generate grid preview plot")
    parser.add_argument("--plot-output", type=Path, default=None,
                        help="Path for grid preview plot")
    args = parser.parse_args()

    print("=" * 60)
    print("  PLOT3D to SU2 Grid Converter")
    print("=" * 60)

    # Read PLOT3D grid
    x, y, idim, jdim = read_plot3d_2d(args.input)

    # Identify boundaries
    boundaries = identify_boundaries(x, y, idim, jdim, args.airfoil_pts)

    # Convert to SU2
    output = args.output or args.input.with_suffix('.su2')
    convert_to_su2(x, y, idim, jdim, output, boundaries)

    # Optional plot
    if args.plot:
        plot_grid(x, y, boundaries, args.plot_output)

    print(f"\n  [OK] Conversion complete.")
    print(f"  SU2 mesh: {output}")


if __name__ == "__main__":
    main()
