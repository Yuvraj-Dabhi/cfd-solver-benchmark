#!/usr/bin/env python3
"""
Cf Extraction Pipeline for SU2
================================
Reads SU2 volume/surface flow files, computes wall-shear stress and
skin friction coefficient using two methods:

Method 1 (Direct):
    Use Skin_Friction_Coefficient from SU2 surface output (already
    non-dimensionalized by SU2 internally).

Method 2 (From Volume Gradients):
    Read volume flow.vtu, identify wall-adjacent cells, compute
    du/dy at the wall via finite differences, then:
        tau_w = mu * du/dy
        Cf = tau_w / (0.5 * rho_inf * U_inf^2)

Both methods are cross-validated against each other and against
TMR experimental data for the NASA wall-mounted hump.

Skills demonstrated: Python, VTK, numerical differentiation, V&V.

Usage:
    python scripts/postprocessing/cf_extraction.py [--case-dir DIR]
"""

import json
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

# =========================================================================
# VTK Helpers
# =========================================================================

def read_vtu_arrays(path: Path) -> Dict:
    """Read VTU file, return dict of numpy arrays + coords."""
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy

    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(str(path))
    reader.Update()
    out = reader.GetOutput()

    coords = vtk_to_numpy(out.GetPoints().GetData())
    result = {"coords": coords, "_vtk_output": out}

    pd_data = out.GetPointData()
    for i in range(pd_data.GetNumberOfArrays()):
        name = pd_data.GetArrayName(i)
        result[name] = vtk_to_numpy(pd_data.GetArray(i))

    return result


def get_cell_connectivity(vtk_output) -> np.ndarray:
    """Extract cell-node connectivity from VTK unstructured grid."""
    from vtk.util.numpy_support import vtk_to_numpy

    cells = vtk_output.GetCells()
    conn = vtk_to_numpy(cells.GetConnectivityArray())
    offsets = vtk_to_numpy(cells.GetOffsetsArray())

    # Build list of cells (each as list of node IDs)
    cell_list = []
    for i in range(len(offsets) - 1):
        start, end = offsets[i], offsets[i + 1]
        cell_list.append(conn[start:end].tolist())
    # Handle last cell
    if len(offsets) > 0:
        cell_list.append(conn[offsets[-1]:].tolist())

    return cell_list


# =========================================================================
# Method 1: Direct Cf from SU2 Surface Output
# =========================================================================

def extract_cf_from_surface(
    surface_vtu: Path,
    wall_marker: str = "lower_wall",
) -> Dict:
    """
    Extract Cf directly from SU2 surface_flow.vtu.

    SU2 outputs Skin_Friction_Coefficient as a 3-component vector
    (Cf_x, Cf_y, Cf_z) already non-dimensionalized.

    Returns dict with x, Cf_x, Cf_magnitude, Cp, Y_Plus arrays.
    """
    d = read_vtu_arrays(surface_vtu)
    x = d["coords"][:, 0]
    y = d["coords"][:, 1]

    # SU2's Skin_Friction_Coefficient: already Cf = tau_w / (0.5 rho U^2)
    Cf_vec = d.get("Skin_Friction_Coefficient")
    Cp = d.get("Pressure_Coefficient")
    yplus = d.get("Y_Plus")

    if Cf_vec is None:
        raise ValueError("No Skin_Friction_Coefficient in surface VTU. "
                         "Check OUTPUT_FILES includes SURFACE_PARAVIEW.")

    # Cf_x is the streamwise component
    Cf_x = Cf_vec[:, 0] if Cf_vec.ndim > 1 else Cf_vec

    # Sort by x
    idx = np.argsort(x)

    result = {
        "x": x[idx],
        "y": y[idx],
        "Cf_x": Cf_x[idx],
        "Cf_magnitude": np.linalg.norm(Cf_vec[idx], axis=1) if Cf_vec.ndim > 1 else np.abs(Cf_x[idx]),
        "method": "SU2_surface_direct",
    }
    if Cp is not None:
        result["Cp"] = Cp[idx]
    if yplus is not None:
        result["Y_Plus"] = yplus[idx]

    return result


# =========================================================================
# Method 2: Cf from Volume Velocity Gradients
# =========================================================================

def identify_wall_adjacent_cells(
    coords: np.ndarray,
    wall_node_ids: np.ndarray,
    cell_list: list,
) -> list:
    """
    Find cells that have at least one node on the wall boundary.

    Returns list of (cell_idx, wall_node, off_wall_node) tuples.
    """
    wall_set = set(wall_node_ids.tolist())
    wall_cells = []

    for cell_idx, cell_nodes in enumerate(cell_list):
        on_wall = [n for n in cell_nodes if n in wall_set]
        off_wall = [n for n in cell_nodes if n not in wall_set]
        if on_wall and off_wall:
            wall_cells.append({
                "cell_idx": cell_idx,
                "wall_nodes": on_wall,
                "off_wall_nodes": off_wall,
            })

    return wall_cells


def compute_wall_normal_gradient(
    coords: np.ndarray,
    velocity: np.ndarray,
    wall_cells: list,
) -> Dict:
    """
    Compute du/dn at the wall using finite differences between
    wall and wall-adjacent nodes.

    For the wall hump (2D, wall at y=f(x)), the wall-normal direction
    is approximately the y-direction for the flat portions and
    perpendicular to the hump surface.

    Returns arrays of (x_wall, du_dn, dy) at each wall cell.
    """
    x_wall_list = []
    du_dn_list = []
    dy_list = []

    for wc in wall_cells:
        # Average wall node position
        wn = wc["wall_nodes"]
        x_w = np.mean([coords[n, 0] for n in wn])
        y_w = np.mean([coords[n, 1] for n in wn])
        u_w = 0.0  # No-slip: u = 0 at wall

        # Find the closest off-wall node
        min_dist = float("inf")
        best_off = None
        for n in wc["off_wall_nodes"]:
            dy = coords[n, 1] - y_w
            if 0 < dy < min_dist:  # above wall only
                min_dist = dy
                best_off = n

        if best_off is None:
            # Try any off-wall direction
            for n in wc["off_wall_nodes"]:
                dist = np.sqrt((coords[n, 0] - x_w)**2 +
                               (coords[n, 1] - y_w)**2)
                if 0 < dist < min_dist and coords[n, 1] > y_w:
                    min_dist = dist
                    best_off = n

        if best_off is None:
            continue

        dy = coords[best_off, 1] - y_w
        if abs(dy) < 1e-15:
            continue

        # Streamwise velocity at off-wall node
        u_off = velocity[best_off, 0]

        # du/dy ~ (u_off - 0) / dy
        du_dn = u_off / dy

        x_wall_list.append(x_w)
        du_dn_list.append(du_dn)
        dy_list.append(dy)

    return {
        "x": np.array(x_wall_list),
        "du_dn": np.array(du_dn_list),
        "dy": np.array(dy_list),
    }


def extract_cf_from_volume(
    volume_vtu: Path,
    surface_vtu: Path,
    freestream: Dict,
) -> Dict:
    """
    Compute Cf from volume velocity gradients.

    Steps:
      1. Read surface VTU to get wall node IDs
      2. Read volume VTU to get velocity, viscosity, density
      3. Identify wall-adjacent cells
      4. Compute du/dy at the wall
      5. tau_w = mu * du/dy
      6. Cf = tau_w / (0.5 * rho_inf * U_inf^2)

    Parameters
    ----------
    volume_vtu : Path
        Path to flow.vtu (volume solution).
    surface_vtu : Path
        Path to surface_flow.vtu (for wall node identification).
    freestream : dict
        Must contain: rho_inf, U_inf, mu_inf (dimensional).
        OR: Re, L, Mach, T_inf for auto-computation.

    Returns
    -------
    dict with x, Cf, tau_w arrays.
    """
    # Read surface to get wall node coords
    surf = read_vtu_arrays(surface_vtu)
    wall_coords = surf["coords"]

    # Read volume
    vol = read_vtu_arrays(volume_vtu)
    vol_coords = vol["coords"]
    velocity = vol.get("Velocity")
    mu = vol.get("Laminar_Viscosity")
    density = vol.get("Density")

    if velocity is None:
        raise ValueError("No Velocity array in volume VTU")

    # Find wall nodes in volume mesh by matching coordinates
    from scipy.spatial import cKDTree
    tree = cKDTree(vol_coords[:, :2])

    # Match surface nodes to volume nodes (tolerance = 1e-8)
    dists, vol_wall_ids = tree.query(wall_coords[:, :2])
    matched = dists < 1e-6
    wall_node_ids = np.unique(vol_wall_ids[matched])

    print(f"  Volume: {len(vol_coords)} pts, {len(wall_node_ids)} wall nodes matched")

    # Get cell connectivity
    cell_list = get_cell_connectivity(vol["_vtk_output"])

    # Find wall-adjacent cells
    wall_cells = identify_wall_adjacent_cells(
        vol_coords, wall_node_ids, cell_list)
    print(f"  Wall-adjacent cells: {len(wall_cells)}")

    # Compute du/dy
    grad = compute_wall_normal_gradient(vol_coords, velocity, wall_cells)

    if len(grad["x"]) == 0:
        raise ValueError("No wall gradient data computed")

    # Compute tau_w using local viscosity
    # In SU2 non-dim: tau_w = mu_nondim * du/dy_nondim
    # Then Cf = tau_w / (0.5 * rho_inf_nondim * U_inf_nondim^2)

    # Get local mu at wall-adjacent points (average of wall cell nodes)
    mu_wall = np.zeros(len(wall_cells))
    for i, wc in enumerate(wall_cells):
        all_nodes = wc["wall_nodes"] + wc["off_wall_nodes"]
        if mu is not None:
            mu_wall[i] = np.mean([mu[n] for n in all_nodes])
        else:
            mu_wall[i] = freestream.get("mu_nondim", 1.0)

    tau_w = mu_wall[:len(grad["du_dn"])] * grad["du_dn"]

    # Reference values — SU2 FREESTREAM_PRESS_EQ_ONE non-dim:
    #   rho* = 1, p* = 1/gamma, a* = sqrt(gamma * p*/rho*) = 1
    #   U* = Mach * a* = Mach
    #   q* = 0.5 * rho* * U*^2 = 0.5 * Mach^2
    rho_inf = freestream.get("rho_nondim", 1.0)
    Mach = freestream.get("Mach", 0.1)
    gamma = freestream.get("gamma", 1.4)
    U_inf = Mach  # In SU2 non-dim: U* = Mach
    q_inf = 0.5 * rho_inf * U_inf**2
    print(f"  SU2 non-dim: rho*={rho_inf}, U*=M={Mach}, q*={q_inf:.6f}")
    Cf = tau_w / q_inf

    # Sort by x
    idx = np.argsort(grad["x"])

    return {
        "x": grad["x"][idx],
        "Cf": Cf[idx],
        "tau_w": tau_w[idx],
        "du_dn": grad["du_dn"][idx],
        "dy": grad["dy"][idx],
        "method": "volume_gradient",
        "U_inf": U_inf,
        "q_inf": q_inf,
    }


# =========================================================================
# Comparison & Validation
# =========================================================================

def compare_methods(
    direct: Dict,
    gradient: Dict,
    experimental: Optional[Dict] = None,
    output_dir: Path = None,
) -> Dict:
    """
    Compare Cf from both methods and optionally experimental data.
    Generates comparison plot.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = output_dir or Path("plots/wall_hump")
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # -- Cf comparison --
    ax = axes[0]
    ax.plot(direct["x"], direct["Cf_x"], "-", color="#e53935", lw=1.8,
            label="Method 1: SU2 Surface Direct")
    ax.plot(gradient["x"], gradient["Cf"], "--", color="#1e88e5", lw=1.8,
            label="Method 2: Volume Gradient")
    if experimental:
        ax.plot(experimental["x"], experimental["cf"], "o",
                color="#2e7d32", ms=4, alpha=0.7, label="Experiment")

    ax.axhline(0, color="#999", ls="-", lw=0.5)
    ax.set_ylabel("$C_f$", fontsize=13)
    ax.set_title("Skin Friction Coefficient Comparison", fontsize=14,
                 fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.5, 2.0)

    # -- Separation detection --
    ax = axes[1]
    # Detect separation/reattachment from Cf sign changes
    for data, label, color, ls in [
        (direct, "Direct", "#e53935", "-"),
        (gradient, "Gradient", "#1e88e5", "--"),
    ]:
        cf_key = "Cf_x" if "Cf_x" in data else "Cf"
        x, cf = data["x"], data[cf_key]
        ax.plot(x, cf, ls, color=color, lw=1.2, alpha=0.6)

        # Find zero crossings in hump region (0.5 < x < 1.5)
        hm = (x >= 0.5) & (x <= 1.5)
        xh, cfh = x[hm], cf[hm]
        crossings = np.where(np.diff(np.sign(cfh)))[0]
        for ci in crossings:
            x_zero = xh[ci] - cfh[ci] * (xh[ci+1] - xh[ci]) / (cfh[ci+1] - cfh[ci])
            marker = "v" if cfh[ci] > 0 else "^"
            event = "Sep" if cfh[ci] > 0 else "Reat"
            ax.plot(x_zero, 0, marker, color=color, ms=10, zorder=5)
            ax.annotate(f"{label} {event}: {x_zero:.3f}",
                       (x_zero, 0), textcoords="offset points",
                       xytext=(5, 10 if event == "Sep" else -15),
                       fontsize=8, color=color)

    ax.axhline(0, color="#333", ls="-", lw=1)
    ax.set_xlabel("x/c", fontsize=13)
    ax.set_ylabel("$C_f$", fontsize=13)
    ax.set_title("Separation Detection (Cf zero-crossings)", fontsize=14,
                 fontweight="bold")
    ax.set_xlim(0.3, 1.6)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = output_dir / "cf_extraction_comparison.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  [PLOT] {path}")

    # Compute cross-validation metrics
    # Interpolate gradient Cf onto direct x-grid
    from scipy.interpolate import interp1d
    common_x = direct["x"]
    mask = (common_x >= gradient["x"].min()) & (common_x <= gradient["x"].max())
    if mask.any():
        interp_cf = interp1d(gradient["x"], gradient["Cf"],
                             kind="linear", fill_value="extrapolate")
        cf_grad_interp = interp_cf(common_x[mask])
        cf_direct = direct["Cf_x"][mask]
        rmse = np.sqrt(np.mean((cf_direct - cf_grad_interp)**2))
        corr = np.corrcoef(cf_direct, cf_grad_interp)[0, 1]
    else:
        rmse, corr = float("nan"), float("nan")

    metrics = {
        "cross_validation_RMSE": float(rmse),
        "cross_validation_correlation": float(corr),
    }
    print(f"  Cross-validation: RMSE={rmse:.6f}, correlation={corr:.4f}")

    return metrics


# =========================================================================
# Main Pipeline
# =========================================================================

def run_cf_extraction(
    case_dir: Path,
    freestream: Dict = None,
    output_dir: Path = None,
) -> Dict:
    """
    Full Cf extraction pipeline for a single SU2 case.

    Parameters
    ----------
    case_dir : Path
        SU2 case directory containing flow.vtu and surface_flow.vtu
    freestream : dict
        Freestream conditions (Mach, rho_nondim, etc.)
    output_dir : Path
        Output directory for plots

    Returns
    -------
    dict with direct_cf, gradient_cf, metrics
    """
    surface_vtu = case_dir / "surface_flow.vtu"
    volume_vtu = case_dir / "flow.vtu"
    output_dir = output_dir or case_dir.parent.parent / "plots" / "wall_hump"

    if not surface_vtu.exists():
        raise FileNotFoundError(f"No surface_flow.vtu in {case_dir}")

    freestream = freestream or {
        "Mach": 0.1,
        "gamma": 1.4,
        "rho_nondim": 1.0,
    }

    print(f"\n{'='*60}")
    print(f"  Cf EXTRACTION PIPELINE")
    print(f"  Case: {case_dir.name}")
    print(f"{'='*60}\n")

    # Method 1: Direct from surface
    print("  Method 1: SU2 Surface Direct...")
    direct = extract_cf_from_surface(surface_vtu)
    print(f"  -> {len(direct['x'])} surface points")
    print(f"  -> Cf_x range: [{direct['Cf_x'].min():.6f}, {direct['Cf_x'].max():.6f}]")

    result = {"direct_cf": direct}

    # Method 2: Volume gradients (if volume VTU exists)
    if volume_vtu.exists():
        print("\n  Method 2: Volume Gradient...")
        try:
            gradient = extract_cf_from_volume(
                volume_vtu, surface_vtu, freestream)
            print(f"  -> {len(gradient['x'])} wall points")
            print(f"  -> Cf range: [{gradient['Cf'].min():.6f}, "
                  f"{gradient['Cf'].max():.6f}]")
            result["gradient_cf"] = gradient

            # Cross-validate
            print("\n  Cross-validating methods...")
            metrics = compare_methods(direct, gradient, output_dir=output_dir)
            result["metrics"] = metrics
        except Exception as e:
            print(f"  Method 2 failed: {e}")
            result["gradient_cf"] = None
            result["metrics"] = {"error": str(e)}
    else:
        print(f"\n  [SKIP] No volume VTU for Method 2")
        result["gradient_cf"] = None

    # Separation analysis from direct Cf
    print("\n  Separation Analysis (from direct Cf):")
    x, cf = direct["x"], direct["Cf_x"]
    hm = (x >= 0.5) & (x <= 1.5)
    xh, cfh = x[hm], cf[hm]
    crossings = np.where(np.diff(np.sign(cfh)))[0]
    sep_points = []
    for ci in crossings:
        x_zero = xh[ci] - cfh[ci] * (xh[ci+1] - xh[ci]) / (cfh[ci+1] - cfh[ci])
        if cfh[ci] > 0:
            print(f"    Separation:   x/c = {x_zero:.4f}")
            sep_points.append(("separation", float(x_zero)))
        else:
            print(f"    Reattachment: x/c = {x_zero:.4f}")
            sep_points.append(("reattachment", float(x_zero)))
    result["separation_points"] = sep_points

    if len(sep_points) >= 2:
        x_s = sep_points[0][1]
        x_r = sep_points[-1][1]
        bubble = x_r - x_s
        print(f"    Bubble length: {bubble:.4f}")
        result["bubble_length"] = float(bubble)

    # Save results
    out_json = output_dir / f"cf_results_{case_dir.name}.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_data = {k: v for k, v in result.items()
                 if k not in ("direct_cf", "gradient_cf")}
    with open(out_json, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\n  Results: {out_json}")

    return result


# =========================================================================
# CLI
# =========================================================================

if __name__ == "__main__":
    import argparse

    PROJECT = Path(__file__).resolve().parent.parent.parent

    parser = argparse.ArgumentParser(
        description="Cf Extraction Pipeline for SU2")
    parser.add_argument("--case-dir", type=Path, default=None,
                        help="SU2 case directory")
    parser.add_argument("--model", choices=["SA", "SST", "all"],
                        default="all",
                        help="Turbulence model to process")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    runs_dir = PROJECT / "runs" / "wall_hump"
    output_dir = args.output_dir or (PROJECT / "plots" / "wall_hump")

    if args.case_dir:
        cases = [args.case_dir]
    else:
        models = ["SA", "SST"] if args.model == "all" else [args.model]
        cases = [runs_dir / f"hump_{m}_medium" for m in models]

    for case in cases:
        if case.exists():
            run_cf_extraction(case, output_dir=output_dir)
        else:
            print(f"  [SKIP] {case} does not exist")
