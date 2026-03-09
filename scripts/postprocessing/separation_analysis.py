#!/usr/bin/env python3
"""
Separation Analysis Module
===========================
Formalised granular error metrics for separated-flow validation cases.

Computes:
  - Separation/reattachment location errors
  - Bubble length error (absolute and percentage)
  - Region-wise Cp and Cf RMSE (fore-body, separation, recovery)
  - Boundary-layer shape factor H = δ*/θ at selected stations
  - Reverse-flow intensity (min Cf in separation region)
  - Peak Cp recovery error

Station locations and measurement windows follow:
  - Greenblatt et al. (2006), CFDVAL2004 Case 3 (wall hump)
  - NACA 0012 TMR validation conventions

References:
  - Greenblatt, Paschal, Yao, Harris (2006), AIAA J. 44(12)
  - TMBWG: tmbwg.github.io/turbmodels
  - Rumsey et al. (2023), AIAA AVIATION 2023 Forum
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT))

# ---------------------------------------------------------------------------
# Experimental reference values
# ---------------------------------------------------------------------------

# Wall hump (Greenblatt et al., 2006, CFDVAL2004 Case 3, no-flow-control)
HUMP_EXP = {
    "x_sep": 0.665,       # Separation location (x/c)
    "x_reat": 1.11,       # Reattachment location (x/c)
    "bubble_length": 0.445,  # L_bubble/c
    "source": "Greenblatt et al. (2006), AIAA J. 44(12)",
}

# Region definitions for wall hump (matching Greenblatt and COMSOL studies)
HUMP_REGIONS = {
    "fore_body":  (-0.50, 0.65),   # Attached flow upstream of separation
    "separation": (0.65, 1.10),    # Separation bubble region
    "recovery":   (1.10, 1.50),    # Downstream recovery
    "full":       (-0.50, 1.50),   # Full domain
}

# Stations for BL profile analysis (x/c values)
HUMP_PROFILE_STATIONS = [0.65, 0.80, 1.00, 1.20]

# NACA 0012 TMR experimental reference
NACA_EXP = {
    "source": "Ladson (1988), Gregory & O'Reilly (1970)",
    "conditions": {"Mach": 0.15, "Re": 6e6},
}


# ===================================================================
# Separation/Reattachment Detection
# ===================================================================
def find_zero_crossings(
    x: np.ndarray, y: np.ndarray
) -> Tuple[Optional[float], Optional[float]]:
    """
    Find separation (positive→negative) and reattachment (negative→positive)
    zero-crossings of Cf, using linear interpolation.

    Only considers crossings in the hump region (x > 0.5).

    Returns
    -------
    (x_sep, x_reat) or (None, None)
    """
    x_sep = None
    x_reat = None
    for i in range(len(y) - 1):
        if y[i] > 0 and y[i + 1] < 0 and x[i] > 0.5:
            # Separation: positive → negative
            x_sep = float(x[i] - y[i] * (x[i + 1] - x[i]) / (y[i + 1] - y[i]))
        if y[i] < 0 and y[i + 1] > 0 and x[i] > 0.8:
            # Reattachment: negative → positive
            x_reat = float(x[i] - y[i] * (x[i + 1] - x[i]) / (y[i + 1] - y[i]))
    return x_sep, x_reat


# ===================================================================
# Region-Wise RMSE
# ===================================================================
def compute_regionwise_rmse(
    x_cfd: np.ndarray,
    y_cfd: np.ndarray,
    x_exp: np.ndarray,
    y_exp: np.ndarray,
    regions: Dict[str, Tuple[float, float]],
) -> Dict[str, Dict[str, float]]:
    """
    Compute RMSE, MAE, and max error within each named region.

    Interpolates CFD values onto experimental measurement locations
    for a fair comparison.

    Parameters
    ----------
    x_cfd, y_cfd : arrays
        CFD surface distribution (sorted by x).
    x_exp, y_exp : arrays
        Experimental data points.
    regions : dict
        {region_name: (x_min, x_max)}.

    Returns
    -------
    dict of region name → {RMSE, MAE, max_error, n_points}
    """
    results = {}
    for name, (xmin, xmax) in regions.items():
        mask = (x_exp >= xmin) & (x_exp <= xmax)
        if mask.sum() == 0:
            results[name] = {"RMSE": None, "MAE": None, "max_error": None, "n_points": 0}
            continue

        x_e = x_exp[mask]
        y_e = y_exp[mask]
        y_c = np.interp(x_e, x_cfd, y_cfd)

        err = y_c - y_e
        results[name] = {
            "RMSE": float(np.sqrt(np.mean(err**2))),
            "MAE": float(np.mean(np.abs(err))),
            "max_error": float(np.max(np.abs(err))),
            "n_points": int(mask.sum()),
        }
    return results


# ===================================================================
# Separation Bubble Metrics
# ===================================================================
def compute_separation_metrics(
    x: np.ndarray,
    Cf: np.ndarray,
    x_sep_exp: float = HUMP_EXP["x_sep"],
    x_reat_exp: float = HUMP_EXP["x_reat"],
) -> Dict[str, Optional[float]]:
    """
    Comprehensive separation bubble metrics.

    Returns
    -------
    dict with:
        x_sep_cfd, x_reat_cfd,
        sep_error, reat_error,
        bubble_length_cfd, bubble_length_exp,
        bubble_error_abs, bubble_error_pct,
        cf_min, cf_min_x (reverse-flow intensity)
    """
    x_sep, x_reat = find_zero_crossings(x, Cf)

    bubble_exp = x_reat_exp - x_sep_exp
    bubble_cfd = (x_reat - x_sep) if (x_sep is not None and x_reat is not None) else None

    # Reverse-flow intensity: minimum Cf in separation region
    sep_mask = (x >= 0.5) & (x <= 1.5)
    if sep_mask.any():
        cf_vals = Cf[sep_mask]
        cf_min_val = float(cf_vals.min())
        cf_min_x = float(x[sep_mask][np.argmin(cf_vals)])
    else:
        cf_min_val = None
        cf_min_x = None

    result = {
        "x_sep_cfd": x_sep,
        "x_reat_cfd": x_reat,
        "x_sep_exp": x_sep_exp,
        "x_reat_exp": x_reat_exp,
        "sep_error": abs(x_sep - x_sep_exp) if x_sep is not None else None,
        "reat_error": abs(x_reat - x_reat_exp) if x_reat is not None else None,
        "bubble_length_cfd": bubble_cfd,
        "bubble_length_exp": bubble_exp,
        "bubble_error_abs": abs(bubble_cfd - bubble_exp) if bubble_cfd is not None else None,
        "bubble_error_pct": (
            abs(bubble_cfd - bubble_exp) / bubble_exp * 100
            if bubble_cfd is not None and bubble_exp > 0
            else None
        ),
        "cf_min": cf_min_val,
        "cf_min_x": cf_min_x,
    }
    return result


# ===================================================================
# Boundary-Layer Shape Factor
# ===================================================================
def compute_shape_factor(
    y: np.ndarray, u: np.ndarray, U_edge: float
) -> Dict[str, float]:
    """
    Compute boundary-layer shape factor H = δ*/θ.

    Parameters
    ----------
    y : array
        Wall-normal coordinates (starting from wall, y >= 0).
    u : array
        Streamwise velocity at each y.
    U_edge : float
        Edge velocity.

    Returns
    -------
    dict with delta_star, theta, H
    """
    if U_edge < 1e-10 or len(y) < 3:
        return {"delta_star": None, "theta": None, "H": None}

    u_norm = u / U_edge

    # Displacement thickness: δ* = ∫(1 - u/U_e) dy
    delta_star = float(np.trapz(1.0 - u_norm, y))

    # Momentum thickness: θ = ∫(u/U_e)(1 - u/U_e) dy
    theta = float(np.trapz(u_norm * (1.0 - u_norm), y))

    H = delta_star / theta if abs(theta) > 1e-15 else None

    return {
        "delta_star": delta_star,
        "theta": theta,
        "H": H,
    }


def compute_shape_factors_at_stations(
    coords: np.ndarray,
    velocity: np.ndarray,
    stations: List[float] = None,
    nx: int = 409,
    ny: int = 109,
) -> Dict[float, Dict[str, float]]:
    """
    Compute BL shape factor H at multiple streamwise stations.

    Parameters
    ----------
    coords : array, shape (N, 2+)
        Point coordinates (x, y, ...).
    velocity : array, shape (N, 2+)
        Velocity components (u, v, ...).
    stations : list of float
        x/c stations. Default: [0.65, 0.80, 1.00, 1.20].
    nx, ny : int
        Structured mesh dimensions.

    Returns
    -------
    dict of x_station → {delta_star, theta, H}
    """
    stations = stations or HUMP_PROFILE_STATIONS

    x_2d = coords[:, 0].reshape(ny, nx)
    y_2d = coords[:, 1].reshape(ny, nx)
    u_2d = velocity[:, 0].reshape(ny, nx)

    results = {}
    for x_st in stations:
        # Find nearest column
        i_col = np.argmin(np.abs(x_2d[0, :] - x_st))

        y_prof = y_2d[:, i_col]
        u_prof = u_2d[:, i_col]

        # Sort by y (wall-normal)
        idx = np.argsort(y_prof)
        y_prof = y_prof[idx]
        u_prof = u_prof[idx]

        # Edge velocity: max of u profile
        U_edge = float(np.max(np.abs(u_prof)))

        sf = compute_shape_factor(y_prof, u_prof, U_edge)
        sf["x_station"] = float(x_2d[0, i_col])
        sf["U_edge"] = U_edge
        results[x_st] = sf

    return results


# ===================================================================
# Full Wall Hump Analysis Pipeline
# ===================================================================
def load_exp_data(exp_dir: Path) -> Dict:
    """Load experimental Cp and Cf data from project files."""
    data = {}

    # Load Cf
    cf_file = exp_dir / "noflow_cf.exp.dat"
    if cf_file.exists():
        x, cf = [], []
        for line in cf_file.read_text().splitlines():
            s = line.strip()
            if not s or s.startswith("#") or s.startswith("%"):
                continue
            parts = s.split()
            if len(parts) >= 2:
                try:
                    x.append(float(parts[0]))
                    cf.append(float(parts[1]))
                except ValueError:
                    pass
        data["cf_x"] = np.array(x)
        data["cf"] = np.array(cf)

    # Load Cp
    cp_file = exp_dir / "noflow_cp.exp.dat"
    if cp_file.exists():
        x, cp = [], []
        for line in cp_file.read_text().splitlines():
            s = line.strip()
            if not s or s.startswith("#") or s.startswith("%"):
                continue
            parts = s.split()
            if len(parts) >= 2:
                try:
                    x.append(float(parts[0]))
                    cp.append(float(parts[1]))
                except ValueError:
                    pass
        data["cp_x"] = np.array(x)
        data["cp"] = np.array(cp)

    return data


def extract_surface_from_vtu(vtu_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract x, Cf_x, Cp from a surface_flow.vtu file."""
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy

    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(str(vtu_path))
    reader.Update()
    out = reader.GetOutput()
    pd = out.GetPointData()
    names = [pd.GetArrayName(i) for i in range(pd.GetNumberOfArrays())]
    coords = vtk_to_numpy(out.GetPoints().GetData())

    x = coords[:, 0]
    idx = np.argsort(x)
    x_s = x[idx]

    # Cf (streamwise component)
    cf = vtk_to_numpy(pd.GetArray(names.index("Skin_Friction_Coefficient")))
    cf_x = cf[:, 0] if cf.ndim > 1 else cf
    cf_s = cf_x[idx]

    # Cp (corrected by upstream reference)
    cp = vtk_to_numpy(pd.GetArray(names.index("Pressure_Coefficient")))
    cp_s = cp[idx]
    cp_ref = cp_s[x_s < -0.3].mean() if (x_s < -0.3).any() else 0.0
    cp_corr = cp_s - cp_ref

    return x_s, cf_s, cp_corr


def run_wall_hump_analysis(
    surface_vtu: str,
    exp_dir: Optional[str] = None,
    volume_vtu: Optional[str] = None,
    label: str = "SA_fine",
) -> Dict:
    """
    Full wall hump separation analysis pipeline.

    Parameters
    ----------
    surface_vtu : str
        Path to surface_flow.vtu.
    exp_dir : str, optional
        Path to experimental data directory.
    volume_vtu : str, optional
        Path to volume flow.vtu (for shape factor computation).
    label : str
        Case label for output.

    Returns
    -------
    dict with all computed metrics.
    """
    exp_dir = Path(exp_dir) if exp_dir else PROJECT / "experimental_data" / "wall_hump" / "csv"

    print("=" * 65)
    print(f"  SEPARATION ANALYSIS: Wall Hump — {label}")
    print("=" * 65)

    # 1. Load surface data
    print("\n[1] Loading surface data...")
    x, cf, cp = extract_surface_from_vtu(surface_vtu)
    print(f"    {len(x)} surface points, x ∈ [{x.min():.3f}, {x.max():.3f}]")

    # 2. Separation metrics
    print("\n[2] Separation/Reattachment...")
    sep = compute_separation_metrics(x, cf)
    print(f"    x_sep:  CFD={sep['x_sep_cfd']:.4f}  exp={sep['x_sep_exp']:.3f}  "
          f"err={sep['sep_error']:.4f}" if sep["sep_error"] else "    x_sep: not found")
    print(f"    x_reat: CFD={sep['x_reat_cfd']:.4f}  exp={sep['x_reat_exp']:.3f}  "
          f"err={sep['reat_error']:.4f}" if sep["reat_error"] else "    x_reat: not found")
    if sep["bubble_error_pct"] is not None:
        print(f"    L_bubble: CFD={sep['bubble_length_cfd']:.4f}c  "
              f"exp={sep['bubble_length_exp']:.3f}c  "
              f"err={sep['bubble_error_pct']:.1f}%")
    print(f"    Cf_min: {sep['cf_min']:.6f} at x/c={sep['cf_min_x']:.4f}")

    # 3. Load experimental data and compute region-wise RMSE
    exp = load_exp_data(exp_dir)
    results = {"label": label, "separation": sep}

    if "cp_x" in exp:
        print("\n[3] Region-wise Cp RMSE...")
        cp_rmse = compute_regionwise_rmse(x, cp, exp["cp_x"], exp["cp"], HUMP_REGIONS)
        results["cp_rmse_by_region"] = cp_rmse
        for region, vals in cp_rmse.items():
            if vals["RMSE"] is not None:
                print(f"    {region:12s}: RMSE={vals['RMSE']:.5f}  "
                      f"MAE={vals['MAE']:.5f}  n={vals['n_points']}")
    else:
        print("\n[3] Cp experimental data not found — skipping region-wise RMSE")

    if "cf_x" in exp:
        print("\n[4] Region-wise Cf RMSE...")
        cf_rmse = compute_regionwise_rmse(x, cf, exp["cf_x"], exp["cf"], HUMP_REGIONS)
        results["cf_rmse_by_region"] = cf_rmse
        for region, vals in cf_rmse.items():
            if vals["RMSE"] is not None:
                print(f"    {region:12s}: RMSE={vals['RMSE']:.6f}  "
                      f"MAE={vals['MAE']:.6f}  n={vals['n_points']}")
    else:
        print("\n[4] Cf experimental data not found — skipping region-wise RMSE")

    # 4. Shape factor (if volume data available)
    if volume_vtu and Path(volume_vtu).exists():
        print("\n[5] Boundary-layer shape factor H...")
        try:
            import vtk
            from vtk.util.numpy_support import vtk_to_numpy

            reader = vtk.vtkXMLUnstructuredGridReader()
            reader.SetFileName(str(volume_vtu))
            reader.Update()
            out = reader.GetOutput()
            pd_out = out.GetPointData()

            coords = vtk_to_numpy(out.GetPoints().GetData())
            vel_arr = vtk_to_numpy(pd_out.GetArray("Velocity"))

            n_pts = coords.shape[0]
            # Auto-detect or use default mesh dimensions
            nx, ny = 409, 109
            if n_pts != nx * ny:
                # Try other common dimensions
                for _nx, _ny in [(205, 55), (103, 28), (817, 217)]:
                    if n_pts == _nx * _ny:
                        nx, ny = _nx, _ny
                        break

            if n_pts == nx * ny:
                sf = compute_shape_factors_at_stations(coords, vel_arr, nx=nx, ny=ny)
                results["shape_factors"] = sf
                for x_st, vals in sf.items():
                    if vals["H"] is not None:
                        print(f"    x/c={x_st:.2f}: H={vals['H']:.3f}  "
                              f"δ*={vals['delta_star']:.6f}  θ={vals['theta']:.6f}")
                    else:
                        print(f"    x/c={x_st:.2f}: H=N/A")
            else:
                print(f"    Cannot determine mesh dims for {n_pts} points — skipping")
        except ImportError:
            print("    VTK not available — skipping shape factor")
        except Exception as e:
            print(f"    Error: {e}")
    else:
        print("\n[5] Volume VTU not provided — skipping shape factor")

    # 5. Summary table
    print("\n" + format_results_table(results))

    return results


# ===================================================================
# NACA 0012 Analysis
# ===================================================================
def run_naca0012_analysis(
    cfd_results: Dict[str, Dict[str, float]],
    exp_dir: Optional[str] = None,
    label: str = "SU2_SA",
) -> Dict:
    """
    NACA 0012 validation analysis.

    Parameters
    ----------
    cfd_results : dict
        {alpha: {CL: ..., CD: ...}} for each angle of attack.
    exp_dir : str, optional
        Path to NACA 0012 experimental data.
    label : str
        Case label.

    Returns
    -------
    dict with metrics per alpha.
    """
    exp_dir = Path(exp_dir) if exp_dir else PROJECT / "experimental_data" / "naca0012" / "csv"

    print("=" * 65)
    print(f"  NACA 0012 VALIDATION ANALYSIS — {label}")
    print("=" * 65)

    # Load CFL3D reference
    cfl3d_file = exp_dir / "cfl3d_sa_forces.csv"
    cfl3d = {}
    if cfl3d_file.exists():
        for line in cfl3d_file.read_text().splitlines()[1:]:
            parts = line.strip().split(",")
            if len(parts) >= 3:
                alpha = float(parts[0])
                cfl3d[alpha] = {"CL": float(parts[1]), "CD": float(parts[2])}

    results = {"label": label, "alphas": {}}

    for alpha, vals in sorted(cfd_results.items()):
        alpha_f = float(alpha)
        entry = {"CL": vals.get("CL"), "CD": vals.get("CD")}

        if alpha_f in cfl3d:
            ref = cfl3d[alpha_f]
            if entry["CL"] is not None and ref["CL"] != 0:
                entry["CL_error_pct"] = abs(entry["CL"] - ref["CL"]) / abs(ref["CL"]) * 100
            if entry["CD"] is not None and ref["CD"] != 0:
                entry["CD_error_pct"] = abs(entry["CD"] - ref["CD"]) / ref["CD"] * 100
            entry["CL_ref"] = ref["CL"]
            entry["CD_ref"] = ref["CD"]

        results["alphas"][alpha_f] = entry

    # Print summary
    print(f"\n  {'Alpha':>6s}  {'CL_CFD':>10s}  {'CL_ref':>10s}  {'CL_err%':>8s}  "
          f"{'CD_CFD':>10s}  {'CD_ref':>10s}  {'CD_err%':>8s}")
    print("  " + "-" * 70)
    for alpha, entry in sorted(results["alphas"].items()):
        cl_s = f"{entry['CL']:.6f}" if entry.get("CL") is not None else "N/A"
        cd_s = f"{entry['CD']:.8f}" if entry.get("CD") is not None else "N/A"
        cl_r = f"{entry.get('CL_ref', 0):.6f}" if "CL_ref" in entry else "---"
        cd_r = f"{entry.get('CD_ref', 0):.8f}" if "CD_ref" in entry else "---"
        cl_e = f"{entry.get('CL_error_pct', 0):.2f}" if "CL_error_pct" in entry else "---"
        cd_e = f"{entry.get('CD_error_pct', 0):.2f}" if "CD_error_pct" in entry else "---"
        print(f"  {alpha:6.1f}  {cl_s:>10s}  {cl_r:>10s}  {cl_e:>8s}  "
              f"{cd_s:>10s}  {cd_r:>10s}  {cd_e:>8s}")

    return results


# ===================================================================
# Formatted Output
# ===================================================================
def format_results_table(results: Dict) -> str:
    """Format analysis results as a publication-quality text table."""
    lines = []
    lines.append("=" * 65)
    lines.append(f"  SEPARATION ANALYSIS SUMMARY — {results.get('label', '')}")
    lines.append("=" * 65)

    sep = results.get("separation", {})
    if sep:
        lines.append("")
        lines.append("  Separation Bubble:")
        lines.append(f"    {'Quantity':<25s} {'CFD':>10s} {'Exp':>10s} {'Error':>10s}")
        lines.append("    " + "-" * 55)

        if sep.get("x_sep_cfd") is not None:
            lines.append(f"    {'x_sep (x/c)':<25s} {sep['x_sep_cfd']:10.4f} "
                         f"{sep['x_sep_exp']:10.3f} {sep['sep_error']:10.4f}")
        if sep.get("x_reat_cfd") is not None:
            lines.append(f"    {'x_reat (x/c)':<25s} {sep['x_reat_cfd']:10.4f} "
                         f"{sep['x_reat_exp']:10.3f} {sep['reat_error']:10.4f}")
        if sep.get("bubble_length_cfd") is not None:
            lines.append(f"    {'L_bubble (x/c)':<25s} {sep['bubble_length_cfd']:10.4f} "
                         f"{sep['bubble_length_exp']:10.3f} "
                         f"{sep['bubble_error_pct']:9.1f}%")
        if sep.get("cf_min") is not None:
            lines.append(f"    {'Cf_min':<25s} {sep['cf_min']:10.6f} "
                         f"{'---':>10s} at x/c={sep['cf_min_x']:.4f}")

    # Region-wise RMSE
    for qty_key, qty_name in [("cp_rmse_by_region", "Cp"), ("cf_rmse_by_region", "Cf")]:
        reg = results.get(qty_key, {})
        if reg:
            lines.append(f"\n  Region-wise {qty_name} RMSE:")
            lines.append(f"    {'Region':<15s} {'RMSE':>10s} {'MAE':>10s} {'Max Err':>10s} {'N':>4s}")
            lines.append("    " + "-" * 49)
            for rname, vals in reg.items():
                if vals["RMSE"] is not None:
                    lines.append(f"    {rname:<15s} {vals['RMSE']:10.5f} "
                                 f"{vals['MAE']:10.5f} {vals['max_error']:10.5f} "
                                 f"{vals['n_points']:4d}")

    # Shape factors
    sf = results.get("shape_factors", {})
    if sf:
        lines.append("\n  Boundary-Layer Shape Factor H = δ*/θ:")
        lines.append(f"    {'x/c':<8s} {'H':>8s} {'δ*':>12s} {'θ':>12s} {'U_edge':>10s}")
        lines.append("    " + "-" * 50)
        for x_st, vals in sorted(sf.items()):
            if vals["H"] is not None:
                lines.append(f"    {x_st:<8.2f} {vals['H']:8.3f} "
                             f"{vals['delta_star']:12.6f} {vals['theta']:12.6f} "
                             f"{vals['U_edge']:10.3f}")

    lines.append("")
    lines.append("=" * 65)
    return "\n".join(lines)


# ===================================================================
# CLI Entry Point
# ===================================================================
def main():
    """Run separation analysis on the wall hump fine-grid case."""
    runs_dir = PROJECT / "runs" / "wall_hump"

    # Default: SA fine grid
    surface_vtu = runs_dir / "hump_SA_fine" / "surface_flow.vtu"
    volume_vtu = runs_dir / "hump_SA_fine" / "flow.vtu"

    if not surface_vtu.exists():
        print(f"Surface VTU not found: {surface_vtu}")
        print("Run the wall hump simulation first.")

        # Try SA medium or coarse as fallback
        for alt in ["hump_SA_medium", "hump_SA_coarse"]:
            alt_vtu = runs_dir / alt / "surface_flow.vtu"
            if alt_vtu.exists():
                surface_vtu = alt_vtu
                volume_vtu = runs_dir / alt / "flow.vtu"
                print(f"Using fallback: {alt}")
                break
        else:
            return

    results = run_wall_hump_analysis(
        str(surface_vtu),
        volume_vtu=str(volume_vtu) if volume_vtu.exists() else None,
        label="SA_fine",
    )

    # Save JSON results
    output_dir = PROJECT / "results" / "separation_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    def _serialise(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    out_file = output_dir / "hump_separation_metrics.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2, default=_serialise)
    print(f"\nResults saved to: {out_file}")


if __name__ == "__main__":
    main()
