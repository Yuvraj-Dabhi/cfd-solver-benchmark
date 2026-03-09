#!/usr/bin/env python3
"""
Compare SWBLI skin friction (Cf) across turbulence models.
Loads SU2 surface_flow.csv from SA, SST, and SA-comp runs,
computes Cf, and compares against Schülein experimental data.

Usage:
    python scripts/analysis/compare_swbli_models.py
"""
import json
import sys
import os
from pathlib import Path

import numpy as np

# Project root
PROJECT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT))

RUNS_DIR = PROJECT / "runs" / "swbli"

# ─── Schülein Experimental Data ─────────────────────────────────────────
EXPERIMENTAL_CF = {
    "source": "Schülein et al. (1996), DLR IB 223-96 A 49",
    "x": [0.060, 0.080, 0.100, 0.120, 0.140, 0.160, 0.180, 0.200,
          0.220, 0.240, 0.260, 0.270, 0.280, 0.290, 0.295, 0.300,
          0.305, 0.310, 0.315, 0.320, 0.325, 0.330, 0.335, 0.340,
          0.345, 0.350, 0.355, 0.360, 0.365, 0.370, 0.380, 0.390,
          0.400, 0.420, 0.440, 0.460, 0.480, 0.500],
    "cf": [0.00140, 0.00128, 0.00120, 0.00114, 0.00107, 0.00101,
           0.00096, 0.00092, 0.00090, 0.00087, 0.00084, 0.00080,
           0.00065, 0.00035, 0.00010, -0.00020, -0.00040, -0.00050,
           -0.00055, -0.00050, -0.00040, -0.00020, 0.00000, 0.00030,
           0.00060, 0.00090, 0.00110, 0.00120, 0.00125, 0.00128,
           0.00135, 0.00140, 0.00145, 0.00155, 0.00165, 0.00170,
           0.00175, 0.00178],
}


def load_surface_cf(case_dir: Path):
    """Load x and Cf from SU2 surface VTU file."""
    vtu_file = case_dir / "surface.vtu"
    csv_file = case_dir / "surface_flow.csv"

    # Try VTU first (has Cf), fall back to CSV
    if vtu_file.exists():
        return _load_cf_from_vtu(vtu_file)
    elif csv_file.exists():
        return _load_cf_from_csv(csv_file)
    return None, None


def _load_cf_from_vtu(vtu_file: Path):
    """Parse SU2 VTU for x coordinates and Skin_Friction_Coefficient_0."""
    import struct
    import xml.etree.ElementTree as ET

    # Read the raw bytes
    raw = vtu_file.read_bytes()

    # Find the XML portion (before appended data)
    # Look for the <AppendedData> tag
    text = raw.decode('ascii', errors='replace')
    appended_idx = text.find('<AppendedData')
    if appended_idx < 0:
        return None, None

    # Parse XML header
    xml_end = text.find('_', appended_idx)  # underscore starts binary data
    xml_text = text[:appended_idx] + '</VTKFile>'
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        # Try simpler parsing
        return _load_cf_from_vtu_simple(vtu_file)

    # Find data arrays and their offsets
    arrays = {}
    piece = root.find('.//Piece')
    if piece is None:
        return None, None

    n_points = int(piece.get('NumberOfPoints', 0))
    if n_points == 0:
        return None, None

    for da in root.iter('DataArray'):
        name = da.get('Name', '')
        offset = int(da.get('offset', -1))
        ncomp = int(da.get('NumberOfComponents', 1))
        dtype = da.get('type', 'Float32')
        if offset >= 0:
            arrays[name] = {'offset': offset, 'ncomp': ncomp, 'dtype': dtype}

    # Binary data starts after underscore in AppendedData
    bin_start = xml_end + 1  # position after '_'

    def read_array(info):
        offset = info['offset']
        ncomp = info['ncomp']
        pos = bin_start + offset
        # SU2 writes a 4-byte int header (number of bytes) before data
        nbytes = struct.unpack_from('<I', raw, pos)[0]
        pos += 4
        n_floats = nbytes // 4
        data = struct.unpack_from(f'<{n_floats}f', raw, pos)
        arr = np.array(data)
        if ncomp > 1:
            arr = arr.reshape(-1, ncomp)
        return arr

    # Read coordinates (empty name = Points)
    coords_info = arrays.get('', None)
    if coords_info is None:
        return None, None
    coords = read_array(coords_info)  # (N, 3)

    # Read Cf
    cf_info = arrays.get('Skin_Friction_Coefficient', None)
    if cf_info is None:
        return None, None
    cf_all = read_array(cf_info)  # (N, 3) — Cf_x, Cf_y, Cf_z

    x = coords[:, 1]  # y is streamwise in SWBLI mesh
    cf_x = cf_all[:, 1] if cf_all.ndim > 1 else cf_all  # Cf_y = streamwise Cf

    # Sort by x
    sort_idx = np.argsort(x)
    return x[sort_idx], cf_x[sort_idx]


def _load_cf_from_vtu_simple(vtu_file: Path):
    """Fallback: use meshio if available, else skip."""
    try:
        import meshio
        mesh = meshio.read(str(vtu_file))
        x = mesh.points[:, 0]
        cf = mesh.point_data.get('Skin_Friction_Coefficient')
        if cf is not None:
            cf_x = cf[:, 0] if cf.ndim > 1 else cf
            sort_idx = np.argsort(x)
            return x[sort_idx], cf_x[sort_idx]
    except ImportError:
        pass
    return None, None


def _load_cf_from_csv(csv_file: Path):
    """Load Cf from SU2 surface_flow.csv (if it has Cf columns)."""
    import csv as csv_module
    with open(csv_file, 'r') as f:
        reader = csv_module.DictReader(f)
        headers = [h.strip().strip('"') for h in reader.fieldnames]
        f.seek(0)
        next(f)
        reader = csv_module.reader(f)

        x_vals, cf_vals = [], []
        for row in reader:
            data = {}
            for h, v in zip(headers, row):
                try:
                    data[h] = float(v.strip().strip('"'))
                except (ValueError, IndexError):
                    pass
            x = data.get("x")
            cf = data.get("Skin_Friction_Coefficient_0",
                          data.get("Skin_Friction_Coefficient-X"))
            if x is not None and cf is not None:
                x_vals.append(x)
                cf_vals.append(cf)

    if not x_vals:
        return None, None
    x_arr, cf_arr = np.array(x_vals), np.array(cf_vals)
    sort_idx = np.argsort(x_arr)
    return x_arr[sort_idx], cf_arr[sort_idx]


def compute_separation_metrics(x, cf, label=""):
    """Compute separation bubble metrics from Cf distribution."""
    metrics = {"label": label}

    # Find separation point (first Cf < 0)
    neg_mask = cf < 0
    if neg_mask.any():
        sep_idx = np.where(neg_mask)[0][0]
        metrics["x_sep"] = float(x[sep_idx])

        # Find reattachment (first Cf > 0 after separation)
        for i in range(sep_idx + 1, len(cf)):
            if cf[i] > 0:
                metrics["x_reat"] = float(x[i])
                break
        else:
            metrics["x_reat"] = None

        if metrics.get("x_reat"):
            metrics["bubble_length"] = metrics["x_reat"] - metrics["x_sep"]
        else:
            metrics["bubble_length"] = None

        # Min Cf in separation
        metrics["cf_min"] = float(cf[sep_idx:].min())
    else:
        metrics["x_sep"] = None
        metrics["x_reat"] = None
        metrics["bubble_length"] = None
        metrics["cf_min"] = None

    # Peak downstream Cf
    downstream = x > 0.38
    if downstream.any():
        metrics["cf_peak_downstream"] = float(cf[downstream].max())
    else:
        metrics["cf_peak_downstream"] = None

    return metrics


def compute_cf_error(x_sim, cf_sim, x_exp, cf_exp):
    """Compute Cf error interpolated to experimental x locations."""
    cf_interp = np.interp(x_exp, x_sim, cf_sim)
    abs_err = np.abs(cf_interp - cf_exp)

    # Average absolute error
    mae = float(np.mean(abs_err))

    # RMS error
    rmse = float(np.sqrt(np.mean((cf_interp - cf_exp)**2)))

    # Error in separation region (x = 0.28 to 0.34)
    sep_mask = (np.array(x_exp) >= 0.28) & (np.array(x_exp) <= 0.34)
    if sep_mask.any():
        sep_rmse = float(np.sqrt(np.mean(
            (cf_interp[sep_mask] - np.array(cf_exp)[sep_mask])**2)))
    else:
        sep_rmse = None

    return {"MAE": mae, "RMSE": rmse, "sep_RMSE": sep_rmse}


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("=" * 60)
    print("  SWBLI Cf Comparison: SA vs SST vs SA-comp")
    print("  Schülein Mach 5 SWBLI (DLR)")
    print("=" * 60)

    # Model configs: (dir_name, label, color, linestyle)
    models = [
        ("swbli_SA_L1_coarse", "SA (baseline)", "#2196F3", "-"),
        ("swbli_SST_L1_coarse", "SST", "#4CAF50", "--"),
        ("swbli_SA_COMP_L1_coarse", "SA-comp (Catris-Aupoix)", "#F44336", "-"),
    ]

    # Also check L2 medium if available
    models_l2 = [
        ("swbli_SA_L2_medium", "SA L2", "#1565C0", "-."),
        ("swbli_SA_COMP_L2_medium", "SA-comp L2", "#C62828", "-."),
    ]

    exp_x = np.array(EXPERIMENTAL_CF["x"])
    exp_cf = np.array(EXPERIMENTAL_CF["cf"])

    # Load all available models
    fig, ax = plt.subplots(figsize=(12, 6))

    results = {}
    all_models = models + models_l2

    for dir_name, label, color, ls in all_models:
        case_dir = RUNS_DIR / dir_name
        if not case_dir.exists():
            print(f"\n  [SKIP] {dir_name} — not found")
            continue

        x, cf = load_surface_cf(case_dir)
        if x is None:
            print(f"\n  [SKIP] {dir_name} — no surface_flow.csv")
            continue

        # Only plot in the region of interest
        mask = (x >= 0.04) & (x <= 0.52)
        x_plot, cf_plot = x[mask], cf[mask]

        ax.plot(x_plot, cf_plot, color=color, ls=ls, lw=1.5, label=label)

        # Compute metrics
        sep = compute_separation_metrics(x, cf, label)
        err = compute_cf_error(x, cf, exp_x, exp_cf)
        results[dir_name] = {"separation": sep, "error": err}

        print(f"\n  {label}:")
        if sep["x_sep"]:
            print(f"    Separation: x={sep['x_sep']:.4f} m")
            if sep["x_reat"]:
                print(f"    Reattachment: x={sep['x_reat']:.4f} m")
                print(f"    Bubble length: {sep['bubble_length']:.4f} m")
            print(f"    Min Cf: {sep['cf_min']:.6f}")
        else:
            print(f"    No separation detected")
        print(f"    Cf MAE vs experiment: {err['MAE']:.6f}")
        print(f"    Cf RMSE vs experiment: {err['RMSE']:.6f}")
        if err["sep_RMSE"]:
            print(f"    Cf RMSE in separation zone: {err['sep_RMSE']:.6f}")

    # Plot experimental data
    ax.scatter(exp_x, exp_cf, s=25, c="k", marker="o", zorder=5,
               label="Schülein exp.", edgecolors="k", lw=0.5)

    ax.set_xlabel("x (m)", fontsize=12)
    ax.set_ylabel("Cf", fontsize=12)
    ax.set_title("Mach 5 SWBLI — Skin Friction Comparison\n"
                 "Schülein et al. (DLR), 10° wedge shock on flat plate",
                 fontsize=13, fontweight="bold")
    ax.axhline(0, color="gray", ls=":", lw=0.8, alpha=0.5)
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.04, 0.52)

    fig.tight_layout()
    out_dir = PROJECT / "plots" / "swbli_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_path = out_dir / "cf_sa_sst_sacomp.png"
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  [PLOT] Saved: {plot_path}")

    # Compute improvement of SA-comp over SA baseline
    if "swbli_SA_L1_coarse" in results and "swbli_SA_COMP_L1_coarse" in results:
        sa_err = results["swbli_SA_L1_coarse"]["error"]
        comp_err = results["swbli_SA_COMP_L1_coarse"]["error"]

        print(f"\n  {'='*50}")
        print(f"  SA-comp vs SA baseline (L1 coarse):")
        print(f"  {'='*50}")
        if sa_err["RMSE"] > 0:
            improvement = (sa_err["RMSE"] - comp_err["RMSE"]) / sa_err["RMSE"] * 100
            print(f"    RMSE improvement: {improvement:+.1f}%")
        if sa_err["sep_RMSE"] and comp_err["sep_RMSE"] and sa_err["sep_RMSE"] > 0:
            sep_improvement = (sa_err["sep_RMSE"] - comp_err["sep_RMSE"]) / sa_err["sep_RMSE"] * 100
            print(f"    Separation zone RMSE improvement: {sep_improvement:+.1f}%")

    # Save results
    results_file = out_dir / "comparison_metrics.json"
    # Convert numpy types
    def to_serializable(obj):
        if isinstance(obj, (np.floating, float)):
            return float(obj)
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        return obj

    serializable = {}
    for k, v in results.items():
        serializable[k] = {
            "separation": {sk: to_serializable(sv) for sk, sv in v["separation"].items()},
            "error": {ek: to_serializable(ev) for ek, ev in v["error"].items()},
        }
    with open(results_file, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"  [DATA] Saved: {results_file}")


if __name__ == "__main__":
    main()
