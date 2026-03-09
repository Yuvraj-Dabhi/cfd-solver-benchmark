#!/usr/bin/env python3
"""
plot_swbli_validation.py — Cf validation for Mach 5 SWBLI
==========================================================
Compares SU2 skin friction against Schülein experimental data.
Shows separation / reattachment prediction for SA and SST.
"""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

try:
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy
except ImportError:
    print("VTK not available"); sys.exit(1)

PROJECT = Path(__file__).resolve().parent
RUNS_DIR = PROJECT / "runs" / "swbli"
EXP_DIR = PROJECT / "experimental_data" / "swbli"
OUT_DIR = PROJECT / "plots" / "swbli"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Style ---
plt.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.titlesize": 14, "axes.labelsize": 12,
    "legend.fontsize": 9, "savefig.dpi": 200, "savefig.bbox": "tight",
})
COLORS = {
    "SA_L1_coarse": "#ef9a9a", "SA_L2_medium": "#e53935", "SA_L3_fine": "#b71c1c",
    "SST_L1_coarse": "#90caf9", "SST_L2_medium": "#1e88e5", "SST_L3_fine": "#0d47a1",
    "exp": "#2e7d32",
}
LS = {"coarse": "--", "medium": "-", "fine": ":"}


def read_vtu(path):
    """Read VTU surface output."""
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(str(path))
    reader.Update()
    out = reader.GetOutput()
    coords = vtk_to_numpy(out.GetPoints().GetData())
    pd = out.GetPointData()
    arrays = {}
    for i in range(pd.GetNumberOfArrays()):
        name = pd.GetArrayName(i)
        arrays[name] = vtk_to_numpy(pd.GetArray(i))
    arrays["coords"] = coords
    return arrays


def find_separation(x, cf):
    """Find separation and reattachment from Cf sign changes."""
    sep, reat = None, None
    sign_changes = np.where(np.diff(np.sign(cf)))[0]
    for i in sign_changes:
        xc = x[i] + (x[i+1] - x[i]) * (-cf[i]) / (cf[i+1] - cf[i])
        if cf[i] > 0 and cf[i+1] < 0 and sep is None:
            sep = xc
        elif cf[i] < 0 and cf[i+1] > 0:
            reat = xc
    return sep, reat


def main():
    print("=" * 60)
    print("  SWBLI — Cf VALIDATION (Mach 5, Schülein)")
    print("=" * 60)

    # Load experimental data
    exp_file = EXP_DIR / "exp_cf.json"
    if exp_file.exists():
        exp = json.loads(exp_file.read_text())
        xe = np.array(exp["x"])
        cfe = np.array(exp["cf"])
        print(f"  Experimental Cf: {len(xe)} points")
        # Experimental sep/reat
        sep_e, reat_e = find_separation(xe, cfe)
        print(f"  Exp separation: x = {sep_e:.4f} m" if sep_e else "  Exp: no separation")
        print(f"  Exp reattachment: x = {reat_e:.4f} m" if reat_e else "  Exp: no reattachment")
    else:
        xe, cfe = None, None
        print("  [WARN] No experimental data found")

    # Scan for simulation results
    sim_data = []
    for model in ["SA", "SST"]:
        for grid in ["L1_coarse", "L2_medium", "L3_fine"]:
            case = f"swbli_{model}_{grid}"
            vtu = RUNS_DIR / case / "surface_flow.vtu"
            if not vtu.exists():
                vtu = RUNS_DIR / case / "surface.vtu"
            if not vtu.exists():
                continue
            label = f"{model}_{grid}"
            print(f"\n  Reading {label}...")
            d = read_vtu(vtu)
            x = d["coords"][:, 0]
            cf_vec = d.get("Skin_Friction_Coefficient")
            if cf_vec is None:
                print(f"    [WARN] No Cf data")
                continue
            cf = cf_vec[:, 0] if cf_vec.ndim > 1 else cf_vec
            # Also get wall pressure for Cp
            P = d.get("Pressure")

            idx = np.argsort(x)
            x, cf = x[idx], cf[idx]
            if P is not None:
                P = P[idx]

            # Filter to bottom wall (y ≈ 0)
            mask = np.abs(d["coords"][idx, 1]) < 0.001
            x, cf = x[mask], cf[mask]
            if P is not None:
                P = P[mask]

            sep, reat = find_separation(x, cf)
            print(f"    Cf range: [{cf.min():.6f}, {cf.max():.6f}]")
            if sep:
                print(f"    Separation:   x = {sep:.4f} m")
            if reat:
                print(f"    Reattachment: x = {reat:.4f} m")
            if sep and reat:
                print(f"    Bubble length: {reat - sep:.4f} m")

            sim_data.append({
                "x": x, "cf": cf, "P": P,
                "model": model, "grid": grid, "label": label,
                "sep": sep, "reat": reat,
            })

    if not sim_data:
        print("\n  No simulation data found!"); return

    # ─── Plot 1: Cf comparison ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))
    if xe is not None:
        ax.plot(xe * 1000, cfe, "o", color=COLORS["exp"], ms=5, alpha=0.8,
                label="Experiment (Schülein)", zorder=5)
    for s in sim_data:
        ls = LS.get(s["grid"].split("_")[-1], "-")
        ax.plot(s["x"] * 1000, s["cf"], ls,
                color=COLORS.get(s["label"], "#333"),
                lw=1.8, label=f'SU2 {s["label"]}', zorder=4)
    ax.axhline(0, color="gray", ls="-", lw=0.8, alpha=0.5)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("$C_f$")
    ax.set_title("Mach 5 SWBLI — Skin Friction Coefficient")
    ax.legend(loc="upper left", ncol=2, framealpha=0.9)
    ax.set_xlim(50, 500)
    ax.grid(True, alpha=0.3)
    fname = OUT_DIR / "swbli_cf_validation.png"
    fig.savefig(fname)
    plt.close(fig)
    print(f"\n  Saved: {fname.name}")

    # ─── Plot 2: Separation region detail ──────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    if xe is not None:
        ax.plot(xe * 1000, cfe, "o", color=COLORS["exp"], ms=6, alpha=0.8,
                label="Experiment (Schülein)", zorder=5)
    for s in sim_data:
        ls = LS.get(s["grid"].split("_")[-1], "-")
        ax.plot(s["x"] * 1000, s["cf"], ls,
                color=COLORS.get(s["label"], "#333"),
                lw=2.0, label=f'SU2 {s["label"]}', zorder=4)
    ax.axhline(0, color="gray", ls="-", lw=0.8, alpha=0.5)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("$C_f$")
    ax.set_title("SWBLI Separation Region — Cf Detail")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.set_xlim(250, 420)
    ax.set_ylim(-0.001, 0.002)
    ax.grid(True, alpha=0.3)
    fname2 = OUT_DIR / "swbli_separation_detail.png"
    fig.savefig(fname2)
    plt.close(fig)
    print(f"  Saved: {fname2.name}")

    # ─── Metrics ──────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("  SWBLI VALIDATION METRICS")
    print("=" * 50)
    for s in sim_data:
        print(f"\n  {s['label']}:")
        if xe is not None and len(s["x"]) > 0:
            # Interpolate sim to exp x for RMSE
            cf_interp = np.interp(xe, s["x"], s["cf"])
            rmse = np.sqrt(np.mean((cf_interp - cfe) ** 2))
            print(f"    Cf RMSE = {rmse:.6f}")
        if s["sep"]:
            print(f"    Separation:   x = {s['sep']*1000:.1f} mm", end="")
            if sep_e:
                print(f"  (exp: {sep_e*1000:.1f} mm, err: {abs(s['sep']-sep_e)/sep_e*100:.1f}%)")
            else:
                print()
        if s["reat"]:
            print(f"    Reattachment: x = {s['reat']*1000:.1f} mm", end="")
            if reat_e:
                print(f"  (exp: {reat_e*1000:.1f} mm, err: {abs(s['reat']-reat_e)/reat_e*100:.1f}%)")
            else:
                print()
        if s["sep"] and s["reat"]:
            bubble = (s["reat"] - s["sep"]) * 1000
            print(f"    Bubble: {bubble:.1f} mm")

    # Save metrics as JSON
    metrics = []
    for s in sim_data:
        m = {"model": s["label"],
             "sep_mm": float(s["sep"]*1000) if s["sep"] else None,
             "reat_mm": float(s["reat"]*1000) if s["reat"] else None}
        if xe is not None:
            cf_interp = np.interp(xe, s["x"], s["cf"])
            m["cf_rmse"] = float(np.sqrt(np.mean((cf_interp - cfe) ** 2)))
        metrics.append(m)
    (OUT_DIR / "swbli_metrics.json").write_text(json.dumps(metrics, indent=2))

    print(f"\n  Output: {OUT_DIR}")
    print("  Done!")


if __name__ == "__main__":
    main()
