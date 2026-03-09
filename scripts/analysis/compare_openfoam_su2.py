#!/usr/bin/env python3
"""Cross-solver comparison: SU2 vs OpenFOAM vs experimental Cp on wall hump."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import os, sys, json

PROJECT = Path(r"c:\Users\yuvra\OneDrive\Desktop\Python\Projects\CFD Solver Benchmark for Flow Separation Prediction")
RESULTS = PROJECT / "results" / "openfoam_comparison"
RESULTS.mkdir(parents=True, exist_ok=True)

# ===================================================================
# 1. Load OpenFOAM Cp
# ===================================================================
of_cp_file = PROJECT / "results" / "fiml_correction" / "wall_cp_openfoam.csv"
of_cp = np.loadtxt(of_cp_file, delimiter=",")
x_of, cp_of = of_cp[:, 0], of_cp[:, 1]
print(f"OpenFOAM: {len(x_of)} pts, x=[{x_of.min():.2f},{x_of.max():.2f}], Cp=[{cp_of.min():.3f},{cp_of.max():.3f}]")

# ===================================================================
# 2. Load SU2 Cp from VTU
# ===================================================================
try:
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy

    # Load surface VTU
    su2_dir = PROJECT / "runs" / "wall_hump" / "hump_SA_medium"
    vtu_file = su2_dir / "surface_flow.vtu"

    if vtu_file.exists():
        reader = vtk.vtkXMLUnstructuredGridReader()
        reader.SetFileName(str(vtu_file))
        reader.Update()
        data = reader.GetOutput()

        pts_vtk = vtk_to_numpy(data.GetPoints().GetData())
        x_su2 = pts_vtk[:, 0]

        cp_arr = data.GetPointData().GetArray("Pressure_Coefficient")
        if cp_arr is not None:
            cp_su2 = vtk_to_numpy(cp_arr)
            # Sort by x
            idx = x_su2.argsort()
            x_su2, cp_su2 = x_su2[idx], cp_su2[idx]
            print(f"SU2 (VTU): {len(x_su2)} pts, x=[{x_su2.min():.2f},{x_su2.max():.2f}]")
        else:
            print("No Pressure_Coefficient in VTU")
            x_su2, cp_su2 = None, None
    else:
        # Try CSV
        csv_file = su2_dir / "surface_flow.csv"
        if csv_file.exists():
            import csv
            with open(csv_file) as f:
                reader = csv.DictReader(f)
                data = list(reader)
            x_su2 = np.array([float(d.get("x", 0)) for d in data])
            cp_su2 = np.array([float(d.get("Pressure_Coefficient", 0)) for d in data])
            idx = x_su2.argsort()
            x_su2, cp_su2 = x_su2[idx], cp_su2[idx]
            print(f"SU2 (CSV): {len(x_su2)} pts")
        else:
            x_su2, cp_su2 = None, None
            print("No SU2 surface data")
except ImportError:
    print("VTK not available, trying CSV")
    x_su2, cp_su2 = None, None

# ===================================================================
# 3. Load experimental Cp
# ===================================================================
exp_cp_file = PROJECT / "experimental_data" / "wall_hump" / "noflow_cp.exp.dat"
if exp_cp_file.exists():
    lines = exp_cp_file.read_text().strip().split("\n")
    exp_data = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 2:
            try:
                exp_data.append([float(parts[0]), float(parts[1])])
            except ValueError:
                pass
    exp_data = np.array(exp_data)
    x_exp, cp_exp = exp_data[:, 0], exp_data[:, 1]
    print(f"Exp Cp: {len(x_exp)} pts, x=[{x_exp.min():.2f},{x_exp.max():.2f}]")
else:
    x_exp, cp_exp = None, None
    print("No experimental Cp data")

# ===================================================================
# 4. Plot Cp comparison
# ===================================================================
fig, ax = plt.subplots(figsize=(12, 6))

if x_exp is not None:
    ax.plot(x_exp, cp_exp, "ko", ms=4, label="Experiment (Greenblatt)")

if x_su2 is not None:
    ax.plot(x_su2, cp_su2, "b-", lw=1.5, label="SU2 SA (medium grid)")

ax.plot(x_of, cp_of, "r--", lw=1.5, label="OpenFOAM SA (Gmsh grid)")

ax.set_xlabel("x/c", fontsize=14)
ax.set_ylabel("Cp", fontsize=14)
ax.set_title("Wall Hump Cp: SU2 vs OpenFOAM vs Experiment", fontsize=16)
ax.legend(fontsize=12)
ax.set_xlim([-0.5, 2.0])
ax.invert_yaxis()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(RESULTS / "cp_cross_solver.png", dpi=150)
print(f"\nSaved: {RESULTS / 'cp_cross_solver.png'}")

# ===================================================================
# 5. Cf Comparison (NEW)
# ===================================================================
# Load experimental Cf
exp_cf_file = PROJECT / "experimental_data" / "wall_hump" / "noflow_cf.exp.dat"
x_exp_cf, cf_exp = None, None
if exp_cf_file.exists():
    lines_cf = exp_cf_file.read_text().strip().split("\n")
    cf_data = []
    for line in lines_cf:
        parts = line.strip().split()
        if len(parts) >= 2:
            try:
                cf_data.append([float(parts[0]), float(parts[1])])
            except ValueError:
                pass
    if cf_data:
        cf_data = np.array(cf_data)
        x_exp_cf, cf_exp = cf_data[:, 0], cf_data[:, 1]
        print(f"Exp Cf: {len(x_exp_cf)} pts")

# Load SU2 Cf from VTU
cf_su2, x_su2_cf = None, None
try:
    if x_su2 is not None:
        import vtk
        from vtk.util.numpy_support import vtk_to_numpy
        su2_dir = PROJECT / "runs" / "wall_hump" / "hump_SA_medium"
        vtu_file = su2_dir / "surface_flow.vtu"
        if vtu_file.exists():
            reader = vtk.vtkXMLUnstructuredGridReader()
            reader.SetFileName(str(vtu_file))
            reader.Update()
            data = reader.GetOutput()
            cf_arr = data.GetPointData().GetArray("Skin_Friction_Coefficient")
            if cf_arr is not None:
                cf_raw = vtk_to_numpy(cf_arr)
                pts_vtk = vtk_to_numpy(data.GetPoints().GetData())
                x_su2_cf = pts_vtk[:, 0]
                cf_su2 = cf_raw[:, 0] if cf_raw.ndim > 1 else cf_raw
                idx = x_su2_cf.argsort()
                x_su2_cf, cf_su2 = x_su2_cf[idx], cf_su2[idx]
except ImportError:
    pass

# Plot Cf comparison
fig2, ax2 = plt.subplots(figsize=(12, 6))
if x_exp_cf is not None:
    ax2.plot(x_exp_cf, cf_exp, "ko", ms=4, label="Experiment (Greenblatt)")
if cf_su2 is not None:
    ax2.plot(x_su2_cf, cf_su2, "b-", lw=1.5, label="SU2 SA (medium)")
# OpenFOAM Cf if available
of_cf_file = PROJECT / "results" / "fiml_correction" / "wall_cf_openfoam.csv"
if of_cf_file.exists():
    of_cf = np.loadtxt(of_cf_file, delimiter=",")
    ax2.plot(of_cf[:, 0], of_cf[:, 1], "r--", lw=1.5, label="OpenFOAM SA")
ax2.axhline(y=0, color="k", ls="--", alpha=0.3)
ax2.set_xlabel("x/c", fontsize=14)
ax2.set_ylabel("Cf", fontsize=14)
ax2.set_title("Wall Hump Cf: SU2 vs OpenFOAM vs Experiment", fontsize=16)
ax2.legend(fontsize=12)
ax2.set_xlim([-0.5, 2.0])
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(RESULTS / "cf_cross_solver.png", dpi=150)
print(f"Saved: {RESULTS / 'cf_cross_solver.png'}")

# ===================================================================
# 6. RMSE Metrics
# ===================================================================
metrics = {}

if x_exp is not None and x_su2 is not None:
    from scipy.interpolate import interp1d
    mask = (x_exp >= x_su2.min()) & (x_exp <= x_su2.max())
    if mask.sum() > 0:
        f_su2 = interp1d(x_su2, cp_su2, bounds_error=False, fill_value="extrapolate")
        cp_su2_interp = f_su2(x_exp[mask])
        rmse_su2 = np.sqrt(np.mean((cp_su2_interp - cp_exp[mask])**2))
        metrics["su2_cp_rmse"] = float(rmse_su2)
        print(f"SU2 Cp RMSE: {rmse_su2:.4f}")

if x_exp is not None:
    from scipy.interpolate import interp1d
    mask = (x_exp >= x_of.min()) & (x_exp <= x_of.max())
    if mask.sum() > 0:
        f_of = interp1d(x_of, cp_of, bounds_error=False, fill_value="extrapolate")
        cp_of_interp = f_of(x_exp[mask])
        rmse_of = np.sqrt(np.mean((cp_of_interp - cp_exp[mask])**2))
        metrics["openfoam_cp_rmse"] = float(rmse_of)
        print(f"OpenFOAM Cp RMSE: {rmse_of:.4f}")

# Cf RMSE
if x_exp_cf is not None and cf_su2 is not None:
    from scipy.interpolate import interp1d
    mask = (x_exp_cf >= x_su2_cf.min()) & (x_exp_cf <= x_su2_cf.max())
    if mask.sum() > 0:
        f_su2_cf = interp1d(x_su2_cf, cf_su2, bounds_error=False, fill_value="extrapolate")
        cf_su2_interp = f_su2_cf(x_exp_cf[mask])
        rmse_su2_cf = np.sqrt(np.mean((cf_su2_interp - cf_exp[mask])**2))
        metrics["su2_cf_rmse"] = float(rmse_su2_cf)
        print(f"SU2 Cf RMSE: {rmse_su2_cf:.6f}")

# ===================================================================
# 7. Separation Metrics Side-by-Side Table
# ===================================================================
print("\n" + "=" * 75)
print("  CROSS-SOLVER SEPARATION METRICS")
print("=" * 75)

def find_sep_reat(x, cf):
    """Find separation and reattachment from Cf zero-crossings."""
    x_s, x_r = None, None
    for i in range(len(cf) - 1):
        if cf[i] > 0 and cf[i+1] < 0 and x[i] > 0.5:
            x_s = float(x[i] - cf[i] * (x[i+1] - x[i]) / (cf[i+1] - cf[i]))
        if cf[i] < 0 and cf[i+1] > 0 and x[i] > 0.8:
            x_r = float(x[i] - cf[i] * (x[i+1] - x[i]) / (cf[i+1] - cf[i]))
    return x_s, x_r

sep_table = {"Experiment": {"x_sep": 0.665, "x_reat": 1.11, "L_bubble": 0.445}}

if cf_su2 is not None:
    xs, xr = find_sep_reat(x_su2_cf, cf_su2)
    sep_table["SU2 SA"] = {
        "x_sep": xs, "x_reat": xr,
        "L_bubble": (xr - xs) if xs and xr else None,
        "Cf_min": float(cf_su2[(x_su2_cf > 0.5) & (x_su2_cf < 1.5)].min()) if cf_su2 is not None else None,
    }

if of_cf_file.exists():
    of_cf_d = np.loadtxt(of_cf_file, delimiter=",")
    xs_of, xr_of = find_sep_reat(of_cf_d[:, 0], of_cf_d[:, 1])
    sep_table["OpenFOAM SA"] = {
        "x_sep": xs_of, "x_reat": xr_of,
        "L_bubble": (xr_of - xs_of) if xs_of and xr_of else None,
        "Cf_min": float(of_cf_d[:, 1][(of_cf_d[:, 0] > 0.5) & (of_cf_d[:, 0] < 1.5)].min()),
    }

print(f"\n  {'Solver':<18s} {'x_sep':>7s} {'x_reat':>7s} {'L_bubble':>9s} {'Cf_min':>10s}")
print("  " + "-" * 55)
for solver, vals in sep_table.items():
    xs = f"{vals.get('x_sep', 0):.3f}" if vals.get("x_sep") else "---"
    xr = f"{vals.get('x_reat', 0):.3f}" if vals.get("x_reat") else "---"
    lb = f"{vals.get('L_bubble', 0):.3f}" if vals.get("L_bubble") else "---"
    cf = f"{vals.get('Cf_min', 0):.6f}" if vals.get("Cf_min") else "---"
    print(f"  {solver:<18s} {xs:>7s} {xr:>7s} {lb:>9s} {cf:>10s}")

metrics["separation_table"] = sep_table

# Solver metadata
metrics["su2_wall_distance"] = "Exact point search (geometric)"
metrics["openfoam_wall_distance"] = "meshWave (PDE-based approximate)"
metrics["openfoam_mesh"] = "Gmsh structured quad 205x55 (11275 cells)"
metrics["su2_mesh"] = "TMR structured quad 205x55 (11016 cells)"
metrics["openfoam_convergence"] = "254 SIMPLE iterations"
metrics["openfoam_version"] = "OpenFOAM 13"

with open(RESULTS / "cross_solver_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2, default=lambda o: float(o) if hasattr(o, 'item') else str(o))
print(f"\nSaved metrics: {RESULTS / 'cross_solver_metrics.json'}")

print("\n=== SUMMARY ===")
for k, v in metrics.items():
    if not isinstance(v, dict):
        print(f"  {k}: {v}")

