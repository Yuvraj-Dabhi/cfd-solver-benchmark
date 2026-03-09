#!/usr/bin/env python3
"""
run_bump_channel.py — TMR 2D Bump-in-Channel Verification Case
================================================================
Based on the SU2 V&V Collection bump-in-channel case.
Runs SA and SST on TMR grids, extracts Cf at x=0.63, 0.75, 0.87,
compares against CFL3D/FUN3D reference data.

Flow conditions:
  M = 0.2, Re = 3×10⁶, T = 300 K, L_ref = 1.0 m

Usage:
    python run_bump_channel.py --model SA --grid 0089x041
    python run_bump_channel.py --plot-only
"""
import argparse, gzip, json, math, shutil, subprocess, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

try:
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy
except ImportError:
    print("VTK required"); sys.exit(1)

PROJECT = Path(__file__).resolve().parent
GRID_DIR = PROJECT / "experimental_data" / "bump_channel" / "grids"
RUNS_DIR = PROJECT / "runs" / "bump_channel"
OUT_DIR = PROJECT / "plots" / "bump_channel"
RUNS_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── TMR Flow Conditions ──────────────────────────────────────────────
MACH = 0.2
T_INF = 300.0
RE = 3.0e6
GAMMA = 1.4; R_GAS = 287.058
A_INF = math.sqrt(GAMMA * R_GAS * T_INF)
U_INF = MACH * A_INF
MU_INF = 1.716e-5 * (T_INF / 273.15)**1.5 * (273.15 + 110.4) / (T_INF + 110.4)
RHO_INF = RE * MU_INF / (U_INF * 1.0)

# CFL3D reference Cf at key stations (from TMR, SA model, finest grid)
CFL3D_REF = {
    "source": "CFL3D (TMR, SA, finest grid)",
    "x_cf": [0.0, 0.063, 0.125, 0.188, 0.25, 0.313, 0.375, 0.438, 0.5,
             0.563, 0.625, 0.688, 0.75, 0.813, 0.875, 0.938, 1.0, 1.063,
             1.125, 1.188, 1.25, 1.313, 1.375, 1.438, 1.5],
    "cf": [0.00365, 0.00360, 0.00355, 0.00348, 0.00340, 0.00338, 0.00342,
           0.00360, 0.00398, 0.00460, 0.00535, 0.00575, 0.00540, 0.00450,
           0.00380, 0.00340, 0.00320, 0.00313, 0.00310, 0.00308, 0.00307,
           0.00306, 0.00305, 0.00304, 0.00303],
}

GRIDS = {
    "89x41": {"file": "bump_89x41.su2", "desc": "Coarse (89x41, 3.5K cells)"},
    "177x81": {"file": "bump_177x81.su2", "desc": "Medium (177x81, 14K cells)"},
}


def get_grid(grid_name):
    """Get grid path (SU2 format, converted from TMR PLOT3D)."""
    grid_path = GRID_DIR / GRIDS[grid_name]["file"]
    if grid_path.exists():
        return grid_path
    print(f"  [ERR] Grid not found: {grid_path}")
    print(f"  Run the PLOT3D-to-SU2 conversion first.")
    return None


def generate_config(case_dir, mesh_file, model="SA", n_iter=99999):
    """Generate SU2 config for bump-in-channel case."""
    turb = "KIND_TURB_MODEL= SA" if model == "SA" else (
        "KIND_TURB_MODEL= SST\nSST_OPTIONS= V1994m\n"
        "FREESTREAM_TURBULENCEINTENSITY= 0.00038729\n"
        "FREESTREAM_TURB2LAMVISCRATIO= 0.009"
    )
    screen_turb = "RMS_NU_TILDE" if model == "SA" else "RMS_TKE"

    cfg = f"""\
% TMR 2D Bump-in-Channel — {model}
% M={MACH}, Re={RE:.0e}, T={T_INF}K
SOLVER= RANS
{turb}
MATH_PROBLEM= DIRECT
RESTART_SOL= NO

MACH_NUMBER= {MACH}
AOA= 0.0
FREESTREAM_TEMPERATURE= {T_INF}
REYNOLDS_NUMBER= {RE}
REYNOLDS_LENGTH= 1.0

REF_ORIGIN_MOMENT_X= 0.0
REF_ORIGIN_MOMENT_Y= 0.0
REF_ORIGIN_MOMENT_Z= 0.0
REF_LENGTH= 1.5
REF_AREA= 1.5

MARKER_HEATFLUX= ( bump, 0.0 )
MARKER_SYM= ( lower_upstream, lower_downstream, upper )
MARKER_INLET= ( inlet, 302.4, 70614.866784, 1.0, 0.0, 0.0 )
MARKER_OUTLET= ( outlet, 68672.8 )
MARKER_PLOTTING= ( bump )
MARKER_MONITORING= ( bump )

NUM_METHOD_GRAD= WEIGHTED_LEAST_SQUARES
NUM_METHOD_GRAD_RECON= LEAST_SQUARES
CFL_NUMBER= 10.0
CFL_ADAPT= YES
CFL_ADAPT_PARAM= ( 0.1, 1.2, 10.0, 1e4 )
ITER= {n_iter}

LINEAR_SOLVER= FGMRES
LINEAR_SOLVER_PREC= ILU
LINEAR_SOLVER_ILU_FILL_IN= 0
LINEAR_SOLVER_ERROR= 1E-10
LINEAR_SOLVER_ITER= 25

CONV_NUM_METHOD_FLOW= ROE
MUSCL_FLOW= YES
SLOPE_LIMITER_FLOW= NONE
TIME_DISCRE_FLOW= EULER_IMPLICIT
VENKAT_LIMITER_COEFF= 0.05

CONV_NUM_METHOD_TURB= SCALAR_UPWIND
MUSCL_TURB= NO
TIME_DISCRE_TURB= EULER_IMPLICIT

CONV_FIELD= RMS_DENSITY
CONV_RESIDUAL_MINVAL= -13

MESH_FILENAME= {mesh_file}
MESH_FORMAT= SU2
TABULAR_FORMAT= CSV
CONV_FILENAME= history
VOLUME_FILENAME= flow
SURFACE_FILENAME= surface_flow
SCREEN_OUTPUT= INNER_ITER WALL_TIME RMS_DENSITY {screen_turb} DRAG
OUTPUT_FILES= RESTART PARAVIEW SURFACE_PARAVIEW SURFACE_CSV
OUTPUT_WRT_FREQ= 10000
HISTORY_WRT_FREQ_INNER= 1
HISTORY_OUTPUT= ITER RMS_RES AERO_COEFF
"""
    path = case_dir / "bump.cfg"
    path.write_text(cfg)
    return path


def run_simulation(case_dir, cfg_path, timeout=7200):
    """Run SU2 on bump-in-channel case."""
    su2 = shutil.which("SU2_CFD") or shutil.which("SU2_CFD.exe") or "SU2_CFD"
    log = case_dir / "su2_log.txt"
    print(f"  [RUN] {su2} {cfg_path.name}")
    t0 = time.time()
    with open(log, "w") as f:
        proc = subprocess.run([su2, cfg_path.name], cwd=str(case_dir),
                              stdout=f, stderr=subprocess.STDOUT, timeout=timeout)
    dt = time.time() - t0
    if proc.returncode != 0:
        tail = log.read_text(errors="replace").splitlines()[-10:]
        print(f"  [FAIL] code={proc.returncode}")
        for l in tail: print(f"    {l}")
        return False
    hist = case_dir / "history.csv"
    if hist.exists():
        n = sum(1 for _ in open(hist, errors="replace")) - 1
        last = hist.read_text(errors="replace").strip().split("\n")[-1]
        cols = last.split(",")
        rms = float(cols[3].strip().strip('"')) if len(cols) > 3 else 0
        conv = "GOOD" if rms < -10 else "FAIR" if rms < -6 else "POOR"
        print(f"  [OK] {n} iter, rms={rms:.2f} ({conv}), time={dt:.0f}s")
    return True


def extract_cf(case_dir):
    """Extract Cf from surface VTU."""
    for name in ["surface_flow.vtu", "surface.vtu"]:
        p = case_dir / name
        if p.exists():
            reader = vtk.vtkXMLUnstructuredGridReader()
            reader.SetFileName(str(p))
            reader.Update()
            out = reader.GetOutput()
            coords = vtk_to_numpy(out.GetPoints().GetData())
            pd = out.GetPointData()
            names = [pd.GetArrayName(i) for i in range(pd.GetNumberOfArrays())]
            if "Skin_Friction_Coefficient" not in names:
                return None
            cf = vtk_to_numpy(pd.GetArray(names.index("Skin_Friction_Coefficient")))
            cf_x = cf[:, 0] if cf.ndim > 1 else cf
            x = coords[:, 0]
            idx = np.argsort(x)
            return {"x": x[idx], "cf": cf_x[idx]}
    return None


def main():
    parser = argparse.ArgumentParser(description="TMR Bump-in-Channel")
    parser.add_argument("--model", nargs="+", default=["SA"], choices=["SA", "SST"])
    parser.add_argument("--grid", nargs="+", default=["89x41"],
                        choices=list(GRIDS.keys()))
    parser.add_argument("--n-iter", type=int, default=99999)
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("  TMR BUMP-IN-CHANNEL Verification")
    print("=" * 60)
    print(f"  M={MACH}, Re={RE:.0e}, T={T_INF}K")

    sim_cf = {}
    for grid in args.grid:
        grid_path = get_grid(grid)
        if grid_path is None: continue

        for model in args.model:
            label = f"{model}_{grid}"
            case_dir = RUNS_DIR / f"bump_{label}"
            case_dir.mkdir(parents=True, exist_ok=True)

            # Copy mesh
            mesh_dst = case_dir / GRIDS[grid]["file"]
            if not mesh_dst.exists():
                shutil.copy2(grid_path, mesh_dst)

            if not args.plot_only:
                cfg = generate_config(case_dir, GRIDS[grid]["file"], model, args.n_iter)
                su2 = shutil.which("SU2_CFD") or shutil.which("SU2_CFD.exe")
                if not su2:
                    print("  [ERR] SU2_CFD not found"); return
                ok = run_simulation(case_dir, cfg, timeout=args.timeout)
                if not ok: continue

            # Extract Cf
            cf_data = extract_cf(case_dir)
            if cf_data:
                sim_cf[label] = cf_data
                # Report Cf at key stations
                for xs in [0.63, 0.75, 0.87]:
                    m = np.abs(cf_data["x"] - xs) < 0.02
                    if m.any():
                        print(f"    Cf(x={xs}) = {cf_data['cf'][m].mean():.6f}")

    # ─── Plot Cf vs x ──────────────────────────────────────────────────
    if sim_cf or not args.plot_only:
        fig, ax = plt.subplots(figsize=(12, 6))

        # CFL3D reference
        ax.plot(CFL3D_REF["x_cf"], CFL3D_REF["cf"], "ks-", ms=4, lw=1.2,
                alpha=0.7, label="CFL3D (TMR ref)", zorder=5)

        colors = {"SA_0089x041": "#ef9a9a", "SA_0177x081": "#e53935", "SA_0353x161": "#b71c1c",
                  "SST_0089x041": "#90caf9", "SST_0177x081": "#1e88e5", "SST_0353x161": "#0d47a1"}
        ls_map = {"0089x041": "--", "0177x081": "-", "0353x161": "-."}

        for label, data in sim_cf.items():
            parts = label.split("_", 1)
            grid_key = parts[1] if len(parts) > 1 else "0089x041"
            ls = ls_map.get(grid_key, "-")
            ax.plot(data["x"], data["cf"], ls, color=colors.get(label, "#333"),
                    lw=1.8, label=f"SU2 {label}", zorder=4)

        ax.set_xlabel("x (m)", fontsize=13)
        ax.set_ylabel("$C_f$", fontsize=13)
        ax.set_title("TMR Bump-in-Channel -- Skin Friction Coefficient", fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.1, 1.6)
        fname = OUT_DIR / "bump_channel_cf.png"
        fig.savefig(fname, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"\n  Saved: {fname}")

    # ─── Plot Cf at key stations  ──────────────────────────────────────
    if sim_cf:
        stations = [0.63, 0.75, 0.87]
        print("\n" + "=" * 50)
        print("  BUMP-IN-CHANNEL METRICS")
        print("=" * 50)
        for label, data in sim_cf.items():
            print(f"\n  {label}:")
            for xs in stations:
                m = np.abs(data["x"] - xs) < 0.02
                if m.any():
                    cf_val = data["cf"][m].mean()
                    # Find CFL3D ref
                    ref_idx = np.argmin(np.abs(np.array(CFL3D_REF["x_cf"]) - xs))
                    cf_ref = CFL3D_REF["cf"][ref_idx]
                    err = abs(cf_val - cf_ref) / cf_ref * 100
                    print(f"    Cf(x={xs}) = {cf_val:.6f}  (CFL3D: {cf_ref:.6f}, err: {err:.1f}%)")

    print(f"\n  Output: {OUT_DIR}")
    print("  Done!")


if __name__ == "__main__":
    main()
