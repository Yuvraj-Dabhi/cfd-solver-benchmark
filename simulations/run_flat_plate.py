#!/usr/bin/env python3
"""
run_flat_plate.py — TMR Flat Plate Verification (MRR Level 0)
=============================================================
Runs SU2 on the TMR 2D zero-pressure-gradient flat plate grid,
extracts U⁺ vs y⁺ at x = 0.97 m, and compares against the law of the wall.

This converts the flat plate verification from formula-vs-formula
to a genuine solver verification (MRR Level 0).

Flow conditions (TMR standard):
  M = 0.2, Re = 5×10⁶ (per meter), T = 300 K, L_ref = 1.0 m

Usage:
    python run_flat_plate.py                   # Run SA on 137x97 grid
    python run_flat_plate.py --model SST       # Run SST
    python run_flat_plate.py --plot-only        # Just plot existing results
"""
import argparse, math, shutil, subprocess, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

try:
    import pyvista as pv
except ImportError:
    print("PyVista required: pip install pyvista"); sys.exit(1)

PROJECT = Path(__file__).resolve().parent
GRID_DIR = PROJECT / "experimental_data" / "flat_plate" / "grids"
RUNS_DIR = PROJECT / "runs" / "flat_plate"
OUT_DIR = PROJECT / "plots" / "flat_plate"
RUNS_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── TMR Flow Conditions ──────────────────────────────────────────────
MACH = 0.2
T_INF = 300.0          # K
RE = 5.0e6             # per meter
RE_LENGTH = 1.0        # m
GAMMA = 1.4
R_GAS = 287.058

# Derived
A_INF = math.sqrt(GAMMA * R_GAS * T_INF)
U_INF = MACH * A_INF
RHO_INF = RE * 1.716e-5 * (T_INF / 273.15)**1.5 * (273.15 + 110.4) / (T_INF + 110.4) / (U_INF * RE_LENGTH)
# More precise: use actual Sutherland viscosity
MU_INF = 1.716e-5 * (T_INF / 273.15)**1.5 * (273.15 + 110.4) / (T_INF + 110.4)
RHO_INF = RE * MU_INF / (U_INF * RE_LENGTH)

GRIDS = {
    "069x049": "mesh_069x049.su2",
    "137x097": "mesh_137x097.su2",
}


def generate_config(case_dir, mesh_file, model="SA", n_iter=99999):
    """Generate SU2 config for flat plate verification."""
    turb_block = ""
    if model == "SA":
        turb_block = "KIND_TURB_MODEL= SA"
    else:
        turb_block = (
            "KIND_TURB_MODEL= SST\n"
            "SST_OPTIONS= V1994m\n"
            "FREESTREAM_TURBULENCEINTENSITY= 0.00038729\n"
            "FREESTREAM_TURB2LAMVISCRATIO= 0.009"
        )

    cfg = f"""\
% TMR Flat Plate — MRR Level 0 Verification
% M = {MACH}, Re = {RE:.0e}, T = {T_INF} K
SOLVER= RANS
{turb_block}
MATH_PROBLEM= DIRECT
RESTART_SOL= NO

MACH_NUMBER= {MACH}
AOA= 0.0
FREESTREAM_TEMPERATURE= {T_INF}
REYNOLDS_NUMBER= {RE}
REYNOLDS_LENGTH= {RE_LENGTH}

REF_ORIGIN_MOMENT_X= 0.25
REF_ORIGIN_MOMENT_Y= 0.0
REF_ORIGIN_MOMENT_Z= 0.0
REF_LENGTH= 1.0
REF_AREA= 2.0

MARKER_HEATFLUX= ( wall, 0.0 )
MARKER_FAR= ( farfield )
MARKER_INLET= ( inlet, 302.4, 117691.7874, 1.0, 0.0, 0.0 )
MARKER_OUTLET= ( outlet, 114455.0 )
MARKER_SYM= ( symmetry )
MARKER_PLOTTING= ( wall )
MARKER_MONITORING= ( wall )

NUM_METHOD_GRAD= WEIGHTED_LEAST_SQUARES
CFL_NUMBER= 400.0
CFL_ADAPT= NO
ITER= {n_iter}

LINEAR_SOLVER= FGMRES
LINEAR_SOLVER_PREC= ILU
LINEAR_SOLVER_ILU_FILL_IN= 0
LINEAR_SOLVER_ERROR= 1E-15
LINEAR_SOLVER_ITER= 25

CONV_NUM_METHOD_FLOW= ROE
MUSCL_FLOW= YES
SLOPE_LIMITER_FLOW= NONE
TIME_DISCRE_FLOW= EULER_IMPLICIT

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
SCREEN_OUTPUT= INNER_ITER WALL_TIME RMS_DENSITY RMS_NU_TILDE DRAG
OUTPUT_FILES= RESTART PARAVIEW SURFACE_PARAVIEW SURFACE_CSV
OUTPUT_WRT_FREQ= 10000
HISTORY_WRT_FREQ_INNER= 1
HISTORY_OUTPUT= ITER RMS_RES AERO_COEFF
"""
    path = case_dir / "flatplate.cfg"
    path.write_text(cfg)
    return path


def run_simulation(case_dir, cfg_path, timeout=7200):
    """Run SU2 on flat plate case."""
    su2 = shutil.which("SU2_CFD") or shutil.which("SU2_CFD.exe") or "SU2_CFD"
    cmd = [su2, cfg_path.name]
    log = case_dir / "su2_log.txt"

    print(f"  [RUN] {su2} {cfg_path.name}")
    t0 = time.time()
    with open(log, "w") as f:
        proc = subprocess.run(cmd, cwd=str(case_dir), stdout=f, stderr=subprocess.STDOUT, timeout=timeout)
    dt = time.time() - t0

    if proc.returncode != 0:
        tail = log.read_text(errors="replace").splitlines()[-10:]
        print(f"  [FAIL] code={proc.returncode}")
        for l in tail: print(f"    {l}")
        return False

    # Check convergence
    hist = case_dir / "history.csv"
    if hist.exists():
        n = sum(1 for _ in open(hist, errors="replace")) - 1
        last = hist.read_text(errors="replace").strip().split("\n")[-1]
        cols = last.split(",")
        rms = float(cols[3].strip().strip('"')) if len(cols) > 3 else 0
        print(f"  [OK] {n} iter, rms={rms:.2f}, time={dt:.0f}s")
    return True


def extract_uplus_yplus(vol_vtu_path, surf_vtu_path, x_station=0.97):
    """Extract U+ vs y+ profile at given x station.
    Uses surface Cf for tau_w (accurate) and volume mesh for velocity profile.
    """
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy

    # Get Cf at x_station from surface VTU
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(str(surf_vtu_path))
    reader.Update()
    sout = reader.GetOutput()
    s_coords = vtk_to_numpy(sout.GetPoints().GetData())
    s_pd = sout.GetPointData()
    s_names = [s_pd.GetArrayName(i) for i in range(s_pd.GetNumberOfArrays())]
    cf_arr = vtk_to_numpy(s_pd.GetArray(s_names.index("Skin_Friction_Coefficient")))
    cf_x = cf_arr[:, 0] if cf_arr.ndim > 1 else cf_arr
    xs = s_coords[:, 0]

    # Cf at station
    mask_97 = np.abs(xs - x_station) < 0.03
    if not mask_97.any():
        print(f"  WARNING: No surface data near x={x_station}")
        return None
    Cf_local = float(np.mean(cf_x[mask_97]))
    tau_w = 0.5 * RHO_INF * U_INF**2 * Cf_local
    u_tau = math.sqrt(abs(tau_w) / RHO_INF)

    print(f"    Cf(x={x_station}) = {Cf_local:.6f}")
    print(f"    tau_w = {tau_w:.4f} Pa, u_tau = {u_tau:.4f} m/s")

    # Read volume mesh and extract profile near x_station
    mesh = pv.read(str(vol_vtu_path))
    coords = mesh.points
    vel = mesh.point_data["Velocity"]

    # Select points in a narrow strip near x_station
    x_tol = 0.02
    mask = np.abs(coords[:, 0] - x_station) < x_tol
    y_sel = coords[mask, 1]
    u_sel = vel[mask, 0]

    # Sort by y, keep only y > 0 (above wall)
    idx = np.argsort(y_sel)
    y_sel = y_sel[idx]
    u_sel = u_sel[idx]

    valid = y_sel > 1e-8
    y_v = y_sel[valid]
    u_v = u_sel[valid]

    if len(y_v) == 0:
        return None

    # Compute y+ and U+
    y_plus = y_v * RHO_INF * u_tau / MU_INF
    u_plus = u_v / u_tau

    return {
        "y": y_v, "u": u_v,
        "y_plus": y_plus, "u_plus": u_plus,
        "u_tau": u_tau, "tau_w": tau_w,
        "Cf": Cf_local,
    }


def plot_uplus_yplus(profiles, out_dir):
    """Generate U⁺ vs y⁺ plot with log law comparison."""
    fig, ax = plt.subplots(figsize=(10, 7))

    # Law of the wall
    yp_visc = np.linspace(0.1, 11, 100)
    up_visc = yp_visc
    ax.plot(yp_visc, up_visc, "k-", lw=1.5, label="$U^+ = y^+$ (viscous sublayer)")

    yp_log = np.logspace(np.log10(30), 4, 200)
    kappa, B = 0.41, 5.0
    up_log = (1 / kappa) * np.log(yp_log) + B
    ax.plot(yp_log, up_log, "k--", lw=1.5, label=f"Log law ($\\kappa$={kappa}, B={B})")

    # Simulation profiles
    colors = {"SA_137x097": "#e53935", "SA_069x049": "#ef9a9a",
              "SST_137x097": "#1e88e5", "SST_069x049": "#90caf9"}
    for label, p in profiles.items():
        if p is None: continue
        ls = "-" if "137" in label else "--"
        ax.plot(p["y_plus"], p["u_plus"], ls, color=colors.get(label, "#333"),
                lw=2.0, label=f"SU2 {label} (Cf={p['Cf']:.5f})", zorder=4)
        print(f"  {label}: u_tau={p['u_tau']:.4f}, Cf={p['Cf']:.6f}")

    ax.set_xscale("log")
    ax.set_xlabel("$y^+$", fontsize=14)
    ax.set_ylabel("$U^+$", fontsize=14)
    ax.set_title("MRR Level 0 — TMR Flat Plate: $U^+$ vs $y^+$ at x = 0.97 m", fontsize=14)
    ax.set_xlim(0.1, 1e4)
    ax.set_ylim(0, 30)
    ax.legend(loc="upper left", fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, which="both")

    # Shade buffer layer
    ax.axvspan(5, 30, alpha=0.05, color="orange", label="_buffer")

    fname = out_dir / "flat_plate_uplus_yplus.png"
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: {fname}")
    return fname


def plot_cf_vs_x(profiles_cf, out_dir):
    """Plot Cf vs x along the flat plate."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Blasius analytical Cf for turbulent BL (Schlichting)
    x = np.linspace(0.01, 2.0, 500)
    Re_x = RHO_INF * U_INF * x / MU_INF
    Cf_blasius = 0.0592 * Re_x**(-1/5)  # Prandtl 1/7th power law
    Cf_schoenherr = 0.455 / (np.log10(Re_x))**2.58  # Schoenherr formula
    ax.plot(x, Cf_blasius, "k:", lw=1.2, alpha=0.6, label="Prandtl 1/7 law")
    ax.plot(x, Cf_schoenherr, "k--", lw=1.2, alpha=0.6, label="Schoenherr")

    for label, data in profiles_cf.items():
        if data is None: continue
        color = "#e53935" if "SA" in label else "#1e88e5"
        ls = "-" if "137" in label else "--"
        ax.plot(data["x"], data["cf"], ls, color=color, lw=1.8, label=f"SU2 {label}")

    ax.set_xlabel("x (m)", fontsize=13)
    ax.set_ylabel("$C_f$", fontsize=13)
    ax.set_title("TMR Flat Plate — Skin Friction Coefficient", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 2.0)

    fname = out_dir / "flat_plate_cf_vs_x.png"
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


def main():
    parser = argparse.ArgumentParser(description="TMR Flat Plate MRR Level 0")
    parser.add_argument("--model", nargs="+", default=["SA"], choices=["SA", "SST"])
    parser.add_argument("--grid", nargs="+", default=["137x097"], choices=list(GRIDS.keys()))
    parser.add_argument("--n-iter", type=int, default=99999)
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("  TMR FLAT PLATE — MRR Level 0 Verification")
    print("=" * 60)
    print(f"  M={MACH}, Re={RE:.0e}, T={T_INF}K, U_inf={U_INF:.2f} m/s")
    print(f"  rho_inf={RHO_INF:.4f}, mu_inf={MU_INF:.6e}")

    profiles = {}
    profiles_cf = {}

    for grid in args.grid:
        mesh_src = GRID_DIR / GRIDS[grid]
        if not mesh_src.exists():
            print(f"  [ERR] Grid not found: {mesh_src}"); continue

        for model in args.model:
            label = f"{model}_{grid}"
            case_dir = RUNS_DIR / f"flatplate_{label}"
            case_dir.mkdir(parents=True, exist_ok=True)

            # Copy mesh
            mesh_dst = case_dir / "mesh.su2"
            if not mesh_dst.exists():
                shutil.copy2(mesh_src, mesh_dst)

            if not args.plot_only:
                cfg = generate_config(case_dir, "mesh.su2", model, args.n_iter)
                su2 = shutil.which("SU2_CFD") or shutil.which("SU2_CFD.exe")
                if not su2:
                    print("  [ERR] SU2_CFD not found"); return
                ok = run_simulation(case_dir, cfg, timeout=args.timeout)
                if not ok: continue

            # Extract U+ vs y+
            vtu = case_dir / "flow.vtu"
            svtu = case_dir / "surface_flow.vtu"
            if not svtu.exists():
                svtu = case_dir / "surface.vtu"
            if not vtu.exists():
                print(f"  [SKIP] No VTU: {vtu}"); continue

            print(f"\n  Extracting U+/y+ for {label}...")
            p = extract_uplus_yplus(vtu, svtu, x_station=0.97)
            profiles[label] = p

            # Extract Cf vs x from surface data
            import vtk
            from vtk.util.numpy_support import vtk_to_numpy
            svtu = case_dir / "surface_flow.vtu"
            if not svtu.exists():
                svtu = case_dir / "surface.vtu"
            if svtu.exists():
                reader = vtk.vtkXMLUnstructuredGridReader()
                reader.SetFileName(str(svtu))
                reader.Update()
                out = reader.GetOutput()
                coords = vtk_to_numpy(out.GetPoints().GetData())
                pd = out.GetPointData()
                names = [pd.GetArrayName(i) for i in range(pd.GetNumberOfArrays())]
                if "Skin_Friction_Coefficient" in names:
                    cf_arr = vtk_to_numpy(pd.GetArray(names.index("Skin_Friction_Coefficient")))
                    cf_x = cf_arr[:, 0] if cf_arr.ndim > 1 else cf_arr
                    x_s = coords[:, 0]
                    idx = np.argsort(x_s)
                    profiles_cf[label] = {"x": x_s[idx], "cf": cf_x[idx]}

    if profiles:
        print("\n  Generating U+ vs y+ plot...")
        plot_uplus_yplus(profiles, OUT_DIR)

    if profiles_cf:
        print("  Generating Cf vs x plot...")
        plot_cf_vs_x(profiles_cf, OUT_DIR)

    print(f"\n  Output: {OUT_DIR}")
    print("  Done!")


if __name__ == "__main__":
    main()
