#!/usr/bin/env python3
"""
Wall Hump — Cp/Cf Validation (VTK reader).

Corrects the Cp offset due to FREESTREAM_PRESS_EQ_ONE non-dim by
subtracting the upstream reference Cp value (zeroing Cp at the inlet).
"""
import sys, json
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import vtk
from vtk.util.numpy_support import vtk_to_numpy

PROJECT = Path(__file__).parent.resolve()
EXP_DIR = PROJECT / "experimental_data" / "wall_hump" / "csv"
RUNS_DIR = PROJECT / "runs" / "wall_hump"
OUT_DIR = PROJECT / "plots" / "wall_hump"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def read_vtu(path):
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(str(path))
    reader.Update()
    out = reader.GetOutput()
    coords = vtk_to_numpy(out.GetPoints().GetData())
    result = {'coords': coords}
    pd = out.GetPointData()
    for i in range(pd.GetNumberOfArrays()):
        result[pd.GetArrayName(i)] = vtk_to_numpy(pd.GetArray(i))
    return result


def load_exp():
    exp = {}
    for name, key in [("exp_cp.csv", "cp"), ("exp_cf.csv", "cf")]:
        p = EXP_DIR / name
        if p.exists():
            df = pd.read_csv(p)
            df.columns = [c.strip() for c in df.columns]
            exp[key] = {'x': df.iloc[:, 0].values, 'val': df.iloc[:, 1].values}
            print(f"  Loaded {name}: {len(df)} pts")
    ref = EXP_DIR / "hump_case_reference.json"
    if ref.exists():
        with open(ref) as f:
            exp['reference'] = json.load(f)
    return exp


# ============================================================================
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "#333", "axes.grid": True,
    "grid.color": "#ddd", "grid.alpha": 0.7,
    "font.family": "sans-serif", "font.size": 12,
    "axes.titlesize": 14, "axes.labelsize": 12,
    "legend.fontsize": 10, "savefig.dpi": 200, "savefig.bbox": "tight",
})
COLORS = {
    'SA_coarse': '#ef9a9a', 'SA_medium': '#e53935', 'SA_fine': '#b71c1c',
    'SST_coarse': '#90caf9', 'SST_medium': '#1e88e5', 'SST_fine': '#0d47a1',
    'exp': '#2e7d32',
}
LINE_STYLE = {'coarse': '--', 'medium': '-', 'fine': ':'}


def main():
    print("=" * 60)
    print("  WALL HUMP — Cp/Cf VALIDATION")
    print("=" * 60)
    exp = load_exp()
    # Experimental references (Greenblatt et al. 2006)
    x_sep_exp = exp.get('reference', {}).get('separation_x', 0.665)
    x_reat_exp = exp.get('reference', {}).get('reattachment_x', 1.10)
    # TMR 7-code ensemble references (CFL3D/FUN3D SA, 817x217 grid)
    # RANS codes systematically overpredict reattachment by 15-20%
    x_sep_tmr = 0.661   # TMR SA ensemble
    x_reat_tmr = 1.26   # TMR SA ensemble (not exp 1.10!)
    # Use experimental for separation (well-predicted), TMR for reattachment context
    x_sep_ref = x_sep_exp
    x_reat_ref = x_reat_exp

    sim_data = []
    # Dynamic pressure for Cp normalization: q_inf = 0.5 * rho * U^2
    # From case config: M=0.1, T=300K, P=101325 Pa
    gamma = 1.4; R_gas = 287.058; T_inf = 300.0; P_inf = 101325.0; M = 0.1
    rho_inf = P_inf / (R_gas * T_inf)
    a_inf = (gamma * R_gas * T_inf) ** 0.5
    U_inf = M * a_inf
    q_inf = 0.5 * rho_inf * U_inf ** 2
    print(f"  q_inf = {q_inf:.2f} Pa (rho={rho_inf:.4f}, U={U_inf:.2f} m/s)")

    for model in ['SA', 'SST']:
      for grid in ['coarse', 'medium', 'fine']:
        vtu = RUNS_DIR / f"hump_{model}_{grid}" / "surface_flow.vtu"
        if not vtu.exists():
            continue
        label = f"{model}_{grid}"
        print(f"\n  Reading {label}...")
        d = read_vtu(vtu)
        x = d['coords'][:, 0]
        # Compute Cp from raw Pressure (not SU2's Pressure_Coefficient)
        P_surf = d.get('Pressure')
        Cf_vec = d.get('Skin_Friction_Coefficient')
        Cf = Cf_vec[:, 0] if Cf_vec is not None and Cf_vec.ndim > 1 else Cf_vec

        idx = np.argsort(x)
        x = x[idx]
        P_surf = P_surf[idx] if P_surf is not None else None
        Cf = Cf[idx] if Cf is not None else None

        # Compute Cp = (P - P_ref) / q_inf, with P_ref at x/c = -1.0 (TMR spec)
        if P_surf is not None:
            ref_mask = np.abs(x - (-1.0)) < 0.1
            if ref_mask.any():
                P_ref = np.mean(P_surf[ref_mask])
            else:
                i_ref = np.argmin(np.abs(x - (-1.0)))
                P_ref = P_surf[i_ref]
            Cp = (P_surf - P_ref) / q_inf
            print(f"  {label}: P_ref at x/c=-1.0: {P_ref:.2f} Pa")
            print(f"  {label}: Cp = (P - P_ref) / q_inf, range: [{Cp.min():.4f}, {Cp.max():.4f}]")
        else:
            Cp = np.zeros_like(x)
            print(f"  {label}: WARNING - No Pressure array found!")

        if Cf is not None:
            print(f"  {label}: Cf: [{Cf.min():.6f}, {Cf.max():.6f}]")

        sim_data.append({'x': x, 'Cp': Cp, 'Cf': Cf, 'model': label})

    if not sim_data:
        print("  No data!"); return

    # ---- Cp full ----
    fig, ax = plt.subplots(figsize=(12, 6))
    if 'cp' in exp:
        ax.plot(exp['cp']['x'], exp['cp']['val'], 'o',
                color=COLORS['exp'], ms=4, alpha=0.7,
                label='Experiment (Greenblatt 2004)', zorder=3)
    for s in sim_data:
        parts = s['model'].split('_')
        grid_key = parts[1] if len(parts) > 1 else 'medium'
        ls = LINE_STYLE.get(grid_key, '-')
        ax.plot(s['x'], s['Cp'], ls, color=COLORS.get(s['model'], '#333'),
                lw=1.8, label=f"SU2 {s['model']}", zorder=4)
    ax.axvline(x_sep_exp, color='#d32f2f', ls=':', lw=1.2, alpha=0.6, label=f'Exp sep x={x_sep_exp}')
    ax.axvline(x_reat_exp, color='#1565c0', ls=':', lw=1.2, alpha=0.6, label=f'Exp reat x={x_reat_exp}')
    ax.axvline(x_reat_tmr, color='#1565c0', ls='--', lw=1.2, alpha=0.4, label=f'TMR SA reat x={x_reat_tmr}')
    ax.set_xlabel('x/c'); ax.set_ylabel('$C_p$')
    ax.set_title('Wall-Mounted Hump — Surface Pressure Coefficient', fontweight='bold')
    ax.invert_yaxis(); ax.set_xlim(-1.0, 2.5); ax.legend(loc='lower right')
    fig.tight_layout(); fig.savefig(OUT_DIR / "wall_hump_cp_validation.png"); plt.close()
    print(f"\n  Saved: wall_hump_cp_validation.png")

    # ---- Cf ----
    fig, ax = plt.subplots(figsize=(12, 6))
    if 'cf' in exp:
        ax.plot(exp['cf']['x'], exp['cf']['val'], 'o',
                color=COLORS['exp'], ms=4, alpha=0.7, label='Experiment', zorder=3)
    for s in sim_data:
        if s['Cf'] is not None:
            ax.plot(s['x'], s['Cf'], '-', color=COLORS.get(s['model'], '#333'),
                    lw=1.8, label=f"SU2 {s['model']}", zorder=4)
    ax.axhline(0, color='#999', ls='-', lw=0.5)
    ax.axvline(x_sep_exp, color='#d32f2f', ls=':', lw=1.2, label=f'Exp sep x={x_sep_exp}')
    ax.axvline(x_reat_exp, color='#1565c0', ls=':', lw=1.2, label=f'Exp reat x={x_reat_exp}')
    ax.axvline(x_reat_tmr, color='#1565c0', ls='--', lw=1.2, alpha=0.5, label=f'TMR SA reat x={x_reat_tmr}')
    ax.set_xlabel('x/c'); ax.set_ylabel('$C_f$')
    ax.set_title('Wall-Mounted Hump — Skin Friction Coefficient', fontweight='bold')
    ax.set_xlim(-0.5, 2.0); ax.legend(loc='best')
    fig.tight_layout(); fig.savefig(OUT_DIR / "wall_hump_cf_validation.png"); plt.close()
    print(f"  Saved: wall_hump_cf_validation.png")

    # ---- Separation detail ----
    fig, ax = plt.subplots(figsize=(10, 6))
    if 'cp' in exp:
        m = (exp['cp']['x'] >= 0.3) & (exp['cp']['x'] <= 1.5)
        ax.plot(exp['cp']['x'][m], exp['cp']['val'][m], 'o',
                color=COLORS['exp'], ms=5, alpha=0.8, label='Experiment', zorder=3)
    for s in sim_data:
        m = (s['x'] >= 0.3) & (s['x'] <= 1.5)
        ax.plot(s['x'][m], s['Cp'][m], '-', color=COLORS.get(s['model'], '#333'),
                lw=2.0, label=f"SU2 {s['model']}", zorder=4)
    ax.axvline(x_sep_ref, color='#d32f2f', ls=':', lw=1.5)
    ax.axvline(x_reat_ref, color='#1565c0', ls=':', lw=1.5)
    ax.set_xlabel('x/c'); ax.set_ylabel('$C_p$')
    ax.set_title('Separation Region Detail', fontweight='bold')
    ax.invert_yaxis(); ax.legend()
    fig.tight_layout(); fig.savefig(OUT_DIR / "wall_hump_separation_detail.png"); plt.close()
    print(f"  Saved: wall_hump_separation_detail.png")

    # ---- Metrics ----
    print("\n  " + "=" * 50)
    print("  VALIDATION METRICS")
    print("  " + "=" * 50)
    for s in sim_data:
        model = s['model']
        if 'cp' in exp:
            m = (exp['cp']['x'] >= -0.5) & (exp['cp']['x'] <= 1.5)
            ex, ev = exp['cp']['x'][m], exp['cp']['val'][m]
            sv = np.interp(ex, s['x'], s['Cp'])
            print(f"\n  {model} Cp (hump region):")
            print(f"    RMSE = {np.sqrt(np.mean((sv-ev)**2)):.4f}")
            print(f"    MAE  = {np.mean(np.abs(sv-ev)):.4f}")
        if s['Cf'] is not None:
            hm = (s['x'] >= 0.5) & (s['x'] <= 1.5)
            xh, cfh = s['x'][hm], s['Cf'][hm]
            sc = np.where(np.diff(np.sign(cfh)))[0]
            if len(sc) >= 1:
                i0 = sc[0]
                x_s = xh[i0] - cfh[i0]*(xh[i0+1]-xh[i0])/(cfh[i0+1]-cfh[i0])
                print(f"  {model} Separation:   x/c = {x_s:.3f}  (exp: {x_sep_exp}, TMR: {x_sep_tmr})")
            if len(sc) >= 2:
                i1 = sc[-1]
                x_r = xh[i1] - cfh[i1]*(xh[i1+1]-xh[i1])/(cfh[i1+1]-cfh[i1])
                print(f"  {model} Reattachment: x/c = {x_r:.3f}  (exp: {x_reat_exp}, TMR: {x_reat_tmr})")
                bl = x_r - x_s
                bl_exp = x_reat_exp - x_sep_exp
                bl_tmr = x_reat_tmr - x_sep_tmr
                print(f"  {model} Bubble length: {bl:.3f}")
                print(f"    vs experiment: {bl_exp:.3f} (err: {abs(bl-bl_exp)/bl_exp*100:.1f}%)")
                print(f"    vs TMR SA:     {bl_tmr:.3f} (err: {abs(bl-bl_tmr)/bl_tmr*100:.1f}%)")
        if 'cf' in exp and s['Cf'] is not None:
            m = (exp['cf']['x'] >= -0.5) & (exp['cf']['x'] <= 1.5)
            ex, ev = exp['cf']['x'][m], exp['cf']['val'][m]
            sv = np.interp(ex, s['x'], s['Cf'])
            print(f"  {model} Cf RMSE = {np.sqrt(np.mean((sv-ev)**2)):.6f}")

    print(f"\n  Output: {OUT_DIR}")
    print("  Done!")


if __name__ == "__main__":
    main()
