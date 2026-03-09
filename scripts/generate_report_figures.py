"""
Generate all figures for the technical report.

Creates publication-quality plots from actual SU2 output and experimental data.

Figures produced:
  1. fig_naca0012_cp.pdf         — Cp distributions at alpha=0,10,15 (SU2 vs CFL3D vs expt)
  2. fig_naca0012_grid_conv.pdf  — Cp grid convergence at alpha=0 (medium/fine/xfine)
  3. fig_naca0012_forces.pdf     — CL(alpha) and CD(alpha) polars (SU2 vs TMR 7-code scatter vs Ladson)
  4. fig_naca0012_cf.pdf         — Cf distributions at alpha=0,10,15 (SU2 vs CFL3D)
  5. fig_naca0012_convergence.pdf— Convergence histories (residual + CD)
  6. fig_naca0012_gci.pdf        — GCI bar chart for CD and CL at alpha=0
  7. fig_wall_hump_cp.pdf        — Hump Cp (experiment only; SU2 unconverged)
  8. fig_wall_hump_cf.pdf        — Hump Cf (experiment only; SU2 unconverged)

NOTE ON WALL HUMP DATA:
  The wall hump SU2 simulations ran for only 2 iterations (history.csv).
  surface_flow.csv contains initial-condition data (all-zero momentum).
  No converged SU2 wall hump results exist in this study.

All figures saved to docs/technical_report/figures/
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# -- Paths ---------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
EXPDATA = ROOT / "experimental_data"
FIGDIR = ROOT / "docs" / "technical_report" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

# -- Plot styling --------------------------------------------------------
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 13,
    'legend.fontsize': 9,
    'legend.framealpha': 0.9,
    'lines.linewidth': 1.5,
    'lines.markersize': 4,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

# -- Helper: compute Cp from SU2 conservative-variable CSV ---------------
def su2_csv_to_cp(csv_path, M_inf, gamma=1.4):
    """
    Read SU2 surface_flow.csv (conservative variables) and compute Cp.

    With FREESTREAM_PRESS_EQ_ONE: p_inf = 1.0
    p = (gamma-1) * (E - 0.5 * (Mx^2 + My^2) / rho)
    q_inf = 0.5 * gamma * p_inf * M^2
    Cp = (p - p_inf) / q_inf
    """
    df = pd.read_csv(csv_path)
    rho = df['Density'].values
    Mx  = df['Momentum_x'].values
    My  = df['Momentum_y'].values
    E   = df['Energy'].values
    x   = df['x'].values
    y   = df['y'].values

    p_inf = 1.0
    q_inf = 0.5 * gamma * p_inf * M_inf**2
    p = (gamma - 1.0) * (E - 0.5 * (Mx**2 + My**2) / rho)
    Cp = (p - p_inf) / q_inf

    return x, y, Cp


def read_history(csv_path):
    """Read SU2 convergence history CSV, return cleaned DataFrame."""
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().strip('"') for c in df.columns]
    return df


def get_rms_rho_col(df):
    """Find the rms[Rho] column name in history DataFrame."""
    candidates = [c for c in df.columns
                  if 'rms' in c.lower() and 'Rho' in c
                  and 'RhoU' not in c and 'RhoV' not in c and 'RhoE' not in c]
    return candidates[0] if candidates else None


def savefig(fig, name):
    """Save figure in both PDF and PNG formats."""
    out = FIGDIR / name
    fig.savefig(out)
    fig.savefig(out.with_suffix('.png'))
    print(f"  Saved: {out}")
    plt.close(fig)


# =====================================================================
# FIGURE 1: NACA 0012 Cp distributions at alpha = 0, 10, 15
#            SU2 (fine grid) vs CFL3D SA reference vs experiments
# =====================================================================
def fig_naca_cp():
    print("Generating NACA 0012 Cp distributions...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    alphas = [0.0, 10.0, 15.0]
    for ax, alpha in zip(axes, alphas):
        a_str = f"{alpha:.1f}"
        a_int = int(alpha)

        # -- SU2 data (fine grid) --
        su2_path = RUNS / "naca0012" / f"alpha_{a_str}_SA_fine" / "surface_flow.csv"
        if su2_path.exists():
            x_su2, y_su2, Cp_su2 = su2_csv_to_cp(su2_path, M_inf=0.15)
            upper = y_su2 >= 0
            lower = y_su2 < 0
            idx_u = np.argsort(x_su2[upper])
            idx_l = np.argsort(x_su2[lower])
            ax.plot(x_su2[upper][idx_u], Cp_su2[upper][idx_u],
                    'b-', linewidth=2, label='SU2 SA (fine)', zorder=3)
            ax.plot(x_su2[lower][idx_l], Cp_su2[lower][idx_l],
                    'b-', linewidth=2, zorder=3)

        # -- CFL3D reference --
        cfl3d_file = EXPDATA / "naca0012" / "csv" / f"cfl3d_sa_cp_alpha{a_int}.csv"
        if cfl3d_file.exists():
            cfl3d = pd.read_csv(cfl3d_file)
            ax.plot(cfl3d['x'], cfl3d['cp'], 'r-', linewidth=1.2,
                    label='CFL3D SA (ref)', alpha=0.8)

        # -- Gregory experiment --
        greg_file = EXPDATA / "naca0012" / "csv" / f"gregory_cp_alpha{a_int}.csv"
        if greg_file.exists():
            greg = pd.read_csv(greg_file)
            xcol = [c for c in greg.columns if 'x' in c.lower()][0]
            cpcol = [c for c in greg.columns if 'cp' in c.lower()][0]
            ax.scatter(greg[xcol], greg[cpcol], s=20, marker='^',
                       c='green', alpha=0.7, label='Gregory (Re=2.88M)', zorder=2)

        # -- Ladson experiment --
        ladson_candidates = list(
            (EXPDATA / "naca0012" / "csv").glob(f"ladson_cp_re6*alpha*free*.csv"))
        if a_int == 0:
            ladson_alpha_candidates = [
                f for f in ladson_candidates if "alpha.0" in f.name]
        else:
            ladson_alpha_candidates = [
                f for f in ladson_candidates if f"alpha{a_int}." in f.name]

        if ladson_alpha_candidates:
            ladson = pd.read_csv(ladson_alpha_candidates[0])
            xcol = [c for c in ladson.columns if 'x' in c.lower()][0]
            cpcol = [c for c in ladson.columns if 'cp' in c.lower()][0]
            ax.scatter(ladson[xcol], ladson[cpcol], s=15, marker='o',
                       c='orange', alpha=0.6, label='Ladson (Re=6M, free)', zorder=2)

        ax.set_xlabel('$x/c$')
        ax.set_ylabel('$C_p$')
        ax.set_title(f'$\\alpha = {a_int}\\degree$')
        ax.invert_yaxis()
        ax.legend(loc='lower right', fontsize=8)
        ax.set_xlim(-0.05, 1.05)

    fig.suptitle('NACA 0012 --- $C_p$ Distribution: SU2 vs CFL3D vs Experiment',
                 fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    savefig(fig, "fig_naca0012_cp.pdf")


# =====================================================================
# FIGURE 2: NACA 0012 grid convergence (Cp at alpha=0)
# =====================================================================
def fig_naca_grid_convergence():
    print("Generating NACA 0012 grid convergence...")
    fig, ax = plt.subplots(figsize=(8, 5))

    grids = [
        ('medium', 'alpha_0.0_SA_medium', '--', 0.6),
        ('fine',   'alpha_0.0_SA_fine',   '-',  0.8),
        ('xfine',  'alpha_0.0_SA_xfine',  '-',  1.0),
    ]
    colors = ['#2196F3', '#1565C0', '#0D47A1']

    for (label, folder, ls, alpha), color in zip(grids, colors):
        csv_path = RUNS / "naca0012" / folder / "surface_flow.csv"
        if csv_path.exists():
            x, y_coord, Cp = su2_csv_to_cp(csv_path, M_inf=0.15)
            upper = y_coord >= 0
            idx = np.argsort(x[upper])
            ax.plot(x[upper][idx], Cp[upper][idx],
                    linestyle=ls, color=color, alpha=alpha,
                    linewidth=1.5, label=f'SU2 SA ({label})')

    # CFL3D reference
    cfl3d_file = EXPDATA / "naca0012" / "csv" / "cfl3d_sa_cp_alpha0.csv"
    if cfl3d_file.exists():
        cfl3d = pd.read_csv(cfl3d_file)
        ax.plot(cfl3d['x'], cfl3d['cp'], 'r-', linewidth=1.2,
                label='CFL3D SA (ref)', alpha=0.7)

    ax.set_xlabel('$x/c$')
    ax.set_ylabel('$C_p$')
    ax.set_title('NACA 0012 Grid Convergence ($\\alpha=0\\degree$, upper surface)',
                 fontweight='bold')
    ax.invert_yaxis()
    ax.legend()
    ax.set_xlim(-0.05, 1.05)

    fig.tight_layout()
    savefig(fig, "fig_naca0012_grid_conv.pdf")


# =====================================================================
# FIGURE 3: CL(alpha) and CD(alpha) polars
#   SU2 vs TMR 7-code scatter vs Ladson experiment
# =====================================================================
def fig_naca_forces():
    print("Generating NACA 0012 force polars (CL and CD vs alpha)...")

    # --- Load TMR reference data (per-code scatter) ---
    tmr_file = EXPDATA / "naca0012" / "csv" / "tmr_sa_reference.json"
    with open(tmr_file) as f:
        tmr = json.load(f)

    alphas_tmr = [0, 10, 15]
    tmr_consensus = {
        'CL': [tmr[f'alpha_{a}']['CL'] for a in alphas_tmr],
        'CD': [tmr[f'alpha_{a}']['CD'] for a in alphas_tmr],
    }

    # Per-code scatter
    code_names = list(tmr['per_code_results'].keys())
    per_code = {}
    for code in code_names:
        cdata = tmr['per_code_results'][code]
        per_code[code] = {
            'CL': [cdata['CL_0'], cdata['CL_10'], cdata['CL_15']],
            'CD': [cdata['CD_0'], cdata['CD_10'], cdata['CD_15']],
        }

    # --- Load SU2 data (best available per angle) ---
    # Use xfine where available, fine otherwise
    su2_best = {}
    su2_cases = [
        (0.0,  'alpha_0.0_SA_xfine'),
        (10.0, 'alpha_10.0_SA_xfine'),
        (15.0, 'alpha_15.0_SA_xfine'),
    ]
    for alpha, folder in su2_cases:
        h = RUNS / "naca0012" / folder / "history.csv"
        if h.exists():
            df = read_history(h)
            su2_best[alpha] = {'CL': df['CL'].iloc[-1], 'CD': df['CD'].iloc[-1]}

    su2_alphas = sorted(su2_best.keys())
    su2_cl = [su2_best[a]['CL'] for a in su2_alphas]
    su2_cd = [su2_best[a]['CD'] for a in su2_alphas]

    # --- Load Ladson experimental data ---
    ladson_file = EXPDATA / "naca0012" / "csv" / "ladson_forces.csv"
    ladson = pd.read_csv(ladson_file)
    # Filter to pre-stall (alpha < 18)
    ladson = ladson[ladson['alpha'] < 18]

    # --- Load Gregory experimental CL ---
    greg_cl_file = EXPDATA / "naca0012" / "csv" / "gregory_cl.csv"
    greg_cl = pd.read_csv(greg_cl_file) if greg_cl_file.exists() else None

    # --- Plot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # === CL(alpha) ===
    # TMR per-code scatter (thin gray lines)
    for i, code in enumerate(code_names):
        ax1.plot(alphas_tmr, per_code[code]['CL'], 's',
                 color='gray', markersize=4, alpha=0.4,
                 label='TMR codes' if i == 0 else None, zorder=1)

    # TMR consensus
    ax1.plot(alphas_tmr, tmr_consensus['CL'], 'r^-', markersize=8,
             linewidth=1.5, label='TMR consensus (7 codes)', zorder=3)

    # Ladson experiment
    ax1.scatter(ladson['alpha'], ladson['CL'], s=15, c='orange',
                alpha=0.5, marker='o', label='Ladson (Re=6M)', zorder=2)

    # Gregory experiment
    if greg_cl is not None:
        ax1.scatter(greg_cl['alpha'], greg_cl['CL'], s=25, c='green',
                    marker='^', alpha=0.6, label='Gregory (Re=2.88M)', zorder=2)

    # SU2
    ax1.plot(su2_alphas, su2_cl, 'bD-', markersize=8, linewidth=2,
             label='SU2 SA (best grid)', zorder=5)

    ax1.set_xlabel(r'$\alpha$ (degrees)')
    ax1.set_ylabel('$C_L$')
    ax1.set_title('Lift Coefficient', fontweight='bold')
    ax1.legend(fontsize=8, loc='upper left')
    ax1.set_xlim(-1, 18)
    ax1.set_ylim(-0.1, 1.8)

    # === CD(alpha) ===
    # TMR per-code scatter
    for i, code in enumerate(code_names):
        ax2.plot(alphas_tmr, per_code[code]['CD'], 's',
                 color='gray', markersize=4, alpha=0.4,
                 label='TMR codes' if i == 0 else None, zorder=1)

    # TMR consensus
    ax2.plot(alphas_tmr, tmr_consensus['CD'], 'r^-', markersize=8,
             linewidth=1.5, label='TMR consensus (7 codes)', zorder=3)

    # Ladson experiment
    ax2.scatter(ladson['alpha'], ladson['CD'], s=15, c='orange',
                alpha=0.5, marker='o', label='Ladson (Re=6M)', zorder=2)

    # SU2
    ax2.plot(su2_alphas, su2_cd, 'bD-', markersize=8, linewidth=2,
             label='SU2 SA (best grid)', zorder=5)

    ax2.set_xlabel(r'$\alpha$ (degrees)')
    ax2.set_ylabel('$C_D$')
    ax2.set_title('Drag Coefficient', fontweight='bold')
    ax2.legend(fontsize=8, loc='upper left')
    ax2.set_xlim(-1, 18)
    ax2.set_ylim(-0.001, 0.035)

    fig.suptitle('NACA 0012 --- Force Coefficients: SU2 vs TMR 7-Code Ensemble vs Experiment',
                 fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    savefig(fig, "fig_naca0012_forces.pdf")


# =====================================================================
# FIGURE 4: NACA 0012 Cf distributions (upper surface)
#   SU2 fine vs CFL3D reference at alpha = 0, 10, 15
# =====================================================================
def fig_naca_cf():
    print("Generating NACA 0012 Cf distributions...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    alphas = [0.0, 10.0, 15.0]
    for ax, alpha in zip(axes, alphas):
        a_str = f"{alpha:.1f}"
        a_int = int(alpha)

        # -- SU2 data (fine grid) --
        # Cf from conservative variables: need wall shear stress
        # We don't have Cf directly, but Cp * pressure-based approximation
        # is unreliable. Instead, show CFL3D reference + note SU2 limitation.
        # Actually, we CAN approximate Cf from the surface_flow.csv by
        # noting that for attached flow, Cf ~ du/dy at wall.
        # But this requires flow interior data. We'll show CFL3D only for
        # direct comparison and note the SU2 limitation.

        # -- CFL3D Cf reference (upper surface) --
        cfl3d_cf = EXPDATA / "naca0012" / "csv" / f"cfl3d_sa_cf_alpha{a_int}_upper_surface.csv"
        if cfl3d_cf.exists():
            cfl3d = pd.read_csv(cfl3d_cf)
            ax.plot(cfl3d['x'], cfl3d['cf'], 'r-', linewidth=1.5,
                    label='CFL3D SA (897×257)', alpha=0.9, zorder=3)

        # Compute SU2 Cf from Cp gradient (rough approximation)
        # Actually better: plot that SU2 cannot provide Cf from surface_flow.csv
        # because it only has conservative variables, not wall shear stress.
        # We note this limitation.

        ax.axhline(0, color='black', linewidth=0.5, alpha=0.3)
        ax.set_xlabel('$x/c$')
        ax.set_ylabel('$C_f$')
        ax.set_title(f'$\\alpha = {a_int}\\degree$')
        ax.legend(loc='upper right', fontsize=8)
        ax.set_xlim(-0.01, 1.01)

    # Annotate trailing-edge separation on alpha=15 panel
    if len(axes) >= 3:
        axes[2].annotate('Trailing-edge\nseparation',
                         xy=(0.92, 0.0), xytext=(0.6, 0.01),
                         fontsize=8, color='red',
                         arrowprops=dict(arrowstyle='->', color='red', lw=1.2),
                         ha='center')

    fig.suptitle('NACA 0012 --- Upper-Surface $C_f$ (CFL3D SA Reference)\n'
                 'SU2 surface_flow.csv does not contain wall-shear data',
                 fontsize=13, fontweight='bold', y=1.04)
    fig.tight_layout()
    savefig(fig, "fig_naca0012_cf.pdf")


# =====================================================================
# FIGURE 5: NACA 0012 convergence histories (residual + CD)
# =====================================================================
def fig_convergence_histories():
    print("Generating NACA 0012 convergence histories...")

    cases = [
        (r'$\alpha=0\degree$ (fine)',   'alpha_0.0_SA_fine',   '#90CAF9'),
        (r'$\alpha=0\degree$ (xfine)',  'alpha_0.0_SA_xfine',  '#1565C0'),
        (r'$\alpha=10\degree$ (fine)',  'alpha_10.0_SA_fine',  '#4CAF50'),
        (r'$\alpha=10\degree$ (xfine)', 'alpha_10.0_SA_xfine', '#2E7D32'),
        (r'$\alpha=15\degree$ (fine)',  'alpha_15.0_SA_fine',  '#EF5350'),
        (r'$\alpha=15\degree$ (xfine)', 'alpha_15.0_SA_xfine', '#B71C1C'),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for label, folder, color in cases:
        hist_path = RUNS / "naca0012" / folder / "history.csv"
        if not hist_path.exists():
            continue
        df = read_history(hist_path)
        rms_col = get_rms_rho_col(df)
        if rms_col is None:
            continue
        iters = np.arange(len(df))

        # Residual history
        ax1.plot(iters, df[rms_col].values, color=color, linewidth=1.0,
                 label=label, alpha=0.85)

        # CD history
        if 'CD' in df.columns:
            ax2.plot(iters, df['CD'].values, color=color, linewidth=1.0,
                     label=label, alpha=0.85)

    # Residual plot formatting
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('$\\log_{10}$ (RMS density residual)')
    ax1.set_title('Residual Convergence', fontweight='bold')
    ax1.axhline(-14, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
    ax1.text(500, -13.5, 'Target ($10^{-14}$)', fontsize=8, color='gray')
    ax1.legend(fontsize=7, ncol=2)
    ax1.set_ylim([-16, 0])

    # CD plot formatting
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('$C_D$')
    ax2.set_title('$C_D$ Convergence', fontweight='bold')
    ax2.legend(fontsize=7, ncol=2)

    fig.suptitle('NACA 0012 --- Convergence Histories', fontsize=14,
                 fontweight='bold', y=1.02)
    fig.tight_layout()
    savefig(fig, "fig_naca0012_convergence.pdf")


# =====================================================================
# FIGURE 6: GCI analysis — bar chart for CD (and CL) at alpha=0
# =====================================================================
def fig_gci():
    print("Generating GCI analysis chart...")

    # Gather CD and CL across three grids at alpha=0
    grids = {
        'medium': {'h': 1/449, 'CD': None, 'CL': None},
        'fine':   {'h': 1/897, 'CD': None, 'CL': None},
        'xfine':  {'h': 1/1793, 'CD': None, 'CL': None},
    }
    for g in grids:
        h = RUNS / "naca0012" / f"alpha_0.0_SA_{g}" / "history.csv"
        if h.exists():
            df = read_history(h)
            grids[g]['CD'] = df['CD'].iloc[-1]
            grids[g]['CL'] = df['CL'].iloc[-1]

    # TMR reference
    tmr_file = EXPDATA / "naca0012" / "csv" / "tmr_sa_reference.json"
    with open(tmr_file) as f:
        tmr = json.load(f)
    cd_ref = tmr['alpha_0']['CD']
    cl_ref = tmr['alpha_0']['CL']

    # Compute GCI (Roache method)
    h1 = grids['xfine']['h']
    h2 = grids['fine']['h']
    h3 = grids['medium']['h']
    r21 = h2 / h1
    r32 = h3 / h2

    cd1 = grids['xfine']['CD']
    cd2 = grids['fine']['CD']
    cd3 = grids['medium']['CD']

    # Observed order
    e32 = cd3 - cd2
    e21 = cd2 - cd1
    if e32 != 0 and e21 != 0:
        p_cd = abs(np.log(abs(e32/e21)) / np.log(r21))
    else:
        p_cd = 2.0

    gci_fine_cd = 1.25 * abs(e21 / cd1) / (r21**p_cd - 1) * 100  # percent

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- Left panel: CD across grids ---
    ax = axes[0]
    grid_labels = ['Medium\n(449×129)', 'Fine\n(897×257)', 'X-Fine\n(1793×513)']
    cd_vals = [grids['medium']['CD'], grids['fine']['CD'], grids['xfine']['CD']]
    h_vals = [grids['medium']['h'], grids['fine']['h'], grids['xfine']['h']]

    bars = ax.bar(grid_labels, [c*1e4 for c in cd_vals], color=['#90CAF9', '#42A5F5', '#1565C0'], 
                  edgecolor='white', linewidth=1.5, width=0.6)
    ax.axhline(cd_ref*1e4, color='red', linestyle='--', linewidth=1.5,
               label=f'TMR reference ({cd_ref*1e4:.2f})')
    
    # GCI band on finest grid
    gci_abs = gci_fine_cd / 100 * cd1
    ax.fill_between([1.7, 2.3], (cd1 - gci_abs)*1e4, (cd1 + gci_abs)*1e4,
                    alpha=0.2, color='blue', label=f'GCI = {gci_fine_cd:.1f}%')
    
    # Annotate bars
    for bar, val in zip(bars, cd_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val*1e4:.2f}', ha='center', fontsize=9, fontweight='bold')
    
    ax.set_ylabel('$C_D \\times 10^4$')
    ax.set_title(f'$C_D$ Grid Convergence ($\\alpha=0\\degree$)\nObserved order $p={p_cd:.2f}$',
                 fontweight='bold')
    ax.legend(fontsize=8)
    ax.set_ylim([70, 85])

    # --- Right panel: Error vs h^p (Richardson extrapolation) ---
    ax2 = axes[1]
    h_arr = np.array([grids[g]['h'] for g in ['xfine', 'fine', 'medium']])
    cd_arr = np.array([grids[g]['CD'] for g in ['xfine', 'fine', 'medium']])
    
    # Richardson extrapolated value
    cd_extrap = cd1 + (cd2 - cd1) / (r21**p_cd - 1)
    errors = abs(cd_arr - cd_extrap) / cd_extrap * 100

    ax2.loglog(h_arr, errors, 'bD-', markersize=8, linewidth=2,
               label='Observed error', zorder=4)
    
    # Reference slopes
    h_ref = np.linspace(h_arr.min()*0.8, h_arr.max()*1.2, 50)
    scale1 = errors[-1] / h_arr[-1]**1
    scale2 = errors[-1] / h_arr[-1]**2
    ax2.loglog(h_ref, scale1 * h_ref**1, 'k:', alpha=0.4, label='$p=1$ slope')
    ax2.loglog(h_ref, scale2 * h_ref**2, 'k--', alpha=0.4, label='$p=2$ slope')
    
    ax2.set_xlabel('Characteristic grid spacing $h$')
    ax2.set_ylabel('$|C_D - C_{D,extrap}|$ / $C_{D,extrap}$ (%)')
    ax2.set_title('Richardson Extrapolation\n(asymptotic-range caveat applies)',
                  fontweight='bold')
    ax2.legend(fontsize=8)

    fig.suptitle('NACA 0012 --- Grid Convergence Index Analysis',
                 fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    savefig(fig, "fig_naca0012_gci.pdf")


# =====================================================================
# FIGURE 7: Wall hump Cp — experiment only
# =====================================================================
def fig_hump_cp():
    print("Generating wall hump Cp (experiment only)...")
    fig, ax = plt.subplots(figsize=(10, 5))

    exp_cp = EXPDATA / "wall_hump" / "csv" / "exp_cp.csv"
    if exp_cp.exists():
        exp = pd.read_csv(exp_cp)
        xcol = exp.columns[0]
        cpcol = exp.columns[1]
        ax.scatter(exp[xcol], exp[cpcol], s=25, marker='o',
                   c='#2E7D32', zorder=4, label='Experiment (Greenblatt 2006)')

    # Reference separation/reattachment
    ax.axvline(0.665, color='red', linestyle=':', alpha=0.5, linewidth=1,
               label='$x_{sep}$ = 0.665 (expt)')
    ax.axvline(1.10,  color='blue', linestyle=':', alpha=0.5, linewidth=1,
               label='$x_{reat}$ = 1.10 (expt)')
    ax.axvspan(0.665, 1.10, alpha=0.06, color='red', zorder=0)

    ax.set_xlabel('$x/c$')
    ax.set_ylabel('$C_p$')
    ax.set_title('Wall-Mounted Hump --- Experimental $C_p$ Reference\n'
                 '(No converged SU2 data available; simulations ran for 2 iterations)',
                 fontweight='bold', fontsize=11)
    ax.invert_yaxis()
    ax.legend(loc='lower right')
    ax.set_xlim(-1.0, 2.5)

    fig.tight_layout()
    savefig(fig, "fig_wall_hump_cp.pdf")


# =====================================================================
# FIGURE 8: Wall hump Cf — experiment only
# =====================================================================
def fig_hump_cf():
    print("Generating wall hump Cf (experiment only)...")
    fig, ax = plt.subplots(figsize=(10, 5))

    exp_cf = EXPDATA / "wall_hump" / "csv" / "exp_cf.csv"
    if exp_cf.exists():
        exp = pd.read_csv(exp_cf)
        xcol = exp.columns[0]
        cfcol = exp.columns[1]
        ax.scatter(exp[xcol], exp[cfcol], s=25, marker='o',
                   c='#2E7D32', zorder=4, label='Experiment (Greenblatt 2006)')

    ax.axvline(0.665, color='red', linestyle=':', alpha=0.5, linewidth=1,
               label='$x_{sep}$ = 0.665')
    ax.axvline(1.10,  color='blue', linestyle=':', alpha=0.5, linewidth=1,
               label='$x_{reat}$ = 1.10')
    ax.axhline(0, color='black', linewidth=0.5, alpha=0.3)
    ax.axvspan(0.665, 1.10, alpha=0.06, color='red', zorder=0)

    ax.set_xlabel('$x/c$')
    ax.set_ylabel('$C_f$')
    ax.set_title('Wall-Mounted Hump --- Experimental $C_f$ Reference\n'
                 '(No converged SU2 data available)',
                 fontweight='bold', fontsize=11)
    ax.legend(loc='upper right')

    fig.tight_layout()
    savefig(fig, "fig_wall_hump_cf.pdf")


# =====================================================================
# MAIN
# =====================================================================
if __name__ == "__main__":
    print(f"Output directory: {FIGDIR}\n")
    fig_naca_cp()
    fig_naca_grid_convergence()
    fig_naca_forces()
    fig_naca_cf()
    fig_convergence_histories()
    fig_gci()
    fig_hump_cp()
    fig_hump_cf()
    print("\nAll figures generated.")
