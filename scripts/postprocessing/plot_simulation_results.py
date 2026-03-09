#!/usr/bin/env python3
"""
NACA 0012 SU2 Simulation Results — Comprehensive Plots
=======================================================
Plots all aerodynamic parameters from the actual SU2 simulation runs.
White background, publication-quality figures.

Plots generated:
  1. Convergence histories (residuals + force monitors)
  2. CL, CD, CM vs α — simulation vs TMR reference
  3. Aerodynamic efficiency (L/D, CEff)
  4. Grid convergence study
  5. Surface Cp distribution (computed from conservative variables)
  6. Simulation accuracy summary
"""

import sys
import json
import csv
import math
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── Paths ───────────────────────────────────────────────────────────────────
PROJECT = Path(__file__).parent.parent.parent.resolve()
RUNS    = PROJECT / "runs" / "naca0012"
OUT     = PROJECT / "plots"
OUT.mkdir(exist_ok=True)

# ── Publication-quality white theme ─────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.edgecolor":    "#333333",
    "axes.labelcolor":   "#222222",
    "axes.grid":         True,
    "grid.color":        "#e0e0e0",
    "grid.alpha":        0.7,
    "grid.linestyle":    "--",
    "text.color":        "#222222",
    "xtick.color":       "#333333",
    "ytick.color":       "#333333",
    "legend.facecolor":  "white",
    "legend.edgecolor":  "#cccccc",
    "legend.fontsize":   9,
    "legend.framealpha":  0.95,
    "font.family":       "sans-serif",
    "font.size":         11,
    "axes.titlesize":    14,
    "axes.labelsize":    12,
    "savefig.dpi":       200,
    "savefig.bbox":      "tight",
    "savefig.facecolor": "white",
    "lines.linewidth":   1.5,
})

# Colour palette (professional, high contrast on white)
PAL = {
    "sim":     "#1565C0",   # deep blue for SU2
    "ref":     "#C62828",   # dark red for CFL3D reference
    "exp":     "#2E7D32",   # green for experiments
    "grid1":   "#1565C0",   # medium
    "grid2":   "#E65100",   # fine
    "grid3":   "#6A1B9A",   # xfine
    "rho":     "#1565C0",
    "rhoU":    "#E65100",
    "rhoV":    "#2E7D32",
    "rhoE":    "#C62828",
    "nu":      "#6A1B9A",
    "cd":      "#C62828",
    "cl":      "#1565C0",
    "grey":    "#888888",
    "band":    "#E3F2FD",
}

GRID_LABELS = {
    "medium": ("225×65", PAL["grid1"], "o"),
    "fine":   ("449×129", PAL["grid2"], "s"),
    "xfine":  ("897×257", PAL["grid3"], "D"),
}


# ── Data Loading ────────────────────────────────────────────────────────────

def load_results_summary():
    """Load the results_summary.json."""
    with open(RUNS / "results_summary.json") as f:
        return json.load(f)


def load_history(case_dir: Path):
    """Load SU2 convergence history CSV."""
    hist = defaultdict(list)
    fpath = case_dir / "history.csv"
    if not fpath.exists():
        return None
    with open(fpath, "r") as f:
        reader = csv.reader(f)
        headers = [h.strip().strip('"') for h in next(reader)]
        for row in reader:
            for h, v in zip(headers, row):
                try:
                    hist[h].append(float(v.strip()))
                except ValueError:
                    pass
    return dict(hist)


def load_surface(case_dir: Path):
    """Load SU2 surface_flow.csv and compute Cp from conservative variables."""
    fpath = case_dir / "surface_flow.csv"
    if not fpath.exists():
        return None
    data = defaultdict(list)
    with open(fpath, "r") as f:
        reader = csv.reader(f)
        headers = [h.strip().strip('"') for h in next(reader)]
        for row in reader:
            for h, v in zip(headers, row):
                try:
                    data[h].append(float(v.strip()))
                except ValueError:
                    pass
    return dict(data)


def compute_cp(surface_data, mach=0.15, gamma=1.4):
    """Compute Cp from conservative variables with FREESTREAM_PRESS_EQ_ONE.

    SU2 with REF_DIMENSIONALIZATION = FREESTREAM_PRESS_EQ_ONE uses:
        p_inf  = 1.0
        rho_inf = 1.0
        a_inf  = sqrt(gamma * p_inf / rho_inf) = sqrt(gamma)
        V_inf  = M * a_inf = M * sqrt(gamma)
        q_inf  = 0.5 * rho_inf * V_inf^2 = 0.5 * gamma * M^2
    """
    rho = np.array(surface_data["Density"])
    rhou = np.array(surface_data["Momentum_x"])
    rhov = np.array(surface_data["Momentum_y"])
    rhoE = np.array(surface_data["Energy"])

    # Velocity
    u = rhou / rho
    v = rhov / rho
    vel2 = u**2 + v**2

    # Pressure from ideal gas: p = (gamma-1) * (rhoE - 0.5*rho*vel^2)
    p = (gamma - 1.0) * (rhoE - 0.5 * rho * vel2)

    # Freestream values for FREESTREAM_PRESS_EQ_ONE
    p_inf = 1.0
    q_inf = 0.5 * gamma * mach**2

    cp = (p - p_inf) / q_inf

    x = np.array(surface_data["x"])
    y = np.array(surface_data["y"])

    return x, y, cp


def get_all_cases():
    """Discover all simulation run directories."""
    cases = []
    for d in sorted(RUNS.iterdir()):
        if d.is_dir() and d.name.startswith("alpha_"):
            parts = d.name.split("_")
            # alpha_0.0_SA_xfine
            alpha = float(parts[1])
            model = parts[2]
            grid = parts[3] if len(parts) > 3 else "unknown"
            cases.append({
                "dir": d,
                "alpha": alpha,
                "model": model,
                "grid": grid,
                "name": d.name,
            })
    return cases


# ── Plotting Functions ──────────────────────────────────────────────────────

def plot_convergence(cases):
    """Plot convergence histories for all xfine runs."""
    xfine_cases = [c for c in cases if c["grid"] == "xfine"]
    if not xfine_cases:
        xfine_cases = cases[:3]

    # Pre-load summary for reference lines
    try:
        summary = load_results_summary()
    except Exception:
        summary = None

    fig, axes = plt.subplots(len(xfine_cases), 2, figsize=(16, 5*len(xfine_cases)),
                              squeeze=False)
    fig.suptitle("SU2 Simulation Convergence  —  NACA 0012 (SA Model, 897×257 Grid)",
                 fontsize=16, fontweight="bold", y=0.995)

    for row, case in enumerate(xfine_cases):
        hist = load_history(case["dir"])
        if hist is None:
            continue

        iters = hist.get("Inner_Iter", list(range(len(hist.get("rms[Rho]", [])))))

        # Left: Residuals
        ax = axes[row, 0]
        for key, label, clr in [
            ("rms[Rho]",  "ρ",    PAL["rho"]),
            ("rms[RhoU]", "ρu",   PAL["rhoU"]),
            ("rms[RhoV]", "ρv",   PAL["rhoV"]),
            ("rms[RhoE]", "ρE",   PAL["rhoE"]),
            ("rms[nu]",   "ν̃",    PAL["nu"]),
        ]:
            if key in hist:
                ax.plot(iters[:len(hist[key])], hist[key],
                        color=clr, lw=1.0, label=label, alpha=0.85)

        ax.set_xlabel("Iteration")
        ax.set_ylabel("log₁₀(RMS Residual)")
        ax.set_title(f"Residuals — α = {case['alpha']}°", fontweight="bold")
        ax.legend(loc="upper right", fontsize=8, ncol=2)
        ax.set_ylim(top=0)

        # Right: Force coefficients
        ax2 = axes[row, 1]
        if "CL" in hist:
            ax2.plot(iters[:len(hist["CL"])], hist["CL"],
                     color=PAL["cl"], lw=1.2, label="CL")
        ax2_r = ax2.twinx()
        if "CD" in hist:
            ax2_r.plot(iters[:len(hist["CD"])], hist["CD"],
                       color=PAL["cd"], lw=1.2, label="CD", ls="--")

        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("CL", color=PAL["cl"])
        ax2_r.set_ylabel("CD", color=PAL["cd"])
        ax2.set_title(f"Force Convergence — α = {case['alpha']}°", fontweight="bold")
        ax2.tick_params(axis="y", labelcolor=PAL["cl"])
        ax2_r.tick_params(axis="y", labelcolor=PAL["cd"])

        # Add reference lines
        if summary and "results" in summary:
            alpha_key = f"alpha_{int(case['alpha'])}"
            if alpha_key in summary["results"]:
                ref = summary["results"][alpha_key]
                if "CL_ref" in ref:
                    ax2.axhline(ref["CL_ref"], color=PAL["ref"], ls=":", lw=1.0, alpha=0.6)
                if "CD_ref" in ref:
                    ax2_r.axhline(ref["CD_ref"], color=PAL["ref"], ls=":", lw=1.0, alpha=0.6)

        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_r.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2,
                   loc="center right", fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT / "sim_convergence.png")
    print(f"  Saved: sim_convergence.png")
    plt.close(fig)


def plot_forces_vs_alpha(summary):
    """Plot CL, CD, CM vs alpha — SU2 simulation vs TMR reference."""
    results = summary["results"]
    alphas_sim, cls_sim, cds_sim, cms_sim = [], [], [], []
    alphas_ref, cls_ref, cds_ref = [], [], []

    for key in sorted(results.keys()):
        r = results[key]
        alpha = float(key.replace("alpha_", ""))
        alphas_sim.append(alpha)
        cls_sim.append(r["CL_sim"])
        cds_sim.append(r["CD_sim"])
        cms_sim.append(r.get("CM_sim", 0))
        alphas_ref.append(alpha)
        cls_ref.append(r["CL_ref"])
        cds_ref.append(r["CD_ref"])

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("SU2 Simulation Results  —  NACA 0012 (SA, 897×257 Grid) vs CFL3D Reference (SA)",
                 fontsize=16, fontweight="bold")

    # ── Row 1: CL, CD, CM vs α ─────────────────────────────────────────────

    # CL vs alpha
    ax = axes[0, 0]
    ax.plot(alphas_ref, cls_ref, "o--", color=PAL["ref"], lw=2, markersize=10,
            label="CFL3D Reference (SA)", markeredgecolor="white", markeredgewidth=1.5,
            zorder=5)
    ax.plot(alphas_sim, cls_sim, "s-", color=PAL["sim"], lw=2, markersize=10,
            label="SU2 Simulation", markeredgecolor="white", markeredgewidth=1.5,
            zorder=6)
    # Thin airfoil theory
    a_line = np.linspace(-2, 16, 50)
    cl_line = 2 * np.pi * np.radians(a_line)
    ax.plot(a_line, cl_line, ":", color=PAL["grey"], lw=1, label="Thin airfoil theory", alpha=0.6)
    ax.set_xlabel("Angle of Attack α (°)")
    ax.set_ylabel("CL")
    ax.set_title("Lift Coefficient", fontweight="bold")
    ax.legend(loc="upper left", fontsize=9)

    # CD vs alpha
    ax = axes[0, 1]
    ax.plot(alphas_ref, cds_ref, "o--", color=PAL["ref"], lw=2, markersize=10,
            label="CFL3D Reference", markeredgecolor="white", markeredgewidth=1.5, zorder=5)
    ax.plot(alphas_sim, cds_sim, "s-", color=PAL["sim"], lw=2, markersize=10,
            label="SU2 Simulation", markeredgecolor="white", markeredgewidth=1.5, zorder=6)
    ax.set_xlabel("Angle of Attack α (°)")
    ax.set_ylabel("CD")
    ax.set_title("Drag Coefficient", fontweight="bold")
    ax.legend(loc="upper left", fontsize=9)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.4f"))

    # CM vs alpha
    ax = axes[0, 2]
    ax.plot(alphas_sim, cms_sim, "s-", color=PAL["sim"], lw=2, markersize=10,
            label="SU2 Simulation", markeredgecolor="white", markeredgewidth=1.5, zorder=6)
    ax.axhline(0, color=PAL["grey"], ls="-", lw=0.5, alpha=0.4)
    ax.set_xlabel("Angle of Attack α (°)")
    ax.set_ylabel("CM (about c/4)")
    ax.set_title("Pitching Moment Coefficient", fontweight="bold")
    ax.legend(loc="lower left", fontsize=9)

    # ── Row 2: Drag polar, L/D, Error summary ──────────────────────────────

    # Drag polar
    ax = axes[1, 0]
    ax.plot(cls_ref, cds_ref, "o--", color=PAL["ref"], lw=2, markersize=10,
            label="CFL3D Reference", markeredgecolor="white", markeredgewidth=1.5, zorder=5)
    ax.plot(cls_sim, cds_sim, "s-", color=PAL["sim"], lw=2, markersize=10,
            label="SU2 Simulation", markeredgecolor="white", markeredgewidth=1.5, zorder=6)
    ax.set_xlabel("CL")
    ax.set_ylabel("CD")
    ax.set_title("Drag Polar", fontweight="bold")
    ax.legend(loc="upper left", fontsize=9)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.4f"))

    # L/D vs alpha
    ax = axes[1, 1]
    ld_sim = [cl / cd if abs(cd) > 1e-6 else 0 for cl, cd in zip(cls_sim, cds_sim)]
    ld_ref = [cl / cd if abs(cd) > 1e-6 else 0 for cl, cd in zip(cls_ref, cds_ref)]
    # Filter out alpha=0 (L/D ≈ 0)
    mask = [i for i, a in enumerate(alphas_sim) if abs(a) > 0.5]
    if mask:
        ax.plot([alphas_ref[i] for i in mask], [ld_ref[i] for i in mask],
                "o--", color=PAL["ref"], lw=2, markersize=10, label="CFL3D Reference",
                markeredgecolor="white", markeredgewidth=1.5, zorder=5)
        ax.plot([alphas_sim[i] for i in mask], [ld_sim[i] for i in mask],
                "s-", color=PAL["sim"], lw=2, markersize=10, label="SU2 Simulation",
                markeredgecolor="white", markeredgewidth=1.5, zorder=6)
    ax.set_xlabel("Angle of Attack α (°)")
    ax.set_ylabel("CL / CD")
    ax.set_title("Lift-to-Drag Ratio", fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)

    # Error summary bar chart
    ax = axes[1, 2]
    labels = []
    cl_errs = []
    cd_errs = []
    for key in sorted(results.keys()):
        r = results[key]
        alpha = key.replace("alpha_", "")
        labels.append(f"α={alpha}°")
        cl_errs.append(abs(r["CL_error_pct"]))
        cd_errs.append(abs(r["CD_error_pct"]))

    x = np.arange(len(labels))
    width = 0.35
    bars1 = ax.bar(x - width/2, cl_errs, width, color=PAL["sim"],
                    label="CL error", edgecolor="white", linewidth=0.8)
    bars2 = ax.bar(x + width/2, cd_errs, width, color=PAL["cd"],
                    label="CD error", edgecolor="white", linewidth=0.8)

    # TMR tolerance band
    ax.axhline(1.0, color=PAL["sim"], ls=":", lw=1.5, alpha=0.5, label="TMR CL tol (1%)")
    ax.axhline(4.0, color=PAL["cd"], ls=":", lw=1.5, alpha=0.5, label="TMR CD tol (4%)")

    # Value labels on bars
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.2,
                f"{h:.1f}%", ha="center", va="bottom", fontsize=9, color=PAL["sim"])
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.2,
                f"{h:.1f}%", ha="center", va="bottom", fontsize=9, color=PAL["cd"])

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Error (%)")
    ax.set_title("Simulation Accuracy vs CFL3D", fontweight="bold")
    ax.legend(loc="upper left", fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / "sim_forces_vs_alpha.png")
    print(f"  Saved: sim_forces_vs_alpha.png")
    plt.close(fig)


def plot_grid_convergence(cases, summary):
    """Plot grid convergence study — CL, CD vs grid size at each alpha."""
    # Group by alpha
    by_alpha = defaultdict(list)
    for c in cases:
        by_alpha[c["alpha"]].append(c)

    alphas_with_grids = {a: cs for a, cs in by_alpha.items()
                         if len(cs) >= 2}

    if not alphas_with_grids:
        print("  [SKIP] Not enough grid levels for convergence study")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Grid Convergence Study  —  NACA 0012 (SA Model)",
                 fontsize=16, fontweight="bold")

    grid_sizes = {"medium": 225*65, "fine": 449*129, "xfine": 897*257}

    for col, (alpha, alpha_cases) in enumerate(sorted(alphas_with_grids.items())[:3]):
        ax_cl = axes[col]

        grids, cls, cds = [], [], []
        for c in sorted(alpha_cases, key=lambda x: grid_sizes.get(x["grid"], 0)):
            h = load_history(c["dir"])
            if h and "CL" in h and "CD" in h:
                grids.append(c["grid"])
                cls.append(h["CL"][-1])
                cds.append(h["CD"][-1])

        if not grids:
            continue

        sizes = [grid_sizes.get(g, 0) for g in grids]
        h_vals = [1.0/np.sqrt(s) for s in sizes]  # characteristic grid spacing

        # CL plot
        colors = [GRID_LABELS.get(g, ("?", PAL["grey"], "x"))[1] for g in grids]
        markers = [GRID_LABELS.get(g, ("?", PAL["grey"], "x"))[2] for g in grids]

        for i, g in enumerate(grids):
            lbl = GRID_LABELS.get(g, (g, PAL["grey"], "x"))
            ax_cl.plot(h_vals[i], cls[i], lbl[2], color=lbl[1], markersize=12,
                       label=f"{lbl[0]}  CL={cls[i]:.4f}",
                       markeredgecolor="white", markeredgewidth=1.5, zorder=5)

        # Reference line
        alpha_key = f"alpha_{int(alpha)}"
        if alpha_key in summary["results"]:
            ref_cl = summary["results"][alpha_key]["CL_ref"]
            ax_cl.axhline(ref_cl, color=PAL["ref"], ls="--", lw=1.5,
                          label=f"CFL3D Ref CL={ref_cl:.4f}", alpha=0.7)

        ax_cl.set_xlabel("Grid Spacing h = 1/√N")
        ax_cl.set_ylabel("CL")
        ax_cl.set_title(f"α = {alpha}°", fontweight="bold")
        ax_cl.legend(loc="best", fontsize=8)
        ax_cl.set_xlim(left=0)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT / "sim_grid_convergence.png")
    print(f"  Saved: sim_grid_convergence.png")
    plt.close(fig)


def load_cfl3d_reference_cp(alpha):
    """Load CFL3D SA reference Cp data for a given alpha."""
    alpha_int = int(alpha)
    ref_path = PROJECT / "experimental_data" / "naca0012" / "csv" / f"cfl3d_sa_cp_alpha{alpha_int}.csv"
    if not ref_path.exists():
        return None, None
    x_ref, cp_ref = [], []
    with open(ref_path, "r") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            if len(row) >= 2:
                x_ref.append(float(row[0]))
                cp_ref.append(float(row[1]))
    return np.array(x_ref), np.array(cp_ref)


def plot_surface_cp(cases):
    """Plot Cp distribution from simulation surface data on the finest grid."""
    xfine_cases = [c for c in cases if c["grid"] == "xfine"]
    if not xfine_cases:
        xfine_cases = cases

    # Filter to unique alphas
    seen = set()
    unique = []
    for c in xfine_cases:
        if c["alpha"] not in seen:
            seen.add(c["alpha"])
            unique.append(c)
    unique.sort(key=lambda c: c["alpha"])

    n_plots = len(unique)
    if n_plots == 0:
        print("  [SKIP] No surface data for Cp plots")
        return

    fig, axes = plt.subplots(1, n_plots, figsize=(7*n_plots, 6), squeeze=False)
    fig.suptitle("SU2 vs CFL3D Surface Pressure  —  NACA 0012 (SA, 897×257 Grid)",
                 fontsize=16, fontweight="bold")

    for i, case in enumerate(unique):
        ax = axes[0, i]
        sdata = load_surface(case["dir"])
        if sdata is None:
            ax.text(0.5, 0.5, "No surface data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=14, color=PAL["grey"])
            continue

        x, y, cp = compute_cp(sdata, mach=0.15)

        # Separate upper and lower surface
        # Sort by x for cleaner plot
        order = np.argsort(x)
        x_sorted = x[order]
        y_sorted = y[order]
        cp_sorted = cp[order]

        upper = y_sorted >= 0
        lower = y_sorted < 0

        # SU2 simulation lines
        ax.plot(x_sorted[upper], cp_sorted[upper], "-", color=PAL["sim"],
                lw=1.5, label="SU2 Upper", alpha=0.9)
        ax.plot(x_sorted[lower], cp_sorted[lower], "-", color=PAL["cd"],
                lw=1.5, label="SU2 Lower", alpha=0.9)

        # CFL3D reference Cp overlay
        x_ref, cp_ref = load_cfl3d_reference_cp(case["alpha"])
        if x_ref is not None:
            ax.plot(x_ref, cp_ref, "--", color=PAL["grey"], lw=1.2,
                    label="CFL3D Ref", alpha=0.7, zorder=3)

        ax.invert_yaxis()
        ax.set_xlabel("x/c")
        ax.set_ylabel("Cp")
        ax.set_title(f"\u03b1 = {case['alpha']}\u00b0", fontweight="bold")
        ax.legend(loc="lower right", fontsize=9)
        ax.set_xlim(-0.02, 1.05)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT / "sim_cp_distribution.png")
    print(f"  Saved: sim_cp_distribution.png")
    plt.close(fig)


def plot_model_comparison(cases):
    """Compare SA vs SST when both models have results on the same grid."""
    models = set(c["model"] for c in cases)
    if "SST" not in models or "SA" not in models:
        print("  [SKIP] Model comparison: need both SA and SST results")
        return

    # Get xfine cases for both models
    sa_cases = sorted([c for c in cases if c["model"] == "SA" and c["grid"] == "xfine"],
                       key=lambda c: c["alpha"])
    sst_cases = sorted([c for c in cases if c["model"] == "SST" and c["grid"] == "xfine"],
                        key=lambda c: c["alpha"])

    if not sa_cases or not sst_cases:
        print("  [SKIP] Model comparison: need xfine results from both models")
        return

    # Find common alphas
    sa_alphas = {c["alpha"]: c for c in sa_cases}
    sst_alphas = {c["alpha"]: c for c in sst_cases}
    common = sorted(set(sa_alphas.keys()) & set(sst_alphas.keys()))
    if not common:
        print("  [SKIP] Model comparison: no common alphas between SA and SST")
        return

    # --- Force Coefficients Comparison ---
    sa_cl, sa_cd, sst_cl, sst_cd = [], [], [], []
    for alpha in common:
        from run_naca0012 import parse_su2_history
        sa_h = parse_su2_history(sa_alphas[alpha]["dir"])
        sst_h = parse_su2_history(sst_alphas[alpha]["dir"])
        if sa_h and sst_h:
            sa_cl.append(sa_h.get("CL", 0))
            sa_cd.append(sa_h.get("CD", 0))
            sst_cl.append(sst_h.get("CL", 0))
            sst_cd.append(sst_h.get("CD", 0))
        else:
            sa_cl.append(None); sa_cd.append(None)
            sst_cl.append(None); sst_cd.append(None)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("SA vs SST Model Comparison  —  NACA 0012 (897×257 Grid)",
                 fontsize=14, fontweight="bold")

    # CL comparison
    ax = axes[0]
    valid = [i for i in range(len(common)) if sa_cl[i] is not None and sst_cl[i] is not None]
    ax.plot([common[i] for i in valid], [sa_cl[i] for i in valid],
            "o-", color=PAL["sim"], lw=2, ms=7, label="SA", zorder=5)
    ax.plot([common[i] for i in valid], [sst_cl[i] for i in valid],
            "s--", color=PAL["cd"], lw=2, ms=7, label="SST", zorder=5)
    ax.set_xlabel(r"$\alpha$ (deg)")
    ax.set_ylabel(r"$C_L$")
    ax.set_title("Lift Coefficient", fontweight="bold")
    ax.legend()

    # CD comparison
    ax = axes[1]
    ax.plot([common[i] for i in valid], [sa_cd[i] for i in valid],
            "o-", color=PAL["sim"], lw=2, ms=7, label="SA", zorder=5)
    ax.plot([common[i] for i in valid], [sst_cd[i] for i in valid],
            "s--", color=PAL["cd"], lw=2, ms=7, label="SST", zorder=5)

    # Load TMR reference for overlay
    tmr_file = PROJECT / "experimental_data" / "naca0012" / "csv" / "tmr_sa_reference.json"
    if tmr_file.exists():
        with open(tmr_file) as f:
            tmr = json.load(f)
        ref_alphas, ref_cd = [], []
        for alpha in common:
            key = f"alpha_{int(alpha)}"
            if key in tmr:
                ref_alphas.append(alpha)
                ref_cd.append(tmr[key]["CD"])
        if ref_cd:
            ax.plot(ref_alphas, ref_cd, "^:", color=PAL["grey"], lw=1.5, ms=6,
                    label="CFL3D Ref (SA)", alpha=0.7)

    ax.set_xlabel(r"$\alpha$ (deg)")
    ax.set_ylabel(r"$C_D$")
    ax.set_title("Drag Coefficient", fontweight="bold")
    ax.legend()

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT / "sim_model_comparison.png", dpi=200)
    print(f"  Saved: sim_model_comparison.png")
    plt.close(fig)

    # --- Cp Distribution Comparison ---
    n_plots = min(len(common), 3)
    cp_alphas = common[:n_plots]

    fig, axes = plt.subplots(1, n_plots, figsize=(7*n_plots, 6), squeeze=False)
    fig.suptitle("SA vs SST Surface Pressure  —  NACA 0012 (897×257 Grid)",
                 fontsize=16, fontweight="bold")

    for i, alpha in enumerate(cp_alphas):
        ax = axes[0, i]

        # SA Cp
        sa_sdata = load_surface(sa_alphas[alpha]["dir"])
        if sa_sdata:
            x, y, cp = compute_cp(sa_sdata, mach=0.15)
            order = np.argsort(x)
            x_s, y_s, cp_s = x[order], y[order], cp[order]
            upper = y_s >= 0
            lower = y_s < 0
            ax.plot(x_s[upper], cp_s[upper], "-", color=PAL["sim"],
                    lw=1.5, label="SA Upper", alpha=0.9)
            ax.plot(x_s[lower], cp_s[lower], "-", color=PAL["cl"],
                    lw=1.0, label="SA Lower", alpha=0.6)

        # SST Cp
        sst_sdata = load_surface(sst_alphas[alpha]["dir"])
        if sst_sdata:
            x, y, cp = compute_cp(sst_sdata, mach=0.15)
            order = np.argsort(x)
            x_s, y_s, cp_s = x[order], y[order], cp[order]
            upper = y_s >= 0
            lower = y_s < 0
            ax.plot(x_s[upper], cp_s[upper], "--", color=PAL["cd"],
                    lw=1.5, label="SST Upper", alpha=0.9)
            ax.plot(x_s[lower], cp_s[lower], "--", color="#E65100",
                    lw=1.0, label="SST Lower", alpha=0.6)

        # CFL3D reference
        x_ref, cp_ref = load_cfl3d_reference_cp(alpha)
        if x_ref is not None:
            ax.plot(x_ref, cp_ref, ":", color=PAL["grey"], lw=1.2,
                    label="CFL3D Ref (SA)", alpha=0.7, zorder=3)

        ax.invert_yaxis()
        ax.set_xlabel("x/c")
        ax.set_ylabel("Cp")
        ax.set_title(f"\u03b1 = {alpha}\u00b0", fontweight="bold")
        ax.legend(loc="lower right", fontsize=8)
        ax.set_xlim(-0.02, 1.05)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT / "sim_model_cp_comparison.png", dpi=200)
    print(f"  Saved: sim_model_cp_comparison.png")
    plt.close(fig)

def plot_simulation_dashboard(summary):
    """Create an at-a-glance dashboard of simulation metrics."""
    results = summary["results"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("SU2 Simulation Dashboard  —  NACA 0012 TMR Validation",
                 fontsize=16, fontweight="bold")

    # ── Panel 1: Iterations to converge ─────────────────────────────────────
    ax = axes[0, 0]
    labels = []
    iters = []
    colors_bar = []
    for key in sorted(results.keys()):
        r = results[key]
        alpha = key.replace("alpha_", "")
        labels.append(f"α={alpha}°")
        iters.append(r["iterations"])
        colors_bar.append(PAL["sim"])

    bars = ax.bar(labels, iters, color=colors_bar, edgecolor="white", linewidth=0.8)
    for bar, n in zip(bars, iters):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f"{n:,}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("Iterations")
    ax.set_title("Convergence Iterations", fontweight="bold")

    # ── Panel 2: Final residual levels ──────────────────────────────────────
    ax = axes[0, 1]
    rms_vals = []
    for key in sorted(results.keys()):
        rms_vals.append(results[key]["rms_rho"])
        alpha = key.replace("alpha_", "")
    ax.barh(labels, rms_vals, color=["#4CAF50" if r < -9 else "#FF9800" if r < -8 else "#F44336"
                                      for r in rms_vals],
            edgecolor="white", linewidth=0.8)
    for i, v in enumerate(rms_vals):
        ax.text(v + 0.05, i, f"{v:.2f}", ha="left", va="center", fontsize=10)
    ax.axvline(-9, color=PAL["grey"], ls=":", lw=1, alpha=0.5)
    ax.text(-9, len(labels)-0.3, "−9 target", fontsize=8, color=PAL["grey"], ha="center")
    ax.set_xlabel("log₁₀(RMS ρ)")
    ax.set_title("Final Residual Level", fontweight="bold")
    ax.invert_xaxis()

    # ── Panel 3: CL comparison table-like ───────────────────────────────────
    ax = axes[1, 0]
    ax.axis("off")

    table_data = [["α", "CL (SU2)", "CL (TMR)", "Error (%)", "CD (SU2)", "CD (TMR)", "Error (%)"]]
    for key in sorted(results.keys()):
        r = results[key]
        alpha = key.replace("alpha_", "")
        table_data.append([
            f"{alpha}°",
            f"{r['CL_sim']:.4f}",
            f"{r['CL_ref']:.4f}",
            f"{r['CL_error_pct']:.1f}%",
            f"{r['CD_sim']:.5f}",
            f"{r['CD_ref']:.5f}",
            f"{r['CD_error_pct']:.1f}%",
        ])

    table = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                      loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.8)

    # Style header
    for j in range(len(table_data[0])):
        table[0, j].set_facecolor("#1565C0")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Colour code errors
    for i in range(1, len(table_data)):
        for j in [3, 6]:  # error columns
            val = float(table_data[i][j].replace("%", ""))
            if abs(val) <= 1.0:
                table[i, j-1].set_facecolor("#E8F5E9")  # green
            elif abs(val) <= 4.0:
                table[i, j-1].set_facecolor("#FFF3E0")  # orange
            else:
                table[i, j-1].set_facecolor("#FFEBEE")  # red

    ax.set_title("Force Coefficient Comparison", fontweight="bold", y=0.95)

    # ── Panel 4: Status + key parameters ────────────────────────────────────
    ax = axes[1, 1]
    ax.axis("off")

    info_lines = [
        f"Solver:  {summary['solver']}",
        f"Model:   {summary['turbulence_model']}",
        f"Grid:    {summary['grid']}",
        f"Mach:    {summary['conditions']['Mach']}",
        f"Re:      {summary['conditions']['Reynolds']:,.0f}",
        "",
        "Numerical Settings:",
        f"  Scheme:   {summary['numerical_settings']['conv_num_method_flow']}",
        f"  Limiter:  {summary['numerical_settings']['slope_limiter']}",
        f"  MG:       {summary['numerical_settings']['multigrid']}",
        f"  Solver:   {summary['numerical_settings']['linear_solver']}",
        "",
        "Key Improvements:",
    ]
    for fix in summary.get("improvement_history", {}).get("key_fixes", [])[:4]:
        info_lines.append(f"  • {fix[:60]}")

    info_text = "\n".join(info_lines)
    ax.text(0.05, 0.95, info_text, transform=ax.transAxes,
            fontsize=9, fontfamily="monospace", verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#F5F5F5",
                      edgecolor="#CCCCCC"))
    ax.set_title("Simulation Configuration", fontweight="bold", y=0.95)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT / "sim_dashboard.png")
    print(f"  Saved: sim_dashboard.png")
    plt.close(fig)


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  SU2 Simulation Results — NACA 0012 Plots")
    print("=" * 60)

    cases = get_all_cases()
    print(f"\nFound {len(cases)} simulation runs:")
    for c in cases:
        print(f"  alpha={c['alpha']:5.1f} deg  {c['model']}  {c['grid']}")

    summary = load_results_summary()

    print("\nGenerating plots (white background)...")
    plot_convergence(cases)
    plot_forces_vs_alpha(summary)
    plot_grid_convergence(cases, summary)
    plot_surface_cp(cases)
    plot_model_comparison(cases)
    plot_simulation_dashboard(summary)

    print(f"\nAll plots saved to: {OUT}/")
    print("Done!")


if __name__ == "__main__":
    main()
