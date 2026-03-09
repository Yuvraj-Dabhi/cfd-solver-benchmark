#!/usr/bin/env python3
"""
NACA 0012 TMR Validation — Comprehensive Aerodynamic Plots
===========================================================
Generates publication-quality plots for all aerodynamic parameters
from the TMR experimental and CFD reference datasets.

Plots:
  1. CL vs alpha       (Ladson, Abbott, Gregory, McCroskey, CFL3D SA)
  2. CD vs alpha        (Ladson 3 grits, CFL3D SA)
  3. Drag polar CD-CL   (Abbott, Ladson, CFL3D SA)
  4. L/D vs alpha       (Ladson, CFL3D SA)
  5-7. Cp vs x/c        (Gregory + Ladson + CFL3D at alpha = 0, 10, 15)
  8-10. Cf vs x/c       (CFL3D SA at alpha = 0, 10, 15)
"""

import sys
import os
from pathlib import Path

# Add project paths
PROJECT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT / "experimental_data" / "naca0012"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from naca0012_tmr_data import (
    parse_ladson_forces, parse_abbott_cl, parse_abbott_cd,
    parse_gregory_cl, parse_mccroskey_cl, parse_cfl3d_forces,
    parse_gregory_cp, parse_ladson_cp, parse_cfl3d_cp, parse_cfl3d_cf,
    TMR_SA_REFERENCE,
)

DATA_DIR = PROJECT / "experimental_data" / "naca0012"
OUT_DIR = PROJECT / "plots"
OUT_DIR.mkdir(exist_ok=True)

# ── Visual style ────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0f1117",
    "axes.facecolor":   "#181b25",
    "axes.edgecolor":   "#3a3f55",
    "axes.labelcolor":  "#e0e0e0",
    "axes.grid":        True,
    "grid.color":       "#2a2e3d",
    "grid.alpha":       0.6,
    "text.color":       "#e0e0e0",
    "xtick.color":      "#b0b0b0",
    "ytick.color":      "#b0b0b0",
    "legend.facecolor": "#1e2130",
    "legend.edgecolor": "#3a3f55",
    "legend.fontsize":  9,
    "font.family":      "sans-serif",
    "font.size":        11,
    "axes.titlesize":   14,
    "axes.labelsize":   12,
    "savefig.dpi":      200,
    "savefig.bbox":     "tight",
    "savefig.facecolor":"#0f1117",
})

# Colour palette
C = {
    "ladson80":   "#4fc3f7",
    "ladson120":  "#29b6f6",
    "ladson180":  "#039be5",
    "abbott":     "#ff8a65",
    "gregory":    "#aed581",
    "mccroskey":  "#ce93d8",
    "cfl3d":      "#ef5350",
    "tmr_band":   "#ffffff",
    "ladson_cp":  "#4fc3f7",
    "gregory_cp": "#aed581",
    "cfl3d_cp":   "#ef5350",
    "cfl3d_cf":   "#ffa726",
}

MARKER = dict(markersize=5, markeredgewidth=0.6, markeredgecolor="#0f1117")


def _split_ladson(ladson, grit_sizes=(17, 18, 18)):
    """Split Ladson data into 3 grit zones (80, 120, 180)."""
    grits = {}
    idx = 0
    for grit, n in zip(["80 grit", "120 grit", "180 grit"], grit_sizes):
        grits[grit] = {
            "alpha": ladson["alpha"][idx:idx+n],
            "CL":    ladson["CL"][idx:idx+n],
            "CD":    ladson["CD"][idx:idx+n],
        }
        idx += n
    return grits


def plot_cl_vs_alpha(ax, ladson_grits, abbott, gregory, mccroskey, cfl3d):
    """Plot lift coefficient vs angle of attack."""
    # McCroskey best-fit slope
    ax.plot(mccroskey["alpha"], mccroskey["CL"], "--",
            color=C["mccroskey"], lw=1.4, label="McCroskey best fit", zorder=2)

    # Abbott (untripped)
    ax.plot(abbott["alpha"], abbott["CL"], "d",
            color=C["abbott"], label="Abbott (untripped)", alpha=0.7, **MARKER)

    # Gregory
    ax.plot(gregory["alpha"], gregory["CL"], "^",
            color=C["gregory"], label="Gregory (Re=3M)", alpha=0.8, **MARKER)

    # Ladson — average of 3 grits for cleaner plot
    for grit_name, gdata in ladson_grits.items():
        ax.plot(gdata["alpha"], gdata["CL"], "o",
                color=C[f"ladson{grit_name.split()[0]}"],
                label=f"Ladson {grit_name}", alpha=0.85, **MARKER)

    # CFL3D SA
    ax.plot(cfl3d["alpha"], cfl3d["CL"], "s",
            color=C["cfl3d"], label="CFL3D SA", markersize=8,
            markeredgewidth=1.2, markeredgecolor="#fff", zorder=5)

    # TMR consensus markers
    for key in ["alpha_0", "alpha_10", "alpha_15"]:
        a = float(key.split("_")[1])
        ref = TMR_SA_REFERENCE[key]
        ax.axhline(ref["CL"], color=C["tmr_band"], alpha=0.08, lw=8)

    ax.set_xlabel("Angle of Attack α (°)")
    ax.set_ylabel("Lift Coefficient  CL")
    ax.set_title("CL  vs  α", fontweight="bold")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.set_xlim(-6, 20)


def plot_cd_vs_alpha(ax, ladson_grits, cfl3d):
    """Plot drag coefficient vs angle of attack (pre-stall only)."""
    for grit_name, gdata in ladson_grits.items():
        # Filter pre-stall only (CD < 0.04)
        mask = [i for i, cd in enumerate(gdata["CD"]) if cd < 0.04]
        alphas = [gdata["alpha"][i] for i in mask]
        cds    = [gdata["CD"][i] for i in mask]
        ax.plot(alphas, cds, "o",
                color=C[f"ladson{grit_name.split()[0]}"],
                label=f"Ladson {grit_name}", alpha=0.85, **MARKER)

    ax.plot(cfl3d["alpha"], cfl3d["CD"], "s",
            color=C["cfl3d"], label="CFL3D SA", markersize=8,
            markeredgewidth=1.2, markeredgecolor="#fff", zorder=5)

    ax.set_xlabel("Angle of Attack α (°)")
    ax.set_ylabel("Drag Coefficient  CD")
    ax.set_title("CD  vs  α  (pre-stall)", fontweight="bold")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_xlim(-6, 20)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.4f"))


def plot_drag_polar(ax, ladson_grits, abbott_cd, cfl3d):
    """Plot drag polar (CD vs CL)."""
    # Abbott CD vs CL (untripped)
    ax.plot(abbott_cd["CL"], abbott_cd["CD"], "d",
            color=C["abbott"], label="Abbott (untripped)", alpha=0.7, **MARKER)

    # Ladson — pre-stall
    for grit_name, gdata in ladson_grits.items():
        mask = [i for i, cd in enumerate(gdata["CD"]) if cd < 0.04]
        cls = [gdata["CL"][i] for i in mask]
        cds = [gdata["CD"][i] for i in mask]
        ax.plot(cls, cds, "o",
                color=C[f"ladson{grit_name.split()[0]}"],
                label=f"Ladson {grit_name}", alpha=0.85, **MARKER)

    # CFL3D
    ax.plot(cfl3d["CL"], cfl3d["CD"], "s",
            color=C["cfl3d"], label="CFL3D SA", markersize=8,
            markeredgewidth=1.2, markeredgecolor="#fff", zorder=5)

    ax.set_xlabel("Lift Coefficient  CL")
    ax.set_ylabel("Drag Coefficient  CD")
    ax.set_title("Drag Polar  CD  vs  CL", fontweight="bold")
    ax.legend(loc="upper left", fontsize=8)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.4f"))


def plot_ld_vs_alpha(ax, ladson_grits, cfl3d):
    """Plot Lift-to-Drag ratio vs alpha."""
    for grit_name, gdata in ladson_grits.items():
        mask = [i for i, cd in enumerate(gdata["CD"]) if 0.001 < cd < 0.04
                and abs(gdata["alpha"][i]) > 0.5]
        alphas = [gdata["alpha"][i] for i in mask]
        lds = [gdata["CL"][i] / gdata["CD"][i] for i in mask]
        ax.plot(alphas, lds, "o",
                color=C[f"ladson{grit_name.split()[0]}"],
                label=f"Ladson {grit_name}", alpha=0.85, **MARKER)

    # CFL3D (skip alpha=0 to avoid division issues)
    mask = [i for i in range(len(cfl3d["alpha"]))
            if abs(cfl3d["alpha"][i]) > 0.5]
    ax.plot([cfl3d["alpha"][i] for i in mask],
            [cfl3d["CL"][i]/cfl3d["CD"][i] for i in mask], "s",
            color=C["cfl3d"], label="CFL3D SA", markersize=8,
            markeredgewidth=1.2, markeredgecolor="#fff", zorder=5)

    ax.set_xlabel("Angle of Attack α (°)")
    ax.set_ylabel("Lift-to-Drag Ratio  CL/CD")
    ax.set_title("L/D  vs  α", fontweight="bold")
    ax.legend(loc="upper right", fontsize=8)


def plot_cp(ax, alpha_deg, gregory_cp, ladson_cp, cfl3d_cp):
    """Plot Cp distribution at a specific alpha."""
    # CFL3D reference (full surface, continuous line)
    zone_key = f"alpha={int(alpha_deg)}"
    cfl3d_zone = None
    for k, v in cfl3d_cp["zones"].items():
        if zone_key.lower() in k.lower():
            cfl3d_zone = v
            break

    if cfl3d_zone:
        ax.plot(cfl3d_zone["x"], cfl3d_zone["cp"], "-",
                color=C["cfl3d_cp"], lw=1.5, label="CFL3D SA", zorder=3)

    # Gregory experimental (upper surface only)
    for k, v in gregory_cp["zones"].items():
        if f"alpha={int(alpha_deg)}" in k.lower():
            ax.plot(v["x/c"], v["cp"], "^",
                    color=C["gregory_cp"], label="Gregory (Re=2.88M)",
                    alpha=0.9, **MARKER)
            break

    # Ladson experimental (Re=6M, free transition preferred)
    for k, v in ladson_cp["zones"].items():
        if f"alpha" in k.lower() and "6 million" in k.lower().replace("=", " "):
            # Match alpha approximately
            try:
                # Extract alpha from zone name like "Re=6 million, alpha=10.0254, free transition"
                parts = k.lower().replace("=", " ").replace(",", " ").split()
                for i, p in enumerate(parts):
                    if "alpha" in p and i+1 < len(parts):
                        zone_alpha = float(parts[i+1].rstrip(","))
                        if abs(zone_alpha - alpha_deg) < 1.0:
                            ax.plot(v["x/c"], v["cp"], "o",
                                    color=C["ladson_cp"],
                                    label="Ladson (Re=6M, free)",
                                    alpha=0.7, markersize=4,
                                    markeredgewidth=0.4,
                                    markeredgecolor="#0f1117")
                            break
            except (ValueError, IndexError):
                continue

    ax.set_xlabel("x/c")
    ax.set_ylabel("Cp")
    ax.set_title(f"Cp Distribution  α = {int(alpha_deg)}°", fontweight="bold")
    ax.invert_yaxis()
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim(-0.02, 1.05)


def plot_cf(ax, alpha_deg, cfl3d_cf):
    """Plot Cf distribution at a specific alpha."""
    zone_key = f"alpha={int(alpha_deg)}"

    for k, v in cfl3d_cf["zones"].items():
        if zone_key.lower() in k.lower():
            surface = "upper" if "upper" in k.lower() else "lower" if "lower" in k.lower() else ""
            lbl = f"CFL3D SA ({surface})" if surface else "CFL3D SA"
            clr = C["cfl3d_cf"] if "upper" in k.lower() else "#66bb6a"
            ax.plot(v["x"], v["cf"], "-", color=clr, lw=1.3, label=lbl, zorder=3)

    ax.set_xlabel("x/c")
    ax.set_ylabel("Cf")
    ax.set_title(f"Skin Friction  α = {int(alpha_deg)}°", fontweight="bold")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(-0.02, 1.05)
    ax.axhline(0, color="#555", lw=0.5, ls="--", alpha=0.5)


def main():
    print("Loading TMR data...")

    # Load all datasets
    ladson  = parse_ladson_forces(DATA_DIR)
    abbott  = parse_abbott_cl(DATA_DIR)
    abbott_cd = parse_abbott_cd(DATA_DIR)
    gregory = parse_gregory_cl(DATA_DIR)
    mccroskey = parse_mccroskey_cl(DATA_DIR)
    cfl3d   = parse_cfl3d_forces(DATA_DIR)

    gregory_cp = parse_gregory_cp(DATA_DIR)
    ladson_cp  = parse_ladson_cp(DATA_DIR)
    cfl3d_cp   = parse_cfl3d_cp(DATA_DIR)
    cfl3d_cf   = parse_cfl3d_cf(DATA_DIR)

    ladson_grits = _split_ladson(ladson)

    # ── Figure 1: Force Coefficients (2×2) ──────────────────────────────────
    print("Plotting force coefficients...")
    fig1, axes1 = plt.subplots(2, 2, figsize=(16, 12))
    fig1.suptitle("NACA 0012  —  Force Coefficients  (M=0.15, Re=6×10⁶)",
                  fontsize=17, fontweight="bold", color="#ffffff", y=0.98)

    plot_cl_vs_alpha(axes1[0, 0], ladson_grits, abbott, gregory, mccroskey, cfl3d)
    plot_cd_vs_alpha(axes1[0, 1], ladson_grits, cfl3d)
    plot_drag_polar(axes1[1, 0], ladson_grits, abbott_cd, cfl3d)
    plot_ld_vs_alpha(axes1[1, 1], ladson_grits, cfl3d)

    fig1.tight_layout(rect=[0, 0, 1, 0.95])
    fig1.savefig(OUT_DIR / "naca0012_force_coefficients.png")
    print(f"  Saved: {OUT_DIR / 'naca0012_force_coefficients.png'}")

    # ── Figure 2: Cp Distributions (1×3) ────────────────────────────────────
    print("Plotting Cp distributions...")
    fig2, axes2 = plt.subplots(1, 3, figsize=(20, 6.5))
    fig2.suptitle("NACA 0012  —  Surface Pressure Coefficient  (CFL3D SA + Experiments)",
                  fontsize=17, fontweight="bold", color="#ffffff", y=1.02)

    for i, alpha in enumerate([0, 10, 15]):
        plot_cp(axes2[i], alpha, gregory_cp, ladson_cp, cfl3d_cp)

    fig2.tight_layout(rect=[0, 0, 1, 0.96])
    fig2.savefig(OUT_DIR / "naca0012_cp_distributions.png")
    print(f"  Saved: {OUT_DIR / 'naca0012_cp_distributions.png'}")

    # ── Figure 3: Cf Distributions (1×3) ────────────────────────────────────
    print("Plotting Cf distributions...")
    fig3, axes3 = plt.subplots(1, 3, figsize=(20, 6.5))
    fig3.suptitle("NACA 0012  —  Skin Friction Coefficient  (CFL3D SA Reference)",
                  fontsize=17, fontweight="bold", color="#ffffff", y=1.02)

    for i, alpha in enumerate([0, 10, 15]):
        plot_cf(axes3[i], alpha, cfl3d_cf)

    fig3.tight_layout(rect=[0, 0, 1, 0.96])
    fig3.savefig(OUT_DIR / "naca0012_cf_distributions.png")
    print(f"  Saved: {OUT_DIR / 'naca0012_cf_distributions.png'}")

    # ── Figure 4: 7-Code Comparison Bar Chart ───────────────────────────────
    print("Plotting 7-code comparison...")
    per_code = TMR_SA_REFERENCE["per_code_results"]
    codes = list(per_code.keys())
    colors_codes = ["#ef5350", "#42a5f5", "#66bb6a", "#ffa726",
                    "#ab47bc", "#26c6da", "#8d6e63"]

    fig4, axes4 = plt.subplots(2, 3, figsize=(18, 10))
    fig4.suptitle("NACA 0012  —  SA Model 7-Code Comparison  (897×257 grid)",
                  fontsize=17, fontweight="bold", color="#ffffff", y=0.98)

    for col, alpha in enumerate([0, 10, 15]):
        # CL row
        vals_cl = [per_code[c][f"CL_{alpha}"] for c in codes]
        bars = axes4[0, col].bar(codes, vals_cl, color=colors_codes,
                                  edgecolor="#0f1117", linewidth=0.8)
        mean_cl = np.mean(vals_cl)
        axes4[0, col].axhline(mean_cl, color="#fff", ls="--", lw=1, alpha=0.5)
        axes4[0, col].set_title(f"CL at α={alpha}°", fontweight="bold")
        axes4[0, col].set_ylabel("CL")
        axes4[0, col].tick_params(axis="x", rotation=35)

        # CD row
        vals_cd = [per_code[c][f"CD_{alpha}"] for c in codes]
        axes4[1, col].bar(codes, vals_cd, color=colors_codes,
                           edgecolor="#0f1117", linewidth=0.8)
        mean_cd = np.mean(vals_cd)
        axes4[1, col].axhline(mean_cd, color="#fff", ls="--", lw=1, alpha=0.5)
        axes4[1, col].set_title(f"CD at α={alpha}°", fontweight="bold")
        axes4[1, col].set_ylabel("CD")
        axes4[1, col].tick_params(axis="x", rotation=35)
        axes4[1, col].yaxis.set_major_formatter(
            ticker.FormatStrFormatter("%.5f"))

    fig4.tight_layout(rect=[0, 0, 1, 0.94])
    fig4.savefig(OUT_DIR / "naca0012_7code_comparison.png")
    print(f"  Saved: {OUT_DIR / 'naca0012_7code_comparison.png'}")

    plt.close("all")
    print(f"\nAll plots saved to: {OUT_DIR}")
    print("Done!")


if __name__ == "__main__":
    main()
