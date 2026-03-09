#!/usr/bin/env python3
"""
Model-Form Uncertainty Analysis
=================================
Frames CFD results within the context of model-form uncertainty:

1. NACA 0012: Uses the 7-code TMR SA ensemble (CFL3D, FUN3D, NTS, JOE,
   SUMB, TURNS, GGNS) as a model-form uncertainty band, treating min–max
   spread as a reference against which SU2's deviations are interpreted.

2. Wall Hump: Cites multi-model literature (SA, SST, v2-f) to show that
   all RANS models over-predict bubble length and reverse-flow intensity,
   framing our SA/SST behaviour as consistent with known RANS deficiencies.

References:
  - TMR: tmbwg.github.io/turbmodels (7-code SA verification)
  - Rumsey et al. (2006), J. Fluids Eng. 128(3)
  - Greenblatt et al. (2006), AIAA J. 44(12)
  - CFDVAL2004 Workshop proceedings
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT))


# ===================================================================
# NACA 0012 -- TMR 7-Code SA Ensemble
# ===================================================================

def load_tmr_ensemble(
    data_dir: Optional[Path] = None,
) -> Dict:
    """
    Load the TMR 7-code SA reference ensemble.

    Returns
    -------
    dict with per-code results and ensemble statistics.
    """
    data_dir = Path(data_dir) if data_dir else (
        PROJECT / "experimental_data" / "naca0012" / "csv"
    )
    ref_file = data_dir / "tmr_sa_reference.json"

    if not ref_file.exists():
        raise FileNotFoundError(f"TMR reference not found: {ref_file}")

    with open(ref_file) as f:
        data = json.load(f)

    return data


def compute_ensemble_statistics(
    tmr_data: Dict,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Compute min, max, mean, std for CL and CD across all 7 codes
    at each angle of attack.

    Returns
    -------
    dict: {alpha_key: {CL: {min, max, mean, std, spread, spread_pct},
                       CD: {min, max, mean, std, spread, spread_pct}}}
    """
    per_code = tmr_data["per_code_results"]
    codes = list(per_code.keys())

    stats = {}
    for alpha_key in ["alpha_0", "alpha_10", "alpha_15"]:
        # Map alpha_key to per-code key format
        alpha_num = alpha_key.split("_")[1]  # "0", "10", "15"

        cl_vals = [per_code[c].get(f"CL_{alpha_num}") for c in codes
                   if per_code[c].get(f"CL_{alpha_num}") is not None]
        cd_vals = [per_code[c].get(f"CD_{alpha_num}") for c in codes
                   if per_code[c].get(f"CD_{alpha_num}") is not None]

        cl_arr = np.array(cl_vals)
        cd_arr = np.array(cd_vals)

        cl_mean = float(np.mean(cl_arr))
        cd_mean = float(np.mean(cd_arr))

        stats[alpha_key] = {
            "alpha": float(alpha_num),
            "CL": {
                "min": float(np.min(cl_arr)),
                "max": float(np.max(cl_arr)),
                "mean": cl_mean,
                "std": float(np.std(cl_arr)),
                "spread": float(np.max(cl_arr) - np.min(cl_arr)),
                "spread_pct": (
                    float((np.max(cl_arr) - np.min(cl_arr)) / abs(cl_mean) * 100)
                    if abs(cl_mean) > 1e-10 else 0.0
                ),
                "values": cl_vals,
            },
            "CD": {
                "min": float(np.min(cd_arr)),
                "max": float(np.max(cd_arr)),
                "mean": cd_mean,
                "std": float(np.std(cd_arr)),
                "spread": float(np.max(cd_arr) - np.min(cd_arr)),
                "spread_pct": float(
                    (np.max(cd_arr) - np.min(cd_arr)) / cd_mean * 100
                ) if cd_mean > 1e-10 else 0.0,
                "values": cd_vals,
            },
            "n_codes": len(codes),
            "codes": codes,
        }

    return stats


def assess_su2_vs_ensemble(
    ensemble_stats: Dict,
    su2_results: Dict[str, Dict[str, float]],
) -> Dict:
    """
    Assess SU2 results against the TMR 7-code ensemble band.

    Parameters
    ----------
    ensemble_stats : dict
        From compute_ensemble_statistics().
    su2_results : dict
        {alpha: {CL: ..., CD: ...}} for SU2 results.
        Alphas as strings or floats: "0", "10", "15".

    Returns
    -------
    dict with assessment per alpha and quantity.
    """
    assessment = {}

    for alpha_key, stats in ensemble_stats.items():
        alpha = stats["alpha"]
        alpha_s = str(int(alpha))

        su2 = su2_results.get(alpha_s) or su2_results.get(alpha) or {}
        if not su2:
            continue

        entry = {"alpha": alpha}
        for qty in ["CL", "CD"]:
            su2_val = su2.get(qty)
            if su2_val is None:
                continue

            s = stats[qty]
            dev_from_mean = su2_val - s["mean"]
            dev_pct = (
                abs(dev_from_mean) / abs(s["mean"]) * 100
                if abs(s["mean"]) > 1e-10 else 0.0
            )

            within_band = s["min"] <= su2_val <= s["max"]

            entry[qty] = {
                "su2_value": su2_val,
                "ensemble_mean": s["mean"],
                "ensemble_min": s["min"],
                "ensemble_max": s["max"],
                "ensemble_spread_pct": s["spread_pct"],
                "deviation_from_mean": dev_from_mean,
                "deviation_pct": dev_pct,
                "within_ensemble_band": within_band,
                "interpretation": (
                    f"SU2 SA {'is WITHIN' if within_band else 'is OUTSIDE'} "
                    f"the 7-code ensemble band "
                    f"(deviation: {dev_pct:.2f}% from mean, "
                    f"ensemble spread: +/-{s['spread_pct']/2:.2f}%)"
                ),
            }

        assessment[alpha_key] = entry

    return assessment


# ===================================================================
# Wall Hump -- Literature Multi-Model Comparison
# ===================================================================

# Published results from CFDVAL2004 workshop and subsequent studies.
# Sources: Rumsey et al. (2006), Greenblatt et al. (2006), TMR,
#          various AIAA papers, COMSOL verification study
HUMP_LITERATURE = {
    "experimental": {
        "x_sep": 0.665,
        "x_reat": 1.10,
        "bubble_length": 0.435,
        "cf_min": -0.00146,  # Greenblatt et al. (2006)
        "source": "Greenblatt et al. (2006)",
    },
    "models": {
        "SA (CFL3D, TMR)": {
            "x_sep": 0.665,
            "x_reat": 1.27,
            "bubble_length": 0.605,
            "source": "TMR / Rumsey (2006)",
            "notes": "Over-predicts bubble length by ~39% vs experiment",
        },
        "SA (FUN3D)": {
            "x_sep": 0.666,
            "x_reat": 1.27,
            "bubble_length": 0.604,
            "source": "TMR",
            "notes": "Similar to CFL3D SA",
        },
        "SST (CFL3D)": {
            "x_sep": 0.660,
            "x_reat": 1.24,
            "bubble_length": 0.580,
            "source": "Rumsey (2006)",
            "notes": "Slightly shorter bubble than SA",
        },
        "v2-f (various)": {
            "x_sep": 0.660,
            "x_reat": 1.18,
            "bubble_length": 0.520,
            "source": "CFDVAL2004 Workshop",
            "notes": "Closest to experiment among linear models",
        },
        "RSM (Lien-Leschziner)": {
            "x_sep": 0.665,
            "x_reat": 1.15,
            "bubble_length": 0.485,
            "source": "CFDVAL2004 Workshop",
            "notes": "Reynolds stress model, improved reattachment",
        },
    },
}


def compute_hump_model_comparison(
    su2_results: Optional[Dict] = None,
) -> Dict:
    """
    Frame SU2 wall hump results within literature multi-model comparison.

    Parameters
    ----------
    su2_results : dict, optional
        {x_sep: ..., x_reat: ..., bubble_length: ..., cf_min: ...}
        If None, uses only literature data.

    Returns
    -------
    dict with comparison table and interpretation.
    """
    exp = HUMP_LITERATURE["experimental"]
    models = HUMP_LITERATURE["models"]

    comparison = {
        "experimental": exp,
        "models": {},
    }

    # Compute errors for literature models
    for name, data in models.items():
        entry = dict(data)
        if data.get("bubble_length") and exp["bubble_length"] > 0:
            entry["bubble_error_pct"] = (
                (data["bubble_length"] - exp["bubble_length"])
                / exp["bubble_length"] * 100
            )
        if data.get("x_reat"):
            entry["reat_error"] = data["x_reat"] - exp["x_reat"]
        comparison["models"][name] = entry

    # Add SU2 results if provided
    if su2_results:
        su2_entry = dict(su2_results)
        su2_entry["source"] = "This study (SU2)"
        if su2_results.get("bubble_length") and exp["bubble_length"] > 0:
            su2_entry["bubble_error_pct"] = (
                (su2_results["bubble_length"] - exp["bubble_length"])
                / exp["bubble_length"] * 100
            )
        if su2_results.get("x_reat"):
            su2_entry["reat_error"] = su2_results["x_reat"] - exp["x_reat"]
        comparison["models"]["SU2 SA (this study)"] = su2_entry

    # Literature consensus
    all_bubbles = [m.get("bubble_length", 0) for m in models.values() if m.get("bubble_length")]
    comparison["consensus"] = {
        "all_overpredict_bubble": all(b > exp["bubble_length"] for b in all_bubbles),
        "bubble_range": [min(all_bubbles), max(all_bubbles)],
        "bubble_mean": float(np.mean(all_bubbles)),
        "separation_well_captured": all(
            abs(m.get("x_sep", 0) - exp["x_sep"]) < 0.01
            for m in models.values() if m.get("x_sep")
        ),
        "interpretation": (
            "All RANS models consistently over-predict the separation bubble length "
            "and delay reattachment. Separation location is well-captured across models "
            "(within +/-0.005c), confirming it is primarily governed by the adverse "
            "pressure gradient geometry. Reattachment is controlled by the turbulence "
            "model's ability to predict the separated shear layer development, "
            "explaining the persistent RANS deficiency."
        ),
    }

    return comparison


# ===================================================================
# Formatted Output
# ===================================================================
def print_naca_ensemble_report(
    ensemble_stats: Dict,
    assessment: Optional[Dict] = None,
) -> str:
    """Print formatted NACA 0012 ensemble uncertainty report."""
    lines = []
    lines.append("=" * 80)
    lines.append("  NACA 0012 -- TMR 7-CODE SA MODEL-FORM UNCERTAINTY ANALYSIS")
    lines.append("=" * 80)
    lines.append(f"\n  Codes: CFL3D, FUN3D, NTS, JOE, SUMB, TURNS, GGNS")
    lines.append(f"  Grid: 897x257 (finest Family I)")
    lines.append(f"  Conditions: Ma=0.15, Re=6e6, fully turbulent\n")

    # Ensemble statistics table
    lines.append(f"  {'Alpha':>6s}  {'Qty':>4s}  {'Min':>10s}  {'Max':>10s}  "
                 f"{'Mean':>10s}  {'Std':>10s}  {'Spread%':>8s}")
    lines.append("  " + "-" * 68)

    for alpha_key in ["alpha_0", "alpha_10", "alpha_15"]:
        s = ensemble_stats[alpha_key]
        alpha = s["alpha"]
        for qty in ["CL", "CD"]:
            q = s[qty]
            fmt = ".6f" if qty == "CL" else ".8f"
            lines.append(
                f"  {alpha:6.1f}  {qty:>4s}  {q['min']:{fmt}}  {q['max']:{fmt}}  "
                f"{q['mean']:{fmt}}  {q['std']:{fmt}}  {q['spread_pct']:7.2f}%"
            )

    # SU2 assessment
    if assessment:
        lines.append(f"\n  SU2 SA Assessment:")
        lines.append(f"  {'Alpha':>6s}  {'Qty':>4s}  {'SU2':>10s}  {'Mean':>10s}  "
                     f"{'Dev%':>8s}  {'In Band?':>10s}")
        lines.append("  " + "-" * 55)
        for alpha_key in ["alpha_0", "alpha_10", "alpha_15"]:
            if alpha_key not in assessment:
                continue
            a = assessment[alpha_key]
            alpha = a["alpha"]
            for qty in ["CL", "CD"]:
                if qty not in a:
                    continue
                q = a[qty]
                fmt = ".6f" if qty == "CL" else ".8f"
                band = "YES" if q["within_ensemble_band"] else "NO"
                lines.append(
                    f"  {alpha:6.1f}  {qty:>4s}  {q['su2_value']:{fmt}}  "
                    f"{q['ensemble_mean']:{fmt}}  {q['deviation_pct']:7.2f}%  "
                    f"{band:>10s}"
                )

    lines.append("\n" + "=" * 80)
    report = "\n".join(lines)
    print(report)
    return report


def print_hump_comparison_report(comparison: Dict) -> str:
    """Print formatted wall hump multi-model comparison."""
    lines = []
    lines.append("=" * 85)
    lines.append("  WALL HUMP -- MULTI-MODEL RANS COMPARISON (Model-Form Uncertainty)")
    lines.append("=" * 85)

    exp = comparison["experimental"]
    lines.append(f"\n  Experimental: x_sep={exp['x_sep']:.3f}  x_reat={exp['x_reat']:.3f}  "
                 f"L_bubble={exp['bubble_length']:.3f}c")
    lines.append(f"  Source: {exp['source']}")

    lines.append(f"\n  {'Model':<28s} {'x_sep':>7s} {'x_reat':>7s} {'L_bubble':>9s} "
                 f"{'dL (%)':<8s} {'Source':>25s}")
    lines.append("  " + "-" * 84)

    # Experimental row
    lines.append(f"  {'Experiment':<28s} {exp['x_sep']:7.3f} {exp['x_reat']:7.3f} "
                 f"{exp['bubble_length']:9.3f} {'---':>8s} "
                 f"{'Greenblatt (2006)':>25s}")

    # Model rows
    for name, data in comparison["models"].items():
        x_s = f"{data.get('x_sep', 0):.3f}" if data.get('x_sep') else "---"
        x_r = f"{data.get('x_reat', 0):.3f}" if data.get('x_reat') else "---"
        bl = f"{data.get('bubble_length', 0):.3f}" if data.get('bubble_length') else "---"
        dl = f"{data.get('bubble_error_pct', 0):+.1f}" if 'bubble_error_pct' in data else "---"
        src = data.get("source", "")[:25]
        lines.append(f"  {name:<28s} {x_s:>7s} {x_r:>7s} {bl:>9s} {dl:>8s} {src:>25s}")

    # Consensus
    con = comparison.get("consensus", {})
    if con:
        lines.append(f"\n  Consensus:")
        lines.append(f"    All RANS models over-predict bubble: "
                     f"{'YES' if con.get('all_overpredict_bubble') else 'NO'}")
        lines.append(f"    Separation well-captured (+/-0.005c):  "
                     f"{'YES' if con.get('separation_well_captured') else 'NO'}")
        if con.get("bubble_range"):
            lines.append(f"    RANS bubble length range: "
                         f"{con['bubble_range'][0]:.3f}-{con['bubble_range'][1]:.3f}c "
                         f"(exp: {exp['bubble_length']:.3f}c)")
        if con.get("interpretation"):
            # Word-wrap interpretation
            interp = con["interpretation"]
            words = interp.split()
            current_line = "    "
            for w in words:
                if len(current_line) + len(w) + 1 > 85:
                    lines.append(current_line)
                    current_line = "    " + w
                else:
                    current_line += " " + w if current_line.strip() else "    " + w
            if current_line.strip():
                lines.append(current_line)

    lines.append("\n" + "=" * 85)
    report = "\n".join(lines)
    print(report)
    return report


# ===================================================================
# CLI Entry Point
# ===================================================================
def main():
    """Run model-form uncertainty analysis for both cases."""
    print("\n" + "=" * 80)
    print("             MODEL-FORM UNCERTAINTY ANALYSIS")
    print("=" * 80)

    # 1. NACA 0012 TMR ensemble
    print("\n\n[1] NACA 0012 -- TMR 7-Code Ensemble\n")
    try:
        tmr = load_tmr_ensemble()
        stats = compute_ensemble_statistics(tmr)

        # Try to load SU2 results if available
        su2_results = {}
        results_dir = PROJECT / "runs" / "naca0012"
        forces_file = results_dir / "force_summary.json"
        if forces_file.exists():
            with open(forces_file) as f:
                su2_results = json.load(f)

        assessment = None
        if su2_results:
            assessment = assess_su2_vs_ensemble(stats, su2_results)

        print_naca_ensemble_report(stats, assessment)

    except FileNotFoundError as e:
        print(f"  {e}")
    except Exception as e:
        print(f"  Error: {e}")

    # 2. Wall hump multi-model comparison
    print("\n\n[2] Wall Hump -- Multi-Model RANS Comparison\n")

    # Try to load SU2 wall hump results
    su2_hump = None
    gci_file = PROJECT / "plots" / "wall_hump" / "gci_asme_results.json"
    if gci_file.exists():
        with open(gci_file) as f:
            gci_data = json.load(f)
        fine = gci_data.get("grids", {}).get("fine", {})
        if fine.get("x_sep") and fine.get("x_reat"):
            su2_hump = {
                "x_sep": fine["x_sep"],
                "x_reat": fine["x_reat"],
                "bubble_length": fine.get("bubble_length",
                                          fine["x_reat"] - fine["x_sep"]),
                "cf_min": fine.get("cf_min"),
            }

    comparison = compute_hump_model_comparison(su2_hump)
    print_hump_comparison_report(comparison)

    # Save results
    output_dir = PROJECT / "results" / "model_form_uncertainty"
    output_dir.mkdir(parents=True, exist_ok=True)

    output = {
        "naca0012_ensemble": {},
        "wall_hump_comparison": comparison,
    }

    try:
        tmr = load_tmr_ensemble()
        output["naca0012_ensemble"]["statistics"] = compute_ensemble_statistics(tmr)
    except Exception:
        pass

    def _serialise(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    out_file = output_dir / "model_form_analysis.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2, default=_serialise)
    print(f"\nResults saved to: {out_file}")


if __name__ == "__main__":
    main()
