#!/usr/bin/env python3
"""
CFD Results Validation Script
==============================
Automated comparison of CFD results against experimental and analytical data.

This script:
1. Loads CFD simulation results
2. Loads reference experimental/analytical data
3. Computes validation metrics (RMSE, MAPE, ASME V&V 20)
4. Generates comparison plots
5. Produces validation report

Usage:
    python validate_results.py --case backward_facing_step --model SA --grid medium
    python validate_results.py --case nasa_hump --model SST --grid fine --output report.pdf
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless operation
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import json

# Force UTF-8 output on Windows terminals.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from experimental_data.data_loader import load_case
from scripts.postprocessing.error_metrics import (
    rmse, mae, nrmse, mape, correlation_coefficient,
    asme_vv20_metric, compute_all_metrics
)
from scripts.postprocessing.extract_profiles import (
    find_separation_point,
    find_reattachment_point
)

# =============================================================================
# Validation Thresholds (from NASA 40% Challenge)
# =============================================================================
VALIDATION_LEVELS = {
    "excellent": {"E": 0.05, "color": "green", "symbol": "✓✓"},
    "good": {"E": 0.20, "color": "blue", "symbol": "✓"},
    "adequate": {"E": 0.40, "color": "orange", "symbol": "~"},
    "poor": {"E": float('inf'), "color": "red", "symbol": "✗"}
}


CASE_ALIASES = {
    "wall_hump": "nasa_hump",
    "hump": "nasa_hump",
    "bfs": "backward_facing_step",
    "naca0012": "naca_0012_stall",
    "gaussian_bump": "boeing_gaussian_bump",
}


def normalize_case_name(case_name: str) -> str:
    """Map common project aliases to canonical experimental-data loader keys."""
    key = case_name.strip()
    return CASE_ALIASES.get(key, key)


def get_validation_level(error_metric):
    """Classify validation quality based on error metric."""
    for level, criteria in VALIDATION_LEVELS.items():
        if error_metric < criteria["E"]:
            return level, criteria
    return "poor", VALIDATION_LEVELS["poor"]


# =============================================================================
# Main Validation Class
# =============================================================================
class CFDValidator:
    """Comprehensive CFD validation against reference data."""

    def __init__(self, case_name: str, model_name: str, grid_level: str,
                 results_dir: Path):
        self.case_name = case_name
        self.case_key = normalize_case_name(case_name)
        self.model_name = model_name
        self.grid_level = grid_level
        self.results_dir = Path(results_dir)

        # Load experimental data (uses experimental_data.data_loader API)
        # Returns ExperimentalData with .wall_data (DataFrame), .separation_metrics (dict),
        # .velocity_profiles (dict of DataFrames)
        self.exp_data = load_case(self.case_key)

        # Initialize results storage
        self.validation_results = {
            "case": case_name,
            "case_key": self.case_key,
            "model": model_name,
            "grid": grid_level,
            "metrics": {},
            "separation": {},
            "profiles": {},
            "overall_status": "PENDING",
            "is_synthetic_data": getattr(self.exp_data, 'is_synthetic', False)
        }

    def load_cfd_wall_data(self):
        """Load CFD wall pressure and friction coefficients."""
        wall_file = self.results_dir / "postProcessing" / "wall_quantities.csv"

        if not wall_file.exists():
            raise FileNotFoundError(
                f"Wall data not found: {wall_file}. "
                "Generate postProcessing outputs before validation."
            )

        return pd.read_csv(wall_file)

    def load_cfd_profiles(self, station: float):
        """Load CFD velocity profile at specified x-station."""
        profile_file = self.results_dir / f"postProcessing/profiles/profile_x{station:.3f}.csv"

        if not profile_file.exists():
            raise FileNotFoundError(
                f"Profile not found: {profile_file}. "
                "Generate profile CSV files in postProcessing/profiles."
            )

        return pd.read_csv(profile_file)

    def validate_separation_bubble(self):
        """Validate separation and reattachment locations."""
        print("\n" + "="*80)
        print("  SEPARATION BUBBLE VALIDATION")
        print("="*80)

        try:
            cfd_wall = self.load_cfd_wall_data()

            # Extract Cf distribution
            x_cfd = cfd_wall['x'].values
            cf_cfd = cfd_wall['Cf'].values

            # Find separation and reattachment
            x_sep_cfd = find_separation_point(x_cfd, cf_cfd)
            x_reat_cfd = find_reattachment_point(x_cfd, cf_cfd)

            # Get experimental references from separation_metrics dict
            x_sep_exp = self.exp_data.separation_metrics.get('x_sep_xH') or \
                       self.exp_data.separation_metrics.get('x_sep_xc')
            x_reat_exp = self.exp_data.separation_metrics.get('x_reat_xH') or \
                        self.exp_data.separation_metrics.get('x_reat_xc')

            results = {}

            if x_sep_cfd is not None and x_sep_exp is not None:
                sep_error = abs(x_sep_cfd - x_sep_exp) / x_sep_exp * 100
                results['x_separation'] = {
                    'cfd': x_sep_cfd,
                    'exp': x_sep_exp,
                    'error_pct': sep_error,
                    'status': 'PASS' if sep_error < 10 else 'FAIL'
                }
                print(f"✓ Separation:    CFD={x_sep_cfd:.3f}, EXP={x_sep_exp:.3f}, "
                      f"Error={sep_error:.1f}%")
            else:
                print("⚠ Separation point detection failed")

            if x_reat_cfd is not None and x_reat_exp is not None:
                reat_error = abs(x_reat_cfd - x_reat_exp) / x_reat_exp * 100
                results['x_reattachment'] = {
                    'cfd': x_reat_cfd,
                    'exp': x_reat_exp,
                    'error_pct': reat_error,
                    'status': 'PASS' if reat_error < 10 else 'FAIL'
                }
                print(f"✓ Reattachment:  CFD={x_reat_cfd:.3f}, EXP={x_reat_exp:.3f}, "
                      f"Error={reat_error:.1f}%")

                # Bubble length
                if x_sep_cfd is not None:
                    bubble_cfd = x_reat_cfd - x_sep_cfd
                    bubble_exp = x_reat_exp - x_sep_exp if x_sep_exp else None

                    if bubble_exp:
                        bubble_error = abs(bubble_cfd - bubble_exp) / bubble_exp * 100
                        results['bubble_length'] = {
                            'cfd': bubble_cfd,
                            'exp': bubble_exp,
                            'error_pct': bubble_error,
                            'status': 'PASS' if bubble_error < 15 else 'FAIL'
                        }
                        print(f"✓ Bubble length: CFD={bubble_cfd:.3f}, EXP={bubble_exp:.3f}, "
                              f"Error={bubble_error:.1f}%")
            else:
                print("⚠ Reattachment point detection failed")

            self.validation_results['separation'] = results
            return results

        except Exception as e:
            print(f"✗ Separation validation failed: {e}")
            return {}

    def validate_wall_quantities(self):
        """Validate wall Cp and Cf distributions."""
        print("\n" + "="*80)
        print("  WALL QUANTITIES VALIDATION")
        print("="*80)

        try:
            cfd_wall = self.load_cfd_wall_data()

            results = {}

            # Pressure coefficient validation
            if self.exp_data.wall_data is not None and 'Cp' in self.exp_data.wall_data.columns:
                x_exp = self.exp_data.wall_data['x'].values
                cp_exp = self.exp_data.wall_data['Cp'].values

                # Interpolate CFD to experimental stations
                cp_cfd_interp = np.interp(x_exp, cfd_wall['x'], cfd_wall['Cp'])

                # Compute metrics (uses lowercase aliases from compute_all_metrics)
                metrics = compute_all_metrics(cp_cfd_interp, cp_exp)

                level, criteria = get_validation_level(metrics['mape'] / 100)

                results['Cp'] = {
                    'rmse': metrics['rmse'],
                    'mae': metrics['mae'],
                    'mape': metrics['mape'],
                    'r2': metrics['r_squared'],
                    'level': level,
                    'status': 'PASS' if level in ['excellent', 'good'] else 'WARN'
                }

                print(f"{criteria['symbol']} Cp:  RMSE={metrics['rmse']:.4f}, "
                      f"MAPE={metrics['mape']:.2f}%, R²={metrics['r_squared']:.4f} [{level.upper()}]")

            # Skin friction coefficient validation
            if self.exp_data.wall_data is not None and 'Cf' in self.exp_data.wall_data.columns:
                x_exp = self.exp_data.wall_data['x'].values
                cf_exp = self.exp_data.wall_data['Cf'].values

                cf_cfd_interp = np.interp(x_exp, cfd_wall['x'], cfd_wall['Cf'])

                metrics = compute_all_metrics(cf_cfd_interp, cf_exp)
                level, criteria = get_validation_level(metrics['mape'] / 100)

                results['Cf'] = {
                    'rmse': metrics['rmse'],
                    'mae': metrics['mae'],
                    'mape': metrics['mape'],
                    'r2': metrics['r_squared'],
                    'level': level,
                    'status': 'PASS' if level in ['excellent', 'good'] else 'WARN'
                }

                print(f"{criteria['symbol']} Cf:  RMSE={metrics['rmse']:.6f}, "
                      f"MAPE={metrics['mape']:.2f}%, R²={metrics['r_squared']:.4f} [{level.upper()}]")

            self.validation_results['metrics'] = results
            return results

        except Exception as e:
            print(f"✗ Wall quantities validation failed: {e}")
            return {}

    def validate_velocity_profiles(self, stations=None):
        """Validate velocity profiles at measurement stations."""
        print("\n" + "="*80)
        print("  VELOCITY PROFILE VALIDATION")
        print("="*80)

        if not self.exp_data.velocity_profiles:
            print("⚠ No experimental velocity profiles available")
            return {}

        results = {}

        # Use all available stations if not specified
        if stations is None:
            stations = list(self.exp_data.velocity_profiles.keys())

        for station in stations:
            if station not in self.exp_data.velocity_profiles:
                continue

            try:
                # Load experimental profile
                exp_profile = self.exp_data.velocity_profiles[station]

                # Load CFD profile
                cfd_profile = self.load_cfd_profiles(float(station))

                # Extract U velocity
                y_exp = exp_profile['y'].values
                u_exp = exp_profile['U'].values

                y_cfd = cfd_profile['y'].values
                u_cfd = cfd_profile['U'].values

                # Interpolate to common y-coordinates
                u_cfd_interp = np.interp(y_exp, y_cfd, u_cfd)

                # Compute metrics
                metrics = compute_all_metrics(u_cfd_interp, u_exp)
                level, criteria = get_validation_level(metrics['mape'] / 100)

                results[f'x_{station}'] = {
                    'rmse': metrics['rmse'],
                    'mape': metrics['mape'],
                    'r2': metrics['r_squared'],
                    'level': level
                }

                print(f"{criteria['symbol']} x={station}:  RMSE={metrics['rmse']:.4f}, "
                      f"MAPE={metrics['mape']:.2f}%, R²={metrics['r_squared']:.4f} [{level.upper()}]")

            except Exception as e:
                print(f"⚠ Profile at x={station} failed: {e}")

        self.validation_results['profiles'] = results
        return results

    def generate_validation_plots(self, output_file: Path):
        """Generate comprehensive validation plots."""
        print("\n" + "="*80)
        print("  GENERATING VALIDATION PLOTS")
        print("="*80)

        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

        try:
            cfd_wall = self.load_cfd_wall_data()

            # 1. Pressure coefficient
            ax1 = fig.add_subplot(gs[0, 0])
            if self.exp_data.wall_data is not None and 'Cp' in self.exp_data.wall_data.columns:
                ax1.plot(self.exp_data.wall_data['x'], self.exp_data.wall_data['Cp'],
                        'ko', label='Experiment', markersize=4)
            ax1.plot(cfd_wall['x'], cfd_wall['Cp'], 'r-', label=f'CFD ({self.model_name})', linewidth=2)
            ax1.set_xlabel('x/H or x/c')
            ax1.set_ylabel('C_p')
            ax1.set_title('Pressure Coefficient')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # 2. Skin friction coefficient
            ax2 = fig.add_subplot(gs[0, 1])
            if self.exp_data.wall_data is not None and 'Cf' in self.exp_data.wall_data.columns:
                ax2.plot(self.exp_data.wall_data['x'], self.exp_data.wall_data['Cf'],
                        'ko', label='Experiment', markersize=4)
            ax2.plot(cfd_wall['x'], cfd_wall['Cf'], 'b-', label=f'CFD ({self.model_name})', linewidth=2)
            ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
            ax2.set_xlabel('x/H or x/c')
            ax2.set_ylabel('C_f')
            ax2.set_title('Skin Friction Coefficient')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            # 3-6. Velocity profiles at key stations
            profile_axes = [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]),
                           fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])]

            stations = list(self.exp_data.velocity_profiles.keys())[:4]

            for ax, station in zip(profile_axes, stations):
                exp_profile = self.exp_data.velocity_profiles[station]

                try:
                    cfd_profile = self.load_cfd_profiles(float(station))

                    ax.plot(exp_profile['U'], exp_profile['y'],
                           'ko', label='Experiment', markersize=3)
                    ax.plot(cfd_profile['U'], cfd_profile['y'],
                           'r-', label='CFD', linewidth=2)
                    ax.set_xlabel('U/U_inf')
                    ax.set_ylabel('y/H or y/c')
                    ax.set_title(f'Velocity Profile at x={station}')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                except Exception:
                    ax.text(0.5, 0.5, f'Profile at x={station}\nnot available',
                           ha='center', va='center', transform=ax.transAxes)

            plt.suptitle(f'{self.case_name} - {self.model_name} - {self.grid_level}',
                        fontsize=16, fontweight='bold')

            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"✓ Plots saved to: {output_file}")
            plt.close()

        except Exception as e:
            print(f"✗ Plot generation failed: {e}")

    def generate_report(self, output_file: Path):
        """Generate JSON validation report."""
        with open(output_file, 'w') as f:
            json.dump(self.validation_results, f, indent=2)

        print(f"\n  Report saved to: {output_file}")

    def _collect_gci(self):
        """Attempt to load GCI results if available."""
        gci_file = PROJECT_ROOT / "plots" / "wall_hump" / "gci_asme_results.json"
        if gci_file.exists():
            try:
                with open(gci_file) as f:
                    gci_data = json.load(f)
                self.validation_results["gci"] = gci_data.get("gci", {})
                self.validation_results["asme_vv20"] = gci_data.get("asme_vv20", {})
                return gci_data
            except Exception:
                pass
        return None

    def emit_json_summary(self, output_file: Path):
        """
        Emit comprehensive JSON summary with all metrics and pass/fail flags.

        This is the primary machine-readable output for CI/CD integration.
        """
        summary = {
            "case": self.case_name,
            "model": self.model_name,
            "grid": self.grid_level,
            "timestamp": __import__("datetime").datetime.now().isoformat(),
            "overall_status": self.validation_results.get("overall_status", "PENDING"),
            "separation": self.validation_results.get("separation", {}),
            "wall_metrics": self.validation_results.get("metrics", {}),
            "profiles": self.validation_results.get("profiles", {}),
            "gci": self.validation_results.get("gci", {}),
            "asme_vv20": self.validation_results.get("asme_vv20", {}),
            "pass_fail": {},
        }

        # Compute pass/fail flags
        tol = {"sep_error_pct": 10, "reat_error_pct": 10, "bubble_error_pct": 15,
               "cp_mape": 20, "cf_mape": 40}
        sep = summary["separation"]
        if sep.get("x_separation"):
            summary["pass_fail"]["x_sep"] = sep["x_separation"]["error_pct"] < tol["sep_error_pct"]
        if sep.get("x_reattachment"):
            summary["pass_fail"]["x_reat"] = sep["x_reattachment"]["error_pct"] < tol["reat_error_pct"]
        if sep.get("bubble_length"):
            summary["pass_fail"]["bubble"] = sep["bubble_length"]["error_pct"] < tol["bubble_error_pct"]
        wall = summary["wall_metrics"]
        if wall.get("Cp"):
            summary["pass_fail"]["Cp_mape"] = wall["Cp"]["mape"] < tol["cp_mape"]
        if wall.get("Cf"):
            summary["pass_fail"]["Cf_mape"] = wall["Cf"]["mape"] < tol["cf_mape"]

        with open(output_file, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  JSON summary: {output_file}")
        return summary

    def emit_markdown_report(self, output_file: Path):
        """
        Generate compact Markdown validation report with tables.

        Follows SU2 V&V repository conventions.
        """
        r = self.validation_results
        lines = []
        lines.append(f"# Validation Report: {self.case_name}")
        lines.append(f"\n**Model**: {self.model_name} | "
                     f"**Grid**: {self.grid_level} | "
                     f"**Status**: {r.get('overall_status', 'PENDING')}")
        
        if r.get('is_synthetic_data', False):
            lines.append("\n> [!WARNING]")
            lines.append("> **SYNTHETIC DATA DEMONSTRATION**")
            lines.append("> The reference data used for this validation is generated from")
            lines.append("> analytical formulas (np.tanh/np.sin), NOT experimental measurements.")
            lines.append("> This report demonstrates the validation pipeline functionality only")
            lines.append("> and MUST NOT be used for turbulence model validation claims.")
        
        lines.append("")

        # Separation metrics table
        sep = r.get("separation", {})
        if sep:
            lines.append("## Separation Bubble")
            lines.append("")
            lines.append("| Metric | CFD | Experiment | Error (%) | Pass? |")
            lines.append("|--------|:---:|:----------:|:---------:|:-----:|")
            for key in ["x_separation", "x_reattachment", "bubble_length"]:
                if key in sep:
                    s = sep[key]
                    pf = "PASS" if s.get("status") == "PASS" else "FAIL"
                    lines.append(f"| {key} | {s['cfd']:.4f} | {s['exp']:.4f} | "
                                 f"{s['error_pct']:.1f} | {pf} |")
            lines.append("")

        # Wall quantities table
        wall = r.get("metrics", {})
        if wall:
            lines.append("## Wall Quantities")
            lines.append("")
            lines.append("| Quantity | RMSE | MAE | MAPE (%) | R^2 | Level |")
            lines.append("|----------|:----:|:---:|:--------:|:---:|:-----:|")
            for key in ["Cp", "Cf"]:
                if key in wall:
                    w = wall[key]
                    lines.append(f"| {key} | {w['rmse']:.5f} | {w['mae']:.5f} | "
                                 f"{w['mape']:.2f} | {w['r2']:.4f} | "
                                 f"{w.get('level', '?').upper()} |")
            lines.append("")

        # Profile validation
        prof = r.get("profiles", {})
        if prof:
            lines.append("## Velocity Profiles")
            lines.append("")
            lines.append("| Station | RMSE | MAPE (%) | R^2 | Level |")
            lines.append("|---------|:----:|:--------:|:---:|:-----:|")
            for key, p in prof.items():
                lines.append(f"| {key} | {p['rmse']:.4f} | {p['mape']:.2f} | "
                             f"{p['r2']:.4f} | {p.get('level', '?').upper()} |")
            lines.append("")

        # GCI
        gci = r.get("gci", {})
        if gci:
            lines.append("## Grid Convergence Index")
            lines.append("")
            lines.append("| Quantity | p_obs | GCI_fine (%) | Asymptotic? |")
            lines.append("|----------|:-----:|:------------:|:-----------:|")
            for qty, g in gci.items():
                p_s = f"{g.get('p', 0):.3f}" if g.get("convergence") == "monotonic" else "OSC"
                gci_s = f"{g.get('gci_fine_pct', 0):.2f}" if g.get("gci_fine_pct") else "---"
                ar = g.get("asym_ratio", 0)
                ar_s = "YES" if 0.9 <= ar <= 1.1 else "NO" if ar else "---"
                lines.append(f"| {qty} | {p_s} | {gci_s} | {ar_s} |")
            lines.append("")

        content = "\n".join(lines)
        output_file.write_text(content)
        print(f"  Markdown report: {output_file}")
        return content

    def emit_csv_summary(self, output_file: Path):
        """
        Emit one-line CSV summary for batch processing.

        Columns: case, model, grid, status, cp_rmse, cf_rmse, cp_mape,
                 cf_mape, x_sep_err, x_reat_err, bubble_err
        """
        r = self.validation_results
        sep = r.get("separation", {})
        wall = r.get("metrics", {})

        row = {
            "case": self.case_name,
            "model": self.model_name,
            "grid": self.grid_level,
            "status": r.get("overall_status", "PENDING"),
            "cp_rmse": wall.get("Cp", {}).get("rmse", ""),
            "cf_rmse": wall.get("Cf", {}).get("rmse", ""),
            "cp_mape": wall.get("Cp", {}).get("mape", ""),
            "cf_mape": wall.get("Cf", {}).get("mape", ""),
            "x_sep_err_pct": sep.get("x_separation", {}).get("error_pct", ""),
            "x_reat_err_pct": sep.get("x_reattachment", {}).get("error_pct", ""),
            "bubble_err_pct": sep.get("bubble_length", {}).get("error_pct", ""),
        }

        import csv as csv_mod
        write_header = not output_file.exists()
        with open(output_file, "a", newline="") as f:
            writer = csv_mod.DictWriter(f, fieldnames=row.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(row)

        print(f"  CSV summary: {output_file}")
        return row

    def run_complete_validation(self, output_dir: Path):
        """Run all validation checks and generate outputs."""
        print("\n" + "="*80)
        print(f"  CFD VALIDATION: {self.case_name} | {self.model_name} | {self.grid_level}")
        print("="*80)

        # Run validation checks
        self.validate_separation_bubble()
        self.validate_wall_quantities()
        self.validate_velocity_profiles()

        # Collect GCI if available
        self._collect_gci()

        # Determine overall status
        sep_entries = self.validation_results['separation']
        wall_entries = self.validation_results['metrics']
        has_sep = bool(sep_entries)
        has_wall = bool(wall_entries)

        sep_ok = has_sep and all(v.get('status') == 'PASS' for v in sep_entries.values())
        wall_ok = has_wall and all(v.get('status') in ['PASS', 'WARN'] for v in wall_entries.values())

        if self.exp_data.is_synthetic:
            self.validation_results['overall_status'] = 'SYNTHETIC_DEMO'
            status_symbol = "DEMO"
            print("\n" + "!"*80)
            print("  ⚠️  WARNING: Validation performed against SYNTHETIC reference data.")
            print("  ⚠️  Results demonstrate pipeline functionality only and are NOT real validation.")
            print("!"*80)
        elif sep_ok and wall_ok:
            self.validation_results['overall_status'] = 'VALIDATED'
            status_symbol = "PASS"
        elif wall_ok and has_sep:
            self.validation_results['overall_status'] = 'ACCEPTABLE'
            status_symbol = "OK"
        elif not has_sep and not has_wall:
            self.validation_results['overall_status'] = 'INSUFFICIENT_DATA'
            status_symbol = "FAIL"
        else:
            self.validation_results['overall_status'] = 'ISSUES'
            status_symbol = "WARN"

        # Generate outputs
        output_dir.mkdir(parents=True, exist_ok=True)

        plot_file = output_dir / f"validation_{self.case_name}_{self.model_name}_{self.grid_level}.pdf"
        self.generate_validation_plots(plot_file)

        report_file = output_dir / f"validation_{self.case_name}_{self.model_name}_{self.grid_level}.json"
        self.generate_report(report_file)

        # New outputs: JSON summary, Markdown report, CSV line
        self.emit_json_summary(
            output_dir / f"summary_{self.case_name}_{self.model_name}_{self.grid_level}.json"
        )
        self.emit_markdown_report(
            output_dir / f"report_{self.case_name}_{self.model_name}_{self.grid_level}.md"
        )
        self.emit_csv_summary(output_dir / "validation_summary.csv")

        # Summary
        print("\n" + "="*80)
        print(f"  [{status_symbol}] OVERALL STATUS: {self.validation_results['overall_status']}")
        print("="*80)

        return self.validation_results





# =============================================================================
# BFS Reattachment Utility  (Driver & Seegmiller 1985)
# =============================================================================
def compute_bfs_reattachment(
    wall_csv: Path,
    x_col: str = "x",
    cf_col: str = "Cf",
    exp_xH_reat: float = 6.26,
    exp_uncertainty: float = 0.10,
    step_height: float = 1.0,
    verbose: bool = True,
) -> dict:
    """Compute the BFS reattachment x/H from a CFD Cf profile and compare to experiment.

    The Backward-Facing Step reference reattachment is x/H = 6.26 ± 0.10
    (Driver & Seegmiller 1985, AIAA J. 23(2):163-171).  Reattachment is
    identified as the first negative-to-positive zero crossing of Cf that
    occurs after an initial positive-to-negative (separation) crossing.

    Parameters
    ----------
    wall_csv : Path
        CSV file containing at minimum ``x`` (= x/H) and ``Cf`` columns.
    x_col, cf_col : str
        Column names for the x-coordinate and skin-friction coefficient.
    exp_xH_reat : float
        Experimental reattachment location (x/H).  Default: 6.26.
    exp_uncertainty : float
        Experimental uncertainty band (±).  Default: 0.10.
    step_height : float
        Step height H used to normalise x if the CSV stores dimensional x.
        Set to 1.0 (default) when x is already non-dimensionalised as x/H.
    verbose : bool
        Print a formatted comparison table.

    Returns
    -------
    dict with keys:
        cfd_xH_reat, exp_xH_reat, exp_uncertainty,
        error_pct, within_uncertainty, status, verdict
    """
    df = pd.read_csv(wall_csv)
    if x_col not in df.columns or cf_col not in df.columns:
        raise ValueError(
            f"wall_csv must contain columns '{x_col}' and '{cf_col}'. "
            f"Found: {list(df.columns)}"
        )

    x  = (df[x_col].values / step_height).astype(float)
    Cf = df[cf_col].values.astype(float)

    # Sort by x in case the CSV is unordered
    order = np.argsort(x)
    x, Cf = x[order], Cf[order]

    cfd_reat = find_reattachment_point(x, Cf)

    result: dict = {
        "cfd_xH_reat":        cfd_reat,
        "exp_xH_reat":        exp_xH_reat,
        "exp_uncertainty":    exp_uncertainty,
        "error_pct":          None,
        "within_uncertainty": None,
        "status":             "UNKNOWN",
        "verdict":            "Reattachment not found in Cf profile.",
    }

    if cfd_reat is not None:
        err_abs = cfd_reat - exp_xH_reat
        err_pct = abs(err_abs) / exp_xH_reat * 100.0
        within  = abs(err_abs) <= exp_uncertainty

        # Status thresholds
        if err_pct < 5.0:
            status, verdict = "EXCELLENT", "Within 5 % of Driver & Seegmiller reference."
        elif err_pct < 10.0:
            status, verdict = "GOOD",      "Within 10 % tolerance."
        elif err_pct < 20.0:
            status, verdict = "ACCEPTABLE","Moderate RANS-bias overshoot; acceptable for RANS."
        else:
            status, verdict = "FAIL",      ("Reattachment error > 20 %. Check inflow BL "
                                            "profile, mesh density near step, or turbulence model.")

        result.update(
            error_pct=err_pct,
            error_abs=err_abs,
            within_uncertainty=within,
            status=status,
            verdict=verdict,
        )

        if verbose:
            unc_flag = " ✓ within uncertainty band" if within else ""
            print("\n" + "=" * 70)
            print("  BFS REATTACHMENT — Driver & Seegmiller (1985) Comparison")
            print("=" * 70)
            print(f"  CFD x/H reattachment  : {cfd_reat:.3f}")
            print(f"  Experiment x/H        : {exp_xH_reat:.2f} ± {exp_uncertainty:.2f}{unc_flag}")
            print(f"  Absolute error        : {err_abs:+.3f} H")
            print(f"  Relative error        : {err_pct:.1f} %")
            print(f"  Status                : [{status}] {verdict}")
            print("=" * 70)
    elif verbose:
        print("\n[BFS] WARNING: Cf zero-crossing (reattachment) not detected in the CFD wall file.")
        print("       Check that the Cf profile covers the full downstream region to at least x/H = 10.")

    return result


# =============================================================================
# Command-Line Interface

# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Validate CFD results against experimental/analytical data'
    )
    parser.add_argument('--case', required=True,
                       help=('Case name (e.g., backward_facing_step, nasa_hump). '
                             'Common aliases are accepted, e.g. wall_hump, bfs, '
                             'naca0012, gaussian_bump.'))
    parser.add_argument('--model', required=True,
                       help='Turbulence model (e.g., SA, SST)')
    parser.add_argument('--grid', required=True,
                       help='Grid level (e.g., coarse, medium, fine)')
    parser.add_argument('--results-dir', type=Path, default=None,
                       help='CFD results directory (default: ./runs/CASE/MODEL_GRID)')
    parser.add_argument('--output', type=Path, default=Path('./validation_results'),
                       help='Output directory for validation results')

    args = parser.parse_args()

    # Construct results directory if not specified
    if args.results_dir is None:
        args.results_dir = Path(f"./runs/{args.case}/{args.model}_{args.grid}")

    if not args.results_dir.exists():
        print(f"Error: Results directory not found: {args.results_dir}")
        sys.exit(1)

    # Run validation
    validator = CFDValidator(
        case_name=args.case,
        model_name=args.model,
        grid_level=args.grid,
        results_dir=args.results_dir
    )

    results = validator.run_complete_validation(args.output)

    # Exit with appropriate code
    if results['overall_status'] == 'VALIDATED':
        sys.exit(0)
    elif results['overall_status'] == 'ACCEPTABLE':
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()
