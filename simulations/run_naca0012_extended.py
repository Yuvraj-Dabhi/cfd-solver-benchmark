#!/usr/bin/env python3
"""
Extended NACA 0012 Runs — Convergence Validation
===================================================
Restarts NACA 0012 α=10° and α=15° with 50k iterations for full
iterative convergence. Invokes ConvergenceChecker to verify CL/CD
have stabilized within 0.1%.

Usage
-----
    python run_naca0012_extended.py [--alpha 10 15] [--dry-run] [--check-only]
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


# CFL3D reference values for SA at various angles (from NASA TMR)
CFL3D_REFERENCE = {
    10: {"CL": 1.0912, "CD": 0.01222, "CDp": 0.00601, "CDv": 0.006205, "CM": 0.00678},
    15: {"CL": 1.5461, "CD": 0.02124, "CDp": 0.01225, "CDv": 0.00899, "CM": 0.0148},
}

CASE_DIR = PROJECT_ROOT / "runs" / "naca0012"


def generate_extended_config(
    alpha_deg: float,
    n_iter: int = 50000,
    restart: bool = True,
    model: str = "SA",
) -> Path:
    """Generate SU2 config for extended NACA 0012 run."""
    run_dir = CASE_DIR / f"naca0012_{model}_alpha{int(alpha_deg)}_extended"
    run_dir.mkdir(parents=True, exist_ok=True)

    aoa_rad = np.radians(alpha_deg)
    restart_flag = "YES" if restart else "NO"

    config = f"""\
% ============================================================
% NACA 0012 Extended Run — {model} at α={alpha_deg}°
% Target: 50k iterations for full convergence validation
% ============================================================

SOLVER= INC_RANS
KIND_TURB_MODEL= {model}
MATH_PROBLEM= DIRECT
RESTART_SOL= {restart_flag}

% --- Freestream ---
INC_DENSITY_MODEL= CONSTANT
INC_DENSITY_INIT= 1.225
INC_VELOCITY_INIT= ({51.4 * np.cos(aoa_rad):.6f}, {51.4 * np.sin(aoa_rad):.6f}, 0.0)
INC_TEMPERATURE_INIT= 300.0
VISCOSITY_MODEL= CONSTANT
MU_CONSTANT= 1.7894e-05
FREESTREAM_NU_FACTOR= 3.0

% --- Reference ---
REF_LENGTH= 1.0
REF_AREA= 1.0
REYNOLDS_NUMBER= 6000000.0
AOA= {alpha_deg}

% --- Boundary Conditions ---
MARKER_HEATFLUX= ( airfoil, 0.0 )
MARKER_FAR= ( farfield )
MARKER_MONITORING= ( airfoil )

% --- Numerics ---
NUM_METHOD_GRAD= GREEN_GAUSS
CFL_NUMBER= 25.0
CFL_ADAPT= YES
CFL_ADAPT_PARAM= ( 0.1, 2.0, 25.0, 1e10 )

CONV_NUM_METHOD_FLOW= FDS
MUSCL_FLOW= YES
SLOPE_LIMITER_FLOW= VENKATAKRISHNAN
VENKAT_LIMITER_COEFF= 0.03

CONV_NUM_METHOD_TURB= SCALAR_UPWIND
MUSCL_TURB= NO

% --- Convergence (strict NASA TMR target) ---
ITER= {n_iter}
CONV_RESIDUAL_MINVAL= -12
CONV_FIELD= RMS_PRESSURE, RMS_NU_TILDE
CONV_STARTITER= 100

% --- Output ---
MESH_FILENAME= mesh_897x257.su2
OUTPUT_FILES= RESTART, PARAVIEW_MULTIBLOCK, SURFACE_PARAVIEW
VOLUME_OUTPUT= RESIDUAL, PRIMITIVE, TURBULENT
SURFACE_OUTPUT= SKIN_FRICTION, PRESSURE_COEFFICIENT, HEAT_FLUX
OUTPUT_WRT_FREQ= 1000
CONV_FILENAME= history
RESTART_FILENAME= restart_flow.dat
SOLUTION_FILENAME= solution_flow.dat
HISTORY_OUTPUT= ITER, RMS_RES, AERO_COEFF, LINSOL
"""

    config_path = run_dir / f"naca0012_extended_{model}_a{int(alpha_deg)}.cfg"
    config_path.write_text(config)
    print(f"Generated config: {config_path}")
    return config_path


def check_convergence(alpha_deg: float, model: str = "SA"):
    """Run convergence check on existing NACA 0012 history files."""
    from scripts.validation.convergence_checker import ConvergenceChecker

    run_dir = CASE_DIR / f"naca0012_{model}_alpha{int(alpha_deg)}_extended"
    history_path = run_dir / "history.csv"

    if not history_path.exists():
        # Try the non-extended directory
        alt_dirs = [
            CASE_DIR / f"naca0012_{model}_alpha{int(alpha_deg)}",
            CASE_DIR / f"alpha{int(alpha_deg)}_{model}",
        ]
        for alt in alt_dirs:
            alt_hist = alt / "history.csv"
            if alt_hist.exists():
                history_path = alt_hist
                break

    if not history_path.exists():
        print(f"No history file found for α={alpha_deg}°. Run simulation first.")
        return None

    print(f"\n{'='*60}")
    print(f"CONVERGENCE CHECK: NACA 0012 {model} α={alpha_deg}°")
    print(f"{'='*60}")
    print(f"History file: {history_path}")

    checker = ConvergenceChecker()
    checker.load_residual_history(str(history_path))
    checker.load_force_history(str(history_path))

    # Residual check
    res_status = checker.check_residual_convergence(target=1e-12)
    print(f"\nResidual convergence:")
    print(f"  Converged to 1e-12: {res_status.converged}")
    print(f"  Final residual: {res_status.final_residual:.2e}")
    print(f"  Orders dropped: {res_status.orders_of_magnitude:.1f}")
    print(f"  Iterations: {res_status.n_iterations}")

    # Force convergence
    force_status = checker.check_force_convergence(window=500, tolerance=0.001)
    print(f"\nForce convergence (last 500 iterations, 0.1% tolerance):")
    print(f"  Converged: {force_status.converged}")
    print(f"  Relative change: {force_status.relative_change:.6f}")

    # Print force values and compare to TMR
    for fname, fhist in checker.forces.items():
        if len(fhist) > 0:
            print(f"  {fname}: {fhist[-1]:.6f}")
            if int(alpha_deg) in CFL3D_REFERENCE and fname in CFL3D_REFERENCE[int(alpha_deg)]:
                ref = CFL3D_REFERENCE[int(alpha_deg)][fname]
                err_pct = abs(fhist[-1] - ref) / abs(ref) * 100
                print(f"    CFL3D reference: {ref:.6f} (error: {err_pct:.2f}%)")

    # Monotone convergence check
    for fname, fhist in checker.forces.items():
        mono = checker.check_monotone_convergence(fhist)
        print(f"  {fname} trend: {mono.trend}")

    # Generate full report
    report = checker.generate_convergence_report()
    report_path = run_dir / "convergence_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport saved to {report_path}")

    return report


def main():
    parser = argparse.ArgumentParser(description="NACA 0012 extended convergence runs")
    parser.add_argument("--alpha", type=float, nargs="+", default=[10, 15],
                        help="Angles of attack to run")
    parser.add_argument("--dry-run", action="store_true",
                        help="Generate configs only")
    parser.add_argument("--check-only", action="store_true",
                        help="Only check convergence of existing runs")
    parser.add_argument("--n-iter", type=int, default=50000,
                        help="Number of iterations")
    parser.add_argument("--model", type=str, default="SA",
                        help="Turbulence model (SA or SST)")
    parser.add_argument("--no-restart", action="store_true",
                        help="Start fresh instead of restarting")
    args = parser.parse_args()

    if args.check_only:
        for alpha in args.alpha:
            check_convergence(alpha, args.model)
        return

    configs = {}
    for alpha in args.alpha:
        config_path = generate_extended_config(
            alpha, args.n_iter,
            restart=not args.no_restart,
            model=args.model,
        )
        configs[alpha] = config_path

    if args.dry_run:
        print(f"\nDry run: generated {len(configs)} configs.")
        return

    # Run SU2 for each alpha
    for alpha, config_path in configs.items():
        run_dir = config_path.parent
        print(f"\nRunning SU2 for α={alpha}° in {run_dir}...")
        try:
            result = subprocess.run(
                ["SU2_CFD", str(config_path)],
                cwd=str(run_dir),
                capture_output=True,
                text=True,
                timeout=14400,  # 4 hour timeout
            )
            if result.returncode == 0:
                print(f"  α={alpha}° completed successfully.")
                check_convergence(alpha, args.model)
            else:
                print(f"  α={alpha}° failed (rc={result.returncode})")
        except FileNotFoundError:
            print("  SU2_CFD not found. Install SU2 and add to PATH.")
        except subprocess.TimeoutExpired:
            print(f"  α={alpha}° timed out after 4 hours.")


if __name__ == "__main__":
    main()
