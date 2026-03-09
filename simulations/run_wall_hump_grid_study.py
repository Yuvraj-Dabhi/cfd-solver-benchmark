#!/usr/bin/env python3
"""
Wall Hump Grid Convergence Study — Setup & Run Script
=====================================================
Sets up SA and SST simulations on coarse/medium/fine grids for
proper grid convergence analysis (Richardson extrapolation / GCI).

Usage:
    python run_wall_hump_grid_study.py --dry-run     # Setup only
    python run_wall_hump_grid_study.py               # Setup and run
    python run_wall_hump_grid_study.py -t 4          # Run with 4 OpenMP threads
"""
import sys
import argparse
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

PROJECT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT))

from run_wall_hump import (
    setup_case, run_simulation, check_solver,
    GRID_LEVELS, CASE_CONFIG,
)


def main():
    parser = argparse.ArgumentParser(description="Wall Hump grid convergence study")
    parser.add_argument("--dry-run", action="store_true",
                        help="Setup cases only, don't run solver")
    parser.add_argument("-t", "--threads", type=int, default=1,
                        help="OpenMP threads for SU2")
    parser.add_argument("--timeout", type=int, default=28800,
                        help="Max wall time per case (seconds)")
    parser.add_argument("--grids", nargs="+",
                        default=["coarse", "medium", "fine"],
                        choices=list(GRID_LEVELS.keys()),
                        help="Grid levels to run")
    parser.add_argument("--models", nargs="+", default=["SA", "SST"],
                        choices=["SA", "SST"],
                        help="Turbulence models to run")
    args = parser.parse_args()

    RUNS_DIR = PROJECT / "runs" / "wall_hump"
    GRIDS_DIR = PROJECT / "experimental_data" / "wall_hump" / "grids"

    print("=" * 60)
    print("  WALL HUMP — GRID CONVERGENCE STUDY")
    print("=" * 60)
    print(f"  Models:  {args.models}")
    print(f"  Grids:   {args.grids}")
    print(f"  Threads: {args.threads}")
    print()

    # Grid dimensions info
    print("  Grid levels:")
    for g in args.grids:
        info = GRID_LEVELS[g]
        print(f"    {g:8s}: {info['dims'][0]:>5d} × {info['dims'][1]:<5d} "
              f"({info['cells']:,d} cells)")
    print()

    # Refinement ratio check
    if len(args.grids) >= 2:
        for i in range(len(args.grids) - 1):
            g1, g2 = args.grids[i], args.grids[i + 1]
            d1, d2 = GRID_LEVELS[g1]["dims"], GRID_LEVELS[g2]["dims"]
            r_i = d2[0] / d1[0]
            r_j = d2[1] / d1[1]
            print(f"  Refinement ratio ({g1} → {g2}): "
                  f"r_i = {r_i:.2f}, r_j = {r_j:.2f}")
        print()

    # Setup cases
    cases = []
    print("  Setting up cases...")
    for model in args.models:
        for grid in args.grids:
            print(f"\n  --- {model} / {grid} ---")
            case_dir = setup_case(
                model=model, grid=grid,
                runs_dir=RUNS_DIR, grids_dir=GRIDS_DIR,
                n_iter=25000,
            )
            cases.append((model, grid, case_dir))

    if args.dry_run:
        print("\n  [DRY RUN] Cases set up. Not running solver.")
        print(f"  Total: {len(cases)} cases")
        print("\n  To run:")
        for model, grid, case_dir in cases:
            print(f"    cd {case_dir}")
            print(f"    SU2_CFD wall_hump.cfg")
        return

    # Check solver
    print()
    if not check_solver():
        print("  SU2_CFD not found. Install SU2 and add to PATH.")
        print("  Cases are set up — run manually when solver is available.")
        return

    # Run all cases
    results = []
    for model, grid, case_dir in cases:
        cfg = case_dir / "wall_hump.cfg"
        if not cfg.exists():
            print(f"  [SKIP] {model}/{grid} — no config file")
            continue
        print(f"\n{'=' * 60}")
        print(f"  Running: {model} / {grid}")
        print(f"{'=' * 60}")
        res = run_simulation(
            case_dir, cfg,
            n_threads=args.threads,
            timeout=args.timeout,
        )
        results.append({"model": model, "grid": grid, **res})

    # Summary
    print(f"\n{'=' * 60}")
    print(f"  GRID CONVERGENCE STUDY — SUMMARY")
    print(f"{'=' * 60}")
    for r in results:
        status = "✓ Converged" if r.get("converged") else "✗ Not converged"
        cd = r.get("CD", "N/A")
        wt = r.get("wall_time_s", 0)
        print(f"  {r['model']:4s} {r['grid']:8s}: CD = {cd}, "
              f"t = {wt:.0f}s, {status}")

    # Richardson extrapolation
    if len(args.grids) >= 3:
        print(f"\n  After all runs complete, compute GCI with:")
        print(f"  >>> from scripts.postprocessing.grid_convergence import richardson_extrapolation")
        print(f"  >>> result = richardson_extrapolation(CD_fine, CD_medium, CD_coarse)")


if __name__ == "__main__":
    main()
