#!/usr/bin/env python3
"""
CFD Benchmark - Quick Start Guide & Runner
===========================================
Interactive guide for running simulations and validation.

This script will:
1. Check your environment
2. Run verification tests
3. Guide you through simulation setup
4. Execute simulations
5. Validate results against reference data
6. Generate reports

Usage: python start_here.py
"""

import sys
import subprocess
from pathlib import Path
import time

PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

# Force UTF-8 output on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

print("""
+==============================================================================+
|                                                                              |
|          CFD SOLVER BENCHMARK FOR FLOW SEPARATION PREDICTION                 |
|                                                                              |
|  A comprehensive benchmarking framework for turbulence models in             |
|  separated flows with formal V&V, grid convergence, and ML augmentation.     |
|                                                                              |
+==============================================================================+
""")

def print_section(title):
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)

def run_command(cmd, description):
    """Run a command and return success status."""
    print(f"\n▶ {description}")
    print(f"  Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("  ✓ Success")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Failed: {e}")
        if e.stderr:
            print(f"  Error: {e.stderr[:200]}")
        return False

def main():
    """Main interactive workflow."""

    # =========================================================================
    # STEP 1: Quick Verification
    # =========================================================================
    print_section("STEP 1: Environment & Code Verification")

    print("""
This will verify:
  • Python dependencies are installed
  • Configuration files are valid
  • Experimental data loads correctly
  • Analytical solutions match theory
  • Error metrics function correctly

Estimated time: 30 seconds
""")

    proceed = input("Run verification? [Y/n]: ").strip().lower()
    if proceed not in ['', 'y', 'yes']:
        print("Verification skipped. Cannot proceed safely.")
        sys.exit(1)

    verify_script = str(PROJECT_ROOT / "run_full_benchmark.py")
    if not run_command([sys.executable, verify_script, "--quick"], "Running verification"):
        print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║  VERIFICATION FAILED - Please fix issues before proceeding                   ║
║                                                                              ║
║  Common fixes:                                                               ║
║  1. Install dependencies: pip install -r requirements.txt                    ║
║  2. Check Python version: python --version (need 3.8+)                       ║
║  3. Ensure all files are present in project directory                        ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")
        sys.exit(1)

    # =========================================================================
    # STEP 2: Full Benchmark Test (Optional)
    # =========================================================================
    print_section("STEP 2: Full Benchmark Test (Optional)")

    print("""
This comprehensive test validates all 150+ checks:
  • All modules and classes
  • Reference data comparisons (NASA Hump, BFS, periodic hill)
  • Analytical verifications (flat plate, law of wall)
  • Grid convergence algorithms
  • ML module functionality

Estimated time: 2-3 minutes
""")

    proceed = input("Run full benchmark test? [Y/n]: ").strip().lower()
    if proceed in ['', 'y', 'yes']:
        benchmark_script = str(PROJECT_ROOT / "run_full_benchmark.py")
        run_command([sys.executable, benchmark_script], "Running full benchmark")
    else:
        print("Skipping full test. Proceeding to simulations.")

    # =========================================================================
    # STEP 3: Choose Simulation Configuration
    # =========================================================================
    print_section("STEP 3: Simulation Configuration")

    # Show available cases from config
    try:
        from config import BENCHMARK_CASES, TURBULENCE_MODELS
        print("\nRegistered benchmark cases:")
        for key, case in BENCHMARK_CASES.items():
            print(f"  • {key}: {case.name} (Tier {case.tier.value})")
        print(f"\nRegistered turbulence models:")
        model_names = [f"{m.abbreviation}" for m in TURBULENCE_MODELS.values()]
        print(f"  {', '.join(model_names)}")
    except ImportError:
        pass

    print("""
Choose your simulation configuration:

1. Quick Test (10-30 min)
   └─ Single case, single model, coarse grid
   └─ Good for: Testing setup, familiarization
   └─ Example: BFS with SA model

2. Grid Convergence Study (2-6 hours)
   └─ Single case, 1-2 models, 3 grid levels
   └─ Good for: Uncertainty quantification, publication
   └─ Example: NASA Hump with SA+SST, coarse/medium/fine

3. Model Comparison (4-12 hours)
   └─ Single case, 5-7 models, medium grid
   └─ Good for: Model benchmarking, best model selection
   └─ Example: BFS with SA, SST, kEpsilon, v2f, realizable-kE

4. Custom Configuration
   └─ Specify cases, models, and grids manually

5. Skip simulations (validation only)
   └─ Use existing simulation results
""")

    choice = input("Select option [1-5]: ").strip()

    if choice == '1':
        # Quick test
        case = "backward_facing_step"
        models = "SA"
        grids = "coarse"
        nproc = 4

        print(f"""
Selected Configuration:
  Case:   {case}
  Model:  {models}
  Grid:   {grids}
  Cores:  {nproc}

Estimated runtime: 10-30 minutes
""")

    elif choice == '2':
        # Grid convergence
        print("\nSelect case:")
        print("  1. backward_facing_step (BFS)")
        print("  2. nasa_hump")
        print("  3. periodic_hill")
        case_choice = input("Case [1-3]: ").strip()

        case_map = {'1': 'backward_facing_step', '2': 'nasa_hump', '3': 'periodic_hill'}
        case = case_map.get(case_choice, 'backward_facing_step')

        models = "SA,SST"
        grids = "coarse,medium,fine"
        nproc = 8

        print(f"""
Selected Configuration:
  Case:   {case}
  Models: {models}
  Grids:  {grids}
  Cores:  {nproc}

Estimated runtime: 2-6 hours
""")

    elif choice == '3':
        # Model comparison
        case = "backward_facing_step"
        models = "SA,SST,kEpsilon,v2f"
        grids = "medium"
        nproc = 10

        print(f"""
Selected Configuration:
  Case:   {case}
  Models: {models}
  Grid:   {grids}
  Cores:  {nproc}

Estimated runtime: 4-12 hours
""")

    elif choice == '4':
        # Custom
        print("\nAvailable cases (from config.BENCHMARK_CASES):")
        try:
            for key in BENCHMARK_CASES.keys():
                print(f"  {key}")
        except NameError:
            print("  backward_facing_step, nasa_hump, periodic_hill, ...")
        case = input("Case: ").strip()

        print("\nAvailable models (from config.TURBULENCE_MODELS):")
        try:
            for key in TURBULENCE_MODELS.keys():
                print(f"  {key}")
        except NameError:
            print("  SA, SST, kEpsilon, v2f, ...")
        models = input("Models (comma-separated): ").strip()

        grids = input("Grid levels (comma-separated, e.g., coarse,medium,fine): ").strip()
        nproc = int(input("Number of CPU cores: ").strip() or "4")

        print(f"""
Selected Configuration:
  Case:   {case}
  Models: {models}
  Grids:  {grids}
  Cores:  {nproc}
""")

    elif choice == '5':
        print("\nSkipping simulations. Proceeding to validation.")
        case = input("Case name: ").strip()
        models = input("Model name: ").strip()
        grids = input("Grid level: ").strip()

        # Skip to validation
        run_validation(case, models, grids)
        return

    else:
        print("Invalid choice. Exiting.")
        sys.exit(1)

    # =========================================================================
    # STEP 4: Run Simulations
    # =========================================================================
    print_section("STEP 4: Running Simulations")

    proceed = input(f"\nProceed with simulation? [Y/n]: ").strip().lower()
    if proceed not in ['', 'y', 'yes']:
        print("Simulation cancelled.")
        sys.exit(0)

    print(f"\n⏱ Starting at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📁 Results will be saved to: ./runs/{case}/")

    # Use batch_manager from scripts/solvers/
    batch_script = str(PROJECT_ROOT / "scripts" / "solvers" / "batch_manager.py")
    cmd = [
        sys.executable, batch_script,
        "--case", case,
        "--models", models,
        "--grids", grids,
        "--nproc", str(nproc)
    ]

    if not run_command(cmd, f"Running CFD simulations"):
        print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  SIMULATION FAILED                                                           ║
║                                                                              ║
║  Possible causes:                                                            ║
║  1. OpenFOAM not installed or not in PATH                                    ║
║  2. Insufficient disk space                                                  ║
║  3. Invalid case/model/grid specification                                    ║
║                                                                              ║
║  Check log files in ./runs/{case}/MODEL_GRID/log.*                           ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")
        sys.exit(1)

    print(f"\n✓ Simulations completed at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # =========================================================================
    # STEP 5: Validation
    # =========================================================================
    run_validation(case, models.split(',')[0], grids.split(',')[0])

    # =========================================================================
    # FINAL MESSAGE
    # =========================================================================
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║  ✓ WORKFLOW COMPLETE                                                         ║
║                                                                              ║
║  Next steps:                                                                 ║
║  1. Review validation results in ./validation_results/                       ║
║  2. Generate full report: python run_full_benchmark.py                       ║
║  3. Run sensitivity analysis (with SALib installed)                          ║
║                                                                              ║
║  Documentation: See SIMULATION_WORKFLOW.md for detailed instructions        ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")


def run_validation(case, model, grid):
    """Run validation against reference data."""
    print_section("STEP 5: Validation Against Reference Data")

    print(f"""
Comparing CFD results with experimental/analytical data:
  Case:  {case}
  Model: {model}
  Grid:  {grid}

This will:
  • Compare separation/reattachment locations
  • Validate Cp and Cf distributions
  • Check velocity profile accuracy
  • Compute error metrics (RMSE, MAPE, R²)
  • Generate validation plots
  • Assess NASA 40% Challenge criteria
""")

    results_dir = Path(f"./runs/{case}/{model}_{grid}")

    if not results_dir.exists():
        print(f"\n⚠ Results directory not found: {results_dir}")
        print("Skipping validation.")
        return

    validate_script = str(PROJECT_ROOT / "scripts" / "validation" / "validate_results.py")
    cmd = [
        sys.executable, validate_script,
        "--case", case,
        "--model", model,
        "--grid", grid,
        "--results-dir", str(results_dir),
        "--output", "./validation_results"
    ]

    if run_command(cmd, "Running validation"):
        print("""
✓ Validation completed successfully!

Check outputs:
  • Plots:  ./validation_results/validation_*.pdf
  • Report: ./validation_results/validation_*.json
""")
    else:
        print("""
⚠ Validation encountered issues.
This may be normal if:
  • Post-processing files are not yet generated
  • Simulation did not converge fully
  • Output format differs from expected

Review simulation logs for details.
""")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user. Exiting.")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
