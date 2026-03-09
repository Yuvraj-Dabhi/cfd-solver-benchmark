#!/usr/bin/env python3
"""
Consolidated Plotting Tool
==========================
Unified interface for generating all benchmark validation plots.
Replaces the individual run scripts with a single CLI using subcommands.

Usage:
  python plot_all.py [command]

Commands:
  beverli      Plot BeVERLI Hill validation
  pareto       Plot Physics Pareto fronts
  swbli        Plot SWBLI validation
  tmr          Plot TMR (NACA 0012) validation
  velocity     Plot Velocity Profiles
  wall_hump    Plot Wall Hump validation
  all          Generate all plots
"""

import argparse
import sys
import os
from pathlib import Path

# Add project root to path
PROJECT = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(PROJECT))

# Import the main functions from the individual plotting modules
from scripts.postprocessing.plot_beverli_hill_validation import main as plot_beverli
from scripts.postprocessing.plot_physics_pareto import main as plot_pareto
from scripts.postprocessing.plot_swbli_validation import main as plot_swbli
from scripts.postprocessing.plot_tmr_validation import main as plot_tmr
from scripts.postprocessing.plot_velocity_profiles import main as plot_velocity
from scripts.postprocessing.plot_wall_hump_validation import main as plot_wall_hump

def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive plotting suite for the CFD Solver Benchmark."
    )
    subparsers = parser.add_subparsers(dest="command", help="Plotting domains")
    
    subparsers.add_parser("beverli", help="Plot BeVERLI Hill validation metrics")
    subparsers.add_parser("pareto", help="Plot Physics Pareto fronts for ML models")
    subparsers.add_parser("swbli", help="Plot SWBLI pressure validation profiles")
    subparsers.add_parser("tmr", help="Plot TMR (NACA 0012) force and Cp comparisons")
    subparsers.add_parser("velocity", help="Plot Velocity Profiles for wall cases")
    subparsers.add_parser("wall_hump", help="Plot Wall Hump Cp/Cf validation")
    subparsers.add_parser("all", help="Generate all validation plots sequentially")
    
    args = parser.parse_args()
    
    commands = {
        "beverli": plot_beverli,
        "pareto": plot_pareto,
        "swbli": plot_swbli,
        "tmr": plot_tmr,
        "velocity": plot_velocity,
        "wall_hump": plot_wall_hump,
    }
    
    if args.command == "all":
        print("============================================================")
        print("  Generating ALL Benchmark Validation Plots")
        print("============================================================")
        for name, func in commands.items():
            print(f"\n---> Running: {name}")
            try:
                func()
            except Exception as e:
                print(f"Error executing {name}: {e}")
                
        print("\nAll plotting tasks completed.")
    elif args.command in commands:
        commands[args.command]()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
