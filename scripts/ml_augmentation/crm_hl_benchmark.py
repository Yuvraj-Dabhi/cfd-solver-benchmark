#!/usr/bin/env python3
"""
CRM-HL / HLPW-5 Benchmark Configuration
==========================================
NASA Common Research Model — High Lift (CRM-HL) benchmark per
AIAA High-Lift Prediction Workshop 5 (HLPW-5) standards.

Key features:
  - CRMHLConfiguration: multi-element high-lift geometry setup
  - WarmStartManager: sequential α-sweep restart strategy
  - DragDecompositionAnalyzer: pressure vs viscous drag analysis
  - NumericalDissipationStudy: upwind vs central scheme comparison
  - WMLESConfiguration: wall-modeled LES with dual time-stepping
  - CRMHLBenchmarkRunner: end-to-end HLPW-5 compliance pipeline

Architecture reference:
  - Rumsey et al. (2023): HLPW-5 test cases and results
  - Kiris et al. (2022): CRM-HL at flight Reynolds numbers
  - Coder (2019): SU2 for high-lift prediction

Usage:
    from scripts.ml_augmentation.crm_hl_benchmark import (
        CRMHLBenchmarkRunner, CRMHLConfiguration,
    )
    runner = CRMHLBenchmarkRunner()
    results = runner.run_alpha_sweep(alpha_range=[0, 4, 8, 12, 16, 20])
"""

import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)
PROJECT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT))


# =============================================================================
# CRM-HL Geometry Configuration
# =============================================================================
@dataclass
class CRMHLConfiguration:
    """
    CRM-HL multi-element high-lift geometry configuration.

    Encodes the NASA CRM-HL geometry parameters including slat,
    main wing, and flap settings per HLPW-5 specifications.

    Attributes
    ----------
    name : str
        Configuration name.
    reference_chord : float
        Reference MAC in meters.
    reference_area : float
        Reference wing area (half-model) in m².
    span : float
        Semi-span in meters.
    reynolds_number : float
        Reynolds number based on MAC.
    mach : float
        Freestream Mach number.
    temperature : float
        Freestream temperature in K.
    slat_deflection : float
        Slat deployment angle in degrees.
    flap_deflection : float
        Flap deployment angle in degrees.
    include_nacelle : bool
        Whether to include nacelle/pylon geometry.
    include_brackets : bool
        Whether to include slat/flap track brackets.
    """
    name: str = "CRM-HL Landing"
    reference_chord: float = 7.00532  # MAC in meters
    reference_area: float = 191.845   # Half-model, m²
    span: float = 29.38              # Semi-span, m
    reynolds_number: float = 5.49e6   # Based on MAC
    mach: float = 0.2
    temperature: float = 280.0        # K
    slat_deflection: float = 30.0     # degrees
    flap_deflection: float = 40.0     # degrees
    include_nacelle: bool = True
    include_brackets: bool = False
    alpha_range: List[float] = field(default_factory=lambda: [0, 4, 8, 12, 16, 20, 21])

    # Grid specifications (HLPW-5 standard grids)
    grid_levels: Dict[str, int] = field(default_factory=lambda: {
        "coarse": 25_000_000,
        "medium": 55_000_000,
        "fine": 130_000_000,
        "extra_fine": 280_000_000,
    })

    def to_su2_config(self, alpha: float = 8.0,
                      grid_level: str = "medium") -> Dict[str, Any]:
        """
        Generate SU2 configuration for CRM-HL at given alpha.

        Returns
        -------
        Dict with SU2 config key-value pairs.
        """
        return {
            "SOLVER": "RANS",
            "KIND_TURB_MODEL": "SST",
            "MATH_PROBLEM": "DIRECT",
            "RESTART_SOL": "NO",
            "MACH_NUMBER": self.mach,
            "AOA": alpha,
            "SIDESLIP_ANGLE": 0.0,
            "FREESTREAM_TEMPERATURE": self.temperature,
            "REYNOLDS_NUMBER": self.reynolds_number,
            "REYNOLDS_LENGTH": self.reference_chord,
            "REF_ORIGIN_MOMENT_X": 0.25 * self.reference_chord,
            "REF_ORIGIN_MOMENT_Y": 0.0,
            "REF_ORIGIN_MOMENT_Z": 0.0,
            "REF_LENGTH": self.reference_chord,
            "REF_AREA": self.reference_area,
            "MARKER_HEATFLUX": "(wing, 0.0, slat, 0.0, flap, 0.0, fuselage, 0.0)",
            "NUM_METHOD_GRAD": "GREEN_GAUSS",
            "CFL_NUMBER": 5.0,
            "CFL_ADAPT": "YES",
            "CFL_ADAPT_PARAM": "(0.1, 2.0, 1.0, 100.0)",
            "ITER": 10000,
            "CONV_RESIDUAL_MINVAL": -10,
            "CONV_FIELD": "RMS_DENSITY",
            "MESH_FILENAME": f"crm_hl_{grid_level}.su2",
            "OUTPUT_FILES": "(RESTART, PARAVIEW, SURFACE_CSV)",
            "VOLUME_OUTPUT": "(COORDINATES, SOLUTION, RESIDUAL, PRIMITIVE, "
                           "MESH_QUALITY)",
        }

    def get_grid_size(self, level: str) -> int:
        """Get number of grid points for specified level."""
        return self.grid_levels.get(level, 55_000_000)


# =============================================================================
# Warm-Start Manager
# =============================================================================
class WarmStartManager:
    """
    Sequential α-sweep restart strategy for high-lift cases.

    Implements warm-starting where each angle of attack is initialized
    from the converged solution at a nearby (lower) α. This dramatically
    improves convergence for near-stall conditions.

    Parameters
    ----------
    alpha_sequence : list of float
        Angles of attack in sweep order.
    restart_dir : str or Path
        Directory for restart files.
    """

    def __init__(self, alpha_sequence: List[float] = None,
                 restart_dir: str = "./restarts"):
        self.alpha_sequence = alpha_sequence or [0, 4, 8, 12, 16, 18, 19, 20, 21]
        self.restart_dir = Path(restart_dir)
        self.convergence_log = {}

    def get_restart_config(self, target_alpha: float) -> Dict[str, Any]:
        """
        Get restart configuration for target angle of attack.

        Parameters
        ----------
        target_alpha : float
            Target angle of attack.

        Returns
        -------
        Dict with restart file path and configuration overrides.
        """
        # Find nearest lower converged α
        available = sorted(self.convergence_log.keys())
        restart_alpha = None
        restart_file = None

        for a in reversed(available):
            if a < target_alpha and self.convergence_log[a].get("converged", False):
                restart_alpha = a
                restart_file = str(self.restart_dir / f"restart_alpha{a:.1f}.dat")
                break

        config = {
            "RESTART_SOL": "YES" if restart_file else "NO",
            "AOA": target_alpha,
        }

        if restart_file:
            config["SOLUTION_FILENAME"] = restart_file
            # Reduce CFL for near-stall conditions
            if target_alpha > 16:
                config["CFL_NUMBER"] = 1.0
                config["CFL_ADAPT_PARAM"] = "(0.05, 1.5, 0.5, 50.0)"
            logger.info("Warm-start α=%.1f from α=%.1f",
                       target_alpha, restart_alpha)

        return config

    def log_convergence(self, alpha: float, converged: bool,
                        residual: float = 0.0, cl: float = 0.0,
                        cd: float = 0.0):
        """Log convergence status for an angle of attack."""
        self.convergence_log[alpha] = {
            "converged": converged,
            "residual": residual,
            "cl": cl,
            "cd": cd,
            "timestamp": time.time(),
        }

    def generate_sweep_plan(self) -> List[Dict[str, Any]]:
        """
        Generate ordered sweep plan with restart recommendations.

        Returns
        -------
        List of dicts with 'alpha', 'restart_from', 'cfl_reduction'.
        """
        plan = []
        for i, alpha in enumerate(self.alpha_sequence):
            entry = {
                "alpha": alpha,
                "restart_from": self.alpha_sequence[i - 1] if i > 0 else None,
                "cfl_reduction": alpha > 16,
                "near_stall": alpha > 18,
                "recommended_iter": 15000 if alpha > 16 else 10000,
            }
            plan.append(entry)
        return plan

    def summary(self) -> str:
        """Generate convergence summary."""
        lines = ["Warm-Start Convergence Summary", "=" * 40]
        for alpha in sorted(self.convergence_log.keys()):
            info = self.convergence_log[alpha]
            status = "✓" if info["converged"] else "✗"
            lines.append(
                f"  α={alpha:5.1f}°  {status}  CL={info['cl']:.4f}  "
                f"CD={info['cd']:.5f}  Res={info['residual']:.2e}")
        return "\n".join(lines)


# =============================================================================
# Drag Decomposition Analyzer
# =============================================================================
class DragDecompositionAnalyzer:
    """
    Decompose total drag into pressure and viscous components,
    with sensitivity to numerical scheme choice.

    Computes:
      - CDp (pressure drag) vs CDv (viscous drag)
      - CDi (induced drag) via Trefftz-plane approach
      - Parasite vs form drag breakdown
      - JST vs Roe scheme artificial dissipation comparison
    """

    def __init__(self):
        self._results = {}

    def decompose(self, surface_cp: np.ndarray, surface_cf: np.ndarray,
                  surface_normals: np.ndarray, surface_areas: np.ndarray,
                  alpha: float = 8.0) -> Dict[str, float]:
        """
        Decompose drag from surface distributions.

        Parameters
        ----------
        surface_cp : ndarray (n_faces,)
            Surface pressure coefficient.
        surface_cf : ndarray (n_faces,)
            Surface skin friction coefficient magnitude.
        surface_normals : ndarray (n_faces, 3)
            Outward surface normals.
        surface_areas : ndarray (n_faces,)
            Face areas.
        alpha : float
            Angle of attack in degrees.

        Returns
        -------
        Dict with CD decomposition.
        """
        alpha_rad = np.radians(alpha)
        cos_a = np.cos(alpha_rad)
        sin_a = np.sin(alpha_rad)

        # Pressure drag: integrate Cp * n_x over surface
        CDp = -np.sum(surface_cp * surface_normals[:, 0] * surface_areas)

        # Viscous drag: integrate Cf in freestream direction
        CDv = np.sum(surface_cf * cos_a * surface_areas)

        # Total
        CD_total = CDp + CDv

        # Lift for L/D
        CL = -np.sum(surface_cp * (
            -surface_normals[:, 0] * sin_a + surface_normals[:, 2] * cos_a
        ) * surface_areas)

        self._results = {
            "CD_total": float(CD_total),
            "CD_pressure": float(CDp),
            "CD_viscous": float(CDv),
            "CD_pressure_pct": float(CDp / (CD_total + 1e-15) * 100),
            "CD_viscous_pct": float(CDv / (CD_total + 1e-15) * 100),
            "CL": float(CL),
            "L_over_D": float(CL / (CD_total + 1e-15)),
            "alpha": alpha,
        }
        return self._results

    def compute_induced_drag(self, CL: float, AR: float = 9.0,
                              e: float = 0.85) -> float:
        """
        Estimate induced drag via lifting-line theory.

        CDi = CL² / (π·e·AR)

        Parameters
        ----------
        CL : float
            Lift coefficient.
        AR : float
            Wing aspect ratio.
        e : float
            Oswald efficiency factor.
        """
        CDi = CL ** 2 / (np.pi * e * AR)
        self._results["CD_induced"] = float(CDi)
        return float(CDi)

    def scheme_sensitivity(self, cd_jst: float, cd_roe: float) -> Dict[str, float]:
        """
        Compare drag predictions between JST and Roe schemes.

        Parameters
        ----------
        cd_jst : float
            Drag coefficient from JST (central + artificial dissipation).
        cd_roe : float
            Drag coefficient from Roe (upwind).

        Returns
        -------
        Dict with scheme comparison metrics.
        """
        diff = cd_jst - cd_roe
        artificial_dissipation = abs(diff)
        return {
            "cd_jst": cd_jst,
            "cd_roe": cd_roe,
            "difference": diff,
            "artificial_dissipation_counts": artificial_dissipation * 1e4,
            "pct_difference": abs(diff) / max(abs(cd_roe), 1e-15) * 100,
            "preferred_scheme": "Roe" if cd_roe < cd_jst else "JST",
        }


# =============================================================================
# Numerical Dissipation Study
# =============================================================================
class NumericalDissipationStudy:
    """
    Study the effect of numerical dissipation on drag prediction.

    Compares upwind (Roe) vs central (JST) schemes with varying
    levels of artificial viscosity (JST sensor coefficients).
    """

    def __init__(self):
        self._study_results = []

    def generate_jst_configs(self, k2_values: List[float] = None,
                              k4_values: List[float] = None) -> List[Dict[str, Any]]:
        """
        Generate SU2 configs with varying JST coefficients.

        Parameters
        ----------
        k2_values : list of float
            JST 2nd-order dissipation coefficients.
        k4_values : list of float
            JST 4th-order dissipation coefficients.
        """
        if k2_values is None:
            k2_values = [0.25, 0.5, 1.0]
        if k4_values is None:
            k4_values = [0.01, 0.02, 0.04]

        configs = []
        for k2 in k2_values:
            for k4 in k4_values:
                configs.append({
                    "CONV_NUM_METHOD_FLOW": "JST",
                    "JST_SENSOR_COEFF": f"({k2}, {k4})",
                    "label": f"JST_k2={k2}_k4={k4}",
                })

        # Add Roe reference
        configs.append({
            "CONV_NUM_METHOD_FLOW": "ROE",
            "MUSCL_FLOW": "YES",
            "SLOPE_LIMITER_FLOW": "VENKATAKRISHNAN",
            "label": "Roe_MUSCL",
        })
        return configs

    def analyze_results(self, cd_values: Dict[str, float]) -> Dict[str, Any]:
        """
        Analyze drag sensitivity to numerical scheme.

        Parameters
        ----------
        cd_values : dict
            label → CD value mapping.

        Returns
        -------
        Dict with sensitivity analysis results.
        """
        values = list(cd_values.values())
        labels = list(cd_values.keys())
        spread = max(values) - min(values) if values else 0

        sensitivity = {
            "n_schemes": len(cd_values),
            "cd_min": min(values) if values else 0,
            "cd_max": max(values) if values else 0,
            "cd_spread": spread,
            "cd_spread_counts": spread * 1e4,
            "best_scheme": labels[int(np.argmin(values))] if values else "N/A",
            "worst_scheme": labels[int(np.argmax(values))] if values else "N/A",
        }
        return sensitivity


# =============================================================================
# WMLES Configuration
# =============================================================================
@dataclass
class WMLESConfiguration:
    """
    Wall-Modeled Large Eddy Simulation configuration for CRM-HL.

    Generates SU2 config for WMLES with proper dual time-stepping
    and subgrid-scale model specification.

    Attributes
    ----------
    sgs_model : str
        Sub-grid scale model ('WALE', 'Sigma', 'Vreman').
    dt_physical : float
        Physical time step in seconds.
    dt_inner_iter : int
        Inner iterations per physical time step.
    n_timesteps : int
        Total number of physical time steps.
    wall_model : str
        Wall model type ('equilibrium', 'nonequilibrium').
    """
    sgs_model: str = "WALE"
    dt_physical: float = 1e-5
    dt_inner_iter: int = 20
    n_timesteps: int = 10000
    wall_model: str = "equilibrium"
    sampling_start: int = 5000      # Start averaging after this step
    spanwise_extent: float = 0.1    # For periodic span

    def to_su2_config(self) -> Dict[str, Any]:
        """Generate WMLES-specific SU2 configuration."""
        return {
            "SOLVER": "INC_NAVIER_STOKES",
            "KIND_TURB_MODEL": "NONE",
            "HYBRID_RANSLES": "SA_DDES" if self.wall_model == "nonequilibrium" else "NONE",
            "TIME_DOMAIN": "YES",
            "TIME_MARCHING": "DUAL_TIME_STEPPING-2ND_ORDER",
            "TIME_STEP": self.dt_physical,
            "MAX_TIME": self.dt_physical * self.n_timesteps,
            "INNER_ITER": self.dt_inner_iter,
            "TIME_ITER": self.n_timesteps,
            "RESTART_SOL": "YES",
            "OUTPUT_WRT_FREQ": 100,
            "CONV_NUM_METHOD_FLOW": "FDS",
            "MUSCL_FLOW": "YES",
            "SLOPE_LIMITER_FLOW": "NONE",
            "CFL_NUMBER": 200.0,
            "WALL_MODEL": self.wall_model.upper(),
            "SGS_MODEL": self.sgs_model,
        }

    def estimate_cost(self, n_cells: int) -> Dict[str, float]:
        """Estimate computational cost for WMLES run."""
        cost_per_step = n_cells * self.dt_inner_iter * 1e-6  # Arbitrary scaling
        total_hours = cost_per_step * self.n_timesteps / 3600
        return {
            "cost_per_step_M": cost_per_step,
            "total_hours": total_hours,
            "n_cells": n_cells,
            "n_timesteps": self.n_timesteps,
            "averaging_steps": self.n_timesteps - self.sampling_start,
        }


# =============================================================================
# CRM-HL Benchmark Runner
# =============================================================================
class CRMHLBenchmarkRunner:
    """
    End-to-end CRM-HL benchmark pipeline for HLPW-5 compliance.

    Orchestrates geometry setup, α-sweep with warm-starting,
    drag analysis, and compliance reporting.

    Parameters
    ----------
    config : CRMHLConfiguration
        Geometry and flow configuration.
    output_dir : str
        Output directory.
    """

    def __init__(self, config: CRMHLConfiguration = None,
                 output_dir: str = "./crm_hl_results"):
        self.config = config or CRMHLConfiguration()
        self.output_dir = Path(output_dir)
        self.warm_start = WarmStartManager(self.config.alpha_range)
        self.drag_analyzer = DragDecompositionAnalyzer()
        self.dissipation_study = NumericalDissipationStudy()
        self._sweep_results = {}

    def generate_alpha_sweep_configs(self, grid_level: str = "medium"
                                       ) -> List[Dict[str, Any]]:
        """
        Generate SU2 configs for complete α-sweep.

        Returns
        -------
        List of SU2 config dicts, one per α.
        """
        configs = []
        sweep_plan = self.warm_start.generate_sweep_plan()

        for entry in sweep_plan:
            alpha = entry["alpha"]
            base_config = self.config.to_su2_config(alpha, grid_level)

            # Apply warm-start settings
            restart_config = self.warm_start.get_restart_config(alpha)
            base_config.update(restart_config)

            # Adjust for near-stall
            if entry["near_stall"]:
                base_config["ITER"] = entry["recommended_iter"]
                base_config["CFL_NUMBER"] = 1.0

            configs.append(base_config)

        return configs

    def log_alpha_result(self, alpha: float, cl: float, cd: float,
                          cm: float = 0.0, converged: bool = True):
        """Log result for a single angle of attack."""
        self._sweep_results[alpha] = {
            "CL": cl, "CD": cd, "CM": cm,
            "L/D": cl / (cd + 1e-15),
            "converged": converged,
        }
        self.warm_start.log_convergence(alpha, converged, cl=cl, cd=cd)

    def predict_clmax(self) -> Dict[str, float]:
        """
        Predict CLmax from the α-sweep results.

        Returns
        -------
        Dict with CLmax, alpha_stall, and stall margin.
        """
        if not self._sweep_results:
            return {"CLmax": 0, "alpha_stall": 0}

        alphas = sorted(self._sweep_results.keys())
        cls = [self._sweep_results[a]["CL"] for a in alphas]

        clmax_idx = int(np.argmax(cls))
        clmax = cls[clmax_idx]
        alpha_stall = alphas[clmax_idx]

        # Compute CL slope in linear region
        if len(alphas) >= 3:
            linear_end = min(3, len(alphas))
            cl_slope = (cls[linear_end - 1] - cls[0]) / (
                alphas[linear_end - 1] - alphas[0] + 1e-15)
        else:
            cl_slope = 0.1  # Default

        return {
            "CLmax": clmax,
            "alpha_stall": alpha_stall,
            "CL_slope_per_deg": cl_slope,
            "alpha_linear_end": alphas[min(2, len(alphas) - 1)],
        }

    def hlpw5_compliance_check(self) -> Dict[str, Any]:
        """
        Check compliance with HLPW-5 submission requirements.

        Returns
        -------
        Dict with compliance status for each requirement.
        """
        checks = {
            "geometry": {
                "crm_hl_config": self.config.name == "CRM-HL Landing",
                "slat_angle": abs(self.config.slat_deflection - 30.0) < 1.0,
                "flap_angle": abs(self.config.flap_deflection - 40.0) < 1.0,
                "nacelle": self.config.include_nacelle,
            },
            "conditions": {
                "mach_02": abs(self.config.mach - 0.2) < 0.01,
                "reynolds_correct": abs(self.config.reynolds_number - 5.49e6) / 5.49e6 < 0.01,
            },
            "alpha_sweep": {
                "n_alphas": len(self._sweep_results),
                "includes_stall": any(a >= 18 for a in self._sweep_results.keys()),
                "all_converged": all(r["converged"] for r in self._sweep_results.values()),
            },
        }
        n_pass = sum(1 for cat in checks.values()
                     for v in cat.values() if v is True)
        n_total = sum(1 for cat in checks.values()
                      for v in cat.values() if isinstance(v, bool))
        checks["overall_compliance_pct"] = n_pass / max(n_total, 1) * 100
        return checks

    def report(self) -> str:
        """Generate comprehensive benchmark report."""
        lines = [
            "CRM-HL / HLPW-5 Benchmark Report",
            "=" * 45,
            f"Configuration: {self.config.name}",
            f"Re = {self.config.reynolds_number:.2e}, M = {self.config.mach}",
            f"Slat: {self.config.slat_deflection}°, Flap: {self.config.flap_deflection}°",
            "",
            "α-Sweep Results:",
            "-" * 45,
        ]

        for alpha in sorted(self._sweep_results.keys()):
            r = self._sweep_results[alpha]
            status = "✓" if r["converged"] else "✗"
            lines.append(
                f"  α={alpha:5.1f}°  {status}  CL={r['CL']:.4f}  "
                f"CD={r['CD']:.5f}  L/D={r['L/D']:.1f}")

        if self._sweep_results:
            clmax = self.predict_clmax()
            lines.extend([
                "",
                f"CLmax = {clmax['CLmax']:.4f} at α = {clmax['alpha_stall']:.1f}°",
                f"CL slope = {clmax['CL_slope_per_deg']:.4f} /deg",
            ])

        return "\n".join(lines)
