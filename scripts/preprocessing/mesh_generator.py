"""
Mesh Generator
==============
Systematic 4-level grid generation for CFD benchmarking. Supports:
  - blockMesh (parametric structured grids)
  - snappyHexMesh (complex geometry refinement)
  - Gmsh (Python API for unstructured)

Grid levels follow a 1.5-2.0× refinement ratio with Δx⁺ targets
(50-100 / 25-50 / 10-25 / 5-10) for proper GCI analysis.
"""

import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import BENCHMARK_CASES


# =============================================================================
# Data Classes
# =============================================================================
@dataclass
class MeshLevel:
    """Specification for one grid refinement level."""
    name: str                   # coarse, medium, fine, xfine
    target_cells: int           # Approximate cell count
    dx_plus_target: str         # Δx⁺ range
    y_plus_target: float = 1.0  # First cell y⁺
    refinement_ratio: float = 1.0  # Relative to previous level


@dataclass
class MeshQuality:
    """Mesh quality metrics after generation."""
    n_cells: int = 0
    max_non_orthogonality: float = 0.0
    max_skewness: float = 0.0
    max_aspect_ratio: float = 0.0
    min_volume: float = 0.0
    avg_y_plus: float = 0.0
    passed: bool = False


# =============================================================================
# Core Mesh Generator
# =============================================================================
class MeshGenerator:
    """Generate systematic multi-level meshes for benchmark cases."""

    # Quality thresholds
    MAX_NON_ORTHO = 70.0
    MAX_SKEWNESS = 4.0
    MAX_ASPECT_RATIO = 100.0

    def __init__(self, case_name: str, case_dir: Path):
        """
        Parameters
        ----------
        case_name : str
            Key from config.BENCHMARK_CASES.
        case_dir : Path
            Root directory for the OpenFOAM case.
        """
        if case_name not in BENCHMARK_CASES:
            raise ValueError(f"Unknown case: {case_name}")

        self.case_name = case_name
        self.case_cfg = BENCHMARK_CASES[case_name]
        self.case_dir = Path(case_dir)
        self.mesh_levels = self._build_levels()

    def _build_levels(self) -> List[MeshLevel]:
        """Build mesh levels from case configuration."""
        mesh_cfg = self.case_cfg.mesh_levels
        dx_cfg = self.case_cfg.delta_x_plus or {}
        y_plus = self.case_cfg.yplus_target

        levels = []
        prev_cells = None
        for name in ["coarse", "medium", "fine", "xfine"]:
            if name not in mesh_cfg:
                continue
            cells = mesh_cfg[name]
            ratio = (cells / prev_cells) ** (1 / 3) if prev_cells else 1.0
            levels.append(MeshLevel(
                name=name,
                target_cells=cells,
                dx_plus_target=dx_cfg.get(name, "N/A"),
                y_plus_target=y_plus,
                refinement_ratio=ratio,
            ))
            prev_cells = cells
        return levels

    # ---- blockMesh Generation ----
    def generate_blockmesh(self, level: MeshLevel) -> Dict:
        """
        Generate a blockMeshDict for structured grids.

        Returns dict with path to generated blockMeshDict and mesh quality.
        """
        # Dispatch to case-specific generators
        generators = {
            "backward_facing_step": self._blockmesh_bfs,
            "flat_plate": self._blockmesh_flat_plate,
            "nasa_hump": self._blockmesh_hump,
        }

        gen = generators.get(self.case_name)
        if gen is None:
            raise NotImplementedError(
                f"blockMesh not implemented for '{self.case_name}'. "
                f"Use snappyHexMesh or Gmsh instead."
            )

        return gen(level)

    def _blockmesh_bfs(self, level: MeshLevel) -> Dict:
        """Backward-facing step: 3-block structured grid."""
        H = self.case_cfg.reference_length
        expansion = 1.125  # Step expansion ratio
        Re = self.case_cfg.reynolds_number
        U = self.case_cfg.reference_velocity

        # Domain dimensions
        upstream = 50 * H
        downstream = 200 * H
        height_inlet = H / (expansion - 1)  # Channel height upstream
        height_outlet = height_inlet + H

        # Cell counts (scale from target)
        scale = (level.target_cells / 40_000) ** 0.5  # 2D scaling
        nx_up = int(50 * scale)
        nx_down = int(200 * scale)
        ny_inlet = int(20 * scale)
        ny_step = int(10 * scale)

        # First cell height for y+ ~ 1
        nu = U * H / Re
        u_tau = U * np.sqrt(0.003 / 2)  # Approximate
        y1 = level.y_plus_target * nu / u_tau

        # Grading ratios
        grade_y = self._compute_grading(height_outlet, ny_inlet + ny_step, y1)

        blockmesh_dict = self._format_blockmesh_dict(
            case="BFS",
            vertices=[
                (-upstream, 0, 0), (0, 0, 0), (downstream, 0, 0),
                (downstream, height_outlet, 0),
                (0, height_outlet, 0), (-upstream, height_outlet, 0),
                (-upstream, H, 0), (0, H, 0),
            ],
            blocks=[
                f"hex (0 1 7 6 ...) ({nx_up} {ny_inlet} 1) simpleGrading (1 {grade_y} 1)",
                f"hex (1 2 3 4 ...) ({nx_down} {ny_inlet + ny_step} 1) simpleGrading (1 {grade_y} 1)",
            ],
            level=level,
        )

        return {"blockMeshDict": blockmesh_dict, "estimated_cells": level.target_cells}

    def _blockmesh_flat_plate(self, level: MeshLevel) -> Dict:
        """Flat plate: simple 2-block structured grid."""
        L = self.case_cfg.reference_length
        h = 1.0  # Domain height
        Re = self.case_cfg.reynolds_number
        U = self.case_cfg.reference_velocity

        nu = U * L / Re
        u_tau = U * np.sqrt(0.003 / 2)
        y1 = level.y_plus_target * nu / u_tau

        scale = (level.target_cells / 20_000) ** 0.5
        nx = int(200 * scale)
        ny = int(100 * scale)
        grade_y = self._compute_grading(h, ny, y1)

        blockmesh_dict = self._format_blockmesh_dict(
            case="FlatPlate",
            vertices=[(0, 0, 0), (L, 0, 0), (L, h, 0), (0, h, 0)],
            blocks=[f"hex (0 1 2 3 ...) ({nx} {ny} 1) simpleGrading (1 {grade_y} 1)"],
            level=level,
        )

        return {"blockMeshDict": blockmesh_dict, "estimated_cells": level.target_cells}

    def _blockmesh_hump(self, level: MeshLevel) -> Dict:
        """NASA hump: multi-block structured (simplified)."""
        c = self.case_cfg.reference_length
        tunnel_h = 0.9144

        scale = (level.target_cells / 150_000) ** 0.5
        nx = int(400 * scale)
        ny = int(200 * scale)

        blockmesh_dict = self._format_blockmesh_dict(
            case="NASAHump",
            vertices=[
                (-2.14 * c, 0, 0), (4.0 * c, 0, 0),
                (4.0 * c, tunnel_h, 0), (-2.14 * c, tunnel_h, 0),
            ],
            blocks=[f"hex (...) ({nx} {ny} 1) simpleGrading (1 ... 1)"],
            level=level,
        )

        return {"blockMeshDict": blockmesh_dict, "estimated_cells": level.target_cells}

    # ---- Utility Methods ----
    @staticmethod
    def _compute_grading(total_height: float, n_cells: int, first_cell: float) -> float:
        """Compute geometric grading ratio for boundary layer meshing."""
        if n_cells <= 1 or first_cell <= 0:
            return 1.0
        # Geometric series: total = first * (r^n - 1) / (r - 1)
        # Approximate iteratively
        r = 1.0
        for _ in range(50):
            total_approx = first_cell * (r**n_cells - 1) / (r - 1) if r != 1 else first_cell * n_cells
            if total_approx < total_height:
                r *= 1.01
            else:
                r /= 1.005
        return round(r, 4)

    @staticmethod
    def _format_blockmesh_dict(case: str, vertices: list, blocks: list,
                                level: MeshLevel) -> str:
        """Format a minimal blockMeshDict template."""
        vert_str = "\n".join(f"    ({v[0]} {v[1]} {v[2]})" for v in vertices)
        block_str = "\n".join(f"    {b}" for b in blocks)

        return f"""/*
    blockMeshDict for {case}
    Grid level: {level.name} (~{level.target_cells} cells)
    Target y+: {level.y_plus_target}, Δx+: {level.dx_plus_target}
*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      blockMeshDict;
}}

scale   1;

vertices
(
{vert_str}
    // ... 3D extrusion vertices
);

blocks
(
{block_str}
);

// Boundary patches to be defined per case
"""

    def check_quality(self, level_name: str = "fine") -> MeshQuality:
        """Run checkMesh and parse quality metrics."""
        quality = MeshQuality()

        check_cmd = f"checkMesh -case {self.case_dir}"
        try:
            result = subprocess.run(
                check_cmd.split(), capture_output=True, text=True, timeout=60
            )
            output = result.stdout

            # Parse key metrics from checkMesh output
            for line in output.split("\n"):
                if "cells:" in line.lower():
                    try:
                        quality.n_cells = int(line.split(":")[-1].strip())
                    except ValueError:
                        pass
                if "max non-orthogonality" in line.lower():
                    try:
                        quality.max_non_orthogonality = float(
                            line.split("=")[-1].strip().split()[0]
                        )
                    except (ValueError, IndexError):
                        pass
                if "max skewness" in line.lower():
                    try:
                        quality.max_skewness = float(
                            line.split("=")[-1].strip().split()[0]
                        )
                    except (ValueError, IndexError):
                        pass

            quality.passed = (
                quality.max_non_orthogonality < self.MAX_NON_ORTHO
                and quality.max_skewness < self.MAX_SKEWNESS
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass  # OpenFOAM not available

        return quality

    def generate_all_levels(self, method: str = "blockmesh") -> Dict[str, Dict]:
        """Generate meshes for all refinement levels."""
        results = {}
        for level in self.mesh_levels:
            if method == "blockmesh":
                results[level.name] = self.generate_blockmesh(level)
            else:
                raise NotImplementedError(f"Method '{method}' not yet implemented")
        return results


# =============================================================================
# y+ Estimator
# =============================================================================
def estimate_yplus(
    Re: float, L: float, U: float, y1: float, nu: Optional[float] = None
) -> float:
    """
    Estimate y⁺ for the first cell.

    Parameters
    ----------
    Re : float
        Reynolds number.
    L : float
        Reference length [m].
    U : float
        Free-stream velocity [m/s].
    y1 : float
        First cell height [m].
    nu : float, optional
        Kinematic viscosity [m²/s]. Computed from Re if not given.

    Returns
    -------
    float
        Estimated y⁺.
    """
    if nu is None:
        nu = U * L / Re
    Cf = 0.059 * Re**(-0.2)  # Turbulent flat-plate correlation
    tau_w = 0.5 * Cf * (U**2)  # Wall shear stress / density
    u_tau = np.sqrt(tau_w)
    return y1 * u_tau / nu


def required_first_cell_height(
    Re: float, L: float, U: float, y_plus_target: float = 1.0,
    nu: Optional[float] = None
) -> float:
    """
    Compute the required first cell height for a target y⁺.

    Returns
    -------
    float
        Required first cell height [m].
    """
    if nu is None:
        nu = U * L / Re
    Cf = 0.059 * Re**(-0.2)
    tau_w = 0.5 * Cf * (U**2)
    u_tau = np.sqrt(tau_w)
    return y_plus_target * nu / u_tau


# =============================================================================
# CLI
# =============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate benchmark meshes")
    parser.add_argument("case", help="Case name from config")
    parser.add_argument("--level", default="all", help="Grid level or 'all'")
    parser.add_argument("--outdir", default=".", help="Output directory")
    args = parser.parse_args()

    gen = MeshGenerator(args.case, Path(args.outdir))

    print(f"Case: {args.case}")
    print(f"Mesh levels: {[l.name for l in gen.mesh_levels]}")

    if args.level == "all":
        results = gen.generate_all_levels()
        for name, result in results.items():
            print(f"\n  {name}: ~{result['estimated_cells']} cells")
    else:
        level = next((l for l in gen.mesh_levels if l.name == args.level), None)
        if level:
            result = gen.generate_blockmesh(level)
            print(f"\n{result['blockMeshDict']}")
        else:
            print(f"Unknown level: {args.level}")
