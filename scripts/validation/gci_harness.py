#!/usr/bin/env python3
"""
Reusable Grid Convergence Index (GCI) Harness
===============================================
Generalized 3-level GCI analysis per Roache (1997) and ASME V&V 20-2009.

Extracts quantities of interest from any case, computes Richardson
extrapolation, observed convergence order, GCI_fine, and asymptotic
range ratio. Produces formatted summary tables matching TMR style.

Usage
-----
    from scripts.validation.gci_harness import GCIStudy

    study = GCIStudy(r21=2.0, r32=2.0)
    study.add_quantity("x_sep", f_coarse=0.72, f_medium=0.69, f_fine=0.67)
    study.add_quantity("Cp_RMSE", f_coarse=0.045, f_medium=0.032, f_fine=0.028)
    study.compute()
    study.summary_table()
    study.to_json("gci_results.json")

References
----------
  - Roache (1997), J. Fluids Eng. 119, pp.681-686
  - Celik et al. (2008), J. Fluids Eng. 130(7), DOI:10.1115/1.2960953
  - ASME V&V 20-2009, Standard for Verification and Validation
"""

import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class GCIResult:
    """Result for a single quantity of interest."""
    name: str
    f_coarse: float
    f_medium: float
    f_fine: float
    observed_order: float = 0.0
    extrapolated_value: float = 0.0
    gci_fine_pct: float = 0.0
    gci_medium_pct: float = 0.0
    asymptotic_ratio: float = 0.0
    in_asymptotic_range: bool = False
    convergence_type: str = ""       # monotonic, oscillatory, divergent
    safety_factor: float = 1.25


@dataclass
class GCIStudy:
    """
    3-level Grid Convergence Index study.

    Parameters
    ----------
    r21 : float
        Grid refinement ratio (fine-to-medium), typically sqrt(N_fine/N_medium).
    r32 : float
        Grid refinement ratio (medium-to-coarse).
    safety_factor : float
        GCI safety factor (1.25 for 3+ grids, 3.0 for 2 grids).
    """

    r21: float = 2.0
    r32: float = 2.0
    safety_factor: float = 1.25
    _quantities: Dict[str, Tuple[float, float, float]] = field(
        default_factory=dict, repr=False
    )
    _results: Dict[str, GCIResult] = field(default_factory=dict, repr=False)

    def add_quantity(
        self,
        name: str,
        f_coarse: float,
        f_medium: float,
        f_fine: float,
    ) -> None:
        """
        Register a scalar quantity of interest for GCI analysis.

        Parameters
        ----------
        name : str
            QoI label (e.g., 'x_sep', 'Cp_RMSE', 'CL').
        f_coarse, f_medium, f_fine : float
            Values on coarse, medium, and fine grids.
        """
        self._quantities[name] = (f_coarse, f_medium, f_fine)

    def compute(self, safety_factor: Optional[float] = None) -> Dict[str, GCIResult]:
        """
        Compute GCI for all registered quantities.

        Parameters
        ----------
        safety_factor : float, optional
            Override the default safety factor.

        Returns
        -------
        dict mapping quantity name → GCIResult.
        """
        Fs = safety_factor or self.safety_factor
        self._results = {}

        for name, (f3, f2, f1) in self._quantities.items():
            result = self._compute_single(name, f3, f2, f1, Fs)
            self._results[name] = result

        return self._results

    def _compute_single(
        self,
        name: str,
        f_coarse: float,
        f_medium: float,
        f_fine: float,
        Fs: float,
    ) -> GCIResult:
        """Compute GCI for a single quantity."""
        r21 = self.r21
        r32 = self.r32

        eps_32 = f_medium - f_coarse   # coarse → medium change
        eps_21 = f_fine - f_medium      # medium → fine change

        result = GCIResult(
            name=name,
            f_coarse=f_coarse,
            f_medium=f_medium,
            f_fine=f_fine,
            safety_factor=Fs,
        )

        # Check convergence type
        if abs(eps_21) < 1e-15 and abs(eps_32) < 1e-15:
            result.convergence_type = "converged"
            result.observed_order = 0.0
            result.extrapolated_value = f_fine
            result.gci_fine_pct = 0.0
            result.gci_medium_pct = 0.0
            result.asymptotic_ratio = 1.0
            result.in_asymptotic_range = True
            return result

        if abs(eps_21) < 1e-15:
            # Fine and medium are identical
            result.convergence_type = "converged"
            result.observed_order = float("inf")
            result.extrapolated_value = f_fine
            result.gci_fine_pct = 0.0
            return result

        ratio = eps_32 / eps_21 if abs(eps_21) > 1e-15 else float("inf")

        if ratio < 0:
            result.convergence_type = "oscillatory"
            # Use absolute values for order estimation
            p_est = abs(math.log(abs(ratio))) / math.log(r21) if abs(ratio) > 0 else 0
            result.observed_order = max(p_est, 0.5)
        elif 0 < ratio < 1:
            result.convergence_type = "divergent"
            result.observed_order = 0.0
            result.extrapolated_value = f_fine
            result.gci_fine_pct = float("nan")
            return result
        else:
            result.convergence_type = "monotonic"
            # Iterative procedure for observed order (Celik et al. 2008)
            try:
                result.observed_order = self._compute_order_iterative(
                    eps_21, eps_32, r21, r32
                )
            except (ValueError, ZeroDivisionError):
                result.observed_order = 1.0  # Fallback

        p = result.observed_order

        # Richardson extrapolation
        if p > 0 and r21 > 1:
            result.extrapolated_value = (
                r21**p * f_fine - f_medium
            ) / (r21**p - 1)
        else:
            result.extrapolated_value = f_fine

        # GCI computation
        if p > 0 and r21 > 1:
            e_fine = abs((f_fine - f_medium) / f_fine) if abs(f_fine) > 1e-15 else 0
            e_medium = abs((f_medium - f_coarse) / f_medium) if abs(f_medium) > 1e-15 else 0

            result.gci_fine_pct = Fs * e_fine / (r21**p - 1) * 100
            result.gci_medium_pct = Fs * e_medium / (r32**p - 1) * 100
        else:
            result.gci_fine_pct = float("nan")
            result.gci_medium_pct = float("nan")

        # Asymptotic range ratio
        if result.gci_fine_pct > 0 and not math.isnan(result.gci_fine_pct):
            result.asymptotic_ratio = (
                result.gci_medium_pct / (r21**p * result.gci_fine_pct)
                if result.gci_fine_pct > 0 else 0
            )
            result.in_asymptotic_range = abs(result.asymptotic_ratio - 1.0) < 0.1
        else:
            result.asymptotic_ratio = 0.0
            result.in_asymptotic_range = False

        return result

    @staticmethod
    def _compute_order_iterative(
        eps_21: float,
        eps_32: float,
        r21: float,
        r32: float,
        max_iter: int = 50,
        tol: float = 1e-6,
    ) -> float:
        """
        Compute observed order using the iterative fixed-point method
        from Celik et al. (2008), Eq. 5.

        For non-uniform refinement ratios (r21 ≠ r32).
        """
        s = 1.0 if eps_32 / eps_21 > 0 else -1.0

        # Initial guess
        p = abs(math.log(abs(eps_32 / eps_21))) / math.log(r21)

        for _ in range(max_iter):
            if r21 != r32:
                q = math.log((r21**p - s) / (r32**p - s))
                p_new = abs(math.log(abs(eps_32 / eps_21)) + q) / math.log(r21)
            else:
                p_new = abs(math.log(abs(eps_32 / eps_21))) / math.log(r21)

            if abs(p_new - p) < tol:
                return max(p_new, 0.0)
            p = p_new

        return max(p, 0.0)

    def summary_table(self) -> str:
        """
        Generate formatted summary table matching TMR/Celik format.

        Returns
        -------
        str
            Formatted table string.
        """
        if not self._results:
            return "No results computed. Call compute() first."

        header = (
            f"{'Quantity':<20s} {'f_coarse':>10s} {'f_medium':>10s} {'f_fine':>10s} "
            f"{'p_obs':>6s} {'f_extrap':>10s} {'GCI_fine%':>9s} {'Asym.R':>7s} "
            f"{'Range?':>7s} {'Type':>12s}"
        )
        sep = "-" * len(header)
        lines = [header, sep]

        for name, r in self._results.items():
            gci_s = f"{r.gci_fine_pct:.3f}" if not math.isnan(r.gci_fine_pct) else "N/A"
            ar_s = f"{r.asymptotic_ratio:.3f}" if r.asymptotic_ratio != 0 else "N/A"
            range_s = "YES" if r.in_asymptotic_range else "NO"

            lines.append(
                f"{name:<20s} {r.f_coarse:>10.6f} {r.f_medium:>10.6f} "
                f"{r.f_fine:>10.6f} {r.observed_order:>6.2f} "
                f"{r.extrapolated_value:>10.6f} {gci_s:>9s} {ar_s:>7s} "
                f"{range_s:>7s} {r.convergence_type:>12s}"
            )

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize results to a dictionary."""
        output = {
            "refinement_ratios": {"r21": self.r21, "r32": self.r32},
            "safety_factor": self.safety_factor,
            "quantities": {},
        }
        for name, r in self._results.items():
            output["quantities"][name] = {
                "f_coarse": r.f_coarse,
                "f_medium": r.f_medium,
                "f_fine": r.f_fine,
                "observed_order": r.observed_order,
                "extrapolated_value": r.extrapolated_value,
                "gci_fine_pct": r.gci_fine_pct if not math.isnan(r.gci_fine_pct) else None,
                "gci_medium_pct": r.gci_medium_pct if not math.isnan(r.gci_medium_pct) else None,
                "asymptotic_ratio": r.asymptotic_ratio,
                "in_asymptotic_range": r.in_asymptotic_range,
                "convergence_type": r.convergence_type,
            }
        return output

    def to_json(self, path: Union[str, Path]) -> None:
        """Save results to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"GCI results saved to {path}")

    @property
    def all_converged(self) -> bool:
        """Check if all quantities have GCI < threshold."""
        from config import CONVERGENCE_CRITERIA
        threshold = CONVERGENCE_CRITERIA.get("gci_threshold", 0.05) * 100  # to %
        return all(
            r.gci_fine_pct < threshold
            for r in self._results.values()
            if not math.isnan(r.gci_fine_pct)
        )


def compute_from_cell_counts(
    n_coarse: int,
    n_medium: int,
    n_fine: int,
    ndim: int = 2,
) -> Tuple[float, float]:
    """
    Compute refinement ratios from cell counts.

    Parameters
    ----------
    n_coarse, n_medium, n_fine : int
        Cell counts on each grid level.
    ndim : int
        Problem dimensionality (2 or 3).

    Returns
    -------
    (r21, r32) : tuple of float
        Fine-to-medium and medium-to-coarse refinement ratios.
    """
    r21 = (n_fine / n_medium) ** (1.0 / ndim)
    r32 = (n_medium / n_coarse) ** (1.0 / ndim)
    return r21, r32


def extract_quantities_from_vtu(
    vtu_path: str,
    quantities: List[str] = None,
) -> Dict[str, np.ndarray]:
    """
    Extract surface data from SU2 VTU file.

    Parameters
    ----------
    vtu_path : str
        Path to surface_flow.vtu.
    quantities : list of str
        Field names to extract (default: Skin_Friction_Coefficient, Pressure_Coefficient).

    Returns
    -------
    dict mapping field name → 1D numpy array, plus 'x' coordinate.
    """
    try:
        import vtk
        from vtk.util.numpy_support import vtk_to_numpy
    except ImportError:
        raise ImportError("VTK is required for VTU extraction. Install with: pip install vtk")

    if quantities is None:
        quantities = ["Skin_Friction_Coefficient", "Pressure_Coefficient"]

    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(str(vtu_path))
    reader.Update()
    output = reader.GetOutput()

    points = output.GetPoints()
    coords = vtk_to_numpy(points.GetData())
    x = coords[:, 0]

    result = {"x": x}
    for qty in quantities:
        arr = output.GetPointData().GetArray(qty)
        if arr is not None:
            result[qty] = vtk_to_numpy(arr)
            if result[qty].ndim > 1:
                result[qty] = np.linalg.norm(result[qty], axis=1)

    return result


def compute_profile_gci(
    profiles_coarse: np.ndarray,
    profiles_medium: np.ndarray,
    profiles_fine: np.ndarray,
    r21: float = 2.0,
    r32: float = 2.0,
    safety_factor: float = 1.25,
) -> Dict[str, float]:
    """
    Compute GCI on full 1D profiles using L2-norm of differences.

    Parameters
    ----------
    profiles_coarse, profiles_medium, profiles_fine : ndarray
        1D profile arrays (must be interpolated to same grid).
    r21, r32 : float
        Refinement ratios.
    safety_factor : float
        GCI safety factor.

    Returns
    -------
    dict with L2-based GCI metrics.
    """
    eps_21 = np.linalg.norm(profiles_fine - profiles_medium)
    eps_32 = np.linalg.norm(profiles_medium - profiles_coarse)
    ref_norm = np.linalg.norm(profiles_fine)

    if ref_norm < 1e-15:
        return {"gci_fine_pct": 0.0, "observed_order": 0.0}

    if abs(eps_21) < 1e-15:
        return {"gci_fine_pct": 0.0, "observed_order": float("inf")}

    ratio = eps_32 / eps_21
    if ratio > 0:
        p = abs(math.log(ratio)) / math.log(r21)
    else:
        p = 1.0  # Default for oscillatory

    e_rel = eps_21 / ref_norm
    gci_fine = safety_factor * e_rel / (r21**p - 1) * 100 if r21**p > 1 else 0

    return {
        "gci_fine_pct": gci_fine,
        "observed_order": p,
        "eps_21_L2": eps_21,
        "eps_32_L2": eps_32,
    }


if __name__ == "__main__":
    # Demo: wall hump GCI study
    print("=== GCI Harness Demo ===\n")

    study = GCIStudy(r21=1.414, r32=1.414, safety_factor=1.25)
    study.add_quantity("x_sep", f_coarse=0.685, f_medium=0.672, f_fine=0.668)
    study.add_quantity("x_reat", f_coarse=1.18, f_medium=1.14, f_fine=1.12)
    study.add_quantity("Cp_RMSE", f_coarse=0.045, f_medium=0.032, f_fine=0.028)

    study.compute()
    print(study.summary_table())
    print(f"\nAll converged (GCI < 5%): {study.all_converged}")
