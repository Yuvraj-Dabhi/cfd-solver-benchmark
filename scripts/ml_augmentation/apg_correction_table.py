#!/usr/bin/env python3
"""
APG Separation-Onset Correction Table
=========================================
Fits a correction factor β = f(Clauser parameter, shape factor H)
for separation prediction improvement. Uses RANS vs experimental
data from benchmark cases.

Theory
------
The Clauser pressure gradient parameter β_c = (δ*/τ_w)(dp/dx)
and shape factor H = δ*/θ are the two most important local
parameters governing adverse pressure gradient boundary layers.

RANS models consistently over- or under-predict separation onset
in APG flows. This module builds a lookup table of correction
factors from validated cases that can be applied to new predictions.

Usage
-----
    from scripts.ml_augmentation.apg_correction_table import APGCorrectionTable

    table = APGCorrectionTable()
    table.build_from_cases(case_results)
    correction = table.lookup(clauser_param=5.0, shape_factor=2.5)
"""

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def compute_clauser_parameter(
    dp_dx: np.ndarray,
    tau_w: np.ndarray,
    delta_star: np.ndarray,
) -> np.ndarray:
    """
    Compute Clauser pressure gradient parameter.

    β_c = (δ*/τ_w) · (dp/dx)

    Parameters
    ----------
    dp_dx : ndarray
        Pressure gradient (Pa/m).
    tau_w : ndarray
        Wall shear stress (Pa).
    delta_star : ndarray
        Displacement thickness (m).

    Returns
    -------
    ndarray
        Clauser parameter β_c. Positive for APG.
    """
    tau_safe = np.where(np.abs(tau_w) > 1e-15, tau_w, 1e-15)
    return delta_star * dp_dx / tau_safe


def compute_shape_factor(
    delta_star: np.ndarray,
    theta: np.ndarray,
) -> np.ndarray:
    """
    Compute boundary layer shape factor.

    H = δ*/θ

    Parameters
    ----------
    delta_star : ndarray
        Displacement thickness.
    theta : ndarray
        Momentum thickness.

    Returns
    -------
    ndarray
        Shape factor H. H > 2.6 indicates near-separation for turbulent BLs.
    """
    theta_safe = np.where(np.abs(theta) > 1e-15, theta, 1e-15)
    return delta_star / theta_safe


@dataclass
class CorrectionEntry:
    """A single entry in the APG correction table."""
    case_name: str
    x_location: float
    clauser_param: float
    shape_factor: float
    cf_rans: float
    cf_exp: float
    correction_beta: float  # β = Cf_exp / Cf_RANS


class APGCorrectionTable:
    """
    Separation-onset correction lookup table.

    Builds a correction mapping from (β_c, H) → correction factor
    using validated CFD-vs-experiment data from benchmark cases.
    """

    def __init__(self):
        self.entries: List[CorrectionEntry] = []
        self._grid_clauser = None
        self._grid_H = None
        self._grid_correction = None

    def add_entry(
        self,
        case_name: str,
        x_location: float,
        clauser_param: float,
        shape_factor: float,
        cf_rans: float,
        cf_exp: float,
    ) -> None:
        """Add a data point to the correction table."""
        correction = cf_exp / cf_rans if abs(cf_rans) > 1e-15 else 1.0
        self.entries.append(CorrectionEntry(
            case_name=case_name,
            x_location=x_location,
            clauser_param=clauser_param,
            shape_factor=shape_factor,
            cf_rans=cf_rans,
            cf_exp=cf_exp,
            correction_beta=correction,
        ))

    def build_from_cases(
        self,
        case_results: Dict[str, Dict],
    ) -> None:
        """
        Build correction table from multiple benchmark case results.

        Parameters
        ----------
        case_results : dict
            Mapping case_name → dict with keys:
                'x': streamwise coordinates
                'clauser': Clauser parameter at each x
                'H': shape factor at each x
                'Cf_rans': RANS skin friction
                'Cf_exp': experimental skin friction
        """
        for case_name, data in case_results.items():
            x = data["x"]
            clauser = data["clauser"]
            H = data["H"]
            cf_rans = data["Cf_rans"]
            cf_exp = data["Cf_exp"]

            for i in range(len(x)):
                if abs(cf_rans[i]) > 1e-10:  # Skip zero Cf_rans
                    self.add_entry(
                        case_name=case_name,
                        x_location=float(x[i]),
                        clauser_param=float(clauser[i]),
                        shape_factor=float(H[i]),
                        cf_rans=float(cf_rans[i]),
                        cf_exp=float(cf_exp[i]),
                    )

        self._build_interpolator()

    def _build_interpolator(self) -> None:
        """Build 2D interpolation grid from entries."""
        if len(self.entries) < 3:
            return

        clauser_vals = np.array([e.clauser_param for e in self.entries])
        H_vals = np.array([e.shape_factor for e in self.entries])
        correction_vals = np.array([e.correction_beta for e in self.entries])

        self._clauser_vals = clauser_vals
        self._H_vals = H_vals
        self._correction_vals = correction_vals

    def lookup(
        self,
        clauser_param: float,
        shape_factor: float,
        n_neighbors: int = 5,
    ) -> float:
        """
        Look up correction factor for given (β_c, H).

        Uses inverse-distance-weighted (IDW) interpolation from
        the nearest entries in the table.

        Parameters
        ----------
        clauser_param : float
            Local Clauser parameter.
        shape_factor : float
            Local shape factor H.
        n_neighbors : int
            Number of nearest neighbors for IDW.

        Returns
        -------
        float
            Interpolated correction factor β.
        """
        if len(self.entries) == 0:
            return 1.0

        # Normalize coordinates for distance computation
        c_range = max(self._clauser_vals.max() - self._clauser_vals.min(), 1e-10)
        h_range = max(self._H_vals.max() - self._H_vals.min(), 1e-10)

        c_norm = (self._clauser_vals - self._clauser_vals.min()) / c_range
        h_norm = (self._H_vals - self._H_vals.min()) / h_range

        query_c = (clauser_param - self._clauser_vals.min()) / c_range
        query_h = (shape_factor - self._H_vals.min()) / h_range

        # Compute distances
        distances = np.sqrt((c_norm - query_c) ** 2 + (h_norm - query_h) ** 2)

        # IDW interpolation
        k = min(n_neighbors, len(self.entries))
        nearest_idx = np.argsort(distances)[:k]
        nearest_dist = distances[nearest_idx]
        nearest_corrections = self._correction_vals[nearest_idx]

        # Handle exact matches
        if nearest_dist[0] < 1e-10:
            return float(nearest_corrections[0])

        weights = 1.0 / nearest_dist
        correction = np.average(nearest_corrections, weights=weights)

        return float(correction)

    def apply_correction(
        self,
        x: np.ndarray,
        Cf_rans: np.ndarray,
        dp_dx: np.ndarray,
        tau_w: np.ndarray,
        delta_star: np.ndarray,
        theta: np.ndarray,
    ) -> np.ndarray:
        """
        Apply APG correction to RANS Cf prediction.

        Parameters
        ----------
        x : ndarray
            Streamwise coordinates.
        Cf_rans : ndarray
            RANS skin friction.
        dp_dx : ndarray
            Pressure gradient.
        tau_w : ndarray
            Wall shear stress.
        delta_star : ndarray
            Displacement thickness.
        theta : ndarray
            Momentum thickness.

        Returns
        -------
        ndarray
            Corrected skin friction coefficient.
        """
        clauser = compute_clauser_parameter(dp_dx, tau_w, delta_star)
        H = compute_shape_factor(delta_star, theta)

        Cf_corrected = np.copy(Cf_rans)
        for i in range(len(x)):
            beta = self.lookup(float(clauser[i]), float(H[i]))
            Cf_corrected[i] *= beta

        return Cf_corrected

    def summary(self) -> str:
        """Print summary of the correction table."""
        if not self.entries:
            return "Empty correction table."

        cases = set(e.case_name for e in self.entries)
        clauser_range = (
            min(e.clauser_param for e in self.entries),
            max(e.clauser_param for e in self.entries),
        )
        H_range = (
            min(e.shape_factor for e in self.entries),
            max(e.shape_factor for e in self.entries),
        )
        beta_range = (
            min(e.correction_beta for e in self.entries),
            max(e.correction_beta for e in self.entries),
        )

        return (
            f"APG Correction Table\n"
            f"  Entries: {len(self.entries)}\n"
            f"  Cases: {', '.join(sorted(cases))}\n"
            f"  Clauser range: [{clauser_range[0]:.2f}, {clauser_range[1]:.2f}]\n"
            f"  H range: [{H_range[0]:.2f}, {H_range[1]:.2f}]\n"
            f"  β correction range: [{beta_range[0]:.3f}, {beta_range[1]:.3f}]"
        )

    def to_json(self, path: str) -> None:
        """Save correction table to JSON."""
        data = {
            "n_entries": len(self.entries),
            "entries": [
                {
                    "case": e.case_name,
                    "x": e.x_location,
                    "clauser": e.clauser_param,
                    "H": e.shape_factor,
                    "Cf_rans": e.cf_rans,
                    "Cf_exp": e.cf_exp,
                    "beta": e.correction_beta,
                }
                for e in self.entries
            ],
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)


if __name__ == "__main__":
    print("=== APG Correction Table Demo ===\n")

    # Build from synthetic case data
    table = APGCorrectionTable()

    np.random.seed(42)
    n = 30
    case_results = {
        "wall_hump": {
            "x": np.linspace(0.5, 1.4, n),
            "clauser": np.linspace(0, 15, n),
            "H": np.linspace(1.4, 3.5, n),
            "Cf_rans": 0.003 * np.exp(-0.5 * np.linspace(0, 5, n)),
            "Cf_exp": 0.003 * np.exp(-0.5 * np.linspace(0, 5, n)) * (
                1 + 0.3 * np.linspace(0, 1, n)
            ),
        },
    }

    table.build_from_cases(case_results)
    print(table.summary())

    # Lookup
    beta = table.lookup(clauser_param=5.0, shape_factor=2.5)
    print(f"\nLookup(β_c=5.0, H=2.5) → correction β = {beta:.4f}")
