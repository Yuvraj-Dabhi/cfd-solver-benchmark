"""
Boundary-Layer Profile Analyzer
================================
Deep boundary-layer diagnostics beyond surface Cf/Cp:
  - Inner-scaled profiles: U⁺(y⁺), k⁺(y⁺), ω⁺(y⁺) at user-specified stations
  - Log-law deviation diagnostic
  - TKE budget: production, dissipation, turbulent transport
  - Clauser parameter β (pressure gradient severity)
  - Shape factor H = δ*/θ (separation proximity)

Usage:
    analyzer = BLProfileAnalyzer(nu=1.5e-5)
    report = analyzer.analyze_station(y, U, k, omega, dpdx, rho=1.225)
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Log-law constants (von Kármán)
KAPPA = 0.41
B_CONSTANT = 5.0


@dataclass
class BLStationReport:
    """Diagnostic report for one boundary-layer station."""
    x_station: float = 0.0
    station_label: str = ""

    # Integral quantities
    delta_99: float = 0.0          # BL thickness [m]
    delta_star: float = 0.0        # Displacement thickness [m]
    theta: float = 0.0             # Momentum thickness [m]
    shape_factor_H: float = 0.0    # δ*/θ
    Re_theta: float = 0.0          # Momentum thickness Reynolds number

    # Wall quantities
    u_tau: float = 0.0             # Friction velocity [m/s]
    Cf_local: float = 0.0          # Local skin friction
    y_plus_first: float = 0.0      # y⁺ of first cell

    # Pressure gradient
    clauser_beta: float = 0.0      # Clauser parameter
    dpdx: float = 0.0              # dp/dx [Pa/m]

    # Log-law diagnostic
    log_law_max_deviation_pct: float = 0.0
    log_law_region_yplus: Tuple[float, float] = (30, 300)

    # Separation status
    is_separated: bool = False
    separation_status: str = "ATTACHED"

    # Profile DataFrames
    inner_scaled: Optional[pd.DataFrame] = None
    tke_budget: Optional[pd.DataFrame] = None


@dataclass
class BLAnalysisReport:
    """Collection of station analyses for one case."""
    case_name: str = ""
    stations: Dict[str, BLStationReport] = field(default_factory=dict)
    overall_status: str = ""

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Boundary-Layer Analysis: {self.case_name}",
            "=" * 60,
            f"{'Station':<15} {'H':<8} {'β':<10} {'Cf':<10} {'Re_θ':<10} {'Status'}",
            "-" * 60,
        ]
        for label, s in self.stations.items():
            lines.append(
                f"{label:<15} {s.shape_factor_H:<8.3f} "
                f"{s.clauser_beta:<10.3f} {s.Cf_local:<10.6f} "
                f"{s.Re_theta:<10.0f} {s.separation_status}"
            )
        return "\n".join(lines)


class BLProfileAnalyzer:
    """
    Analyze boundary-layer velocity/turbulence profiles at specified stations.

    Parameters
    ----------
    nu : float
        Kinematic viscosity [m²/s].
    U_inf : float
        Freestream velocity [m/s].
    rho : float
        Reference density [kg/m³].
    """

    def __init__(
        self,
        nu: float = 1.5e-5,
        U_inf: float = 1.0,
        rho: float = 1.225,
    ):
        self.nu = nu
        self.U_inf = U_inf
        self.rho = rho

    def analyze_station(
        self,
        y: np.ndarray,
        U: np.ndarray,
        k: Optional[np.ndarray] = None,
        omega: Optional[np.ndarray] = None,
        epsilon: Optional[np.ndarray] = None,
        dpdx: float = 0.0,
        x_station: float = 0.0,
        station_label: str = "",
    ) -> BLStationReport:
        """
        Full boundary-layer diagnostic at one station.

        Parameters
        ----------
        y : array
            Wall-normal coordinate [m] (y=0 at wall).
        U : array
            Streamwise velocity [m/s].
        k : array, optional
            Turbulent kinetic energy [m²/s²].
        omega : array, optional
            Specific dissipation rate [1/s].
        epsilon : array, optional
            Dissipation rate [m²/s³]. If not given, computed from k and omega.
        dpdx : float
            Streamwise pressure gradient [Pa/m].
        x_station : float
            Streamwise location.
        station_label : str
            Label for the station (e.g., "x/c = 0.65").
        """
        report = BLStationReport(
            x_station=x_station,
            station_label=station_label,
            dpdx=dpdx,
        )

        # Ensure sorted by y
        sort_idx = np.argsort(y)
        y = y[sort_idx]
        U = U[sort_idx]
        if k is not None:
            k = k[sort_idx]
        if omega is not None:
            omega = omega[sort_idx]
        if epsilon is not None:
            epsilon = epsilon[sort_idx]

        # --- Wall quantities ---
        tau_w = self._estimate_wall_shear(y, U)
        report.u_tau = np.sqrt(np.abs(tau_w) / self.rho)
        report.Cf_local = tau_w / (0.5 * self.rho * self.U_inf ** 2)
        report.is_separated = tau_w < 0
        report.separation_status = "SEPARATED" if report.is_separated else "ATTACHED"

        # --- Integral quantities ---
        U_edge = self._find_edge_velocity(U)
        report.delta_99 = self._compute_delta99(y, U, U_edge)
        report.delta_star = self._compute_delta_star(y, U, U_edge)
        report.theta = self._compute_theta(y, U, U_edge)
        report.shape_factor_H = (
            report.delta_star / report.theta if report.theta > 1e-15 else 0.0
        )
        report.Re_theta = self.U_inf * report.theta / self.nu

        # --- Clauser parameter ---
        if abs(tau_w) > 1e-15:
            report.clauser_beta = (report.delta_star / tau_w) * dpdx
        else:
            report.clauser_beta = float("inf") if dpdx != 0 else 0.0

        # --- Inner-scaled profiles ---
        if report.u_tau > 1e-15:
            y_plus = y * report.u_tau / self.nu
            U_plus = U / report.u_tau
            report.y_plus_first = y_plus[0] if len(y_plus) > 0 else 0.0

            inner_data = {"y_plus": y_plus, "U_plus": U_plus}
            if k is not None:
                inner_data["k_plus"] = k / report.u_tau ** 2
            if omega is not None:
                inner_data["omega_plus"] = omega * self.nu / report.u_tau ** 2

            report.inner_scaled = pd.DataFrame(inner_data)

            # Log-law deviation
            report.log_law_max_deviation_pct = self._log_law_deviation(
                y_plus, U_plus
            )

        # --- TKE budget ---
        if k is not None and (epsilon is not None or omega is not None):
            report.tke_budget = self._compute_tke_budget(
                y, U, k, omega, epsilon
            )

        # Classify separation proximity from shape factor
        if report.shape_factor_H > 3.5:
            report.separation_status = "SEPARATED (H > 3.5)"
        elif report.shape_factor_H > 2.5:
            report.separation_status = "NEAR-SEPARATION (H > 2.5)"

        return report

    def analyze_case(
        self,
        stations: Dict[str, Dict[str, np.ndarray]],
        case_name: str = "",
    ) -> BLAnalysisReport:
        """
        Analyze multiple stations for one benchmark case.

        Parameters
        ----------
        stations : dict
            {station_label: {"y": array, "U": array, "k": array, ...}}
        case_name : str
            Benchmark case name.
        """
        report = BLAnalysisReport(case_name=case_name)
        for label, data in stations.items():
            station = self.analyze_station(
                y=data["y"],
                U=data["U"],
                k=data.get("k"),
                omega=data.get("omega"),
                epsilon=data.get("epsilon"),
                dpdx=data.get("dpdx", 0.0),
                x_station=data.get("x_station", 0.0),
                station_label=label,
            )
            report.stations[label] = station

        # Overall status
        n_sep = sum(1 for s in report.stations.values() if s.is_separated)
        if n_sep == 0:
            report.overall_status = "FULLY ATTACHED"
        elif n_sep == len(report.stations):
            report.overall_status = "FULLY SEPARATED"
        else:
            report.overall_status = f"PARTIAL SEPARATION ({n_sep}/{len(report.stations)} stations)"

        return report

    # =========================================================================
    # Internal computations
    # =========================================================================
    def _estimate_wall_shear(self, y: np.ndarray, U: np.ndarray) -> float:
        """Estimate wall shear stress from first two points."""
        if len(y) < 2:
            return 0.0
        dy = y[1] - y[0]
        if dy < 1e-15:
            return 0.0
        dUdy = (U[1] - U[0]) / dy
        return self.rho * self.nu * dUdy

    def _find_edge_velocity(self, U: np.ndarray) -> float:
        """Estimate BL edge velocity (99% of freestream)."""
        if len(U) == 0:
            return self.U_inf
        return max(np.max(U), self.U_inf)

    def _compute_delta99(
        self, y: np.ndarray, U: np.ndarray, U_edge: float
    ) -> float:
        """BL thickness where U = 0.99 * U_edge."""
        threshold = 0.99 * U_edge
        above = np.where(U >= threshold)[0]
        if len(above) > 0:
            return float(y[above[0]])
        return float(y[-1])

    def _compute_delta_star(
        self, y: np.ndarray, U: np.ndarray, U_edge: float
    ) -> float:
        """Displacement thickness: δ* = ∫(1 - U/U_e) dy."""
        if U_edge < 1e-15:
            return 0.0
        integrand = 1.0 - U / U_edge
        return float(np.trapezoid(np.maximum(integrand, 0), y))

    def _compute_theta(
        self, y: np.ndarray, U: np.ndarray, U_edge: float
    ) -> float:
        """Momentum thickness: θ = ∫(U/U_e)(1 - U/U_e) dy."""
        if U_edge < 1e-15:
            return 0.0
        u_ratio = U / U_edge
        integrand = u_ratio * (1.0 - u_ratio)
        return float(np.trapezoid(np.maximum(integrand, 0), y))

    def _log_law_deviation(
        self,
        y_plus: np.ndarray,
        U_plus: np.ndarray,
        yp_min: float = 30,
        yp_max: float = 300,
    ) -> float:
        """
        Compute maximum deviation from log law in the log-layer region.

        Returns deviation as percentage.
        """
        mask = (y_plus >= yp_min) & (y_plus <= yp_max)
        if not np.any(mask):
            return 0.0

        yp_log = y_plus[mask]
        up_cfd = U_plus[mask]
        up_log = (1.0 / KAPPA) * np.log(yp_log) + B_CONSTANT

        deviation = np.abs(up_cfd - up_log) / np.maximum(np.abs(up_log), 1e-10) * 100
        return float(np.max(deviation))

    def _compute_tke_budget(
        self,
        y: np.ndarray,
        U: np.ndarray,
        k: np.ndarray,
        omega: Optional[np.ndarray] = None,
        epsilon: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """
        Compute TKE budget terms: production P_k, dissipation ε.

        P_k = -<u'v'> * dU/dy ≈ ν_t * (dU/dy)²
        """
        dUdy = np.gradient(U, y)

        # Dissipation
        if epsilon is not None:
            eps = epsilon
        elif omega is not None:
            # ε = C_mu * k * ω  (with C_mu = 0.09 for k-ω)
            eps = 0.09 * k * omega
        else:
            eps = np.zeros_like(k)

        # Turbulent viscosity: ν_t = k / ω (simplified)
        if omega is not None:
            nu_t = k / np.maximum(omega, 1e-15)
        else:
            nu_t = 0.09 * k ** 2 / np.maximum(eps, 1e-15)

        # Production
        P_k = nu_t * dUdy ** 2

        # P/ε ratio
        P_eps_ratio = P_k / np.maximum(eps, 1e-15)

        return pd.DataFrame({
            "y": y,
            "P_k": P_k,
            "epsilon": eps,
            "nu_t": nu_t,
            "P_eps_ratio": P_eps_ratio,
        })


def compute_shape_factor(
    y: np.ndarray, U: np.ndarray, U_inf: float
) -> float:
    """Quick shape factor computation: H = δ*/θ."""
    u_ratio = U / max(U_inf, 1e-15)
    delta_star = np.trapezoid(np.maximum(1.0 - u_ratio, 0), y)
    theta = np.trapezoid(np.maximum(u_ratio * (1.0 - u_ratio), 0), y)
    return delta_star / max(theta, 1e-15)


def compute_clauser_beta(
    delta_star: float, tau_w: float, dpdx: float
) -> float:
    """Clauser pressure gradient parameter: β = (δ*/τ_w) * dp/dx."""
    if abs(tau_w) < 1e-15:
        return float("inf") if dpdx != 0 else 0.0
    return (delta_star / tau_w) * dpdx
