"""
NACA 0012 Transition Model Study
=================================
Systematic comparison of fully-turbulent vs γ-Reθ transition model
on NACA 0012 at α = 0° and α = 10°.

Investigates:
  - Whether transition changes CL/CD predictions
  - Laminar separation bubble effects on Cp
  - Impact on stall prediction

Usage:
    study = NACA0012TransitionStudy()
    configs = study.generate_all_configs(output_dir=Path("runs"))
    table = study.build_comparison_table(su2_results)
"""

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


@dataclass
class TransitionStudyConfig:
    """Configuration for one NACA 0012 transition study case."""
    alpha_deg: float
    model: str
    transition: bool  # True = γ-Reθ, False = fully turbulent
    label: str = ""

    def __post_init__(self):
        if not self.label:
            trans_label = "transition" if self.transition else "fully_turb"
            self.label = f"NACA0012_a{self.alpha_deg}_{self.model}_{trans_label}"


@dataclass
class TransitionResult:
    """Result from one transition study case."""
    config: TransitionStudyConfig
    cl: float = 0.0
    cd: float = 0.0
    cm: float = 0.0
    x_transition: float = 0.0  # Location of laminar-turbulent transition
    converged: bool = False


# Study matrix: α=0° and α=10° with SA, SST, γ-Reθ
STUDY_MATRIX = [
    # α = 0° — baseline verification
    TransitionStudyConfig(alpha_deg=0.0, model="SA", transition=False),
    TransitionStudyConfig(alpha_deg=0.0, model="SST", transition=False),
    TransitionStudyConfig(alpha_deg=0.0, model="gammaReTheta", transition=True),
    # α = 10° — near-stall regime
    TransitionStudyConfig(alpha_deg=10.0, model="SA", transition=False),
    TransitionStudyConfig(alpha_deg=10.0, model="SST", transition=False),
    TransitionStudyConfig(alpha_deg=10.0, model="gammaReTheta", transition=True),
]

# TMR reference values for comparison
TMR_NACA0012_REFERENCE = {
    "alpha_0_CFL3D_SA": {"CL": 0.0, "CD": 0.00819, "CM": 0.0, "source": "TMR 2DN00"},
    "alpha_10_CFL3D_SA": {"CL": 1.0909, "CD": 0.01231, "source": "TMR 2DN00"},
    "alpha_10_FUN3D_SA": {"CL": 1.0912, "CD": 0.01222, "source": "TMR 2DN00"},
}

# Experimental reference
EXPERIMENTAL_REF = {
    "alpha_0": {"CL": 0.0, "CD": 0.0082, "source": "Gregory & O'Reilly (1970)"},
    "alpha_10": {"CL": 1.09, "CD": 0.0120, "source": "Gregory & O'Reilly (1970)"},
}


class NACA0012TransitionStudy:
    """
    Generate and compare transition vs fully-turbulent results for NACA 0012.
    """

    def __init__(self, study_matrix: Optional[List[TransitionStudyConfig]] = None):
        self.matrix = study_matrix or STUDY_MATRIX

    def generate_all_configs(
        self,
        output_dir: Path,
        mesh_file: str = "mesh_naca0012.su2",
    ) -> Dict[str, Path]:
        """Generate SU2 configs for all study cases."""
        from scripts.solvers.su2_runner import SU2Runner

        configs = {}
        for cfg in self.matrix:
            case_dir = output_dir / cfg.label
            runner = SU2Runner(
                case_dir=case_dir,
                case_name="naca_0012_stall",
                model=cfg.model,
                mesh_file=mesh_file,
            )
            runner.setup_case()

            # Patch AOA in the generated config
            config_path = case_dir / "config.cfg"
            self._patch_aoa(config_path, cfg.alpha_deg)

            configs[cfg.label] = config_path
            logger.info(f"  Generated: {cfg.label}")

        return configs

    def _patch_aoa(self, config_path: Path, alpha: float):
        """Patch angle of attack in generated SU2 config."""
        if not config_path.exists():
            return
        text = config_path.read_text()
        text = text.replace("AOA= 0.0", f"AOA= {alpha}")
        config_path.write_text(text)

    def build_comparison_table(
        self,
        su2_results: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> str:
        """Build markdown comparison table."""
        lines = [
            "## NACA 0012 Transition Study",
            "",
            "| α (°) | Model | Type | CL | CD | ΔCL vs Exp | ΔCD vs Exp | Source |",
            "|-------|-------|------|-----|-----|------------|------------|--------|",
        ]

        for cfg in self.matrix:
            alpha_key = f"alpha_{int(cfg.alpha_deg)}"
            exp = EXPERIMENTAL_REF.get(alpha_key, {})
            exp_cl = exp.get("CL", 0)
            exp_cd = exp.get("CD", 0)

            if su2_results and cfg.label in su2_results:
                r = su2_results[cfg.label]
                cl, cd = r.get("CL", 0), r.get("CD", 0)
                dcl = cl - exp_cl if exp_cl else 0
                dcd = cd - exp_cd if exp_cd else 0
                trans_type = "γ-Reθ" if cfg.transition else "Fully Turb"
                lines.append(
                    f"| {cfg.alpha_deg:.0f} | {cfg.model} | {trans_type} | "
                    f"{cl:.4f} | {cd:.5f} | {dcl:+.4f} | {dcd:+.5f} | SU2 |"
                )

        # Add TMR reference rows
        lines.append("")
        lines.append("### TMR Reference (CFL3D/FUN3D)")
        lines.append("")
        lines.append("| α (°) | Solver | CL | CD | Source |")
        lines.append("|-------|--------|-----|-----|--------|")
        for key, ref in TMR_NACA0012_REFERENCE.items():
            alpha = "0" if "alpha_0" in key else "10"
            solver = key.split("_")[-2] + "_" + key.split("_")[-1]
            lines.append(
                f"| {alpha} | {solver} | {ref.get('CL', 0):.4f} | "
                f"{ref.get('CD', 0):.5f} | {ref['source']} |"
            )

        lines.extend([
            "",
            "### Key Questions",
            "1. Does γ-Reθ change CL at α=0°? (Should be minimal — fully turbulent at Re=6M)",
            "2. Does γ-Reθ change drag at α=0°? (Yes — transition delays skin friction onset)",
            "3. Does γ-Reθ change stall prediction at α=10°? (Possible — laminar bubble affects TE separation)",
        ])

        return "\n".join(lines)

    def get_study_summary(self) -> str:
        """Get study configuration summary."""
        lines = [
            "NACA 0012 Transition Study Configuration",
            "=" * 50,
            f"Cases: {len(self.matrix)}",
        ]
        for cfg in self.matrix:
            trans = "γ-Reθ" if cfg.transition else "Fully Turb"
            lines.append(f"  α={cfg.alpha_deg:5.1f}°  {cfg.model:<15s} ({trans})")
        return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    study = NACA0012TransitionStudy()
    print(study.get_study_summary())

    # Generate configs
    output = PROJECT_ROOT / "runs" / "naca0012_transition"
    configs = study.generate_all_configs(output)
    print(f"\nGenerated {len(configs)} configs")

    # Demo table with hypothetical results
    demo_results = {
        "NACA0012_a0.0_SA_fully_turb": {"CL": 0.0000, "CD": 0.00825},
        "NACA0012_a0.0_SST_fully_turb": {"CL": 0.0001, "CD": 0.00818},
        "NACA0012_a0.0_gammaReTheta_transition": {"CL": 0.0000, "CD": 0.00780},
        "NACA0012_a10.0_SA_fully_turb": {"CL": 1.089, "CD": 0.01235},
        "NACA0012_a10.0_SST_fully_turb": {"CL": 1.085, "CD": 0.01240},
        "NACA0012_a10.0_gammaReTheta_transition": {"CL": 1.092, "CD": 0.01210},
    }
    print(study.build_comparison_table(demo_results))
