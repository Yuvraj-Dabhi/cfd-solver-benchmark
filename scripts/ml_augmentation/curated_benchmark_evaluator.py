#!/usr/bin/env python3
"""
Curated Benchmark Evaluator & Metrics Contract
===============================================
Standardizes the computation of prediction metrics (RMSE, MAE, physics-
constraint violations) across different ML closures and baseline RANS models,
mirroring the style of McConkey et al. (2021) "A curated dataset for
data-driven turbulence modelling."

For a given set of predictions and targets, this module computes:
  - URMS / UMAE: RMS / MAE error in mean velocity field (Ux, Uy)
  - kRMS / kMAE: RMS / MAE error in turbulent kinetic energy
  - uvRMS, uuRMS, vvRMS: Component-wise Reynolds-stress errors
  - realizability_violation: Lumley triangle bound violation fraction
  - separation_error: Bubble-length / reattachment-point errors from Cf

The `BenchmarkMetricsContract` provides a plug-in API that allows external
models to be registered and evaluated with identical metrics, enabling
standardised comparison across research groups.
"""

import json
import logging
import csv
import numpy as np
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


# =====================================================================
# Core Evaluator
# =====================================================================

class CuratedBenchmarkEvaluator:
    """
    Evaluator for ML closures against the McConkey curated turbulence dataset.

    Computes RMSE, MAE, physics-constraint metrics, and separation metrics.
    """

    def __init__(self, target_names: List[str]):
        """
        Parameters
        ----------
        target_names : list of str
            The ordered list of variables present in the `targets` array.
            Expected to contain combinations of:
            "Ux", "Uy", "k_dns", "uu_dns", "uv_dns", "vv_dns", "Cp", "Cf".
        """
        self.target_names = target_names
        self._var_indices = {name: i for i, name in enumerate(target_names)}

    # ----- Core metric helpers -----

    def _compute_rmse(self, pred: np.ndarray, target: np.ndarray) -> float:
        """Compute Root Mean Squared Error, ignoring NaNs."""
        mask = np.isfinite(pred) & np.isfinite(target)
        if not np.any(mask):
            return float('nan')
        return float(np.sqrt(np.mean((pred[mask] - target[mask])**2)))

    def _compute_mae(self, pred: np.ndarray, target: np.ndarray) -> float:
        """Compute Mean Absolute Error, ignoring NaNs."""
        mask = np.isfinite(pred) & np.isfinite(target)
        if not np.any(mask):
            return float('nan')
        return float(np.mean(np.abs(pred[mask] - target[mask])))

    # ----- Physics constraint metrics -----

    def _compute_realizability_violation(
        self,
        predictions: np.ndarray,
    ) -> float:
        """
        Check Lumley triangle realizability bounds on predicted Reynolds stresses.

        The Reynolds-stress tensor must satisfy:
          - Diagonal components (uu, vv, ww) >= 0
          - 2*k = uu + vv + ww > 0
          - |uv| <= sqrt(uu * vv)  (Cauchy-Schwarz)

        Returns
        -------
        float
            Fraction of points violating at least one realizability constraint.
        """
        has_uu = "uu_dns" in self._var_indices
        has_vv = "vv_dns" in self._var_indices
        has_uv = "uv_dns" in self._var_indices

        if not (has_uu and has_vv):
            return float('nan')

        uu = predictions[:, self._var_indices["uu_dns"]]
        vv = predictions[:, self._var_indices["vv_dns"]]

        n = len(uu)
        violations = np.zeros(n, dtype=bool)

        # Positivity of normal stresses
        violations |= (uu < -1e-12)
        violations |= (vv < -1e-12)

        # Cauchy-Schwarz on shear stress
        if has_uv:
            uv = predictions[:, self._var_indices["uv_dns"]]
            safe_product = np.maximum(uu, 0) * np.maximum(vv, 0)
            violations |= (uv**2 > safe_product + 1e-12)

        valid = np.isfinite(uu) & np.isfinite(vv)
        if not np.any(valid):
            return float('nan')

        return float(np.sum(violations[valid]) / np.sum(valid))

    def _compute_separation_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        x_coords: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Compute separation-specific metrics from Cf predictions.

        Detects separation point (Cf sign change negative) and reattachment
        (Cf sign change positive) from wall-friction coefficient distributions.

        Parameters
        ----------
        predictions, targets : ndarray (N, n_vars)
        x_coords : ndarray (N,), optional
            Streamwise coordinate for Cf-based detection.

        Returns
        -------
        dict with keys: 'Cf_RMSE', 'Cf_MAE', and optionally 'sep_point_error',
            'reat_point_error', 'bubble_length_error'.
        """
        result = {}

        if "Cf" not in self._var_indices:
            return result

        idx = self._var_indices["Cf"]
        cf_pred = predictions[:, idx]
        cf_targ = targets[:, idx]

        result["Cf_RMSE"] = self._compute_rmse(cf_pred, cf_targ)
        result["Cf_MAE"] = self._compute_mae(cf_pred, cf_targ)

        if x_coords is not None and len(x_coords) == len(cf_pred):
            # Sort by x
            order = np.argsort(x_coords)
            x_s = x_coords[order]
            cf_p = cf_pred[order]
            cf_t = cf_targ[order]

            def _find_sign_changes(cf_arr, x_arr):
                """Find x-locations where Cf changes sign."""
                seps, reats = [], []
                for i in range(len(cf_arr) - 1):
                    if cf_arr[i] >= 0 and cf_arr[i + 1] < 0:
                        # Linear interpolation for separation
                        frac = cf_arr[i] / (cf_arr[i] - cf_arr[i + 1] + 1e-30)
                        seps.append(x_arr[i] + frac * (x_arr[i + 1] - x_arr[i]))
                    elif cf_arr[i] < 0 and cf_arr[i + 1] >= 0:
                        frac = -cf_arr[i] / (cf_arr[i + 1] - cf_arr[i] + 1e-30)
                        reats.append(x_arr[i] + frac * (x_arr[i + 1] - x_arr[i]))
                return seps, reats

            seps_p, reats_p = _find_sign_changes(cf_p, x_s)
            seps_t, reats_t = _find_sign_changes(cf_t, x_s)

            if seps_p and seps_t:
                result["sep_point_error"] = abs(seps_p[0] - seps_t[0])
            if reats_p and reats_t:
                result["reat_point_error"] = abs(reats_p[0] - reats_t[0])
            if seps_p and reats_p and seps_t and reats_t:
                bubble_pred = reats_p[0] - seps_p[0]
                bubble_targ = reats_t[0] - seps_t[0]
                result["bubble_length_error"] = abs(bubble_pred - bubble_targ)

        return result

    # ----- Main evaluation entry point -----

    def evaluate_predictions(
        self,
        model_name: str,
        predictions: np.ndarray,
        targets: np.ndarray,
        case_labels: Optional[List[str]] = None,
        x_coords: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate full predictions array against targets.

        Parameters
        ----------
        model_name : str
            Name of the model being evaluated.
        predictions : ndarray (N, n_targets)
            Predicted values matching the target variables.
        targets : ndarray (N, n_targets)
            Ground truth DNS/LES values.
        case_labels : list of str, optional
            If provided, also compute breakdown of errors per geometry.
        x_coords : ndarray (N,), optional
            Streamwise coordinates for separation metric computation.

        Returns
        -------
        results : dict
            Dictionary containing RMSE, MAE, physics metrics, and
            optionally per-case breakdowns.
        """
        if predictions.shape != targets.shape:
            raise ValueError(
                f"Prediction shape {predictions.shape} must match "
                f"target shape {targets.shape}"
            )

        res = {"model": model_name}

        # Variable-level metrics
        field_metrics = [
            ("Ux", "URMS_x", "UMAE_x"),
            ("Uy", "URMS_y", "UMAE_y"),
            ("k_dns", "kRMS", "kMAE"),
            ("uu_dns", "uuRMS", "uuMAE"),
            ("uv_dns", "uvRMS", "uvMAE"),
            ("vv_dns", "vvRMS", "vvMAE"),
        ]

        overall = {}
        for var, rms_name, mae_name in field_metrics:
            if var in self._var_indices:
                idx = self._var_indices[var]
                overall[rms_name] = self._compute_rmse(
                    predictions[:, idx], targets[:, idx]
                )
                overall[mae_name] = self._compute_mae(
                    predictions[:, idx], targets[:, idx]
                )

        # Combined velocity magnitude RMSE/MAE
        if "Ux" in self._var_indices and "Uy" in self._var_indices:
            ix = self._var_indices["Ux"]
            iy = self._var_indices["Uy"]
            vel_pred = np.sqrt(predictions[:, ix]**2 + predictions[:, iy]**2)
            vel_targ = np.sqrt(targets[:, ix]**2 + targets[:, iy]**2)
            overall["U_mag_RMS"] = self._compute_rmse(vel_pred, vel_targ)
            overall["U_mag_MAE"] = self._compute_mae(vel_pred, vel_targ)

        # Physics-constraint metrics
        overall["realizability_violation"] = (
            self._compute_realizability_violation(predictions)
        )

        # Separation metrics
        sep_metrics = self._compute_separation_metrics(
            predictions, targets, x_coords
        )
        overall.update(sep_metrics)

        res["overall"] = overall

        # Per-case metrics
        if case_labels is not None:
            if len(case_labels) != len(targets):
                logger.warning(
                    "Length of case_labels does not match targets. "
                    "Skipping per-case breakdown."
                )
            else:
                per_case = {}
                case_labels_arr = np.array(case_labels)

                for case in np.unique(case_labels_arr):
                    mask = case_labels_arr == case
                    case_metrics = {}

                    for var, rms_name, mae_name in field_metrics:
                        if var in self._var_indices:
                            idx = self._var_indices[var]
                            case_metrics[rms_name] = self._compute_rmse(
                                predictions[mask, idx], targets[mask, idx]
                            )
                            case_metrics[mae_name] = self._compute_mae(
                                predictions[mask, idx], targets[mask, idx]
                            )

                    if "U_mag_RMS" in overall:
                        vp = np.sqrt(
                            predictions[mask, ix]**2 + predictions[mask, iy]**2
                        )
                        vt = np.sqrt(
                            targets[mask, ix]**2 + targets[mask, iy]**2
                        )
                        case_metrics["U_mag_RMS"] = self._compute_rmse(vp, vt)
                        case_metrics["U_mag_MAE"] = self._compute_mae(vp, vt)

                    case_metrics["realizability_violation"] = (
                        self._compute_realizability_violation(predictions[mask])
                    )
                    per_case[case] = case_metrics

                res["per_case"] = per_case

        return res

    # ----- Formatting -----

    def format_as_markdown_table(
        self,
        evaluations: List[Dict],
        metrics_to_show: Optional[List[str]] = None,
    ) -> str:
        """Format evaluations as a Markdown table."""
        if not evaluations:
            return "*No evaluations provided.*"

        if metrics_to_show is None:
            metrics_to_show = [
                "U_mag_RMS", "U_mag_MAE", "kRMS",
                "uuRMS", "uvRMS", "vvRMS",
                "realizability_violation",
            ]

        # Filter to metrics present in at least one evaluation
        actual_metrics = []
        for m in metrics_to_show:
            for ev in evaluations:
                if m in ev.get("overall", {}):
                    if m not in actual_metrics:
                        actual_metrics.append(m)

        if not actual_metrics:
            return "*No matching metrics found.*"

        lines = []
        header = "| Model | " + " | ".join(actual_metrics) + " |"
        sep = "| :--- | " + " | ".join([":---:"] * len(actual_metrics)) + " |"
        lines.append(header)
        lines.append(sep)

        for ev in evaluations:
            model = ev["model"]
            overall = ev.get("overall", {})
            vals = []
            for m in actual_metrics:
                val = overall.get(m, float('nan'))
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    vals.append("-")
                else:
                    vals.append(f"{val:.5f}")
            lines.append(f"| **{model}** | " + " | ".join(vals) + " |")

        return "\n".join(lines)


# =====================================================================
# Benchmark Metrics Contract — Plug-in API
# =====================================================================

class BenchmarkMetricsContract:
    """
    Standardised plug-in API for evaluating external ML models against
    the curated turbulence benchmark.

    Usage
    -----
    >>> contract = BenchmarkMetricsContract(target_names=["Ux", "Uy", ...])
    >>> contract.register_model("MyModel", my_predict_fn)
    >>> results = contract.evaluate_all(test_features, test_targets)
    >>> contract.export_results(Path("results"), format="json")

    Where ``my_predict_fn(features: np.ndarray) -> np.ndarray`` maps
    (N, n_features) → (N, n_targets).
    """

    def __init__(self, target_names: List[str]):
        """
        Parameters
        ----------
        target_names : list of str
            Ordered variable names matching the target columns.
        """
        self.target_names = target_names
        self._evaluator = CuratedBenchmarkEvaluator(target_names)
        self._models: Dict[str, Callable] = {}
        self._results: List[Dict] = []

    def register_model(
        self,
        name: str,
        predict_fn: Callable[[np.ndarray], np.ndarray],
    ) -> None:
        """
        Register an external model for evaluation.

        Parameters
        ----------
        name : str
            Model identifier (e.g., "MyTBNN_v2").
        predict_fn : callable
            Function mapping features (N, n_features) to predictions
            (N, n_targets).
        """
        if name in self._models:
            logger.warning(f"Model '{name}' already registered, overwriting.")
        self._models[name] = predict_fn
        logger.info(f"Registered model '{name}' for benchmark evaluation.")

    def list_models(self) -> List[str]:
        """Return names of all registered models."""
        return list(self._models.keys())

    def evaluate_model(
        self,
        name: str,
        features: np.ndarray,
        targets: np.ndarray,
        case_labels: Optional[List[str]] = None,
        x_coords: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate a single registered model.

        Parameters
        ----------
        name : str
            Registered model name.
        features : ndarray (N, n_features)
        targets : ndarray (N, n_targets)
        case_labels : list of str, optional
        x_coords : ndarray (N,), optional

        Returns
        -------
        dict
            Evaluation results.
        """
        if name not in self._models:
            raise KeyError(f"Model '{name}' not registered.")

        predict_fn = self._models[name]
        predictions = predict_fn(features)

        return self._evaluator.evaluate_predictions(
            model_name=name,
            predictions=predictions,
            targets=targets,
            case_labels=case_labels,
            x_coords=x_coords,
        )

    def evaluate_all(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        case_labels: Optional[List[str]] = None,
        x_coords: Optional[np.ndarray] = None,
    ) -> List[Dict[str, Any]]:
        """
        Evaluate all registered models and store results.

        Returns
        -------
        list of dict
            One evaluation dict per model.
        """
        self._results = []
        for name in self._models:
            try:
                ev = self.evaluate_model(
                    name, features, targets, case_labels, x_coords
                )
                self._results.append(ev)
                logger.info(f"Evaluated '{name}' successfully.")
            except Exception as e:
                logger.error(f"Evaluation of '{name}' failed: {e}")
                self._results.append({"model": name, "error": str(e)})
        return self._results

    def evaluate_predictions_direct(
        self,
        model_name: str,
        predictions: np.ndarray,
        targets: np.ndarray,
        case_labels: Optional[List[str]] = None,
        x_coords: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate pre-computed predictions (no registration required).

        This is a convenience method for models that are already trained
        and produce predictions outside the contract framework.
        """
        ev = self._evaluator.evaluate_predictions(
            model_name=model_name,
            predictions=predictions,
            targets=targets,
            case_labels=case_labels,
            x_coords=x_coords,
        )
        self._results.append(ev)
        return ev

    def get_results(self) -> List[Dict]:
        """Return all accumulated evaluation results."""
        return list(self._results)

    def format_results_markdown(
        self,
        metrics_to_show: Optional[List[str]] = None,
    ) -> str:
        """Format accumulated results as Markdown table."""
        return self._evaluator.format_as_markdown_table(
            self._results, metrics_to_show
        )

    # ----- Export -----

    def export_results(
        self,
        output_dir: Path,
        fmt: str = "json",
    ) -> Path:
        """
        Export evaluation results.

        Parameters
        ----------
        output_dir : Path
            Directory for output files.
        fmt : str
            One of ``"json"``, ``"csv"``, ``"markdown"``, ``"all"``.

        Returns
        -------
        Path
            Path to the main output file.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        main_path = None

        if fmt in ("json", "all"):
            path = output_dir / "benchmark_results.json"
            # Make results JSON-serialisable
            serialisable = []
            for r in self._results:
                sr = {}
                for k, v in r.items():
                    if isinstance(v, dict):
                        sr[k] = {
                            sk: (float(sv) if isinstance(sv, (np.floating, float))
                                 else sv)
                            for sk, sv in v.items()
                        }
                    else:
                        sr[k] = v
                serialisable.append(sr)
            with open(path, "w") as f:
                json.dump(serialisable, f, indent=2, default=str)
            logger.info(f"JSON results exported to {path}")
            main_path = main_path or path

        if fmt in ("csv", "all"):
            path = output_dir / "benchmark_results.csv"
            if self._results:
                # Flatten overall metrics into columns
                fieldnames = ["model"]
                for r in self._results:
                    for k in r.get("overall", {}):
                        if k not in fieldnames:
                            fieldnames.append(k)
                with open(path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    for r in self._results:
                        row = {"model": r.get("model", "")}
                        row.update(r.get("overall", {}))
                        # Convert numpy types
                        row = {k: (float(v) if isinstance(v, np.floating) else v)
                               for k, v in row.items()}
                        writer.writerow(row)
            logger.info(f"CSV results exported to {path}")
            main_path = main_path or path

        if fmt in ("markdown", "all"):
            path = output_dir / "benchmark_results.md"
            md = self.format_results_markdown()
            with open(path, "w") as f:
                f.write("# ML-Turbulence Benchmark Results\n\n")
                f.write(md)
                f.write("\n")
            logger.info(f"Markdown results exported to {path}")
            main_path = main_path or path

        return main_path
