# Benchmark Cases & Turbulence Models

20 cases across 6 separation categories, spanning 3 complexity tiers.

---

## Tier 1 — Canonical (Code Verification)

| Case | Mach | Re | Separation Type |
|------|------|-----|-----------------|
| Flat Plate | 0.2 | 5×10⁶/m | Attached BL (MRR Level 0) |
| Bump-in-Channel | 0.2 | 3×10⁶/m | Mild pressure gradient |
| NACA 0012 | 0.15 | 6×10⁶ | Trailing-edge |
| Wall Hump | 0.1 | 936,000 | Pressure-induced |
| Periodic Hill | — | 10,595 | Curvature-driven (DNS ref) |

## Tier 2 — Intermediate (Validation)

| Case | Mach | Re | Separation Type |
|------|------|-----|-----------------|
| Backward-Facing Step | — | 36,000 | Geometry-fixed (2 datasets) |
| Boeing Gaussian Bump | 0.2 | 2×10⁶ | Smooth-body APG |
| SWBLI (Mach 5) | 5.0 | 3.7×10⁷/m | Shock-induced |
| SWBLI (Mach 2.85) | 2.85 | 7.5×10⁶/m | Axisymmetric shock |
| Bachalo-Johnson | 0.875 | 1.37×10⁶ | Transonic shock |
| ERCOFTAC T3A/T3B | — | varies | Bypass transition *— scaffolded, awaiting runs* |
| BeVERLI Hill | — | 250,000 | 3D smooth body |

## Tier 3 — Complex / 3D

| Case | Mach | Re | Separation Type |
|------|------|-----|-----------------|
| Axisymmetric Jet | 0.5 | 570,000 | Free shear (Witze) |
| 3D Bump-in-Channel | 0.2 | 2×10⁶ | 3D corner separation |
| NASA CRM | 0.85 | 5×10⁶ | Wing-body junction (DPW-5) |
| Heated Jet (AJM163H) | 0.5 | 575,600 | Thermal jet mixing *— scaffolded, awaiting runs* |
| ZBOT Micro-g | — | — | VOF multiphase *— scaffolded, awaiting runs* |

---

## Turbulence Models

15 models registered across RANS, transition, and scale-resolving categories:

| Model | Type | Key Cases |
|-------|------|-----------|
| SA | RANS | All 5 core cases |
| SA-QCR | RANS | Gaussian bump, 3D bump |
| SA-RC | RANS | 3D bump (rotation correction) |
| SST | RANS | Wall hump, SWBLI, BFS |
| k-ε | RANS | BFS, jet (round-jet anomaly) |
| v2-f | RANS | Wall hump (F₂ sensitivity) |
| EASM | RANS | Complex 3D flows |
| RSM | RANS | TMR cross-reference (CFL3D data) |
| γ-Reθ SST | Transition | T3A, T3B, NACA 0012 (α=0°) |
| DDES / SBES / SAS | Hybrid | Periodic hill, BeVERLI |
| WMLES-EQWM / NEQWM | LES | 3D bump reference |

---

## ML-Turbulence Benchmark Suite (McConkey Alignment)

The project serves as a formal ML-turbulence benchmark aligned with McConkey et al. (2021). Five flow cases are mapped to standardised geometries for direct comparison against published DNS/LES references.

### Matched Cases

| Project Case | Curated Geometry | Re | Reference |
|---|---|---|---|
| Periodic Hill | `periodic_hills_alpha1.0` | 10,595 | Breuer DNS |
| NASA Wall Hump | `parametric_bump_h42` | 936,000 | Greenblatt PIV |
| Backward-Facing Step | `backward_facing_step` | 36,000 | Driver & Seegmiller |
| Boeing Gaussian Bump | `gaussian_bump_3d` | 2×10⁶ | NASA WMLES |
| NASA Juncture Flow | `wing_body_junction` | 2.4×10⁶ | NASA TMR |

### Benchmark Tasks & Baselines

8 benchmark tasks are defined across the 5 cases, covering Reynolds-stress fields, Cp/Cf distributions, and separation metrics. Baseline errors are provided for 9 model classes:

| Model Class | Type | Example Task Error (PH_RS_field) |
|---|---|---|
| SA | RANS | 0.350 |
| SST | RANS | 0.250 |
| Random Forest | Simple ML | 0.180 |
| Vanilla MLP | Simple ML | 0.140 |
| TBNN | Advanced ML | 0.085 |
| FIML | Advanced ML | 0.095 |
| Diffusion Surrogate | Advanced ML | 0.070 |
| DeepONet | Advanced ML | 0.090 |

### Metrics Contract API

External models can be plug-in evaluated using `BenchmarkMetricsContract`:

```python
from scripts.ml_augmentation.curated_benchmark_evaluator import BenchmarkMetricsContract

contract = BenchmarkMetricsContract(target_names=["Ux", "Uy", "uu_dns", "uv_dns", "vv_dns"])
contract.register_model("MyModel", my_predict_fn)
results = contract.evaluate_all(test_features, test_targets)
contract.export_results(Path("output"), fmt="all")  # JSON + CSV + Markdown
```

Metrics computed: RMSE, MAE, realizability violation (Lumley triangle), Cf-based separation/reattachment error, bubble length error.

```bash
# Run the full benchmark
python scripts/run_curated_benchmark.py --fast

# Specific cases only
python scripts/run_curated_benchmark.py --cases periodic_hill nasa_hump
```

---

## Cross-Solver Comparison

SU2 results are compared against published CFL3D/FUN3D values from [NASA TMR](https://turbmodels.larc.nasa.gov/). An OpenFOAM cross-solver comparison on the wall hump is also included (see §3.9 of the technical report).

```python
from scripts.comparison.tmr_reference_runner import TMRReferenceComparison
comp = TMRReferenceComparison(tolerance_pct=2.0)
report = comp.compare("naca_0012_stall", {"CL": 1.089, "CD": 0.01235})
print(report.summary())
```

See [`docs/SOLVER_TRICKS.md`](SOLVER_TRICKS.md) for numerical method rationale, CFL strategy, SA variant selection, and grid generation best practices.
