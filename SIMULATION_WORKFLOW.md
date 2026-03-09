# Simulation Workflow

This project supports **five practical paths** — from quick sanity checks to full
CFD simulations and advanced ML/UQ analysis. All workflows are designed to be
reproducible and automated.

**Author:** Yuvraj Singh · **Updated:** March 2026

---

## 1) Fast Verification (No SU2 Required)

Run a quick environment and data sanity check:

```bash
python run_full_benchmark.py --quick
```

This verifies Python dependencies, data file integrity, and basic imports.

---

## 2) Automated Project Health Check

Run the integrated benchmark validation script (106 checks across all cases):

```bash
python run_full_benchmark.py
```

Run the full test suite (1563+ tests across 65 files):

```bash
python -m pytest tests/ -v

# Run specific test suites
python -m pytest tests/test_benchmark.py -v
python -m pytest tests/test_ml_physics.py -v
python -m pytest tests/test_constrained_rans_recalibration.py -v
python -m pytest tests/test_pod_transformer_closure.py -v
```

Run the reproducible V&V report harness (JSON + CSV + Markdown output):

```bash
python scripts/benchmark_harness.py --report --format json csv md
```

---

## 3) Launch CFD Simulations (Requires SU2)

### Core TMR Cases

```bash
# Flat Plate — MRR Level 0 code verification
python simulations/run_flat_plate.py --model SA --grid 137x097

# NACA 0012 — grid convergence + force validation
python simulations/run_naca0012.py --alpha 0 10 15 --model SA --grid xfine

# Wall Hump — SA & SST comparison
python simulations/run_wall_hump.py --model SA SST --grid medium fine

# Bump-in-Channel — TMR code-to-code verification
python simulations/run_bump_channel.py --model SA --grid 177x081
```

### Extended Validation Cases

```bash
# SWBLI — Mach 5 shock-induced separation
python simulations/run_swbli.py --model SA SST --grid L1

# Backward-Facing Step — multi-dataset validation
python simulations/run_bfs_validation.py

# Periodic Hill — DNS reference comparison
python simulations/run_periodic_hill.py --model SA SST

# BeVERLI Hill — 3D smooth body separation
python simulations/run_beverli_hill.py --model SA SST

# Boeing Gaussian Bump — smooth-body APG
python simulations/run_gaussian_bump.py --model SA SST SA-RC

# Axisymmetric Jet — Witze reference
python simulations/run_axisymmetric_jet.py --model SA SST k-epsilon
```

### Advanced / 3D Cases

```bash
# 3D Bump-in-Channel — WMLES reference
python simulations/run_bump_3d_channel.py --model SA-QCR

# NASA CRM — DPW-5 wing-body junction
python simulations/run_nasa_crm.py --model SA --grid medium

# Transition — ERCOFTAC T3 cases
python simulations/run_transition_t3.py --case T3A T3B

# Heated Jet — thermal mixing
python simulations/run_heated_jet.py --model SA SST

# ZBOT — micro-gravity VOF
python simulations/run_zbot_vof.py
```

---

## 4) Validate & Post-Process Results

### Case-Specific Validation

```bash
python scripts/validation/validate_results.py --case wall_hump --model SA --grid medium \
    --results-dir runs/wall_hump/hump_SA_medium
```

### Grid Convergence & Uncertainty Quantification

```bash
# Multi-case GCI (Celik et al. 2008)
python scripts/analysis/run_gci_all_cases.py

# Input-parameter sensitivity (±10% OAT)
python scripts/analysis/run_input_uq_study.py

# Wall hump GCI study
python scripts/analysis/compute_wall_hump_gci.py
```

---

## 5) Plot & Summarize

```bash
# All validation plots — consolidated CLI
python scripts/postprocessing/plot_all.py all

# Individual domains
python scripts/postprocessing/plot_all.py tmr          # NACA 0012 force & Cp
python scripts/postprocessing/plot_all.py wall_hump    # Hump Cp/Cf vs Greenblatt
python scripts/postprocessing/plot_all.py swbli        # SWBLI pressure profiles
python scripts/postprocessing/plot_all.py beverli      # BeVERLI Hill validation
python scripts/postprocessing/plot_all.py velocity     # Wall-case velocity profiles
python scripts/postprocessing/plot_all.py pareto       # Physics Pareto fronts

# Full simulation dashboard (NACA 0012 SU2 results)
python scripts/postprocessing/plot_simulation_results.py
```

> [!NOTE]
> **Known RANS Limitations (document before presenting):**
> - **Wall Hump:** RANS-SA/SST reattach 20–30 % too far downstream (Greenblatt exp: x/c≈1.10; RANS: x/c≈1.25–1.30). Plots will show the offset clearly.
> - **BFS:** Expected x/H = 6.26 ± 0.10 (Driver & Seegmiller 1985). Run `compute_bfs_reattachment()` from `scripts/validation/validate_results.py` on your Cf CSV to get the comparison automatically.
> - **NACA 0012:** Fully-turbulent assumption used. Compare only against Ladson fixed-transition data; Abbott/von Doenhoff is untripped and will show a much lower CD.


---

## 6) ML-Augmented RANS Pipeline

### Physics-Informed ML Closures

```bash
# Curated ML-turbulence benchmark (McConkey alignment)
python scripts/run_curated_benchmark.py --fast

# TBNN tensor-basis closure
python -c "from scripts.ml_augmentation.tbnn_closure import TBNNModel; ..."

# FIML field inversion
python -c "from scripts.ml_augmentation.fiml_pipeline import FIMLPipeline; ..."

# PINN boundary layer correction
python -c "from scripts.ml_augmentation.pinn_boundary_layer import PINNBLCorrection; ..."
```

### DRL Active Flow Control

```bash
# PPO wall-hump separation suppression
python scripts/ml_augmentation/run_drl_wall_hump.py --episodes 200

# DRL benchmark (PPO vs constant blowing vs periodic forcing)
python -c "from scripts.ml_augmentation.design_control_uq_workflows import DRLFlowControlBenchmark; ..."
```

### Constrained RANS Recalibration

```bash
# Physics-penalty SST coefficient optimization (Bin 2024)
python -c "
from scripts.ml_augmentation.constrained_rans_recalibration import ConstrainedRecalibrator
recalibrator = ConstrainedRecalibrator(
    penalty_weights={'log_layer': 10.0, 'realizability': 5.0, 'decay': 2.0}
)
result = recalibrator.optimize(target_cases=['wall_hump', 'periodic_hill'])
print(result.report())
"
```

### POD + Transformer ROM Closure

```bash
# Easy-attention Transformer closure for POD energy recovery
python -c "
from scripts.ml_augmentation.pod_transformer_closure import PODTransformerClosure, ClosureConfig
config = ClosureConfig(n_modes_retained=10, use_easy_attention=True)
closure = PODTransformerClosure(config)
# closure.train(rom, snapshots, dns_snapshots)
"
```

### ML Physics Constraint Benchmark

```bash
# Galilean invariance & realizability comparison (Vanilla MLP vs TBNN)
python scripts/ml_augmentation/physics_informed_benchmark.py
```

### Foundation Model Alignment Study

```bash
# Foundation model vs custom domain model comparison
python -c "
from scripts.ml_augmentation.foundation_model_alignment import run_alignment_study
run_alignment_study()
"
```

### Hypersonic Extrapolation Benchmark

```bash
# OOD evaluation: low-Mach-trained ML on M=5 SWBLI + heated jet
python -c "
from scripts.ml_augmentation.hypersonic_extrapolation import run_benchmark_suite
results = run_benchmark_suite()
print(results['markdown_report'])
"
```

---

## 7) Advanced UQ Pipeline

```bash
# Bayesian PCE + MCMC inversion
python -c "from scripts.ml_augmentation.bayesian_pce_uq import BayesianPCEFramework; ..."

# Calibrated stochastic closures (BNN + ensemble + diffusion)
python -c "from scripts.ml_augmentation.calibrated_stochastic_closures import ...; ..."

# Error budget (RSS combination: GCI + model + ML + input)
python -c "from scripts.analysis.uq_summary_report import ...; ..."
```

---

## 8) Cross-Case Generalization Studies

```bash
# LOO extrapolation (6 flows × 4 architectures + UQ)
python -c "from scripts.ml_augmentation.loo_extrapolation_study import ...; ..."

# Cross-case generalization (Srivastava 2024 protocol)
python -c "from scripts.ml_augmentation.cross_case_generalization import ...; ..."
```

---

## 9) Operator Learning & Temporal Case Studies

```bash
# DeepONet vs FNO comparison for SWBLI/transonic
python -c "from scripts.ml_augmentation.operator_temporal_case_studies import ...; ..."

# ConvLSTM temporal surrogate for unsteady BFS
python -c "from scripts.ml_augmentation.operator_temporal_case_studies import ...; ..."
```

---

## 10) GNN Mesh Adaptation

```bash
# Anisotropic Hessian metric prediction + adaptation pipeline
python -c "from scripts.ml_augmentation.gnn_mesh_adaptation import ...; ..."
```

---

## Notes

- `tests/` contains **1563+ fast synthetic/unit tests** across 65 files.
- Large solver jobs should be launched through batch scripts in `scripts/solvers/`.
- Generated reports are written to `output/` and case-specific folders under `results/`.
- The ML pipeline (74 modules in `scripts/ml_augmentation/`) runs independently of SU2.
- NVIDIA PhysicsNeMo DoMINO integration is available as a mock interface
  (`physicsnemo_domino_integration.py`) for future GPU hardware availability.
- All ML modules include comprehensive docstrings with usage examples and references.
