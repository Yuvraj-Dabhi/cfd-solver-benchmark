# CFD Solver Benchmark for Flow Separation Prediction

[![CI](https://github.com/Yuvraj-Dabhi/cfd-solver-benchmark/actions/workflows/ci.yml/badge.svg)](https://github.com/Yuvraj-Dabhi/cfd-solver-benchmark/actions/workflows/ci.yml)

**[Python 3.10+]** | **[SU2 v8.4.0]** | **[1616+ Tests Passed]** | **[20 Benchmark Cases]** | **[15 Turbulence Models]** | **[74 ML Modules]**

Systematic benchmarking of RANS turbulence models for separated-flow prediction,
with physics-informed ML augmentation and uncertainty quantification. Built on the
SU2 solver and Python, spanning 20 benchmark cases across 6 separation categories —
from code verification (flat plate) through complex 3D flows (NASA CRM, 3D bump).

**Author:** Yuvraj Singh · **Date:** February–March 2026

---

## Quick Start

```bash
pip install -r requirements.txt        # Install dependencies
python run_full_benchmark.py           # Run 106 verification checks (no SU2 needed)
python -m pytest tests/ -v             # Run 1616+ tests
python scripts/benchmark_harness.py --report  # Generate V&V report
```

See **[docs/QUICK_START.md](docs/QUICK_START.md)** for the full installation guide, simulation commands, and reproducible V&V harness.

---

## Key Results

| Case | Model | Key Metric | Status |
|------|-------|------------|--------|
| Flat Plate | SA | U⁺ vs y⁺: **<1% error** vs log law (κ=0.41, B=5.0); Cf(x=0.97)=0.00271 | Verified |
| Bump-in-Channel | SA | Cf at x=0.63: **0.00509** (rms=−13.01); note grid 177×81 vs NASA's 1409×641 | Verified |
| NACA 0012 (α=0°) | SA | CD=0.01235 (fully-turbulent; tripped); error: **3.2%** vs CFL3D | Validated |
| NACA 0012 (α=10°) | SA | CL error: **1.6%**; CD error: 5.4% (fully-turbulent assumption) | Validated |
| NACA 0012 (α=15°) | SA | Simulation unconverged on xfine; medium/fine grid results only | Issues |
| Wall Hump | SA | Cp/Cf vs Greenblatt PIV; RANS overpredicts reattachment x/c by ~20–30% (known bias) | Synthetic Demo |
| SWBLI (M=5) | SST | Separation onset: **8.5% error** | Synthetic Demo |
| BFS (Driver & Seegmiller) | SA/SST/k-ε | x_reat/H cross-dataset validated | Synthetic Demo |
| NACA 0012 Transition | γ-Reθ SST | Transition onset vs fully turbulent | Validated |
| ML: | `distribution_surrogate.py` | Multi-output MLP: (AoA, Re, Mach, H, dCp/dx, Reθ) → 80-pt Cp + 80-pt Cf *(Note: limited out-of-distribution validity; ref: Forrester et al. 2008)* | Validated (Real Data) |
| ML: Tier 1 MLP Surrogate | MLP | Trained on Ladson (n=22); CV R²=0.98; *needs independent test set* | Validated (Real Data) |
| ML: FIML β-Correction | FIML | 22.1% Cf RMSE reduction (wall hump, synthetic) | Synthetic Metrics |
| UQ: Error Budget | RSS | GCI + Model + ML + Input uncertainty combined | Quantified |

**Maturity key:** Quantified = end-to-end experiment with numeric results; Implemented = module coded and unit-tested but awaiting full training/validation on CFD data.

### V&V Accuracy Caveats

> [!IMPORTANT]
> **Wall-Mounted Hump (NASA TMR 2DWMH):** All RANS models systematically predict reattachment too far downstream. The Greenblatt et al. (2006) experiment measures reattachment at x/c ≈ 1.10. SA and SST predictions overshoot by **~20–30%** (x/c ≈ 1.25–1.30). This is a well-documented RANS deficiency.
>
> **NACA 0012 — Fully-Turbulent Assumption:** All SU2 runs use a fully-turbulent (tripped) boundary layer. At Re=6×10⁶, tripped data yield CD ≈ 0.012–0.020; *untripped* experiments yield CD ≈ 0.005–0.008. Comparisons are valid only against Ladson fixed-transition data.
>
> **SWBLI (M≈5):** SA predicts a larger separation zone than SST (separation onset error ~14% vs 8.5%), consistent with literature on SA over-amplifying separation in hypersonic interactions.
>
> **Backward-Facing Step:** Reference reattachment is x/H = 6.26 ±0.10 (Driver & Seegmiller 1985). Any discrepancy > 5% indicates incorrect inflow BL or mesh coarseness.
>
> **Bump-in-Channel:** Current grid (177×81) is significantly coarser than NASA's verified finest (1409×641). Full Richardson extrapolation requires 3+ grid levels.

---

## Project Structure

```
CFD Solver Benchmark/
├── config.py                       Central registry (20 cases, 15 models)
├── run_full_benchmark.py           Master orchestrator (106 checks)
├── start_here.py                   Guided project entry point
│
├── simulations/                    SU2 simulation runner scripts (21 scripts)
│   ├── run_naca0012.py             SU2 TMR runner (NACA 0012)
│   ├── run_wall_hump.py            SU2 TMR runner (wall hump)
│   ├── run_flat_plate.py           TMR flat plate (MRR Level 0)
│   └── ...                         (21 simulation scripts total)
│
├── scripts/                        230+ Python files across 13 sub-packages
│   ├── benchmark_harness.py        Single-command V&V report (JSON/CSV/MD)
│   ├── analysis/                   GCI, error metrics, sensitivity, UQ
│   ├── ml_augmentation/            Physics-informed ML pipeline (74 modules)
│   ├── postprocessing/             Surface data, plotting, validation figures
│   ├── preprocessing/              Grid conversion (PLOT3D → SU2)
│   ├── comparison/                 Multi-code/experiment comparison
│   ├── validation/                 Post-simulation validation
│   ├── models/                     14 turbulence model wrappers
│   └── ...                         solvers, orchestration, utils
│
├── experimental_data/              TMR grids, CFL3D reference, experiments
│   ├── naca0012/                   TMR reference + Ladson + Gregory (26 CSV)
│   ├── wall_hump/                  Greenblatt Cp/Cf + TMR grids
│   └── ...                         beverli_hill, gaussian_bump, etc.
│
├── tests/                          pytest suite (1616+ tests, 72 files)
├── runs/                           SU2 simulation outputs
├── results/                        Post-processed results, JSON, plots
├── output/                         Generated benchmark reports
├── plots/                          Generated validation figures
│
├── docs/                           Documentation
│   ├── QUICK_START.md              Installation & first run
│   ├── BENCHMARK_CASES.md          20 cases + 15 turbulence models
│   ├── ML_PIPELINE.md             74 ML/AI modules inventory
│   ├── UQ_PIPELINE.md             UQ methodology + error budget
│   ├── technical_report/           LaTeX report (main.pdf)
│   ├── SOLVER_TRICKS.md           Numerical method rationale
│   ├── best_practices_guide.md    Grid/solver best practices
│   └── troubleshooting_guide.md   Common issues & fixes
│
├── SIMULATION_WORKFLOW.md          Complete workflow guide (10 sections)
├── Dockerfile                      Containerized environment
├── requirements.txt                Python dependencies
└── archive/                        Early planning docs (historical)
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [Quick Start](docs/QUICK_START.md) | Installation, running simulations, V&V harness |
| [Benchmark Cases](docs/BENCHMARK_CASES.md) | 20 cases, 15 turbulence models, McConkey alignment |
| [ML/AI Pipeline](docs/ML_PIPELINE.md) | 74 ML modules: closures, surrogates, GNN, DRL, UQ |
| [UQ Pipeline](docs/UQ_PIPELINE.md) | GCI, input sensitivity, error budget |
| [Simulation Workflow](SIMULATION_WORKFLOW.md) | Step-by-step guide (10 workflow sections) |
| [Solver Tricks](docs/SOLVER_TRICKS.md) | CFL strategy, SA variants, grid generation |
| [Troubleshooting](docs/troubleshooting_guide.md) | Common issues and fixes |

---

## Test Coverage

**1616+ tests** across 72 test files covering all modules, cases, and ML pipelines.
*(Note: These tests verify Python code logic, data parsing, and ML tensor dimensions; they do not verify CFD physics outcomes).*

```bash
python -m pytest tests/ -v                              # Full suite
python -m pytest tests/test_benchmark.py -v             # Core benchmark
python -m pytest tests/test_ml_physics.py -v            # ML physics modules
python -m pytest tests/test_gnn_meshgraphnet.py -v      # GNN modules
```

---

## Reference Data Sources

| Source | URL |
|--------|-----|
| NASA Turbulence Modeling Resource | https://turbmodels.larc.nasa.gov/ |
| TMR (new site, Jan 2026) | https://tmbwg.github.io/turbmodels/ |
| SU2 V&V Repository | https://github.com/su2code/VandV |
| McConkey et al. 2021 | Scientific Data (895k-point DNS/LES dataset for TBNN) |
| DPW-5 (NASA CRM) | https://aiaa-dpw.larc.nasa.gov/ |

---

## Requirements

- **Python** 3.10+ with NumPy, SciPy, Matplotlib, Pandas, scikit-learn
- **SU2** v8.4.0 (for simulations; not needed for testing/validation)
- **PyTorch** (optional, for TBNN, deep ensemble, GNN, neural operators)
- **PyTorch Geometric** (optional, for GNN-FIML and MeshGraphNet)

```bash
pip install -r requirements.txt
```

---

## License

MIT License
