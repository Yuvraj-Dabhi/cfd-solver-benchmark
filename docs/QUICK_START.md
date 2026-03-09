# Quick Start Guide

Get up and running with the CFD Solver Benchmark in minutes.

**Prerequisites:** Python 3.10+, SU2 v8.4.0 (only for running simulations)

---

## Installation

```bash
# Clone and install dependencies
pip install -r requirements.txt
```

**Optional dependencies:**
- **PyTorch** — for TBNN, deep ensemble, GNN, neural operators
- **PyTorch Geometric** — for GNN-FIML and MeshGraphNet
- **GMSH** — for mesh generation
- **LaTeX** — for report compilation

---

## Verification (No SU2 Needed)

```bash
# Run full benchmark checks (106 checks across all cases)
python run_full_benchmark.py

# Run complete test suite (1616+ tests)
python -m pytest tests/ -v
```

---

## Reproducible V&V Harness

A single command generates all metrics and writes machine-readable + human-readable reports:

```bash
# Full report (no SU2 rerun required)
python scripts/benchmark_harness.py --report

# Specific cases only
python scripts/benchmark_harness.py --report --cases wall_hump bfs

# JSON + CSV only (CI/CD integration)
python scripts/benchmark_harness.py --report --format json csv
```

Outputs (written to `output/`):

| File | Format | Contents |
|------|--------|----------|
| `benchmark_report.md` | Markdown | Tables + figure links + V&V status |
| `benchmark_summary.json` | JSON | All metrics, GCI, uncertainty budgets |
| `benchmark_summary.csv` | CSV | One-row-per-case for batch processing |
| `per_case/*.json` | JSON | Individual case validation data |

This chains: **post-processing → GCI (Celik 2008) → ASME V&V 20 → uncertainty budget → report**.

---

## Running Simulations (Requires SU2)

```bash
# Flat Plate — MRR Level 0 verification
python simulations/run_flat_plate.py --model SA --grid 137x097

# NACA 0012 — full simulation
python simulations/run_naca0012.py --alpha 0 10 15 --model SA --grid xfine

# Wall Hump — SA & SST comparison
python simulations/run_wall_hump.py --model SA SST --grid medium fine

# SWBLI — Mach 5 shock-induced separation
python simulations/run_swbli.py --model SA SST --grid L1

# Backward-Facing Step — multi-dataset validation
python simulations/run_bfs_validation.py

# Transition — ERCOFTAC T3 cases
python simulations/run_transition_t3.py --case T3A T3B
```

See [SIMULATION_WORKFLOW.md](../SIMULATION_WORKFLOW.md) for the full step-by-step workflow (10 sections covering all 20 cases, UQ pipeline, ML augmentation, and more).

---

## Interactive Guide

For a guided walkthrough:

```bash
python start_here.py
```

This interactive script checks your environment, runs verification, and walks you through simulation configuration.

---

## Requirements

| Dependency | Version | Purpose |
|-----------|---------|---------|
| Python | 3.10+ | Core runtime |
| NumPy, SciPy, Matplotlib, Pandas | latest | Data processing & plotting |
| scikit-learn | latest | ML utilities |
| SU2 | v8.4.0 | CFD solver (optional) |
| PyTorch | ≥2.0 | ML models (optional) |
| PyTorch Geometric | latest | GNN modules (optional) |

```bash
pip install -r requirements.txt
```

---

## Next Steps

- **Benchmark cases & turbulence models:** [BENCHMARK_CASES.md](BENCHMARK_CASES.md)
- **ML/AI pipeline (74 modules):** [ML_PIPELINE.md](ML_PIPELINE.md)
- **Uncertainty quantification:** [UQ_PIPELINE.md](UQ_PIPELINE.md)
- **Simulation workflow:** [../SIMULATION_WORKFLOW.md](../SIMULATION_WORKFLOW.md)
- **Technical report:** [`docs/technical_report/main.pdf`](technical_report/main.pdf)
