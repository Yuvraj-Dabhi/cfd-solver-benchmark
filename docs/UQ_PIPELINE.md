# Uncertainty Quantification Pipeline

Formal UQ framework combining numerical, model-form, input, and ML epistemic uncertainties following ASME V&V 20 and Celik et al. (2008) standards.

---

## Multi-Case GCI (10 Cases)

Grid Convergence Index computed via Celik et al. (2008) for all eligible cases with 3+ mesh levels. Typical GCI_fine < 5% on fine grids.

```bash
# Run multi-case GCI study
python scripts/analysis/run_gci_all_cases.py

# Wall hump specific GCI
python scripts/analysis/compute_wall_hump_gci.py
```

---

## Input-Parameter Sensitivity

±10% one-at-a-time (OAT) perturbation of turbulence intensity (TI), pressure, velocity, and turbulence viscosity ratio (TVR) across wall hump, BFS, NACA 0012, and axisymmetric jet.

```bash
python scripts/analysis/run_input_uq_study.py
```

---

## Error Budget (RSS Combination)

| Source | Typical Range | Notes |
|--------|---------------|-------|
| Model uncertainty | 10–45% | Dominant unquantified ignorance (turbulence model form) |
| ML Epistemic Limits | 2–6% | Predictive calibration bounds via BNN/DDIM |
| Input uncertainty | 1–5% | TI most critical parameter |
| Numerical uncertainty | <5% | GCI on fine grids |

Uncertainties are combined via root-sum-square (RSS) to produce a total uncertainty budget per case and metric.

---

## Advanced UQ Methods

| Method | Module | Description |
|--------|--------|-------------|
| Bayesian PCE + MCMC | `bayesian_pce_uq.py` | Polynomial chaos expansion surrogate + MCMC Bayesian inversion + eigenspace perturbation |
| Calibrated Stochastic Closures | `calibrated_stochastic_closures.py` | Coverage-calibrated BNN/Ensemble/Diffusion + space-dependent aggregation |
| Conformal Prediction | `conformal_prediction.py` | Distribution-free UQ with finite-sample coverage guarantees |
| Deep Ensemble | `deep_ensemble.py` | Epistemic uncertainty from ensemble disagreement (N=5) |
| Bayesian DNN | `bayesian_dnn_closure.py` | MC-Dropout variational inference for structural uncertainty |

All UQ modules are in `scripts/ml_augmentation/` and can be run independently of SU2.

---

## Verification & Validation Summary

| Metric | Result |
|--------|--------|
| Flat plate U⁺ vs y⁺ | **<1% error** in viscous sublayer and log layer |
| GCI (fine grid, α=0°) | 3.65% (< 5% Roache criterion) |
| CL accuracy (all α) | < 2% error |
| Wall hump velocity profiles | Match Greenblatt PIV at 4 stations |
| SWBLI separation | SST: 8.5% error vs experiment |
| GCI across 10 cases | 15 quantities verified |
| Pipeline integration tests | **1616+/1616+ passed** |

---

## Reference Standards

- **ASME V&V 20-2009** — Standard for Verification and Validation in Computational Fluid Dynamics and Heat Transfer
- **Celik et al. (2008)** — "Procedure for Estimation and Reporting of Uncertainty Due to Discretization in CFD Applications," *Journal of Fluids Engineering*
- **Roache (1998)** — *Verification and Validation in Computational Science and Engineering*
