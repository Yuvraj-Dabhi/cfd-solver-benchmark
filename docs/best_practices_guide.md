# Best Practices Guide

Recommended practices for CFD benchmarking, based on AIAA G-077, ASME V&V 20-2009, NASA TMR methodology, and lessons from DPW/HLPW workshops.

---

## 1. Grid Independence

> **Always perform a systematic grid convergence study before comparing to experiment.**

- Use **≥3 grid levels** with refinement ratio r ≥ 1.3 (r = 2 preferred)
- Report **GCI** with safety factor Fs = 1.25 (3 grids) or 3.0 (2 grids)
- Verify asymptotic range: GCI₃₂ / (r^p × GCI₂₁) ≈ 1.0
- Track **multiple quantities**: separation point, Cf_max, reattachment length

### Target GCI Thresholds

| Level | GCI | Interpretation |
|-------|-----|---------------|
| Grid independent | < 5% | Ready for validation |
| Acceptable | 5-10% | Acceptable with caution |
| Needs refinement | > 10% | Further refinement required |

---

## 2. Turbulence Model Selection

### Decision Tree

```
Is separation geometric (step/ramp)?
├─ Yes → SA or SST sufficient (< 5% error)
└─ No: smooth-body/pressure-gradient separation?
   ├─ 2D → SST (best general), v²-f (best for diffusers)
   ├─ 3D corner → SA-QCR or RSM (capture secondary flows)
   └─ Shock-induced → SST or SA-RC
```

### Model Strengths Summary

| Model | Best for | Avoid for |
|-------|----------|-----------|
| SA | BFS, flat plate | 3D juncture flows |
| SA-QCR | Corner/juncture | Simple 2D |
| SST | General APG, hump | Strong curvature |
| k-ε | Attached flows | Any separation |
| v²-f | Diffusers | Complex 3D |
| RSM | Anisotropy, secondary flows | Simple 2D (overkill) |
| DDES | Massive separation | Attached BL |
| WMLES | High-Re separated flows | Low-Re cases |

### First-order → Second-order Strategy

1. Start with **upwind** (scheme 0) to establish initial solution
2. Map converged field to **linearUpwind** (scheme 1)
3. Compare with **LUST** (scheme 3) for scheme sensitivity

---

## 3. Boundary Conditions

### Inlet Turbulence

- Report **both** turbulence intensity (TI) and length scale (L_t)
- For wind tunnels: TI = 0.5-5%, L_t ≈ 0.1 × tunnel height
- Sensitivity: run ±50% TI variation to assess impact

### Wall Treatment

| y⁺ range | Approach | Wall function |
|----------|----------|--------------|
| y⁺ ≈ 1 | Resolved | None (direct) |
| 5 < y⁺ < 30 | Buffer layer | **Avoid** (inaccurate) |
| 30 < y⁺ < 300 | Wall function | `nutUSpaldingWallFunction` |

### Outlet

- Place outlet **far downstream** (≥ 20H for BFS, ≥ 4c for airfoils)
- Use `zeroGradient` for velocity, `fixedValue` for pressure

---

## 4. Convergence Criteria

### Residual Targets

| Field | Target | Notes |
|-------|--------|-------|
| Pressure | < 10⁻⁵ | Most critical for separation |
| Velocity | < 10⁻⁵ | |
| Turbulence | < 10⁻⁵ | Can be harder to converge |

### Beyond Residuals

- Monitor **integral quantities** (drag, separation point) until plateau
- Check **mass balance** (should be < 10⁻⁸ of inlet flux)
- For unsteady: verify **statistical stationarity** before averaging

---

## 5. Validation Methodology

### NASA MRR (Model Readiness Review) Levels

| MRR | Description | Requirement |
|-----|-------------|-------------|
| 1 | Unit problems verified | Flat plate Cf < 1% error |
| 2 | Canonical validation | BFS x_R < 5% error |
| 3 | Multiple benchmarks | ≥3 cases validated |
| 4 | Complex application | 3D configuration |

### ASME V&V 20-2009

The validation metric compares error E against combined uncertainty U_val:

```
E = |S - D|            (comparison error)
U_val = √(U_num² + U_input² + U_exp²)  (validation uncertainty)
```

- **|E| < U_val**: Validated at this uncertainty level
- **|E| > U_val**: Model-form error exists

### Reporting

Always include:
1. Grid convergence data (GCI)
2. Numerical scheme sensitivity
3. Boundary condition sensitivity (TI, outlet location)
4. Uncertainty bands on experimental data
5. Statistical significance (ANOVA for multi-model comparison)

---

## 6. Data Management

### Directory Structure

```
results/
├── <case_name>/
│   ├── <model>/
│   │   ├── <mesh_level>/
│   │   │   ├── case_setup/          # Input files
│   │   │   ├── solution/            # Converged fields
│   │   │   ├── postProcessing/      # Extracted profiles
│   │   │   └── log.<solver>         # Solver log
│   │   └── convergence_study.json   # GCI results
│   └── comparison/                  # Cross-model plots
└── summary_report.md
```

### Reproducibility Checklist

- [ ] All input files version-controlled
- [ ] Solver version recorded (e.g., OpenFOAM v2312)
- [ ] Mesh generation scripts included
- [ ] Random seeds fixed for UQ/ML
- [ ] Hardware specs noted for timing comparisons

---

## 7. Common Pitfalls

| Pitfall | Consequence | Prevention |
|---------|-------------|-----------|
| Skipping grid study | Unreliable results | Always run ≥3 grids |
| Using k-ε for separation | Misses recirculation | Use SST or better |
| y⁺ in buffer layer | Inaccurate Cf | Target y⁺=1 or y⁺>30 |
| Ignoring inlet TI | Up to 15% Cf variation | Run sensitivity study |
| First-order only | Excessive numerical diffusion | Always use ≥ second-order |
| Not reporting uncertainty | Incomplete validation | Include GCI + experimental bounds |

---

## 8. Publication Checklist

- [ ] Grid convergence reported with GCI and Richardson extrapolation
- [ ] ≥2 numerical schemes compared
- [ ] Inlet BC sensitivity assessed
- [ ] V&V methodology cited (AIAA G-077, ASME V&V 20)
- [ ] Experimental uncertainty included in comparisons
- [ ] Statistical significance tests for multi-model ranking
- [ ] Code and data availability statement
