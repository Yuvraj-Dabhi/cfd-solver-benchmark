# Solver Tricks & Numerical Method Documentation

This document records the numerical method choices, solver tricks, and their rationale
for the CFD Solver Benchmark. These decisions are based on NASA TMR guidelines,
ASME V&V 20-2009, and practical experience with SU2 v8.4.0.

---

## 1. Second-Order Discretization Everywhere

**Rule:** Both flow and turbulence variables use MUSCL reconstruction (second-order).

| Setting | Value | Rationale |
|---------|-------|-----------|
| `MUSCL_FLOW` | `YES` | Standard for all RANS verification |
| `MUSCL_TURB` | `YES` | NASA TMR: first-order is "inadequate for verification" |
| `SLOPE_LIMITER_FLOW` | `VENKATAKRISHNAN` | Preserves 2nd-order in smooth regions |
| `SLOPE_LIMITER_TURB` | `VENKATAKRISHNAN` | Prevents oscillations in turbulence variables |
| `CONV_NUM_METHOD_FLOW` | `ROE` | Roe's approximate Riemann solver — robust for mixed flows |
| `CONV_NUM_METHOD_TURB` | `SCALAR_UPWIND` | Standard upwind for scalar transport |

> **Why not first-order?** First-order upwind introduces numerical diffusion of the same
> magnitude as the turbulence model's physical diffusion. This masks modeling errors and
> produces artificially smooth solutions that appear converged but are inaccurate.

---

## 2. Convergence Criteria

| Parameter | Default | V&V Strict | Notes |
|-----------|---------|------------|-------|
| `CONV_RESIDUAL_MINVAL` | -10 | -12 | Log₁₀ of RMS residual drop |
| `CONV_STARTITER` | 10 | 10 | Ignore initial transients |
| Max iterations | 5000 | 30000+ | For near-stall angles |

**Iterative convergence checks:**
- Monitor CL/CD over last 500 iterations — should stabilize to <0.01% variation
- For airfoil cases, CL stabilization is more important than residual level
- For wall hump, Cf at reattachment is the most residual-sensitive metric

---

## 3. CFL Strategy

```
CFL_NUMBER= 10.0
CFL_ADAPT= YES
CFL_ADAPT_PARAM= (0.5, 1.5, 1.0, 100.0)
```

- **Ramp-up:** Start at CFL=1.0 (parameter 3), grow by factor 1.5, reduce by 0.5 on divergence
- **Cap:** Maximum CFL=100.0 prevents runaway for stiff problems
- **SWBLI tip:** For Mach >3 flows, start CFL=1.0 and cap at 10.0; use `CFL_ADAPT_PARAM= (0.3, 1.2, 1.0, 10.0)`

---

## 4. SA Variants: When to Use What

| Variant | SU2 Option | Use Case |
|---------|------------|----------|
| SA (baseline) | `KIND_TURB_MODEL= SA` | Default for most flows |
| SA-RC | `SA_OPTIONS= RC` | Curved walls (hump, bump, diffuser) |
| SA-QCR | `SA_OPTIONS= QCR2000` | Corner/juncture flows (DPW requirement) |
| SA-neg | `SA_OPTIONS= NEGATIVE` | Better convergence for complex flows |

---

## 5. Dual-Time Stepping for Unsteady Flows

For SWBLI, transonic buffet, or other inherently unsteady problems:

```
TIME_MARCHING= DUAL_TIME_STEPPING-2ND_ORDER
TIME_STEP= 1e-5
MAX_TIME= 1.0
TIME_ITER= 1000
INNER_ITER= 50
UNST_CFL_NUMBER= 5.0
```

- Inner iterations converge the pseudo-steady problem at each physical time step
- `UNST_CFL_NUMBER` should be lower than steady CFL for stability
- For hypersonic SWBLI, use 50-100 inner iterations

---

## 6. Line-Implicit Solving (Reference)

NASA Wind-US uses line-implicit solving for stiff hypersonic problems (Ref: NPARC Alliance
Wind-US documentation). The SU2 equivalent approach:

- Use ILU preconditioner (default in SU2)
- Enable MUSCL + ROE for the flow convection
- Set under-relaxation factors ~0.7 for momentum, ~0.3 for pressure
- For very stiff problems, reduce CFL to 1-5 and increase max iterations

---

## 7. Grid Generation Best Practices

### Geometric Stretching

For wall-resolved RANS, use geometric stretching ratios:
- **y+ ≈ 1** at the first cell (mandatory for SA and SST without wall functions)
- **Growth ratio ≤ 1.2** normal to wall (1.15 preferred for V&V)
- **Δx⁺ ≈ Δz⁺ ≈ 50-100** streamwise/spanwise (for 2D RANS, only streamwise matters)

### First Cell Height Estimation

```python
# Flat-plate correlation for first cell height
Re_x = U_inf * x / nu
Cf = 0.026 / Re_x**(1/7)
tau_w = 0.5 * rho * U_inf**2 * Cf
u_tau = sqrt(tau_w / rho)
y_1 = y_plus_target * nu / u_tau
```

### Geometric Conservation Law (GCL)

When using moving/deforming meshes (not applicable to current steady cases), the GCL must
be satisfied to maintain conservation. SU2 handles this automatically with the `GRID_MOVEMENT`
options, but for time-accurate solutions, verify that `VOLUME_FILENAME` shows no spurious
mass/momentum source terms.

---

## 8. Cross-Solver Verification

We compare SU2 results against published CFL3D and FUN3D values from NASA TMR:

| Case | Metric | CFL3D | FUN3D | Tolerance |
|------|--------|-------|-------|-----------|
| Flat Plate | Cf(x=5) | 0.002967 | 0.002969 | <2% |
| NACA 0012 α=10° | CL | 1.0909 | 1.0912 | <2% |
| NACA 0012 α=10° | CD | 0.01231 | 0.01222 | <2% |
| Wall Hump (SA) | x_reat | 1.28 | — | <5% |
| BFS (SA) | x_reat | 6.28 | — | <5% |

SU2 results deviating >2% from CFL3D/FUN3D warrant investigation for:
1. Grid topology differences (C-grid vs O-grid)
2. Far-field boundary distance
3. Turbulence model implementation differences
4. Limiter effects

Use `scripts/comparison/tmr_reference_runner.py` for automated comparison.
