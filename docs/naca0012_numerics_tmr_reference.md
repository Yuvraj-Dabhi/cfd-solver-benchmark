# NACA 0012 Numerical Analysis — TMR Grid & Case Reference
## Source: NASA Turbulence Modeling Resource
**URL:** https://turbmodels.larc.nasa.gov/naca0012numerics_val.html  
**Grids:** https://turbmodels.larc.nasa.gov/naca0012numerics_grids.html  
**SA Results:** https://turbmodels.larc.nasa.gov/naca0012numerics_val_sa.html  
**SA+PV Results:** https://turbmodels.larc.nasa.gov/naca0012numerics_val_sa_withpv.html  
**MRR Level:** 4 (highest)

---

## 1. Case Definition

### 1.1 Flow Conditions
| Parameter | Value |
|-----------|-------|
| Mach number | M = 0.15 |
| Reynolds number (per chord) | Re = 6,000,000 |
| Angle of attack | alpha = 0, 10, 15 deg |
| Reference temperature | T_ref = 540 R (300 K) |
| Boundary layers | Fully turbulent |
| Turbulence model | SA (standard), SA-neg recommended |

### 1.2 Corrected NACA 0012 Airfoil Definition (Sharp TE)
The original NACA 0012 formula is used from x=0 to x=1.008930411365 (where the TE 
is naturally sharp), then scaled down by that factor. The resulting airfoil has a 
**maximum thickness of ~11.894%** relative to its chord.

**Corrected formula (use this, not the original):**
```
y = +/- 0.594689181 * [0.298222773*sqrt(x) - 0.127125232*x 
                       - 0.357907906*x^2 + 0.291984971*x^3 
                       - 0.105174606*x^4]
```

**Original NACA 0012 formula (for reference):**
```
y = +/- 0.6 * [0.2969*sqrt(x) - 0.1260*x - 0.3516*x^2 
               + 0.2843*x^3 - 0.1015*x^4]
```

**IMPORTANT:** There was a historical typo in the last coefficient on the TMR page
(0.105174696 vs correct 0.105174606). Although the difference is O(10^-8), grids 
have been regenerated with the correct formula as of 6/23/2014.

### 1.3 Fluid Properties
| Property | Value | Notes |
|----------|-------|-------|
| Pr | 0.72 | Molecular Prandtl number (constant) |
| Pr_t | 0.90 | Turbulent Prandtl number (constant) |
| gamma | 1.4 | Heat capacity ratio |
| Viscosity law | Sutherland's | See below |

**Sutherland's Law:**
```
mu(T) = mu_ref * (T/T_ref)^(3/2) * (T_ref + S_T) / (T + S_T)

mu_ref = 1.716e-5 Pa*s  (at T_ref = 273.15 K = 491.67 R)
S_T    = 110.4 K = 198.72 R

Nondimensional form (for this case):
mu/mu_inf = (T/T_inf)^(3/2) * (T_inf + S_T) / (T + S_T)
where T_inf = 540 R
```

### 1.4 Boundary Conditions
| Boundary | Condition | Details |
|----------|-----------|---------|
| **Farfield** | Inviscid characteristic | Point vortex correction **recommended** |
| **Airfoil wall** | Adiabatic no-slip | T wall determined by solution |
| **SA inflow** | nu_hat = 3*nu | Standard TMR (both FUN3D and CFL3D) |

**Point Vortex Correction (strongly recommended):**
- Reference: Thomas & Salas (1986), AIAA J. 24(7):1074-1080, DOI:10.2514/3.9394
- Although farfield is ~500c, correction is noticeable at the precision levels 
  being investigated
- Results with PV correction on 500c grid approximate infinite-domain results

---

## 2. TMR Grid Families

### 2.1 Common Properties (All Families)
| Property | Value |
|----------|-------|
| Farfield extent | ~500 chord lengths |
| Grid topology | C-grid |
| Min wall spacing | 1e-7 chord |
| Wall-normal stretch rate | ~1.02 (near wall) |
| LE spacing (all families) | 0.0000125c |
| Points on airfoil (finest) | 4,097 |
| Wake points (TE to outflow) | 1,537 |
| Finest grid | 7169 x 2049 |
| Coarsest grid | 113 x 33 |
| Nested levels | 7 (each is every-other-point of finer) |

### 2.2 Family Differences: Trailing Edge Spacing
| Family | TE Spacing | Relative | Status |
|--------|-----------|----------|--------|
| **Family I** | 0.000125c | baseline (1x) | Standard (same as validation grids) |
| **Family II** | 0.0000125c | 10x finer | **Best for grid convergence** |
| **Family III** | 0.0000375c | 3.33x finer | Intermediate |

### 2.3 Grid Levels (All Families)

| Level | Grid Size | Airfoil Points | Total Cells | h/h_finest |
|-------|-----------|----------------|-------------|------------|
| 1 (finest) | 7169 x 2049 | 4,097 | ~14.7M | 1.0 |
| 2 | 3585 x 1025 | 2,049 | ~3.67M | 2.0 |
| 3 | 1793 x 513 | 1,025 | ~920K | 4.0 |
| 4 | 897 x 257 | 513 | ~230K | 8.0 |
| 5 | 449 x 129 | 257 | ~58K | 16.0 |
| 6 | 225 x 65 | 129 | ~14.6K | 32.0 |
| 7 (coarsest) | 113 x 33 | 65 | ~3.7K | 64.0 |

### 2.4 Grid Download URLs

#### Family I (2D PLOT3D, gzipped)
```
Level 1: https://turbmodels.larc.nasa.gov/NACA0012numerics_grids/n0012familyI.1.p2dfmt.gz (242 MB)
Level 2: https://turbmodels.larc.nasa.gov/NACA0012numerics_grids/n0012familyI.2.p2dfmt.gz (58 MB)
Level 3: https://turbmodels.larc.nasa.gov/NACA0012numerics_grids/n0012familyI.3.p2dfmt.gz (13 MB)
Level 4: https://turbmodels.larc.nasa.gov/NACA0012numerics_grids/n0012familyI.4.p2dfmt.gz (<10 MB)
Level 5: https://turbmodels.larc.nasa.gov/NACA0012numerics_grids/n0012familyI.5.p2dfmt.gz (<10 MB)
Level 6: https://turbmodels.larc.nasa.gov/NACA0012numerics_grids/n0012familyI.6.p2dfmt.gz (<10 MB)
Level 7: https://turbmodels.larc.nasa.gov/NACA0012numerics_grids/n0012familyI.7.p2dfmt.gz (<10 MB)
```

#### Family II (2D PLOT3D, gzipped) — RECOMMENDED
```
Level 1: https://turbmodels.larc.nasa.gov/NACA0012numerics_grids/n0012familyII.1.p2dfmt.gz (242 MB)
Level 2: https://turbmodels.larc.nasa.gov/NACA0012numerics_grids/n0012familyII.2.p2dfmt.gz (58 MB)
Level 3: https://turbmodels.larc.nasa.gov/NACA0012numerics_grids/n0012familyII.3.p2dfmt.gz (13 MB)
Level 4: https://turbmodels.larc.nasa.gov/NACA0012numerics_grids/n0012familyII.4.p2dfmt.gz (<10 MB)
Level 5: https://turbmodels.larc.nasa.gov/NACA0012numerics_grids/n0012familyII.5.p2dfmt.gz (<10 MB)
Level 6: https://turbmodels.larc.nasa.gov/NACA0012numerics_grids/n0012familyII.6.p2dfmt.gz (<10 MB)
Level 7: https://turbmodels.larc.nasa.gov/NACA0012numerics_grids/n0012familyII.7.p2dfmt.gz (<10 MB)
```

#### Family III (2D PLOT3D, gzipped, finest only)
```
Level 1: https://turbmodels.larc.nasa.gov/NACA0012numerics_grids/n0012familyIII.1.p2dfmt.gz (242 MB)
```

#### CGNS Unstructured (3D hex) — for OpenFOAM/SU2
Family I and II also available as unstructured CGNS hexahedral grids.
URL pattern: `n0012family{I,II}.{1-7}.hex.cgns.gz`

### 2.5 Grid Format Details
**PLOT3D format:** Formatted, multi-grid, 2D (nbl=1). **Must use double precision!**
```fortran
read(2,*) nbl
read(2,*) (idim(n), jdim(n), n=1, nbl)
do n=1, nbl
  read(2,*) ((x(i,j,n), i=1,idim(n)), j=1,jdim(n)),
             ((y(i,j,n), i=1,idim(n)), j=1,jdim(n))
enddo
```

---

## 3. Grid-Converged Reference Results (SA Model)

### 3.1 With Point Vortex Correction (recommended)
**At alpha = 10 deg, M = 0.15, Re = 6M, T_ref = 540R**

Grid-converged values from Family II grids (FUN3D + CFL3D agreement):

| Coefficient | Grid-Converged Value | Notes |
|-------------|---------------------|-------|
| **CL** | 1.0912 - 1.0913 | Lift coefficient |
| **CD** | 0.01221 - 0.01222 | Total drag coefficient |
| **CDp** | 0.00601 - 0.00602 | Pressure drag |
| **CDv** | 0.006204 - 0.006205 | Viscous (friction) drag |
| **CM** | 0.00677 - 0.00679 | Pitching moment (at 0.25c) |

### 3.2 Key Findings on Grid Convergence
1. **TE spacing is critical:** Family I (coarse TE) converges to WRONG values 
   for CL and CM. Family II (fine TE) converges correctly.
2. **Family I would need >> 7169x2049** to start converging to correct values.
3. **Family III (intermediate TE)** only just starting to turn toward correct 
   values on the finest grid.
4. **Lesson:** Trailing edge resolution dominates convergence for lift and moment.

### 3.3 Code Settings Used
| Setting | CFL3D | FUN3D |
|---------|-------|-------|
| SA variant | SA (standard) | SA-neg |
| S-tilde clipping | Method (a) | Method (c) [ICCFD7-1902] |
| Turbulence advection | 1st order (default) | 2nd order |
| nu_hat_freestream | 3*nu | 3*nu |
| Pr / Pr_t | 0.72 / 0.90 | 0.72 / 0.90 |
| Viscosity | Sutherland's law | Sutherland's law |

### 3.4 Reference Data Files (downloadable)
```
Force/moment data:
  https://turbmodels.larc.nasa.gov/NACA0012numerics_val/fun3d_results_sa_withN.dat
  https://turbmodels.larc.nasa.gov/NACA0012numerics_val/cfl3d_results_sa_withN.dat

Surface Cp data (finest Family II grid):
  https://turbmodels.larc.nasa.gov/NACA0012numerics_val/fun3d_cp_sa.dat
  https://turbmodels.larc.nasa.gov/NACA0012numerics_val/cfl3d_cp_sa.dat

Surface Cf data (finest Family II grid):
  https://turbmodels.larc.nasa.gov/NACA0012numerics_val/fun3d_cf_sa.dat
  https://turbmodels.larc.nasa.gov/NACA0012numerics_val/cfl3d_cf_sa.dat
```

---

## 4. Quantities of Interest (Validation Metrics)

For comparison at alpha = 0, 10, 15 deg:
1. **CL** — Lift coefficient
2. **CD** — Drag coefficient (total, pressure, viscous)
3. **CM** — Moment coefficient (about 0.25c)
4. **Cp(x/c)** — Surface pressure coefficient distribution
5. **Cf(x/c)** — Surface skin friction coefficient distribution

---

## 5. Key References

1. **Rumsey (2016)** — Primary numerics analysis paper  
   "The Influence of Trailing Edge Grid Resolution on the NACA 0012 Airfoil 
   Including Near-Field and Far-Field Effects"  
   *AIAA Journal* 54(9), pp. 2563-2588  
   DOI: 10.2514/1.J054555

2. **Rumsey (2015)** — Conference version  
   AIAA Paper 2015-1746  
   DOI: 10.2514/6.2015-1746

3. **Thomas & Salas (1986)** — Point vortex correction  
   *AIAA Journal* 24(7), pp. 1074-1080  
   DOI: 10.2514/3.9394

4. **DPW-5 and DPW-6** — This case used as verification test case  
   https://aiaa-dpw.larc.nasa.gov/Workshop5/workshop5.html  
   https://aiaa-dpw.larc.nasa.gov/

---

## 6. Implications for Project

### For Grid Convergence Study
- Use **Family II grids** (fine TE spacing = 0.0000125c)
- At minimum use levels 7, 6, 5, 4 (113x33 through 897x257)
- For publication quality, include level 3 (1793x513) 
- Richardson extrapolation should yield CL ~ 1.091 ± 0.001

### For Code Verification
- Compare against grid-converged CL = 1.0912, CD = 0.01221
- Plot CL, CD, CM vs 1/N (N = total grid points) for convergence rate
- Expect 2nd-order convergence on smoothly-nested grids

### For Scheme Sensitivity
- Compare 1st-order vs 2nd-order turbulence advection (CFL3D vs FUN3D difference)
- Compare S-tilde clipping methods (a) vs (c)

### Grid Download Priority
For this project, recommended download order:
1. Family II, Level 5 (449x129) — quick testing (~58K cells)
2. Family II, Level 4 (897x257) — main runs (~230K cells)  
3. Family II, Level 3 (1793x513) — grid convergence (~920K cells)
4. Family II, Level 6 (225x65) — coarsest for GCI study (~14.6K cells)
