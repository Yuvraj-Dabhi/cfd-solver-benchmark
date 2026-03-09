# Spalart-Allmaras Turbulence Model — Complete TMR Reference
## Source: NASA Turbulence Modeling Resource
**URL:** https://turbmodels.larc.nasa.gov/spalart.html#sa  
**Last Updated on TMR:** 12/05/2024  
**Primary Reference:** Spalart, P. R. and Allmaras, S. R., "A One-Equation Turbulence Model for
Aerodynamic Flows," *Recherche Aerospatiale*, No. 1, 1994, pp. 5-21.

---

## 1. Overview

The Spalart-Allmaras (SA) model is a **one-equation linear eddy viscosity model** using the 
Boussinesq assumption for the constitutive relation. The `-2/3 ρk δ_ij` term is generally 
ignored because k is not readily available. For compressible flow with heat transfer, this 
model is implemented with:
- **Pr = 0.72** (molecular Prandtl number)
- **Pr_t = 0.90** (turbulent Prandtl number)
- **Sutherland's law** for dynamic viscosity
- **Perfect gas** assumed

---

## 2. Model Variants

### 2.1 General Model Forms
| Variant | Description | Recommended? |
|---------|-------------|--------------|
| **(SA)** | Standard — original published version without trip term | Yes (standard) |
| **(SA-neg)** | Handles negative ν̂ values robustly | **Recommended** (identical results to SA, better numerics) |
| **(SA-noft2)** | Drops f_t2 term (commonly used) | Yes, for most practical problems |
| **(SA-Ia)** | Includes trip term | Rarely used |

### 2.2 Corrections (can be combined with General Model)
| Correction | Purpose | Key Reference |
|-----------|---------|---------------|
| **(SA-RC)** | Rotation/Curvature correction | Shur et al. (2000), AIAA J. 38(5) |
| **(SA-R)** | Rotation correction (simpler than RC) | Dacles-Mariani et al. (1995) |
| **(SA-KL)** | Kato-Launder correction | Kato & Launder (1993) |
| **(SA-LRe)** | Low Reynolds number correction | Spalart & Garbaruk (2020), AIAA J. 58(5) |
| **(SA-comp)** | Mixing layer compressibility correction | Spalart (2000), AIAA 2000-2306 |
| **(SA-rough)** | Wall roughness correction | Aupoix & Spalart (2003) |
| **(SA-TC)** | Transverse curvature free-shear correction | Spalart & Garbaruk (2021) |
| **(SA-QCR2000)** | Quadratic constitutive relation (nonlinear) | Spalart (2000) |
| **(SA-QCR2013)** | QCR v2013 (adds -2/3ρk approx.) | Mani et al. (2013) |
| **(SA-QCR2020)** | QCR v2020 (improved, NASA Juncture Flow) | Rumsey et al. (2020), AIAA J. 58(10) |
| **(SA-QCR2024)** | QCR v2024 (corner flow improvement) | Tamaki & Kawai (2024), JFM 980 |
| **(SA-Helicity)** | Velocity helicity for energy backscatter | Liu et al. (2011), **NOT Galilean invariant** |

### 2.3 Other SA Versions (NOT compatible with General Model)
| Version | Description |
|---------|-------------|
| **(SA-noft2-Catris)** | Compressible form by Catris & Aupoix (2000) |
| **(SA-noft2-Edwards)** | Improved near-wall convergence (Edwards & Chandra, 1996) |
| **(SA-fv3)** | Not recommended (unusual transition behavior) |
| **(SA-noft2-salsa)** | Strain-adaptive formulation (Rung et al., 2003) |

---

## 3. Standard SA Model Equations

### 3.1 Transport Equation (non-conservation form)
The model solves for ν̂ (SA working variable, written as "hat" on TMR for display):

```
Dν̂/Dt = cb1 * S̃ * ν̂                          — Production
       - cw1 * fw * (ν̂/d)²                    — Destruction (wall)
       + (1/σ) * [∇·((ν + ν̂)∇ν̂) + cb2*(∇ν̂)²] — Diffusion
```

### 3.2 Turbulent Eddy Viscosity
```
ν_t = ν̂ * fv1

fv1 = χ³ / (χ³ + cv1³)

χ = ν̂ / ν
```
where ν is the molecular kinematic viscosity and ρ is the density.

### 3.3 Additional Definitions
```
S̃ = S + (ν̂/(κ²d²)) * fv2        when S̃ ≥ 0 (see Note 1c below)

S = |Ω|  (magnitude of vorticity)

fv2 = 1 - χ/(1 + χ*fv1)

fw = g * [(1 + cw3⁶)/(g⁶ + cw3⁶)]^(1/6)

g = r + cw2 * (r⁶ - r)

r = min(ν̂/(S̃*κ²*d²), 10)
```

**Note:** d is the minimum distance to the nearest wall. Must be computed as true 
minimum distance (not along grid lines or nearest grid point).

### 3.4 Note 1: Preventing Numerical Issues with S̃ and r

Three methods exist for preventing S̃ from going to zero or negative:

**(a)** Limit `ν̂/(κ²d²) * fv2` term to be > 0  
**(b)** Spalart recommends limiting `S̃` to be no smaller than `0.3 * S`  
**(c)** **Recommended (especially for SA-neg):** Replace the S̃ equation:

Define:
```
S_bar = (ν̂/(κ²d²)) * fv2
```

Then:
```
S̃ = S + S_bar           when S_bar ≥ -cv2 * S
S̃ = S + S*(cv2²*S + cv3*S_bar) / ((cv3-2*cv2)*S - S_bar)   when S_bar < -cv2 * S
```

where `cv2 = 0.7` and `cv3 = 0.9`.

**Guard on r:** Whenever `S̃ = 0`, set `r = 10`.

---

## 4. Model Constants

| Constant | Value | Notes |
|----------|-------|-------|
| σ | 2/3 | Diffusion coefficient |
| κ | 0.41 | von Kármán constant |
| cb1 | 0.1355 | Production coefficient |
| cb2 | 0.622 | Diffusion coefficient |
| cw1 | cb1/κ² + (1+cb2)/σ | Wall destruction coefficient |
| cw2 | 0.3 | Wall function coefficient |
| cw3 | 2.0 | Wall function coefficient |
| cv1 | 7.1 | Eddy viscosity function |
| cv2 | 0.7 | For Note 1(c) S̃ limiter |
| cv3 | 0.9 | For Note 1(c) S̃ limiter |
| ct1 | 1.0 | Trip term (SA-Ia only) |
| ct2 | 2.0 | Trip term (SA-Ia only) |
| ct3 | 1.2 | ft2 term (0 for SA-noft2) |
| ct4 | 0.5 | ft2 term |

### Derived constant:
```
cw1 = cb1/κ² + (1+cb2)/σ
    = 0.1355/0.41² + (1+0.622)/(2/3)
    = 0.1355/0.1681 + 1.622/0.6667
    = 0.8062 + 2.4333
    = 3.2391 (approximately 3.24)
```

---

## 5. Boundary Conditions

### 5.1 Standard (without trip term)
| Location | ν̂ Value | ν_t Value |
|----------|---------|-----------|
| **Wall** | ν̂ = 0 | ν_t = 0 |
| **Farfield / Inflow** | ν̂ = 3ν to 5ν | ν_t ≈ 0.210438ν to 1.294ν |

**Reference for inflow BCs:**
- Spalart (2000), "Trends in Turbulence Treatments," AIAA 2000-2306
- Spalart & Rumsey (2007), AIAA Journal 45(10), pp. 2544-2553

### 5.2 With Trip Term (SA-Ia)
| Location | ν̂ Value | ν_t Value |
|----------|---------|-----------|
| **Farfield** | ν̂ = ν/10 | ν_t ≈ 6.04e-05 ν |

---

## 6. SA-neg Model (Recommended Variant)

Same as standard SA when ν̂ ≥ 0 (including Note 1(c) as standard for this model).

When ν̂ < 0, solve:
```
Dν̂/Dt = cb1 * (1 - ct3) * S * ν̂           — Production (with + sign)
       + cw1 * (ν̂/d)²                       — Destruction (opposite sign!)
       + (1/σ) * [∇·((ν + ν̂*fn)∇ν̂) + cb2*(∇ν̂)²]  — Diffusion

fn = (cn1 + χ³)/(cn1 - χ³)
cn1 = 16
```

**Key:** ν_t = 0 when ν̂ < 0.

The SA-neg model also provides a **conservation form** (combining with mass conservation):
```
∂(ρν̂)/∂t + ∇·(ρν̂u) = [RHS terms from SA]
```

---

## 7. SA-noft2 Variant

Same as standard SA but **ct3 = 0** (ft2 term removed entirely).

For most problems with appropriate inflow conditions (ν̂ ≥ 3ν), results are 
essentially identical to standard SA.

**Important for DES/DDES:** The ft2 term in standard SA may cause undesirable delay 
in transition in the RANS region. SA-noft2 recommended for DES/DDES base model.

---

## 8. Rotation/Curvature Correction (SA-RC)

The production term is multiplied by rotation function f_r1:
```
P_new = cb1 * (f_r1) * S̃ * ν̂
```

where:
```
f_r1 = (1 + c_r1) * (2r*/(1+r*)) * [1 - c_r3*tan⁻¹(c_r2*r̃*)] - c_r1

r* = S/Ω    (ratio of strain rate to rotation rate)

r̃* = D²S_ij / (D Lag)   (Lagrangian derivative term)
```

Constants: `c_r1 = 1.0`, `c_r2 = 12.0`, `c_r3 = 1.0`

**Note:** SA-RC production term can be negative.

---

## 9. Quadratic Constitutive Relation (QCR)

### 9.1 QCR2000
Instead of linear Boussinesq, use:
```
τ_ij = τ_ij^B + c_cr1 * (O_ik * τ_jk^B + O_jk * τ_ik^B)
```

where O_ik is an antisymmetric normalized rotation tensor:
```
O_ik = 2*W_ik / √(∂u_m/∂x_n * ∂u_m/∂x_n)

W_ik = 0.5*(∂u_i/∂x_k - ∂u_k/∂x_i)
```

**Constant:** `c_cr1 = 0.3`

### 9.2 QCR2020
```
τ_ij = τ_ij^B + c_cr1*(O_ik*τ_jk^B + O_jk*τ_ik^B) 
     + c_cr4*S_ij_hat*fw² + c_cr5*(k_approx)*δ_ij
```

Constants: `c_cr1 = 0.3`, `c_cr4 = -0.02`, `c_cr5 = -2/3`

Uses fw function from SA model. Wall distance d is required.

### 9.3 Naming Convention
- SA + QCR2000 → SA-QCR2000
- SA-noft2 + QCR2000 → SA-noft2-QCR2000
- **SA-RC-QCR2000** is a very common combination

---

## 10. Compressibility Correction (SA-comp)

Additional term on RHS of SA equation for compressible mixing layers:
```
Additional = c5 * (ν̂/a)² * (∂²u_i/∂x_j∂x_j)
```

where a is local speed of sound and `c5` is the model constant.

Based on Shur et al. (1995) but in modified form.

---

## 11. Wall Roughness Correction (SA-rough)

Boeing method — augment the distance function:
```
d_new = d + 0.03 * k_s
```

where k_s is the equivalent sand grain roughness height.

At the wall (d=0), the BC for ν̂ is replaced (see TMR for specific value).

---

## 12. Low Reynolds Number Correction (SA-LRe)

Changes cw2 from constant to a function:
```
cw2 = cw2(Re_δ)
```

With boundary-layer-thickness-based Reynolds number.

**Reference:** Spalart & Garbaruk (2020), AIAA J. 58(5), pp. 1903-1905

---

## 13. Important Implementation Notes

### 13.1 Wall Distance Computation — CRITICAL

**This is one of the most common sources of implementation error in the SA model.**

The variable `d` in the SA model is the **true minimum Euclidean distance** from 
each field point (grid point or cell center) to the nearest solid wall surface. 
Getting this wrong introduces **grid-dependent errors** that pollute all results.

#### CORRECT: True Minimum Distance (Green)
```
d(P) = min over all wall surfaces { perpendicular distance from P to wall segment }
```
The closest point on the wall may lie **between wall grid points**, not at a grid 
point. The algorithm must compute the distance from the field point to each wall 
**segment** (or face in 3D), not just to wall **nodes**.

#### INCORRECT Method 1: Distance Along Grid Lines (Pink)
```
WRONG: Follow the j-direction grid line from the field point down to the wall
```
This only works if the grid is perfectly wall-normal at every point. For any 
non-orthogonal grid, this gives the wrong answer. This method is grid-dependent 
AND yields different results on different grid topologies.

#### INCORRECT Method 2: Distance to Nearest Wall Grid Point (Red)
```
WRONG: d = min over wall grid points { ||P - P_wall|| }
```
The nearest point on the continuous wall surface may fall between grid points. 
This is **inaccurate** (though often closer to correct than Method 1). As the 
grid is refined, this method converges to the correct answer, but on coarse 
grids the error can be significant, particularly near regions of high wall 
curvature.

#### INCORRECT Method 3: Distance to Nearest Wall Cell Center
```
WRONG: d = min over wall cell centers { ||P - P_cell_center|| }
```
Same issue as Method 2, but even less accurate because cell centers may be 
further from the actual wall surface.

#### Visual Concept (from TMR, curved wall case):
```
                      * field point
                     /|\ 
                    / | \
          RED (wrong)/|  \ RED (wrong)
                  /   |   \
                 /    | GREEN (correct: true min dist to wall surface)
                /     |     \
           ----*------*------*---- wall surface (continuous)
               ^  closest ^  ^
               |  point   |  |
          wall grid    between  wall grid
          point       grid pts  point
```
The GREEN distance goes to the nearest point on the wall **surface** (which 
lies between two grid points). The RED distances go to grid points on the wall, 
which are farther away. In this case both red distances are longer than green.

#### Visual Concept (from TMR, non-orthogonal grid case):
```
     grid lines (non-orthogonal)
      \   |   /
       \  |  /
        \ | /
    PINK \|/ (following grid line: WRONG)
          * field point
          |
          | GREEN (true perpendicular to wall: CORRECT)
          |
    ------+------------ wall surface
```
Following the grid line (pink) to the wall gives a DIFFERENT (longer) distance 
than the true perpendicular distance (green). This is why "distance along grid 
lines" is fundamentally incorrect for non-orthogonal grids.

#### Special Case: Sharp Convex Corners (e.g., Airfoil Trailing Edge)
```
              * field point (in wake)
             /  
            /
           / d = distance to corner point
          /
    -----*  <-- sharp TE corner
          \
           \  airfoil surface
```
When the nearest wall point is a sharp convex corner (like an airfoil trailing 
edge), the correct minimum distance is the **distance to that corner point**, 
NOT a wall normal. There is no well-defined wall normal at a sharp corner.

#### Multi-Zone Grids
The nearest wall may be in a **different grid zone** than the field point. 
The wall distance computation must consider ALL wall surfaces from ALL zones, 
not just the walls in the local zone. This is particularly important for:
- Overset/chimera grids
- Multi-block structured grids
- Grids with internal walls or obstacles

#### Recommended Algorithms for Correct Wall Distance
1. **Brute-force search:** For each field point, compute distance to every wall 
   face/segment. Exact but O(N_field * N_wall). Acceptable for 2D.
2. **KD-tree / BVH search:** Build a spatial tree of wall segments. 
   O(N_field * log(N_wall)). Recommended for 3D.
3. **Signed distance field:** Solve the Eikonal equation |grad(d)| = 1. 
   Efficient for large grids but less exact.
4. **OpenFOAM:** Uses `wallDist` class with Poisson-equation or search methods.
5. **SU2:** Computes wall distance in preprocessing; verify the method used.

#### Consequence of Incorrect Wall Distance
- **Production term** (cb1 * S_tilde * nu_hat): S_tilde depends on d through fv2
- **Destruction term** (cw1 * fw * (nu_hat/d)^2): directly proportional to 1/d^2
- **r parameter**: r = nu_hat/(S_tilde * kappa^2 * d^2): also 1/d^2 dependence
- A 10% error in d near the wall produces ~20% error in destruction term
- This makes the SA solution **grid-topology dependent** (different grids give 
  different answers even at the same refinement level)


### 13.2 Compressible Flow Implementation
- Standard SA: density does NOT appear inside derivatives
- SA-noft2-Catris: includes density corrections for compressible flows
- Some implementations incorrectly introduce density — this alters predictions for 
  supersonic flows

### 13.3 Turbulence Index at Walls
For detecting transition:
```
i_t = ν_t_wall / ν
```

### 13.4 Conservation Form (from SA-neg paper, ICCFD7-1902)
For compressible flows, combining SA with mass conservation yields:
```
∂(ρν̂)/∂t + ∇·(ρν̂u) = ρ*cb1*S̃*ν̂ - ρ*cw1*fw*(ν̂/d)² 
                       + (1/σ)*[∇·(ρ(ν+ν̂)∇ν̂) + cb2*ρ*(∇ν̂)²]
```

---

## 14. Key References (Ordered by Importance for This Project)

1. **Spalart & Allmaras (1994)** — Original model  
   *Recherche Aerospatiale*, No. 1, pp. 5-21  

2. **Allmaras, Johnson & Spalart (2012)** — SA-neg and clarifications  
   ICCFD7-1902  
   https://www.iccfd.org/iccfd7/assets/pdf/papers/ICCFD7-1902_paper.pdf  

3. **Spalart & Rumsey (2007)** — Inflow conditions  
   *AIAA Journal* 45(10), pp. 2544-2553  
   DOI: 10.2514/1.29373  

4. **Spalart (2000)** — Trends in turbulence treatments  
   AIAA 2000-2306  
   DOI: 10.2514/6.2000-2306  

5. **Shur et al. (2000)** — SA-RC rotation/curvature  
   *AIAA Journal* 38(5), pp. 784-792  
   DOI: 10.2514/2.1058  

6. **Rumsey et al. (2020)** — QCR2020  
   *AIAA Journal* 58(10), pp. 4374-4384  
   DOI: 10.2514/1.J059683  

7. **Spalart & Garbaruk (2020)** — SA-LRe  
   *AIAA Journal* 58(5), pp. 1903-1905  
   DOI: 10.2514/1.J059489  

8. **Edwards & Chandra (1996)** — SA-Edwards modification  
   *AIAA Journal* 34(4), pp. 756-763  
   DOI: 10.2514/3.13137  

9. **Catris & Aupoix (2000)** — Compressible density corrections  
   *Aerospace Science and Technology* 4, pp. 1-11  
   DOI: 10.1016/S1270-9638(00)00112-7  

10. **Aupoix & Spalart (2003)** — Wall roughness  
    *Int. J. Heat and Fluid Flow* 24(4), pp. 454-462  
    DOI: 10.1016/S0142-727X(03)00043-2  

---

## 15. Compressible RANS Implementation

For compressible flows, the SA model is implemented as described on the TMR page 
"Implementing Turbulence Models into the Compressible RANS Equations" 
(https://turbmodels.larc.nasa.gov/implementrans.html):

- **Perfect gas** assumed
- **Pr = 0.72** (molecular)
- **Pr_t = 0.90** (turbulent)  
- **Sutherland's law** for dynamic viscosity:
  ```
  μ(T) = μ_ref * (T/T_ref)^(3/2) * (T_ref + S_T)/(T + S_T)
  ```
  where typically `μ_ref = 1.716e-5 Pa·s`, `T_ref = 273.15 K`, `S_T = 110.4 K`.

---

## 16. OpenFOAM / SU2 Implementation Notes

### OpenFOAM
- RASModel name: `SpalartAllmaras`
- Field variable: `nuTilda` (ν̂)
- For QCR: requires custom modification or specific OF version
- For DDES: `SpalartAllmarasDDES`

### SU2
- Turbulence model name: `SA`
- SA-neg available as: `SA_NEG`
- QCR available via config option

---

## 17. Relevance to CFD Benchmark Project

### For NACA 0012 Validation Case
- Use **SA-neg** (standard recommended form)
- Inflow BC: ν̂ = 3ν (standard TMR recommendation)
- Wall BC: ν̂ = 0 (no-slip)
- Compute proper wall distance d

### For Flow Separation Prediction
- SA alone: overpredicts separation bubble 20-35%
- SA-QCR: required for corner/juncture flows
- SA-RC: beneficial for rotation-dominated flows
- SA-noft2: recommended for DES/DDES base model

### For NASA 40% Challenge
- Standard SA as baseline
- SA-comp for transonic mixing layers
- SA-RC-QCR2000 for 3D complex cases

### Grid Requirements (from TMR notes)
- y+ ≤ 1 at walls (resolves viscous sublayer)
- True minimum wall distance computation essential
- Grid convergence study mandatory (at least 3 grid levels)
