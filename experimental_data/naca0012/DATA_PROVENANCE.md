# NACA 0012 Experimental Data — Provenance & Usage Guide

Source: [NASA TMR 2DN00](https://turbmodels.larc.nasa.gov/naca0012_val.html)

**Case conditions**: M = 0.15, Re = 6×10⁶, fully turbulent, sharp TE

## Which Dataset to Use

| Comparison | Primary Dataset | Notes |
|-----------|----------------|-------|
| **CL vs α** | Ladson tripped | Most appropriate for fully turbulent CFD at Re=6M |
| **CD vs CL** | Ladson tripped | Tripped data essential — untripped CD levels are much lower |
| **Cp vs x/c** | Gregory & O'Reilly | Better LE resolution; more two-dimensional despite Re=3M |
| **Cf vs x/c** | CFL3D SA reference | No experimental Cf data available |
| **CL slope** | McCroskey best fit | dCL/dα reference line for linear regime |

## Experimental Sources

### Ladson (1988) — **PRIMARY for forces**
- **Citation**: Ladson, C. L., "Effects of Independent Variation of Mach and Reynolds
  Numbers on the Low-Speed Aerodynamic Characteristics of the NACA 0012 Airfoil Section,"
  NASA TM 4074, October 1988
- **URL**: https://ntrs.nasa.gov/citations/19880019495
- **Conditions**: Re=6M, M=0.15, **tripped** (3 grit sizes: 80, 120, 180)
- **Data**: CL vs α, CD vs α
- **Use for**: Force coefficient validation against fully turbulent CFD

### Gregory & O'Reilly (1970) — **PRIMARY for Cp**
- **Citation**: Gregory, N. and O'Reilly, C. L., "Low-Speed Aerodynamic
  Characteristics of NACA 0012 Aerofoil Sections, including the Effects of
  Upper-Surface Roughness Simulation Hoar Frost," R&M 3726, Jan 1970
- **Conditions**: Re=2.88M (lower Re), **tripped**
- **Data**: CL vs α, Cp vs x/c (upper surface, α=0, 10, 15)
- **Use for**: Surface pressure validation (better LE peak resolution)
- **Caveat**: CL not significantly affected by Re difference; CD is (~10% higher at Re=3M)

### Abbott & von Doenhoff (1959)
- **Citation**: Abbott, I. H. and von Doenhoff, A. E., "Theory of Wing Sections,"
  Dover Publications, New York, 1959
- **Conditions**: Re=6M, **NOT tripped**
- **Data**: CL vs α, CD vs CL
- **Use for**: Context only — **not appropriate for fully turbulent CFD drag comparison**

### McCroskey (1987)
- **Citation**: McCroskey, W. J., "A Critical Assessment of Wind Tunnel Results for
  the NACA 0012 Airfoil," AGARD CP-429, July 1988; also NASA TM 100019
- **URL**: https://ntrs.nasa.gov/citations/19880002254
- **Data**: Best-fit CL-α slope line for Re=6M, M=0.15
- **Formula**: dCL/dα = (0.1025 + 0.00485·log₁₀(Re/10⁶)) / √(1−M²)

### Ladson et al (1987) — Cp data
- **Citation**: Ladson, C. L., Hill, A. S., and Johnson, Jr., W. G., "Pressure
  Distributions from High Reynolds Number Transonic Tests of an NACA 0012 Airfoil
  in the Langley 0.3-Meter Transonic Cryogenic Tunnel," NASA TM 100526
- **URL**: https://ntrs.nasa.gov/citations/19880009181
- **Data**: Cp vs x/c at multiple Re (3M, 6M, 9M) and conditions
- **Caveat**: Does not resolve LE suction peak well; model aspect ratio only 1.333

## CFD Reference

### CFL3D SA (7-code consensus)
- **Grid**: 897×257 (Family I)
- **Codes**: CFL3D, FUN3D, NTS, JOE, SUMB, TURNS, GGNS
- **Spread**: Max 1% CL difference, 4% CD difference between codes
- **Data**: CL/CD at α=0/10/15, Cp and Cf distributions
- **Use for**: Code-to-code verification baseline

## Important Caveats

1. **2D experiments are difficult** — especially near stall (α > 12°), experiments
   are far from two-dimensional
2. **Tripping matters for drag** — untripped CD₀ is significantly lower than tripped
3. **No Cf data exists** — skin friction predictions are of interest but cannot be validated
4. **Grid sensitivity at high α** — CFD results at α > 15° are highly grid-dependent;
   results should be viewed qualitatively
