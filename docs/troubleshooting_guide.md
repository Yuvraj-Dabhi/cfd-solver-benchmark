# Troubleshooting Guide

Reference for common issues when running the CFD Solver Benchmark pipeline.

---

## 1. Installation & Dependencies

### SALib import error
```
ModuleNotFoundError: No module named 'SALib'
```
**Fix:** `pip install SALib>=1.4`

### TensorFlow not found (ML module)
The ML module falls back to scikit-learn if TensorFlow is unavailable. For full PINN support:
```bash
pip install tensorflow>=2.10
```

### Gmsh / PyVista missing
Only needed for advanced mesh operations and 3D visualization:
```bash
pip install gmsh pyvista
```

---

## 2. OpenFOAM Issues

### Solver not found
```
FileNotFoundError: simpleFoam not found
```
**Fix:** Source OpenFOAM environment:
```bash
source /opt/openfoam2312/etc/bashrc
# Verify:
simpleFoam -help
```

### Floating point exception at startup
**Cause:** Usually bad mesh or boundary conditions.
**Fix:**
1. Run `checkMesh` first
2. Verify `y⁺ < 1` on wall-resolved grids
3. Start with first-order upwind (`scheme_id=0`), then switch to second-order

### Divergence after N iterations
**Cause:** CFL too high, bad relaxation, or inadequate mesh.
**Fix:**
1. Reduce relaxation factors: `U: 0.5`, `p: 0.2`
2. Use `--scheme-id 0` (first-order upwind) for initial stability
3. Check for negative cell volumes: `checkMesh -allGeometry`

### GAMG solver failure
```
GAMG: Solving for p, Initial residual = nan
```
**Fix:**
- Switch to `PCG` with `DIC` preconditioner for pressure
- Add `nNonOrthogonalCorrectors 1;` if mesh is non-orthogonal

### Wall function issues (y⁺ too high)
**Symptoms:** Poor Cf/Cp predictions on coarse grids.
**Fix:**
- Check y⁺ with `postProcess -func yPlus`
- For wall-resolved: target y⁺ ≈ 1
- For wall functions: target 30 < y⁺ < 300
- Use `nutUSpaldingWallFunction` for wide y⁺ range

---

## 3. SU2 Issues

### Config parse error
**Fix:** Check for trailing whitespace or missing `=` signs in `config.cfg`.

### Non-convergence with adaptive CFL
**Fix:** Reduce initial CFL and adapt parameters:
```
CFL_NUMBER= 1.0
CFL_ADAPT_PARAM= (0.5, 1.2, 1.0, 50.0)
```

### Mesh format error
SU2 expects `.su2` mesh format. Convert from other formats:
```bash
# From Gmsh
gmsh -3 mesh.geo -format su2 -o mesh.su2
```

---

## 4. Mesh Generation

### blockMesh crashes
**Fix:**
1. Verify `blockMeshDict` syntax: `blockMesh -check`
2. Ensure all vertex coordinates are correct
3. Check edge grading ratios are > 0

### Required y⁺ too small for available memory
The required first cell height for y⁺=1 can demand very fine meshes at high Re.
**Fix:** Use wall functions on coarse grids (y⁺ ≈ 30-100) and resolve walls only on fine/xfine levels.

---

## 5. Post-Processing

### Grid convergence shows "DIVERGENT"
**Cause:** Solutions are not in the asymptotic convergence range.
**Fix:**
- Add a fourth, coarser grid to verify convergence trend
- Ensure refinement ratio ≥ 1.3 (ideally 2.0)
- Check that the quantity of interest is mesh-sensitive

### Separation point not detected
**Cause:** Cf never crosses zero (no separation predicted).
**Fix:**
- Verify turbulence model is appropriate for the case
- Use finer mesh near expected separation
- Check if the model underpredicts adverse pressure gradient (common with k-ε)

### ASME V&V 20 shows "NOT VALIDATED"
**Cause:** |E| > U_val (error exceeds validation uncertainty).
**Fix:**
- Reduce numerical uncertainty via grid refinement
- Include input uncertainty in the combined metric
- Some model-form errors are inherent (e.g., RANS for 3D flows)

---

## 6. HPC / Parallel

### MPI segfault
**Fix:**
1. Verify consistent OpenFOAM/MPI versions
2. Use `scotch` decomposition (more robust than `simple`)
3. Check for sufficient memory per process

### SLURM job killed (OOM)
**Fix:** Increase `--mem-per-cpu` or reduce mesh size per processor.

### Inconsistent parallel results
**Fix:** Ensure `decomposeParDict` uses `scotch` method. Results should be identical regardless of processor count (within floating-point tolerance).

---

## 7. ML & PINN

### Loss not decreasing (PINN)
**Fix:**
1. Normalize inputs and outputs
2. Reduce `lambda_physics` initially, increase gradually
3. Increase collocation point density near walls

### Feature extraction returns NaN
**Cause:** Division by zero in `k/epsilon` or zero strain rate.
**Fix:** Add small epsilon: `k / (epsilon + 1e-10)`

---

## 8. Quick Diagnostic Checklist

| Symptom | First thing to check |
|---------|---------------------|
| Solver diverges | Relaxation factors, mesh quality |
| Wrong Cf | y⁺ value, turbulence model BCs |
| No separation | Model choice (avoid k-ε for APG) |
| GCI > 20% | Mesh refinement ratio, asymptotic range |
| Slow convergence | Switch to multigrid (GAMG), increase CFL |
| MPI crash | Memory, decomposition method |
