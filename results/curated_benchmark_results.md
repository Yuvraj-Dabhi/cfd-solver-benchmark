### Benchmark Results on McConkey et al. Dataset Splits

**Training Geometries:** `bump_h38, bump_h20`
**Testing Geometries:** `bump_h26, bump_h31`

| Model | U_mag_RMS | kRMS | uuRMS | uvRMS | vvRMS |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline (RANS-correction)** | 0.00343 | 0.00361 | 0.00000 | 0.00000 | 0.00000 |
| **TBNN Closure** | 1.73296 | 0.57871 | 5.84932 | 2.70321 | 7.50576 |
| **FIML MLP** | 3.32200 | 1.52570 | 1.35315 | 6.95519 | 1.89499 |
| **PINN BL** | 0.00102 | 0.00099 | 0.00000 | 0.00000 | 0.00000 |
