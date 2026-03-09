"""Quick functional verification of all newly created modules."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

print("=" * 55)
print("  Functional Verification of New Modules")
print("=" * 55)

# 1. y+ estimator
from scripts.preprocessing.yplus_estimator import (
    required_first_cell_height, estimate_yplus, boundary_layer_thickness
)
y1 = required_first_cell_height(Re=5e6, L=1.0, U=50.0, y_plus_target=1.0)
yp = estimate_yplus(Re=5e6, L=1.0, U=50.0, y1=y1)
bl = boundary_layer_thickness(1.0, 5e6)
assert abs(yp - 1.0) < 0.01, f"y+ mismatch: {yp}"
assert bl["delta_99"] > 0
print(f"[PASS] yplus_estimator: dy1={y1:.4e} m, y+={yp:.2f}, delta99={bl['delta_99']:.6f}")

# 2. physics_diagnostics
from scripts.postprocessing.physics_diagnostics import (
    boussinesq_validity, production_dissipation_ratio, lumley_triangle_invariants, run_all_diagnostics
)
N = 50
S = np.random.randn(N, 3, 3) * 0.1
S = 0.5 * (S + np.swapaxes(S, -2, -1))  # Symmetric
tau = np.random.randn(N, 3, 3) * 0.01
tau = 0.5 * (tau + np.swapaxes(tau, -2, -1))
k = np.abs(np.random.randn(N)) * 0.1 + 0.01
bv = boussinesq_validity(S, tau, k)
assert len(bv.values) == N
print(f"[PASS] physics_diagnostics: Boussinesq validity — {bv.summary}")

# 3. scheme_sensitivity
from scripts.postprocessing.scheme_sensitivity import analyze_scheme_sensitivity
quantities = {
    "x_reat": {"1st-order": 6.0, "2nd-order standard": 6.26, "limited": 6.20, "LUST": 6.22},
}
results = analyze_scheme_sensitivity(quantities)
assert "x_reat" in results
assert results["x_reat"].cv > 0
print(f"[PASS] scheme_sensitivity: x_reat spread={results['x_reat'].spread:.4f}, CV={results['x_reat'].cv:.2f}%")

# 4. visualization (import-only; no display needed)
from scripts.comparison.visualization import (
    plot_cf_comparison, plot_velocity_profiles, plot_law_of_wall,
    plot_grid_convergence, plot_mape_heatmap, plot_accuracy_vs_cost,
    plot_sobol_indices, close_all
)
close_all()
print("[PASS] visualization: all 8 plot functions imported successfully")

# 5. feature_extraction
from scripts.ml_augmentation.feature_extraction import extract_invariant_features, normalize_features
dudx = np.random.randn(100, 3, 3) * 0.1
k_vals = np.abs(np.random.randn(100)) + 0.01
eps_vals = np.abs(np.random.randn(100)) + 0.01
wd = np.abs(np.random.randn(100)) + 0.001
feats = extract_invariant_features(dudx, k_vals, eps_vals, wd)
assert feats.n_features == 14
norm_feats, params = normalize_features(feats, method="standard")
assert "mean" in params
print(f"[PASS] feature_extraction: {feats.n_features} features, {feats.n_points} points")

# 6. dataset
from scripts.ml_augmentation.dataset import DatasetBuilder
builder = DatasetBuilder()
builder.add_periodic_hills(alpha_values=[1.0], n_points=100)
ds = builder.build(name="test")
assert ds.n_samples == 100
train, val, test = ds.split()
assert train.n_samples + val.n_samples + test.n_samples == ds.n_samples
print(f"[PASS] dataset: {ds.n_samples} samples, train={train.n_samples}, val={val.n_samples}, test={test.n_samples}")

# 7. evaluate
from scripts.ml_augmentation.evaluate import evaluate_predictions, check_realizability
y_true = np.random.randn(200, 3)
y_pred = y_true + np.random.randn(200, 3) * 0.1
ev = evaluate_predictions(y_true, y_pred, model_name="test")
assert ev.r2 > 0.5
b_pred = np.random.randn(200, 3) * 0.01
rlz = check_realizability(b_pred)
assert "realizability_fraction" in rlz
print(f"[PASS] evaluate: RMSE={ev.rmse:.4f}, R²={ev.r2:.4f}, realizability={rlz['realizability_fraction']:.1f}%")

# 8. ROM
from scripts.ml_augmentation.rom import GalerkinROM, DEIM
snaps = np.random.randn(100, 20)
rom = GalerkinROM(n_modes=5)
rom.fit(snaps)
result = rom.predict(snapshot=snaps[:, 0])
assert result.n_modes <= 5
assert result.energy_captured > 0
print(f"[PASS] rom: {result.n_modes} modes, {result.energy_captured:.1f}% energy, err={result.reconstruction_error:.4e}")

# 9. DRL flow control
from scripts.ml_augmentation.drl_flow_control import FlowControlEnv
env = FlowControlEnv(n_actuators=3)
obs = env.reset(seed=42)
action = np.array([0.05, 0.0, -0.05])
obs2, r, term, trunc, info = env.step(action)
assert "bubble_length" in info
print(f"[PASS] drl_flow_control: bubble={info['bubble_length']:.2f}, reduction={info['bubble_reduction']:.1f}%")

# 10. flat_plate_verification
from scripts.validation.flat_plate_verification import verify_cf, law_of_wall, spalding_law
x = np.array([0.5, 1.0])
U = 50.0; nu = 1.5e-5
Re_vals = U * x / nu
cf_analytical = 0.059 / Re_vals**0.2
results = verify_cf(x, cf_analytical, U, nu)
assert all(r.passed for r in results)
yp = np.logspace(-1, 3, 50)
up = law_of_wall(yp)
assert len(up) == 50
up_sp = spalding_law(yp)
assert len(up_sp) == 50
print(f"[PASS] flat_plate_verification: {len(results)} Cf checks passed, law-of-wall OK")

print(f"\n{'=' * 55}")
print("  ALL 10 NEW MODULES VERIFIED SUCCESSFULLY [OK]")
print(f"{'=' * 55}")
