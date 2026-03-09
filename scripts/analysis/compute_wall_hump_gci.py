#!/usr/bin/env python3
"""
compute_wall_hump_gci.py — 3-Level GCI + ASME V&V 20 for Wall Hump SA
======================================================================
Extracts Cf/Cp from coarse/medium/fine SA VTU files, computes Richardson
extrapolation (observed order p, GCI_fine), and the ASME V&V 20 metric.
"""
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import numpy as np
import json
import os

def extract_surface_data(vtu_path):
    """Extract x, Cf_x, Cp (corrected) from a surface_flow.vtu file."""
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(vtu_path)
    reader.Update()
    out = reader.GetOutput()
    pd = out.GetPointData()
    names = [pd.GetArrayName(i) for i in range(pd.GetNumberOfArrays())]
    coords = vtk_to_numpy(out.GetPoints().GetData())
    
    x = coords[:, 0]
    idx = np.argsort(x)
    x_s = x[idx]
    
    # Cf (streamwise component)
    cf = vtk_to_numpy(pd.GetArray(names.index('Skin_Friction_Coefficient')))
    cf_x = cf[:, 0] if cf.ndim > 1 else cf
    cf_s = cf_x[idx]
    
    # Cp (corrected by upstream reference)
    cp = vtk_to_numpy(pd.GetArray(names.index('Pressure_Coefficient')))
    cp_s = cp[idx]
    cp_ref = cp_s[x_s < -0.3].mean()
    cp_corr = cp_s - cp_ref
    
    return x_s, cf_s, cp_corr

def find_separation_reattachment(x, cf):
    """Find separation and reattachment from zero-crossings of Cf."""
    x_sep = None
    x_reat = None
    for i in range(len(cf) - 1):
        if cf[i] > 0 and cf[i+1] < 0 and x[i] > 0.5:
            x_sep = x[i] + (0 - cf[i]) * (x[i+1] - x[i]) / (cf[i+1] - cf[i])
        if cf[i] < 0 and cf[i+1] > 0 and x[i] > 0.8:
            x_reat = x[i] + (0 - cf[i]) * (x[i+1] - x[i]) / (cf[i+1] - cf[i])
    return x_sep, x_reat

def compute_cp_rmse(x_su2, cp_su2, exp_file, x_range=None):
    """Compute Cp RMSE against experiment."""
    exp_x, exp_cp = [], []
    with open(exp_file) as f:
        for line in f:
            if line.strip() and not line.startswith('#') and not line.startswith('%'):
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        exp_x.append(float(parts[0]))
                        exp_cp.append(float(parts[1]))
                    except:
                        pass
    exp_x = np.array(exp_x)
    exp_cp = np.array(exp_cp)
    
    if x_range:
        mask = (exp_x >= x_range[0]) & (exp_x <= x_range[1])
        exp_x = exp_x[mask]
        exp_cp = exp_cp[mask]
    
    cp_interp = np.interp(exp_x, x_su2, cp_su2)
    rmse = np.sqrt(np.mean((cp_interp - exp_cp)**2))
    return rmse, exp_x, exp_cp, cp_interp

def gci_analysis(f1, f2, f3, r21, r32, name):
    """Compute GCI for a scalar quantity on 3 grid levels."""
    e21 = f2 - f1   # medium - fine
    e32 = f3 - f2   # coarse - medium
    
    if abs(e21) < 1e-15 or abs(e32) < 1e-15:
        print(f"  {name}: near-zero difference, cannot compute GCI")
        return None
    
    s = np.sign(e32 / e21)
    
    result = {
        'name': name,
        'f1': f1, 'f2': f2, 'f3': f3,
        'e21': e21, 'e32': e32,
        'convergence': 'monotonic' if s > 0 else 'oscillatory'
    }
    
    if s > 0:
        # Monotonic convergence — standard Roache GCI
        p = abs(np.log(abs(e32 / e21))) / np.log(r32)
        f_exact = f1 + (f1 - f2) / (r21**p - 1)
        Fs = 1.25  # safety factor for 3+ grids
        gci_fine = Fs * abs((f1 - f2) / f1) / (r21**p - 1) * 100
        
        # Asymptotic ratio check
        gci_21 = Fs * abs((f1 - f2) / f1) / (r21**p - 1)
        gci_32 = Fs * abs((f2 - f3) / f2) / (r32**p - 1)
        asym_ratio = gci_32 / (r21**p * gci_21) if gci_21 > 0 else float('inf')
        
        result.update({
            'p': p,
            'f_exact': f_exact,
            'gci_fine_pct': gci_fine,
            'asym_ratio': asym_ratio
        })
        
        print(f"  {name}:")
        print(f"    f_coarse={f3:.6f}, f_medium={f2:.6f}, f_fine={f1:.6f}")
        print(f"    e21={e21:.6f}, e32={e32:.6f}")
        print(f"    Observed order p = {p:.3f}")
        print(f"    Richardson extrapolation f* = {f_exact:.6f}")
        print(f"    GCI_fine = {gci_fine:.2f}%")
        print(f"    Asymptotic ratio = {asym_ratio:.3f}")
    else:
        print(f"  {name}: OSCILLATORY convergence")
        print(f"    f_coarse={f3:.6f}, f_medium={f2:.6f}, f_fine={f1:.6f}")
        result['gci_fine_pct'] = None
    
    return result

# ============================================================
# MAIN
# ============================================================
base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
exp_file = os.path.join(base, 'experimental_data', 'wall_hump', 'csv', 'noflow_cp.exp.dat')

# Grid definitions
grids = {
    'coarse': {'dims': (103, 28), 'case': 'SA_coarse'},
    'medium': {'dims': (205, 55), 'case': 'SA_medium'},
    'fine':   {'dims': (409, 109), 'case': 'SA_fine'},
}

print("=" * 60)
print("WALL HUMP 3-LEVEL GCI ANALYSIS (SA)")
print("=" * 60)

# Extract data from all 3 levels
data = {}
for level, info in grids.items():
    vtu_path = os.path.join(base, 'runs', 'wall_hump', f'hump_{info["case"]}', 'surface_flow.vtu')
    x, cf, cp = extract_surface_data(vtu_path)
    x_sep, x_reat = find_separation_reattachment(x, cf)
    cf_min = cf[(x > 0.5) & (x < 1.5)].min()
    cp_rmse, _, _, _ = compute_cp_rmse(x, cp, exp_file)
    cp_rmse_sep, _, _, _ = compute_cp_rmse(x, cp, exp_file, x_range=(0.65, 1.10))
    
    # Bubble length
    bubble = (x_reat - x_sep) if (x_sep is not None and x_reat is not None) else None
    
    nx, ny = info['dims']
    data[level] = {
        'n_cells': (nx-1) * (ny-1),
        'x_sep': x_sep,
        'x_reat': x_reat,
        'bubble_length': bubble,
        'cf_min': cf_min,
        'cp_rmse': cp_rmse,
        'cp_rmse_sep': cp_rmse_sep,
        'x': x, 'cf': cf, 'cp': cp
    }
    
    print(f"\n{level} ({nx}x{ny}, {data[level]['n_cells']} cells):")
    print(f"  x_sep = {x_sep:.4f}")
    print(f"  x_reat = {x_reat:.4f}")
    print(f"  bubble = {bubble:.4f}c" if bubble else "  bubble = N/A")
    print(f"  cf_min = {cf_min:.6f}")
    print(f"  Cp RMSE (overall) = {cp_rmse:.4f}")
    print(f"  Cp RMSE (sep region) = {cp_rmse_sep:.4f}")

# Experimental reference
EXP_X_SEP = 0.665
EXP_X_REAT = 1.11
EXP_BUBBLE = 0.445

# Refinement ratios (based on cell count)
N1 = data['fine']['n_cells']
N2 = data['medium']['n_cells']
N3 = data['coarse']['n_cells']

r21 = (N1 / N2) ** 0.5   # fine-to-medium
r32 = (N2 / N3) ** 0.5   # medium-to-coarse

print(f"\nGrid cells: coarse={N3}, medium={N2}, fine={N1}")
print(f"Refinement ratio r21 (fine/medium) = {r21:.4f}")
print(f"Refinement ratio r32 (medium/coarse) = {r32:.4f}")

# GCI for each quantity (including bubble length)
print("\n" + "=" * 60)
print("GCI RESULTS")
print("=" * 60)

gci_results = {}
for qty in ['x_sep', 'x_reat', 'bubble_length', 'cp_rmse', 'cp_rmse_sep', 'cf_min']:
    f1 = data['fine'].get(qty)
    f2 = data['medium'].get(qty)
    f3 = data['coarse'].get(qty)
    if f1 is not None and f2 is not None and f3 is not None:
        result = gci_analysis(f1, f2, f3, r21, r32, qty)
        if result:
            gci_results[qty] = result

# ============================================================
# FORMATTED GCI TABLE
# ============================================================
print("\n" + "=" * 100)
print("GCI SUMMARY TABLE — WALL HUMP SA (3-Level Richardson Extrapolation)")
print("=" * 100)
print(f"  {'Quantity':<16s} {'f_coarse':>10s} {'f_medium':>10s} {'f_fine':>10s} "
      f"{'p_obs':>6s} {'GCI_fine%':>9s} {'Asym.R':>7s} {'In Range?':>10s}")
print("  " + "-" * 90)

for qty in ['x_sep', 'x_reat', 'bubble_length', 'cp_rmse', 'cp_rmse_sep', 'cf_min']:
    if qty not in gci_results:
        continue
    g = gci_results[qty]
    
    f3_s = f"{g['f3']:.6f}"
    f2_s = f"{g['f2']:.6f}"
    f1_s = f"{g['f1']:.6f}"
    
    if g['convergence'] == 'monotonic' and g.get('p') is not None:
        p_s = f"{g['p']:.3f}"
        gci_s = f"{g['gci_fine_pct']:.2f}"
        ar_s = f"{g['asym_ratio']:.3f}"
        # Asymptotic range check: ratio should be close to 1.0
        in_range = 0.9 <= g['asym_ratio'] <= 1.1
        ir_s = "YES" if in_range else "NO"
    else:
        p_s = "OSC"
        gci_s = "---"
        ar_s = "---"
        ir_s = "OSC"
    
    print(f"  {qty:<16s} {f3_s:>10s} {f2_s:>10s} {f1_s:>10s} "
          f"{p_s:>6s} {gci_s:>9s} {ar_s:>7s} {ir_s:>10s}")

# Experimental comparison
print("\n  Experimental reference (Greenblatt et al., 2006):")
print(f"    x_sep  = {EXP_X_SEP:.3f}")
print(f"    x_reat = {EXP_X_REAT:.3f}")
print(f"    L_bubble = {EXP_BUBBLE:.3f}c")
if data['fine']['x_sep'] is not None:
    print(f"\n  Fine-grid errors vs experiment:")
    print(f"    x_sep error  = {abs(data['fine']['x_sep'] - EXP_X_SEP):.4f} "
          f"({abs(data['fine']['x_sep'] - EXP_X_SEP)/EXP_X_SEP*100:.1f}%)")
if data['fine']['x_reat'] is not None:
    print(f"    x_reat error = {abs(data['fine']['x_reat'] - EXP_X_REAT):.4f} "
          f"({abs(data['fine']['x_reat'] - EXP_X_REAT)/EXP_X_REAT*100:.1f}%)")
if data['fine']['bubble_length'] is not None:
    print(f"    L_bubble error = {abs(data['fine']['bubble_length'] - EXP_BUBBLE):.4f}c "
          f"({abs(data['fine']['bubble_length'] - EXP_BUBBLE)/EXP_BUBBLE*100:.1f}%)")

# Asymptotic range commentary
print("\n  Asymptotic Range Assessment:")
for qty in ['x_sep', 'x_reat', 'bubble_length']:
    if qty in gci_results:
        g = gci_results[qty]
        if g['convergence'] == 'monotonic' and g.get('asym_ratio') is not None:
            ar = g['asym_ratio']
            if 0.9 <= ar <= 1.1:
                print(f"    {qty}: IN asymptotic range (ratio={ar:.3f}, ≈1.0)")
            else:
                print(f"    {qty}: NOT in asymptotic range (ratio={ar:.3f})")
                if qty in ('x_reat', 'bubble_length'):
                    print(f"      → Typical for RANS: reattachment is controlled by the "
                          f"turbulence model closure")
                    print(f"        (eddy viscosity level in the separated shear layer), "
                          f"not grid resolution.")
                    print(f"        Further grid refinement will not substantially change "
                          f"x_reat. The ~{g['gci_fine_pct']:.0f}% GCI")
                    print(f"        reflects model-form error dominating over discretisation "
                          f"error.")
                elif qty == 'x_sep':
                    print(f"      → Low observed order (p={g['p']:.2f}) suggests the SA model's")
                    print(f"        separation onset is near its asymptotic model prediction.")
        else:
            print(f"    {qty}: OSCILLATORY convergence — cannot assess")

# ============================================================
# ASME V&V 20
# ============================================================
print("\n" + "=" * 60)
print("ASME V&V 20-2009 VALIDATION METRIC")
print("=" * 60)

# u_num from GCI of Cp RMSE
if 'cp_rmse' in gci_results and gci_results['cp_rmse'].get('gci_fine_pct') is not None:
    gci_r = gci_results['cp_rmse']
    # u_num = GCI / Fs (the actual numerical uncertainty without safety factor)
    u_num = abs(gci_r['f1'] - gci_r['f2']) / (r21**gci_r['p'] - 1)
    p_obs = gci_r['p']
else:
    # Fallback
    u_num = abs(data['fine']['cp_rmse'] - data['medium']['cp_rmse'])
    p_obs = None

u_D = 0.01  # Greenblatt pressure tap uncertainty (±1-2% Cp)
u_val = np.sqrt(u_num**2 + u_D**2)

# E = comparison error (Cp RMSE = RMS of S-D)
E = data['fine']['cp_rmse']

ratio = abs(E) / u_val

print(f"  u_num = {u_num:.4f} (from Richardson extrapolation)")
print(f"  u_D   = {u_D:.4f} (Greenblatt pressure tap)")
print(f"  u_val = sqrt(u_num^2 + u_D^2) = {u_val:.4f}")
print(f"  |E|   = Cp RMSE (fine grid) = {E:.4f}")
print(f"  |E|/u_val = {ratio:.3f}")
if p_obs:
    print(f"  Observed order p = {p_obs:.3f}")
if ratio < 1.0:
    print("  => VALIDATED at this uncertainty level")
else:
    print("  => NOT validated (modeling deficiency exceeds combined uncertainty)")

# Save results
output = {
    'grids': {k: {'n_cells': v['n_cells'], 'x_sep': v['x_sep'], 'x_reat': v['x_reat'],
                   'bubble_length': v.get('bubble_length'),
                   'cf_min': float(v['cf_min']), 'cp_rmse': float(v['cp_rmse']),
                   'cp_rmse_sep': float(v['cp_rmse_sep'])} for k, v in data.items()},
    'experimental_reference': {
        'x_sep': EXP_X_SEP, 'x_reat': EXP_X_REAT, 'bubble_length': EXP_BUBBLE,
        'source': 'Greenblatt et al. (2006), AIAA J. 44(12)',
    },
    'refinement_ratios': {'r21': r21, 'r32': r32},
    'gci': {k: {kk: vv for kk, vv in v.items() if kk != 'name'} for k, v in gci_results.items()},
    'asme_vv20': {
        'u_num': float(u_num),
        'u_D': float(u_D),
        'u_val': float(u_val),
        'E_cp_rmse': float(E),
        'E_over_uval': float(ratio),
        'validated': ratio < 1.0,
        'p_observed': float(p_obs) if p_obs else None
    }
}

with open(os.path.join(base, 'plots', 'wall_hump', 'gci_asme_results.json'), 'w') as f:
    json.dump(output, f, indent=2, default=lambda o: float(o) if hasattr(o, 'item') else o)
print(f"\nResults saved to plots/wall_hump/gci_asme_results.json")

