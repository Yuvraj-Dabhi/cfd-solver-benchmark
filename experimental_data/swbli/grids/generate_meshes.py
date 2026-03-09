"""Generate SWBLI meshes using GMSH API for L1 (coarse), L2 (medium), L3 (fine)."""
import gmsh
import sys
import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
GEO_FILE = SCRIPT_DIR / "swbli.geo"
OUTPUT_DIR = SCRIPT_DIR

LEVELS = {
    "L1_coarse": 1.0,
    "L2_medium": 1.41,
    "L3_fine": 2.0,
}

for name, f_val in LEVELS.items():
    print(f"Generating {name} (f={f_val})...")
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 1)

    # Read the geo file content and replace the f value
    geo_text = GEO_FILE.read_text()
    # Replace the first assignment of f
    import re
    geo_text = re.sub(r'^f\s*=\s*[\d.]+;', f'f = {f_val};', geo_text, count=1, flags=re.MULTILINE)

    # Write temp geo file
    tmp_geo = SCRIPT_DIR / f"_tmp_{name}.geo"
    tmp_geo.write_text(geo_text)

    gmsh.open(str(tmp_geo))
    gmsh.model.mesh.generate(2)

    # Export as SU2
    out_file = OUTPUT_DIR / f"swbli_{name}.su2"
    gmsh.write(str(out_file))

    # Get mesh stats
    nodes = gmsh.model.mesh.getNodes()
    n_nodes = len(nodes[0])
    elems_2d = gmsh.model.mesh.getElements(dim=2)
    n_elems = sum(len(e) for e in elems_2d[1])
    print(f"  {n_nodes} nodes, {n_elems} elements -> {out_file.name}")

    gmsh.finalize()
    tmp_geo.unlink()

print("\nDone!")
