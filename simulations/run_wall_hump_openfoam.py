#!/usr/bin/env python3
"""
Set up and run NASA Wall Hump in OpenFOAM via WSL.
Converts the SU2 medium grid (205×55) and runs simpleFoam with SA.
"""
import os, sys, shutil, subprocess, json
from pathlib import Path
import numpy as np

PROJECT = Path(__file__).resolve().parent
CASE_DIR = PROJECT / "runs" / "wall_hump" / "hump_SA_medium_openfoam"
SU2_MESH = PROJECT / "runs" / "wall_hump" / "hump_SA_medium" / "mesh.su2"
GRIDS_DIR = PROJECT / "experimental_data" / "wall_hump" / "grids"

# Flow conditions (TMR specification — match SU2 exactly)
U_INF = 34.6   # m/s  (M=0.1, a~346 m/s)
RHO   = 1.1766  # kg/m³
NU    = 1.58e-5  # kinematic viscosity
RE    = 936000.0
CHORD = 1.0
MU_T_RATIO = 3.0  # SA: nu_tilde_inf = 3*nu

# ===================================================================
# Step 1: Convert SU2 mesh to OpenFOAM format
# ===================================================================
def convert_su2_to_openfoam_mesh():
    """
    Parse SU2 mesh and write OpenFOAM polyMesh.
    The SU2 wall hump mesh is 2D structured.
    We'll write a simple converter for the structured mesh.
    """
    print("[1/4] Converting SU2 mesh to OpenFOAM format...")

    # Parse SU2 mesh
    points = []
    elements = []
    markers = {}

    with open(SU2_MESH, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if line.startswith('NPOIN=') or line.startswith('NPOIN ='):
            npts = int(line.split('=')[1].strip().split()[0])
            for j in range(npts):
                i += 1
                parts = lines[i].strip().split()
                x, y = float(parts[0]), float(parts[1])
                points.append([x, y, 0.0])
            print(f"  Points: {len(points)}")

        elif line.startswith('NELEM=') or line.startswith('NELEM ='):
            nelem = int(line.split('=')[1].strip())
            for j in range(nelem):
                i += 1
                parts = lines[i].strip().split()
                etype = int(parts[0])
                nodes = [int(p) for p in parts[1:]]
                # Remove trailing index if present
                if etype == 9:  # quad
                    elements.append(nodes[:4])
                elif etype == 5:  # tri
                    elements.append(nodes[:3])
            print(f"  Elements: {len(elements)}")

        elif line.startswith('MARKER_TAG=') or line.startswith('MARKER_TAG ='):
            tag = line.split('=')[1].strip()
            i += 1
            nmark = int(lines[i].strip().split('=')[1].strip())
            edges = []
            for j in range(nmark):
                i += 1
                parts = lines[i].strip().split()
                etype = int(parts[0])
                nodes = [int(p) for p in parts[1:]]
                if etype == 3:  # line segment
                    edges.append(nodes[:2])
            markers[tag] = edges
            print(f"  Marker '{tag}': {len(edges)} edges")

        i += 1

    points = np.array(points)
    return points, elements, markers


def write_openfoam_case(points, elements, markers):
    """Write OpenFOAM case directory structure."""
    print("\n[2/4] Writing OpenFOAM case directory...")

    # Create directory structure
    poly_dir = CASE_DIR / "constant" / "polyMesh"
    poly_dir.mkdir(parents=True, exist_ok=True)
    (CASE_DIR / "0").mkdir(exist_ok=True)
    (CASE_DIR / "system").mkdir(exist_ok=True)

    # --- Write points ---
    npts = len(points)
    # Extrude to 3D (add z=0 and z=0.1 layers)
    pts_3d = []
    for p in points:
        pts_3d.append(p)  # z=0 face
    for p in points:
        pts_3d.append([p[0], p[1], 0.1])  # z=0.1 face

    with open(poly_dir / "points", 'w') as f:
        f.write(foam_header("vectorField", "points"))
        f.write(f"{len(pts_3d)}\n(\n")
        for p in pts_3d:
            f.write(f"({p[0]} {p[1]} {p[2]})\n")
        f.write(")\n")

    # --- Build faces and cells ---
    # For 2D: each quad element becomes one cell with 6 faces (4 internal/boundary + 2 empty z-faces)
    n2d = npts  # offset for back plane

    # Internal faces + boundary faces
    all_faces = []
    cell_faces = [[] for _ in range(len(elements))]
    owner = []
    neighbour = []
    boundary_faces = {tag: [] for tag in markers}
    front_faces = []
    back_faces = []

    # Map edge->element for boundary identification
    edge_to_cells = {}
    for ci, elem in enumerate(elements):
        n = len(elem)
        for k in range(n):
            e = tuple(sorted([elem[k], elem[(k+1) % n]]))
            edge_to_cells.setdefault(e, []).append(ci)

    # Identify boundary edges
    boundary_edges = set()
    for tag, edges in markers.items():
        for e in edges:
            boundary_edges.add(tuple(sorted(e)))

    # Internal faces: shared between two cells
    internal_face_set = set()
    for ci, elem in enumerate(elements):
        n = len(elem)
        for k in range(n):
            e = tuple(sorted([elem[k], elem[(k+1) % n]]))
            if e in internal_face_set:
                continue
            cells = edge_to_cells.get(e, [])
            p0, p1 = e
            if len(cells) == 2 and e not in boundary_edges:
                # Internal face: owner=lower cell, neighbour=higher cell
                c_own = min(cells)
                c_nei = max(cells)
                face = [p0, p1, p1 + n2d, p0 + n2d]
                fi = len(all_faces)
                all_faces.append(face)
                owner.append(c_own)
                neighbour.append(c_nei)
                internal_face_set.add(e)

    n_internal = len(all_faces)

    # Boundary faces
    for tag, edges in markers.items():
        start = len(all_faces)
        for e in edges:
            p0, p1 = e
            cells = edge_to_cells.get(tuple(sorted(e)), [])
            if cells:
                c_own = cells[0]
            else:
                c_own = 0
            face = [p0, p1, p1 + n2d, p0 + n2d]
            all_faces.append(face)
            owner.append(c_own)
        boundary_faces[tag] = (start, len(all_faces) - start)

    # Front and back (empty) faces
    front_start = len(all_faces)
    for ci, elem in enumerate(elements):
        # Back face (z=0)
        face = list(elem)
        all_faces.append(face)
        owner.append(ci)
    back_start = len(all_faces)
    for ci, elem in enumerate(elements):
        # Front face (z=0.1)
        face = [n + n2d for n in elem]
        face.reverse()  # Normal points outward (+z)
        all_faces.append(face)
        owner.append(ci)

    # Write faces
    with open(poly_dir / "faces", 'w') as f:
        f.write(foam_header("faceList", "faces"))
        f.write(f"{len(all_faces)}\n(\n")
        for face in all_faces:
            f.write(f"{len(face)}({' '.join(str(n) for n in face)})\n")
        f.write(")\n")

    # Write owner
    with open(poly_dir / "owner", 'w') as f:
        f.write(foam_header("labelList", "owner"))
        f.write(f"{len(owner)}\n(\n")
        for o in owner:
            f.write(f"{o}\n")
        f.write(")\n")

    # Write neighbour
    with open(poly_dir / "neighbour", 'w') as f:
        f.write(foam_header("labelList", "neighbour"))
        f.write(f"{len(neighbour)}\n(\n")
        for n in neighbour:
            f.write(f"{n}\n")
        f.write(")\n")

    # Write boundary
    # Map SU2 markers to OpenFOAM patches
    su2_to_of = {
        'lower_wall': 'humpWall',
        'wall': 'humpWall',
        'hump': 'humpWall',
        'lower': 'humpWall',
        'upper_wall': 'upperWall',
        'upper': 'upperWall',
        'top': 'upperWall',
        'inlet': 'inlet',
        'outlet': 'outlet',
        'inflow': 'inlet',
        'outflow': 'outlet',
    }

    with open(poly_dir / "boundary", 'w') as f:
        f.write(foam_header("polyBoundaryMesh", "boundary"))
        n_patches = len(boundary_faces) + 2  # +front +back
        f.write(f"{n_patches}\n(\n")
        for tag, (start, count) in boundary_faces.items():
            of_name = su2_to_of.get(tag.lower(), tag)
            of_type = "wall" if "wall" in of_name.lower() or "hump" in of_name.lower() else "patch"
            f.write(f"    {of_name}\n    {{\n")
            f.write(f"        type            {of_type};\n")
            f.write(f"        nFaces          {count};\n")
            f.write(f"        startFace       {start};\n")
            f.write(f"    }}\n")
        # Front
        f.write(f"    frontAndBack_back\n    {{\n")
        f.write(f"        type            empty;\n")
        f.write(f"        nFaces          {len(elements)};\n")
        f.write(f"        startFace       {front_start};\n")
        f.write(f"    }}\n")
        # Back
        f.write(f"    frontAndBack_front\n    {{\n")
        f.write(f"        type            empty;\n")
        f.write(f"        nFaces          {len(elements)};\n")
        f.write(f"        startFace       {back_start};\n")
        f.write(f"    }}\n")
        f.write(")\n")

    print(f"  polyMesh written: {len(all_faces)} faces, {len(elements)} cells")
    print(f"  Internal faces: {n_internal}")
    for tag, (start, count) in boundary_faces.items():
        print(f"  {tag}: {count} faces")

    return list(boundary_faces.keys())


def foam_header(class_name, object_name):
    return f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       {class_name};
    object      {object_name};
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

"""


def write_initial_conditions(patch_names):
    """Write 0/ directory files for SA turbulence model."""
    print("\n[3/4] Writing initial/boundary conditions...")

    bc_dir = CASE_DIR / "0"

    # Map patch names for boundary conditions
    wall_patches = [p for p in patch_names if 'wall' in p.lower() or 'hump' in p.lower()]
    inlet_patches = [p for p in patch_names if 'inlet' in p.lower() or 'inflow' in p.lower()]
    outlet_patches = [p for p in patch_names if 'outlet' in p.lower() or 'outflow' in p.lower()]
    slip_patches = [p for p in patch_names if 'upper' in p.lower() or 'top' in p.lower()]

    def bc_entry(patch, bc_type, value=None, extra=""):
        s = f"    {patch}\n    {{\n        type            {bc_type};\n"
        if value is not None:
            s += f"        value           {value};\n"
        if extra:
            s += extra
        s += f"    }}\n"
        return s

    def empty_bc():
        return "    frontAndBack_back\n    {\n        type            empty;\n    }\n" + \
               "    frontAndBack_front\n    {\n        type            empty;\n    }\n"

    # --- U ---
    with open(bc_dir / "U", 'w') as f:
        f.write(foam_header("volVectorField", "U"))
        f.write(f'dimensions      [0 1 -1 0 0 0 0];\n\n')
        f.write(f'internalField   uniform ({U_INF} 0 0);\n\n')
        f.write('boundaryField\n{\n')
        for p in wall_patches:
            f.write(bc_entry(p, "noSlip"))
        for p in inlet_patches:
            f.write(bc_entry(p, "fixedValue", f"uniform ({U_INF} 0 0)"))
        for p in outlet_patches:
            f.write(bc_entry(p, "zeroGradient"))
        for p in slip_patches:
            f.write(bc_entry(p, "slip"))
        f.write(empty_bc())
        f.write('}\n')

    # --- p ---
    with open(bc_dir / "p", 'w') as f:
        f.write(foam_header("volScalarField", "p"))
        f.write(f'dimensions      [0 2 -2 0 0 0 0];\n\n')
        p_val = 101325.0 / RHO  # kinematic pressure p/rho
        f.write(f'internalField   uniform {p_val};\n\n')
        f.write('boundaryField\n{\n')
        for p in wall_patches:
            f.write(bc_entry(p, "zeroGradient"))
        for p in inlet_patches:
            f.write(bc_entry(p, "zeroGradient"))
        for p in outlet_patches:
            f.write(bc_entry(p, "fixedValue", f"uniform {p_val}"))
        for p in slip_patches:
            f.write(bc_entry(p, "slip"))
        f.write(empty_bc())
        f.write('}\n')

    # --- nuTilda (SA working variable) ---
    nu_tilda_inf = MU_T_RATIO * NU  # 3 * nu
    with open(bc_dir / "nuTilda", 'w') as f:
        f.write(foam_header("volScalarField", "nuTilda"))
        f.write(f'dimensions      [0 2 -1 0 0 0 0];\n\n')
        f.write(f'internalField   uniform {nu_tilda_inf:.6e};\n\n')
        f.write('boundaryField\n{\n')
        for p in wall_patches:
            f.write(bc_entry(p, "fixedValue", "uniform 0"))
        for p in inlet_patches:
            f.write(bc_entry(p, "fixedValue", f"uniform {nu_tilda_inf:.6e}"))
        for p in outlet_patches:
            f.write(bc_entry(p, "zeroGradient"))
        for p in slip_patches:
            f.write(bc_entry(p, "slip"))
        f.write(empty_bc())
        f.write('}\n')

    # --- nut (turbulent kinematic viscosity) ---
    nut_inf = MU_T_RATIO * NU
    with open(bc_dir / "nut", 'w') as f:
        f.write(foam_header("volScalarField", "nut"))
        f.write(f'dimensions      [0 2 -1 0 0 0 0];\n\n')
        f.write(f'internalField   uniform {nut_inf:.6e};\n\n')
        f.write('boundaryField\n{\n')
        for p in wall_patches:
            f.write(bc_entry(p, "nutLowReWallFunction",
                             "uniform 0"))
        for p in inlet_patches:
            f.write(bc_entry(p, "calculated", f"uniform {nut_inf:.6e}"))
        for p in outlet_patches:
            f.write(bc_entry(p, "calculated", f"uniform {nut_inf:.6e}"))
        for p in slip_patches:
            f.write(bc_entry(p, "slip"))
        f.write(empty_bc())
        f.write('}\n')

    print("  Written: U, p, nuTilda, nut")


def write_system_files():
    """Write system/ directory files."""
    sys_dir = CASE_DIR / "system"

    # --- controlDict ---
    with open(sys_dir / "controlDict", 'w') as f:
        f.write(foam_header("dictionary", "controlDict"))
        f.write("""application     simpleFoam;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         20000;
deltaT          1;
writeControl    timeStep;
writeInterval   20000;
purgeWrite      1;
writeFormat     ascii;
writePrecision  8;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable true;

functions
{
    wallShearStress
    {
        type            wallShearStress;
        libs            ("libfieldFunctionObjects.so");
        writeControl    writeStep;
        patches         (humpWall);
    }

    yPlus
    {
        type            yPlus;
        libs            ("libfieldFunctionObjects.so");
        writeControl    writeStep;
    }

    forces
    {
        type            forces;
        libs            ("libforces.so");
        writeControl    timeStep;
        writeInterval   100;
        patches         (humpWall);
        rho             rhoInf;
        rhoInf          1.1766;
        CofR            (0 0 0);
    }
}
""")

    # --- fvSchemes ---
    with open(sys_dir / "fvSchemes", 'w') as f:
        f.write(foam_header("dictionary", "fvSchemes"))
        f.write("""ddtSchemes
{
    default         steadyState;
}

gradSchemes
{
    default         Gauss linear;
    grad(U)         cellLimited Gauss linear 1;
    grad(nuTilda)   cellLimited Gauss linear 1;
}

divSchemes
{
    default         none;
    div(phi,U)      bounded Gauss linearUpwindV grad(U);
    div(phi,nuTilda) bounded Gauss linearUpwind grad(nuTilda);
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}

laplacianSchemes
{
    default         Gauss linear corrected;
}

interpolationSchemes
{
    default         linear;
}

snGradSchemes
{
    default         corrected;
}

wallDist
{
    method          meshWave;
}
""")

    # --- fvSolution ---
    with open(sys_dir / "fvSolution", 'w') as f:
        f.write(foam_header("dictionary", "fvSolution"))
        f.write("""solvers
{
    p
    {
        solver          GAMG;
        tolerance       1e-8;
        relTol          0.01;
        smoother        GaussSeidel;
    }

    U
    {
        solver          smoothSolver;
        smoother        GaussSeidel;
        tolerance       1e-8;
        relTol          0.01;
    }

    nuTilda
    {
        solver          smoothSolver;
        smoother        GaussSeidel;
        tolerance       1e-8;
        relTol          0.01;
    }
}

SIMPLE
{
    nNonOrthogonalCorrectors 1;
    consistent      yes;

    residualControl
    {
        p               1e-6;
        U               1e-6;
        nuTilda         1e-6;
    }
}

relaxationFactors
{
    fields
    {
        p               0.3;
    }
    equations
    {
        U               0.7;
        nuTilda         0.7;
    }
}
""")

    # --- turbulenceProperties ---
    const_dir = CASE_DIR / "constant"
    with open(const_dir / "turbulenceProperties", 'w') as f:
        f.write(foam_header("dictionary", "turbulenceProperties"))
        f.write("""simulationType  RAS;

RAS
{
    RASModel        SpalartAllmaras;
    turbulence      on;
    printCoeffs     on;
}
""")

    # --- momentumTransport (OpenFOAM 13 naming) ---
    with open(const_dir / "momentumTransport", 'w') as f:
        f.write(foam_header("dictionary", "momentumTransport"))
        f.write("""simulationType  RAS;

RAS
{
    model           SpalartAllmaras;
    turbulence      on;
    printCoeffs     on;
}
""")

    # --- transportProperties ---
    with open(const_dir / "transportProperties", 'w') as f:
        f.write(foam_header("dictionary", "transportProperties"))
        f.write(f"""transportModel  Newtonian;
nu              {NU:.6e};
""")

    print("  Written: controlDict, fvSchemes, fvSolution, turbulenceProperties, transportProperties")


def run_openfoam():
    """Run simpleFoam via WSL."""
    print("\n[4/4] Running simpleFoam (SA) via WSL...")

    # Convert Windows path to WSL path
    win_path = str(CASE_DIR).replace('\\', '/')
    # WSL mounts C: as /mnt/c
    wsl_path = '/mnt/c' + win_path[2:]  # Remove C:

    cmd = f"""
cd '{wsl_path}' && \
source /opt/openfoam13/etc/bashrc && \
simpleFoam > of_log.txt 2>&1
"""

    print(f"  WSL path: {wsl_path}")
    print(f"  Running simpleFoam...")

    proc = subprocess.Popen(
        ['wsl', '-d', 'Ubuntu', '--', 'bash', '-c', cmd],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )

    return proc


# ===================================================================
# MAIN
# ===================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("  NASA Wall Hump — OpenFOAM Setup (SA, Medium Grid 205×55)")
    print("=" * 70)

    # Parse SU2 mesh
    points, elements, markers = convert_su2_to_openfoam_mesh()

    print(f"\nSU2 markers found: {list(markers.keys())}")

    # Write OpenFOAM case
    patch_names = write_openfoam_case(points, elements, markers)
    write_initial_conditions(patch_names)
    write_system_files()

    print(f"\nCase directory: {CASE_DIR}")
    print("To run: wsl -d Ubuntu -- bash -c 'cd <path> && source /opt/openfoam13/etc/bashrc && simpleFoam'")

    # Launch
    proc = run_openfoam()
    print(f"PID: {proc.pid}")
    print("Running in background...")
