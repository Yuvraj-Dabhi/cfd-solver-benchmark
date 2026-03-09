"""
OpenFOAM case generator utilities.
Automates the creation of system/, constant/, and 0/ directories.
"""
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

def foam_header(class_name: str, object_name: str, location: str = "") -> str:
    """Generate the standard OpenFOAM file header."""
    header = f"""/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2312                                 |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       {class_name};"""
    if location:
        header += f"\n    location    \"{location}\";"
    header += f"""
    object      {object_name};
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

"""
    return header


class FoamCaseGenerator:
    """Generates standard OpenFOAM case structures from Python configurations."""

    def __init__(self, case_dir: Path):
        self.case_dir = case_dir
        self.system_dir = case_dir / "system"
        self.constant_dir = case_dir / "constant"
        self.zero_dir = case_dir / "0"

    def setup_directories(self, clean: bool = True):
        """Create case directory structure."""
        if clean and self.case_dir.exists():
            shutil.rmtree(self.case_dir)
        
        self.system_dir.mkdir(parents=True, exist_ok=True)
        self.constant_dir.mkdir(parents=True, exist_ok=True)
        self.zero_dir.mkdir(parents=True, exist_ok=True)

    def write_controlDict(
        self,
        application: str = "interFoam",
        startFrom: str = "startTime",
        startTime: float = 0,
        stopAt: str = "endTime",
        endTime: float = 10.0,
        deltaT: float = 0.001,
        writeControl: str = "adjustable",
        writeInterval: float = 0.5,
        maxCo: float = 0.5,
        maxAlphaCo: float = 0.5,
        maxDeltaT: float = 0.1,
        adjustTimeStep: bool = True,
        extra_functions: str = "",
    ):
        """Write system/controlDict."""
        content = foam_header("dictionary", "controlDict")
        content += f"""application     {application};
startFrom       {startFrom};
startTime       {startTime};
stopAt          {stopAt};
endTime         {endTime};
deltaT          {deltaT};
writeControl    {writeControl};
writeInterval   {writeInterval};
purgeWrite      0;
writeFormat     ascii;
writePrecision  8;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable true;

adjustTimeStep  {"yes" if adjustTimeStep else "no"};
maxCo           {maxCo};
"""
        if application == "interFoam":
            content += f"maxAlphaCo      {maxAlphaCo};\n"
        
        content += f"maxDeltaT       {maxDeltaT};\n"
        
        if extra_functions:
            content += f"\n// Extra Functions\n{extra_functions}\n"

        (self.system_dir / "controlDict").write_text(content)

    def write_fvSchemes(self, template_str: str):
        """Write system/fvSchemes from template string."""
        content = foam_header("dictionary", "fvSchemes")
        content += template_str
        (self.system_dir / "fvSchemes").write_text(content)

    def write_fvSolution(self, template_str: str):
        """Write system/fvSolution from template string."""
        content = foam_header("dictionary", "fvSolution")
        content += template_str
        (self.system_dir / "fvSolution").write_text(content)

    def write_decomposeParDict(self, numberOfSubdomains: int = 4):
        """Write system/decomposeParDict."""
        content = foam_header("dictionary", "decomposeParDict")
        content += f"""numberOfSubdomains {numberOfSubdomains};
method          scotch;
"""
        (self.system_dir / "decomposeParDict").write_text(content)

    def write_transportProperties(self, template_str: str):
        """Write constant/transportProperties."""
        content = foam_header("dictionary", "transportProperties")
        content += template_str
        (self.constant_dir / "transportProperties").write_text(content)
        
    def write_momentumTransport(self, model: str = "kOmegaSST"):
        """Write constant/momentumTransport (OpenFOAM v2312+ style)."""
        content = foam_header("dictionary", "momentumTransport")
        # Support laminar or RAS
        sim_type = "RAS" if model != "laminar" else "laminar"
        content += f"""simulationType  {sim_type};

{sim_type}
{{
    model           {model};
    turbulence      on;
    printCoeffs     on;
}}
"""
        (self.constant_dir / "momentumTransport").write_text(content)

    def write_g(self, vector: Tuple[float, float, float] = (0, -9.81, 0)):
        """Write constant/g for gravity field."""
        content = foam_header("uniformDimensionedVectorField", "g", location="constant")
        content += f"""dimensions      [0 1 -2 0 0 0 0];
value           ( {vector[0]} {vector[1]} {vector[2]} );
"""
        (self.constant_dir / "g").write_text(content)

    def write_0_field(self, field_name: str, dim: str, internal_field: str, boundaries: Dict[str, str]):
        """Write a field file to the 0/ directory."""
        # Simple heuristic for class name
        if field_name == "U":
            class_name = "volVectorField"
        elif field_name == "p_rgh" or field_name == "p":
            class_name = "volScalarField"
        else:
            class_name = "volScalarField"

        content = foam_header(class_name, field_name, location="0")
        content += f"dimensions      {dim};\n\n"
        content += f"internalField   {internal_field};\n\n"
        content += "boundaryField\n{\n"
        for patch, bc in boundaries.items():
            content += f"    {patch}\n    {{\n{bc}    }}\n"
        content += "}\n"
        
        (self.zero_dir / field_name).write_text(content)
