#!/usr/bin/env python3
"""Stub for backward compatibility."""
import sys
import subprocess
import warnings
from pathlib import Path

if __name__ == "__main__":
    warnings.warn(
        "This script has moved to scripts/analysis/compute_wall_hump_gci.py. "
        "This stub will be removed in a future version.",
        DeprecationWarning,
        stacklevel=2
    )
    script = Path(__file__).resolve().parent / "scripts" / "analysis" / "compute_wall_hump_gci.py"
    sys.exit(subprocess.call([sys.executable, str(script)] + sys.argv[1:]))
