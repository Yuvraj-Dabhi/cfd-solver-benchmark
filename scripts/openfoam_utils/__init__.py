"""
OpenFOAM integration utilities for the CFD Solver Benchmark project.
Provides automated case generation, file formatting, and log parsing.
"""

from .foam_case_generator import FoamCaseGenerator, foam_header
from .foam_log_parser import FoamLogParser

__all__ = ["FoamCaseGenerator", "foam_header", "FoamLogParser"]
