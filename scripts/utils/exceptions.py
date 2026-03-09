"""
Custom Exception Hierarchy for CFD Benchmark Framework
=======================================================
Typed exceptions for clear error categorization and robust pipeline handling.

Usage:
    from scripts.utils.exceptions import SolverError, ConvergenceError

    try:
        runner.run()
    except ConvergenceError as e:
        logger.warning(f"Case did not converge: {e}")
    except SolverError as e:
        logger.error(f"Solver crashed: {e}")
"""


class CFDBenchmarkError(Exception):
    """Base exception for all CFD benchmark framework errors."""
    pass


# ---------------------------------------------------------------------------
# Solver Errors
# ---------------------------------------------------------------------------
class SolverError(CFDBenchmarkError):
    """Error during CFD solver execution (SU2, OpenFOAM)."""
    def __init__(self, solver: str, message: str, case_dir: str = ""):
        self.solver = solver
        self.case_dir = case_dir
        super().__init__(f"[{solver}] {message}" + (f" (case: {case_dir})" if case_dir else ""))


class SolverNotFoundError(SolverError):
    """CFD solver binary not found on PATH."""
    def __init__(self, solver: str):
        super().__init__(solver, f"Solver binary '{solver}' not found on PATH. "
                         "Ensure it is installed and accessible.")


class ConvergenceError(SolverError):
    """Solver did not converge to target residual."""
    def __init__(self, solver: str, target_residual: float,
                 achieved_residual: float, iterations: int, case_dir: str = ""):
        self.target_residual = target_residual
        self.achieved_residual = achieved_residual
        self.iterations = iterations
        super().__init__(
            solver,
            f"Did not converge: target={target_residual:.0e}, "
            f"achieved={achieved_residual:.2e} after {iterations} iterations",
            case_dir,
        )


class SolverCrashError(SolverError):
    """Solver terminated abnormally (non-zero exit code, NaN, etc.)."""
    def __init__(self, solver: str, exit_code: int, case_dir: str = "",
                 stderr: str = ""):
        self.exit_code = exit_code
        self.stderr = stderr
        msg = f"Crashed with exit code {exit_code}"
        if stderr:
            msg += f"\n  stderr: {stderr[:500]}"
        super().__init__(solver, msg, case_dir)


# ---------------------------------------------------------------------------
# Mesh Errors
# ---------------------------------------------------------------------------
class MeshError(CFDBenchmarkError):
    """Error related to mesh operations."""
    pass


class MeshNotFoundError(MeshError):
    """Mesh file not found."""
    def __init__(self, mesh_path: str):
        self.mesh_path = mesh_path
        super().__init__(f"Mesh file not found: {mesh_path}")


class MeshQualityError(MeshError):
    """Mesh quality check failed (negative volumes, high skewness, etc.)."""
    def __init__(self, message: str, min_quality: float = 0.0):
        self.min_quality = min_quality
        super().__init__(f"Mesh quality issue: {message}")


# ---------------------------------------------------------------------------
# Data Errors
# ---------------------------------------------------------------------------
class DataError(CFDBenchmarkError):
    """Error in data loading, parsing, or validation."""
    pass


class DataNotFoundError(DataError):
    """Required data file not found."""
    def __init__(self, filepath: str, data_type: str = ""):
        self.filepath = filepath
        desc = f" ({data_type})" if data_type else ""
        super().__init__(f"Data file not found{desc}: {filepath}")


class DataFormatError(DataError):
    """Data file has unexpected format."""
    def __init__(self, filepath: str, expected: str, got: str = ""):
        self.filepath = filepath
        msg = f"Unexpected format in {filepath}: expected {expected}"
        if got:
            msg += f", got {got}"
        super().__init__(msg)


# ---------------------------------------------------------------------------
# Configuration Errors
# ---------------------------------------------------------------------------
class ConfigurationError(CFDBenchmarkError):
    """Invalid or missing configuration."""
    def __init__(self, parameter: str, message: str = ""):
        self.parameter = parameter
        super().__init__(f"Configuration error for '{parameter}': {message}")


class InvalidModelError(ConfigurationError):
    """Requested turbulence model not available."""
    def __init__(self, model: str, available: list):
        super().__init__(
            model,
            f"Model '{model}' not available. Choose from: {available}",
        )


# ---------------------------------------------------------------------------
# Validation Errors
# ---------------------------------------------------------------------------
class ValidationError(CFDBenchmarkError):
    """Validation check failed."""
    def __init__(self, metric: str, expected: float, actual: float,
                 tolerance: float):
        self.metric = metric
        self.expected = expected
        self.actual = actual
        self.tolerance = tolerance
        deviation = abs(actual - expected) / max(abs(expected), 1e-15) * 100
        super().__init__(
            f"Validation failed for {metric}: expected={expected:.6f}, "
            f"actual={actual:.6f} (deviation={deviation:.2f}%, "
            f"tolerance={tolerance*100:.1f}%)"
        )
