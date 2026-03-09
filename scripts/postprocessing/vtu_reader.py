"""
VTU/VTK File Reader (pyvista/meshio-based)
==========================================
Read SU2 VTK output files using industry-standard libraries
instead of raw vtk bindings.

Features:
  - Read .vtu, .vtk, .vtp files
  - Extract surface data (wall Cf, Cp)
  - Volume slicing at arbitrary stations
  - Automatic field name detection

Usage:
    reader = VTUReader("flow.vtu")
    wall = reader.extract_surface("wall")
    profile = reader.slice_at_x(x_station=0.65)

Falls back to meshio if pyvista is not installed.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Try pyvista first, then meshio
_HAS_PYVISTA = False
_HAS_MESHIO = False
try:
    import pyvista as pv
    _HAS_PYVISTA = True
    logger.debug("pyvista available for VTU reading")
except ImportError:
    pass

try:
    import meshio
    _HAS_MESHIO = True
    logger.debug("meshio available for VTU reading")
except ImportError:
    pass


@dataclass
class VTUField:
    """Metadata for a field in VTU data."""
    name: str
    n_components: int
    field_type: str  # "point" or "cell"
    dtype: str


class VTUReader:
    """
    Read SU2/OpenFOAM VTK output files.

    Uses pyvista for full functionality (slicing, surface extraction),
    falls back to meshio for basic data reading.

    Parameters
    ----------
    filepath : str or Path
        Path to VTU/VTK/VTP file.
    """

    def __init__(self, filepath):
        self.filepath = Path(filepath)
        if not self.filepath.exists():
            from scripts.utils.exceptions import DataNotFoundError
            raise DataNotFoundError(str(self.filepath), "VTU/VTK file")

        self._mesh = None
        self._meshio_mesh = None
        self._load()

    def _load(self):
        """Load the mesh using best available library."""
        if _HAS_PYVISTA:
            try:
                self._mesh = pv.read(str(self.filepath))
                logger.info(
                    f"Loaded {self.filepath.name}: "
                    f"{self._mesh.n_points} points, "
                    f"{self._mesh.n_cells} cells"
                )
                return
            except Exception as e:
                logger.warning(f"pyvista failed: {e}, trying meshio")

        if _HAS_MESHIO:
            try:
                self._meshio_mesh = meshio.read(str(self.filepath))
                n_pts = len(self._meshio_mesh.points)
                logger.info(f"Loaded {self.filepath.name} via meshio: {n_pts} points")
                return
            except Exception as e:
                logger.error(f"meshio also failed: {e}")

        logger.warning(
            "Neither pyvista nor meshio available. "
            "Install with: pip install pyvista meshio"
        )

    @property
    def is_loaded(self) -> bool:
        return self._mesh is not None or self._meshio_mesh is not None

    def list_fields(self) -> List[VTUField]:
        """List all fields in the dataset."""
        fields = []
        if self._mesh is not None:
            for name in self._mesh.point_data:
                arr = self._mesh.point_data[name]
                fields.append(VTUField(
                    name=name,
                    n_components=arr.shape[1] if arr.ndim > 1 else 1,
                    field_type="point",
                    dtype=str(arr.dtype),
                ))
            for name in self._mesh.cell_data:
                arr = self._mesh.cell_data[name]
                fields.append(VTUField(
                    name=name,
                    n_components=arr.shape[1] if arr.ndim > 1 else 1,
                    field_type="cell",
                    dtype=str(arr.dtype),
                ))
        elif self._meshio_mesh is not None:
            for name, arr in self._meshio_mesh.point_data.items():
                fields.append(VTUField(
                    name=name,
                    n_components=arr.shape[1] if arr.ndim > 1 else 1,
                    field_type="point",
                    dtype=str(arr.dtype),
                ))
        return fields

    def get_points(self) -> np.ndarray:
        """Get point coordinates as (N, 3) array."""
        if self._mesh is not None:
            return np.array(self._mesh.points)
        elif self._meshio_mesh is not None:
            return self._meshio_mesh.points
        return np.array([])

    def get_field(self, name: str) -> Optional[np.ndarray]:
        """Get a field by name."""
        if self._mesh is not None:
            if name in self._mesh.point_data:
                return np.array(self._mesh.point_data[name])
            if name in self._mesh.cell_data:
                return np.array(self._mesh.cell_data[name])
        elif self._meshio_mesh is not None:
            if name in self._meshio_mesh.point_data:
                return self._meshio_mesh.point_data[name]
        return None

    def extract_surface(
        self,
        boundary_name: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Extract surface data from the mesh.

        Parameters
        ----------
        boundary_name : str, optional
            Boundary name to extract (pyvista only).

        Returns empty DataFrame if extraction not possible.
        """
        if self._mesh is None:
            logger.warning("Surface extraction requires pyvista")
            return pd.DataFrame()

        try:
            surface = self._mesh.extract_surface()
            points = np.array(surface.points)
            data = {"x": points[:, 0], "y": points[:, 1], "z": points[:, 2]}

            for name in surface.point_data:
                arr = np.array(surface.point_data[name])
                if arr.ndim == 1:
                    data[name] = arr
                else:
                    for j in range(arr.shape[1]):
                        data[f"{name}_{j}"] = arr[:, j]

            return pd.DataFrame(data)
        except Exception as e:
            logger.warning(f"Surface extraction failed: {e}")
            return pd.DataFrame()

    def slice_at_x(
        self,
        x_station: float,
        normal: Tuple[float, float, float] = (1, 0, 0),
    ) -> pd.DataFrame:
        """
        Extract a slice at a given x-station (pyvista only).

        Parameters
        ----------
        x_station : float
            Streamwise location for slice.
        normal : tuple
            Slice normal direction.
        """
        if self._mesh is None:
            logger.warning("Slicing requires pyvista")
            return pd.DataFrame()

        try:
            sliced = self._mesh.slice(
                normal=normal,
                origin=(x_station, 0, 0),
            )
            points = np.array(sliced.points)
            data = {"x": points[:, 0], "y": points[:, 1], "z": points[:, 2]}

            for name in sliced.point_data:
                arr = np.array(sliced.point_data[name])
                if arr.ndim == 1:
                    data[name] = arr
                else:
                    for j in range(arr.shape[1]):
                        data[f"{name}_{j}"] = arr[:, j]

            df = pd.DataFrame(data)
            # Sort by y for profile extraction
            return df.sort_values("y").reset_index(drop=True)
        except Exception as e:
            logger.warning(f"Slice extraction failed: {e}")
            return pd.DataFrame()

    def to_dataframe(self) -> pd.DataFrame:
        """Convert entire mesh to DataFrame (point data only)."""
        points = self.get_points()
        if len(points) == 0:
            return pd.DataFrame()

        data = {"x": points[:, 0], "y": points[:, 1], "z": points[:, 2]}

        fields = self.list_fields()
        for f in fields:
            if f.field_type == "point":
                arr = self.get_field(f.name)
                if arr is not None:
                    if arr.ndim == 1:
                        data[f.name] = arr
                    else:
                        for j in range(arr.shape[1]):
                            data[f"{f.name}_{j}"] = arr[:, j]

        return pd.DataFrame(data)


def read_su2_vtk(filepath, fields: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Convenience function to read SU2 VTK output to DataFrame.

    Parameters
    ----------
    filepath : str or Path
        Path to VTK/VTU file.
    fields : list, optional
        Specific fields to extract. If None, extracts all.
    """
    reader = VTUReader(filepath)
    if fields:
        points = reader.get_points()
        data = {"x": points[:, 0], "y": points[:, 1], "z": points[:, 2]}
        for name in fields:
            arr = reader.get_field(name)
            if arr is not None:
                if arr.ndim == 1:
                    data[name] = arr
                else:
                    for j in range(arr.shape[1]):
                        data[f"{name}_{j}"] = arr[:, j]
        return pd.DataFrame(data)
    return reader.to_dataframe()
