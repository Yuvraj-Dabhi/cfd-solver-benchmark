#!/usr/bin/env python3
"""
SU2 Mesh → Graph Converter for GNN-Based Flow Prediction
==========================================================
Parses SU2 unstructured mesh files into PyTorch Geometric ``Data``
objects, preserving the full mesh topology as a graph suitable for
MeshGraphNet and Physics-Guided GNN models.

Pipeline:
  1. Parse ``.su2`` file → node coordinates + element connectivity
  2. Build graph edges from element adjacency (undirected)
  3. Compute node features: position, wall distance, boundary flags
  4. Compute edge features: relative displacement, Euclidean distance
  5. Optionally augment with SU2 solution data (Cp, Cf, ν_t)

References:
  - Pfaff et al. (2021), "Learning Mesh-Based Simulation with Graph
    Networks", ICML (MeshGraphNet — Google DeepMind)
  - Bonnet et al. (2022), "An Airfoil Design Methodology Using Graph
    Neural Networks", AIAA Aviation
"""

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)
PROJECT = Path(__file__).resolve().parent.parent.parent

# ---------------------------------------------------------------------------
# Optional dependency guard (consistent with project pattern)
# ---------------------------------------------------------------------------
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import torch_geometric
    from torch_geometric.data import Data, Dataset
    HAS_TORCH_GEOMETRIC = True
except ImportError:
    HAS_TORCH_GEOMETRIC = False


# ============================================================================
# SU2 Element Types (VTK numbering used by SU2)
# ============================================================================
SU2_ELEMENT_TYPES = {
    3: ("Line", 2),
    5: ("Triangle", 3),
    9: ("Quadrilateral", 4),
    10: ("Tetrahedron", 4),
    12: ("Hexahedron", 8),
    13: ("Prism", 6),
    14: ("Pyramid", 5),
}


# ============================================================================
# Data Structures
# ============================================================================
@dataclass
class SU2MeshData:
    """Raw data parsed from an SU2 mesh file."""
    coords: np.ndarray           # (N, ndim) node coordinates
    elements: List[Tuple[int, ...]]  # list of (type, n0, n1, ...) tuples
    n_dim: int = 2
    n_points: int = 0
    n_elements: int = 0
    boundary_markers: Dict[str, List[Tuple[int, ...]]] = field(
        default_factory=dict
    )
    boundary_node_ids: Set[int] = field(default_factory=set)
    wall_boundary_names: Tuple[str, ...] = (
        "wall", "airfoil", "hump", "bump", "body", "surface",
    )


@dataclass
class GraphData:
    """Graph representation of a mesh (numpy-based, framework-agnostic)."""
    node_features: np.ndarray     # (N, F_node)
    edge_index: np.ndarray        # (2, E) — COO format
    edge_features: np.ndarray     # (E, F_edge)
    node_coords: np.ndarray       # (N, ndim) — for visualization
    wall_distance: np.ndarray     # (N,)
    boundary_mask: np.ndarray     # (N,) bool
    n_nodes: int = 0
    n_edges: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# SU2 Mesh Parser
# ============================================================================
def parse_su2_mesh(mesh_path: Union[str, Path]) -> SU2MeshData:
    """
    Parse an SU2 ``.su2`` mesh file.

    Reads the ``NDIME``, ``NPOIN``, ``NELEM``, and ``MARKER_*`` sections.

    Parameters
    ----------
    mesh_path : str or Path
        Path to the ``.su2`` mesh file.

    Returns
    -------
    SU2MeshData
        Parsed mesh data with coordinates, elements, and boundaries.
    """
    mesh_path = Path(mesh_path)
    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh file not found: {mesh_path}")

    mesh = SU2MeshData(coords=np.array([]), elements=[])
    coords_list = []
    current_section = None
    marker_tag = None
    marker_elems = []
    points_remaining = 0
    elems_remaining = 0
    marker_elems_remaining = 0

    with open(mesh_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('%'):
                continue

            # --- Section headers ---
            if line.startswith("NDIME="):
                mesh.n_dim = int(line.split("=")[1].strip())
                continue
            if line.startswith("NPOIN="):
                parts = line.split("=")[1].strip().split()
                mesh.n_points = int(parts[0])
                points_remaining = mesh.n_points
                current_section = "POINTS"
                continue
            if line.startswith("NELEM="):
                mesh.n_elements = int(line.split("=")[1].strip())
                elems_remaining = mesh.n_elements
                current_section = "ELEMENTS"
                continue
            if line.startswith("MARKER_TAG="):
                # Save previous marker
                if marker_tag is not None and marker_elems:
                    mesh.boundary_markers[marker_tag] = marker_elems
                marker_tag = line.split("=")[1].strip().lower()
                marker_elems = []
                current_section = "MARKER"
                continue
            if line.startswith("MARKER_ELEMS="):
                marker_elems_remaining = int(line.split("=")[1].strip())
                continue
            if line.startswith("NMARK="):
                # Save last marker
                if marker_tag is not None and marker_elems:
                    mesh.boundary_markers[marker_tag] = marker_elems
                    marker_tag = None
                    marker_elems = []
                continue

            # --- Section data ---
            if current_section == "POINTS" and points_remaining > 0:
                parts = line.split()
                coord = [float(parts[i]) for i in range(mesh.n_dim)]
                coords_list.append(coord)
                points_remaining -= 1
                if points_remaining == 0:
                    current_section = None
                continue

            if current_section == "ELEMENTS" and elems_remaining > 0:
                parts = [int(p) for p in line.split()]
                elem_type = parts[0]
                if elem_type in SU2_ELEMENT_TYPES:
                    _, n_nodes_per_elem = SU2_ELEMENT_TYPES[elem_type]
                    node_ids = tuple(parts[1:1 + n_nodes_per_elem])
                    mesh.elements.append((elem_type,) + node_ids)
                elems_remaining -= 1
                if elems_remaining == 0:
                    current_section = None
                continue

            if current_section == "MARKER" and marker_elems_remaining > 0:
                parts = [int(p) for p in line.split()]
                elem_type = parts[0]
                if elem_type in SU2_ELEMENT_TYPES:
                    _, n_nodes_per_elem = SU2_ELEMENT_TYPES[elem_type]
                    node_ids = tuple(parts[1:1 + n_nodes_per_elem])
                    marker_elems.append((elem_type,) + node_ids)
                    mesh.boundary_node_ids.update(node_ids)
                marker_elems_remaining -= 1
                continue

    # Save final marker
    if marker_tag is not None and marker_elems:
        mesh.boundary_markers[marker_tag] = marker_elems

    mesh.coords = np.array(coords_list, dtype=np.float64)

    # Identify wall boundary nodes
    wall_nodes = set()
    for bname, belems in mesh.boundary_markers.items():
        if any(wn in bname for wn in mesh.wall_boundary_names):
            for elem in belems:
                wall_nodes.update(elem[1:])  # Skip element type
    mesh.boundary_node_ids = wall_nodes if wall_nodes else mesh.boundary_node_ids

    logger.info(
        "Parsed SU2 mesh: %d nodes, %d elements, %d dim, %d boundaries",
        mesh.n_points, len(mesh.elements), mesh.n_dim,
        len(mesh.boundary_markers),
    )
    return mesh


# ============================================================================
# Graph Builder
# ============================================================================
def _compute_wall_distance(
    coords: np.ndarray,
    wall_node_ids: Set[int],
) -> np.ndarray:
    """Compute minimum distance from each node to the nearest wall node."""
    n = len(coords)
    if not wall_node_ids:
        # No wall identified — use y-coordinate as proxy
        return np.abs(coords[:, 1]) if coords.shape[1] >= 2 else np.ones(n)

    wall_coords = coords[list(wall_node_ids)]
    dists = np.full(n, np.inf)
    # Chunk to avoid memory blow-up for large meshes
    chunk_size = 5000
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        diff = coords[start:end, np.newaxis, :] - wall_coords[np.newaxis, :, :]
        dists[start:end] = np.sqrt((diff ** 2).sum(axis=-1)).min(axis=1)
    return dists


def build_graph_from_mesh(
    mesh: SU2MeshData,
    include_position: bool = True,
) -> GraphData:
    """
    Convert parsed SU2 mesh into a graph with node/edge features.

    Node features (per node):
      - Position ``(x, y[, z])`` (optional, controlled by include_position)
      - Wall distance (scalar)
      - Boundary flag (0 or 1)

    Edge features (per directed edge):
      - Relative displacement ``Δx = x_j - x_i`` (ndim components)
      - Euclidean distance ``||Δx||``

    Parameters
    ----------
    mesh : SU2MeshData
        Parsed mesh from ``parse_su2_mesh()``.
    include_position : bool
        Whether to include raw coordinates as node features.

    Returns
    -------
    GraphData
    """
    coords = mesh.coords
    n = len(coords)
    ndim = mesh.n_dim

    # --- Build edge set from element connectivity ---
    edge_set: Set[Tuple[int, int]] = set()
    for elem in mesh.elements:
        elem_type = elem[0]
        nodes = elem[1:]
        n_nodes_elem = len(nodes)
        # Connect all node pairs within the element
        for i in range(n_nodes_elem):
            for j in range(i + 1, n_nodes_elem):
                ni, nj = nodes[i], nodes[j]
                if ni < n and nj < n:  # Safety check
                    edge_set.add((ni, nj))
                    edge_set.add((nj, ni))  # Undirected

    if not edge_set:
        # Fallback: create edges from boundary elements
        for bname, belems in mesh.boundary_markers.items():
            for elem in belems:
                nodes = elem[1:]
                for i in range(len(nodes)):
                    for j in range(i + 1, len(nodes)):
                        ni, nj = nodes[i], nodes[j]
                        if ni < n and nj < n:
                            edge_set.add((ni, nj))
                            edge_set.add((nj, ni))

    edge_list = sorted(edge_set)
    edge_index = np.array(edge_list, dtype=np.int64).T  # (2, E)
    n_edges = edge_index.shape[1]

    # --- Edge features: relative displacement + distance ---
    src = edge_index[0]
    dst = edge_index[1]
    dx = coords[dst] - coords[src]  # (E, ndim)
    dist = np.linalg.norm(dx, axis=1, keepdims=True)  # (E, 1)
    dist_safe = np.maximum(dist, 1e-15)
    direction = dx / dist_safe  # (E, ndim) normalized
    edge_features = np.hstack([dx, dist, direction])  # (E, 2*ndim + 1)

    # --- Node features ---
    wall_dist = _compute_wall_distance(coords, mesh.boundary_node_ids)
    boundary_mask = np.zeros(n, dtype=np.float64)
    for nid in mesh.boundary_node_ids:
        if nid < n:
            boundary_mask[nid] = 1.0

    feature_parts = []
    if include_position:
        feature_parts.append(coords)              # (N, ndim)
    feature_parts.append(wall_dist.reshape(-1, 1))  # (N, 1)
    feature_parts.append(boundary_mask.reshape(-1, 1))  # (N, 1)
    node_features = np.hstack(feature_parts)  # (N, F_node)

    graph = GraphData(
        node_features=node_features,
        edge_index=edge_index,
        edge_features=edge_features,
        node_coords=coords,
        wall_distance=wall_dist,
        boundary_mask=boundary_mask.astype(bool),
        n_nodes=n,
        n_edges=n_edges,
        metadata={
            "mesh_file": "",
            "n_dim": ndim,
            "n_boundary_markers": len(mesh.boundary_markers),
            "boundary_names": list(mesh.boundary_markers.keys()),
        },
    )
    return graph


# ============================================================================
# High-Level Convenience Functions
# ============================================================================
def load_su2_mesh_as_graph(
    mesh_path: Union[str, Path],
    include_position: bool = True,
) -> GraphData:
    """
    One-call convenience: parse SU2 mesh and build graph.

    Parameters
    ----------
    mesh_path : str or Path
        Path to ``.su2`` mesh file.
    include_position : bool
        Include raw coordinates as node features.

    Returns
    -------
    GraphData
    """
    mesh = parse_su2_mesh(mesh_path)
    graph = build_graph_from_mesh(mesh, include_position=include_position)
    graph.metadata["mesh_file"] = str(mesh_path)
    return graph


def augment_graph_with_solution(
    graph: GraphData,
    solution_fields: Dict[str, np.ndarray],
) -> GraphData:
    """
    Add solution fields (Cp, Cf, ν_t, velocity, etc.) as extra node features.

    Parameters
    ----------
    graph : GraphData
        Graph from ``build_graph_from_mesh``.
    solution_fields : dict
        Mapping of field name → array of shape ``(N,)`` or ``(N, d)``.
        Common fields: ``Cp``, ``Cf``, ``nu_t``, ``velocity``.

    Returns
    -------
    GraphData with updated node_features.
    """
    extra = []
    for name, arr in solution_fields.items():
        arr = np.asarray(arr, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if len(arr) != graph.n_nodes:
            logger.warning(
                "Field '%s' has %d values but mesh has %d nodes — skipping",
                name, len(arr), graph.n_nodes,
            )
            continue
        extra.append(arr)
        logger.debug("Added solution field '%s' (%d cols)", name, arr.shape[1])

    if extra:
        graph.node_features = np.hstack(
            [graph.node_features] + extra
        )

    return graph


def graph_to_pyg(
    graph: GraphData,
    target: Optional[np.ndarray] = None,
) -> Any:
    """
    Convert ``GraphData`` to a PyTorch Geometric ``Data`` object.

    Parameters
    ----------
    graph : GraphData
        Graph from the numpy-based pipeline.
    target : ndarray, optional
        Target values ``(N,)`` or ``(N, d)`` (e.g., β field).

    Returns
    -------
    torch_geometric.data.Data

    Raises
    ------
    ImportError
        If PyTorch Geometric is not installed.
    """
    if not HAS_TORCH or not HAS_TORCH_GEOMETRIC:
        raise ImportError(
            "PyTorch and torch-geometric are required. "
            "Install with: pip install torch torch-geometric"
        )

    data = Data(
        x=torch.tensor(graph.node_features, dtype=torch.float32),
        edge_index=torch.tensor(graph.edge_index, dtype=torch.long),
        edge_attr=torch.tensor(graph.edge_features, dtype=torch.float32),
        pos=torch.tensor(graph.node_coords, dtype=torch.float32),
    )
    data.wall_distance = torch.tensor(
        graph.wall_distance, dtype=torch.float32
    )
    data.boundary_mask = torch.tensor(
        graph.boundary_mask, dtype=torch.bool
    )

    if target is not None:
        target = np.asarray(target, dtype=np.float32)
        if target.ndim == 1:
            target = target.reshape(-1, 1)
        data.y = torch.tensor(target, dtype=torch.float32)

    return data


# ============================================================================
# Synthetic Mesh Generator (for testing)
# ============================================================================
def generate_synthetic_mesh(
    nx: int = 20,
    ny: int = 10,
    lx: float = 2.0,
    ly: float = 0.5,
    seed: int = 42,
) -> SU2MeshData:
    """
    Generate a synthetic 2D triangular mesh for testing.

    Creates a rectangular domain [0, lx] × [0, ly] with a structured
    quad grid that is split into triangles, then mildly perturbed.

    Parameters
    ----------
    nx, ny : int
        Grid points in x and y directions.
    lx, ly : float
        Domain size.
    seed : int
        Random seed for perturbation.

    Returns
    -------
    SU2MeshData
    """
    rng = np.random.default_rng(seed)
    n_points = nx * ny

    # Generate structured grid
    x = np.linspace(0, lx, nx)
    y = np.linspace(0, ly, ny)
    xx, yy = np.meshgrid(x, y, indexing='ij')
    coords = np.column_stack([xx.ravel(), yy.ravel()])

    # Mild perturbation (keep boundaries fixed)
    for i in range(n_points):
        ix = i // ny
        iy = i % ny
        if 0 < ix < nx - 1 and 0 < iy < ny - 1:
            coords[i, 0] += rng.uniform(-0.3, 0.3) * (lx / nx)
            coords[i, 1] += rng.uniform(-0.3, 0.3) * (ly / ny)

    # Triangulate: split each quad into 2 triangles
    elements = []
    for ix in range(nx - 1):
        for iy in range(ny - 1):
            n0 = ix * ny + iy
            n1 = (ix + 1) * ny + iy
            n2 = (ix + 1) * ny + (iy + 1)
            n3 = ix * ny + (iy + 1)
            elements.append((5, n0, n1, n3))  # 5 = Triangle
            elements.append((5, n1, n2, n3))

    # Boundary markers
    bottom = []  # y = 0 (wall)
    top = []     # y = ly (farfield)
    for ix in range(nx - 1):
        n0_b = ix * ny
        n1_b = (ix + 1) * ny
        bottom.append((3, n0_b, n1_b))  # 3 = Line
        n0_t = ix * ny + (ny - 1)
        n1_t = (ix + 1) * ny + (ny - 1)
        top.append((3, n0_t, n1_t))

    wall_nodes = {ix * ny for ix in range(nx)}  # y=0 nodes

    mesh = SU2MeshData(
        coords=coords,
        elements=elements,
        n_dim=2,
        n_points=n_points,
        n_elements=len(elements),
        boundary_markers={"wall": bottom, "farfield": top},
        boundary_node_ids=wall_nodes,
    )
    return mesh


def generate_synthetic_graph(
    nx: int = 20,
    ny: int = 10,
    **kwargs,
) -> GraphData:
    """Generate a synthetic mesh graph for testing."""
    mesh = generate_synthetic_mesh(nx=nx, ny=ny, **kwargs)
    return build_graph_from_mesh(mesh)


# ============================================================================
# Demo / Main
# ============================================================================
def _demo():
    """Demonstrate SU2 mesh → graph conversion."""
    print("=" * 65)
    print("  SU2 Mesh → Graph Converter Demo")
    print("=" * 65)

    # 1. Synthetic mesh
    print("\n  Generating synthetic 20×10 triangular mesh...")
    mesh = generate_synthetic_mesh(nx=20, ny=10)
    print(f"  Nodes: {mesh.n_points}")
    print(f"  Elements: {len(mesh.elements)}")
    print(f"  Dim: {mesh.n_dim}")
    print(f"  Boundaries: {list(mesh.boundary_markers.keys())}")
    print(f"  Wall nodes: {len(mesh.boundary_node_ids)}")

    # 2. Build graph
    print("\n  Building graph...")
    graph = build_graph_from_mesh(mesh)
    print(f"  Graph nodes: {graph.n_nodes}")
    print(f"  Graph edges: {graph.n_edges}")
    print(f"  Node features: {graph.node_features.shape}")
    print(f"  Edge features: {graph.edge_features.shape}")
    print(f"  Wall distance range: [{graph.wall_distance.min():.4f}, "
          f"{graph.wall_distance.max():.4f}]")

    # 3. Augment with synthetic solution
    print("\n  Augmenting with synthetic solution fields...")
    rng = np.random.default_rng(42)
    solution = {
        "Cp": -0.6 * np.exp(-((graph.node_coords[:, 0] - 1.0) ** 2) / 0.1),
        "nu_t": rng.lognormal(0, 1, graph.n_nodes) * 1e-5,
    }
    graph = augment_graph_with_solution(graph, solution)
    print(f"  Updated node features: {graph.node_features.shape}")

    # 4. Convert to PyG (if available)
    if HAS_TORCH_GEOMETRIC:
        print("\n  Converting to PyTorch Geometric Data...")
        beta_target = np.ones(graph.n_nodes)
        beta_target[50:100] = 1.3  # Synthetic separation correction
        pyg_data = graph_to_pyg(graph, target=beta_target)
        print(f"  PyG Data: {pyg_data}")
        print(f"    x: {pyg_data.x.shape}")
        print(f"    edge_index: {pyg_data.edge_index.shape}")
        print(f"    edge_attr: {pyg_data.edge_attr.shape}")
        print(f"    y: {pyg_data.y.shape}")
    else:
        print("\n  [SKIP] torch-geometric not installed — PyG conversion skipped")

    # 5. Try loading a real SU2 mesh if available
    sample_meshes = list((PROJECT / "runs").rglob("*.su2"))
    if sample_meshes:
        real_mesh_path = sample_meshes[0]
        print(f"\n  Loading real SU2 mesh: {real_mesh_path.name}...")
        try:
            real_graph = load_su2_mesh_as_graph(real_mesh_path)
            print(f"  Real graph: {real_graph.n_nodes} nodes, "
                  f"{real_graph.n_edges} edges")
        except Exception as e:
            print(f"  [WARN] Could not load real mesh: {e}")

    print(f"\n{'=' * 65}")
    print("  Demo complete!")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _demo()
