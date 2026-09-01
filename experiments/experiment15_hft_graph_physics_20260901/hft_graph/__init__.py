"""HFT fully-coupled graph schema and conversion helpers."""

from .schema import (
    CapacitanceEdge,
    ConductorSegment,
    ConductanceEdge,
    GeometricEdge,
    HFTGraph,
    MagneticEdge,
    PotentialNode,
)
from .converters import MatrixBundle, graph_to_matrices, matrix_to_graph

__all__ = [
    "PotentialNode", "ConductorSegment", "MagneticEdge",
    "CapacitanceEdge", "ConductanceEdge", "GeometricEdge", "HFTGraph",
    "MatrixBundle", "graph_to_matrices", "matrix_to_graph",
]
