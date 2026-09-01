"""Typed, dependency-free data schema for fully coupled HFT graphs.

The graph has two node domains:
1. potential nodes: electrical potentials and reference conductors;
2. conductor segments: winding regions with two electrical boundaries.

Different edge families are intentionally not collapsed into one adjacency:
incidence defines galvanic continuity, magnetic edges define L/M coupling,
capacitance and conductance edges connect potentials, and geometric edges are
AI-only structural relations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class PotentialNode:
    id: str
    node_type: str = "winding"
    winding: Optional[str] = None
    terminal_type: Optional[str] = None
    is_external: bool = False
    is_internal: bool = True
    is_reference: bool = False
    boundary: str = "floating"
    xyz_m: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    size_m: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    material_region: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConductorSegment:
    id: str
    winding: str
    order: int
    start_node: str
    end_node: str
    turns: float = 1.0
    length_m: float = 0.0
    area_m2: float = 0.0
    conductor_type: str = "copper"
    material: str = "Cu"
    xyz_m: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    layer: Optional[int] = None
    physical_section: Optional[int] = None
    forced_boundary: bool = False
    resistance_ohm: Any = 0.0
    self_inductance_h: Any = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MagneticEdge:
    segment_i: str
    segment_j: str
    mutual_inductance_h: Any
    parameterization: str = "constant"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CapacitanceEdge:
    node_i: str
    node_j: str
    capacitance_f: Any
    subtype: str = "generic"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConductanceEdge:
    node_i: str
    node_j: str
    conductance_s: Any
    subtype: str = "dielectric"
    parameterization: str = "constant"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GeometricEdge:
    source_id: str
    target_id: str
    relation: str = "adjacent"
    distance_m: float = 0.0
    merge_allowed: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HFTGraph:
    potential_nodes: List[PotentialNode]
    conductor_segments: List[ConductorSegment]
    magnetic_edges: List[MagneticEdge] = field(default_factory=list)
    capacitance_edges: List[CapacitanceEdge] = field(default_factory=list)
    conductance_edges: List[ConductanceEdge] = field(default_factory=list)
    geometric_edges: List[GeometricEdge] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def node_map(self) -> Dict[str, PotentialNode]:
        return {x.id: x for x in self.potential_nodes}

    def segment_map(self) -> Dict[str, ConductorSegment]:
        return {x.id: x for x in self.conductor_segments}

    def external_node_ids(self) -> List[str]:
        return [x.id for x in self.potential_nodes if x.is_external and not x.is_reference]

    def reference_node_ids(self) -> List[str]:
        return [x.id for x in self.potential_nodes if x.is_reference]

    def to_dict(self) -> Dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)

    def to_pyg(self):
        """Optional HeteroData adapter; core physics never depends on PyG."""
        try:
            import torch
            from torch_geometric.data import HeteroData
        except ImportError as exc:
            raise RuntimeError("PyTorch Geometric is optional and not installed") from exc
        data = HeteroData()
        data["potential"].x = torch.tensor([
            [float(n.is_external), float(n.is_internal), float(n.is_reference), *n.xyz_m]
            for n in self.potential_nodes
        ], dtype=torch.float64)
        data["segment"].x = torch.tensor([
            [float(s.order), float(s.turns), float(s.length_m), float(s.area_m2), *s.xyz_m]
            for s in self.conductor_segments
        ], dtype=torch.float64)
        nidx = {n.id: i for i, n in enumerate(self.potential_nodes)}
        sidx = {s.id: i for i, s in enumerate(self.conductor_segments)}
        starts = [[sidx[s.id], nidx[s.start_node]] for s in self.conductor_segments]
        ends = [[sidx[s.id], nidx[s.end_node]] for s in self.conductor_segments]
        data["segment", "starts_at", "potential"].edge_index = torch.tensor(starts).T.long()
        data["segment", "ends_at", "potential"].edge_index = torch.tensor(ends).T.long()
        return data
