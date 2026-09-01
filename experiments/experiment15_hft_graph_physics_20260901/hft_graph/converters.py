"""Bidirectional graph/matrix conversion with explicit sign conventions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from .schema import (CapacitanceEdge, ConductorSegment, ConductanceEdge,
                     HFTGraph, MagneticEdge, PotentialNode)
from .validation import validate_graph


@dataclass
class MatrixBundle:
    A: torch.Tensor
    R: torch.Tensor
    L: torch.Tensor
    C: torch.Tensor
    G: torch.Tensor
    node_order: List[str]
    segment_order: List[str]
    external_indices: torch.Tensor
    metadata: Dict

    def to(self, device: torch.device | str) -> "MatrixBundle":
        return MatrixBundle(self.A.to(device), self.R.to(device), self.L.to(device),
                            self.C.to(device), self.G.to(device), self.node_order,
                            self.segment_order, self.external_indices.to(device), self.metadata)


def _scalar(value, *, dtype, device):
    return torch.as_tensor(value, dtype=dtype, device=device)


def _stamp_pair(matrix, i: int, j: Optional[int], value):
    matrix[i, i] = matrix[i, i] + value
    if j is not None:
        matrix[j, j] = matrix[j, j] + value
        matrix[i, j] = matrix[i, j] - value
        matrix[j, i] = matrix[j, i] - value


def graph_to_matrices(graph: HFTGraph, *, dtype=torch.float64, device="cpu") -> MatrixBundle:
    validate_graph(graph)
    retained = [n for n in graph.potential_nodes if not n.is_reference]
    node_order = [n.id for n in retained]
    segment_order = [s.id for s in sorted(graph.conductor_segments,
                                           key=lambda x: (x.winding, x.order, x.id))]
    ni = {x: k for k, x in enumerate(node_order)}
    si = {x: k for k, x in enumerate(segment_order)}
    n, b = len(node_order), len(segment_order)
    A = torch.zeros((n, b), dtype=dtype, device=device)
    R = torch.zeros((b, b), dtype=dtype, device=device)
    L = torch.zeros((b, b), dtype=dtype, device=device)
    C = torch.zeros((n, n), dtype=dtype, device=device)
    G = torch.zeros((n, n), dtype=dtype, device=device)
    smap = graph.segment_map()
    for sid in segment_order:
        k, s = si[sid], smap[sid]
        if s.start_node in ni:
            A[ni[s.start_node], k] = 1.0
        if s.end_node in ni:
            A[ni[s.end_node], k] = -1.0
        R[k, k] = _scalar(s.resistance_ohm, dtype=dtype, device=device)
        L[k, k] = _scalar(s.self_inductance_h, dtype=dtype, device=device)
    for e in graph.magnetic_edges:
        i, j = si[e.segment_i], si[e.segment_j]
        value = _scalar(e.mutual_inductance_h, dtype=dtype, device=device)
        L[i, j] = value
        L[j, i] = value
    refs = set(graph.reference_node_ids())
    for e in graph.capacitance_edges:
        i = ni.get(e.node_i); j = ni.get(e.node_j)
        if i is None and e.node_i not in refs:
            raise ValueError(f"unknown capacitance endpoint {e.node_i}")
        if j is None and e.node_j not in refs:
            raise ValueError(f"unknown capacitance endpoint {e.node_j}")
        if i is None and j is None:
            continue
        value = _scalar(e.capacitance_f, dtype=dtype, device=device)
        _stamp_pair(C, j if i is None else i, None if (i is None or j is None) else j, value)
    for e in graph.conductance_edges:
        i = ni.get(e.node_i); j = ni.get(e.node_j)
        if i is None and e.node_i not in refs:
            raise ValueError(f"unknown conductance endpoint {e.node_i}")
        if j is None and e.node_j not in refs:
            raise ValueError(f"unknown conductance endpoint {e.node_j}")
        if i is None and j is None:
            continue
        value = _scalar(e.conductance_s, dtype=dtype, device=device)
        _stamp_pair(G, j if i is None else i, None if (i is None or j is None) else j, value)
    external = torch.tensor([ni[x] for x in graph.external_node_ids()], dtype=torch.long, device=device)
    return MatrixBundle(A, R, L, C, G, node_order, segment_order, external,
                        {"source": graph.metadata.get("source"),
                         "capacitance_convention": "reduced_nodal_Maxwell",
                         "reference_nodes": graph.reference_node_ids()})


def matrix_to_graph(bundle: MatrixBundle, template: Optional[HFTGraph] = None,
                    tol: float = 1e-18) -> HFTGraph:
    """Convert matrices back to a graph.

    Exact physical metadata and the identity of multiple reference conductors
    are not identifiable from reduced matrices alone. Passing ``template``
    provides that missing information and enables an exact round trip.
    Otherwise a canonical single-reference graph is constructed.
    """
    if template is not None:
        import copy
        graph = copy.deepcopy(template)
        # Values are refreshed from the matrices while physical metadata stays.
        segs = {s.id: s for s in graph.conductor_segments}
        for k, sid in enumerate(bundle.segment_order):
            segs[sid].resistance_ohm = bundle.R[k, k]
            segs[sid].self_inductance_h = bundle.L[k, k]
        graph.magnetic_edges = []
        for i in range(len(bundle.segment_order)):
            for j in range(i + 1, len(bundle.segment_order)):
                if abs(float(bundle.L[i, j].detach().cpu())) > tol:
                    graph.magnetic_edges.append(MagneticEdge(bundle.segment_order[i],
                                                              bundle.segment_order[j], bundle.L[i, j]))
        return graph
    ref = PotentialNode("reference_0", node_type="reference", is_reference=True,
                        is_internal=False, boundary="reference")
    nodes = [PotentialNode(x, is_external=(i in bundle.external_indices.detach().cpu().tolist()),
                           is_internal=(i not in bundle.external_indices.detach().cpu().tolist()))
             for i, x in enumerate(bundle.node_order)] + [ref]
    segs = []
    for k, sid in enumerate(bundle.segment_order):
        plus = torch.where(bundle.A[:, k] > 0.5)[0]
        minus = torch.where(bundle.A[:, k] < -0.5)[0]
        start = bundle.node_order[int(plus[0])] if len(plus) else ref.id
        end = bundle.node_order[int(minus[0])] if len(minus) else ref.id
        segs.append(ConductorSegment(sid, "unknown", k, start, end,
                                     resistance_ohm=bundle.R[k, k],
                                     self_inductance_h=bundle.L[k, k]))
    mags = [MagneticEdge(bundle.segment_order[i], bundle.segment_order[j], bundle.L[i, j])
            for i in range(len(segs)) for j in range(i + 1, len(segs))
            if abs(float(bundle.L[i, j].detach().cpu())) > tol]
    caps = []
    for i in range(len(nodes) - 1):
        for j in range(i + 1, len(nodes) - 1):
            if float(bundle.C[i, j]) < -tol:
                caps.append(CapacitanceEdge(nodes[i].id, nodes[j].id, -bundle.C[i, j]))
        ground = bundle.C[i, i] + bundle.C[i].sum() - bundle.C[i, i]
        if float(ground) > tol:
            caps.append(CapacitanceEdge(nodes[i].id, ref.id, ground, "ground"))
    cond = []
    for i in range(len(nodes) - 1):
        for j in range(i + 1, len(nodes) - 1):
            if float(bundle.G[i, j]) < -tol:
                cond.append(ConductanceEdge(nodes[i].id, nodes[j].id, -bundle.G[i, j]))
        ground = bundle.G[i, i] + bundle.G[i].sum() - bundle.G[i, i]
        if float(ground) > tol:
            cond.append(ConductanceEdge(nodes[i].id, ref.id, ground))
    return HFTGraph(nodes, segs, mags, caps, cond,
                    metadata={"source": "canonical_matrix_reconstruction"})
