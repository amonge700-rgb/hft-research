"""Schema-level structural validation."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict

from .schema import HFTGraph


class GraphValidationError(ValueError):
    pass


def validate_graph(graph: HFTGraph) -> Dict[str, int]:
    node_ids = [x.id for x in graph.potential_nodes]
    seg_ids = [x.id for x in graph.conductor_segments]
    if len(node_ids) != len(set(node_ids)):
        raise GraphValidationError("duplicate potential-node id")
    if len(seg_ids) != len(set(seg_ids)):
        raise GraphValidationError("duplicate conductor-segment id")
    nodes, segs = set(node_ids), set(seg_ids)
    refs = {x.id for x in graph.potential_nodes if x.is_reference}
    if not refs:
        raise GraphValidationError("at least one mathematical/physical reference node is required")
    for s in graph.conductor_segments:
        if s.start_node not in nodes or s.end_node not in nodes:
            raise GraphValidationError(f"segment {s.id} has a missing endpoint")
        if s.start_node == s.end_node:
            raise GraphValidationError(f"segment {s.id} is a self loop")
    for e in graph.magnetic_edges:
        if e.segment_i not in segs or e.segment_j not in segs:
            raise GraphValidationError("magnetic edge references a missing segment")
    for family in (graph.capacitance_edges, graph.conductance_edges):
        for e in family:
            if e.node_i not in nodes or e.node_j not in nodes:
                raise GraphValidationError("potential-coupling edge references a missing node")
            if e.node_i == e.node_j:
                raise GraphValidationError("potential-coupling self loop is invalid")
    by_winding = defaultdict(list)
    for s in graph.conductor_segments:
        by_winding[s.winding].append(s.order)
    for winding, orders in by_winding.items():
        if sorted(orders) != list(range(len(orders))):
            raise GraphValidationError(f"non-contiguous order indices in winding {winding}")
    duplicate_caps = Counter((min(e.node_i, e.node_j), max(e.node_i, e.node_j), e.subtype)
                             for e in graph.capacitance_edges)
    if any(v > 1 for v in duplicate_caps.values()):
        raise GraphValidationError("duplicate capacitance edge; aggregate it explicitly")
    return {
        "potential_nodes": len(node_ids), "segments": len(seg_ids),
        "magnetic_edges": len(graph.magnetic_edges),
        "capacitance_edges": len(graph.capacitance_edges),
        "conductance_edges": len(graph.conductance_edges),
        "references": len(refs),
    }
