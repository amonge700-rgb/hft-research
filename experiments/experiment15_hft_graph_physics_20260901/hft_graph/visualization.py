"""Small dependency-light graph visualisation helpers."""

from __future__ import annotations

import json
from pathlib import Path

from .schema import HFTGraph


def save_graph_json(graph: HFTGraph, path) -> None:
    Path(path).write_text(json.dumps(graph.to_dict(), ensure_ascii=False, indent=2, default=str),
                          encoding="utf-8")


def save_graph_svg(graph: HFTGraph, path) -> None:
    """Write a compact schematic SVG without importing NetworkX."""
    nodes = [x for x in graph.potential_nodes if not x.is_reference]
    pos = {n.id: (80 + 90*i, 90 if n.winding == "primary" else 310)
           for i, n in enumerate(nodes)}
    width = max(900, 120 + 90*len(nodes))
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="430">',
             '<rect width="100%" height="100%" fill="white"/>']
    for e in graph.capacitance_edges:
        if e.node_i in pos and e.node_j in pos:
            x1,y1=pos[e.node_i]; x2,y2=pos[e.node_j]
            lines.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#d95f02" stroke-width="1"/>')
    for s in graph.conductor_segments:
        if s.start_node in pos and s.end_node in pos:
            x1,y1=pos[s.start_node]; x2,y2=pos[s.end_node]
            lines.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#1b6ca8" stroke-width="4"/>')
    for n in nodes:
        x,y=pos[n.id]
        lines.append(f'<circle cx="{x}" cy="{y}" r="8" fill="#333"/>')
        lines.append(f'<text x="{x-18}" y="{y-14}" font-size="12">{n.id}</text>')
    lines.append('</svg>')
    Path(path).write_text("\n".join(lines), encoding="utf-8")
