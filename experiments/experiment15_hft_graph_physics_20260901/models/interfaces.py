"""Interfaces only: EXP-015 deliberately does not implement/train EXP-016 AI."""
from dataclasses import dataclass
from typing import Any, Dict
import torch

@dataclass
class ReductionOutput:
    assignment: torch.Tensor
    reduced_graph: Any
    diagnostics: Dict[str,Any]

class GraphReducer(torch.nn.Module):
    def forward(self, graph, task_context) -> ReductionOutput:
        raise NotImplementedError("implemented in EXP-016 after the physics baseline is frozen")
