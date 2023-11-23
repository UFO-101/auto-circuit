# Based on:
# https://github.com/ArthurConmy/Automatic-Circuit-Discovery/blob/main/acdc/docstring/utils.py  # noqa: E501
from typing import List, Set
import torch as t

from auto_circuit.types import Edge

def docstring_true_edges(model: t.nn.Module) -> Set[Edge]:
    """the manual graph, from Stefan"""
    assert model.cfg.model_name == "Attn_Only_4L512W_C4_Code" # type: ignore

    edges_present: List[str] = []
    edges_present.append("A0.5->A1.4.V")
    edges_present.append("Resid Start->A0.5.V")
    edges_present.append("Resid Start->A2.0.Q")
    edges_present.append("A0.5->A2.0.Q")
    edges_present.append("Resid Start->A2.0.K")
    edges_present.append("A0.5->A2.0.K")
    edges_present.append("A1.4->A2.0.V")
    edges_present.append("Resid Start->A1.4.V")
    edges_present.append("Resid Start->A1.2.Q")
    edges_present.append("Resid Start->A1.2.K")
    edges_present.append("A0.5->A1.2.Q")
    edges_present.append("A0.5->A1.2.K")

    for layer_3_head in ["0", "6"]:
        edges_present.append(f"A3.{layer_3_head}->Resid End")
        edges_present.append(f"A1.4->A3.{layer_3_head}.Q")
        edges_present.append(f"Resid Start->A3.{layer_3_head}.V")
        edges_present.append(f"A0.5->A3.{layer_3_head}.V")
        edges_present.append(f"A2.0->A3.{layer_3_head}.K")
        edges_present.append(f"A1.2->A3.{layer_3_head}.K")
    
    # reflects the value in the docstring appendix of the manual circuit as of 12th June
    assert len(edges_present) == 24, len(edges_present)

    edges: Set[Edge] = model.edges  # type: ignore
    true_edges: Set[Edge] = set()
    for edge in edges:
        if edge.name in edges_present:
            true_edges.add(edge)
    return true_edges
