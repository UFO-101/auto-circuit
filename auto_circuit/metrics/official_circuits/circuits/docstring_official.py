# Based on:
# https://github.com/ArthurConmy/Automatic-Circuit-Discovery/blob/main/acdc/docstring/utils.py  # noqa: E501
# I added the token positions based on the findings in the paper
from typing import Dict, List, Set

from auto_circuit.types import Edge
from auto_circuit.utils.patchable_model import PatchableModel


def docstring_true_edges(
    model: PatchableModel, token_positions: bool = False, seq_start_idx: int = 0
) -> Set[Edge]:
    """
    the manual graph, from Stefan
    !!! Note: !!!
    The sequence positions assume prompts of length 40, as in docstring_prompts.json
    """
    assert model.cfg.model_name == "Attn_Only_4L512W_C4_Code"

    edges_present: Dict[str, List[int]] = {}
    edges_present["A0.5->A1.4.V"] = [13]
    edges_present["Resid Start->A0.5.V"] = [13, 15, 34]
    edges_present["Resid Start->A2.0.Q"] = [15]
    edges_present["A0.5->A2.0.Q"] = [15]
    # edges_present["Resid Start->A2.0.K"] = [14]  # Embedding at 14 doesn't vary
    edges_present["A0.5->A2.0.K"] = [13]  # Not entirely sure about this one
    edges_present["A1.4->A2.0.V"] = [14]
    edges_present["Resid Start->A1.2.K"] = [11, 13, 27, 34]
    edges_present["Resid Start->A1.4.V"] = [13, 34]
    edges_present["Resid Start->A1.2.Q"] = [27, 34]
    edges_present["A0.5->A1.2.Q"] = [34]
    edges_present["A0.5->A1.2.K"] = [13, 34]

    for layer_3_head in ["0", "6"]:
        edges_present[f"A3.{layer_3_head}->Resid End"] = [40]
        edges_present[f"A1.4->A3.{layer_3_head}.Q"] = [40]
        edges_present[f"Resid Start->A3.{layer_3_head}.V"] = [15]
        edges_present[f"A0.5->A3.{layer_3_head}.V"] = [15]
        edges_present[f"A2.0->A3.{layer_3_head}.K"] = [15]
        edges_present[f"A1.2->A3.{layer_3_head}.K"] = [34]

    true_edges: Set[Edge] = set()
    for edge in model.edges:
        if edge.name in edges_present.keys():
            if token_positions:
                for tok_pos in edges_present[edge.name]:
                    true_edges.add(Edge(edge.src, edge.dest, tok_pos - seq_start_idx))
            else:
                true_edges.add(edge)

    # reflects the value in the docstring appendix of the manual circuit as of 12th June
    assert len(true_edges) == 31

    return true_edges
