# Based on:
# https://github.com/ArthurConmy/Automatic-Circuit-Discovery/blob/main/acdc/docstring/utils.py  # noqa: E501
# I added the token positions based on the findings in the paper
from typing import Dict, List, Set

from auto_circuit.types import Edge
from auto_circuit.utils.patchable_model import PatchableModel


def docstring_true_edges(
    model: PatchableModel,
    token_positions: bool = False,
    word_idxs: Dict[str, int] = {},
    seq_start_idx: int = 0,
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
                assert edge.seq_idx is not None
                if (edge.seq_idx + seq_start_idx) in edges_present[edge.name]:
                    true_edges.add(edge)
            else:
                true_edges.add(edge)

    # reflects the value in the docstring appendix of the manual circuit as of 12th June
    assert len(true_edges) == 31 if token_positions else 24

    return true_edges


def docstring_node_based_official_edges(
    model: PatchableModel,
    token_positions: bool = False,
    word_idxs: Dict[str, int] = {},
    seq_start_idx: int = 0,
) -> Set[Edge]:
    """
    The heads not ablated in the final check in the Docstring blog post:
    https://www.alignmentforum.org/posts/u6KXXmKFbXfWzoAXn#Putting_it_all_together
    """
    heads_in_circuit = {
        "A1.4",
        "A2.0",
        "A3.0",
        "A3.6",
        "A0.5",
        "A1.2",
        "A0.2",
        "A0.4",
        # Extras for improved performance
        "A1.0",
        "A0.1",
        "A2.3",
    }
    heads_not_in_circ = set()
    for src in model.srcs:
        if src.head_idx is not None and src.name not in heads_in_circuit:
            heads_not_in_circ.add(src.name)

    official_edges: Set[Edge] = set()
    for edge in model.edges:
        src_head_str = edge.src.name
        if edge.dest.name[-1] in ["Q", "K", "V"]:
            dest_head_str = edge.dest.name[:-2]
        else:
            dest_head_str = edge.dest.name
        if src_head_str in heads_not_in_circ or dest_head_str in heads_not_in_circ:
            continue
        official_edges.add(edge)
    return official_edges
