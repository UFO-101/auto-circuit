# Based on:
# https://github.com/ArthurConmy/Automatic-Circuit-Discovery/blob/main/acdc/docstring/utils.py  # noqa: E501
# I added the token positions based on my reading of the paper
from typing import Dict, List, Optional, Set, Tuple

from auto_circuit.types import Edge
from auto_circuit.utils.patchable_model import PatchableModel


def docstring_true_edges(
    model: PatchableModel,
    token_positions: bool = False,
    word_idxs: Dict[str, int] = {},
    seq_start_idx: int = 0,
) -> Set[Edge]:
    """
    The manual graph, from Stefan
    !!! Note: !!!
    The sequence positions assume prompts of length 40, as in docstring_prompts.json
    """
    assert model.cfg.model_name == "Attn_Only_4L512W_C4_Code"
    assert model.separate_qkv

    A_def_tok_idx = word_idxs.get("A_def", 0)
    B_def_tok_idx = word_idxs.get("B_def", 0)
    B_comma_tok_idx = word_idxs.get(",_B", 0)
    C_def_tok_idx = word_idxs.get("C_def", 0)
    A_doc_tok_idx = word_idxs.get("A_doc", 0)
    # param_2_tok_idx = word_idxs.get("param_2", 0) A0.2 not included for some reason
    B_doc_tok_idx = word_idxs.get("B_doc", 0)
    param_3_tok_idx = word_idxs.get("param_3", 0)

    if token_positions:
        assert param_3_tok_idx > 0, "Must provide word_idxs if token_positions is True"

    edges_present: Dict[str, List[int]] = {}
    edges_present["A0.5->A1.4.V"] = [B_def_tok_idx, B_doc_tok_idx]
    edges_present["Resid Start->A0.5.V"] = [B_def_tok_idx, C_def_tok_idx, B_doc_tok_idx]
    edges_present["Resid Start->A2.0.Q"] = [C_def_tok_idx]
    edges_present["A0.5->A2.0.Q"] = [C_def_tok_idx]
    edges_present["Resid Start->A2.0.K"] = []  # [B_comma_tok_idx] ',_B' doesn't vary?
    edges_present["A0.5->A2.0.K"] = [B_def_tok_idx]  # Not entirely sure about this
    edges_present["A1.4->A2.0.V"] = [B_comma_tok_idx]
    edges_present["Resid Start->A1.4.V"] = [B_def_tok_idx, B_doc_tok_idx]
    edges_present["Resid Start->A1.2.K"] = [
        A_def_tok_idx,
        B_def_tok_idx,
        A_doc_tok_idx,
        B_doc_tok_idx,
    ]  # This is an inclusive guess as to which tokens might matter
    edges_present["Resid Start->A1.2.Q"] = [A_doc_tok_idx, B_doc_tok_idx]
    edges_present["A0.5->A1.2.Q"] = [B_doc_tok_idx]
    edges_present["A0.5->A1.2.K"] = [B_def_tok_idx, B_doc_tok_idx]

    for layer_3_head in ["0", "6"]:
        edges_present[f"A3.{layer_3_head}->Resid End"] = [param_3_tok_idx]
        edges_present[f"A1.4->A3.{layer_3_head}.Q"] = [param_3_tok_idx]
        edges_present[f"Resid Start->A3.{layer_3_head}.V"] = [C_def_tok_idx]
        edges_present[f"A0.5->A3.{layer_3_head}.V"] = [C_def_tok_idx]
        edges_present[f"A2.0->A3.{layer_3_head}.K"] = [C_def_tok_idx]
        edges_present[f"A1.2->A3.{layer_3_head}.K"] = [A_doc_tok_idx, B_doc_tok_idx]

    true_edges: Set[Edge] = set()
    for edge in model.edges:
        if edge.name in edges_present.keys():
            if token_positions:
                assert edge.seq_idx is not None
                if (edge.seq_idx + seq_start_idx) in edges_present[edge.name]:
                    true_edges.add(edge)
            else:
                true_edges.add(edge)

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

    word_idxs.get("A_def", 0)
    B_def_tok_idx = word_idxs.get("B_def", 0)
    B_comma_tok_idx = word_idxs.get(",_B", 0)
    C_def_tok_idx = word_idxs.get("C_def", 0)
    A_doc_tok_idx = word_idxs.get("A_doc", 0)
    # param_2_tok_idx = word_idxs.get("param_2", 0) A0.2 not included for some reason
    B_doc_tok_idx = word_idxs.get("B_doc", 0)
    param_3_tok_idx = word_idxs.get("param_3", 0)

    if token_positions:
        assert param_3_tok_idx > 0, "Must provide word_idxs if token_positions is True"

    heads_in_circuit: Dict[str, List[int]] = {
        "A1.4": [B_comma_tok_idx],
        "A2.0": [C_def_tok_idx],
        "A3.0": [param_3_tok_idx],
        "A3.6": [param_3_tok_idx],
        "A0.5": [B_def_tok_idx, C_def_tok_idx, B_doc_tok_idx],
        "A1.2": [A_doc_tok_idx, B_doc_tok_idx],
        "A0.2": [B_doc_tok_idx],
        "A0.4": [param_3_tok_idx],
        # Extras for improved performance. Post does not specify token positions.
        "A1.0": [],
        "A0.1": [],
        "A2.3": [],
    }

    heads_to_keep: Set[Tuple[str, Optional[int]]] = set()
    for head_name, tok_idxs in heads_in_circuit.items():
        if token_positions:
            for tok_idx in tok_idxs:
                heads_to_keep.add((head_name, tok_idx))
        else:
            heads_to_keep.add((head_name, None))

    official_edges: Set[Edge] = set()
    not_official_edges: Set[Edge] = set()
    for edge in model.edges:
        src_is_head = edge.src.head_idx is not None
        if token_positions:
            assert edge.seq_idx is not None
            src_head_key = (edge.src.name, edge.seq_idx + seq_start_idx)
        else:
            assert not token_positions
            src_head_key = (edge.src.name, None)

        if src_is_head and src_head_key not in heads_to_keep:
            not_official_edges.add(edge)
            continue

        official_edges.add(edge)
    return official_edges
