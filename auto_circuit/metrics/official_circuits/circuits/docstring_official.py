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
    The Docstring circuit, from the
    [blog post](https://www.alignmentforum.org/posts/u6KXXmKFbXfWzoAXn) by Stefan
    Heimersheim and Jett Janiak (2023).

    The exact set of edges was defined by Stephan in the
    [ACDC repo](https://github.com/ArthurConmy/Automatic-Circuit-Discovery/blob/main/acdc/docstring/utils.py).
    The token positions are based on my reading of the paper, but I have been over them
    briefly with Stefan and he endorsed them as reasonable.

    Args:
        model: A patchable TransformerLens GPT-2 `HookedTransformer` model.
        token_positions: Whether to distinguish between token positions when returning
            the set of circuit edges. If `True`, we require that the `model` has
            `seq_len` not `None` (ie. separate edges for each token position) and that
            `word_idxs` is provided.
        word_idxs: A dictionary defining the index of specific named tokens in the
            circuit definition. For this circuit, the required tokens positions are:
            <ul>
                <li><code>A_def</code></li>
                <li><code>B_def</code></li>
                <li><code>,_B</code></li>
                <li><code>C_def</code></li>
                <li><code>A_doc</code></li>
                <li><code>B_doc</code></li>
                <li><code>param_3</code></li>
            </ul>
        seq_start_idx: Offset to add to all of the token positions in `word_idxs`.
            This is useful when using KV caching to skip the common prefix of the
            prompt.

    Returns:
        The set of edges in the circuit.
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
    The attention heads in the Docstring circuit, from the
    [blog post](https://www.alignmentforum.org/posts/u6KXXmKFbXfWzoAXn) by Stefan
    Heimersheim and Jett Janiak (2023).

    In the blog post, to measure
    [the overall performance](https://www.alignmentforum.org/posts/u6KXXmKFbXfWzoAXn#Putting_it_all_together)
    of the circuit, the authors Node Patch the heads in the circuit, rather than Edge
    Patching the specific edges they find. We include this variation to enable
    replication of these results.

    The token positions are based on my reading of the paper, but some were unspecified
    so in those cases we include all token positions between `A_def` and `param_3`.

    Args:
        model: A patchable TransformerLens GPT-2 `HookedTransformer` model.
        token_positions: Whether to distinguish between token positions when returning
            the set of circuit edges. If `True`, we require that the `model` has
            `seq_len` not `None` (ie. separate edges for each token position) and that
            `word_idxs` is provided.
        word_idxs: A dictionary defining the index of specific named tokens in the
            circuit definition. For this circuit, the required tokens positions are:
            <ul>
                <li><code>A_def</code></li>
                <li><code>B_def</code></li>
                <li><code>,_B</code></li>
                <li><code>C_def</code></li>
                <li><code>A_doc</code></li>
                <li><code>B_doc</code></li>
                <li><code>param_3</code></li>
            </ul>
        seq_start_idx: Offset to add to all of the token positions in `word_idxs`.
            This is useful when using KV caching to skip the common prefix of the
            prompt.

    Returns:
        The set of edges in the circuit.
    """

    word_idxs.get("A_def", 0)
    B_def_tok_idx = word_idxs.get("B_def", 0)
    B_comma_tok_idx = word_idxs.get(",_B", 0)
    C_def_tok_idx = word_idxs.get("C_def", 0)
    A_doc_tok_idx = word_idxs.get("A_doc", 0)
    # param_2_tok_idx = word_idxs.get("param_2", 0) A0.2 not included for some reason
    B_doc_tok_idx = word_idxs.get("B_doc", 0)
    param_3_tok_idx = word_idxs.get("param_3", 0)

    all_tok_idxs = [
        B_def_tok_idx,
        B_comma_tok_idx,
        C_def_tok_idx,
        A_doc_tok_idx,
        # param_2_tok_idx,
        B_doc_tok_idx,
        param_3_tok_idx,
    ]
    tok_idx_range: List[int] = list(range(min(all_tok_idxs), max(all_tok_idxs) + 1))

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
        "A1.0": tok_idx_range,
        "A0.1": tok_idx_range,
        "A2.3": tok_idx_range,
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
