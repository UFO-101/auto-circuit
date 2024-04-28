from typing import Dict, List, Set

from auto_circuit.types import Edge
from auto_circuit.utils.patchable_model import PatchableModel


def tracr_reverse_true_edges(
    model: PatchableModel,
    token_positions: bool = False,
    word_idxs: Dict[str, int] = {},
    seq_start_idx: int = 0,
) -> Set[Edge]:
    """
    The canonical circuit for tracr-reverse according to Miller et al. (Forthcoming).
    As discussed in the paper, this circuit is the set of edges that must be preserved
    when Resample Ablation is used.

    Args:
        model: A patchable TransformerLens tracr-reverse `HookedTransformer` model.
        token_positions: Whether to distinguish between token positions when returning
            the set of circuit edges. If `True`, we require that the `model` has
            `seq_len` not `None` (ie. separate edges for each token position) and that
            `word_idxs` is provided.
        word_idxs: A dictionary defining the index of specific named tokens in the
            circuit definition. This variable is not used in this circuit, instead we
            assume a sequence of length 6 (including BOS).
        seq_start_idx: Offset to add to all of the token positions in `word_idxs`.
            This is useful when using KV caching to skip the common prefix of the
            prompt.

    Returns:
        The set of edges in the circuit.

    Note:
        The sequence positions assume prompts of length 6 (including BOS), as in
        tracr/tracr_reverse_len_5_prompts.json
    """
    assert model.cfg.model_name == "tracr-reverse"
    assert model.separate_qkv

    # tok_seq_pos = [1, 2, 3, 4, 5]
    tok_seq_pos = [1, 2, 4, 5]
    edges_present: Dict[str, List[int]] = {}
    edges_present["Resid Start->A3.0.V"] = tok_seq_pos
    edges_present["A3.0->Resid End"] = tok_seq_pos

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


def tracr_reverse_acdc_edges(
    model: PatchableModel,
    token_positions: bool = False,
    word_idxs: Dict[str, int] = {},
    seq_start_idx: int = 0,
) -> Set[Edge]:
    """
    The canonical circuit for tracr-reverse according to
    [Conmy et al. (2023)](https://arxiv.org/abs/2304.14997). Based on the
    [ACDC repo](https://github.com/ArthurConmy/Automatic-Circuit-Discovery/blob/main/acdc/tracr_task/utils.py).

    As discussed in Miller et al. (forthcoming), this circuit is (intended to be) the
    set of edges that must be preserved when Zero Ablation is used.

    Args:
        model: A patchable TransformerLens tracr-reverse `HookedTransformer` model.
        token_positions: Whether to distinguish between token positions when returning
            the set of circuit edges. If `True`, we require that the `model` has
            `seq_len` not `None` (ie. separate edges for each token position) and that
            `word_idxs` is provided.
        word_idxs: A dictionary defining the index of specific named tokens in the
            circuit definition. This variable is not used in this circuit, instead we
            assume a sequence of length 6 (including BOS).
        seq_start_idx: Offset to add to all of the token positions in `word_idxs`.
            This is useful when using KV caching to skip the common prefix of the
            prompt.

    Returns:
        The set of edges in the circuit.

    Note:
        The sequence positions assume prompts of length 6 (including BOS), as in
        tracr/tracr_reverse_len_5_prompts.json
    """
    assert model.cfg.model_name == "tracr-reverse"
    assert model.separate_qkv

    # tok_seq_pos = [1, 2, 3, 4, 5]
    tok_seq_pos = [1, 2, 4, 5]
    edges_present: Dict[str, List[int]] = {}

    edges_present["A3.0->Resid End"] = tok_seq_pos
    edges_present["MLP 2->A3.0.Q"] = tok_seq_pos
    edges_present["Resid Start->A3.0.K"] = tok_seq_pos
    edges_present["Resid Start->A3.0.V"] = tok_seq_pos
    edges_present["MLP 1->MLP 2"] = tok_seq_pos
    edges_present["MLP 0->MLP 1"] = tok_seq_pos
    edges_present["Resid Start->MLP 1"] = tok_seq_pos
    edges_present["A0.0->MLP 0"] = tok_seq_pos
    edges_present["Resid Start->MLP 0"] = tok_seq_pos
    edges_present["Resid Start->A0.0.V"] = tok_seq_pos

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
