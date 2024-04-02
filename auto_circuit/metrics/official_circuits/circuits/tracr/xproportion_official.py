from typing import Dict, List, Set

from auto_circuit.types import Edge
from auto_circuit.utils.patchable_model import PatchableModel


def tracr_xproportion_official_edges(
    model: PatchableModel,
    token_positions: bool = False,
    word_idxs: Dict[str, int] = {},
    seq_start_idx: int = 0,
) -> Set[Edge]:
    """
    !!! Note: !!!
    The sequence positions assume prompts of length 6 (including BOS), as in
    tracr/tracr_xproportion_len_5_prompts.json
    """
    assert model.cfg.model_name == "tracr-xproportion"
    assert model.separate_qkv

    tok_seq_pos = [1, 2, 3, 4, 5]
    edges_present: Dict[str, List[int]] = {}
    edges_present["Resid Start->MLP 0"] = tok_seq_pos
    edges_present["MLP 0->A1.0.V"] = tok_seq_pos
    edges_present["A1.0->Resid End"] = tok_seq_pos

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


def tracr_xproportion_acdc_edges(
    model: PatchableModel,
    token_positions: bool = False,
    word_idxs: Dict[str, int] = {},
    seq_start_idx: int = 0,
) -> Set[Edge]:
    """
    The canonical circuit for tracr-reverse according to Conmy et al.
    (https://arxiv.org/abs/2304.14997).
    We question the correctness of this circuit in Miller et al. (forthcoming).
    !!! Note: !!!
    The sequence positions assume prompts of length 6 (including BOS), as in
    tracr/tracr_xproportion_len_5_prompts.json
    """
    assert model.cfg.model_name == "tracr-xproportion"
    assert model.separate_qkv

    tok_seq_pos = [1, 2, 3, 4, 5]
    edges_present: Dict[str, List[int]] = {}

    edges_present["A1.0->Resid End"] = tok_seq_pos
    edges_present["Resid Start->A1.0.Q"] = tok_seq_pos
    edges_present["Resid Start->A1.0.K"] = tok_seq_pos
    edges_present["MLP 0->A1.0.V"] = tok_seq_pos
    edges_present["Resid Start->MLP 0"] = tok_seq_pos

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
