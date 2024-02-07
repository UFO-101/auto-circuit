from typing import Dict, List, Set

from auto_circuit.types import Edge
from auto_circuit.utils.patchable_model import PatchableModel


def tracr_reverse_true_edges(
    model: PatchableModel, token_positions: bool = False, seq_start_idx: int = 0
) -> Set[Edge]:
    """
    !!! Note: !!!
    The sequence positions assume prompts of length 6 (including BOS), as in
    tracr/tracr_reverse_len_5_prompts.json
    """
    assert model.cfg.model_name == "Attn_Only_4L512W_C4_Code"

    edges_present: Dict[str, List[int]] = {}
    # edges_present["Resid Start->A2.0.Q"] = [15]

    # reflects the value in the docstring appendix of the manual circuit as of 12th June
    assert len(edges_present) == 24, len(edges_present)

    true_edges: Set[Edge] = set()
    for edge in model.edges:
        if edge.name in edges_present.keys():
            if token_positions:
                for tok_pos in edges_present[edge.name]:
                    true_edges.add(Edge(edge.src, edge.dest, tok_pos - seq_start_idx))
            else:
                true_edges.add(edge)
    return true_edges
