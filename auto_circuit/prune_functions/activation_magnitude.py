from typing import Dict, Set

import torch as t
from torch.utils.data import DataLoader

from auto_circuit.data import PromptPairBatch
from auto_circuit.types import Edge, SrcNode
from auto_circuit.utils.graph_utils import get_sorted_src_outs


def activation_magnitude_prune_scores(
    model: t.nn.Module, train_data: DataLoader[PromptPairBatch]
) -> Dict[Edge, float]:
    """Prune scores are the mean activation magnitude of each edge."""
    edges: Set[Edge] = model.edges  # type: ignore
    act_dict: Dict[Edge, t.Tensor] = {}
    with t.inference_mode():
        for batch in train_data:
            src_outs: Dict[SrcNode, t.Tensor] = get_sorted_src_outs(model, batch.clean)
            for edge in edges:
                if edge in act_dict:
                    act_dict[edge] += src_outs[edge.src]
                else:
                    act_dict[edge] = src_outs[edge.src]

    prune_scores = {}
    for edge, act in act_dict.items():
        prune_scores[edge] = t.mean(t.abs(act)).item()
    return prune_scores
