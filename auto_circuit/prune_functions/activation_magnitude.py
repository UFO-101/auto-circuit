from functools import partial
from typing import Dict, Tuple

import torch as t
from torch.utils.data import DataLoader

from auto_circuit.data import PromptPairBatch
from auto_circuit.types import Edge
from auto_circuit.utils import graph_edges


def output_hook(
    module: t.nn.Module,
    input: Tuple[t.Tensor, ...],
    output: t.Tensor,
    edge: Edge,
    act_dict: Dict[Edge, t.Tensor],
) -> None:
    if edge not in act_dict:
        act_dict[edge] = output[edge.src.t_idx]
    else:
        act_dict[edge] += output[edge.src.t_idx]


def activation_magnitude_prune_scores(
    model: t.nn.Module,
    train_data: DataLoader[PromptPairBatch],
) -> Dict[Edge, float]:
    """Prune scores are the mean activation magnitude of each edge."""
    edges = graph_edges(model)
    act_dict: Dict[Edge, t.Tensor] = {}
    handles = []
    try:
        for edge in edges:
            handle = edge.src.module.register_forward_hook(
                partial(output_hook, edge=edge, act_dict=act_dict)
            )
            handles.append(handle)
        for batch in train_data:
            model(batch.clean)
    finally:
        [handle.remove() for handle in handles]

    prune_scores = {}
    for edge, act in act_dict.items():
        prune_scores[edge] = t.mean(t.abs(act)).item()
    return prune_scores
