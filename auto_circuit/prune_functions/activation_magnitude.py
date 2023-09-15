from functools import partial
from typing import Dict, Tuple

import torch as t
from torch.utils.data import DataLoader

from auto_circuit.data import PromptPairBatch
from auto_circuit.types import Edge
from auto_circuit.utils.graph_utils import graph_edges
from auto_circuit.utils.misc import remove_hooks


def output_hook(
    module: t.nn.Module,
    input: Tuple[t.Tensor, ...],
    output: t.Tensor,
    edge: Edge,
    act_dict: Dict[Edge, t.Tensor],
) -> None:
    if edge not in act_dict:
        act_dict[edge] = output[edge.src.out_idx]
    else:
        act_dict[edge] += output[edge.src.out_idx]


def activation_magnitude_prune_scores(
    model: t.nn.Module, factorized: bool, train_data: DataLoader[PromptPairBatch]
) -> Dict[Edge, float]:
    """Prune scores are the mean activation magnitude of each edge."""
    edges = graph_edges(model, factorized)
    act_dict: Dict[Edge, t.Tensor] = {}
    with remove_hooks() as handles:
        for edge in edges:
            handle = edge.src.module(model).register_forward_hook(
                partial(output_hook, edge=edge, act_dict=act_dict)
            )
            handles.add(handle)
        for batch in train_data:
            model(batch.clean)

    prune_scores = {}
    for edge, act in act_dict.items():
        prune_scores[edge] = t.mean(t.abs(act)).item()
    return prune_scores
