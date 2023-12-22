from typing import Dict

import torch as t

from auto_circuit.tasks import Task
from auto_circuit.types import Edge, PruneScores, SrcNode
from auto_circuit.utils.graph_utils import get_sorted_src_outs


def activation_magnitude_prune_scores(task: Task) -> PruneScores:
    """Prune scores are the mean activation magnitude of each edge."""
    model = task.model
    act_dict: Dict[Edge, t.Tensor] = {}
    with t.inference_mode():
        for batch in task.train_loader:
            src_outs: Dict[SrcNode, t.Tensor] = get_sorted_src_outs(model, batch.clean)
            for edge in model.edges:
                if edge in act_dict:
                    act_dict[edge] += src_outs[edge.src]
                else:
                    act_dict[edge] = src_outs[edge.src]

    prune_scores = {}
    for edge, act in act_dict.items():
        prune_scores[edge] = t.mean(t.abs(act)).item()
    return prune_scores
