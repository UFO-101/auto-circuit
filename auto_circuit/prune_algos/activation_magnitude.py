from typing import Dict

import torch as t

from auto_circuit.tasks import Task
from auto_circuit.types import PruneScores, SrcNode
from auto_circuit.utils.graph_utils import get_sorted_src_outs


def activation_magnitude_prune_scores(task: Task) -> PruneScores:
    """Prune scores are the mean activation magnitude of each edge."""
    model = task.model
    prune_scores = model.new_prune_scores()
    n_batches = len(task.train_loader)
    with t.inference_mode():
        for batch in task.train_loader:
            src_outs: Dict[SrcNode, t.Tensor] = get_sorted_src_outs(model, batch.clean)
            src_out_stack = t.stack(list(src_outs.values()))
            src_out_means = src_out_stack.mean(dim=list(range(1, src_out_stack.ndim)))
            # prune_scores shape = seq_shape + head_shape + [prev_src_count]
            for mod, ps in prune_scores.items():
                n_srcs = ps.size(-1)
                edge_acts = src_out_means[:n_srcs]
                if ps.ndim >= 2:
                    edge_acts = edge_acts.unsqueeze(0).repeat(ps.shape[-2], 1)
                if ps.ndim >= 3:
                    edge_acts = edge_acts.unsqueeze(0).repeat(ps.shape[-3], 1, 1)
                prune_scores[mod] += edge_acts.abs() / n_batches
    return prune_scores
