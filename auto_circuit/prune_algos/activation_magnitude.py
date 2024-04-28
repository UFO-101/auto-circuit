from typing import Optional, Set

import torch as t

from auto_circuit.data import PromptDataLoader
from auto_circuit.types import AblationType, Edge, PruneScores
from auto_circuit.utils.ablation_activations import src_ablations
from auto_circuit.utils.patchable_model import PatchableModel


def activation_magnitude_prune_scores(
    model: PatchableModel,
    dataloader: PromptDataLoader,
    official_edges: Optional[Set[Edge]],
) -> PruneScores:
    """
    Simple baseline circuit discovery algorithm. Prune scores are the mean activation
    magnitude of each edge.

    Args:
        model: The model to find the circuit for.
        dataloader: The dataloader to use for input.
        official_edges: Not used.

    Returns:
        An ordering of the edges by importance to the task. Importance is equal to the
            absolute value of the score assigned to the edge.
    """
    prune_scores = model.new_prune_scores()
    n_batches = len(dataloader)
    with t.inference_mode():
        for batch in dataloader:
            src_outs = src_ablations(model, batch.clean, AblationType.RESAMPLE)
            src_out_means = src_outs.mean(dim=list(range(1, src_outs.ndim)))
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
