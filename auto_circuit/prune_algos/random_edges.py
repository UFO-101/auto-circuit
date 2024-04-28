from typing import Optional, Set

import torch as t

from auto_circuit.data import PromptDataLoader
from auto_circuit.types import Edge, PruneScores
from auto_circuit.utils.patchable_model import PatchableModel


def random_prune_scores(
    model: PatchableModel,
    dataloader: PromptDataLoader,
    official_edges: Optional[Set[Edge]],
) -> PruneScores:
    """Prune scores are the mean activation magnitude of each edge."""
    """
    Random baseline circuit discovery algorithm. Prune scores are random values.

    Args:
        model: The model to find the circuit for.
        dataloader: Not used.
        official_edges: Not used.

    Returns:
        An ordering of the edges by importance to the task. Importance is equal to the
            absolute value of the score assigned to the edge.
    """
    prune_scores: PruneScores = {}
    for mod_name, patch_mask in model.patch_masks.items():
        prune_scores[mod_name] = t.rand_like(patch_mask.data)
    return prune_scores
