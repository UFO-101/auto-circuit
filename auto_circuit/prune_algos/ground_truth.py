from typing import Optional, Set

from auto_circuit.data import PromptDataLoader
from auto_circuit.types import Edge, PruneScores
from auto_circuit.utils.patchable_model import PatchableModel


def ground_truth_prune_scores(
    model: PatchableModel,
    dataloader: PromptDataLoader,
    official_edges: Optional[Set[Edge]],
) -> PruneScores:
    """
    Assigns `1` for edges that are in the ground truth circuit, `0` otherwise.

    Args:
        model: The model on which this circuit was discovered.
        dataloader: Not used.
        official_edges: The edges of the circuit.

    Returns:
        An ordering of the edges by importance to the task. Importance is equal to the
            absolute value of the score assigned to the edge.
    """
    prune_scores: PruneScores = model.new_prune_scores()

    if official_edges is None:
        raise ValueError("Official edges must be provided for ground truth pruning.")
    for edge in official_edges:
        prune_scores[edge.dest.module_name][edge.patch_idx] = 1.0
    return prune_scores
