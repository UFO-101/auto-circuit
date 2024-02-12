import torch as t

from auto_circuit.tasks import Task
from auto_circuit.types import PruneScores


def random_prune_scores(task: Task) -> PruneScores:
    """Prune scores are the mean activation magnitude of each edge."""
    prune_scores: PruneScores = {}
    for mod_name, patch_mask in task.model.patch_masks.items():
        prune_scores[mod_name] = t.rand_like(patch_mask.data)
    return prune_scores
