import torch as t

from auto_circuit.tasks import Task
from auto_circuit.types import PruneScores


def random_prune_scores(task: Task) -> PruneScores:
    """Prune scores are the mean activation magnitude of each edge."""
    prune_scores = {}
    for edge in task.model.edges:
        prune_scores[edge] = t.rand(1).item()
    return prune_scores
