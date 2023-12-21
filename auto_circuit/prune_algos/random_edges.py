from typing import Set

import torch as t

from auto_circuit.tasks import Task
from auto_circuit.types import Edge, PruneScores


def random_prune_scores(task: Task) -> PruneScores:
    """Prune scores are the mean activation magnitude of each edge."""
    edges: Set[Edge] = task.model.edges  # type: ignore
    prune_scores = {}
    for edge in edges:
        prune_scores[edge] = t.rand(1).item()
    return prune_scores
