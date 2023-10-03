from typing import Dict, Set

import torch as t
from torch.utils.data import DataLoader

from auto_circuit.data import PromptPairBatch
from auto_circuit.types import Edge


def random_prune_scores(
    model: t.nn.Module,
    train_data: DataLoader[PromptPairBatch],
) -> Dict[Edge, float]:
    """Prune scores are the mean activation magnitude of each edge."""
    edges: Set[Edge] = model.edges  # type: ignore
    prune_scores = {}
    for edge in edges:
        prune_scores[edge] = t.rand(1).item()
    return prune_scores
