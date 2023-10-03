from typing import Dict, List, Set, Tuple

import torch as t

from auto_circuit.types import Edge


def measure_roc(
    model: t.nn.Module,
    prune_scores: Dict[Edge, float],
    test_edge_counts: List[int],
    correct_edges: Set[Edge],
) -> Set[Tuple[float, float]]:
    edges: Set[Edge] = model.edges  # type: ignore
    incorrect_edges = edges - correct_edges
    points: Set[Tuple[float, float]] = set()
    current_pred_edges: Set[Edge] = set()
    prune_scores = dict(sorted(prune_scores.items(), key=lambda x: x[1], reverse=True))
    for edge_idx, (edge, _) in enumerate(prune_scores.items()):
        current_pred_edges.add(edge)
        edge_count = edge_idx + 1
        if edge_count in test_edge_counts:
            true_positives = len(correct_edges & current_pred_edges)
            true_positive_rate = true_positives / len(correct_edges)
            false_positives = len(incorrect_edges & current_pred_edges)
            false_positive_rate = false_positives / len(incorrect_edges)
            points.add((false_positive_rate, true_positive_rate))
    return points
