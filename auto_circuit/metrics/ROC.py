from typing import Dict, List, Set, Tuple

import torch as t

from auto_circuit.types import Edge, Measurements
from auto_circuit.utils.graph_utils import edge_counts_util


def measure_roc(
    model: t.nn.Module,
    prune_scores: Dict[Edge, float],
    correct_edges: Set[Edge],
) -> Measurements:
    edges: Set[Edge] = model.edges  # type: ignore

    prune_scores = {edge: val for edge, val in prune_scores.items() if edge in edges}
    sort_ps = dict(sorted(prune_scores.items(), key=lambda x: abs(x[1]), reverse=True))

    test_edge_counts = edge_counts_util(edges, prune_scores=sort_ps)

    incorrect_edges = edges - correct_edges
    points: List[Tuple[float, float]] = []
    current_pred_edges: Set[Edge] = set()
    for edge_idx, (edge, _) in enumerate(sort_ps.items()):
        current_pred_edges.add(edge)
        edge_count = edge_idx + 1
        if edge_count in test_edge_counts or edge_count == len(edges):
            true_positives = len(correct_edges & current_pred_edges)
            true_positive_rate = true_positives / len(correct_edges)
            false_positives = len(incorrect_edges & current_pred_edges)
            false_positive_rate = false_positives / len(incorrect_edges)
            points.append((false_positive_rate, true_positive_rate))
    return points
