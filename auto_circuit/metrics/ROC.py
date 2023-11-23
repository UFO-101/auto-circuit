from typing import Dict, List, Set, Tuple

import torch as t

from auto_circuit.types import Edge, EdgeCounts
from auto_circuit.utils.graph_utils import edge_counts_util


def is_head_node(edge: Edge) -> bool:
    valid_starts = ["A", "R"]
    return edge.src.name[0] in valid_starts and edge.dest.name[0] in valid_starts


def measure_roc(
    model: t.nn.Module,
    prune_scores: Dict[Edge, float],
    correct_edges: Set[Edge],
    head_nodes_only: bool = False,
    group_edges: bool = False,
) -> List[Tuple[float, float]]:
    edges: Set[Edge] = model.edges  # type: ignore

    # Filter edges that are not between attention heads
    edges = set(filter(is_head_node, edges)) if head_nodes_only else edges
    prune_scores = {edge: val for edge, val in prune_scores.items() if edge in edges}
    sort_ps = dict(sorted(prune_scores.items(), key=lambda x: abs(x[1]), reverse=True))

    edge_counts_type = EdgeCounts.GROUPS if group_edges else EdgeCounts.LOGARITHMIC
    test_edge_counts = edge_counts_util(edges, edge_counts_type, sort_ps)

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
