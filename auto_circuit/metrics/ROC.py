from typing import List, Optional, Set, Tuple

from auto_circuit.types import Edge, Measurements, PrunedOutputs, PruneScores, Task
from auto_circuit.utils.custom_tqdm import tqdm
from auto_circuit.utils.graph_utils import edge_counts_util


def measure_roc(
    task: Task,
    prune_scores: Optional[PruneScores],
    pruned_outs: Optional[PrunedOutputs],
) -> Measurements:
    """Measure ROC curve."""
    assert prune_scores is not None
    correct_edges = task.true_edges
    edges: Set[Edge] = task.model.edges  # type: ignore

    test_edge_counts = edge_counts_util(edges, prune_scores=prune_scores)

    prune_scores = {e: prune_scores.get(e, 0.0) for e in edges}
    assert len(prune_scores) == len(edges)
    sort_ps = dict(sorted(prune_scores.items(), key=lambda x: abs(x[1]), reverse=True))
    sorted_edges = list(sort_ps.keys())

    incorrect_edges = edges - correct_edges
    points: List[Tuple[float, float]] = []
    current_edges: Set[Edge] = set()
    for edge_idx in (prune_score_pbar := tqdm(range(len(sort_ps) + 1))):
        if edge_idx in test_edge_counts:
            prune_score_pbar.set_description_str(f"ROC for {edge_idx} edges")
            true_positives = len(correct_edges & current_edges)
            true_positive_rate = true_positives / len(correct_edges)
            false_positives = len(incorrect_edges & current_edges)
            false_positive_rate = false_positives / len(incorrect_edges)
            points.append((false_positive_rate, true_positive_rate))
        if edge_idx < len(edges):
            current_edges.add(sorted_edges[edge_idx])
    return points
