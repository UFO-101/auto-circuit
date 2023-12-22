from auto_circuit.tasks import Task
from auto_circuit.types import PruneScores


def ground_truth_prune_scores(task: Task) -> PruneScores:
    """Return 1 for edges that are in the ground truth circuit, 0 otherwise."""
    prune_scores = {}
    ground_truth_edges = task.true_edges
    for edge in task.model.edges:
        if edge in ground_truth_edges:
            prune_scores[edge] = 1.0
    return prune_scores
