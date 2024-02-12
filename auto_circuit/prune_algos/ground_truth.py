from auto_circuit.tasks import Task
from auto_circuit.types import PruneScores


def ground_truth_prune_scores(task: Task) -> PruneScores:
    """Return 1 for edges that are in the ground truth circuit, 0 otherwise."""
    prune_scores: PruneScores = task.model.new_prune_scores()

    ground_truth_edges = task.true_edges
    if ground_truth_edges is None:
        raise ValueError("This task does not have a true edge function")
    for edge in ground_truth_edges:
        prune_scores[edge.dest.module_name][edge.patch_idx] = 1.0
    return prune_scores
