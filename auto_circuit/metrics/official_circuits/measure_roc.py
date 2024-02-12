from typing import List, Tuple

from auto_circuit.tasks import TASK_DICT, Task
from auto_circuit.types import (
    AlgoMeasurements,
    Measurements,
    PruneScores,
    TaskMeasurements,
    TaskPruneScores,
)
from auto_circuit.utils.custom_tqdm import tqdm
from auto_circuit.utils.graph_utils import edge_counts_util
from auto_circuit.utils.tensor_ops import desc_prune_scores, prune_scores_threshold


def measure_roc(task_prune_scores: TaskPruneScores) -> TaskMeasurements:
    task_measurements: TaskMeasurements = {}
    for task_key, algo_prune_scores in (task_pbar := tqdm(task_prune_scores.items())):
        task = TASK_DICT[task_key]
        task_pbar.set_description_str(f"Measuring ROC Task: {task.name}")
        algo_measurements: AlgoMeasurements = {}
        for algo_key, prune_scores in (algo_pbar := tqdm(algo_prune_scores.items())):
            algo_pbar.set_description_str(f"Measuring ROC Pruning with {algo_key}")
            algo_measurement = measure_task_roc(task, prune_scores)
            algo_measurements[algo_key] = algo_measurement
        task_measurements[task_key] = algo_measurements
    return task_measurements


def measure_task_roc(
    task: Task,
    prune_scores: PruneScores,
) -> Measurements:
    """Measure ROC curve."""
    correct_edges = task.true_edges
    if correct_edges is None:
        raise ValueError("This task does not have a true edge function")
    n_correct = len(correct_edges)
    n_incorrect = task.model.n_edges - n_correct
    test_edge_counts = edge_counts_util(task.model.edges, prune_scores=prune_scores)
    desc_ps = desc_prune_scores(prune_scores)

    correct_ps: PruneScores = task.model.circuit_prune_scores(correct_edges, bool=True)
    incorrect_edges: PruneScores = dict([(mod, ~ps) for mod, ps in correct_ps.items()])

    points: List[Tuple[float, float]] = []
    for edge_count in (edge_count_pbar := tqdm(test_edge_counts)):
        edge_count_pbar.set_description_str(f"ROC for {edge_count} edges")
        threshold = prune_scores_threshold(desc_ps, edge_count)
        true_positives, false_positives = 0, 0
        for mod, ps in prune_scores.items():
            ps_circuit = ps >= threshold
            correct_circuit, incorrect_circuit = correct_ps[mod], incorrect_edges[mod]
            true_positives += (ps_circuit & correct_circuit).sum().item()
            false_positives += (ps_circuit & incorrect_circuit).sum().item()
        true_positive_rate = true_positives / n_correct
        false_positive_rate = false_positives / n_incorrect
        points.append((false_positive_rate, true_positive_rate))
    return points
