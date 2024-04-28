from typing import List, Set, Tuple

from auto_circuit.tasks import TASK_DICT
from auto_circuit.types import (
    AlgoMeasurements,
    Edge,
    EdgeCounts,
    Measurements,
    PruneScores,
    TaskMeasurements,
    TaskPruneScores,
)
from auto_circuit.utils.custom_tqdm import tqdm
from auto_circuit.utils.graph_utils import edge_counts_util
from auto_circuit.utils.patchable_model import PatchableModel
from auto_circuit.utils.tensor_ops import desc_prune_scores, prune_scores_threshold


def measure_roc(task_prune_scores: TaskPruneScores) -> TaskMeasurements:
    """
    Wrapper of
    [`measure_task_roc`][auto_circuit.metrics.official_circuits.measure_roc.measure_task_roc]
    that measures the ROC curve for each task and algorithm.
    """
    task_measurements: TaskMeasurements = {}
    for task_key, algo_prune_scores in (task_pbar := tqdm(task_prune_scores.items())):
        task = TASK_DICT[task_key]
        task_pbar.set_description_str(f"Measuring ROC Task: {task.name}")
        algo_measurements: AlgoMeasurements = {}
        for algo_key, prune_scores in (algo_pbar := tqdm(algo_prune_scores.items())):
            algo_pbar.set_description_str(f"Measuring ROC Pruning with {algo_key}")
            official_edge = task.model.true_edges
            if official_edge is None:
                raise ValueError("This task does not have a true edge function")
            algo_measurement = measure_task_roc(task.model, official_edge, prune_scores)
            algo_measurements[algo_key] = algo_measurement
        task_measurements[task_key] = algo_measurements
    return task_measurements


def measure_task_roc(
    model: PatchableModel,
    official_edges: Set[Edge],
    prune_scores: PruneScores,
    all_edges: bool = False,
) -> Measurements:
    """
    Finds points for the Receiver Operating Characteristic (ROC) curve that
    measures the performance of the given `prune_scores` at classifying which edges
    are in or out of the circuit defined by `official_edges`.

    Args:
        model: The model to measure the ROC curve for.
        official_edges: The edges that define the correct circuit.
        prune_scores: The pruning scores to measure the ROC curve for. The scores
            define an ordering of the edges in the model. We sweep through the scores
            in descending order, including the top-k edges in the circuit.
        all_edges: By default we calculate the True Positive Rate (TRP) and False
            Positive Rate (FPR) for the set of [number of edges] determined by passing
            `prune_scores` to
            [`edge_counts_util`][auto_circuit.utils.graph_utils.edge_counts_util]. If
            `all_edges` is `True`, we instead calculate the TRP and FPR for every number
            `0, 1, 2, ..., model.n_edges` of edges.

    Returns:
        A list of points that define the ROC curve. Each point is a tuple of the form
            `(FPR, TPR)`. The first point is always `(0, 0)` and the last point is
            always `(1, 1)`.
    """
    n_official = len(official_edges)
    n_complement = model.n_edges - n_official
    if all_edges:
        test_edge_counts = edge_counts_util(model.edges, test_counts=EdgeCounts.ALL)
    else:
        test_edge_counts = edge_counts_util(model.edges, prune_scores=prune_scores)
    desc_ps = desc_prune_scores(prune_scores)

    correct_ps: PruneScores = model.circuit_prune_scores(official_edges, bool=True)
    incorrect_edges: PruneScores = dict([(mod, ~ps) for mod, ps in correct_ps.items()])

    points: List[Tuple[float, float]] = [(0.0, 0.0)]
    for edge_count in (edge_count_pbar := tqdm(test_edge_counts)):
        edge_count_pbar.set_description_str(f"ROC for {edge_count} edges")
        threshold = prune_scores_threshold(desc_ps, edge_count)
        true_positives, false_positives = 0, 0
        for mod, ps in prune_scores.items():
            ps_circuit = ps.abs() >= threshold
            correct_circuit, incorrect_circuit = correct_ps[mod], incorrect_edges[mod]
            true_positives += (ps_circuit & correct_circuit).sum().item()
            false_positives += (ps_circuit & incorrect_circuit).sum().item()
        true_positive_rate = true_positives / n_official
        false_positive_rate = false_positives / n_complement
        points.append((false_positive_rate, true_positive_rate))
    points.append((1.0, 1.0))
    return points
