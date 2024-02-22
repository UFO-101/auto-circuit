from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import plotly.graph_objects as go
import torch as t

from auto_circuit.metrics.area_under_curve import average_auc_plot
from auto_circuit.metrics.prune_metrics.prune_metrics import (
    PRUNE_METRIC_DICT,
    PruneMetric,
)
from auto_circuit.metrics.prune_metrics.prune_metrics_plot import edge_patching_plot
from auto_circuit.prune import run_circuits
from auto_circuit.prune_algos.prune_algos import PRUNE_ALGO_DICT
from auto_circuit.tasks import TASK_DICT
from auto_circuit.types import (
    CircuitOutputs,
    PatchType,
    PruneMetricMeasurements,
    TaskPruneScores,
)
from auto_circuit.utils.custom_tqdm import tqdm
from auto_circuit.utils.graph_utils import edge_counts_util


def default_factory() -> Dict[Any, Dict[Any, Any]]:
    return defaultdict(dict)


def measure_prune_metrics(
    metrics: List[PruneMetric],
    task_prune_scores: TaskPruneScores,
    patch_type: PatchType,
    reverse_clean_corrupt: bool = False,
    test_edge_counts: Optional[List[int]] = None,
) -> PruneMetricMeasurements:
    measurements: PruneMetricMeasurements = defaultdict(default_factory)
    for task_key, algo_prune_scores in (task_pbar := tqdm(task_prune_scores.items())):
        task = TASK_DICT[task_key]
        task_pbar.set_description_str(f"Task: {task.name}")
        test_loader = task.test_loader
        for algo_key, prune_scores in (algo_pbar := tqdm(algo_prune_scores.items())):
            algo = PRUNE_ALGO_DICT[algo_key]
            algo_pbar.set_description_str(f"Pruning with {algo.name}")
            default_edge_counts = edge_counts_util(
                edges=task.model.edges,
                test_counts=None,
                prune_scores=prune_scores,
                true_edge_count=task.true_edge_count,
            )
            circuit_outs: CircuitOutputs = run_circuits(
                model=task.model,
                dataloader=test_loader,
                test_edge_counts=test_edge_counts or default_edge_counts,
                prune_scores=prune_scores,
                patch_type=patch_type,
                reverse_clean_corrupt=reverse_clean_corrupt,
            )
            for metric in (metric_pbar := tqdm(metrics)):
                metric_pbar.set_description_str(f"Measuring {metric.name}")
                measurement = metric.metric_func(task, circuit_outs)
                measurements[metric.key][task.key][algo.key] = measurement
            del circuit_outs
            t.cuda.empty_cache()
    return measurements


def measurement_figs(
    measurements: PruneMetricMeasurements, auc_plots: bool = False
) -> Tuple[go.Figure, ...]:
    figs = []
    for metric_key, task_measurements in measurements.items():
        token_circuit = TASK_DICT[list(task_measurements.keys())[0]].token_circuit
        if metric_key not in PRUNE_METRIC_DICT:
            print(f"Skipping unknown metric: {metric_key}")
            continue
        metric = PRUNE_METRIC_DICT[metric_key]
        data, y_max = [], 0.0
        for task_key, algo_measurements in task_measurements.items():
            task = TASK_DICT[task_key]
            # Assert all tasks have the same token_circuit value
            assert task.token_circuit == token_circuit

            for algo_key, points in algo_measurements.items():
                algo = PRUNE_ALGO_DICT[algo_key]
                if len(points) > 1:
                    for x, y in points:
                        data.append(
                            {
                                "Task": task.name,
                                "Algorithm": algo.short_name,
                                "X": max(x, 0.5) if metric.log_x else x,
                                "Y": y
                                if metric.y_min is None
                                else max(y, metric.y_min),
                            }
                        )
                        # !!!! Make multiple different ones if not sharing y-axis
                        # Also, why are the x-values not quite right?
                        y_max = max(y_max, y)
        y_max = None if metric.y_min is None or not metric.y_axes_match else y_max
        figs.append(
            edge_patching_plot(
                data,
                task_measurements,
                metric.name,
                metric.log_x,
                metric.log_y,
                metric.y_axes_match,
                token_circuit,
                y_max,
                metric.y_min,
            )
        )
        if auc_plots:
            figs.append(
                average_auc_plot(
                    task_measurements,
                    metric.log_x,
                    metric.log_y,
                    metric.y_min,
                    metric.lower_better,
                )
            )
    return tuple(figs)
