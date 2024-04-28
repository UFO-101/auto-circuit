import math
from collections import defaultdict
from typing import Dict, Optional

import plotly.express as px
import plotly.graph_objects as go

from auto_circuit.prune_algos.prune_algos import PRUNE_ALGO_DICT, RANDOM_PRUNE_ALGO
from auto_circuit.tasks import TASK_DICT
from auto_circuit.types import (
    COLOR_PALETTE,
    AlgoKey,
    AlgoMeasurements,
    Measurements,
    PruneMetricKey,
    PruneMetricMeasurements,
    TaskKey,
    TaskMeasurements,
)


def measurements_auc(
    points: Measurements,
    log_x: bool,
    log_y: bool,
    y_min: Optional[float],
    eps: float = 1e-3,
) -> float:
    """
    Calculate the area under the curve of a set of points.

    Args:
        points: A list of (x, y) points.
        log_x: Whether to log the x values. All values must be `>= 0`. `0` values are
            mapped to `0`.
        log_y: Whether to log the y values. (See `y_min`.)
        y_min: Must be provided if `log_y` is `True` and must be greater than zero. If
            `log_y` is `True`, the y values are all set to `max(y, y_min)`.
        eps: A small value to ensure the returned area is greater than zero.
    """
    points = sorted(points, key=lambda x: x[0])
    assert points[0][0] == 0
    y_baseline = points[0][1]
    area = 0.0
    for i in range(1, len(points)):
        x1, y1 = points[i - 1]
        x2, y2 = points[i]
        assert x1 <= x2
        if log_y:
            assert y_min is not None
            y1 = max(y1, y_min)
            y2 = max(y2, y_min)
        if log_x and (x1 == 0 or x2 == 0):
            continue
        if log_x:
            assert x1 > 0 and x2 > 0
            x1 = math.log(x1, 10)
            x2 = math.log(x2, 10)
        if log_y:
            assert y_min is not None
            y1 = math.log(y1, 10) - math.log(y_min, 10)
            y2 = math.log(y2, 10) - math.log(y_min, 10)
        height_1, height_2 = y1 - y_baseline, y2 - y_baseline
        area += (x2 - x1) * (height_1 + height_2) / 2.0
    return max(area, eps)


def algo_measurements_auc(
    algo_measurements: AlgoMeasurements,
    log_x: bool,
    log_y: bool,
    y_min: Optional[float] = None,
) -> Dict[AlgoKey, float]:
    """
    Wrapper that runs
    [`measurements_auc`][auto_circuit.metrics.area_under_curve.measurements_auc] on each
    algorithm's measurements.
    """
    algo_measurements_auc = {}
    for algo, measurements in algo_measurements.items():
        if len(measurements) > 1:
            algo_measurements_auc[algo] = measurements_auc(
                measurements, log_x, log_y, y_min
            )
    return algo_measurements_auc


def task_measurements_auc(
    task_measurements: TaskMeasurements,
    log_x: bool,
    log_y: bool,
    y_min: Optional[float] = None,
) -> Dict[TaskKey, Dict[AlgoKey, float]]:
    """
    Wrapper that runs
    [`measurements_auc`][auto_circuit.metrics.area_under_curve.measurements_auc] for
    each task and algorithm.
    """
    return {
        task_key: algo_measurements_auc(algo_measurements, log_x, log_y, y_min)
        for task_key, algo_measurements in task_measurements.items()
    }


def metric_measurements_auc(
    points: PruneMetricMeasurements,
    log_x: bool,
    log_y: bool,
    y_min: Optional[float] = None,
) -> Dict[PruneMetricKey, Dict[TaskKey, Dict[AlgoKey, float]]]:
    """
    Wrapper that runs
    [`measurements_auc`][auto_circuit.metrics.area_under_curve.measurements_auc] for
    each metric, task, and algorithm.
    """
    return {
        metric_key: task_measurements_auc(task_measurements, log_x, log_y, y_min)
        for metric_key, task_measurements in points.items()
    }


def average_auc_plot(
    task_measurements: TaskMeasurements,
    log_x: bool,
    log_y: bool,
    y_min: Optional[float],
    inverse: bool,
) -> go.Figure:
    """
    A bar chart of the average AUC for each algorithm across all tasks.
    See [`measurements_auc`][auto_circuit.metrics.area_under_curve.measurements_auc].

    Returns:
        A Plotly figure.
    """
    task_algo_aucs: Dict[TaskKey, Dict[AlgoKey, float]] = task_measurements_auc(
        task_measurements, log_x, log_y, y_min
    )
    data, totals = [], defaultdict(float)
    for task_key, algo_aucs in task_algo_aucs.items():
        task = TASK_DICT[task_key]
        for algo_key, auc in reversed(algo_aucs.items()):
            algo = PRUNE_ALGO_DICT[algo_key]
            normalized_auc = auc / (
                algo_aucs[RANDOM_PRUNE_ALGO.key] * len(task_algo_aucs)
            )
            if inverse:
                normalized_unit = 1 / len(task_algo_aucs)
                normalized_auc = normalized_unit + (normalized_unit - normalized_auc)
            data.append(
                {
                    "Algorithm": algo.name,
                    "Task": task.name,
                    "Normalized AUC": normalized_auc,
                }
            )
            totals[algo] += normalized_auc
    algo_count = len(set([d["Algorithm"] for d in data]))
    fig = px.bar(
        data,
        y="Algorithm",
        x="Normalized AUC",
        color="Task",
        orientation="h",
        color_discrete_sequence=COLOR_PALETTE[algo_count:],
    )
    fig.update_layout(
        # title=f"Normalized Area Under {metric_name} Curve",
        template="plotly",
        width=550,
        legend=dict(orientation="h", yanchor="bottom", y=-0.55, xanchor="left", x=0.0),
    )
    max_total = max(totals.values())
    fig.add_trace(
        go.Scatter(
            x=[totals[algo] + max_total * 0.2 for algo in totals],
            y=[algo.name for algo in totals.keys()],
            mode="text",
            text=[f"{totals[algo]:.2f}" for algo in totals],
            textposition="middle left",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0.02 for _ in totals],
            y=[algo.name for algo in totals.keys()],
            mode="text",
            text=[f"{algo.name}" for algo in totals.keys()],
            textposition="middle right",
            showlegend=False,
            textfont=dict(color="white"),
        )
    )
    fig.update_yaxes(visible=False, showticklabels=False)
    fig.update_xaxes(title="Normalized Inverse AUC") if inverse else None
    return fig
