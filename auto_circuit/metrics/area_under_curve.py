import math
from typing import Dict, Optional

from auto_circuit.types import (
    AlgoMeasurements,
    Measurements,
    Metric,
    MetricMeasurements,
    PruneAlgo,
    Task,
    TaskMeasurements,
)


def measurements_auc(
    points: Measurements, log_x: bool, log_y: bool, y_min: Optional[float]
) -> float:
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
    return area


def algo_measurements_auc(
    algo_measurements: AlgoMeasurements,
    log_x: bool,
    log_y: bool,
    y_min: Optional[float] = None,
) -> Dict[PruneAlgo, float]:
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
) -> Dict[Task, Dict[PruneAlgo, float]]:
    return {
        task: algo_measurements_auc(algo_measurements, log_x, log_y, y_min)
        for task, algo_measurements in task_measurements.items()
    }


def metric_measurements_auc(
    points: MetricMeasurements, log_x: bool, log_y: bool, y_min: Optional[float] = None
) -> Dict[Metric, Dict[Task, Dict[PruneAlgo, float]]]:
    return {
        metric: task_measurements_auc(task_measurements, log_x, log_y, y_min)
        for metric, task_measurements in points.items()
    }
