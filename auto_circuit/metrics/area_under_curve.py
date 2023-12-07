import math
from typing import Dict

from auto_circuit.types import (
    Y_MIN,
    AlgoMeasurements,
    Measurements,
    MetricMeasurements,
    TaskMeasurements,
)


def measurements_auc(points: Measurements, log_x: bool, log_y: bool) -> float:
    points = sorted(points, key=lambda x: x[0])
    area = 0.0
    for i in range(1, len(points)):
        x1, y1 = points[i - 1]
        x2, y2 = points[i]
        if log_y:
            y1 = max(y1, Y_MIN)
            y2 = max(y2, Y_MIN)
        if log_x and (x1 == 0 or x2 == 0):
            continue
        if log_x:
            assert x1 > 0 and x2 > 0
            x1 = math.log(x1, 10)
            x2 = math.log(x2, 10)
        if log_y:
            y1 = math.log(y1, 10) - math.log(Y_MIN, 10)
            y2 = math.log(y2, 10) - math.log(Y_MIN, 10)
        area += (x2 - x1) * (y1 + y2) / 2.0
    return area


def algo_measurements_auc(
    algo_measurements: AlgoMeasurements, log_x: bool, log_y: bool
) -> Dict[str, float]:
    return {
        algo: measurements_auc(measurements, log_x, log_y)
        for algo, measurements in algo_measurements.items()
    }


def task_measurements_auc(
    task_measurements: TaskMeasurements, log_x: bool, log_y: bool
) -> Dict[str, Dict[str, float]]:
    return {
        task: algo_measurements_auc(algo_measurements, log_x, log_y)
        for task, algo_measurements in task_measurements.items()
    }


def metric_measurements_auc(
    points: MetricMeasurements, log_x: bool, log_y: bool
) -> Dict[str, Dict[str, Dict[str, float]]]:
    return {
        metric: task_measurements_auc(task_measurements, log_x, log_y)
        for metric, task_measurements in points.items()
    }
