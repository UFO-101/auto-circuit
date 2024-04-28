from collections import defaultdict
from typing import Dict

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from auto_circuit.prune_algos.prune_algos import PRUNE_ALGO_DICT
from auto_circuit.tasks import TASK_DICT
from auto_circuit.types import (
    COLOR_PALETTE,
    Measurements,
    TaskMeasurements,
)


def task_roc_plot(task_measurements: TaskMeasurements) -> go.Figure:
    """
    Wrapper that takes the output of
    [`measure_roc`][auto_circuit.metrics.official_circuits.measure_roc.measure_roc],
    extracts the names of the tasks and algorithms, and plots the ROC curves
    using [`roc_plot`][auto_circuit.metrics.official_circuits.roc_plot.roc_plot].
    """
    task_algo_name_measurements: Dict[str, Dict[str, Measurements]] = defaultdict(dict)
    for task_key, algo_measurements in task_measurements.items():
        task_name = TASK_DICT[task_key].name
        for algo_key, measurements in algo_measurements.items():
            algo_name = PRUNE_ALGO_DICT[algo_key].short_name
            task_algo_name_measurements[task_name][algo_name] = measurements
    return roc_plot(task_algo_name_measurements)


def roc_plot(
    taskname_measurements: Dict[str, Dict[str, Measurements]],
    variable_width: bool = False,
) -> go.Figure:
    """
    Plots the pessimistic Receiver Operating Characteristic (ROC) curve for a nested
    dictionary of measurements. The outer dictionary has keys that are the task names
    and the inner dictionary has keys that are the algorithm names.

    Args:
        taskname_measurements: A nested dictionary with keys corresponding to task names
            (outer), algorithm names (inner), and values corresponding to the points of
            the ROC curve.
        variable_width: If True, the lines corresponding to different to different
            algorithms will have different widths. This helps distinguish overlapping
            lines.

    Returns:
        A plotly figure.
    """
    titles = list(taskname_measurements.keys())
    fig = make_subplots(rows=1, cols=len(taskname_measurements), subplot_titles=titles)
    fig.update_traces(line=dict(shape="hv"), mode="lines")
    taskname_measurements = dict(
        sorted(taskname_measurements.items(), key=lambda x: x[0])
    )
    for task_idx, (task_key, algo_measurements) in enumerate(
        taskname_measurements.items()
    ):
        for algo_idx, (algo_name, measurements) in enumerate(algo_measurements.items()):
            width_delta = 8
            max_width = (width_delta / 2) + (len(algo_measurements) - 1) * width_delta
            line_width = max_width - algo_idx * width_delta
            fig.add_scatter(
                row=1,
                col=task_idx + 1,
                x=[x for x, _ in measurements],
                y=[y for _, y in measurements],
                mode="markers+text" if len(measurements) == 1 else "lines",
                text=algo_name,
                line=dict(width=line_width if variable_width else 2),
                textposition="middle right",
                showlegend=task_idx == 0,
                # marker=dict(color="black", size=10, symbol="x-thin"),
                marker_line_width=2,
                name=algo_name,
                marker_color=COLOR_PALETTE[algo_idx],
            )
    fig.update_xaxes(
        matches=None,
        title="False Positive Rate",
        scaleanchor="y",
        scaleratio=1,
        range=[-0.02, 1.02],
    )
    fig.update_yaxes(range=[-0.02, 1.02], scaleanchor="x", scaleratio=1)
    fig.update_layout(
        # title=f"{main_title}: ROC Curves",
        template="plotly",
        yaxis_title="True Positive Rate",
        height=500,
        # width=335 * len(set([d["Task"] for d in data])) + 280,
        width=365 * len(taskname_measurements) - 10,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.6,
            xanchor="left",
            # x=0.0,
            entrywidthmode="fraction",
            entrywidth=0.6,
        ),
    )
    return fig
