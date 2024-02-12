from typing import Any, Dict, List

import plotly.express as px
import plotly.graph_objects as go

from auto_circuit.prune_algos.prune_algos import PRUNE_ALGO_DICT
from auto_circuit.tasks import TASK_DICT
from auto_circuit.types import TaskMeasurements


def roc_plot(task_measurements: TaskMeasurements) -> go.Figure:
    data = []
    token_circuit = TASK_DICT[list(task_measurements.keys())[0]].token_circuit
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
                            "X": x,
                            "Y": y,
                        }
                    )
    return roc_fig(data, task_measurements)


def roc_fig(
    data: List[Dict[str, Any]], task_measurements: TaskMeasurements
) -> go.Figure:
    data = sorted(data, key=lambda x: (x["Algorithm"], x["Task"], x["X"]))
    fig = px.scatter(data, x="X", y="Y", facet_col="Task", color="Algorithm")
    fig.update_traces(line=dict(shape="hv"), mode="lines")
    task_measurements = dict(sorted(task_measurements.items(), key=lambda x: x[0]))
    for task_idx, algo_measurements in enumerate(task_measurements.values()):
        for algo_key, measurements in algo_measurements.items():
            algo = PRUNE_ALGO_DICT[algo_key]
            if len(measurements) == 1:
                x, y = measurements[0]
                fig.add_scattergl(
                    row=1,
                    col=task_idx + 1,
                    x=[x],
                    y=[y],
                    mode="markers+text",
                    text=algo.short_name if algo.short_name else algo.name,
                    textposition="middle right",
                    showlegend=task_idx == 0,
                    marker=dict(color="black", size=10, symbol="x-thin"),
                    marker_line_width=2,
                    name=f"{algo.short_name} = {algo.name}",
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
        width=365 * len(set([d["Task"] for d in data])) - 10,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.75,
            xanchor="left",
            x=0.0,
            entrywidthmode="fraction",
            entrywidth=0.3,
        ),
    )
    return fig
