import plotly.graph_objects as go
from plotly.subplots import make_subplots

from auto_circuit.prune_algos.prune_algos import PRUNE_ALGO_DICT
from auto_circuit.tasks import TASK_DICT
from auto_circuit.types import COLOR_PALETTE, TaskMeasurements


def roc_plot(task_measurements: TaskMeasurements) -> go.Figure:
    titles = [TASK_DICT[task_key].name for task_key in task_measurements.keys()]
    fig = make_subplots(rows=1, cols=len(task_measurements), subplot_titles=titles)
    fig.update_traces(line=dict(shape="hv"), mode="lines")
    task_measurements = dict(sorted(task_measurements.items(), key=lambda x: x[0]))
    for task_idx, (task_key, algo_measurements) in enumerate(task_measurements.items()):
        for algo_idx, (algo_key, measurements) in enumerate(algo_measurements.items()):
            algo = PRUNE_ALGO_DICT[algo_key]
            fig.add_scatter(
                row=1,
                col=task_idx + 1,
                x=[x for x, _ in measurements],
                y=[y for _, y in measurements],
                mode="markers+text" if len(measurements) == 1 else "lines",
                text=algo.short_name,
                textposition="middle right",
                showlegend=task_idx == 0,
                # marker=dict(color="black", size=10, symbol="x-thin"),
                marker_line_width=2,
                name=algo.short_name,
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
        width=365 * len(task_measurements) - 10,
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
