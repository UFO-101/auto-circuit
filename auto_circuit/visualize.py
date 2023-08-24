from typing import Dict

import plotly.graph_objects as go

from auto_circuit.types import EdgeCounts, ExperimentType


def kl_vs_edges_plot(
    data: Dict[str, Dict[int, float]],
    experiment_type: ExperimentType,
    edge_counts: EdgeCounts,
) -> go.Figure:
    fig = go.Figure()

    for label, d in data.items():
        x = list(d.keys())
        y = list(d.values())
        x = [max(0.5, x_i) for x_i in x]
        y = [max(2e-5, y_i) for y_i in y]
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=label))

    fig.update_layout(
        title=f"KL Div vs. Edges: {experiment_type.input_type} input, patching \
            {experiment_type.patch_type} edges",
        xaxis_title="Edges",
        xaxis_type="log" if edge_counts == EdgeCounts.LOGARITHMIC else "linear",
        yaxis_title="KL Divergence",
        yaxis_type="log",
        template="plotly",
    )

    return fig
