from typing import Dict, List, Tuple

import plotly.graph_objects as go

from auto_circuit.types import ExperimentType


def kl_vs_edges_plot(
    data: List[Tuple[str, Dict[int, float]]], experiment_type: ExperimentType
) -> go.Figure:
    fig = go.Figure()

    for label, d in data:
        x = list(d.keys())
        y = list(d.values())
        x = [max(0.2, x_i) for x_i in x]
        y = [max(2e-3, y_i) for y_i in y]
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=label))

    fig.update_layout(
        title=f"KL Div vs. Edges: {experiment_type.input_type} input, patching \
            {experiment_type.patch_type} edges",
        xaxis_title="Edges",
        xaxis_type="log",
        yaxis_title="KL Divergence",
        yaxis_type="log",
        template="plotly",
    )

    return fig
