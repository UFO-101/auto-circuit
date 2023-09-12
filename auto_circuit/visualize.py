from collections import defaultdict
from typing import Any, Dict

import plotly.graph_objects as go
import pygraphviz as pgv
import torch as t

from auto_circuit.types import DestNode, Edge, EdgeCounts, ExperimentType, SrcNode
from auto_circuit.utils.graph_utils import (
    get_dest_ins,
    get_src_outs,
    graph_dest_nodes,
    graph_edges,
    graph_src_nodes,
)


def kl_vs_edges_plot(
    data: Dict[str, Dict[int, float]],
    experiment_type: ExperimentType,
    edge_counts: EdgeCounts,
    factorized: bool,
) -> go.Figure:
    fig = go.Figure()

    for label, d in data.items():
        x = list(d.keys())
        y = list(d.values())
        x = [max(0.5, x_i) for x_i in x]
        y = [max(2e-5, y_i) for y_i in y]
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=label))

    fig.update_layout(
        title=(
            f"Task Pruning: {experiment_type.input_type} input, patching"
            f" {experiment_type.patch_type} edges"
            f" ({'factorized' if factorized else 'unfactorized'} model)"
        ),
        xaxis_title="Edges",
        xaxis_type="log" if edge_counts == EdgeCounts.LOGARITHMIC else "linear",
        yaxis_title="KL Divergence",
        yaxis_type="log",
        template="plotly",
    )
    return fig


def parse_name(n: str) -> str:
    return n[:-2] if n.startswith("A") and len(n) > 4 else n
    # return n.split(".")[0] if n.startswith("A") else n


def draw_graph(
    model: t.nn.Module,
    factorized: bool,
    input: t.Tensor,
    edge_label_override: Dict[Edge, Any] = {},
) -> None:
    edges = graph_edges(model, factorized)
    src_nodes = graph_src_nodes(model, factorized)
    dest_nodes = graph_dest_nodes(model, factorized)
    G = pgv.AGraph(strict=False, directed=True)

    src_outs: Dict[SrcNode, t.Tensor] = get_src_outs(model, src_nodes, input)
    dest_ins: Dict[DestNode, t.Tensor] = get_dest_ins(model, dest_nodes, input)
    node_ins: Dict[str, t.Tensor] = dict([(k.name, v) for k, v in dest_ins.items()])
    node_ins = defaultdict(lambda: t.tensor([0.0]), node_ins)

    for edge in edges:
        patched_edge = False
        label = src_outs[edge.src]
        if patched_edge := (edge in edge_label_override):
            label = edge_label_override[edge]
        G.add_edge(
            edge.src.name + f"\n{str(node_ins[edge.src.name].tolist())}",
            edge.dest.name + f"\n{str(node_ins[edge.dest.name].tolist())}",
            label=str(label.tolist()) if type(label) == t.Tensor else str(label),
            color="red" if patched_edge else "black",
        )

    G.layout(prog="dot")  # neato
    G.draw("graphviz.png")
