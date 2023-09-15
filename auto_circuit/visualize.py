from collections import defaultdict
from typing import Any, Dict, Optional

import plotly.graph_objects as go
import pygraphviz as pgv
import torch as t
from IPython.display import display

from auto_circuit.types import (
    DestNode,
    Edge,
    EdgeCounts,
    ExperimentType,
    SrcNode,
    TensorIndex,
)
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


def head(n: str) -> str:
    return n[:-2] if n.startswith("A") and len(n) > 4 else n
    # return n.split(".")[0] if n.startswith("A") else n


def t_fmt(x: Any, idx: Optional[TensorIndex] = None) -> str:
    if type(x) == t.Tensor and (arr := x[idx].squeeze()).ndim == 1:
        return "[" + ",\n".join([f"{v:.3f}".rstrip("0") for v in arr.tolist()]) + "]"
    return str(x[idx]).lstrip("tensor(").rstrip(")") if type(x) == t.Tensor else str(x)


def draw_graph(
    model: t.nn.Module,
    factorized: bool,
    input: t.Tensor,
    edge_label_override: Dict[Edge, Any] = {},
    output_idx: TensorIndex = (slice(None), -1),
    display_ipython: bool = True,
    file_path: Optional[str] = None,
) -> None:
    edges = graph_edges(model, factorized)
    src_nodes = graph_src_nodes(model, factorized)
    dest_nodes = graph_dest_nodes(model, factorized)
    G = pgv.AGraph(strict=False, directed=True)

    src_outs: Dict[SrcNode, t.Tensor] = get_src_outs(model, src_nodes, input)
    dest_ins: Dict[DestNode, t.Tensor] = get_dest_ins(model, dest_nodes, input)
    node_ins: Dict[str, t.Tensor] = dict(
        [(head(k.name), v) for k, v in dest_ins.items()]
    )
    node_lbls = defaultdict(str, node_ins)

    for e in edges:
        patched_edge = False
        label = src_outs[e.src]
        if patched_edge := (e in edge_label_override):
            label = edge_label_override[e]
        G.add_edge(
            head(e.src.name) + f"\n{t_fmt(node_lbls[head(e.src.name)], output_idx)}",
            head(e.dest.name) + f"\n{t_fmt(node_lbls[head(e.dest.name)], output_idx)}",
            label=t_fmt(label, output_idx),
            color="red" if patched_edge else "black",
        )

    if display_ipython or file_path:
        G.node_attr.update(fontsize="7")
        G.edge_attr.update(fontsize="7")
        G.layout(prog="dot")  # neato
    if display_ipython:
        display(G)
    if file_path:
        G.draw(file_path)
