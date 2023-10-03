from collections import defaultdict
from typing import Any, Dict, Optional, Set, Tuple

import plotly.graph_objects as go
import pygraphviz as pgv
import torch as t
from IPython.display import display
from transformer_lens import HookedTransformerKeyValueCache

from auto_circuit.types import (
    DestNode,
    Edge,
    EdgeCounts,
    ExperimentType,
    SrcNode,
)
from auto_circuit.utils.graph_utils import (
    get_sorted_dest_ins,
    get_sorted_src_outs,
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
        mode = "lines+markers" if len(d) > 1 else "markers"
        fig.add_trace(go.Scatter(x=x, y=y, mode=mode, name=label))

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


def roc_plot(data: Dict[str, Set[Tuple[float, float]]]) -> go.Figure:
    fig = go.Figure()

    for label, points in data.items():
        points = sorted(points, key=lambda x: x[0])
        x, y = zip(*points)
        fig.add_trace(
            go.Scatter(x=x, y=y, mode="lines", name=label, line=dict(shape="hv"))
        )

    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_xaxes(constrain="domain")
    fig.update_layout(
        title="ROC Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        template="plotly",
    )
    return fig


def head(n: str) -> str:
    return n[:-2] if n.startswith("A") and len(n) > 4 else n
    # return n.split(".")[0] if n.startswith("A") else n


def t_fmt(x: Any, idx: Any = None) -> str:
    if type(x) == t.Tensor and (arr := x[idx].squeeze()).ndim == 1:
        return "[" + ",\n".join([f"{v:.3f}".rstrip("0") for v in arr.tolist()]) + "]"
    return str(x[idx]).lstrip("tensor(").rstrip(")") if type(x) == t.Tensor else str(x)


def draw_graph(
    model: t.nn.Module,
    input: t.Tensor,
    edge_label_override: Dict[Edge, Any] = {},
    output_dim: int = 1,
    kv_cache: Optional[HookedTransformerKeyValueCache] = None,
    display_ipython: bool = True,
    file_path: Optional[str] = None,
) -> None:
    edges: Set[Edge] = model.edges  # type: ignore
    output_idx = tuple([slice(None)] * output_dim + [-1])
    G = pgv.AGraph(strict=False, directed=True)

    src_outs: Dict[SrcNode, t.Tensor] = get_sorted_src_outs(model, input, kv_cache)
    dest_ins: Dict[DestNode, t.Tensor] = get_sorted_dest_ins(model, input, kv_cache)
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
