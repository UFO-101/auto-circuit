from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

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
    Node,
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
    y_axis_title: str,
    factorized: bool,
    log_y_axis: bool = True,
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
        yaxis_title=y_axis_title,
        yaxis_type="log" if log_y_axis else "linear",
        template="plotly",
    )
    return fig


def roc_plot(title: str, data: Dict[str, Set[Tuple[float, float]]]) -> go.Figure:
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
        title="ROC Curve: " + title,
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        template="plotly",
    )
    return fig


def head(n: str) -> str:
    return n[:-2] if n.startswith("A") and len(n) > 5 else n
    # return n.split(".")[0] if n.startswith("A") else n


def t_fmt(x: Any, idx: Any = None) -> str:
    if type(x) == t.Tensor and (arr := x[idx].squeeze()).ndim == 1:
        return "[" + ", ".join([f"{v:.2f}".rstrip("0") for v in arr.tolist()[:2]]) + "]"
    return str(x[idx]).lstrip("tensor(").rstrip(")") if type(x) == t.Tensor else str(x)


def draw_graph(
    model: t.nn.Module,
    input: t.Tensor,
    edge_label_override: Dict[Edge, Any] = {},
    kv_cache: Optional[HookedTransformerKeyValueCache] = None,
    display_ipython: bool = True,
    file_path: Optional[str] = None,
) -> None:
    edges: Set[Edge] = model.edges  # type: ignore
    output_idx = slice(None)
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


def model_network_sankey(
    model: t.nn.Module,
    nodes: Set[Node],
    edges: Set[Edge],
    input: t.Tensor,
    seq_idx: Optional[int] = None,
    seq_len: Optional[int] = None,
    edge_label_override: Dict[Edge, Any] = {},
    patched_edge_only: bool = False,
    kv_cache: Optional[HookedTransformerKeyValueCache] = None,
) -> go.Sankey:
    seq_dim: int = model.seq_dim  # type: ignore
    assert (seq_idx is None) == (seq_len is None)
    if seq_idx is None:
        label_slice = (
            (0,)
            if input.size(0) < 5
            else tuple([0] + [slice(None)] * (seq_dim - 1) + [-1])
        )
    else:
        label_slice = tuple([0] + [slice(None)] * (seq_dim - 1) + [seq_idx])

    src_outs: Dict[SrcNode, t.Tensor] = get_sorted_src_outs(model, input, kv_cache)
    dest_ins: Dict[DestNode, t.Tensor] = get_sorted_dest_ins(model, input, kv_cache)
    node_ins: Dict[str, t.Tensor] = dict(
        [(head(k.name), v) for k, v in dest_ins.items()]
    )
    node_labels = defaultdict(str, node_ins)

    node_info: Set[Tuple[str, int]] = set([(head(n.name), n.lyr) for n in nodes])
    graph_node_dict = dict([(n[0], i) for i, n in enumerate(node_info)])
    n_lyrs = max([n.lyr for n in nodes])
    max([n.head_idx for n in nodes if n.head_idx is not None])
    graph_nodes = {
        "label": [
            n[0]
            + ("<br>Input:<br>" if node_labels[n[0]] != "" else "")
            + (t_fmt(node_labels[n[0]], label_slice))
            for n in node_info
        ],
        "x": [(n[1] + 1) / (n_lyrs + 2) for n in node_info],
        # Set y so fixed works?
        # "y": [(n.head_idx + 1) / (n_heads + 2)
        # if n.head_idx is not None else 0.5 for n in nodes],
        "pad": 10,
    }
    sources, targets, labels, colors = [], [], [], []
    for e in edges:
        patched_edge = False
        label = src_outs[e.src]
        if patched_edge := (e in edge_label_override):
            label = edge_label_override[e]
        # if patched_edge or not patched_edge_only:
        sources.append(graph_node_dict[head(e.src.name)])
        targets.append(graph_node_dict[head(e.dest.name)])
        labels.append(e.name + "<br>" + t_fmt(label, label_slice))
        normal_color = "rgba(0,0,0,0.0)" if patched_edge_only else "rgba(0,0,0,0.2)"
        colors.append("rgba(255,0,0,0.4)" if patched_edge else normal_color)

    if seq_idx is not None:
        assert seq_len is not None
        margin = 1 / (seq_len * 10)
        vert = [(seq_idx / seq_len) + margin, ((seq_idx + 1) / seq_len) - margin]
    else:
        vert = [0, 1]

    return go.Sankey(
        arrangement="fixed",
        node=graph_nodes,
        link={
            "arrowlen": 25,
            "source": sources,
            "target": targets,
            # "value": [1, 2, 1, 1, 1, 1, 1, 2]
            "value": [1 for _ in range(len(sources))],
            "label": labels,
            "color": colors,
        },
        domain={
            "y": vert,
        },
    )


def draw_seq_graph(
    model: t.nn.Module,
    input: t.Tensor,
    edge_label_override: Dict[Edge, Any] = {},
    patched_edge_only: bool = False,
    kv_cache: Optional[HookedTransformerKeyValueCache] = None,
    display_ipython: bool = True,
    file_path: Optional[str] = None,
) -> None:
    nodes: Set[Node] = model.nodes  # type: ignore
    # nodes_l = list(sorted(nodes, key=lambda x: (x.lyr, x.head_idx)))
    edge_dict: Dict[Optional[int], List[Edge]] = model.edge_dict  # type: ignore
    n_lyrs = max([n.lyr for n in nodes])
    seq_len: Optional[int] = model.seq_len  # type: ignore

    sankeys = []
    for seq_idx, seq_edges in edge_dict.items():
        sankeys.append(
            model_network_sankey(
                model,
                nodes,
                set(seq_edges),
                input,
                seq_idx,
                seq_len,
                edge_label_override,
                patched_edge_only,
                kv_cache,
            )
        )

    layout = go.Layout(
        title="Patched Network",
        height=700 * len(sankeys),
        width=300 * n_lyrs,
    )
    fig = go.Figure(data=sankeys, layout=layout)
    fig.show()
