from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import plotly.express as px
import plotly.graph_objects as go
import pygraphviz as pgv
import torch as t
from IPython.display import display
from ordered_set import OrderedSet
from transformer_lens import HookedTransformerKeyValueCache

from auto_circuit.types import (
    DestNode,
    Edge,
    Node,
    SrcNode,
)
from auto_circuit.utils.graph_utils import (
    get_sorted_dest_ins,
    get_sorted_src_outs,
)

Y_MIN = 1e-6


def edge_patching_plot(
    data: List[Dict[str, Any]],
    metric_name: str,
    y_axes_match: bool,
    kl_max: Optional[float] = None,
) -> go.Figure:
    data = sorted(data, key=lambda x: (x["Algorithm"], x["Task"]))
    fig = px.line(
        data,
        x="X",
        y="Y",
        facet_col="Task",
        color="Algorithm",
        log_x=True,
        log_y=True,
        range_y=None if kl_max is None else [Y_MIN, kl_max * 2],
        facet_col_spacing=0.04,
    )
    fig.update_layout(
        title=f"Task Pruning: {metric_name} vs. Patched Edges",
        yaxis_title=metric_name,
        template="plotly",
        width=1300,
    )
    fig.update_yaxes(matches=None, showticklabels=True) if not y_axes_match else None
    fig.update_xaxes(matches=None, title="Patched Edges")
    return fig


def roc_plot(data: List[Dict[str, Any]]) -> go.Figure:
    data = sorted(data, key=lambda x: (x["Algorithm"], x["Task"], x["X"]))
    fig = px.scatter(data, x="X", y="Y", facet_col="Task", color="Algorithm")
    fig.update_traces(line=dict(shape="hv"), mode="lines")
    fig.update_xaxes(
        matches=None,
        title="False Positive Rate",
        scaleanchor="y",
        scaleratio=1,
        range=[-0.02, 1.02],
    )
    fig.update_yaxes(range=[-0.02, 1.02], scaleanchor="x", scaleratio=1)
    fig.update_layout(
        title="Task Pruning: ROC Curves",
        template="plotly",
        yaxis_title="True Positive Rate",
        width=1300,
    )
    return fig


def head(n: str) -> str:
    return n[:-2] if n.startswith("A") and len(n) > 5 else n


def t_fmt(x: Any, idx: Any = None, replace_line_break: str = "\n") -> str:
    if type(x) != t.Tensor:
        return str(x)
    if (arr := x[idx].squeeze()).ndim == 1:
        return f"[{', '.join([f'{v:.2f}'.rstrip('0') for v in arr.tolist()[:2]])} ...]"
    return str(x[idx]).lstrip("tensor(").rstrip(")").replace("\n", replace_line_break)


def draw_graph(
    model: t.nn.Module,
    input: t.Tensor,
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
        label = src_outs[e.src]
        if patched_edge := (e.patch_mask(model)[e.patch_idx].item() == 1.0):
            label = e.dest.module(model).patch_src_outs[e.src.idx]  # type: ignore
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


def net_viz(
    model: t.nn.Module,
    seq_edges: Set[Edge],
    input: t.Tensor,
    prune_scores: Dict[Edge, float],
    vert_interval: Tuple[float, float],
    seq_idx: Optional[int] = None,
    show_all: bool = False,
    kv_cache: Optional[HookedTransformerKeyValueCache] = None,
) -> go.Sankey:
    nodes: OrderedSet[Node] = OrderedSet(model.nodes)  # type: ignore
    seq_dim: int = model.seq_dim  # type: ignore
    seq_sl = (-1 if input.size(-1) < 5 else slice(None)) if seq_idx is None else seq_idx
    label_slice = tuple([0] + [slice(None)] * (seq_dim - 1) + [seq_sl])

    src_outs: Dict[SrcNode, t.Tensor] = get_sorted_src_outs(model, input, kv_cache)
    dest_ins: Dict[DestNode, t.Tensor] = get_sorted_dest_ins(model, input, kv_cache)
    node_ins = dict([(head(k.name), v) for k, v in dest_ins.items()])
    node_labels = defaultdict(str, node_ins)

    # Define the sankey nodes
    viz_nodes = dict([(head(n.name), n) for n in nodes])
    viz_node_head_idxs = [n.head_idx for n in viz_nodes.values()]
    n_layers = max([n.layer for n in nodes])
    n_head = max([n.head_idx for n in nodes if n.head_idx is not None])
    graph_nodes = {
        "label": [
            name
            + ("<br>In: " if node_labels[name] != "" else "")
            + (t_fmt(node_labels[name], label_slice, "<br>"))
            for name, _ in viz_nodes.items()
        ],
        "x": [(n.layer + 1) / (n_layers + 2) for n in viz_nodes.values()],
        "y": [0.5 if h is None else (h + 1) / (n_head + 2) for h in viz_node_head_idxs],
        "pad": 10,
    }

    # Define the sankey edges
    sources, targets, values, labels, colors = [], [], [], [], []
    seq_prune_score = 0.0
    for e in seq_edges:
        label = src_outs[e.src]
        if patched_edge := (e.patch_mask(model)[e.patch_idx].item() == 1.0):
            label = e.dest.module(model).patch_src_outs[e.src.idx]  # type: ignore
            seq_prune_score += abs(prune_scores[e]) if e in prune_scores else 0.0
        sources.append(list(viz_nodes).index(head(e.src.name)))
        targets.append(list(viz_nodes).index(head(e.dest.name)))
        values.append(abs(prune_scores[e]) if e in prune_scores else 0.8)
        labels.append(e.name + "<br>" + t_fmt(label, label_slice, "<br>"))
        normal_color = "rgba(0,0,0,0.0)" if show_all else "rgba(0,0,0,0.2)"
        ptch_col = f"rgba({'255,0,0' if prune_scores.get(e, 0) > 0 else '0,0,255'},0.3)"
        colors.append(ptch_col if patched_edge else normal_color)

    return go.Sankey(
        arrangement="snap",
        node=graph_nodes,
        link={
            "arrowlen": 25,
            "source": sources,
            "target": targets,
            "value": values,
            "label": labels,
            "color": colors,
        },
        domain={"y": vert_interval},
    )


def draw_seq_graph(
    model: t.nn.Module,
    input: t.Tensor,
    prune_scores: Dict[Edge, float],
    show_all: bool = True,
    kv_cache: Optional[HookedTransformerKeyValueCache] = None,
    seq_labels: Optional[List[str]] = None,
    display_ipython: bool = True,
    file_path: Optional[str] = None,
) -> None:
    nodes: Set[Node] = model.nodes  # type: ignore
    edge_dict: Dict[Optional[int], List[Edge]] = model.edge_dict  # type: ignore
    n_layers = max([n.layer for n in nodes])
    seq_len: Optional[int] = model.seq_len  # type: ignore

    # Calculate the vertical interval for each sub-diagram
    total_prune_score = sum([abs(v) for v in prune_scores.values()])
    sankey_heights: Dict[Optional[int], float] = defaultdict(float)
    for edge, score in prune_scores.items():
        sankey_heights[edge.seq_idx] += abs(score)
    for seq_idx in edge_dict:
        if seq_idx not in sankey_heights:
            sankey_heights[seq_idx] = total_prune_score / (len(edge_dict) * 2)
    margin_height: float = total_prune_score / ((n_figs := len(sankey_heights)) * 4)
    total_height = sum(sankey_heights.values()) + margin_height * (n_figs - 1)
    intervals, interval_end = {}, 0.0
    for seq_idx, height in sorted(sankey_heights.items(), key=lambda x: (x is None, x)):
        interval_start = interval_end + (margin_height if len(intervals) > 0 else 0)
        interval_end = interval_start + height
        intervals[seq_idx] = interval_start / total_height, interval_end / total_height

    # Draw the sankey for each token position
    sankeys = []
    for idx, seq_edges in edge_dict.items():
        e_set, vert = set(seq_edges), intervals[idx]
        viz = net_viz(model, e_set, input, prune_scores, vert, idx, show_all, kv_cache)
        sankeys.append(viz)

    layout = go.Layout(
        title="Patched Network",
        height=500 * len(sankeys),
        width=300 * n_layers,
        plot_bgcolor="blue",
    )
    fig = go.Figure(data=sankeys, layout=layout)
    for idx, seq_label in enumerate(seq_labels) if seq_labels else []:
        assert seq_len is not None
        y_range: Tuple[float, float] = fig.data[idx].domain["y"]  # type: ignore
        fig.add_annotation(
            x=0.05,
            y=(y_range[0] + y_range[1]) / 2,
            text=f"<b>{seq_label}</b>",
            showarrow=False,
            xref="paper",
            yref="paper",
        )
    if display_ipython:
        fig.show()
    if file_path:
        fig.write_image(file_path)
