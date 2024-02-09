from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import plotly.graph_objects as go
import torch as t
from ordered_set import OrderedSet

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
from auto_circuit.utils.misc import repo_path_to_abs_path
from auto_circuit.utils.patchable_model import PatchableModel


def head(n: str) -> str:
    return n[:-2] if n.startswith("A") and len(n) > 5 else n


def t_fmt(x: Any, idx: Any = None, replace_line_break: str = "\n") -> str:
    if type(x) != t.Tensor:
        return str(x)
    if (arr := x[idx].squeeze()).ndim == 1:
        return f"[{', '.join([f'{v:.2f}'.rstrip('0') for v in arr.tolist()[:2]])} ...]"
    return str(x[idx]).lstrip("tensor(").rstrip(")").replace("\n", replace_line_break)


def net_viz(
    model: PatchableModel,
    seq_edges: Set[Edge],
    input: t.Tensor,
    prune_scores: Dict[Edge, float],
    vert_interval: Tuple[float, float],
    seq_idx: Optional[int] = None,
    show_prune_scores: bool = False,
    show_all_edges: bool = False,
) -> go.Sankey:
    nodes: OrderedSet[Node] = OrderedSet(model.nodes)
    seq_dim: int = model.seq_dim
    seq_sl = (-1 if input.size(-1) < 5 else slice(None)) if seq_idx is None else seq_idx
    label_slice = tuple([0] + [slice(None)] * (seq_dim - 1) + [seq_sl])

    src_outs: Dict[SrcNode, t.Tensor] = get_sorted_src_outs(model, input)
    dest_ins: Dict[DestNode, t.Tensor] = get_sorted_dest_ins(model, input)
    node_ins = dict([(head(k.name), v) for k, v in dest_ins.items()])
    defaultdict(str, node_ins)

    # Define the sankey nodes
    viz_nodes = dict([(head(n.name), n) for n in nodes])
    viz_node_head_idxs = [n.head_idx for n in viz_nodes.values()]
    n_layers = max([n.layer for n in nodes])
    n_head = max([n.head_idx for n in nodes if n.head_idx is not None])
    graph_nodes = {
        "label": [
            name
            # + ("<br>In: " if node_labels[name] != "" else "")
            # + (t_fmt(node_labels[name], label_slice, "<br>"))
            for name, _ in viz_nodes.items()
        ],
        "x": [(n.layer + 1) / (n_layers + 2) for n in viz_nodes.values()],
        "y": [0.5 if h is None else (h + 1) / (n_head + 2) for h in viz_node_head_idxs],
        "pad": 10,
    }

    # Define the sankey edges
    sources, targets, values, labels, colors = [], [], [], [], []
    for e in seq_edges:
        label = src_outs[e.src]

        if show_prune_scores:
            patched_edge = e in prune_scores
        else:
            if patched_edge := (e.patch_mask(model)[e.patch_idx].item() == 1.0):
                label = e.dest.module(model).patch_src_outs[e.src.idx]  # type: ignore

        sources.append(list(viz_nodes).index(head(e.src.name)))
        targets.append(list(viz_nodes).index(head(e.dest.name)))
        values.append(
            abs(prune_scores[e])
            if e in prune_scores
            else 0.8
            if show_all_edges
            else 0.01
        )
        labels.append(e.name + "<br>" + t_fmt(label, label_slice, "<br>"))
        normal_color = "rgba(0,0,0,0.2)" if show_all_edges else "rgba(0,0,0,0.0)"
        ptch_col = f"rgba({'0,0,255' if prune_scores.get(e, 0) > 0 else '255,0,0'},0.3)"
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
    model: PatchableModel,
    input: t.Tensor,
    prune_scores: Dict[Edge, float],
    show_prune_scores: bool = False,  # Show edges batched on mask or prune scores
    show_all_edges: bool = False,
    show_all_seq_pos: bool = False,
    seq_labels: Optional[List[str]] = None,
    display_ipython: bool = True,
    file_path: Optional[str] = None,
) -> None:
    n_layers = max([n.layer for n in model.nodes])

    # Calculate the vertical interval for each sub-diagram
    total_prune_score = sum([abs(v) for v in prune_scores.values()])
    sankey_heights: Dict[Optional[int], float] = defaultdict(float)
    for edge, score in prune_scores.items():
        sankey_heights[edge.seq_idx] += abs(score)
    for seq_idx in model.edge_dict:
        if seq_idx not in sankey_heights and show_all_seq_pos:
            sankey_heights[seq_idx] = total_prune_score / (len(model.edge_dict) * 2)
    margin_height: float = total_prune_score / ((n_figs := len(sankey_heights)) * 2)
    total_height = sum(sankey_heights.values()) + margin_height * (n_figs - 1)
    intervals, interval_start = {}, total_height
    for seq_idx, height in sorted(sankey_heights.items(), key=lambda x: (x is None, x)):
        interval_end = interval_start - (margin_height if len(intervals) > 0 else 0)
        interval_start = interval_end - height
        intervals[seq_idx] = max(interval_start / total_height, 1e-6), min(
            interval_end / total_height, 1 - 1e-6
        )

    # Draw the sankey for each token position
    sankeys = []
    for seq_idx, vert_interval in intervals.items():
        edge_set = set(model.edge_dict[seq_idx])
        viz = net_viz(
            model,
            edge_set,
            input,
            prune_scores,
            vert_interval,
            seq_idx,
            show_prune_scores,
            show_all_edges,
        )
        sankeys.append(viz)

    layout = go.Layout(
        height=250 * len(sankeys),
        width=300 * n_layers,
        plot_bgcolor="blue",
    )
    fig = go.Figure(data=sankeys, layout=layout)
    for fig_idx, seq_idx in enumerate(intervals.keys()) if seq_labels else []:
        assert seq_labels is not None
        seq_label = seq_labels[seq_idx]
        assert model.seq_len is not None
        y_range: Tuple[float, float] = fig.data[fig_idx].domain["y"]  # type: ignore
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
        absolute_path: Path = repo_path_to_abs_path(file_path)
        fig.write_image(str(absolute_path))
