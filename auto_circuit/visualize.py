from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import torch as t
from ordered_set import OrderedSet
from transformer_lens import HookedTransformerKeyValueCache

from auto_circuit.metrics.area_under_curve import task_measurements_auc
from auto_circuit.prune_algos.prune_algos import (
    PRUNE_ALGO_DICT,
    RANDOM_PRUNE_ALGO,
)
from auto_circuit.tasks import TASK_DICT
from auto_circuit.types import (
    AlgoKey,
    DestNode,
    Edge,
    Node,
    SrcNode,
    TaskKey,
    TaskMeasurements,
)
from auto_circuit.utils.graph_utils import (
    get_sorted_dest_ins,
    get_sorted_src_outs,
)
from auto_circuit.utils.misc import repo_path_to_abs_path

# Define a colorblind-friendly palette
color_palette = [
    "#377eb8",  # blue
    "#ff7f00",  # orange
    "#4daf4a",  # green
    "#f781bf",  # pink
    "#e41a1c",  # red
    "#984ea3",  # purple
    "#a65628",  # brown
    "#999999",  # grey
    "#dede00",  # yellow
]

# Create or modify a template
template = pio.templates["plotly"]
template.layout.colorway = color_palette  # type: ignore
template.layout.font.size = 19  # type: ignore

# Set the template as the default
pio.templates.default = "plotly"


def edge_patching_plot(
    data: List[Dict[str, Any]],
    task_measurements: TaskMeasurements,
    metric_name: str,
    log_x: bool,
    log_y: bool,
    y_axes_match: bool,
    token_circuit: bool,
    y_max: Optional[float],
    y_min: Optional[float],
) -> go.Figure:
    data = sorted(data, key=lambda x: (x["Algorithm"], x["Task"]))
    fig = px.line(
        data,
        x="X",
        y="Y",
        facet_col="Task",
        color="Algorithm",
        log_x=log_x,
        log_y=log_y,
        range_y=None if y_max is None else [y_min, y_max * 0.8],
        # range_y=[-45, 120],
        facet_col_spacing=0.03 if y_axes_match else 0.06,
    )
    fig.update_layout(
        # title=f"{main_title}: {metric_name} vs. Patched Edges",
        yaxis_title=metric_name,
        template="plotly",
        # width=335 * len(set([d["Task"] for d in data])) + 280,
        width=365 * len(set([d["Task"] for d in data])) - 10,
        height=600,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-1.15,
            xanchor="left",
            x=0.0,
            entrywidthmode="fraction",
            entrywidth=0.67,
        ),
    )
    task_measurements = dict(sorted(task_measurements.items(), key=lambda x: x[0]))
    for task_idx, algo_measurements in enumerate(task_measurements.values()):
        for algo_key, measurements in algo_measurements.items():
            algo = PRUNE_ALGO_DICT[algo_key]
            pos = "middle right" if algo.short_name == "GT" else "middle left"
            if len(measurements) == 1:
                x, y = measurements[0]
                fig.add_scattergl(
                    row=1,
                    col=task_idx + 1,
                    x=[x],
                    y=[y],
                    mode="markers+text",
                    text=algo.short_name if algo.short_name else algo.name,
                    textposition=pos,
                    showlegend=task_idx == 0,
                    marker=dict(color="black", size=10, symbol="x-thin"),
                    marker_line_width=2,
                    name=f"{algo.short_name} = {algo.name}",
                )

    fig.update_yaxes(matches=None, showticklabels=True) if not y_axes_match else None
    fig.update_xaxes(matches=None, title="Circuit Edges")
    return fig


def roc_plot(
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
        height=600,
        # width=335 * len(set([d["Task"] for d in data])) + 280,
        width=365 * len(set([d["Task"] for d in data])) - 10,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-1.15,
            xanchor="left",
            x=0.0,
            entrywidthmode="fraction",
            entrywidth=0.67,
        ),
    )
    return fig


def average_auc_plot(
    task_measurements: TaskMeasurements,
    metric_name: str,
    log_x: bool,
    log_y: bool,
    y_min: Optional[float],
    inverse: bool,
) -> go.Figure:
    """A bar chart of the average AUC for each algorithm across all tasks."""
    task_algo_aucs: Dict[TaskKey, Dict[AlgoKey, float]] = task_measurements_auc(
        task_measurements, log_x, log_y, y_min
    )
    data, totals = [], defaultdict(float)
    for task_key, algo_aucs in task_algo_aucs.items():
        task = TASK_DICT[task_key]
        for algo_key, auc in reversed(algo_aucs.items()):
            algo = PRUNE_ALGO_DICT[algo_key]
            normalized_auc = auc / (
                algo_aucs[RANDOM_PRUNE_ALGO.key] * len(task_algo_aucs)
            )
            if inverse:
                normalized_unit = 1 / len(task_algo_aucs)
                normalized_auc = normalized_unit + (normalized_unit - normalized_auc)
            data.append(
                {
                    "Algorithm": algo.name,
                    "Task": task.name,
                    "Normalized AUC": normalized_auc,
                }
            )
            totals[algo] += normalized_auc
    algo_count = len(set([d["Algorithm"] for d in data]))
    fig = px.bar(
        data,
        y="Algorithm",
        x="Normalized AUC",
        color="Task",
        orientation="h",
        color_discrete_sequence=color_palette[algo_count:],
    )
    fig.update_layout(
        # title=f"Normalized Area Under {metric_name} Curve",
        template="plotly",
        width=550,
        legend=dict(orientation="h", yanchor="bottom", y=-0.55, xanchor="left", x=0.0),
    )
    max_total = max(totals.values())
    fig.add_trace(
        go.Scatter(
            x=[totals[algo] + max_total * 0.2 for algo in totals],
            y=[algo.name for algo in totals.keys()],
            mode="text",
            text=[f"{totals[algo]:.2f}" for algo in totals],
            textposition="middle left",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0.02 for _ in totals],
            y=[algo.name for algo in totals.keys()],
            mode="text",
            text=[f"{algo.name}" for algo in totals.keys()],
            textposition="middle right",
            showlegend=False,
            textfont=dict(color="white"),
        )
    )
    fig.update_yaxes(visible=False, showticklabels=False)
    fig.update_xaxes(title="Normalized Inverse AUC") if inverse else None
    return fig


def head(n: str) -> str:
    return n[:-2] if n.startswith("A") and len(n) > 5 else n


def t_fmt(x: Any, idx: Any = None, replace_line_break: str = "\n") -> str:
    if type(x) != t.Tensor:
        return str(x)
    if (arr := x[idx].squeeze()).ndim == 1:
        return f"[{', '.join([f'{v:.2f}'.rstrip('0') for v in arr.tolist()[:2]])} ...]"
    return str(x[idx]).lstrip("tensor(").rstrip(")").replace("\n", replace_line_break)


def net_viz(
    model: t.nn.Module,
    seq_edges: Set[Edge],
    input: t.Tensor,
    prune_scores: Dict[Edge, float],
    vert_interval: Tuple[float, float],
    seq_idx: Optional[int] = None,
    show_prune_scores: bool = False,
    show_all_edges: bool = False,
    kv_cache: Optional[HookedTransformerKeyValueCache] = None,
) -> go.Sankey:
    nodes: OrderedSet[Node] = OrderedSet(model.nodes)  # type: ignore
    seq_dim: int = model.seq_dim  # type: ignore
    seq_sl = (-1 if input.size(-1) < 5 else slice(None)) if seq_idx is None else seq_idx
    label_slice = tuple([0] + [slice(None)] * (seq_dim - 1) + [seq_sl])

    src_outs: Dict[SrcNode, t.Tensor] = get_sorted_src_outs(model, input, kv_cache)
    dest_ins: Dict[DestNode, t.Tensor] = get_sorted_dest_ins(model, input, kv_cache)
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
    model: t.nn.Module,
    input: t.Tensor,
    prune_scores: Dict[Edge, float],
    show_prune_scores: bool = False,  # Show edges batched on mask or prune scores
    show_all_edges: bool = False,
    show_all_seq_pos: bool = False,
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
        if seq_idx not in sankey_heights and show_all_seq_pos:
            sankey_heights[seq_idx] = total_prune_score / (len(edge_dict) * 2)
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
        edge_set = set(edge_dict[seq_idx])
        viz = net_viz(
            model,
            edge_set,
            input,
            prune_scores,
            vert_interval,
            seq_idx,
            show_prune_scores,
            show_all_edges,
            kv_cache,
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
        assert seq_len is not None
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
