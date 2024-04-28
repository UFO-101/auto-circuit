from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import plotly.graph_objects as go
import torch as t
from ordered_set import OrderedSet

from auto_circuit.types import (
    Edge,
    Node,
    PruneScores,
)
from auto_circuit.utils.misc import repo_path_to_abs_path
from auto_circuit.utils.patchable_model import PatchableModel


def node_name(n: str) -> str:
    if n == "Resid End":
        return ""
    elif n == "Resid Start":
        return "Embed"
    else:
        return n[:-2] if n[-1] in ["Q", "K", "V"] else n


def t_fmt(x: Any, seq_dim: int, seq_idx: Optional[int], line_break: str = "\n") -> str:
    if type(x) != t.Tensor:
        return str(x)
    seq_sl = (-1 if x.size(-1) < 5 else slice(None)) if seq_idx is None else seq_idx
    idx = tuple([0] + [slice(None)] * (seq_dim - 1) + [seq_sl])
    if (arr := x[idx].squeeze()).ndim == 1:
        return f"[{', '.join([f'{v:.2f}'.rstrip('0') for v in arr.tolist()[:2]])} ...]"
    return str(x[idx]).lstrip("tensor(").rstrip(")").replace("\n", line_break)


def net_viz(
    model: PatchableModel,
    seq_edges: Set[Edge],
    prune_scores: Optional[PruneScores],
    vert_interval: Tuple[float, float],
    seq_idx: Optional[int] = None,
    show_all_edges: bool = False,
) -> go.Sankey:
    """
    Draw the sankey diagram for a single token position.
    If `prune_scores` is `None`, the diagram will show the current activations and edge
    scores of the model. If `prune_scores` is provided, the diagram will use these edge
    scores and won't show activations.

    Args:
        model: The model to visualize.
        seq_edges: The edges to visualize. This should be the edges at a single token
            position if `model.seq_len` is not `None`. Otherwise, this should be all the
            edges in the model.
        prune_scores: The edge scores to use for the visualization. If `None`, the
            current activations and mask values of the model will be visualized instead.
        vert_interval: The vertical interval to place the diagram in the figure. Must
            be in the range `(0, 1)`. This is used by
            [`draw_seq_graph`][auto_circuit.visualize.draw_seq_graph] to place the
            diagrams for each token position in the figure. If you are using this
            function to create a standalone diagram, you can set this to `(0, 1)`.
        seq_idx: The token position being visualized, this is used to get the correct
            slice of activations (if `prune_scores` is `None`) to label the edges.
        show_all_edges: If `True`, all edges will be shown, even if their edge score is
            close to zero. If `False`, only edges with a non-zero edge score will be
            shown.

    Returns:
        The sankey diagram for the given token position.

    Note:
        This is a lower level function, it is generally recommended to use
        [`draw_seq_graph`][auto_circuit.visualize.draw_seq_graph] instead.

    """
    nodes: OrderedSet[Node] = OrderedSet(model.nodes)

    # Define the sankey nodes
    viz_nodes: Dict[str, Node] = dict([(node_name(n.name), n) for n in nodes])
    node_idxs: Dict[str, int] = dict([(n, i) for i, n in enumerate(viz_nodes.keys())])
    lyr_nodes: Dict[int, List[str]] = defaultdict(list)
    for n in viz_nodes.values():
        lyr_nodes[n.layer].append(n.name)
    graph_nodes = {"label": ["" for _, _ in viz_nodes.items()]}

    # Define the sankey edges
    sources, targets, values, labels, colors = [], [], [], [], []
    included_layer_nodes: Dict[int, List[str]] = defaultdict(list)
    for e in seq_edges:
        if prune_scores is None:
            no_edge_score_error = "Visualization requires patch mode or PruneScores."
            assert e.dest.module(model).curr_src_outs is not None, no_edge_score_error
            edge_score = e.patch_mask(model).data[e.patch_idx].item()
            if edge_score == 1.0:  # Show the patched edge activation
                lbl = e.dest.module(model).patch_src_outs[e.src.src_idx]
            else:
                lbl = e.dest.module(model).curr_src_outs[e.src.src_idx]
        else:
            edge_score = prune_scores[e.dest.module_name][e.patch_idx].item()
            lbl = None

        if edge_score < 1e-6 and not show_all_edges:
            continue

        sources.append(node_idxs[node_name(e.src.name)])
        graph_nodes["label"][node_idxs[node_name(e.src.name)]] = node_name(e.src.name)
        targets.append(node_idxs[node_name(e.dest.name)])
        graph_nodes["label"][node_idxs[node_name(e.dest.name)]] = node_name(e.dest.name)
        values.append(0.8 if prune_scores is None else edge_score)
        lbl = t_fmt(lbl, model.seq_dim, seq_idx, "<br>")
        lbl = e.name + "<br>" + lbl + f"<br>{edge_score:.2f}"
        labels.append(lbl)
        if edge_score == 0:
            edge_color = "rgba(0,0,0,0.1)"
        elif edge_score > 0:
            edge_color = "rgba(0,0,255,0.3)"
        else:
            edge_color = "rgba(255,0,0,0.3)"
        colors.append(edge_color)
        included_layer_nodes[e.src.layer].append(e.src.name)
        included_layer_nodes[e.dest.layer].append(e.dest.name)

    # # Add ghost edges to horizontally align nodes to the correct layer
    for i in lyr_nodes.keys():
        if i not in included_layer_nodes:
            included_layer_nodes[i] = [lyr_nodes[i][0]]
    ordered_lyr_nodes = [nodes for _, nodes in sorted(included_layer_nodes.items())]
    ghost_edge_val = 1e-6
    for lyr_1_nodes, lyr_2_nodes in zip(ordered_lyr_nodes[:-1], ordered_lyr_nodes[1:]):
        first_lyr_1_node = lyr_1_nodes[0]
        first_lyr_2_node = lyr_2_nodes[0]
        for lyr_1_node in lyr_1_nodes:
            sources.append(node_idxs[node_name(lyr_1_node)])
            targets.append(node_idxs[node_name(first_lyr_2_node)])
            values.append(ghost_edge_val)
            labels.append("")
            colors.append("rgba(0,255,0,0.0)")
        for lyr_2_node in lyr_2_nodes:
            sources.append(node_idxs[node_name(first_lyr_1_node)])
            targets.append(node_idxs[node_name(lyr_2_node)])
            values.append(ghost_edge_val)
            labels.append("")
            colors.append("rgba(0,255,0,0.0)")

    return go.Sankey(
        arrangement="perpendicular",
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
    prune_scores: Optional[PruneScores] = None,
    show_all_edges: bool = False,
    show_all_seq_pos: bool = False,
    seq_labels: Optional[List[str]] = None,
    display_ipython: bool = True,
    file_path: Optional[str] = None,
) -> None:
    """
    Draw the sankey for all token positions in a
    [`PatchableModel`][auto_circuit.utils.patchable_model.PatchableModel] (drawn
    separately for each token position if the model has a `seq_len`).

    If `prune_scores` is `None`, the diagram will show the current activations and mask
    values of the model. If `prune_scores` is provided, the diagram will use these edge
    scores and won't show activations.

    The mask values or `prune_scores` are used to set the width of each edge.

    Args:
        model: The model to visualize.
        prune_scores: The edge scores to use for the visualization. If `None`, the
            current activations and mask values of the model will be visualized instead.
        show_all_edges: If `True`, all edges will be shown, even if their edge score is
            close to zero. If `False`, only edges with a non-zero edge score will be
            shown.
        show_all_seq_pos: If `True`, the diagram will show all token positions, even if
            they have no non-zero edge values. If `False`, only token positions with
            non-zero edge values will be shown.
        seq_labels: The labels for each token position.
        display_ipython: If `True`, the diagram will be displayed in the current
            ipython environment.
        file_path: If provided, the diagram will be saved to this file path. The file
            extension determines the format of the saved image.
    """
    n_layers = max([n.layer for n in model.nodes])
    seq_len = model.seq_len or 1

    # Calculate the vertical interval for each sub-diagram
    if prune_scores is None:
        edge_scores = model.current_patch_masks_as_prune_scores()
    else:
        edge_scores = prune_scores
    total_ps = max(sum([v.abs().sum().item() for v in edge_scores.values()]), 1e-2)
    if seq_len > 1:
        sankey_heights: Dict[Optional[int], float] = {}
        for patch_mask in edge_scores.values():
            ps_seq_tots = patch_mask.abs().sum(dim=list(range(1, patch_mask.ndim)))
            for seq_idx, ps_seq_tot in enumerate(ps_seq_tots):
                if ps_seq_tot > 0 or show_all_seq_pos:
                    if seq_idx not in sankey_heights:
                        sankey_heights[seq_idx] = 0
                    sankey_heights[seq_idx] += ps_seq_tot.item()

        for seq_idx in model.edge_dict.keys():
            min_height = total_ps / (len(model.edge_dict) * 2)
            if show_all_seq_pos:
                sankey_heights[seq_idx] = max(sankey_heights[seq_idx], min_height)
        margin_height: float = total_ps / ((n_figs := len(sankey_heights)) * 2)
        total_height = sum(sankey_heights.values()) + margin_height * (n_figs - 1)
        intervals, interval_start = {}, total_height
        for seq_idx, height in sorted(
            sankey_heights.items(), key=lambda x: (x is None, x)
        ):
            interval_end = interval_start - (margin_height if len(intervals) > 0 else 0)
            interval_start = interval_end - height
            intervals[seq_idx] = max(interval_start / total_height, 1e-6), min(
                interval_end / total_height, 1 - 1e-6
            )
    else:
        intervals = {list(model.edge_dict.keys())[0]: (0, 1)}

    # Draw the sankey for each token position
    sankeys = []
    for seq_idx, vert_interval in intervals.items():
        edge_set = set(model.edge_dict[seq_idx])
        viz = net_viz(
            model=model,
            seq_edges=edge_set,
            prune_scores=prune_scores,
            vert_interval=vert_interval,
            seq_idx=seq_idx,
            show_all_edges=show_all_edges,
        )
        sankeys.append(viz)

    layout = go.Layout(
        height=max(250 * len(sankeys), 400),
        width=max(110 * n_layers, 600),
        plot_bgcolor="blue",
    )
    fig = go.Figure(data=sankeys, layout=layout)
    for fig_idx, seq_idx in enumerate(intervals.keys()) if seq_labels else []:
        assert seq_labels is not None
        seq_label = "All tokens" if seq_idx is None else seq_labels[seq_idx]
        y_range: Tuple[float, float] = fig.data[fig_idx].domain["y"]  # type: ignore
        fig.add_annotation(
            x=-0.17,
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
