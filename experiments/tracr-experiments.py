#%%
from collections import defaultdict
from copy import deepcopy
from functools import partial
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import torch as t
from plotly.subplots import make_subplots

from auto_circuit.data import load_datasets_from_json
from auto_circuit.metrics.official_circuits.circuits.tracr.reverse_official import (
    tracr_reverse_acdc_edges,
    tracr_reverse_true_edges,
)
from auto_circuit.metrics.official_circuits.circuits.tracr.xproportion_official import (
    tracr_xproportion_acdc_edges,
    tracr_xproportion_official_edges,
)
from auto_circuit.metrics.official_circuits.measure_roc import measure_task_roc
from auto_circuit.model_utils.tracr.tracr_models import TRACR_TASK_KEY, get_tracr_model
from auto_circuit.prune_algos.ACDC import acdc_prune_scores
from auto_circuit.prune_algos.mask_gradient import mask_gradient_prune_scores
from auto_circuit.prune_algos.prune_algos import PruneAlgo
from auto_circuit.prune_algos.subnetwork_probing import subnetwork_probing_prune_scores
from auto_circuit.types import (
    COLOR_PALETTE,
    AlgoKey,
    Edge,
    Measurements,
    PruneScores,
)
from auto_circuit.utils.custom_tqdm import tqdm
from auto_circuit.utils.graph_utils import (
    patchable_model,
)
from auto_circuit.utils.misc import repo_path_to_abs_path
from auto_circuit.utils.patchable_model import PatchableModel

#%%

acdc_algo = PruneAlgo(
    key="ACDC",
    name="ACDC",
    func=partial(
        acdc_prune_scores,
        tao_exps=list(range(-25, -1)),
        tao_bases=[1, 3, 5, 7, 9],
        faithfulness_target="kl_div",
    ),
)
acdc_algo_mse = PruneAlgo(
    key="ACDC",
    name="ACDC",
    func=partial(
        acdc_prune_scores,
        tao_exps=list(range(-25, -1)),
        tao_bases=[1, 3, 5, 7, 9],
        faithfulness_target="mse",
    ),
)
sp_algo = PruneAlgo(
    key="Subnetwork Probing",
    name="Subnetwork Probing",
    _short_name="SP",
    func=partial(
        subnetwork_probing_prune_scores,
        learning_rate=0.1,
        epochs=500,
        regularize_lambda=1e-3,
        mask_fn="hard_concrete",
        show_train_graph=True,
    ),
)
sp_algo_mse = PruneAlgo(
    key="Subnetwork Probing",
    name="Subnetwork Probing",
    _short_name="SP",
    func=partial(
        subnetwork_probing_prune_scores,
        learning_rate=0.1,
        epochs=500,
        regularize_lambda=1e-3,
        mask_fn="hard_concrete",
        show_train_graph=True,
        faithfulness_target="mse",
    ),
)
hisp_algo_mse = PruneAlgo(
    key="Head Importance Scoring",
    name="Head Importance Scoring",
    _short_name="HISP",
    func=partial(
        mask_gradient_prune_scores,
        grad_function="logit",
        answer_function="mse",
        mask_val=0.0,
    ),
)
hisp_algo = PruneAlgo(
    key="Head Importance Scoring",
    name="Head Importance Scoring",
    _short_name="HISP",
    func=partial(
        mask_gradient_prune_scores,
        grad_function="logit",
        answer_function="avg_diff",
        mask_val=0.0,
    ),
)

TRACR_TASK_KEYS: Tuple[TRACR_TASK_KEY, ...] = ("reverse", "xproportion")
# TRACR_TASK_KEYS: Tuple[TRACR_TASK_KEY, ...] = ("xproportion",)
ACDC_EDGES: Tuple[bool, ...] = (True, False)
EDGE_DISCOVERY: Tuple[bool, ...] = (True, False)
# ACDC_EDGES: Tuple[bool, ...] = (False,)
xproportion_algos: List[PruneAlgo] = [acdc_algo_mse, sp_algo_mse, hisp_algo_mse]
# xproportion_algos: List[PruneAlgo] = [hisp_algo_mse]
reverse_algos: List[PruneAlgo] = [acdc_algo, sp_algo, hisp_algo]
tracr_task_algos: Dict[TRACR_TASK_KEY, List[PruneAlgo]] = {
    "reverse": reverse_algos,
    "xproportion": xproportion_algos,
}

acdc_circ_results: Dict[
    bool, Dict[TRACR_TASK_KEY, Dict[AlgoKey, Measurements]]
] = defaultdict(lambda: defaultdict(dict))
my_circ_results: Dict[
    bool, Dict[TRACR_TASK_KEY, Dict[AlgoKey, Measurements]]
] = defaultdict(lambda: defaultdict(dict))

device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")


def node_ps_to_edge_ps(
    fctrzd_model: PatchableModel,
    unfctrzd_model: PatchableModel,
    fctrzd_ps: PruneScores,
    unfctrzd_ps: PruneScores,
) -> PruneScores:
    unfctrzd_src_scrs: Dict[str, t.Tensor] = {}
    for edge in unfctrzd_model.edges:
        unfctrzd_src_scrs[edge.src.name] = unfctrzd_ps[edge.dest.module_name][
            edge.patch_idx
        ]

    unfctrzd_edge_ps = deepcopy(fctrzd_ps)
    for edge in fctrzd_model.edges:
        node_scores: List[Optional[t.Tensor]] = [None, None]
        for idx, node_name in enumerate([edge.src.name, edge.dest.name]):
            if node_name == "Resid End":
                continue
            if node_name[-1] in "QKV":
                node_name = node_name[:-2]
            node_scores[idx] = unfctrzd_src_scrs[node_name]
        assert not (node_scores[0] is None and node_scores[1] is None)
        edge_score = t.min(t.stack([scr for scr in node_scores if scr is not None]))
        unfctrzd_edge_ps[edge.dest.module_name][edge.patch_idx] = edge_score
    return unfctrzd_edge_ps


for acdc_edges in tqdm(ACDC_EDGES):
    for tracr_task_key in tqdm(TRACR_TASK_KEYS):
        model, _ = get_tracr_model(tracr_task_key, str(device))

        path = repo_path_to_abs_path(
            f"datasets/tracr/tracr_{tracr_task_key}_len_5_prompts.json"
        )
        _, test_loader = load_datasets_from_json(
            model=None,
            path=path,
            device=device,
            prepend_bos=False,
            batch_size=200,
            train_test_size=(0, 200),
            shuffle=True,
            return_seq_length=False,
            tail_divergence=False,
        )
        unfctrzd_model = deepcopy(model)
        unfctrzd_model = patchable_model(
            model=model,
            factorized=False,
            slice_output="not_first_seq",
            seq_len=None,
            separate_qkv=True,
            device=device,
        )
        fctrzd_model = deepcopy(model)
        fctrzd_model = patchable_model(
            model=model,
            factorized=True,
            slice_output="not_first_seq",
            seq_len=None,
            separate_qkv=True,
            device=device,
        )

        assert tracr_task_key in ("reverse", "xproportion")
        if acdc_edges:
            results = acdc_circ_results
            if tracr_task_key == "reverse":
                official_edge_func = tracr_reverse_acdc_edges
            else:
                official_edge_func = tracr_xproportion_acdc_edges
        else:
            results = my_circ_results
            if tracr_task_key == "reverse":
                official_edge_func = tracr_reverse_true_edges
            else:
                official_edge_func = tracr_xproportion_official_edges
        official_edges: Set[Edge] = official_edge_func(fctrzd_model)

        for algo in tracr_task_algos[tracr_task_key]:
            fctrzd_ps: PruneScores = algo.func(fctrzd_model, test_loader, None)
            unfctrzd_ps: PruneScores = algo.func(unfctrzd_model, test_loader, None)
            unfctrzd_edge_ps: PruneScores = node_ps_to_edge_ps(
                fctrzd_model, unfctrzd_model, fctrzd_ps, unfctrzd_ps
            )
            edge_roc_measurements: Measurements = measure_task_roc(
                fctrzd_model, official_edges, fctrzd_ps, all_edges=True
            )
            node_roc_measurements: Measurements = measure_task_roc(
                fctrzd_model, official_edges, unfctrzd_edge_ps, all_edges=True
            )
            results[False][tracr_task_key][algo.short_name] = node_roc_measurements
            results[True][tracr_task_key][algo.short_name] = edge_roc_measurements

#%%
# Make 2 figures each 2x2, one for our circuit and one for their circuit, showing tasks
# as columns and Node vs Edge Search as rows
for results in [acdc_circ_results, my_circ_results]:
    row_titles = ["Node Search", "Edge Search"]
    column_titles = ["Reverse", "X-Proportion"]
    fig = make_subplots(
        rows=2,
        cols=len(results),
        row_titles=row_titles,
        column_titles=column_titles,
        x_title="False Positive Rate",
        y_title="True Positive Rate",
    )
    for edge_discovery, taskname_results in results.items():
        for task_idx, (task_key, algo_measurements) in enumerate(
            taskname_results.items()
        ):
            for algo_idx, (algo_name, measurements) in enumerate(
                algo_measurements.items()
            ):
                width_delta = 8
                max_width = (width_delta / 2) + (
                    len(algo_measurements) - 1
                ) * width_delta
                line_width = max_width - algo_idx * width_delta
                fig.add_scatter(
                    row=2 if edge_discovery else 1,
                    col=task_idx + 1,
                    x=[x for x, _ in measurements],
                    y=[y for _, y in measurements],
                    mode="markers+text" if len(measurements) == 1 else "lines",
                    text=algo_name,
                    line=dict(width=line_width),
                    textposition="middle right",
                    showlegend=task_idx == 0 and edge_discovery,
                    marker_line_width=2,
                    name=algo_name,
                    marker_color=COLOR_PALETTE[algo_idx],
                )
    fig.update_xaxes(matches=None, scaleanchor="y", scaleratio=1, range=[-0.0, 1.0])
    fig.update_layout(
        height=800,
        width=720,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="left"),
    )
    fig.update_annotations(font_size=22)
    fig.update_traces(line=dict(shape="hv"), mode="lines")
    fig.show()
    folder: Path = repo_path_to_abs_path("figures/figures-12")
    fig_name = "theirs" if results is acdc_circ_results else "ours"
    fig.write_image(str(folder / f"tracr-{fig_name}.pdf"))
# return fig

#%%
# Make a 4x1 figure showing Edge Search for both tasks on their circuit and our circuit
fig = make_subplots(
    rows=1,
    cols=len(acdc_circ_results) * 2,
    column_titles=[
        " ".join(reversed(p))
        for p in product(["(Theirs)", "(Ours)"], ["Reverse", "X-Proportion"])
    ],
    horizontal_spacing=0.02,
    x_title="False Positive Rate",
    y_title="True Positive Rate",
    shared_yaxes=True,
)
for results_idx, results in enumerate([acdc_circ_results, my_circ_results]):
    for edge_discovery, taskname_results in results.items():
        if not edge_discovery:
            continue
        for task_idx, (task_key, algo_measurements) in enumerate(
            taskname_results.items()
        ):
            for algo_idx, (algo_name, measurements) in enumerate(
                algo_measurements.items()
            ):
                width_delta = 8
                max_width = (width_delta / 2) + (
                    len(algo_measurements) - 1
                ) * width_delta
                line_width = max_width - algo_idx * width_delta
                fig.add_scatter(
                    row=1,
                    col=(results_idx * 2) + task_idx + 1,
                    x=[x for x, _ in measurements],
                    y=[y for _, y in measurements],
                    mode="markers+text" if len(measurements) == 1 else "lines",
                    text=algo_name,
                    line=dict(width=line_width),
                    textposition="middle right",
                    showlegend=results_idx == 0 and task_idx == 0,
                    marker_line_width=2,
                    name=algo_name,
                    marker_color=COLOR_PALETTE[algo_idx],
                )
fig.update_annotations(font_size=20)
fig.update_xaxes(
    matches=None,
    scaleanchor="y",
    scaleratio=1,
    range=[-0.0, 1.0],
)
fig.update_layout(
    height=400,
    width=1160,
)
fig.update_traces(line=dict(shape="hv"), mode="lines")
fig.show()
folder: Path = repo_path_to_abs_path("figures/figures-12")
fig.write_image(str(folder / "tracr-edge-based-search.pdf"))
# %%
