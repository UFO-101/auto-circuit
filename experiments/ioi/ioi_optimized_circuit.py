#%%
"""
Plot
"""
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import plotly.graph_objects as go
import torch as t
from plotly import subplots

from auto_circuit.experiment_utils import (
    IOI_CIRCUIT_TYPE,
    ioi_circuit_single_template_logit_diff_percent,
    load_tl_model,
)
from auto_circuit.metrics.prune_scores_similarity import (
    prune_score_similarities,
)
from auto_circuit.types import COLOR_PALETTE, AblationType, AlgoKey, PruneScores
from auto_circuit.utils.custom_tqdm import tqdm
from auto_circuit.utils.misc import repo_path_to_abs_path

device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")
gpt2 = load_tl_model("gpt2", device)


def circ_name(
    circ_type: IOI_CIRCUIT_TYPE,
    learned: bool,
    n_edge: int,
    include_edge_count: bool = True,
) -> str:
    unit = "Nodes" if circ_type == IOI_CIRCUIT_TYPE.NODES else "Edges"
    if not learned:
        if include_edge_count:
            return f"Official {circ_type} ({n_edge} {unit})"
        else:
            return f"Official {circ_type}"
    else:
        if include_edge_count:
            return f"Learned {n_edge} {unit}"
        else:
            return "Learned"


#%%
ABBA_TEMPLATE, BABA_TEMPLATE, ALL_TEMPLATE = "ABBA", "BABA", "Average"
results: Dict[
    str, Dict[int, Dict[Tuple[IOI_CIRCUIT_TYPE, bool, int], Tuple[float, float]]]
] = defaultdict(lambda: defaultdict(dict))
# templates = [ABBA_TEMPLATE, BABA_TEMPLATE]
templates = [ABBA_TEMPLATE]
# template_idxs = [0, 4, 8]  # Out of 15
template_idxs = [0]  # Out of 15
edge_counts = set()
circ_types: List[IOI_CIRCUIT_TYPE] = [IOI_CIRCUIT_TYPE.NODES]
# circ_types: List[IOI_CIRCUIT_TYPE] = [IOI_CIRCUIT_TYPE.NODES, IOI_CIRCUIT_TYPE.EDGES,
# IOI_CIRCUIT_TYPE.EDGES_MLP_0_ONLY]
algo_prune_scores: Dict[AlgoKey, PruneScores] = {}
for template in (template_pbar := tqdm(templates)):
    template_pbar.set_description(f"Template: {template}")
    for template_idx in (template_idx_pbar := tqdm(template_idxs)):
        template_idx_pbar.set_description(f"Template Index: {template_idx}")
        for circ_type in (circ_type_pbar := tqdm(circ_types)):
            circ_type_pbar.set_description(f"Circuit Type: {circ_type}")
            for learned in (learned_pbar := tqdm([True, False])):
                # for learned in (learned_pbar := tqdm([False])):
                learned_pbar.set_description(f"Learned: {learned}")
                (
                    edge_count,
                    logit_diff_percent_mean,
                    logit_diff_percent_std,
                    _,
                    ps,
                ) = ioi_circuit_single_template_logit_diff_percent(
                    gpt2=gpt2,
                    dataset_size=100,
                    prepend_bos=True,
                    template=template,
                    template_idx=template_idx,
                    factorized=circ_type
                    in [IOI_CIRCUIT_TYPE.EDGES, IOI_CIRCUIT_TYPE.EDGES_MLP_0_ONLY],
                    circuit=circ_type,
                    ablation_type=AblationType.TOKENWISE_MEAN_CORRUPT,
                    learned=learned,
                    learned_faithfulness_target="logit_diff_percent",
                )
                results[template][template_idx][(circ_type, learned, edge_count)] = (
                    logit_diff_percent_mean,
                    logit_diff_percent_std,
                )
                algo_key: AlgoKey = circ_name(circ_type, learned, edge_count, False)
                algo_prune_scores[algo_key] = ps
                edge_counts.add(edge_count)

#%%


# Plot a bar chart showing the average logit difference percentage for each template,
# template_idx and circuit type.
fig = subplots.make_subplots(
    rows=1,
    cols=len(templates),
    shared_xaxes=True,
    shared_yaxes=True,
    column_titles=templates,
)
for i, template in enumerate(templates):
    for j, template_idx in enumerate(template_idxs):
        result = results[template][template_idx]
        fig.add_trace(
            go.Bar(
                name=str(template_idx),
                x=[circ_name(*circ_type_info) for circ_type_info in result.keys()],
                y=[mean for mean, _ in result.values()],
                error_y=dict(
                    type="data", array=[std for _, std in result.values()], visible=True
                ),
                showlegend=(i == 0),
                marker_color=COLOR_PALETTE[j],
            ),
            row=1,
            col=i + 1,
        )
fig.add_hline(y=100, line_dash="dot")
fig.update_layout(yaxis_title="Logit Difference Percent", width=700, height=1050)
# Add legend title "Template"
fig.update_layout(legend_title_text="Prompt<br>Template")
fig.show()

folder: Path = repo_path_to_abs_path("figures/figures-12")
fig.write_image(str(folder / "ioi-learned-vs-official.pdf"))

#%%

edge_count_sims: Dict[
    int, Dict[AlgoKey, Dict[AlgoKey, float]]
] = prune_score_similarities(algo_prune_scores, list(edge_counts))

col_count = len(edge_count_sims)
fig = subplots.make_subplots(
    rows=1,
    cols=col_count,
    shared_xaxes=True,
    shared_yaxes=True,
    column_titles=list(edge_count_sims.keys()),
)
algo_count = 0
for count_idx, algo_sims in enumerate(edge_count_sims.values()):
    algo_count = len(algo_sims)
    x_strs = list(reversed(algo_sims.keys()))
    y_strs = list(algo_sims.keys())
    heatmap = []
    for similarity_dict in algo_sims.values():
        row = [sim_score for sim_score in similarity_dict.values()]
        heatmap.append(list(reversed(row)))
    fig.add_trace(
        go.Heatmap(
            x=x_strs,
            y=y_strs,
            z=heatmap,
            colorscale="Viridis",
            showscale=False,
            text=heatmap,
            texttemplate="%{text:.0%}",
            textfont={"size": 19},
        ),
        row=1,
        col=count_idx + 1,
    )
# fig.update_layout(yaxis_scaleanchor="x")
fig.update_layout(plot_bgcolor="rgba(0,0,0,0)")
fig.update_layout(
    width=col_count * 70 * algo_count + 200,
    height=1 * 50 * algo_count + 200,
)

fig.show()

fig.write_image(str(folder / "ioi-learned-vs-official-similarity.pdf"))

# %%
