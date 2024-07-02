#%%
"""
Compare the faithfulness of the edge-based IOI circuits to the node-based IOI circuit,
on ABBA and BABA templates, with different ablation types.
"""
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import plotly.graph_objects as go
import torch as t
from plotly import subplots
from scipy.stats import ttest_ind

from auto_circuit.experiment_utils import (
    IOI_CIRCUIT_TYPE,
    ioi_circuit_single_template_logit_diff_percent,
    load_tl_model,
)
from auto_circuit.types import COLOR_PALETTE, AblationType
from auto_circuit.utils.custom_tqdm import tqdm
from auto_circuit.utils.misc import repo_path_to_abs_path

#%%

device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")
gpt2 = load_tl_model("gpt2", device)
N_TEMPLATES = 15  # Out of 15
ABBA_TEMPLATE, BABA_TEMPLATE, ALL_TEMPLATE = "ABBA", "BABA", "Average"
results: Dict[
    IOI_CIRCUIT_TYPE, Dict[bool, Dict[AblationType, Dict[str, Tuple[int, float]]]]
] = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
points: Dict[
    IOI_CIRCUIT_TYPE, Dict[bool, Dict[AblationType, List[t.Tensor]]]
] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

tok_positions = [True, False]
ablation_types = [AblationType.RESAMPLE, AblationType.TOKENWISE_MEAN_CORRUPT]

true_circs: List[IOI_CIRCUIT_TYPE] = [
    IOI_CIRCUIT_TYPE.NODES,
    IOI_CIRCUIT_TYPE.EDGES,
    # IOI_CIRCUIT_TYPE.EDGES_MLP_0_ONLY,
]
n_edge = 0
for true_circ in (true_circ_pbar := tqdm(true_circs)):
    true_circ_pbar.set_description(f"True Circ: {true_circ}")
    for tok_pos in (tok_pos_pbar := tqdm(tok_positions)):
        tok_pos_pbar.set_description(f"Tok Pos: {tok_pos}")
        for ablation_type in (ablation_type_pbar := tqdm(ablation_types)):
            ablation_type_pbar.set_description(
                f"Tok Pos: {tok_pos}, Ablation Type: {ablation_type}"
            )
            batch_size_avg_percent = 0
            for template in (template_pbar := tqdm([ABBA_TEMPLATE, BABA_TEMPLATE])):
                template_pbar.set_description(f"Template: {template}")
                template_logit_diff_perc = []
                for template_idx in (template_idx_pbar := tqdm(range(N_TEMPLATES))):
                    template_idx_pbar.set_description(f"Template Index: {template_idx}")
                    (
                        n_edge,
                        logit_diff_percent,
                        _,
                        percents,
                        _,
                    ) = ioi_circuit_single_template_logit_diff_percent(
                        gpt2=gpt2,
                        dataset_size=100,
                        prepend_bos=False,
                        template=template,
                        template_idx=template_idx,
                        factorized=False
                        if true_circ == IOI_CIRCUIT_TYPE.NODES
                        else True,
                        circuit=true_circ,
                        ablation_type=ablation_type,
                        tok_pos=tok_pos,
                    )
                    template_logit_diff_perc.append(logit_diff_percent)
                    points[true_circ][tok_pos][ablation_type].append(percents)
                template_avg_percent = sum(template_logit_diff_perc) / N_TEMPLATES
                results[true_circ][tok_pos][ablation_type][template] = (
                    n_edge,
                    template_avg_percent,
                )
                batch_size_avg_percent += template_avg_percent
            batch_size_avg_percent /= 2
            results[true_circ][tok_pos][ablation_type][ALL_TEMPLATE] = (
                n_edge,
                batch_size_avg_percent,
            )

#%%


def tok_ablate_name(ablate: AblationType, n_edge: Optional[int] = None) -> str:
    if ablate == AblationType.TOKENWISE_MEAN_CORRUPT:
        return "Mean (ABC)"
    return str(ablate)


# Plot a bar chart showing the average logit difference percentage for each template and
# ablation type.
fig = subplots.make_subplots(
    rows=len(tok_positions),
    cols=len(true_circs),
    # shared_xaxes=True,
    shared_yaxes=True,
    row_titles=["Separate Token Edges", "All Token Edges"],
    column_titles=[" ".join(str(c).split("_")) for c in true_circs],
    # y_title="Logit Difference Recovered",
)
for i, circ in enumerate(true_circs):
    for j, tok_pos in enumerate(tok_positions):
        for k, template in enumerate([ABBA_TEMPLATE, BABA_TEMPLATE, ALL_TEMPLATE]):
            fig.add_trace(
                go.Bar(
                    name=str(template),
                    x=[
                        tok_ablate_name(
                            ablate, results[circ][tok_pos][ablate][template][0]
                        )
                        for ablate in ablation_types
                    ],
                    y=[
                        results[circ][tok_pos][ablate][template][1]
                        for ablate in ablation_types
                    ],
                    showlegend=(i == 0 and j == 0),
                    marker_color=COLOR_PALETTE[k],
                ),
                row=j + 1,
                col=i + 1,
            )
fig.update_annotations(font_size=22)
fig.add_hline(y=100, line_dash="dot")
fig.add_hline(y=100)
fig.update_layout(width=800, height=800)
fig.update_yaxes(title_text="Logit Diff Recovered", row=1, col=1)
fig.update_yaxes(title_text="Logit Diff Recovered", row=2, col=1)
fig.show()

folder: Path = repo_path_to_abs_path("figures/figures-12")
# Save figure as pdf in figures folder
fig.write_image(str(folder / "ioi_edges_faithfulness.png"), scale=4)
fig.write_image(str(folder / "ioi_edges_faithfulness.pdf"))

fig = subplots.make_subplots(
    rows=1,
    cols=len(tok_positions) * len(true_circs),
    # shared_xaxes=True,
    shared_yaxes=True,
    # row_titles=["Separate Token Edges", "All Token Edges"],
    # column_titles=[" ".join(str(c).split("_")) for c in true_circs],
    column_titles=[
        "Nodes/Specific Toks",
        "Nodes/All Toks",
        "Edges/Specific Toks",
        "Edges/All Toks",
    ],
    # y_title="Logit Difference Recovered",
)
for i, circ in enumerate(true_circs):
    for j, tok_pos in enumerate(tok_positions):
        col_idx = i * len(tok_positions) + j + 1
        print("col_idx", col_idx)
        print("circ", circ, "tok_pos", tok_pos)
        for k, ablate in enumerate(ablation_types):
            fig.add_trace(
                go.Box(
                    name=tok_ablate_name(ablate),
                    y=t.stack(points[circ][tok_pos][ablate]).flatten().tolist(),
                    showlegend=False,
                    marker_color=COLOR_PALETTE[k],
                ),
                row=1,
                col=col_idx,
            )
            fig.update_yaxes(row=j + 1, col=i + 1, range=[-700, 700])
fig.update_annotations(font_size=20)
fig.add_hline(
    y=100,
    line_dash="dot",
    annotation_text="Perfect Faithfulness",
    annotation_position="top left",
    annotation_font_size=14,
    row=1,  # type: ignore
    col=1,  # type: ignore
    annotation_bgcolor="rgb(228,236,246)",
)
fig.add_hline(y=100, line_dash="dot")
fig.add_hline(y=0)
margin = 20
fig.update_layout(
    width=1400,
    height=600,
    margin=dict(l=margin * 4, r=margin, b=margin * 3.5, t=margin * 2),
)
fig.update_yaxes(title_text="Logit Diff Recovered", row=1, col=1)
fig.show()

fig.write_image(str(folder / "ioi_edges_boxplot.png"), scale=4)
fig.write_image(str(folder / "ioi_edges_boxplot.pdf"))

#%%
# Check statistical significance between Resample and Mean ablation using t-test

for i, circ in enumerate(true_circs):
    for j, tok_pos in enumerate(tok_positions):
        print("circ", circ, "tok_pos", tok_pos)
        resample = (
            t.stack(points[circ][tok_pos][AblationType.RESAMPLE]).flatten().tolist()
        )
        mean = (
            t.stack(points[circ][tok_pos][AblationType.TOKENWISE_MEAN_CORRUPT])
            .flatten()
            .tolist()
        )
        res = ttest_ind(resample, mean)
        print(res)
