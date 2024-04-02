#%%
"""
Compare the faithfulness of the node-based IOI circuit at different batch sizes, on the
ABBA and BABA templates.
"""
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import plotly.graph_objects as go
import torch as t
from plotly import subplots

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

ABBA_TEMPLATE, BABA_TEMPLATE, ALL_TEMPLATE = "ABBA", "BABA", "Average"
batch_size_percents: Dict[
    bool, Dict[int, Dict[str, Tuple[float, float, Optional[t.Tensor]]]]
] = defaultdict(lambda: defaultdict(dict))
# batch_sizes = [2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
batch_sizes = [2, 5, 10, 20, 50, 100, 200]
# batch_sizes = [10, 50, 100, 200]
mean_logit_diffs: List[bool] = [True, False]
# mean_logit_diffs: List[bool] = [False]
col_titles: List[str] = [
    "% Average Logit Diff",
    "Average Logit Diff %",
    "Individual Predictions",
    "Individual Predictions",
]

for mean_logit_diff in (mean_logit_diff_pbar := tqdm(mean_logit_diffs)):
    mean_logit_diff_pbar.set_description(f"Individual Diff: {mean_logit_diff}")
    for test_batch_size in (size_pbar := tqdm(batch_sizes)):
        size_pbar.set_description(f"Test Batch Size: {test_batch_size}")
        batch_size_avg_percent_mean, batch_size_avg_percent_std = 0, 0
        for template in (template_pbar := tqdm([ABBA_TEMPLATE, BABA_TEMPLATE])):
            template_pbar.set_description(f"Template: {template}")
            template_logit_diff_perc_mean, template_logit_diff_perc_std = [], []
            template_logit_diff_points = []
            n_templates = 15
            for template_idx in (template_idx_pbar := tqdm(range(n_templates))):
                template_idx_pbar.set_description(f"Template Index: {template_idx}")
                (
                    _,
                    logit_diff_percent,
                    logit_diff_std,
                    logit_diff_points,
                    _,
                ) = ioi_circuit_single_template_logit_diff_percent(
                    gpt2=gpt2,
                    dataset_size=200,
                    prepend_bos=False,
                    template=template,
                    template_idx=template_idx,
                    factorized=True,
                    circuit=IOI_CIRCUIT_TYPE.EDGES,
                    ablation_type=AblationType.BATCH_ALL_TOK_MEAN,
                    diff_of_mean_logit_diff=mean_logit_diff,
                    batch_size=test_batch_size,
                )
                template_logit_diff_perc_mean.append(logit_diff_percent)
                template_logit_diff_perc_std.append(logit_diff_std)
                template_logit_diff_points.append(logit_diff_points)
            template_avg_percent_mean = sum(template_logit_diff_perc_mean) / n_templates
            template_avg_percent_std = sum(template_logit_diff_perc_std) / n_templates
            template_all_points = t.cat(template_logit_diff_points, dim=0)
            batch_size_percents[mean_logit_diff][test_batch_size][template] = (
                template_avg_percent_mean,
                template_avg_percent_std,
                template_all_points,
            )
            batch_size_avg_percent_mean += template_avg_percent_mean
            batch_size_avg_percent_std += template_avg_percent_std
        batch_size_avg_percent_mean /= 2
        batch_size_avg_percent_std /= 2
        batch_size_percents[mean_logit_diff][test_batch_size][ALL_TEMPLATE] = (
            batch_size_avg_percent_mean,
            batch_size_avg_percent_std,
            None,
        )

#%%
# Plot a graph showing the average logit difference percentage for each template at
# each batch size.

col_titles: List[str] = [
    "[Average Logit Diff] %",
    "Average [Logit Diff %]",
    "ABBA Predictions",
    "BABA Predictions",
]
fig = subplots.make_subplots(
    rows=1,
    cols=len(mean_logit_diffs) + 2,
    # shared_yaxes=False,
    # shared_xaxes=False,
    column_titles=col_titles,
    x_title="ABC Dataset Size",
    y_title="Logit Difference Recovered",
)
for col, mean_logit_diff in enumerate(mean_logit_diffs, start=1):
    for i, template in enumerate([ABBA_TEMPLATE, BABA_TEMPLATE, ALL_TEMPLATE]):
        x = batch_sizes
        y = [
            batch_size_percents[mean_logit_diff][batch_size][template][0]
            for batch_size in x
        ]
        y_stds = [
            batch_size_percents[mean_logit_diff][batch_size][template][1]
            for batch_size in x
        ]
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines+markers",
                name=template,
                showlegend=col == 1,
                line=dict(color=COLOR_PALETTE[i]),
            ),
            row=1,
            col=col,
        )

# Plot all the individual points in the third column
for i, template in enumerate([ABBA_TEMPLATE, BABA_TEMPLATE]):
    for batch_size in batch_sizes:
        points = batch_size_percents[False][batch_size][template][2]
        assert points is not None
        points = points.cpu().tolist()
        x = [batch_size] * len(points)
        fig.add_trace(
            go.Box(
                x=x,
                y=points,
                name=template,
                showlegend=False,
                marker=dict(
                    color=COLOR_PALETTE[i],
                    opacity=0.25,
                ),
            ),
            row=1,
            col=len(mean_logit_diffs) + 1 + i,
        )

[fig.update_xaxes(type="log", row=1, col=i + 1) for i in range(4)]
[fig.update_yaxes(range=[-200, 500], row=1, col=col) for col in [3, 4]]
# [fig.update_yaxes(range=[55, 135], row=1, col=col) for col in [1, 2]]
fig.update_annotations(font_size=20)
margin = 20
fig.update_layout(
    # yaxis_title="Logit Difference Percent",
    width=1400,
    height=400,
    margin=dict(l=margin * 4, r=margin, b=margin * 3.5, t=margin * 2),
)

fig.add_hline(
    y=87,
    line_dash="dot",
    annotation_text="Reported Faithfulness",
    annotation_position="bottom right",
    annotation_font_size=14,
    row=1,  # type: ignore
    col=1,  # type: ignore
)
fig.add_hline(
    y=100,
    annotation_text="Perfect Faithfulness",
    annotation_position="top right",
    annotation_font_size=14,
    row=1,  # type: ignore
    col=1,  # type: ignore
)
fig.add_hline(y=87, line_dash="dot")
fig.add_hline(y=100)

fig.show()
folder: Path = repo_path_to_abs_path("figures/figures-12")
# Save figure as pdf in figures folder
fig.write_image(str(folder / "ioi_edge_all_tok_mean_abc_size_faithfulness.pdf"))

# %%
