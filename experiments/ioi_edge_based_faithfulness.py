#%%
from collections import defaultdict
from typing import Dict

import plotly.graph_objects as go
import torch as t
from plotly import subplots

from auto_circuit.experiment_utils import (
    ioi_circuit_single_template_logit_diff_percent,
    load_tl_model,
)
from auto_circuit.types import COLOR_PALETTE, AblationType
from auto_circuit.utils.custom_tqdm import tqdm

device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")
gpt2 = load_tl_model("gpt2", device)
#%%
N_TEMPLATES = 3  # Out of 15
ABBA_TEMPLATE, BABA_TEMPLATE, ALL_TEMPLATE = "ABBA", "BABA", "Average"
results: Dict[str, Dict[AblationType, Dict[str, float]]] = defaultdict(
    lambda: defaultdict(dict)
)
ablation_types = [AblationType.RESAMPLE, AblationType.TOKENWISE_MEAN_CORRUPT]
true_circs = ["Nodes", "Edges", "Edges (MLP 0 only)"]
for true_circ in (true_circ_pbar := tqdm(true_circs)):
    true_circ_pbar.set_description(f"True Circ: {true_circ}")
    for ablation_type in (ablation_type_pbar := tqdm(ablation_types)):
        ablation_type_pbar.set_description(f"Ablation Type: {ablation_type}")
        batch_size_avg_percent = 0
        for template in (template_pbar := tqdm([ABBA_TEMPLATE, BABA_TEMPLATE])):
            template_pbar.set_description(f"Template: {template}")
            template_logit_diff_perc = []
            for template_idx in (template_idx_pbar := tqdm(range(N_TEMPLATES))):
                template_idx_pbar.set_description(f"Template Index: {template_idx}")
                logit_diff_percent = ioi_circuit_single_template_logit_diff_percent(
                    gpt2=gpt2,
                    test_batch_size=100,
                    prepend_bos=False,
                    template=template,
                    template_idx=template_idx,
                    factorized=False if true_circ == "Nodes" else True,
                    true_circuit=true_circ,
                    ablation_type=ablation_type,
                )
                template_logit_diff_perc.append(logit_diff_percent)
            template_avg_percent = sum(template_logit_diff_perc) / N_TEMPLATES
            results[true_circ][ablation_type][template] = template_avg_percent
            batch_size_avg_percent += template_avg_percent
        batch_size_avg_percent /= 2
        results[true_circ][ablation_type][ALL_TEMPLATE] = batch_size_avg_percent

#%%


def ablate_name(ablation_type: AblationType) -> str:
    if ablation_type == AblationType.TOKENWISE_MEAN_CORRUPT:
        return "Mean (ABC)"
    return str(ablation_type)


# Plot a bar chart showing the average logit difference percentage for each template and
# ablation type.
fig = subplots.make_subplots(
    rows=1,
    cols=len(true_circs),
    shared_xaxes=True,
    shared_yaxes=True,
    column_titles=true_circs,
)
for i, circ in enumerate(true_circs):
    for j, template in enumerate([ABBA_TEMPLATE, BABA_TEMPLATE, ALL_TEMPLATE]):
        fig.add_trace(
            go.Bar(
                name=str(template),
                x=[ablate_name(a) for a in ablation_types],
                y=[results[circ][ablate][template] for ablate in ablation_types],
                showlegend=(i == 0),
                marker_color=COLOR_PALETTE[j],
            ),
            row=1,
            col=i + 1,
        )
fig.add_hline(y=100, line_dash="dot")
fig.update_layout(yaxis_title="Logit Difference Percent", width=700, height=550)
fig.show()

#%%
