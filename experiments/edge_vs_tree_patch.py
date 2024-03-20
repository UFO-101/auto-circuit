#%%
from collections import defaultdict
from typing import Dict

import plotly.graph_objects as go
import torch as t
from plotly import subplots

from auto_circuit.experiment_utils import (
    IOI_CIRCUIT_TYPE,
    ioi_circuit_single_template_logit_diff_percent,
    load_tl_model,
)
from auto_circuit.types import COLOR_PALETTE, AblationType, PatchType
from auto_circuit.utils.custom_tqdm import tqdm

device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")
gpt2 = load_tl_model("gpt2", device)
#%%
N_TEMPLATES = 1  # Out of 15
ABBA_TEMPLATE, BABA_TEMPLATE, ALL_TEMPLATE = "ABBA", "BABA", "Average"
results: Dict[IOI_CIRCUIT_TYPE, Dict[PatchType, Dict[str, float]]] = defaultdict(
    lambda: defaultdict(dict)
)
patch_types = [PatchType.TREE_PATCH, PatchType.EDGE_PATCH]
official_circs = [
    IOI_CIRCUIT_TYPE.NODES,
    IOI_CIRCUIT_TYPE.EDGES,
    IOI_CIRCUIT_TYPE.EDGES_MLP_0_ONLY,
]
for true_circ in (true_circ_pbar := tqdm(official_circs)):
    true_circ_pbar.set_description(f"True Circ: {true_circ}")
    for patch_type in (patch_type_pbar := tqdm(patch_types)):
        patch_type_pbar.set_description(f"Ablation Type: {patch_type}")
        batch_size_avg_percent = 0
        for template in (template_pbar := tqdm([ABBA_TEMPLATE, BABA_TEMPLATE])):
            template_pbar.set_description(f"Template: {template}")
            template_logit_diff_perc = []
            for template_idx in (template_idx_pbar := tqdm(range(N_TEMPLATES))):
                template_idx_pbar.set_description(f"Template Index: {template_idx}")
                (
                    edge_count,
                    logit_diff_percent,
                    _,
                    _,
                    _,
                ) = ioi_circuit_single_template_logit_diff_percent(
                    gpt2=gpt2,
                    dataset_size=100,
                    prepend_bos=False,
                    template=template,
                    template_idx=template_idx,
                    factorized=False if true_circ == IOI_CIRCUIT_TYPE.NODES else True,
                    circuit=true_circ,
                    ablation_type=AblationType.RESAMPLE,
                    patch_type=patch_type,
                )
                template_logit_diff_perc.append(logit_diff_percent)
            template_avg_percent = sum(template_logit_diff_perc) / N_TEMPLATES
            results[true_circ][patch_type][template] = template_avg_percent
            batch_size_avg_percent += template_avg_percent
        batch_size_avg_percent /= 2
        results[true_circ][patch_type][ALL_TEMPLATE] = batch_size_avg_percent

#%%


# Plot a bar chart showing the average logit difference percentage for each template and
# ablation type.
fig = subplots.make_subplots(
    rows=1,
    cols=len(official_circs),
    shared_xaxes=True,
    shared_yaxes=True,
    column_titles=[str(circ) for circ in official_circs],
)
for i, circ in enumerate(official_circs):
    for j, template in enumerate([ABBA_TEMPLATE, BABA_TEMPLATE, ALL_TEMPLATE]):
        fig.add_trace(
            go.Bar(
                name=str(template),
                x=[str(a) for a in patch_types],
                y=[results[circ][patch][template] for patch in patch_types],
                showlegend=(i == 0),
                marker_color=COLOR_PALETTE[j],
            ),
            row=1,
            col=i + 1,
        )
fig.add_hline(y=100, line_dash="dot")
fig.update_layout(yaxis_title="Logit Difference Percent", width=700, height=550)
fig.show()
