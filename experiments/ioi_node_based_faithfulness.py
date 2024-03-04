#%%
from collections import defaultdict
from typing import Dict

import plotly.graph_objects as go
import torch as t
import transformer_lens as tl

from auto_circuit.experiment_utils import (
    ioi_node_circuit_single_template_logit_diff_percent,
)
from auto_circuit.utils.custom_tqdm import tqdm

device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")
gpt2 = tl.HookedTransformer.from_pretrained(
    "gpt2",
    device=str(device),
    fold_ln=True,
    center_writing_weights=True,
    center_unembed=True,
)
gpt2.cfg.use_attn_result = True
gpt2.cfg.use_attn_in = True
gpt2.cfg.use_split_qkv_input = True
gpt2.cfg.use_hook_mlp_in = True
gpt2.eval()
for param in gpt2.parameters():
    param.requires_grad = False

ABBA_TEMPLATE, BABA_TEMPLATE, ALL_TEMPLATE = "ABBA", "BABA", "Average"
batch_size_percents: Dict[int, Dict[str, float]] = defaultdict(dict)
batch_sizes = [2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
for test_batch_size in (size_pbar := tqdm(batch_sizes)):
    size_pbar.set_description(f"Test Batch Size: {test_batch_size}")
    batch_size_avg_percent = 0
    for template in (template_pbar := tqdm([ABBA_TEMPLATE, BABA_TEMPLATE])):
        template_pbar.set_description(f"Template: {template}")
        template_logit_diff_perc = []
        n_templates = 15
        for template_idx in (template_idx_pbar := tqdm(range(n_templates))):
            template_idx_pbar.set_description(f"Template Index: {template_idx}")
            logit_diff_percent = ioi_node_circuit_single_template_logit_diff_percent(
                gpt2=gpt2,
                test_batch_size=test_batch_size,
                prepend_bos=False,
                template=template,
                template_idx=template_idx,
            )
            template_logit_diff_perc.append(logit_diff_percent)
        template_avg_percent = sum(template_logit_diff_perc) / n_templates
        batch_size_percents[test_batch_size][template] = template_avg_percent
        batch_size_avg_percent += template_avg_percent
    batch_size_avg_percent /= 2
    batch_size_percents[test_batch_size][ALL_TEMPLATE] = batch_size_avg_percent

#%%
# Plot a graph showing the average logit difference percentage for each template at
# each batch size.

fig = go.Figure()
for template in (template_pbar := tqdm([ABBA_TEMPLATE, BABA_TEMPLATE, ALL_TEMPLATE])):
    template_pbar.set_description(f"Template: {template}")
    x = list(batch_size_percents.keys())
    y = [batch_size_percents[batch_size][template] for batch_size in x]
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", name=template))
fig.update_xaxes(type="log")
margin = 20
fig.update_layout(
    xaxis_title="Dataset Size",
    yaxis_title="Avg Logit Difference Percent",
    width=700,
    height=450,
    margin=dict(l=margin, r=margin, b=margin, t=margin),
)
fig.add_hline(
    y=87,
    line_dash="dot",
    annotation_text="Reported Faithfulness",
    annotation_position="bottom right",
    annotation_font_size=16,
)
fig.show()
