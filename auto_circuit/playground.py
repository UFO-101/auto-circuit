#%%
import os
import random

import numpy as np
import pygraphviz as pgv
import torch as t
import torch.backends.mps
import transformer_lens

import auto_circuit
import auto_circuit.data
import auto_circuit.run_experiments
import auto_circuit.utils
from auto_circuit.prune_functions.activation_magnitude import (
    activation_magnitude_prune_scores,
)
from auto_circuit.prune_functions.parameter_integrated_gradients import (
    BaselineWeights,
    parameter_integrated_grads_prune_scores,
)
from auto_circuit.run_experiments import (
    get_test_edge_counts,
    measure_kl_div,
    run_pruned,
)
from auto_circuit.types import ActType, EdgeCounts, ExperimentType
from auto_circuit.visualize import kl_vs_edges_plot

np.random.seed(0)
torch.manual_seed(0)
random.seed(0)
os.environ["TOKENIZERS_PARALLELISM"] = "False"
#%%

device = (
    "cuda"
    if t.cuda.is_available()
    else "mps"
    if True and torch.backends.mps.is_available()
    else "cpu"
)
model = transformer_lens.HookedTransformer.from_pretrained("gpt2-small", device=device)
# cfg = transformer_lens.HookedTransformerConfig(
#     d_vocab=20, n_layers=2, d_model=4, n_ctx=64, n_heads=2, d_head=2, act_fn="gelu"
# )
# model = transformer_lens.HookedTransformer(cfg, model.tokenizer)
# model.init_weights()

model.cfg.use_attn_result = True
model.cfg.use_split_qkv_input = True
model.cfg.use_hook_mlp_in = True
# model = t.compile(model)

#%%
repo_root = "/Users/josephmiller/Documents/auto-circuit"
data_file = "datasets/indirect_object_identification.json"
data_path = f"{repo_root}/{data_file}"

experiment_type = ExperimentType(input_type=ActType.CLEAN, patch_type=ActType.CORRUPT)

train_loader, test_loader = auto_circuit.data.load_datasets_from_json(
    model.tokenizer,
    data_path,
    device=device,
    prepend_bos=True,
    batch_size=4,
    train_test_split=[0.75, 0.25],
    length_limit=32,
)
pig_prune_scores = parameter_integrated_grads_prune_scores(
    model, train_loader, BaselineWeights.ZERO, samples=50
)
act_mag_prune_scores = activation_magnitude_prune_scores(model, train_loader)
#%%
test_edge_counts = get_test_edge_counts(model, EdgeCounts.LOGARITHMIC, True)
pig_pruned_outs = run_pruned(
    model, test_loader, experiment_type, test_edge_counts, pig_prune_scores
)
act_mag_pruned_outs = run_pruned(
    model, test_loader, experiment_type, test_edge_counts, act_mag_prune_scores
)
pig_kl_vs_edges_clean, pig_kl_vs_edges_corrupt = measure_kl_div(
    model, test_loader, pig_pruned_outs
)
act_mag_kl_vs_edges_clean, act_mag_kl_vs_edges_corrupt = measure_kl_div(
    model, test_loader, act_mag_pruned_outs
)
kl_divs = [
    ("pig clean", pig_kl_vs_edges_clean),
    ("pig corrupt", pig_kl_vs_edges_corrupt),
    ("act mag clean", act_mag_kl_vs_edges_clean),
    ("act mag corrupt", act_mag_kl_vs_edges_corrupt),
]
kl_vs_edges_plot(kl_divs, experiment_type).show()

#%%
def parse_name(n: str) -> str:
    return n[:-2] if n.startswith("A") and len(n) > 4 else n
    # return n.split(".")[0] if n.startswith("A") else n


edges = auto_circuit.utils.graph_edges(model)
G = pgv.AGraph(strict=False, directed=True)
for edge in edges:
    G.add_edge(parse_name(edge.src.name), parse_name(edge.dest.name))
G.layout(prog="dot")  # neato
G.draw("graphviz.png")
