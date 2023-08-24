#%%
import os
import random
from functools import partial
from typing import Callable, Dict, List

import numpy as np
import pygraphviz as pgv
import torch as t
import torch.backends.mps
import transformer_lens as tl

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
from auto_circuit.prune_functions.random_edges import random_prune_scores
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
toy_model = False
if toy_model:
    cfg = tl.HookedTransformerConfig(
        d_vocab=20,
        n_layers=3,
        d_model=4,
        n_ctx=64,
        n_heads=2,
        d_head=2,
        act_fn="gelu",
        tokenizer_name="gpt2",
        device=device,
    )
    model = tl.HookedTransformer(cfg)
    model.init_weights()
else:
    model = tl.HookedTransformer.from_pretrained("gpt2-small", device=device)

model.cfg.use_attn_result = True
model.cfg.use_split_qkv_input = True
model.cfg.use_hook_mlp_in = True
# model = t.compile(model)

repo_root = "/Users/josephmiller/Documents/auto-circuit"
data_file = "datasets/indirect_object_identification.json"
data_path = f"{repo_root}/{data_file}"

#%%
# ---------- Config ----------
experiment_type = ExperimentType(input_type=ActType.CLEAN, patch_type=ActType.CORRUPT)
factorized = True
pig_baseline, pig_samples = BaselineWeights.ZERO, 50
edge_counts = EdgeCounts.LOGARITHMIC

train_loader, test_loader = auto_circuit.data.load_datasets_from_json(
    model.tokenizer,
    data_path,
    device=device,
    prepend_bos=True,
    batch_size=4,
    train_test_split=[0.75, 0.25],
    length_limit=32,
)
# ----------------------------
#%%

prune_funcs: Dict[str, Callable] = {
    f"PIG ({pig_baseline.name.lower()} Base, {pig_samples} iter)": partial(
        parameter_integrated_grads_prune_scores,
        baseline_weights=pig_baseline,
        samples=pig_samples,
    ),
    "Act Mag": activation_magnitude_prune_scores,
    "Random": random_prune_scores,
}
prune_scores_dict = dict(
    [(n, f(model, factorized, train_loader)) for n, f in prune_funcs.items()]
)
#%%
test_edge_counts = get_test_edge_counts(model, factorized, edge_counts, True)
pruned_outs_dict: Dict[str, Dict[int, List[t.Tensor]]] = {}
for prune_func_str in prune_scores_dict.keys():
    prune_scores = prune_scores_dict[prune_func_str]
    pruned_outs_dict[prune_func_str] = run_pruned(
        model, factorized, test_loader, experiment_type, test_edge_counts, prune_scores
    )
kl_divs: Dict[str, Dict[int, float]] = {}
for prune_func_str, pruned_outs in pruned_outs_dict.items():
    kl_clean, kl_corrupt = measure_kl_div(model, test_loader, pruned_outs)
    kl_divs[prune_func_str + " clean"] = kl_clean
    kl_divs[prune_func_str + " corr"] = kl_corrupt
#%%
kl_vs_edges_plot(kl_divs, experiment_type, edge_counts).show()

#%%
def parse_name(n: str) -> str:
    return n[:-2] if n.startswith("A") and len(n) > 4 else n
    # return n.split(".")[0] if n.startswith("A") else n


edges = auto_circuit.utils.graph_edges(model, factorized)
G = pgv.AGraph(strict=False, directed=True)
for edge in edges:
    print("edge", edge, "src.name", edge.src.name, "dest.name", edge.dest.name)
    # G.add_edge(parse_name(edge.src.name), parse_name(edge.dest.name))
    G.add_edge(
        edge.src.name + f"\n{str(edge.src.weight)}[{edge.src.weight_t_idx}]",
        edge.dest.name + f"\n{str(edge.dest.weight)}[{edge.dest.weight_t_idx}]",
    )
G.layout(prog="dot")  # neato
G.draw("graphviz.png")

# %%
