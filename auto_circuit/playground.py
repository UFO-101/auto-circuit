#%%
import os
import pickle
import random
from datetime import datetime
from functools import partial
from typing import Callable, Dict

import numpy as np
import torch as t
import torch.backends.mps
import transformer_lens as tl

import auto_circuit
import auto_circuit.data
import auto_circuit.prune
from auto_circuit.prune_functions.activation_magnitude import activation_magnitude_prune_scores
from auto_circuit.prune_functions.ioi_official import ioi_true_edges_prune_scores
from auto_circuit.prune_functions.random_edges import random_prune_scores
import auto_circuit.utils.graph_utils
from auto_circuit.prune import (
    measure_kl_div,
    run_pruned,
)
from auto_circuit.prune_functions.ACDC import acdc_edge_counts, acdc_prune_scores
from auto_circuit.prune_functions.parameter_integrated_gradients import (
    BaselineWeights,
    parameter_integrated_grads_prune_scores,
)
from auto_circuit.types import ActType, Edge, EdgeCounts, ExperimentType
from auto_circuit.utils.custom_tqdm import tqdm
from auto_circuit.utils.graph_utils import edge_counts_util
from auto_circuit.utils.misc import percent_gpu_mem_used
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
print("device", device)
toy_model = False
if toy_model:
    cfg = tl.HookedTransformerConfig(
        d_vocab=50257,
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
model.tokenizer.padding_side = "left"
# model = t.compile(model)

# repo_root = "/Users/josephmiller/Documents/auto-circuit"
repo_root = "/home/dev/auto-circuit"
data_file = "datasets/indirect_object_identification.json"
data_path = f"{repo_root}/{data_file}"
print(percent_gpu_mem_used())

#%%
# ---------- Config ----------
experiment_type = ExperimentType(input_type=ActType.CLEAN, patch_type=ActType.CORRUPT)
factorized = True
pig_baseline, pig_samples = BaselineWeights.ZERO, 50
edge_counts = EdgeCounts.LOGARITHMIC
# acdc_tao_range, acdc_tao_step = (1e-6, 2e-5), 2e-6
# acdc_tao_range, acdc_tao_step = (1e-8, 2e-7), 2e-8
one_tao = 6e-2
acdc_tao_range, acdc_tao_step = (one_tao, one_tao), one_tao

train_loader, test_loader = auto_circuit.data.load_datasets_from_json(
    model.tokenizer,
    data_path,
    device=device,
    prepend_bos=True,
    batch_size=8,
    train_test_split=[0.5, 0.5],
    length_limit=64,
)
# ----------------------------
#%%
prune_funcs: Dict[str, Callable] = {
    # f"PIG ({pig_baseline.name.lower()} Base, {pig_samples} iter)": partial(
    #     parameter_integrated_grads_prune_scores,
    #     baseline_weights=pig_baseline,
    #     samples=pig_samples,
    # ),
    # "Act Mag": activation_magnitude_prune_scores,
    # "Random": random_prune_scores,
    # f"ACDC (\u03C4={acdc_tao_range})": partial(
    #     acdc_prune_scores, tao_range=acdc_tao_range, tao_step=acdc_tao_step, patch_slice=(slice(None), slice(-5, None))
    # ),
    # "Subnetwork Probing": partial(
    #     subnetwork_probing_prune_scores, learning_rate=1e-2, epochs=200, max_lambda=30
    # ),
    "IOI Official": ioi_true_edges_prune_scores,
}
# prune_scores_dict: Dict[str, Dict[Edge, float]] = {}
for name, prune_func in (prune_score_pbar := tqdm(prune_funcs.items())):
    prune_score_pbar.set_description_str(f"Computing prune scores: {name}")
    new_prune_scores = prune_func(model, factorized, train_loader)
    if name in prune_scores_dict:
        prune_scores_dict[name].update(new_prune_scores)
    else:
        prune_scores_dict[name] = new_prune_scores
#%%
# SAVE PRUNE SCORES DICT
now = datetime.now()
dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
file_postfix = "(ACDC-factorized-gpt2-1-tao-25000-removed)"
with open(
    f"{repo_root}/.prune_scores_cache/prune_scores_dict-{dt_string}-{file_postfix}.pkl",
    "wb",
) as f:
    pickle.dump(prune_scores_dict, f)
#%%
# LOAD PRUNE SCORES DICT
date = "17-09-2023_17-27-27"
pth = f"{repo_root}/.prune_scores_cache/prune_scores_dict-{date}-{file_postfix}.pkl"
if date is not None:
    with open(pth, "rb") as f:
        loaded_prune_scores = pickle.load(f)
        if prune_scores_dict is None:
            prune_scores_dict = loaded_prune_scores
        else:
            for k, v in loaded_prune_scores.items():
                if k in prune_scores_dict:
                    prune_scores_dict[k].update(v)
                else:
                    prune_scores_dict[k] = v

#%%
test_edge_counts = edge_counts_util(model, factorized, edge_counts)
# pruned_outs_dict: Dict[str, Dict[int, List[t.Tensor]]] = {}
kl_divs: Dict[str, Dict[int, float]] = {}
for prune_func_str, prune_scores in (
    prune_func_pbar := tqdm(prune_scores_dict.items())
):
    prune_func_pbar.set_description_str(f"Pruning with {prune_func_str} scores")
    print("BEFORE prune_func_str", prune_func_str, percent_gpu_mem_used())
    test_edge = (
        acdc_edge_counts(model, factorized, experiment_type, prune_scores)
        if prune_func_str.startswith("ACDC") or prune_func_str.startswith("IOI")
        else test_edge_counts
    )
    pruned_outs = run_pruned(
        model, factorized, test_loader, experiment_type, test_edge, prune_scores
    )
    kl_clean, kl_corrupt = measure_kl_div(model, test_loader, pruned_outs)
    kl_divs[prune_func_str + " clean"] = kl_clean
    kl_divs[prune_func_str + " corr"] = kl_corrupt
    del pruned_outs
    t.cuda.empty_cache()
    print("AFTER prune_func_str", prune_func_str, percent_gpu_mem_used())
#%%
kl_vs_edges_plot(kl_divs, experiment_type, edge_counts, factorized).show()

#%%

# %%
