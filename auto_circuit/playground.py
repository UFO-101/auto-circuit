#%%
import os
import pickle
import random
from datetime import datetime
from functools import partial
from typing import Callable, Dict, Set, Tuple

import numpy as np
import torch as t
import torch.backends.mps
import transformer_lens as tl

import auto_circuit
import auto_circuit.data
import auto_circuit.prune
import auto_circuit.utils.graph_utils
from auto_circuit.metrics.answer_prob import measure_answer_prob
from auto_circuit.metrics.kl_div import measure_kl_div
from auto_circuit.metrics.official_circuits.docstring_official import (
    docstring_true_edges,
)
from auto_circuit.metrics.ROC import measure_roc
from auto_circuit.prune import run_pruned
from auto_circuit.prune_functions.activation_magnitude import (
    activation_magnitude_prune_scores,
)
from auto_circuit.prune_functions.integrated_edge_gradients import (
    integrated_edge_gradients_prune_scores,
)
from auto_circuit.prune_functions.random_edges import random_prune_scores
from auto_circuit.prune_functions.simple_gradient import simple_gradient_prune_scores

# from auto_circuit.prune_functions.parameter_integrated_gradients import (
#     BaselineWeights,
# )
from auto_circuit.types import Edge, EdgeCounts, PatchType
from auto_circuit.utils.custom_tqdm import tqdm
from auto_circuit.utils.graph_utils import edge_counts_util, prepare_model
from auto_circuit.utils.misc import percent_gpu_mem_used
from auto_circuit.visualize import kl_vs_edges_plot, roc_plot

np.random.seed(0)
torch.manual_seed(0)
random.seed(0)
os.environ["TOKENIZERS_PARALLELISM"] = "False"
#%%

device = "cuda" if t.cuda.is_available() else "cpu"
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
    # model = tl.HookedTransformer.from_pretrained("gpt2-small", device=device)
    model = tl.HookedTransformer.from_pretrained("attn-only-4l", device=device)
    # model = tl.HookedTransformer.from_pretrained("tiny-stories-33M", device=device)

model.cfg.use_attn_result = True
model.cfg.use_split_qkv_input = True
model.cfg.use_hook_mlp_in = True
assert model.tokenizer is not None
# model = t.compile(model)
model.eval()

# repo_root = "/Users/josephmiller/Documents/auto-circuit"
repo_root = "/home/dev/auto-circuit"
# data_file = "datasets/indirect_object_identification.json"
# data_file = "datasets/greater_than_gpt2-small_prompts.json"
data_file = "datasets/docstring_prompts.json"
# data_file = "datasets/animal_diet_short_prompts.json"
# data_file = "datasets/mini_prompts.json"
data_path = f"{repo_root}/{data_file}"
print(percent_gpu_mem_used())

#%%
# ---------- Config ----------
experiment_type = PatchType.PATH_PATCH
factorized = True
# pig_baseline, pig_samples = BaselineWeights.ZERO, 50
default_edge_count_type = EdgeCounts.LOGARITHMIC
one_tao = 6e-2
acdc_tao_range, acdc_tao_step = (one_tao, one_tao), one_tao

train_loader, test_loader = auto_circuit.data.load_datasets_from_json(
    model.tokenizer,
    data_path,
    device=device,
    prepend_bos=True,
    batch_size=32,
    train_test_split=[0.5, 0.5],
    length_limit=64,
    pad=True,
)
prepare_model(
    model, factorized=factorized, slice_output=True, seq_len=None, device=device
)
edges: Set[Edge] = model.edges  # type: ignore
prune_scores_dict: Dict[str, Dict[Edge, float]] = {}

# ----------------------------
#%%
prune_funcs: Dict[str, Callable] = {
    # f"PIG ({pig_baseline.name.lower()} Base, {pig_samples} iter)": partial(
    #     parameter_integrated_grads_prune_scores,
    #     baseline_weights=pig_baseline,
    #     samples=pig_samples,
    # ),
    "Act Mag": activation_magnitude_prune_scores,
    "Random": random_prune_scores,
    # "ACDC": partial(
    #     acdc_prune_scores,
    #     # tao_exps=list(range(-6, 1)),
    #     tao_exps=[-5],
    #     tao_bases=[1],
    # ),
    "Integrated edge gradients": partial(
        integrated_edge_gradients_prune_scores,
        samples=50,
    ),
    "Prob Gradient": partial(simple_gradient_prune_scores, grad_function="prob"),
    "Exp Logit Gradient": partial(simple_gradient_prune_scores, grad_function="logit_exp"),
    "Logit Gradient": partial(simple_gradient_prune_scores, grad_function="logit"),
    "Logprob Gradient": partial(simple_gradient_prune_scores, grad_function="logprob"),
    # "Subnetwork Probing": partial(
    #     subnetwork_probing_prune_scores,
    #     learning_rate=0.1,
    #     epochs=500,
    #     regularize_lambda=10,
    #     mask_fn=None,  # "hard_concrete",
    #     dropout_p=0.0,
    #     init_val=1.0,
    #     show_train_graph=True,
    # ),
}
for name, prune_func in (prune_score_pbar := tqdm(prune_funcs.items())):
    prune_score_pbar.set_description_str(f"Computing prune scores: {name}")
    new_prune_scores = prune_func(model, train_loader)
    if name in prune_scores_dict:
        prune_scores_dict[name].update(new_prune_scores)
    else:
        prune_scores_dict[name] = new_prune_scores

#%%
# ps = prune_scores_dict["Integrated Edge Gradients"]
# ps = dict(sorted(ps.items(), key=lambda x: abs(x[1]), reverse=True)[:30])
# run_pruned(
#     model=model,
#     data_loader=test_loader,
#     test_edge_counts=[],
#     prune_scores=ps,
#     patch_type=PatchType.PATH_PATCH,
#     render_graph=True,
#     render_patched_edge_only=True,
#     seq_labels="BOS The cows ate some".split(),
# )
#%%
# SAVE / LOAD PRUNE SCORES DICT
save = False
load = False
if save:
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
    file_postfix = "(ACDC-factorized-gpt2-MEGA-RUN)"
    file_name = f"prune_scores_dict-{dt_string}-{file_postfix}"
    with open(f"{repo_root}/.prune_scores_cache/{file_name}.pkl", "wb") as f:
        pickle.dump(prune_scores_dict, f)
if load:
    ioi_acdc_pth = "IOI-ACDC-prune-scores"
    # ioi_sp_pth = "IOI-Subnetwork-Edge-Probing-prune-scores"
    ioi_sp_pth = "IOI-Subnetwork-Edge-Probing-Dropout-05-Epochs-800"
    # for pth in [ioi_acdc_pth, ioi_sp_pth]:
    for pth in [ioi_acdc_pth]:
        with open(f"{repo_root}/.prune_scores_cache/{pth}.pkl", "rb") as f:
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
kl_divs: Dict[str, Dict[int, float]] = {}
for prune_func_str, prune_scores in (prune_func_pbar := tqdm(prune_scores_dict.items())):
    prune_func_pbar.set_description_str(f"Pruning with {prune_func_str} scores")
    group_edges = prune_func_str.startswith("ACDC") or prune_func_str.startswith("IOI")
    edge_count_type = EdgeCounts.GROUPS if group_edges else default_edge_count_type
    test_edge_counts = edge_counts_util(edges, edge_count_type, prune_scores)
    pruned_outs = run_pruned(model, test_loader, test_edge_counts, prune_scores)
    kl_clean, kl_corrupt = measure_kl_div(model, test_loader, pruned_outs)
    kl_divs[prune_func_str + " clean"] = kl_clean
    kl_divs[prune_func_str + " corr"] = kl_corrupt
    del pruned_outs
    t.cuda.empty_cache()

kl_vs_edges_plot(
    kl_divs, default_edge_count_type, PatchType.PATH_PATCH, "KL Divergence"
).show()
#%%
roc: Dict[str, Set[Tuple[float, float]]] = {}
for func_str, prune_scores in (prune_func_pbar := tqdm(prune_scores_dict.items())):
    prune_func_pbar.set_description_str(f"Pruning with {func_str} scores")
    group_edges = func_str.startswith("ACDC") or func_str.startswith("IOI")
    # correct_edges = ioi_true_edges(model)
    # roc[func_str] = measure_roc(model, prune_scores, correct_edges, True, group_edges)
    # correct_edges = greaterthan_true_edges(model)
    # roc[func_str] = measure_roc(model, prune_scores, correct_edges, False, group_edges)
    correct_edges = docstring_true_edges(model)
    roc[func_str] = measure_roc(model, prune_scores, correct_edges, False, group_edges)
roc_plot("Greaterthan", roc).show()

#%%
answer_probs: Dict[str, Dict[int, float]] = {}
for prune_func_str, prune_scores in (func_pbar := tqdm(prune_scores_dict.items())):
    func_pbar.set_description_str(f"Pruning with {prune_func_str} scores")
    group_edges = prune_func_str.startswith("ACDC") or prune_func_str.startswith("IOI")
    edge_counts_type = EdgeCounts.GROUPS if group_edges else default_edge_count_type
    test_edge_counts = edge_counts_util(edges, edge_counts_type, prune_scores)
    pruned_outs = run_pruned(
        model,
        test_loader,
        test_edge_counts,
        prune_scores,
    )
    answer_probs[prune_func_str] = measure_answer_prob(
        model, test_loader, pruned_outs, prob_func="softmax"
    )
    del pruned_outs

kl_vs_edges_plot(
    answer_probs,
    default_edge_count_type,
    PatchType.PATH_PATCH,
    "Correct Token Prob",
    factorized,
).show()

#%%
