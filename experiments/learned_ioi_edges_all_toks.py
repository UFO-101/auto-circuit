#%%
"""This code learns an edge-based circuit for the IOI task with no distinction between
different tokens."""
from pathlib import Path
from typing import List

import torch as t

from auto_circuit.data import load_datasets_from_json
from auto_circuit.experiment_utils import load_tl_model
from auto_circuit.metrics.official_circuits.circuits.ioi_official import (
    ioi_head_based_official_edges,
    ioi_true_edges,
    ioi_true_edges_mlp_0_only,
)
from auto_circuit.metrics.prune_metrics.answer_diff_percent import answer_diff_percent
from auto_circuit.metrics.prune_metrics.kl_div import (
    measure_kl_div,
)
from auto_circuit.prune import run_circuits
from auto_circuit.prune_algos.circuit_probing import circuit_probing_prune_scores
from auto_circuit.types import (
    AblationType,
    CircuitOutputs,
    Measurements,
    PatchType,
    PruneScores,
)
from auto_circuit.utils.graph_utils import patchable_model
from auto_circuit.utils.misc import load_cache, repo_path_to_abs_path, save_cache

#%%

device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")
gpt2 = load_tl_model("gpt2", device)

gpt2 = patchable_model(
    model=gpt2,
    factorized=True,
    slice_output="last_seq",
    seq_len=None,
    separate_qkv=True,
    device=device,
)

paths: List[Path] = []
for template in ["ABBA", "BABA"]:
    for template_idx in range(15):
        path = repo_path_to_abs_path(
            f"datasets/ioi/ioi_{template}_template_{template_idx}_prompts.json"
        )
        paths.append(path)

train_loader, test_loader = load_datasets_from_json(
    model=gpt2,
    path=paths,
    device=device,
    prepend_bos=True,
    batch_size=64,
    train_test_size=(2000, 200),
    shuffle=True,
    return_seq_length=False,
    tail_divergence=False,
)

# path = repo_path_to_abs_path("datasets/ioi/ioi_ABBA_template_0_prompts.json")
# train_loader, test_loader = load_datasets_from_json(
#     model=gpt2,
#     path=path,
#     device=device,
#     prepend_bos=True,
#     batch_size=64,
#     train_test_size=(900, 100),
#     shuffle=True,
#     return_seq_length=True,
#     tail_divergence=False,
# )

# gpt2 = patchable_model(
#     model=gpt2,
#     factorized=True,
#     slice_output="last_seq",
#     seq_len=train_loader.seq_len,
#     separate_qkv=True,
#     device=device,
# )

#%%
compute_prune_scores = True
load_prune_scores = False
save_prune_scores = True

learned_prune_scores: PruneScores = {}
cache_folder_name = ".prune_scores_cache"

if compute_prune_scores:
    learned_prune_scores = circuit_probing_prune_scores(
        model=gpt2,
        dataloader=train_loader,
        official_edges=ioi_true_edges(gpt2, token_positions=False),
        learning_rate=0.1,
        epochs=100,
        regularize_lambda=1,
        mask_fn="hard_concrete",
        dropout_p=0.0,
        show_train_graph=True,
        circuit_sizes=["true_size"],
        tree_optimisation=True,
        faithfulness_target="kl_div",
        validation_dataloader=test_loader,
    )
if load_prune_scores:
    filename = "ioi_no_toks_prune_scores-11-03-2024_22-44-19.pkl"
    learned_prune_scores = load_cache(cache_folder_name, filename)
if save_prune_scores:
    base_filename = "ioi_no_toks_mean_clean_corrupt_prune_scores"
    save_cache(learned_prune_scores, cache_folder_name, base_filename)


#%%

ioi_official_edges = ioi_true_edges(gpt2, token_positions=False)
ioi_official_edges_no_mlps = ioi_true_edges_mlp_0_only(gpt2, token_positions=False)
ioi_official_nodes = ioi_head_based_official_edges(gpt2, token_positions=False)

# ioi_official_edges = ioi_true_edges(gpt2, token_positions=True,
# word_idxs=test_loader.word_idxs)
# ioi_official_edges_no_mlps = ioi_true_edges_mlp_0_only(gpt2, token_positions=True,
# word_idxs=test_loader.word_idxs)
# ioi_official_nodes = ioi_head_based_official_edges(gpt2, token_positions=True,
# word_idxs=test_loader.word_idxs)

official_edge_ps = gpt2.circuit_prune_scores(ioi_official_edges)
official_edge_no_mlps_ps = gpt2.circuit_prune_scores(ioi_official_edges_no_mlps)
official_node_ps = gpt2.circuit_prune_scores(ioi_official_nodes)

for ps, name in [
    (learned_prune_scores, "Learned"),
    (official_edge_ps, "Official Edges"),
    (official_edge_no_mlps_ps, "Official Edges No MLPs"),
    (official_node_ps, "Official Nodes"),
]:
    circ_outs: CircuitOutputs = run_circuits(
        model=gpt2,
        dataloader=test_loader,
        test_edge_counts=[
            len(ioi_official_nodes),
            len(ioi_official_edges_no_mlps),
            len(ioi_official_edges),
        ],
        prune_scores=ps,
        patch_type=PatchType.TREE_PATCH,
        ablation_type=AblationType.RESAMPLE,
        render_graph=False,
        render_all_edges=False,
    )
    logit_diff_percent_means, logit_diff_percent_std, _ = answer_diff_percent(
        gpt2, test_loader, circ_outs
    )
    print(name, "prune scores")
    for n_edge, logit_diff_percent in logit_diff_percent_means:
        print(f"n_edge: {n_edge}, logit_diff_percent: {logit_diff_percent}")
    kl_divs: Measurements = measure_kl_div(gpt2, test_loader, circ_outs)
    for n_edge, kl_div in kl_divs:
        print(f"n_edge: {n_edge}, kl_div: {kl_div}")
    print()
    print()
    print()
# %%
