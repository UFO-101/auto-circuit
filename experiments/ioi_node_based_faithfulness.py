#%%
import torch as t
import transformer_lens as tl

from auto_circuit.data import load_datasets_from_json
from auto_circuit.metrics.official_circuits.circuits.ioi_official import (
    ioi_head_based_official_edges,
)
from auto_circuit.metrics.prune_metrics.answer_diff_percent import answer_diff_percent
from auto_circuit.prune import run_circuits
from auto_circuit.types import (
    AblationType,
    CircuitOutputs,
    Measurements,
    PatchType,
    PruneScores,
)
from auto_circuit.utils.graph_utils import (
    edge_counts_util,
    patchable_model,
)
from auto_circuit.utils.misc import repo_path_to_abs_path

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

train_loader, test_loader = load_datasets_from_json(
    model=gpt2,
    path=repo_path_to_abs_path("datasets/ioi_single_template_prompts.json"),
    device=device,
    batch_size=100,
    train_test_split=[100, 100],
    length_limit=200,
    return_seq_length=True,
    tail_divergence=False,
)

gpt2 = patchable_model(
    model=gpt2,
    factorized=False,
    slice_output="last_seq",
    seq_len=train_loader.seq_len,
    separate_qkv=True,
    device=device,
)

ioi_node_edges = ioi_head_based_official_edges(
    gpt2, token_positions=True, seq_start_idx=train_loader.diverge_idx
)
circuit_ps: PruneScores = gpt2.circuit_prune_scores(ioi_node_edges)

circ_outs: CircuitOutputs = run_circuits(
    model=gpt2,
    dataloader=train_loader,
    test_edge_counts=edge_counts_util(gpt2.edges, prune_scores=circuit_ps),
    prune_scores=circuit_ps,
    patch_type=PatchType.TREE_PATCH,
    ablation_type=AblationType.TOKENWISE_MEAN_CORRUPT,
)
logit_diff_percents: Measurements = answer_diff_percent(gpt2, train_loader, circ_outs)
for n_edge, logit_diff_percent in logit_diff_percents:
    print(f"{n_edge} edges: {logit_diff_percent:.2f}%")
