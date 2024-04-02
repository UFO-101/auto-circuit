#%%
from collections import defaultdict
from typing import Dict, List

import plotly.graph_objects as go
import torch as t

from auto_circuit.data import BatchKey, load_datasets_from_json
from auto_circuit.experiment_utils import (
    load_tl_model,
)
from auto_circuit.metrics.official_circuits.circuits.sports_players_official import (
    sports_players_probe_true_edges,
)
from auto_circuit.types import AblationType
from auto_circuit.utils.ablation_activations import batch_src_ablations
from auto_circuit.utils.graph_utils import patch_mode, patchable_model, set_all_masks
from auto_circuit.utils.misc import repo_path_to_abs_path

device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")
model = load_tl_model("pythia-2.8b-deduped", device)

ans_toks = []
for answer in [" football", " basketball", " baseball"]:
    ans_tok = model.to_tokens(answer, prepend_bos=False)[0][0]
    ans_toks.append(ans_tok)
ans_toks = t.stack(ans_toks) if isinstance(ans_toks, list) else ans_toks
ans_embeds = model.unembed.W_U[:, ans_toks]
probe = model.blocks[16].attn.W_V[20] @ model.blocks[16].attn.W_O[20] @ ans_embeds

#%%
path = repo_path_to_abs_path(
    "datasets/sports-players/sports_players_pythia-2.8b-deduped_names.json"
)
_, test_loader = load_datasets_from_json(
    model=model,
    path=path,
    device=device,
    prepend_bos=True,
    batch_size=10,
    train_test_size=(0, 100),
    shuffle=True,
    return_seq_length=True,
    tail_divergence=True,
)

model = patchable_model(
    model=model,
    factorized=True,
    slice_output="last_seq",
    seq_len=test_loader.seq_len,
    separate_qkv=False,
    kv_caches=(test_loader.kv_cache,),
    device=device,
)

#%%
sports_players_edges = sports_players_probe_true_edges(
    model,
    word_idxs=test_loader.word_idxs,
    token_positions=True,
    seq_start_idx=test_loader.diverge_idx,
)
set_all_masks(model, 1.0)
for edge in sports_players_edges:
    edge.patch_mask(model).data[edge.patch_idx] = 0.0

patches: Dict[BatchKey, t.Tensor] = batch_src_ablations(
    model=model,
    dataloader=test_loader,
    ablation_type=AblationType.TOKENWISE_MEAN_CORRUPT,
    # ablation_type=AblationType.ZERO,
    # ablation_type=AblationType.RESAMPLE,
    # clean_corrupt="corrupt",
)

#%%
def batch_answers_to_probe_ans_idxs(batch_answers: t.Tensor) -> t.Tensor:
    ans_idxs = t.zeros_like(batch_answers)
    for idx, ans in enumerate(batch_answers):
        ans_idxs[idx] = (ans == ans_toks).nonzero(as_tuple=False).squeeze()
    return ans_idxs.flatten()


layers = list(range(2, 19))
# layers = list(range(2, 4))
correct: Dict[int, List[float]] = defaultdict(list)
for batch in test_loader:
    with patch_mode(model, patch_src_outs=patches[batch.key]):
        _, cache = model.run_with_cache(batch.clean)
    batch_ans = batch_answers_to_probe_ans_idxs(batch.answers)

    for layer in layers:
        embed = cache[f"blocks.{layer}.hook_resid_post"][model.out_slice]
        mean_act = (
            cache[f"blocks.{layer}.hook_resid_post"][model.out_slice]
            .detach()
            .clone()
            .mean(dim=0)
        )
        with t.inference_mode():
            probe_out = (embed - mean_act) @ probe
            probe_ans = probe_out.argmax(dim=-1)
            probe_correct = probe_ans == batch_ans
            correct[layer].append(probe_correct.float().mean().item())
correct_percents = {
    layer: sum(corrects) / len(corrects) for layer, corrects in correct.items()
}

# Plot layers vs. average probability of each sport
fig = go.Figure()
fig.add_trace(
    go.Scatter(x=list(correct_percents.keys()), y=list(correct_percents.values()))
)
fig.update_layout(
    xaxis_title="Layer",
    yaxis_title="Accuracy",
)
fig.show()

# %%
