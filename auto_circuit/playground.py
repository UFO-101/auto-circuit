#%%
import os
import random
from typing import Dict, Optional, Set

import numpy as np
import torch as t
import torch.backends.mps
import transformer_lens as tl

import auto_circuit
import auto_circuit.data
import auto_circuit.prune
import auto_circuit.utils.graph_utils

# from auto_circuit.prune_functions.parameter_integrated_gradients import (
#     BaselineWeights,
# )
from auto_circuit.types import Edge, EdgeCounts, PatchType
from auto_circuit.utils.graph_utils import prepare_model
from auto_circuit.utils.misc import percent_gpu_mem_used, repo_path_to_abs_path


def rotation_matrix(x: t.Tensor, y: t.Tensor) -> t.Tensor:
    # Check that neither x or y is a zero vector
    assert t.all(x != 0), "x is a zero vector"
    assert t.all(y != 0), "y is a zero vector"

    assert x.device == y.device
    device = x.device

    # Normalize x to get u
    u = x / t.norm(x)

    # Project y onto the orthogonal complement of u and normalize to get v
    v = y - t.dot(u, y) * u
    v = v / t.norm(v)

    # Calculate cos(theta) and sin(theta)
    cos_theta = t.dot(x, y) / (t.norm(x) * t.norm(y))
    sin_theta = t.sqrt(1 - cos_theta**2)

    # Rotation matrix in the plane spanned by u and v
    R_theta = t.tensor([[cos_theta, -sin_theta], [sin_theta, cos_theta]], device=device)

    # Construct the full rotation matrix
    uv = t.stack([u, v], dim=1)
    identity = t.eye(len(x), device=device)
    R = identity - t.outer(u, u) - t.outer(v, v) + uv @ R_theta @ uv.T

    return R


def get_most_similar_embeddings(
    model: tl.HookedTransformer,
    resid: t.Tensor,
    answer: Optional[str] = None,
    top_k: int = 10,
):
    show_answer_rank = answer is not None
    answer = " cheese" if answer is None else answer
    unembeded = model.unembed(resid.unsqueeze(0).unsqueeze(0))
    answer_token = model.to_tokens(answer, prepend_bos=False).squeeze()
    answer_str_token = model.to_str_tokens(answer, prepend_bos=False)
    assert len(answer_str_token) == 1
    logits = unembeded.squeeze()
    probs = logits.softmax(dim=-1)

    sorted_token_probs, sorted_token_values = probs.sort(descending=True)
    # Janky way to get the index of the token in the sorted list
    correct_rank = torch.arange(len(sorted_token_values))[
        (sorted_token_values == answer_token).cpu()
    ].item()
    if show_answer_rank:
        print(
            f'\n"{answer_str_token[0]}" token rank:',
            f"{correct_rank: <8}",
            f"\nLogit: {logits[answer_token].item():5.2f}",
            f"Prob: {probs[answer_token].item():6.2%}",
        )
    for i in range(top_k):
        print(
            f"Top {i}th token. Logit: {logits[sorted_token_values[i]].item():5.2f}",
            f"Prob: {sorted_token_probs[i].item():6.2%}",
            f'Token: "{model.to_string(sorted_token_values[i])}"',
        )
    print(" ")


np.random.seed(0)
torch.manual_seed(0)
random.seed(0)
os.environ["TOKENIZERS_PARALLELISM"] = "False"

device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")
print("device", device)
model = tl.HookedTransformer.from_pretrained("gpt2-small", device=device)
# model = tl.HookedTransformer.from_pretrained("attn-only-4l", device=device)
# model = tl.HookedTransformer.from_pretrained("tiny-stories-33M", device=device)

model.cfg.use_attn_result = True
model.cfg.use_split_qkv_input = True
model.cfg.use_hook_mlp_in = True
assert model.tokenizer is not None
# model = t.compile(model)
model.eval()

# data_file = "datasets/indirect_object_identification.json"
# data_file = "datasets/greater_than_gpt2-small_prompts.json"
data_file = "datasets/docstring_prompts.json"
# data_file = "datasets/animal_diet_short_prompts.json"
# data_file = "datasets/mini_prompts.json"
print(percent_gpu_mem_used())
#%%

toks = (
    model.tokenizer([" zero", " one", " two", " three"], return_tensors="pt")[
        "input_ids"
    ]
    .squeeze()  # type: ignore
    .to(device)
)
print(toks)
num_embeds = model.embed(toks)
one_to_two_rotation = rotation_matrix(num_embeds[1], num_embeds[2])
assert t.allclose(num_embeds[2], one_to_two_rotation @ num_embeds[1], atol=1e-2)
one_rotated = one_to_two_rotation @ num_embeds[1]
print("one rotated to two")
get_most_similar_embeddings(model, one_rotated, top_k=10)

two_rotated = one_to_two_rotation @ num_embeds[2]
print("two embedded")
get_most_similar_embeddings(model, num_embeds[2], top_k=10)
print("two rotated to three")
get_most_similar_embeddings(model, two_rotated, top_k=10)


layernorm_num_embeds = model.blocks[0].ln1(num_embeds)  # type: ignore
layernorm_one_to_two_rotation = one_to_two_rotation

print("layernorm one")
get_most_similar_embeddings(model, layernorm_num_embeds[1], top_k=10)
layernorm_one_rotated = layernorm_one_to_two_rotation @ layernorm_num_embeds[1]
print("layernorm one rotated to two")
get_most_similar_embeddings(model, layernorm_one_rotated, top_k=10)

print("layernorm two")
get_most_similar_embeddings(model, layernorm_num_embeds[2], top_k=10)
layernorm_two_rotated = layernorm_one_to_two_rotation @ layernorm_num_embeds[2]
print("layernorm two rotated to three")
get_most_similar_embeddings(model, layernorm_two_rotated, top_k=10)

#%%
#%%

monarch_toks = (
    model.tokenizer([" man", " woman", " king", " queen"], return_tensors="pt")[
        "input_ids"
    ]
    .squeeze()  # type: ignore
    .to(device)
)
print(monarch_toks)
man, woman, king, queen = model.embed(monarch_toks)
man_to_women_rotation = rotation_matrix(man, woman)
assert t.allclose(woman, man_to_women_rotation @ man, atol=1e-1)
man_rotated = man_to_women_rotation @ man
print("man rotated to woman")
get_most_similar_embeddings(model, man_rotated, top_k=10)

print("king embedded")
get_most_similar_embeddings(model, king, top_k=10)
print("king rotated to queen")
king_rotated = man_to_women_rotation @ king
get_most_similar_embeddings(model, king_rotated, top_k=10)


layernorm_man, layernorm_woman, layernorm_king, layernorm_queen = model.blocks[0].ln1(
    t.stack([man, woman, king, queen])
)  # type: ignore
layernorm_man_to_woman_rotation = rotation_matrix(layernorm_man, layernorm_woman)
# layernorm_man_to_woman_rotation = man_to_women_rotation

print("layernorm man")
get_most_similar_embeddings(model, layernorm_man, top_k=10)
print("layernorm man rotated to woman")
layernorm_man_rotated = layernorm_man_to_woman_rotation @ layernorm_man
get_most_similar_embeddings(model, layernorm_man_rotated, top_k=10)

print("layernorm king")
get_most_similar_embeddings(model, layernorm_king, top_k=10)
layernorm_king_rotated = layernorm_man_to_woman_rotation @ layernorm_king
print("layernorm king rotated to queen")
get_most_similar_embeddings(model, layernorm_king_rotated, top_k=10)


#%%
monarch_toks = (
    model.tokenizer([" man", " woman", " king", " queen"], return_tensors="pt")[
        "input_ids"
    ]
    .squeeze()  # type: ignore
    .to(device)
)
print(monarch_toks)
man, woman, king, queen = model.embed(monarch_toks)
man_to_women_translation = woman - man
assert t.allclose(woman, man_to_women_translation + man, atol=1e-3)
print("man translated to woman")
man_translated = man_to_women_translation + man
get_most_similar_embeddings(model, man_translated, top_k=10)

print("king embedded")
get_most_similar_embeddings(model, king, top_k=10)
print("king translated to queen")
king_translated = man_to_women_translation + king
get_most_similar_embeddings(model, king_translated, top_k=10)


layernorm_man, layernorm_woman, layernorm_king, layernorm_queen = model.blocks[0].ln1(
    t.stack([man, woman, king, queen])
)  # type: ignore
# layernorm_man_to_woman_translation = layernorm_woman - layernorm_man
layernorm_man_to_woman_translation = man_to_women_translation

print("layernorm man")
get_most_similar_embeddings(model, layernorm_man, top_k=10)
print("layernorm man translated to woman")
layernorm_man_translated = layernorm_man_to_woman_translation + layernorm_man
get_most_similar_embeddings(model, layernorm_man_translated, top_k=10)

print("layernorm king")
get_most_similar_embeddings(model, layernorm_king, top_k=10)
layernorm_king_translated = layernorm_man_to_woman_translation + layernorm_king
print("layernorm king translated to queen")
get_most_similar_embeddings(model, layernorm_king_translated, top_k=10)
#%%
out_weight = model.blocks[3].mlp.module.W_out[200]  # type: ignore
get_most_similar_embeddings(model, out_weight)
layernorm_out_weight = model.blocks[-1].ln2(out_weight)  # type: ignore
get_most_similar_embeddings(model, layernorm_out_weight)

#%%
# ---------- Config ----------
experiment_type = PatchType.EDGE_PATCH
factorized = True
# pig_baseline, pig_samples = BaselineWeights.ZERO, 50
default_edge_count_type = EdgeCounts.LOGARITHMIC
one_tao = 6e-2
acdc_tao_range, acdc_tao_step = (one_tao, one_tao), one_tao

train_loader, test_loader = auto_circuit.data.load_datasets_from_json(
    model.tokenizer,
    repo_path_to_abs_path(data_file),
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
