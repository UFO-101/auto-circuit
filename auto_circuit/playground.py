#%%
import json
import os
from typing import Dict, Tuple

import blobfile as bf
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import torch as t
import torch.backends.mps
import transformer_lens as tl
from torch.nn.utils import parametrizations

from auto_circuit.utils.misc import get_most_similar_embeddings

# from auto_circuit.prune_functions.parameter_integrated_gradients import (
#     BaselineWeights,
# )


def rotation_matrix(
    x: t.Tensor, y: t.Tensor, lerp: float = 1.0
) -> Tuple[t.Tensor, t.Tensor]:
    # Based on: https://math.stackexchange.com/questions/598750/finding-the-rotation-matrix-in-n-dimensions
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

    # Interpolate between the identity matrix and the rotation matrix
    if lerp != 1.0:
        theta = t.atan2(sin_theta, cos_theta)
        cos_theta = t.cos(theta * lerp)
        sin_theta = t.sin(theta * lerp)

    # Rotation matrix in the plane spanned by u and v
    R_theta = t.tensor([[cos_theta, -sin_theta], [sin_theta, cos_theta]], device=device)

    # Construct the full rotation matrix
    uv = t.stack([u, v], dim=1)
    identity = t.eye(len(x), device=device)
    R = identity - t.outer(u, u) - t.outer(v, v) + uv @ R_theta @ uv.T

    return R, uv


# np.random.seed(0)
# torch.manual_seed(0)
# random.seed(0)
os.environ["TOKENIZERS_PARALLELISM"] = "False"

device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")
print("device", device)
model = tl.HookedTransformer.from_pretrained(
    # "pythia-410m-deduped",
    "gpt2",
    fold_ln=True,
    center_writing_weights=True,
    center_unembed=False,
    # "tiny-stories-2L-33M",
    device=device
    # "tiny-stories-33M", device=device
)
model.eval()
#%%
modelb16 = tl.HookedTransformer.from_pretrained(
    # "pythia-410m-deduped",
    "gpt2",
    fold_ln=True,
    center_writing_weights=True,
    center_unembed=False,
    # "tiny-stories-2L-33M",
    device=device,
    dtype="bfloat16"
    # "tiny-stories-33M", device=device
)
model16 = tl.HookedTransformer.from_pretrained(
    # "pythia-410m-deduped",
    "gpt2",
    fold_ln=True,
    center_writing_weights=True,
    center_unembed=False,
    # "tiny-stories-2L-33M",
    device=device,
    dtype="float16"
    # "tiny-stories-33M", device=device
)
model16.eval()
modelb16.eval()
#%%
test_prompt = "The sun rises in the"
tl.utils.test_prompt(test_prompt, "stone", model, top_k=5)
tl.utils.test_prompt(test_prompt, "stone", model16, top_k=5)
tl.utils.test_prompt(test_prompt, "stone", modelb16, top_k=5)
#%%


country_to_captial: Dict[str, str] = {
    "country": "capital",
    "France": "Paris",
    "Hungary": "Budapest",
    "China": "Beijing",
    "Germany": "Berlin",
    # "Italy": "Rome",
    "Japan": "Tokyo",
    # "Russia": "Moscow",
    # 'Canada': 'Ottawa',
    # 'Australia': 'Canberra',
    # "Egypt": "Cairo",
    # 'Turkey': 'Ankara',
    # "Spain": "Madrid",
    "Sweden": "Stockholm",
    "Norway": "Oslo",
    "Denmark": "Copenhagen",
    "Finland": "Helsinki",
    "Poland": "Warsaw",
    "Indonesia": "Jakarta",
    # "Thailand": "Bangkok",
    "Cuba": "Havana",
    "Chile": "Santiago",
    "Greece": "Athens",
    "Portugal": "Lisbon",
    # "Austria": "Vienna",
    # "Belgium": "Brussels",
    "Philippines": "Manila",
    "Peru": "Lima",
    "Ireland": "Dublin",
    "Israel": "Jerusalem",
    # 'Switzerland': 'Bern',
    "Netherlands": "Amsterdam",
    "Singapore": "Singapore",  # Interesting case
    # "Pakistan": "Islamabad",
    "Lebanon": "Beirut",
}
present_to_past: Dict[str, str] = {
    "present": "past",
    "is": "was",
    "run": "ran",
    "eat": "ate",
    "drink": "drank",
    "go": "went",
    "see": "saw",
    "hear": "heard",
    "speak": "spoke",
    "write": "wrote",
    # "read": "read",
    "do": "did",
    "have": "had",
    "give": "gave",
    "take": "took",
    "make": "made",
    "know": "knew",
    "think": "thought",
    "find": "found",
    "tell": "told",
    "become": "became",
    "leave": "left",
    "feel": "felt",
    # "put": "put",
    "bring": "brought",
    "begin": "began",
    "keep": "kept",
    "hold": "held",
    "stand": "stood",
    "play": "played",
    "light": "lit",
}
male_to_female: Dict[str, str] = {
    "male": "female",
    "king": "queen",
    "actor": "actress",
    "brother": "sister",
    "father": "mother",
    "son": "daughter",
    "nephew": "niece",
    "uncle": "aunt",
    "wizard": "witch",
    "prince": "princess",
    "husband": "wife",
    "boy": "girl",
    "man": "woman",
    "hero": "heroine",
    "lord": "lady",
    "monk": "nun",
    "groom": "bride",
    "bull": "cow",
    "god": "goddess",
}

for k, v in country_to_captial.items():
    tl.utils.test_prompt(f"The capital of {k} is the city of", v, model, top_k=5)
#%%

word_mapping = present_to_past
# for i, (k, v) in enumerate(word_mapping.items()):
#     print("key:", model.to_str_tokens(" " + k, prepend_bos=False))
#     print("value:", model.to_str_tokens(" " + v, prepend_bos=False))
key_toks = model.to_tokens(
    [" " + s for s in word_mapping.keys()], prepend_bos=False
).squeeze()
val_toks = model.to_tokens(
    [" " + s for s in word_mapping.values()], prepend_bos=False
).squeeze()

# print("EMBEDDINGS")
# key_embeds = model.embed(key_toks).detach().clone()  # [n_toks, embed_dim]
# val_embeds = model.embed(val_toks).detach().clone()  # [n_toks, embed_dim]
# print("average key embedding norm:", key_embeds.norm(dim=-1).mean().item())
# print("average val embedding norm:", val_embeds.norm(dim=-1).mean().item())
# key_embeds = model.ln_final(model.embed(key_toks)).detach().clone()
# val_embeds = model.ln_final(model.embed(val_toks)).detach().clone()
# key_embeds = model.blocks[3].ln2(model.embed(key_toks)).detach().clone()
# val_embeds = model.blocks[3].ln2(model.embed(val_toks)).detach().clone()
# print("LAYERNORM NO BIAS NO SCALE")
key_embeds = (
    t.nn.functional.layer_norm(model.embed(key_toks), [model.cfg.d_model])
    .detach()
    .clone()
)  # [n_toks, embed_dim]
val_embeds = (
    t.nn.functional.layer_norm(model.embed(val_toks), [model.cfg.d_model])
    .detach()
    .clone()
)  # [n_toks, embed_dim]
# val_embeds = model.ln_final(model.embed(val_toks)).detach().clone()
print("average key embedding norm:", key_embeds.norm(dim=-1).mean().item())
print("average val embedding norm:", val_embeds.norm(dim=-1).mean().item())

idxs = torch.randperm(len(word_mapping))
train_len = key_embeds.shape[0] - 6
concept_key_embed, train_key_embeds, test_key_embeds = (
    key_embeds[0],
    key_embeds[idxs[1 : 1 + train_len]],
    key_embeds[idxs[1 + train_len :]],
)
concept_val_embed, train_val_embeds, test_val_embeds = (
    val_embeds[0],
    val_embeds[idxs[1 : 1 + train_len]],
    val_embeds[idxs[1 + train_len :]],
)
# ln_final_bias = model.ln_final.b.detach().clone()

# linear_map = t.zeros(
#     [val_embeds.shape[1], key_embeds.shape[1]], device=device, requires_grad=True
# )
translate = t.zeros([key_embeds.shape[1]], device=device, requires_grad=True)
scale = t.ones([key_embeds.shape[1]], device=device, requires_grad=True)
translate_2 = t.zeros([key_embeds.shape[1]], device=device, requires_grad=True)
rotation_vectors = t.rand([2, key_embeds.shape[1]], device=device, requires_grad=True)

learned_rotation = t.nn.Linear(
    key_embeds.shape[1], key_embeds.shape[1], bias=False, device=device
)

linear_map = parametrizations.orthogonal(learned_rotation, "weight")

# optim = t.optim.Adam(linear_map.parameters(), lr=0.01)
optim = t.optim.Adam(list(linear_map.parameters()) + [translate], lr=0.01)
# optim = t.optim.Adam([translate], lr=0.01)
# optim = t.optim.Adam([linear_map, translate], lr=0.01)
# optim = t.optim.Adam([rotation_vectors], lr=0.01)
# optim = t.optim.Adam([rotation_vectors, translate], lr=0.01)
# optim = t.optim.Adam([rotation_vectors, translate, translate_2], lr=0.01)
# optim = t.optim.Adam([linear_map, translate, translate_2], lr=0.01)
# optim = t.optim.Adam([rotation_vectors, translate, scale], lr=0.01)
# optim = t.optim.Adam([translate, scale], lr=0.01)
# optim = t.optim.Adam([scale], lr=0.01)


def pred_from_embeds(embeds: t.Tensor, lerp: float = 1.0) -> t.Tensor:
    # linear_map, proj = rotation_matrix(
    #     rotation_vectors[0], rotation_vectors[1], lerp=lerp
    # )
    # pred = learned_rotation(embeds)
    pred = learned_rotation(embeds + translate) - translate
    # pred = embeds @ linear_map
    # pred = embeds + (translate * lerp)
    # pred = embeds * scale
    # pred = (embeds @ linear_map) + translate
    # pred = ((embeds + translate_2) @ linear_map) + translate
    # pred = ((embeds + translate) @ linear_map) - translate
    # pred = ((embeds - ln_final_bias) @ linear_map) + ln_final_bias
    # pred = ((embeds * scale) + translate) @ linear_map
    # pred = ((embeds @ linear_map)* scale) + translate
    # pred = (embeds * scale) + translate
    return pred


def loss_fn(pred: t.Tensor, target: t.Tensor) -> t.Tensor:
    # loss = (target - pred).pow(2).mean()
    # loss = 1 - t.nn.functional.cosine_similarity(pred, target).mean()
    loss = 1 - t.nn.functional.cosine_similarity(pred, target).mean()
    return loss


losses = []
for epoch in range(1000):
    optim.zero_grad()
    pred = pred_from_embeds(train_key_embeds)
    loss = loss_fn(pred, train_val_embeds)
    loss.backward()
    optim.step()
    losses.append(loss.item()) if epoch % 10 == 0 else None

px.line(y=losses).show()

# linear_map = rotation_matrix(rotation_vectors[0], rotation_vectors[1])  # type: ignore
test_pred = pred_from_embeds(test_key_embeds)
print("Test loss:", loss_fn(test_pred, test_val_embeds).item())

print("Train data example")
get_most_similar_embeddings(
    model, train_key_embeds[0], top_k=5, apply_ln_final=False, apply_unembed=True
)
print()
train_pred_0 = pred_from_embeds(train_key_embeds[0])
get_most_similar_embeddings(
    model, train_pred_0, top_k=5, apply_ln_final=False, apply_unembed=True
)

for i in range(5):
    print("Test data example")
    get_most_similar_embeddings(
        model, test_key_embeds[i], top_k=5, apply_ln_final=False, apply_unembed=True
    )
    print()
    test_pred_i = pred_from_embeds(test_key_embeds[i])
    get_most_similar_embeddings(
        model, test_pred_i, top_k=5, apply_ln_final=False, apply_unembed=True
    )

#%%
get_most_similar_embeddings(model, train_key_embeds[0], apply_embed=True)
#%%
linear_map, proj = rotation_matrix(rotation_vectors[0], rotation_vectors[1], lerp=1.0)
projected_keys = ((key_embeds[1:]) @ proj).detach().clone().cpu()
projected_vals = ((val_embeds[1:]) @ proj).detach().clone().cpu()
projected_translate = (translate @ proj).detach().clone().cpu()

projected_rotated_keys = (
    ((((translate + key_embeds[1:]) @ linear_map) - translate) @ proj)
    .detach()
    .clone()
    .cpu()
)

keys_x, keys_y = projected_keys[:, 0], projected_keys[:, 1]
vals_x, vals_y = projected_vals[:, 0], projected_vals[:, 1]
preds_x, preds_y = projected_rotated_keys[:, 0], projected_rotated_keys[:, 1]

# Create a scatter plot
fig = go.Figure()

# Adding scatter plot for keys
fig.add_trace(
    go.Scatter(
        x=keys_x,
        y=keys_y,
        mode="markers+text",
        name="Keys",
        text=list(word_mapping.keys())[1:],
        marker=dict(size=10, color="blue"),
    )
)

# Adding scatter plot for values
fig.add_trace(
    go.Scatter(
        x=vals_x,
        y=vals_y,
        mode="markers+text",
        name="Vals",
        text=list(word_mapping.values())[1:],
        marker=dict(size=10, color="red"),
    )
)

# Adding scatter plot for predictions
# fig.add_trace(go.Scatter(x=preds_x, y=preds_y,
#                             mode='markers+text',
#                             name='Predictions',
#                             text=list(word_mapping.values())[1:],
#                             marker=dict(size=10, color='orange')))


# Adding lines connecting keys and values
for i in range(keys_x.shape[0]):
    fig.add_trace(
        go.Scatter(
            x=[keys_x[i], vals_x[i]],
            y=[keys_y[i], vals_y[i]],
            mode="lines",
            line=dict(color="grey", width=1),
            showlegend=False,
        )
    )

# # Adding lines connecting keys and predictions
# for i in range(keys_x.shape[0]):
#     fig.add_trace(go.Scatter(x=[keys_x[i], preds_x[i]],
#                              y=[keys_y[i], preds_y[i]],
#                              mode='lines',
#                              line=dict(color='green', width=1),
#                              showlegend=False))

# fig.add_trace(go.Scatter(x=[projected_translate[0]],
#                             y=[projected_translate[1]],
#                             mode='markers+text',
#                             name='Translate',
#                             text=["Translate"],
#                             marker=dict(size=10, color='green')))

# Update layout for a better look
fig.update_layout(
    title="2D Scatter plot of Keys and Vals with Connections",
    xaxis_title="Dimension 1",
    yaxis_title="Dimension 2",
    legend_title="Legend",
)

# Show plot
fig.show()

# print("Interpolated test data example")
# get_most_similar_embeddings(model, test_key_embeds[0], top_k=3)
# for lerp in t.linspace(0, 1, 10):
#     print(f"lerp: {lerp.item():.2f}")
#     test_pred_i = pred_from_embeds(test_key_embeds[0], lerp=lerp.item())
#     get_most_similar_embeddings(model, test_pred_i, top_k=10)

#%%
# Resid_delta_mlp, Layer 3 are the most interpretable
layer_index = 6  # in range(12)
autoencoder_input = ["mlp_post_act", "resid_delta_mlp"][0]
feature_idx = 25890
base_url = "az://openaipublic/sparse-autoencoder/gpt2-small/"
weight_file = f"{autoencoder_input}/autoencoders/{layer_index}.pt"
feat_file = f"{autoencoder_input}/collated_activations/{layer_index}/{feature_idx}.json"
url_to_get = base_url + feat_file

bf.stat(url_to_get)

with bf.BlobFile(url_to_get, mode="rb") as f:
    data = json.load(f)

examples = data["most_positive_activation_records"]
# examples = data["random_sample"]
cell_vals = [d["activations"] for d in examples]
text = [d["tokens"] for d in examples]
fig = ff.create_annotated_heatmap(cell_vals, annotation_text=text, colorscale="Viridis")
prompt_len = min([len(acts) for acts in cell_vals])
fig.layout.width = 75 * prompt_len  # type: ignore
fig.layout.height = 32 * len(cell_vals)  # type: ignore
fig.show()
