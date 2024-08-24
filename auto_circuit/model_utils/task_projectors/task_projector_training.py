# %%
import math
from datetime import datetime
from math import pi
from typing import Dict

import plotly.express as px
import plotly.graph_objs as go
import torch as t
import transformer_lens as tl
from plotly.subplots import make_subplots
from torch.nn.functional import kl_div, log_softmax, relu
from transformer_lens import HookedTransformerKeyValueCache

from auto_circuit.data import PromptDataLoader, load_datasets_from_json
from auto_circuit.model_utils.task_projectors.projector_transformer import (
    ProjectorTransformer,
    get_projector_model,
)
from auto_circuit.model_utils.task_projectors.task_projector import TaskProjector
from auto_circuit.types import AutoencoderInput, BatchKey
from auto_circuit.utils.custom_tqdm import tqdm
from auto_circuit.utils.misc import (
    get_most_similar_embeddings,
    repo_path_to_abs_path,
)

# Constants are copied from the paper's code
mask_p, left, right, temp = 0.9, -0.1, 1.1, 2 / 3
p = (mask_p - left) / (right - left)
init_mask_val = math.log(p / (1 - p))
regularize_const = temp * math.log(-left / right)

# ISN'T THIS WHOLE THING BASICALLY EQUIVALENT TO SVD ON ACTIVATIONS?


def train_model(
    model: tl.HookedTransformer,
    dataloader: PromptDataLoader,
    input_act: AutoencoderInput,
    layernorm: bool,
    regularize_lambda: float,
    learning_rate: float,
    n_epochs: int,
) -> ProjectorTransformer:
    model.eval()
    device = model.cfg.device

    batch_diverge_idx = next(iter(dataloader)).batch_diverge_idx
    print("train_batch.batch_diverge_idx:", batch_diverge_idx)

    # Save default logprobs and initialize KV cache (one for each batch size)
    default_logprobs: Dict[BatchKey, t.Tensor] = {}
    kv_caches = {}
    for batch in (batch_pbar := tqdm(dataloader)):
        tks = batch.clean.to(device)
        batch_size = tks.shape[0]
        kv_cache = HookedTransformerKeyValueCache.init_cache(
            model.cfg, model.cfg.device, batch_size
        )
        common_prefix_batch = tks[:, : batch.batch_diverge_idx]
        with t.inference_mode():
            model(common_prefix_batch, past_kv_cache=kv_cache)
            default_logprobs[batch.key] = log_softmax(model(tks)[:, -1], dim=-1).clone()
        kv_cache.freeze()
        kv_caches[batch_size] = kv_cache

    model.cfg.use_attn_in = True
    projector_model: ProjectorTransformer = get_projector_model(
        model,
        input_act,
        mask_fn=None,
        load_pretrained=False,
        new_instance=False,
        # layers=[0, 2, 5, 14, 22],
        layer_idxs=[2, 5, 23],
        # layer_idxs=[2],
        # layers=[0, 2, 22],
        seq_idxs=[i - batch_diverge_idx for i in [15, 19]],
        layernorm=layernorm,
        load_file_task_name="sports_players",
        load_file_date="18-01-2024_22-16-14",
    )

    train_params = []
    for name, param in projector_model.named_parameters():
        param.requires_grad = False
    for _, module in projector_model.named_modules():
        if isinstance(module, TaskProjector):
            module.train()
            for name, param in module.named_parameters():
                print(name, param.shape)
                param.requires_grad = True
                train_params.append(param)

    optim = t.optim.adam.Adam(train_params, lr=learning_rate)

    loss_history, kl_loss_history, regularize_loss_history = [], [], []
    for epoch in (epoch_pbar := tqdm(range(n_epochs))):
        for batch in (batch_pbar := tqdm(dataloader)):
            toks = batch.clean.to(device)
            kv_cache = kv_caches[toks.shape[0]]
            toks = toks[:, batch.batch_diverge_idx :]
            optim.zero_grad()
            final_tok_logits = projector_model(toks, past_kv_cache=kv_cache)[:, -1]
            proj_logprobs = log_softmax(final_tok_logits, dim=-1)
            # nll = (-(default_probs * proj_logprobs).sum(dim=-1)).mean()
            kl = kl_div(
                proj_logprobs.flatten(end_dim=-2),
                default_logprobs[batch.key].flatten(end_dim=-2),
                reduction="batchmean",
                log_target=True,
            )

            # dim_weights = [p.dim_weights for p in projector_model.projectors]
            # reg = t.sigmoid(t.stack(dim_weights) - regularize_const).mean()
            # reg = t.stack(dim_weights)  # type: ignore
            # reg = ((reg.clamp(0, 1) * pi / 2).sin() + relu(reg - 1) ** 2).mean()

            eigvals = [t.linalg.eigvalsh(p.linear) for p in projector_model.projectors]
            eigvals = t.stack(eigvals).abs()
            # reg = (eigvals.pow(1/3) + t.nn.functional.relu(eigvals - 1)).mean()
            reg = ((eigvals.clamp(0, 1) * pi / 2).sin() + relu(eigvals - 1) ** 2).mean()

            loss = kl + reg * regularize_lambda
            loss.backward()
            optim.step()
            loss_history.append(loss.item())
            kl_loss_history.append(kl.item())
            regularize_loss_history.append(reg.item())
            desc = f"Loss: {loss.item():.3f} KL: {kl.item():.3f} Reg: {reg.item():.3f}"
            batch_pbar.set_description_str(desc)
            epoch_pbar.set_description_str(f"Epoch: {epoch} " + desc)

    data = {
        "Epoch": list(range(len(loss_history))),
        "Total Loss": loss_history,
        "KL Div": kl_loss_history,
        "Regularize": regularize_loss_history,
    }
    px.line(data, x="Epoch", y=["Total Loss", "KL Div", "Regularize"]).show()
    return projector_model


# model_name, dataset_name = "tiny-stories-33M", "roneneldan/TinyStories"
# model, dataloader = get_model_and_data(model_name, dataset_name)


# model_name = "pythia-70m-deduped"
# dataset_name = "datasets/capital_cities_pythia-70m-deduped_prompts.json"
model_name = "pythia-410m-deduped"
dataset_name = "datasets/sports-players/sports_players_pythia-410m-deduped_prompts.json"
input_act: AutoencoderInput = "resid"
seq_start_idx = 13
layernorm = False

model = tl.HookedTransformer.from_pretrained_no_processing(model_name, device="cpu")
assert model.cfg.device is not None
train_dataloader, test_dataloader = load_datasets_from_json(
    model,
    repo_path_to_abs_path(dataset_name),
    device=t.device(model.cfg.device),
    prepend_bos=True,
    batch_size=6,
    train_test_size=(800, 200),
    return_seq_length=True,
    shuffle=True,
    pad=True,
)
projector_model = train_model(
    model,
    train_dataloader,
    input_act,
    layernorm,
    regularize_lambda=0.5,
    learning_rate=1e-3,
    n_epochs=200,
)

# %%
batch_diverge_idx = next(iter(test_dataloader)).batch_diverge_idx
seq_idxs = (
    [idx + batch_diverge_idx for idx in projector_model.seq_idxs]
    if projector_model.seq_idxs
    else None
)
# %%
for layer, projector in zip(projector_model.layer_idxs, projector_model.projectors):
    projector.to("cpu")
    task_name = "sports_players"
    folder = repo_path_to_abs_path(".projector_cache")
    filename_pt_1 = f"projector_{model_name}_layer_{layer}"
    filename_pt_2 = f"_seq_{projector_model.seq_idxs}_task_{task_name}"
    filename_pt_3 = f"_layernorm_{layernorm}"
    filename = filename_pt_1 + filename_pt_2 + filename_pt_3
    dt_string = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    t.save(projector.state_dict(), folder / f"{filename}-{dt_string}.pt")
# %%
model = tl.HookedTransformer.from_pretrained_no_processing(model_name, device="cpu")
model.cfg.use_attn_in = True
# %%
default_logprobs = []
projected_logprobs = []

test_kv_caches = {}
dataloader = test_dataloader
for batch in (batch_pbar := tqdm(dataloader)):
    tks = batch.clean.to(model.cfg.device)
    batch_size = tks.shape[0]
    kv_cache = HookedTransformerKeyValueCache.init_cache(
        model.cfg, model.cfg.device, batch_size
    )
    common_prefix_batch = tks[:, : batch.batch_diverge_idx]
    with t.inference_mode():
        model(common_prefix_batch, past_kv_cache=kv_cache)
    kv_cache.freeze()
    test_kv_caches[batch_size] = kv_cache

for batch in dataloader:
    default_logprobs.append(log_softmax(model(batch.clean)[:, -1], dim=-1))
    toks = batch.clean[:, batch_diverge_idx:]
    batch_size = toks.shape[0]
    kv_cache = test_kv_caches[batch_size]
    proj_out = projector_model(toks, past_kv_cache=kv_cache)[:, -1]
    projected_logprobs.append(log_softmax(proj_out, dim=-1))

    for tok, proj in zip(toks, proj_out):
        print(model.to_string(tok))
        get_most_similar_embeddings(model, proj, top_k=10)


avg_kl_div = t.nn.functional.kl_div(
    t.cat(projected_logprobs).flatten(end_dim=-2),
    t.cat(default_logprobs).flatten(end_dim=-2),
    reduction="batchmean",
    log_target=True,
)
print("avg_kl_div", avg_kl_div.item())
# print("avg sqr eigvals", t.stack(eigvals).pow(2).mean().item())
# print("avg_abs_eigvals", t.stack(eigvals).abs().mean().item())
# print(eigvals[0])
# %%
eigvals = []
for projector in projector_model.projectors:
    eigvals.append(t.linalg.eigvalsh(projector.linear))

# Create an NxM subplot grid
n_layers = len(eigvals)
column_titles = [f"Layer {i}" for i in projector_model.layer_idxs]
row_count = len(seq_idxs) if seq_idxs else eigvals[0].shape[0]
row_titles = model.to_str_tokens(next(iter(train_dataloader)).clean[0])
row_titles = [row_titles[i] for i in seq_idxs] if seq_idxs else row_titles

fig = make_subplots(rows=row_count, cols=n_layers, shared_yaxes=True)
for col, tensor in enumerate(eigvals, start=1):
    for row in range(row_count):
        points = tensor[row].detach().clone().cpu()
        fig.add_trace(
            go.Scatter(y=points, showlegend=False), row=row_count - row, col=col
        )
fig.update_layout(height=200 * row_count, width=200 * n_layers)
for col, title in enumerate(column_titles, start=1):
    fig.update_xaxes(title_text=title, row=1, col=col, side="top")

# Adding row titles
for row, title in enumerate(row_titles):
    fig.update_yaxes(title_text=title, row=row_count - row, col=1)
fig.show()

# %%
eigvals, eigvectors = [], []
for projector in projector_model.projectors:
    vals, vecs = t.linalg.eigh(projector.linear)
    eigvals.append(vals.detach().clone())
    eigvectors.append(vecs.detach().clone())

activations = []
answers = []
labels = []
for batch in train_dataloader:
    answers.extend(model.to_string(batch.answers))
    labels.extend(
        model.to_string(batch.clean[:, batch_diverge_idx : batch_diverge_idx + 3])
    )
    with t.inference_mode():
        _, cache = model.run_with_cache(batch.clean)
    for idx, projector in enumerate(projector_model.projectors):
        act = cache[projector.wrapped_hook.name]
        if act.ndim == 4:
            act = act[:, :, 0]
        if layernorm:
            act = t.nn.functional.layer_norm(act, act.shape[-1:])
        if len(activations) <= idx:
            activations.append(act)
        else:
            activations[idx] = t.cat([activations[idx], act], dim=0)
print("len(activations)", len(activations))
print("activations[0].shape", activations[0].shape)

answer_cols = ["red", "green", "blue", "yellow", "purple", "orange"]
answer_col_map = {answer: col for answer, col in zip(list(set(answers)), answer_cols)}
colors = [answer_col_map[answer] for answer in answers]

fig = make_subplots(rows=row_count, cols=n_layers, shared_yaxes=True)
for col, (vals, vecs, acts) in enumerate(
    zip(eigvals, eigvectors, activations), start=1
):
    top_3_vals = vals.abs().topk(3).indices
    gathers_idxs = top_3_vals.unsqueeze(-2).expand(-1, vecs.size(-2), -1)
    top_3_vecs = t.gather(vecs, -1, gathers_idxs)
    proj_acts = (acts[:, seq_idxs].unsqueeze(-2) @ top_3_vecs).squeeze(-2)
    for row in range(row_count):
        xs, ys, zs = [proj_acts[:, row, i].tolist() for i in range(3)]
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                showlegend=False,
                mode="markers",
                hovertext=labels,
                marker=dict(color=colors),
            ),
            row=row_count - row,
            col=col,
        )
fig.update_layout(height=200 * row_count, width=200 * n_layers)
for col, title in enumerate(column_titles, start=1):
    fig.update_xaxes(title_text=title, row=1, col=col, side="top")

for row, title in enumerate(row_titles):
    fig.update_yaxes(title_text=title, row=row_count - row, col=1)
fig.show()

# %%
