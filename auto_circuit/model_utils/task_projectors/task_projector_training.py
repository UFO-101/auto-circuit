#%%
import math
from math import pi

import plotly.express as px
import plotly.graph_objs as go
import torch as t
import transformer_lens as tl
from plotly.subplots import make_subplots
from torch.nn.functional import kl_div, log_softmax, relu

from auto_circuit.data import PromptDataLoader, load_datasets_from_json
from auto_circuit.model_utils.task_projectors.projector_transformer import (
    ProjectorTransformer,
    get_projector_model,
)
from auto_circuit.model_utils.task_projectors.task_projector import TaskProjector
from auto_circuit.utils.custom_tqdm import tqdm
from auto_circuit.utils.misc import repo_path_to_abs_path

# Constants are copied from the paper's code
mask_p, left, right, temp = 0.9, -0.1, 1.1, 2 / 3
p = (mask_p - left) / (right - left)
init_mask_val = math.log(p / (1 - p))
regularize_const = temp * math.log(-left / right)


# ISN'T THIS WHOLE THING BASICALLY EQUIVALENT TO SVD ON ACTIVATIONS?


def train_model(
    model: tl.HookedTransformer,
    dataloader: PromptDataLoader,
    n_latents: int,
    regularize_lambda: float,
    learning_rate: float,
    n_epochs: int,
) -> ProjectorTransformer:
    model.eval()
    device = model.cfg.device

    # REMEMBER TO TRY POST LAYERNORM
    projector_model: ProjectorTransformer = get_projector_model(
        model,
        "resid_delta_mlp",
        "hard_concrete",
        load_pretrained=False,
        seq_len=dataloader.seq_len,
        new_instance=True,
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

    optim = t.optim.Adam(train_params, lr=learning_rate)

    loss_history, cross_entropy_loss_history, regularize_loss_history = [], [], []
    for epoch in (epoch_pbar := tqdm(range(n_epochs))):
        for batch in (batch_pbar := tqdm(dataloader)):
            toks = batch.clean.to(device)
            optim.zero_grad()
            with t.no_grad():
                default_probs = log_softmax(model(toks)[:, -1], dim=-1).clone()
            proj_lobgprobs = log_softmax(projector_model(toks)[:, -1], dim=-1)
            # nll = (-(default_probs * proj_lobgprobs).sum(dim=-1)).mean()
            nll = kl_div(
                proj_lobgprobs.flatten(end_dim=-2),
                default_probs.flatten(end_dim=-2),
                reduction="batchmean",
                log_target=True,
            )

            # dim_weights = [p.dim_weights for p in projector_model.projectors]
            # reg = t.sigmoid(t.stack(dim_weights) - regularize_const).mean()
            # reg = t.stack(dim_weights).mean()  # type: ignore

            eigvals = [t.linalg.eigvalsh(p.linear) for p in projector_model.projectors]
            eigvals = t.stack(eigvals).abs()
            # reg = (eigvals.pow(1/3) + t.nn.functional.relu(eigvals - 1)).mean()
            reg = ((eigvals.clamp(0, 1) * pi / 2).sin() + relu(eigvals - 1)).mean()

            loss = nll + reg * regularize_lambda
            loss.backward()
            optim.step()
            loss_history.append(loss.item())
            cross_entropy_loss_history.append(nll.item())
            regularize_loss_history.append(reg.item())
            desc = (
                f"Loss: {loss.item():.3f} NLL: {nll.item():.3f} Reg: {reg.item():.3f}"
            )
            batch_pbar.set_description_str(desc)
            epoch_pbar.set_description_str(f"Epoch: {epoch} " + desc)

    data = {
        "Epoch": list(range(len(loss_history))),
        "Total Loss": loss_history,
        "Cross Entropy Loss": cross_entropy_loss_history,
        "Regularize Loss": regularize_loss_history,
    }
    px.line(
        data, x="Epoch", y=["Total Loss", "Cross Entropy Loss", "Regularize Loss"]
    ).show()
    return projector_model


# model_name, dataset_name = "tiny-stories-33M", "roneneldan/TinyStories"
# model, dataloader = get_model_and_data(model_name, dataset_name)


model_name, dataset_name = (
    "pythia-70m-deduped",
    "datasets/capital_cities_pythia-70m-deduped_prompts.json",
)
model = tl.HookedTransformer.from_pretrained_no_processing(model_name, device="cpu")
assert model.cfg.device is not None
train_dataloader, test_dataloader = load_datasets_from_json(
    model.tokenizer,
    repo_path_to_abs_path(dataset_name),
    device=t.device(model.cfg.device),
    prepend_bos=True,
    batch_size=16,
    train_test_split=[0.8, 0.2],
    length_limit=1000,
    return_seq_length=True,
    random_subet=True,
    pad=True,
)
projector_model = train_model(
    model,
    train_dataloader,
    n_latents=64,
    regularize_lambda=1,
    learning_rate=1e-2,
    n_epochs=1500,
)

# %%
for layer, projector in enumerate(projector_model.projectors):
    projector.to("cpu")
    cache_dir = repo_path_to_abs_path(".projector_cache").resolve()
    task_name = "capital_cities"
    filename = f"projector_{model_name}_layer_{layer}_task_{task_name}.pt"
    with open(cache_dir / filename, "wb") as f:
        t.save(projector.state_dict(), f)
# %%
default_logprobs = []
projected_logprobs = []
for batch in train_dataloader:
    default_logprobs.append(log_softmax(model(batch.clean)[:, -1], dim=-1))
    projected_logprobs.append(log_softmax(projector_model(batch.clean)[:, -1], dim=-1))

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
#%%
eigvals = []
for projector in projector_model.projectors:
    eigvals.append(t.linalg.eigvalsh(projector.linear))

# Create an NxM subplot grid
seq_len = eigvals[0].shape[0]
n_layers = len(eigvals)
row_titles = model.to_str_tokens(next(iter(train_dataloader)).clean[0])
column_titles = [f"Layer {i}" for i in range(n_layers)]
fig = make_subplots(rows=seq_len, cols=n_layers, shared_yaxes=True)
for col, tensor in enumerate(eigvals, start=1):
    for row in range(seq_len):
        fig.add_trace(
            go.Scatter(y=tensor[row].detach().clone().cpu(), showlegend=False),
            row=seq_len - row,
            col=col,
        )
# n_rows, n_cols = eigvals[0].shape[0], len(eigvals)
# data = t.stack(eigvals).transpose(0, 1).flatten(end_dim=-2).detach().clone().cpu()
# fig = px.bar(data, facet_col=0, facet_col_wrap=len(eigvals))
fig.update_layout(height=200 * seq_len, width=200 * n_layers)
for col, title in enumerate(column_titles, start=1):
    fig.update_xaxes(title_text=title, row=1, col=col, side="top")

# Adding row titles
for row, title in enumerate(row_titles):
    fig.update_yaxes(title_text=title, row=seq_len - row, col=1)
fig.show()

#%%
eigvals, eigvectors = [], []
for projector in projector_model.projectors:
    vals, vecs = t.linalg.eigh(projector.linear)
    eigvals.append(vals.detach().clone())
    eigvectors.append(vecs.detach().clone())

activations = []
for batch in train_dataloader:
    with t.inference_mode():
        _, cache = model.run_with_cache(batch.clean)
    for block in range(model.cfg.n_layers):
        mlp_out = cache[f"blocks.{block}.hook_mlp_out"]
        if len(activations) <= block:
            activations.append(mlp_out)
        else:
            activations[block] = t.cat([activations[block], mlp_out], dim=0)
print("activations[0].shape", activations[0].shape)
label_tok_idx = 4
clean_batch = next(iter(train_dataloader)).clean
labels = [
    str_toks[label_tok_idx]
    for str_toks in [
        model.to_str_tokens(clean_batch[i]) for i in range(clean_batch.shape[0])
    ]
]

fig = make_subplots(rows=seq_len, cols=n_layers, shared_yaxes=True)
for col, (vals, vecs, acts) in enumerate(
    zip(eigvals, eigvectors, activations), start=1
):
    print("acts.shape", acts.shape)
    top_3_vals = vals.abs().topk(3).indices
    print("col", col, "top_3_vals", top_3_vals.shape, "vecs", vecs.shape)
    # top_3_vecs = vecs[t.arange(vecs.shape[0])[:, None, None], :, top_3_vals]
    top_3_vecs = t.gather(
        vecs, -1, top_3_vals.unsqueeze(-2).expand(-1, vecs.size(-2), -1)
    )
    print("top_3_vecs.shape", top_3_vecs.shape)
    projected_acts = (acts.unsqueeze(-2) @ top_3_vecs).squeeze(-2)
    print("projected_acts.shape", projected_acts.shape)
    for row in range(seq_len):
        xs, ys, zs = [projected_acts[:, row, i].tolist() for i in range(3)]
        fig.add_trace(
            go.Scatter(x=xs, y=ys, showlegend=False, mode="markers", text=labels),
            row=seq_len - row,
            col=col,
        )
# n_rows, n_cols = eigvals[0].shape[0], len(eigvals)
# data = t.stack(eigvals).transpose(0, 1).flatten(end_dim=-2).detach().clone().cpu()
# fig = px.bar(data, facet_col=0, facet_col_wrap=len(eigvals))
fig.update_layout(height=200 * seq_len, width=200 * n_layers)
for col, title in enumerate(column_titles, start=1):
    fig.update_xaxes(title_text=title, row=1, col=col, side="top")

# Adding row titles
for row, title in enumerate(row_titles):
    fig.update_yaxes(title_text=title, row=seq_len - row, col=1)
fig.show()

#%%
# dim_weight_counts = []
eigvals = []
for projector in projector_model.projectors:
    eigvals.append(t.linalg.eigvalsh(projector.linear))
    #     dim_weight_counts.append(projector.discretize_dim_weights())

# if len(dim_weight_counts) > 0:
#     print("avg_dim_weights", sum(dim_weight_counts) / len(dim_weight_counts))
