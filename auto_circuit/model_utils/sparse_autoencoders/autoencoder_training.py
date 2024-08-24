# %%
from typing import Any, Dict, Tuple

import plotly.express as px
import torch as t
import transformer_lens as tl
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding

from auto_circuit.data import PromptDataLoader, load_datasets_from_json
from auto_circuit.model_utils.sparse_autoencoders.autoencoder_transformer import (
    AutoencoderTransformer,
    sae_model,
)
from auto_circuit.model_utils.sparse_autoencoders.sparse_autoencoder import (
    SparseAutoencoder,
)
from auto_circuit.utils.custom_tqdm import tqdm
from auto_circuit.utils.misc import repo_path_to_abs_path
from datasets import Dataset, load_dataset


def get_model_and_data(
    model_name: str, dataset_name: str, batch_size: int
) -> Tuple[tl.HookedTransformer, DataLoader[Any]]:
    model = tl.HookedTransformer.from_pretrained_no_processing(model_name)
    cache_dir = repo_path_to_abs_path(".dataset_cache").resolve()
    text_dataset = load_dataset(dataset_name, split="train", cache_dir=str(cache_dir))
    assert isinstance(text_dataset, Dataset)

    def tokenize(x: Dict[str, Any]) -> Any:
        assert model.tokenizer is not None
        model.tokenizer.padding_side = "right"
        return model.tokenizer(x["text"])

    dataset = text_dataset.map(tokenize, batched=True, batch_size=10000)
    dataset.set_format(type="torch", columns=["input_ids"])

    assert model.tokenizer is not None
    collator = DataCollatorWithPadding(model.tokenizer)
    loader = DataLoader(dataset, batch_size, False, collate_fn=collator)  # type: ignore
    return model, loader


def train_model(
    model: tl.HookedTransformer,
    dataloader: PromptDataLoader,
    n_latents: int,
    l1_lambda: float,
    learning_rate: float,
    n_epochs: int,
) -> AutoencoderTransformer:
    model.eval()
    device = model.cfg.device

    autoencoder_model: AutoencoderTransformer = sae_model(
        model,
        sae_input="resid_delta_mlp",
        load_pretrained=False,
        n_latents=n_latents,
        new_instance=True,
    )

    train_params = []
    for _, module in autoencoder_model.named_modules():
        if isinstance(module, SparseAutoencoder):
            module.train()
            for param in module.parameters():
                param.requires_grad = True
                train_params.append(param)
        else:
            module.eval()
            for param in module.parameters():
                param.requires_grad = False

    optim = t.optim.adam.Adam(train_params, lr=learning_rate)

    loss_history, cross_entropy_loss_history, l1_loss_history = [], [], []
    for epoch in (epoch_pbar := tqdm(range(n_epochs))):
        for batch in (batch_pbar := tqdm(dataloader)):
            # toks = batch["input_ids"].to(device)
            # mask = batch["attention_mask"].to(device)
            toks = batch.clean.to(device)
            autoencoder_model.reset_activated_latents(toks.shape[0], toks.shape[1])
            optim.zero_grad()
            with t.no_grad():
                default_probs = t.nn.functional.softmax(model(toks), -1).clone()
            sae_loprobs = t.nn.functional.log_softmax(autoencoder_model(toks), dim=-1)
            nll = (-(default_probs * sae_loprobs).sum(dim=-1)).mean()
            latents = [
                s.latent_total_act for s in autoencoder_model.sparse_autoencoders
            ]
            l1 = (t.stack(latents).sum(dim=-1)).mean()  # Post ReLU (>0) so it is L1
            loss = nll + l1 * l1_lambda
            loss.backward()
            optim.step()

            loss_history.append(loss.item())
            cross_entropy_loss_history.append(nll.item())
            l1_loss_history.append(l1.item())
            desc = f"Loss: {loss.item():.3f} NLL: {nll.item():.3f} L1: {l1.item():.3f}"
            batch_pbar.set_description_str(desc)
            epoch_pbar.set_description_str(f"Epoch: {epoch} " + desc)

    if False:
        for layer, sae in enumerate(autoencoder_model.sparse_autoencoders):
            cache_dir = repo_path_to_abs_path(".models").resolve()
            filename = f"custom_{n_latents}_sae_{model_name}_layer_{layer}.pt"
            with open(cache_dir / filename, "wb") as f:
                t.save(sae.state_dict(), f)
    data = {
        "Epoch": list(range(len(loss_history))),
        "Total Loss": loss_history,
        "Cross Entropy Loss": cross_entropy_loss_history,
        "L1 Loss": l1_loss_history,
    }
    px.line(data, x="Epoch", y=["Total Loss", "Cross Entropy Loss", "L1 Loss"]).show()
    return autoencoder_model


# model_name, dataset_name = "tiny-stories-33M", "roneneldan/TinyStories"
# model, dataloader = get_model_and_data(model_name, dataset_name)


model_name, dataset_name = (
    "pythia-70m-deduped",
    "datasets/capital_cities_pythia-70m-deduped_prompts.json",
)
model = tl.HookedTransformer.from_pretrained_no_processing(model_name)
assert model.cfg.device is not None
train_dataloader, test_dataloader = load_datasets_from_json(
    model,
    repo_path_to_abs_path(dataset_name),
    device=t.device(model.cfg.device),
    prepend_bos=True,
    batch_size=10,
    train_test_size=(900, 100),
    return_seq_length=False,
    shuffle=True,
    pad=True,
)
encoder_model = train_model(
    model,
    train_dataloader,
    n_latents=64,
    l1_lambda=0.01,
    learning_rate=0.001,
    n_epochs=2000,
)

# %%
default_logprobs = []
sae_logprobs = []
l0_norms = []
for batch in test_dataloader:
    encoder_model.reset_activated_latents(batch.clean.shape[0], batch.clean.shape[1])
    default_logprobs.append(t.nn.functional.log_softmax(model(batch.clean), dim=-1))
    sae_logprobs.append(t.nn.functional.log_softmax(encoder_model(batch.clean), dim=-1))
    for sae in encoder_model.sparse_autoencoders:
        l0_norms.append(sae.latent_total_act.flatten(end_dim=-2) > 0)

avg_kl_div = t.nn.functional.kl_div(
    t.cat(sae_logprobs).flatten(end_dim=-2),
    t.cat(default_logprobs).flatten(end_dim=-2),
    reduction="batchmean",
    log_target=True,
)
print("avg_kl_div", avg_kl_div.item())
avg_l0_norm = t.cat(l0_norms).sum(dim=-1).float().mean()
print("avg_l0_norm", avg_l0_norm.item())
