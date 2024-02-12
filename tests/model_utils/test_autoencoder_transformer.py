#%%
from typing import Optional

import pytest
import torch as t
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint

from auto_circuit.data import load_datasets_from_json
from auto_circuit.model_utils.sparse_autoencoders.autoencoder_transformer import (
    AutoencoderTransformer,
    sae_model,
)
from auto_circuit.model_utils.sparse_autoencoders.sparse_autoencoder import (
    load_autoencoder,
)
from auto_circuit.prune import run_circuits
from auto_circuit.tasks import Task
from auto_circuit.types import AutoencoderInput, PatchType
from auto_circuit.utils.misc import repo_path_to_abs_path, run_prompt
from auto_circuit.utils.patchable_model import PatchableModel

# from tests.conftest import gpt2


def normalized_mean_squared_error(
    reconstruction: t.Tensor,
    original_input: t.Tensor,
) -> t.Tensor:
    """
    Source: https://github.com/openai/sparse_autoencoder/blob/main/sparse_autoencoder/loss.py
    :param reconstruction: output of Autoencoder.decode (shape: [..., n_inputs])
    :param original_input: input of Autoencoder.encode (shape: [..., n_inputs])
    :return: normalized mean squared error (shape: [1])
    """
    return (
        ((reconstruction - original_input) ** 2).mean(dim=-1)
        / (original_input**2).mean(dim=-1)
    ).mean()


MODEL_NAMES = ["pythia-70m-deduped"]
# MODEL_NAMES = ["gpt2", "pythia-70m-deduped"]


@pytest.mark.parametrize("hooked_transformer", MODEL_NAMES, indirect=True)
def test_single_autoencoder_output_similarity(
    hooked_transformer: HookedTransformer,
    layer_idx: int = 3,
    autoencoder_input: AutoencoderInput = "resid_delta_mlp",
    pythia_size: Optional[str] = "0_8192",
):
    """Check that the output of a single autoencoder is similar to the input."""
    model = hooked_transformer
    autoencoder = load_autoencoder(
        HookPoint(), model, layer_idx, autoencoder_input, pythia_size
    )

    prompt = "This is an example of a prompt that"
    tokens = model.to_tokens(prompt)  # (1, n_tokens)
    with t.inference_mode():
        _, activation_cache = model.run_with_cache(tokens, remove_batch_dim=True)
    if autoencoder_input == "mlp_post_act":
        input_tensor = activation_cache[
            "blocks..mlp.hook_post"
        ]  # (n_tokens, n_neurons)
    else:
        assert autoencoder_input == "resid_delta_mlp"
        input_tensor = activation_cache[
            f"blocks.{layer_idx}.hook_mlp_out"
        ]  # (n_tokens, n_residual_channels)

    with t.inference_mode():
        reconstruction = autoencoder(input_tensor)

    error = normalized_mean_squared_error(reconstruction, input_tensor)
    assert error < 0.2


@pytest.mark.parametrize("hooked_transformer", MODEL_NAMES, indirect=True)
def test_autoencoder_transformer_output_similarity(
    hooked_transformer: HookedTransformer,
    autoencoder_input: AutoencoderInput = "resid_delta_mlp",
    pythia_size: Optional[str] = "0_8192",
    print_top_k: Optional[int] = None,
):
    """Check that the output of the AutoencoderTransformer is similar to the default."""
    default_model = hooked_transformer
    encoder_model = sae_model(
        default_model,
        autoencoder_input,
        load_pretrained=True,
        pythia_size=pythia_size,
        new_instance=True,
    )
    prompt = "Michael Jackson was a"
    tokens = default_model.to_tokens(prompt)  # (1, n_tokens)
    with t.inference_mode():
        if print_top_k:
            run_prompt(default_model, prompt, top_k=print_top_k)
            run_prompt(encoder_model, prompt, top_k=print_top_k)

        default_logits = default_model(tokens)
        autoencoder_logits = encoder_model(tokens)
        error = normalized_mean_squared_error(default_logits, autoencoder_logits)
        assert error < 0.1
    del encoder_model


@pytest.mark.parametrize("hooked_transformer", MODEL_NAMES, indirect=True)
def test_prune_latents_with_dataset(
    hooked_transformer: HookedTransformer,
    autoencoder_input: AutoencoderInput = "resid_delta_mlp",
    pythia_size: Optional[str] = "0_8192",
    print_top_k: Optional[int] = None,
):
    default_model = hooked_transformer
    encoder_model = sae_model(
        hooked_transformer,
        autoencoder_input,
        load_pretrained=True,
        pythia_size=pythia_size,
        new_instance=True,
    )

    train_loader, test_loader = load_datasets_from_json(
        model=default_model,
        path=repo_path_to_abs_path("datasets/mini_prompts.json"),
        device=t.device("cpu"),
        prepend_bos=True,
        batch_size=1,
        train_test_split=[1, 1],
        length_limit=2,
        return_seq_length=True,
    )
    encoder_model._prune_latents_with_dataset(
        train_loader, 100, seq_len=train_loader.seq_len
    )

    toks: t.Tensor = test_loader.dataset[0].clean
    prompts = default_model.tokenizer.decode(toks, True)  # type: ignore
    with t.inference_mode():
        if print_top_k:
            run_prompt(default_model, prompts, top_k=print_top_k)
            run_prompt(encoder_model, prompts, top_k=print_top_k)

        default_logits = default_model(toks)
        autoencoder_logits = encoder_model(toks)
        error = normalized_mean_squared_error(default_logits, autoencoder_logits)
        assert error < 0.1
    del encoder_model


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_task_autoencoder_transformer_edges(model_name: str):
    """Load an autoencoder model, make it a PatchableModel.
    The Edge Patch all the edges going to the output node.
    Check the output is the same as the unpatched model run on the clean input."""
    task = Task(
        key="test_task_autoencoder_transformer_edges",
        name="test_task_autoencoder_transformer_edges",
        batch_size=1,
        batch_count=1,
        token_circuit=True,
        _model_def=model_name,
        _dataset_name="capital_cities_pythia-70m-deduped_prompts",
        autoencoder_input="resid_delta_mlp",
        autoencoder_max_latents=100,
        autoencoder_pythia_size="0_8192",
        autoencoder_prune_with_corrupt=False,
    )
    model = task.model
    assert isinstance(model, PatchableModel)
    assert isinstance(model.wrapped_model, AutoencoderTransformer)

    resid_end_mod_name = max(model.dests, key=lambda x: x.layer).module_name
    prune_scores = model.new_prune_scores()
    prune_scores[resid_end_mod_name] = t.ones_like(prune_scores[resid_end_mod_name])
    edge_count = int(prune_scores[resid_end_mod_name].sum().int().item())

    with t.inference_mode():
        batch = next(iter(task.test_loader))
        clean_logits = model(batch.clean)[model.out_slice]
        pruned_outs = run_circuits(
            model, task.test_loader, [edge_count], prune_scores, PatchType.EDGE_PATCH
        )
    patched_out = pruned_outs[edge_count][batch.key]
    assert patched_out.shape == clean_logits.shape
    assert t.allclose(patched_out, clean_logits, atol=1e-5)


# model_name = "gpt2"
# model_name = "pythia-70m-deduped"
# autoencoder_input = "resid_delta_mlp"
# pythia_size = "2_32768"
# model = hooked_transformer(None, model_name=model_name)

# for i in range(3):
# test_single_autoencoder_output_similarity(model)

# test_autoencoder_transformer_output_similarity(model)
# test_prune_latents_with_dataset(model, print_top_k=5)
# test_task_autoencoder_transformer_edges(model_name)
# %%
