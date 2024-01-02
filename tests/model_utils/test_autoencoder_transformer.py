#%%
from typing import Optional

import torch as t
import transformer_lens as tl
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint

from auto_circuit.data import load_datasets_from_json
from auto_circuit.model_utils.autoencoder_transformer import (
    AutoencoderHook,
    AutoencoderTransformer,
    autoencoder_model,
)
from auto_circuit.prune import run_pruned
from auto_circuit.tasks import Task
from auto_circuit.types import AutoencoderInput, PatchType
from auto_circuit.utils.misc import repo_path_to_abs_path
from auto_circuit.utils.patchable_model import PatchableModel


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


def test_single_autoencoder_output_similarity(
    gpt2: HookedTransformer,
    autoencoder_input: AutoencoderInput = "resid_delta_mlp",
    layer_idx: int = 6,
):
    """Check that the output of a single autoencoder is similar to the input."""
    autoencoder = AutoencoderHook(HookPoint(), layer_idx, autoencoder_input)
    model = gpt2

    prompt = "This is an example of a prompt that"
    tokens = model.to_tokens(prompt)  # (1, n_tokens)
    with t.inference_mode():
        _, activation_cache = model.run_with_cache(tokens, remove_batch_dim=True)
    if autoencoder_input == "mlp_post_act":
        input_tensor = activation_cache[
            f"blocks.{layer_idx}.mlp.hook_post"
        ]  # (n_tokens, n_neurons)
    elif autoencoder_input == "resid_delta_mlp":
        input_tensor = activation_cache[
            f"blocks.{layer_idx}.hook_mlp_out"
        ]  # (n_tokens, n_residual_channels)

    with t.inference_mode():
        reconstruction = autoencoder(input_tensor)

    error = normalized_mean_squared_error(reconstruction, input_tensor)
    assert error < 0.2


def test_autoencoder_transformer_output_similarity(
    gpt2: HookedTransformer,
    autoencoder_input: AutoencoderInput = "resid_delta_mlp",
    print_top_k: Optional[int] = None,
):
    """Check that the output of the AutoencoderTransformer is similar to the input."""
    default_model = gpt2
    encoder_model = autoencoder_model(
        default_model, autoencoder_input, new_instance=True
    )
    prompt = "Michael Jackson was a"
    tokens = default_model.to_tokens(prompt)  # (1, n_tokens)
    with t.inference_mode():
        if print_top_k:
            tl.utils.test_prompt(prompt, "singer", default_model, top_k=print_top_k)
            tl.utils.test_prompt(prompt, "singer", encoder_model, top_k=print_top_k)

        default_logits = default_model(tokens)
        autoencoder_logits = encoder_model(tokens)
        error = normalized_mean_squared_error(default_logits, autoencoder_logits)
        assert error < 0.1
    del encoder_model


def test_prune_latents_with_dataset(
    gpt2: HookedTransformer, print_top_k: Optional[int] = None
):
    autoencoder_input: AutoencoderInput = "resid_delta_mlp"
    default_model = gpt2
    encoder_model = autoencoder_model(gpt2, autoencoder_input, new_instance=True)

    train_loader, test_loader = load_datasets_from_json(
        tokenizer=default_model.tokenizer,
        path=repo_path_to_abs_path("datasets/mini_prompts.json"),
        device=t.device("cpu"),
        prepend_bos=True,
        batch_size=1,
        train_test_split=[1, 1],
        length_limit=2,
    )
    encoder_model._prune_latents_with_dataset(train_loader, latent_threshold=0.01)

    toks: t.Tensor = test_loader.dataset[0].clean
    prompts = gpt2.tokenizer.decode(toks, skip_special_tokens=True)  # type: ignore
    with t.inference_mode():
        if print_top_k:
            tl.utils.test_prompt(prompts, "grass", default_model, top_k=print_top_k)
            tl.utils.test_prompt(prompts, "grass", encoder_model, top_k=print_top_k)

        default_logits = default_model(toks)
        autoencoder_logits = encoder_model(toks)
        error = normalized_mean_squared_error(default_logits, autoencoder_logits)
        assert error < 0.1
    del encoder_model


def test_task_autoencoder_transformer_edges():
    """Load an autoencoder model, make it a PatchableModel.
    The Edge Patch all the edges going to the output node.
    Check the output is the same as the unpatched model run on the clean input."""
    autoencoder_input: AutoencoderInput = "resid_delta_mlp"

    task = Task(
        key="test_task_autoencoder_transformer_edges",
        name="test_task_autoencoder_transformer_edges",
        batch_size=1,
        batch_count=1,
        token_circuit=False,
        _model_def="gpt2",
        _dataset_name="greaterthan_gpt2-small_prompts",
        autoencoder_input=autoencoder_input,
        autoencoder_latent_threshold=0.1,
        autoencoder_prune_with_corrupt=False,
    )
    model = task.model
    assert isinstance(model, PatchableModel)
    assert isinstance(model.wrapped_model, AutoencoderTransformer)
    resid_end_node = max(model.dests, key=lambda x: x.layer)
    resid_end_edges = {e: 1.0 for e in model.edges if e.dest == resid_end_node}
    edge_count = len(resid_end_edges)
    with t.inference_mode():
        batch = next(iter(task.test_loader))
        clean_logits = model(batch.clean)[model.out_slice]
        pruned_outs = run_pruned(
            model, task.test_loader, [edge_count], resid_end_edges, PatchType.EDGE_PATCH
        )
    patched_out = pruned_outs[edge_count][0]
    assert patched_out.shape == clean_logits.shape
    assert t.allclose(patched_out, clean_logits, atol=1e-5)


# gpt2 = gpt2()
# autoencoder_input = "resid_delta_mlp"

# for i in range(3):
#     test_single_autoencoder_output_similarity(gpt2, autoencoder_input, i)

# test_autoencoder_transformer_output_similarity(gpt2, "resid_delta_mlp", 10)
# test_prune_latents_with_dataset(gpt2, print_top_k=10)
# test_task_autoencoder_transformer_edges()
# %%
