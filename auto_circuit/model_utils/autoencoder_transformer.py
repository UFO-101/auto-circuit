from copy import deepcopy
from itertools import count
from time import time
from typing import Any, List, Set

import blobfile as bf
import torch as t
from sparse_autoencoder import Autoencoder
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint

from auto_circuit.data import PromptDataLoader
from auto_circuit.types import AutoencoderInput, DestNode, SrcNode
from auto_circuit.utils.custom_tqdm import tqdm
from auto_circuit.utils.misc import repo_path_to_abs_path
from auto_circuit.utils.patchable_model import PatchableModel

CHUNK_SIZE = 1000 * 2**20  # 1000 MB


class AutoencoderHook(t.nn.Module):
    wrapped_hook: HookPoint
    autoencoder: Autoencoder

    def __init__(
        self, hook: HookPoint, layer_idx: int, autoencoder_input: AutoencoderInput
    ):
        super().__init__()
        self.wrapped_hook = hook
        blob_prefix = "az://openaipublic/sparse-autoencoder/gpt2-small"
        blobfile_path = f"{blob_prefix}/{autoencoder_input}/autoencoders/{layer_idx}.pt"
        # tempdir = Path(tempfile.gettempdir())
        cache_dir = repo_path_to_abs_path(".autoencoder_cache")
        cache_filepath = cache_dir / f"gpt2_{autoencoder_input}_{layer_idx}.pt"
        if not cache_filepath.exists():
            with bf.BlobFile(blobfile_path, mode="rb") as blob, open(
                cache_filepath, "wb"
            ) as cache_file:
                print(
                    "Downloading autoencoder"
                    + f"{autoencoder_input}, layer {layer_idx} to {cache_filepath}..."
                )
                start_time = time()
                block = blob.read(CHUNK_SIZE)
                print(f"Done. Took {time() - start_time:.2f} seconds.")
                cache_file.write(block)
            # bf.copy(blobfile_path, str(cache_filepath))
        with open(cache_filepath, "rb") as f:
            state_dict = t.load(f)

        self.autoencoder = Autoencoder.from_state_dict(state_dict)  # type: ignore
        self.reset_activated_latents()
        self.latent_threshold = 0.0

    def reset_activated_latents(self):
        self.activated_latents = t.zeros_like(
            self.autoencoder.latent_bias, dtype=t.bool
        )

    def forward(self, input: t.Tensor) -> t.Tensor:
        latents_pre_act, latents, recons = self.autoencoder(input)
        self.activated_latents |= (
            (latents > self.latent_threshold).flatten(end_dim=-2).any(dim=0)
        )
        return self.wrapped_hook(recons)

    def prune_latents(self, idxs: t.Tensor):
        assert idxs.ndim == 1
        state_dict = self.autoencoder.state_dict()

        # Check all tensors are on the same device
        prev_device = state_dict["latent_bias"].device
        assert all(t.device == prev_device for t in state_dict.values())

        new_state_dict = {
            "pre_bias": state_dict["pre_bias"].clone(),
            "latent_bias": state_dict["latent_bias"][idxs].clone(),
            "stats_last_nonzero": state_dict["stats_last_nonzero"][idxs].clone(),
            "encoder.weight": state_dict["encoder.weight"][idxs].clone(),
            "decoder.weight": state_dict["decoder.weight"][:, idxs].clone(),
        }
        del self.autoencoder
        self.autoencoder = Autoencoder.from_state_dict(new_state_dict)
        self.autoencoder.to(prev_device)
        self.reset_activated_latents()


class AutoencoderTransformer(t.nn.Module):
    wrapped_model: t.nn.Module
    autoencoder_hooks: List[AutoencoderHook]

    def __init__(
        self, wrapped_model: t.nn.Module, autoencoder_hooks: List[AutoencoderHook]
    ):
        super().__init__()
        self.autoencoder_hooks = autoencoder_hooks

        if isinstance(wrapped_model, PatchableModel):
            self.wrapped_model = wrapped_model.wrapped_model
        else:
            self.wrapped_model = wrapped_model

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.wrapped_model(*args, **kwargs)

    def _prune_latents_with_dataset(
        self,
        dataloader: PromptDataLoader,
        latent_threshold: float = 0.0,
        include_corrupt: bool = False,
    ):
        """
        !In place operation!
        Prune the weights of the autoencoder to remove latents that are never activated
        by the dataset. This can reduce the number of edges in the factorized model by a
        factor of 10 or more.
        """
        for hook in self.autoencoder_hooks:
            hook.latent_threshold = latent_threshold
            hook.reset_activated_latents()

        print("Running dataset for autoencoder pruning...")
        unpruned_logits = []
        with t.inference_mode():
            for batch_idx, batch in (batch_pbar := tqdm(enumerate(dataloader))):
                batch_pbar.set_description_str(
                    f"Pruning Autoencoder: Batch {batch_idx}"
                )
                for input_idx, single_input in (
                    input_pbar := tqdm(enumerate(batch.clean))
                ):
                    input_pbar.set_description_str(f"Clean Batch Input {input_idx}")
                    out = self.forward(single_input.unsqueeze(0))  # Run one at a time
                    unpruned_logits.append(out)
                if include_corrupt:
                    for input_idx, single_input in (
                        input_pbar := tqdm(enumerate(batch.corrupt))
                    ):
                        input_pbar.set_description_str(
                            f"Corrupt Batch Input {input_idx}"
                        )
                        out = self.forward(single_input.unsqueeze(0))
                        unpruned_logits.append(out)

        latent_counts = []
        for hook in self.autoencoder_hooks:
            latents_to_keep_idxs = t.where(hook.activated_latents)[0]
            latent_counts.append(len(latents_to_keep_idxs))
            hook.prune_latents(latents_to_keep_idxs)

        pruned_logits = []
        with t.inference_mode():
            for batch_idx, batch in (batch_pbar := tqdm(enumerate(dataloader))):
                batch_pbar_str = f"Testing Pruned Autoencoder: Batch {batch_idx}"
                batch_pbar.set_description_str(batch_pbar_str)
                out = self.forward(batch.clean)
                pruned_logits.append(out)
                if include_corrupt:
                    out = self.forward(batch.corrupt)
                    pruned_logits.append(out)

        flat_pruned_logits = t.flatten(t.stack(pruned_logits), end_dim=-2)
        flat_unpruned_logits = t.flatten(t.stack(unpruned_logits), end_dim=-2)
        kl_div = t.nn.functional.kl_div(
            t.nn.functional.log_softmax(flat_pruned_logits, dim=-1),
            t.nn.functional.log_softmax(flat_unpruned_logits, dim=-1),
            reduction="batchmean",
            log_target=True,
        )

        print(
            "Done. Autoencoder latent counts:",
            latent_counts,
            ". Pruned vs. Unpruned KL Div:",
            kl_div.item(),
        )

    def run_with_cache(self, *args: Any, **kwargs: Any) -> Any:
        return self.wrapped_model.run_with_cache(*args, **kwargs)

    def add_hook(self, *args: Any, **kwargs: Any) -> Any:
        return self.wrapped_model.add_hook(*args, **kwargs)

    @property
    def cfg(self) -> Any:
        return self.wrapped_model.cfg

    @property
    def tokenizer(self) -> Any:
        return self.wrapped_model.tokenizer

    @property
    def input_to_embed(self) -> Any:
        return self.wrapped_model.input_to_embed

    @property
    def blocks(self) -> Any:
        return self.wrapped_model.blocks

    def to_tokens(self, *args: Any, **kwargs: Any) -> Any:
        return self.wrapped_model.to_tokens(*args, **kwargs)

    def to_str_tokens(self, *args: Any, **kwargs: Any) -> Any:
        return self.wrapped_model.to_str_tokens(*args, **kwargs)

    def to_string(self, *args: Any, **kwargs: Any) -> Any:
        return self.wrapped_model.to_string(*args, **kwargs)

    def __str__(self) -> str:
        return self.wrapped_model.__str__()

    def __repr__(self) -> str:
        return self.wrapped_model.__repr__()


def autoencoder_model(
    model: HookedTransformer,
    autoencoder_input: AutoencoderInput,
    new_instance: bool = True,
) -> AutoencoderTransformer:
    if new_instance:
        model = deepcopy(model)
    assert model.cfg.model_name == "gpt2"
    autoencoder_hooks: List[AutoencoderHook] = []
    for layer_idx in range(model.cfg.n_layers):
        if autoencoder_input == "mlp_post_act":
            hook_point = model.blocks[layer_idx].mlp.hook_post
            autoencoder_hook = AutoencoderHook(hook_point, layer_idx, autoencoder_input)
            autoencoder_hook.to(model.cfg.device)
            setattr(model.blocks[layer_idx].mlp, "hook_post", autoencoder_hook)
        else:
            assert autoencoder_input == "resid_delta_mlp"
            hook_point = model.blocks[layer_idx].hook_mlp_out
            autoencoder_hook = AutoencoderHook(hook_point, layer_idx, autoencoder_input)
            autoencoder_hook.to(model.cfg.device)
            setattr(model.blocks[layer_idx], "hook_mlp_out", autoencoder_hook)
        autoencoder_hooks.append(autoencoder_hook)
    return AutoencoderTransformer(model, autoencoder_hooks)


def factorized_src_nodes(model: AutoencoderTransformer) -> Set[SrcNode]:
    """Get the source part of each edge in the factorized graph, grouped by layer.
    Graph is factorized following the Mathematical Framework paper."""
    assert model.cfg.use_attn_result  # Get attention head outputs separately
    assert (
        model.cfg.use_attn_in
    )  # Get attention head inputs separately (but Q, K, V are still combined)
    assert model.cfg.use_split_qkv_input  # Separate Q, K, V input for each head
    assert model.cfg.use_hook_mlp_in  # Get MLP input BEFORE layernorm
    layers, idxs = count(), count()
    nodes = set()
    nodes.add(
        SrcNode(
            name="Resid Start",
            module_name="blocks.0.hook_resid_pre",
            layer=next(layers),
            idx=next(idxs),
            weight="embed.W_E",
        )
    )

    for block_idx in range(model.cfg.n_layers):
        layer = next(layers)
        for head_idx in range(model.cfg.n_heads):
            nodes.add(
                SrcNode(
                    name=f"A{block_idx}.{head_idx}",
                    module_name=f"blocks.{block_idx}.attn.hook_result",
                    layer=layer,
                    idx=next(idxs),
                    head_dim=2,
                    head_idx=head_idx,
                    weight=f"blocks.{block_idx}.attn.W_O",
                    weight_head_dim=0,
                )
            )
        layer = next(layers)
        for latent_idx in range(
            model.blocks[block_idx].hook_mlp_out.autoencoder.n_latents
        ):
            nodes.add(
                SrcNode(
                    name=f"MLP {block_idx} Latent {latent_idx}",
                    module_name=f"blocks.{block_idx}.hook_mlp_out.autoencoder.latent_outs",
                    layer=layer,
                    idx=next(idxs),
                    head_dim=-2,
                    head_idx=latent_idx,
                    weight=f"blocks.{block_idx}.hook_mlp_out.autoencoder.decoder.weight",
                    weight_head_dim=0,
                )
            )
    return nodes


def factorized_dest_nodes(model: AutoencoderTransformer) -> Set[DestNode]:
    """Get the destination part of each edge in the factorized graph, grouped by layer.
    Graph is factorized following the Mathematical Framework paper."""
    assert model.cfg.use_attn_result  # Get attention head outputs separately
    assert (
        model.cfg.use_attn_in
    )  # Get attention head inputs separately (but Q, K, V are still combined)
    # assert model.cfg.use_split_qkv_input  # Separate Q, K, V input for each head
    assert model.cfg.use_hook_mlp_in  # Get MLP input BEFORE layernorm
    layers, idxs = count(1), count()
    nodes = set()
    for block_idx in range(model.cfg.n_layers):
        layer = next(layers)
        for head_idx in range(model.cfg.n_heads):
            nodes.add(
                DestNode(
                    name=f"A{block_idx}.{head_idx}",
                    module_name=f"blocks.{block_idx}.hook_attn_in",
                    layer=layer,
                    idx=next(idxs),
                    head_dim=2,
                    head_idx=head_idx,
                )
            )
        nodes.add(
            DestNode(
                name=f"MLP {block_idx}",
                module_name=f"blocks.{block_idx}.hook_mlp_in",
                layer=next(layers),
                idx=next(idxs),
                weight=f"blocks.{block_idx}.mlp.W_in",
            )
        )
    nodes.add(
        DestNode(
            name="Resid End",
            module_name=f"blocks.{model.cfg.n_layers - 1}.hook_resid_post",
            layer=next(layers),
            idx=next(idxs),
            weight="unembed.W_U",
        )
    )
    return nodes
