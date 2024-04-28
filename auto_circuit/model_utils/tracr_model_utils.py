#%%
# Based on: https://github.com/ArthurConmy/Automatic-Circuit-Discovery/blob/main/acdc/tracr_task/utils.py
from typing import (
    Any,
    Literal,
    Set,
    Tuple,
)

import einops
import numpy as np
import torch
from tracr.compiler import compiling
from tracr.compiler.assemble import AssembledTransformerModel
from tracr.compiler.lib import make_frac_prevs
from tracr.rasp import rasp
from transformer_lens import HookedTransformer, HookedTransformerConfig

BOS = "BOS"
REVERSE_VOCAB: Set[Any] = {1, 2, 3}
XPROPORTION_VOCAB: Set[Any] = {"w", "x", "y", "z"}
MAX_SEQ_LEN = 5

TRACR_TASK_KEY = Literal["reverse", "xproportion"]
"""
Identifier of a Tracr model. Currently supported models:
<ul>
    <li><code>"reverse"</code></li>
    <li><code>"xproportion"</code></li>
</ul>
"""


def get_tracr_model(
    tracr_task_key: TRACR_TASK_KEY, device: str
) -> Tuple[HookedTransformer, AssembledTransformerModel]:
    """
    Load the weights of a Tracr model and convert it to a HookedTransformer model.

    Adapted from Neel Nanda's TransformerLens port of tracr.

    Args:
        tracr_task_key: Identifier of the Tracr model.
        device: Device to load the model on.

    Returns:
        A tuple of the HookedTransformer model and the original
            AssembledTransformerModel.
    """

    def make_length():
        all_true_selector = rasp.Select(rasp.tokens, rasp.tokens, rasp.Comparison.TRUE)
        return rasp.SelectorWidth(all_true_selector)

    if tracr_task_key == "reverse":
        length = make_length()  # `length` is not a primitive in our implementation.
        opp_index = length - rasp.indices - 1
        flip = rasp.Select(rasp.indices, opp_index, rasp.Comparison.EQ)
        reverse = rasp.Aggregate(flip, rasp.tokens)
        model = compiling.compile_rasp_to_model(
            reverse,
            vocab=REVERSE_VOCAB,
            max_seq_len=MAX_SEQ_LEN,
            compiler_bos=BOS,
        )
    elif tracr_task_key == "xproportion":
        model = compiling.compile_rasp_to_model(
            make_frac_prevs(rasp.tokens == "x"),
            vocab=XPROPORTION_VOCAB,
            max_seq_len=MAX_SEQ_LEN,
            compiler_bos=BOS,
        )
    else:
        raise ValueError(f"Unknown task {tracr_task_key}")

    # Extract the model config from the Tracr model, and create a blank
    # HookedTransformer object

    n_heads = model.model_config.num_heads
    n_layers = model.model_config.num_layers
    d_head = model.model_config.key_size
    d_mlp = model.model_config.mlp_hidden_size
    act_fn = "relu"
    normalization_type = "LN" if model.model_config.layer_norm else None
    attention_type = "causal" if model.model_config.causal else "bidirectional"

    n_ctx = model.params["pos_embed"]["embeddings"].shape[0]
    # Equivalent to length of vocab, with BOS and PAD at the end
    d_vocab = model.params["token_embed"]["embeddings"].shape[0]
    # Residual stream width, I don't know of an easy way to infer it from the above
    # config.
    d_model = model.params["token_embed"]["embeddings"].shape[1]

    if tracr_task_key == "reverse":
        # Equivalent to length of vocab, WITHOUT BOS and PAD at the end because we never
        # care about these outputs
        d_vocab_out = model.params["token_embed"]["embeddings"].shape[0] - 2
    elif tracr_task_key == "xproportion":
        # This task outputs a real number, so we only need the first residual dimension
        d_vocab_out = 1

    cfg = HookedTransformerConfig(
        model_name=f"tracr-{tracr_task_key}",
        n_layers=n_layers,
        d_model=d_model,
        d_head=d_head,
        n_ctx=n_ctx,
        d_vocab=d_vocab,
        d_vocab_out=d_vocab_out,
        d_mlp=d_mlp,
        n_heads=n_heads,
        act_fn=act_fn,
        attention_dir=attention_type,
        normalization_type=normalization_type,
        use_attn_result=True,
        use_split_qkv_input=True,
        device=device,
    )
    tl_model = HookedTransformer(cfg)
    if "use_hook_mlp_in" in tl_model.cfg.to_dict():
        tl_model.set_use_hook_mlp_in(True)

    # Extract the state dict, and do some reshaping so that everything has a n_heads
    # dimension
    sd = {}
    sd["pos_embed.W_pos"] = model.params["pos_embed"]["embeddings"]
    sd["embed.W_E"] = model.params["token_embed"]["embeddings"]
    # Equivalent to max_seq_len plus one, for the BOS

    # The unembed is just a projection onto the first few elements of the residual
    # stream, these store output tokens
    # This is a NumPy array, the rest are Jax Arrays, but w/e it's fine.
    sd["unembed.W_U"] = np.eye(d_model, d_vocab_out)

    for lyr in range(n_layers):
        sd[f"blocks.{lyr}.attn.W_K"] = einops.rearrange(
            model.params[f"transformer/layer_{lyr}/attn/key"]["w"],
            "d_model (n_heads d_head) -> n_heads d_model d_head",
            d_head=d_head,
            n_heads=n_heads,
        )
        sd[f"blocks.{lyr}.attn.b_K"] = einops.rearrange(
            model.params[f"transformer/layer_{lyr}/attn/key"]["b"],
            "(n_heads d_head) -> n_heads d_head",
            d_head=d_head,
            n_heads=n_heads,
        )
        sd[f"blocks.{lyr}.attn.W_Q"] = einops.rearrange(
            model.params[f"transformer/layer_{lyr}/attn/query"]["w"],
            "d_model (n_heads d_head) -> n_heads d_model d_head",
            d_head=d_head,
            n_heads=n_heads,
        )
        sd[f"blocks.{lyr}.attn.b_Q"] = einops.rearrange(
            model.params[f"transformer/layer_{lyr}/attn/query"]["b"],
            "(n_heads d_head) -> n_heads d_head",
            d_head=d_head,
            n_heads=n_heads,
        )
        sd[f"blocks.{lyr}.attn.W_V"] = einops.rearrange(
            model.params[f"transformer/layer_{lyr}/attn/value"]["w"],
            "d_model (n_heads d_head) -> n_heads d_model d_head",
            d_head=d_head,
            n_heads=n_heads,
        )
        sd[f"blocks.{lyr}.attn.b_V"] = einops.rearrange(
            model.params[f"transformer/layer_{lyr}/attn/value"]["b"],
            "(n_heads d_head) -> n_heads d_head",
            d_head=d_head,
            n_heads=n_heads,
        )
        sd[f"blocks.{lyr}.attn.W_O"] = einops.rearrange(
            model.params[f"transformer/layer_{lyr}/attn/linear"]["w"],
            "(n_heads d_head) d_model -> n_heads d_head d_model",
            d_head=d_head,
            n_heads=n_heads,
        )
        sd[f"blocks.{lyr}.attn.b_O"] = model.params[
            f"transformer/layer_{lyr}/attn/linear"
        ]["b"]

        sd[f"blocks.{lyr}.mlp.W_in"] = model.params[
            f"transformer/layer_{lyr}/mlp/linear_1"
        ]["w"]
        sd[f"blocks.{lyr}.mlp.b_in"] = model.params[
            f"transformer/layer_{lyr}/mlp/linear_1"
        ]["b"]
        sd[f"blocks.{lyr}.mlp.W_out"] = model.params[
            f"transformer/layer_{lyr}/mlp/linear_2"
        ]["w"]
        sd[f"blocks.{lyr}.mlp.b_out"] = model.params[
            f"transformer/layer_{lyr}/mlp/linear_2"
        ]["b"]

    # Convert weights to tensors and load into the tl_model

    for k, v in sd.items():
        # I cannot figure out a neater way to go from a Jax array to a numpy array lol
        sd[k] = torch.tensor(np.array(v))

    tl_model.load_state_dict(sd, strict=False)
    return tl_model, model


# tl_model, tracr_model = get_tracr_model("reverse", "cpu")
# #%%
# tl_model, tracr_model = get_tracr_model("reverse", "cpu")
# tracr_encoding = tracr_model.input_encoder.encode(["BOS", 3, 2, 1])
# print(tracr_encoding)
# tracr_model.apply(["BOS", 3, 4, 1])

# #%%
# import torch as t
# tl_model, tracr_model = get_tracr_model("xproportion", "cpu")
# tracr_output = tracr_model.apply(["BOS", "x", "x", "y", "z"])
# print("tracr_output:", tracr_output.decoded)

# tracr_encoding = tracr_model.input_encoder.encode(["BOS", "x", "x", "y", "z"])
# tl_output = tl_model(t.tensor(tracr_encoding))
# print("tl_output:", tl_output)
