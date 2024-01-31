#%%
# Based on: https://github.com/ArthurConmy/Automatic-Circuit-Discovery/blob/main/acdc/tracr_task/utils.py
from typing import (
    Literal,
    Tuple,
)

import einops
import numpy as np
import torch
from tracr.compiler import compiling
from tracr.compiler.assemble import AssembledTransformerModel
from tracr.rasp import rasp
from transformer_lens import HookedTransformer, HookedTransformerConfig

BOS = "BOS"
REVERSE_VOCAB = {1, 2, 3}
PROPORTION_VOCAB = {"w", "x", "y", "z"}
MAX_SEQ_LEN = 5


def get_tracr_model(
    task: Literal["reverse", "proportion"], device: str
) -> Tuple[HookedTransformer, AssembledTransformerModel]:
    """
    This function adapts Neel's TransformerLens porting of tracr
    """

    def make_length():
        all_true_selector = rasp.Select(rasp.tokens, rasp.tokens, rasp.Comparison.TRUE)
        return rasp.SelectorWidth(all_true_selector)

    if task == "reverse":
        length = make_length()  # `length` is not a primitive in our implementation.
        opp_index = length - rasp.indices - 1
        flip = rasp.Select(rasp.indices, opp_index, rasp.Comparison.EQ)
        reverse = rasp.Aggregate(flip, rasp.tokens)
        model = compiling.compile_rasp_to_model(
            reverse,
            vocab={1, 2, 3},
            max_seq_len=5,
            compiler_bos=BOS,
        )
        out = model.apply([BOS, 1, 2, 3])

    elif task == "proportion":
        from tracr.compiler.lib import make_frac_prevs

        model = compiling.compile_rasp_to_model(
            make_frac_prevs(rasp.tokens == 1),
            vocab={"w", "x", "y", "z"},
            max_seq_len=5,
            compiler_bos=BOS,
        )

        out = model.apply(["BOS", "w", "x", "y", "z"])

    else:
        raise ValueError(f"Unknown task {task}")

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

    # Equivalent to length of vocab, WITHOUT BOS and PAD at the end because we never
    # care about these outputs
    d_vocab_out = model.params["token_embed"]["embeddings"].shape[0] - 2

    cfg = HookedTransformerConfig(
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

    # Create helper functions to do the tokenization and de-tokenization

    INPUT_ENCODER = model.input_encoder
    OUTPUT_ENCODER = model.output_encoder

    def create_model_input(input, input_encoder=INPUT_ENCODER, device=device):
        encoding = input_encoder.encode(input)
        return torch.tensor(encoding).unsqueeze(dim=0).to(device)

    if task == "reverse":  # this doesn't make sense for proportion

        def decode_model_output(
            logits, output_encoder=OUTPUT_ENCODER, bos_token=INPUT_ENCODER.bos_token
        ):
            max_output_indices = logits.squeeze(dim=0).argmax(dim=-1)
            decoded_output = output_encoder.decode(max_output_indices.tolist())
            decoded_output_with_bos = [bos_token] + decoded_output[1:]
            return decoded_output_with_bos

    # We can now run the model!
    if task == "reverse":
        input = [BOS, 1, 2, 3]
        out = model.apply(input)
        print("Original Decoding:", out.decoded)

        input_tokens_tensor = create_model_input(input)
        logits = tl_model(input_tokens_tensor)
        decoded_output = decode_model_output(logits)
        print("TransformerLens Replicated Decoding:", decoded_output)

    elif task == "proportion":
        input = [BOS, "x", "w", "w", "x"]
        out = model.apply(input)
        print("Original Decoding:", out.decoded)

        input_tokens_tensor = create_model_input(input)
        logits = tl_model(input_tokens_tensor)
        # decoded_output = decode_model_output(logits)
        # print("TransformerLens Replicated Decoding:", decoded_output)

    else:
        raise ValueError("Task must be either 'reverse' or 'proportion'")

    # Lets cache all intermediate activations in the model, and check that they're the
    # same:

    logits, cache = tl_model.run_with_cache(input_tokens_tensor)

    for layer in range(tl_model.cfg.n_layers):
        print(
            f"Layer {layer} Attn Out Equality Check:",
            np.isclose(
                cache["attn_out", layer].detach().cpu().numpy(),
                np.array(out.layer_outputs[2 * layer]),
            ).all(),
        )
        print(
            f"Layer {layer} MLP Out Equality Check:",
            np.isclose(
                cache["mlp_out", layer].detach().cpu().numpy(),
                np.array(out.layer_outputs[2 * layer + 1]),
            ).all(),
        )

    # Look how pretty and ordered the final residual stream is!
    #
    # (The logits are the first 3 dimensions of the residual stream, and we can see
    # that they're flipped!)

    import plotly.express as px

    im = cache["resid_post", -1].detach().cpu().numpy()[0]
    px.imshow(
        im,
        color_continuous_scale="Blues",
        labels={"x": "Residual Stream", "y": "Position"},
        y=[str(i) for i in input],
    ).show()

    return create_model_input, tl_model


get_tracr_model("reverse", "cpu")
