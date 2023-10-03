# Some of this code is copied from:
# https://github.com/stevenxcao/subnetwork-probing
# From the paper Low-Complexity Probing via Finding Subnetworks:
# https://arxiv.org/abs/2104.03514

import math
from typing import Dict, Set

import plotly.graph_objects as go
import torch as t
from torch.nn.functional import kl_div, log_softmax
from torch.utils.data import DataLoader

from auto_circuit.data import PromptPairBatch
from auto_circuit.types import Edge
from auto_circuit.utils.custom_tqdm import tqdm
from auto_circuit.utils.graph_utils import (
    get_sorted_src_outs,
    mask_fn_mode,
    patch_mode,
    train_mask_mode,
)
from auto_circuit.utils.patch_wrapper import MaskFn

# Constants are copied from the paper's code
mask_p, left, right, temp = 0.9, -0.1, 1.1, 2 / 3
p = (mask_p - left) / (right - left)
init_mask_val = math.log(p / (1 - p))
regularize_const = temp * math.log(-left / right)

SP = "Subnetwork Probing"


def subnetwork_probing_prune_scores(
    model: t.nn.Module,
    train_data: DataLoader[PromptPairBatch],
    output_dim: int = 1,
    learning_rate: float = 0.1,
    epochs: int = 20,
    regularize_lambda: float = 10,
    mask_fn: MaskFn = "hard_concrete",
    init_val: float = -init_mask_val,
    show_train_graph: bool = False,
) -> Dict[Edge, float]:
    """Prune scores are the mean activation magnitude of each edge."""
    output_idx = tuple([slice(None)] * output_dim + [-1])

    corrupt_logprobs: Dict[str, t.Tensor] = {}
    with t.inference_mode():
        for batch in train_data:
            corrupt_out = model(batch.corrupt)[output_idx]
            corrupt_logprobs[batch.key] = log_softmax(corrupt_out, dim=-1)

    src_outs_dict: Dict[int, t.Tensor] = {}
    for batch in train_data:
        patch_outs = get_sorted_src_outs(model, batch.corrupt)
        src_outs_dict[batch.key] = t.stack(list(patch_outs.values()))

    losses, kl_divs, regularizes = [], [], []
    with train_mask_mode(model, init_val) as patch_masks, mask_fn_mode(model, mask_fn):
        optim = t.optim.Adam(patch_masks, lr=learning_rate)
        for epoch in (epoch_pbar := tqdm(range(epochs))):
            desc = f"Loss: {losses[-1]:.3f}, KL: {kl_divs[-1]:.3f}" if epoch > 0 else ""
            epoch_pbar.set_description_str(f"{SP} Epoch {epoch} " + desc, refresh=False)
            for batch_idx, batch in (batch_pbar := tqdm(enumerate(train_data))):
                batch_pbar.set_description_str(f"{SP} Batch {batch_idx}", refresh=False)

                patch_src_outs = src_outs_dict[batch.key].clone().detach()
                with patch_mode(model, t.zeros_like(patch_src_outs), patch_src_outs):
                    masked_logprobs = log_softmax(
                        model(batch.clean)[output_idx], dim=-1
                    )
                    kl_div_term = kl_div(
                        masked_logprobs,
                        corrupt_logprobs[batch.key],
                        reduction="batchmean",
                        log_target=True,
                    )
                    masks = t.cat([patch_mask.flatten() for patch_mask in patch_masks])
                    regularize_term = t.sigmoid(masks - regularize_const).mean()
                    loss = kl_div_term + regularize_term * regularize_lambda
                    losses.append(loss.item())
                    kl_divs.append(kl_div_term.item())
                    regularizes.append(regularize_term.item() * regularize_lambda)
                    model.zero_grad()
                    loss.backward()
                    optim.step()

    if show_train_graph:
        title = "Subnetwork Probing Loss History"
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=losses, name="Loss"))
        fig.add_trace(go.Scatter(y=kl_divs, name="KL Divergence"))
        fig.add_trace(go.Scatter(y=regularizes, name="Regularization"))
        fig.update_layout(title=title, xaxis_title="Iteration", yaxis_title="Loss")
        fig.show()

    edges: Set[Edge] = model.edges  # type: ignore
    return dict([(e, e.patch_mask(model)[e.patch_idx].item()) for e in edges])
