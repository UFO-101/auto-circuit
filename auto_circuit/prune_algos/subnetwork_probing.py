# Some of this code is copied from:
# https://github.com/stevenxcao/subnetwork-probing
# From the paper Low-Complexity Probing via Finding Subnetworks:
# https://arxiv.org/abs/2104.03514

import math
from typing import Dict

import plotly.graph_objects as go
import torch as t
from torch.nn.functional import kl_div, log_softmax

from auto_circuit.tasks import Task
from auto_circuit.types import PruneScores
from auto_circuit.utils.custom_tqdm import tqdm
from auto_circuit.utils.graph_utils import (
    get_sorted_src_outs,
    mask_fn_mode,
    patch_mode,
    set_all_masks,
    train_mask_mode,
)
from auto_circuit.utils.patch_wrapper import MaskFn, sample_hard_concrete

# Constants are copied from the paper's code
mask_p, left, right, temp = 0.9, -0.1, 1.1, 2 / 3
p = (mask_p - left) / (right - left)
init_mask_val = math.log(p / (1 - p))
regularize_const = temp * math.log(-left / right)

SP = "Subnetwork Probing"


def subnetwork_probing_prune_scores(
    task: Task,
    learning_rate: float = 0.1,
    epochs: int = 20,
    regularize_lambda: float = 10,
    mask_fn: MaskFn = "hard_concrete",
    dropout_p: float = 0.0,
    init_val: float = -init_mask_val,
    show_train_graph: bool = False,
    regularize_to_true_circuit_size: bool = False,
    tree_optimisation: bool = False,
) -> PruneScores:
    """Prune scores are the mean activation magnitude of each edge."""
    model = task.model
    out_slice = model.out_slice
    true_size = 100 if task.true_edges is None else len(task.true_edges)
    total_edges = len(model.edges)
    inv_size = total_edges - true_size if tree_optimisation else true_size

    clean_logprobs: Dict[str, t.Tensor] = {}
    with t.inference_mode():
        for batch in task.train_loader:
            clean_out = model(batch.clean)[out_slice]
            clean_logprobs[batch.key] = log_softmax(clean_out, dim=-1)

    src_outs_dict: Dict[int, t.Tensor] = {}
    for batch in task.train_loader:
        patch_batch = batch.corrupt if tree_optimisation else batch.clean
        patch_outs = get_sorted_src_outs(model, patch_batch)
        src_outs_dict[batch.key] = t.stack(list(patch_outs.values()))

    losses, kl_divs, regularizes = [], [], []
    set_all_masks(model, val=init_val)
    with train_mask_mode(model) as patch_masks, mask_fn_mode(model, mask_fn, dropout_p):
        optim = t.optim.Adam(patch_masks, lr=learning_rate)
        for epoch in (epoch_pbar := tqdm(range(epochs))):
            desc = f"Loss: {losses[-1]:.3f}, KL: {kl_divs[-1]:.3f}" if epoch > 0 else ""
            epoch_pbar.set_description_str(f"{SP} Epoch {epoch} " + desc, refresh=False)
            for batch in task.train_loader:
                input_batch = batch.clean if tree_optimisation else batch.corrupt
                patch_outs = src_outs_dict[batch.key].clone().detach()
                with patch_mode(model, t.zeros_like(patch_outs), patch_outs):
                    train_logprob = log_softmax(model(input_batch)[out_slice], dim=-1)
                    kl_div_term = kl_div(
                        train_logprob,
                        clean_logprobs[batch.key],
                        reduction="batchmean",
                        log_target=True,
                    )
                    masks = t.cat([patch_mask.flatten() for patch_mask in patch_masks])
                    if regularize_to_true_circuit_size:
                        mask_sample = sample_hard_concrete(masks, batch_size=1).sum()
                        if tree_optimisation:
                            regularize_term = t.relu(inv_size - mask_sample) / inv_size
                            regularize_direction = 1
                        else:
                            regularize_term = (
                                t.relu(mask_sample - true_size) / true_size
                            )
                            regularize_direction = 1
                    else:
                        regularize_term = t.sigmoid(masks - regularize_const).mean()
                        regularize_direction = -1 if tree_optimisation else 1
                    loss = (
                        kl_div_term
                        + regularize_direction * regularize_term * regularize_lambda
                    )
                    losses.append(loss.item())
                    kl_divs.append(kl_div_term.item())
                    regularizes.append(
                        regularize_direction
                        * regularize_term.item()
                        * regularize_lambda
                    )
                    model.zero_grad()
                    loss.backward()
                    optim.step()
        min_val = abs(min([t.min(mask).item() for mask in patch_masks]))
        max_val = abs(max([t.min(mask).item() for mask in patch_masks]))

    if show_train_graph:
        title = "Subnetwork Probing Loss History"
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=losses, name="Loss"))
        fig.add_trace(go.Scatter(y=kl_divs, name="KL Divergence"))
        fig.add_trace(go.Scatter(y=regularizes, name="Regularization"))
        fig.update_layout(title=title, xaxis_title="Iteration", yaxis_title="Loss")
        fig.show()

    if tree_optimisation:
        ps = dict(
            [
                (e, max_val - e.patch_mask(model)[e.patch_idx].item())
                for e in model.edges
            ]
        )
    else:
        ps = dict(
            [
                (e, min_val + e.patch_mask(model)[e.patch_idx].item())
                for e in model.edges
            ]
        )
    if regularize_to_true_circuit_size:
        sorted_ps = sorted(ps.items(), key=lambda x: x[1], reverse=True)
        return dict([(e, 1.0) for e, _ in sorted_ps[:true_size]])
    else:
        return ps
