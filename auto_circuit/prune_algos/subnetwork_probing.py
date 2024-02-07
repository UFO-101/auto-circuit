# Some of this code is copied from:
# https://github.com/stevenxcao/subnetwork-probing
# From the paper Low-Complexity Probing via Finding Subnetworks:
# https://arxiv.org/abs/2104.03514

import math
from typing import Dict, Literal, Optional, Set

import plotly.graph_objects as go
import torch as t
from torch.nn.functional import kl_div, log_softmax

from auto_circuit.tasks import Task
from auto_circuit.types import Edge, PruneScores
from auto_circuit.utils.custom_tqdm import tqdm
from auto_circuit.utils.graph_utils import (
    get_sorted_src_outs,
    mask_fn_mode,
    patch_mode,
    set_all_masks,
    train_mask_mode,
)
from auto_circuit.utils.patch_wrapper import sample_hard_concrete
from auto_circuit.utils.tensor_ops import MaskFn, batch_avg_answer_val

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
    circuit_size: Optional[int] = None,
    tree_optimisation: bool = False,
    avoid_edges: Optional[Set[Edge]] = None,
    avoid_lambda: float = 1.0,
    faithfulness_target: Literal["kl_div", "answer", "wrong_answer"] = "kl_div",
) -> PruneScores:
    """Optimize the patch masks using gradient descent."""
    model = task.model
    out_slice = model.out_slice
    n_edges = len(model.edges)
    n_avoid = len(avoid_edges or [])

    clean_logprobs: Dict[int, t.Tensor] = {}
    with t.inference_mode():
        for batch in task.train_loader:
            clean_out = model(batch.clean)[out_slice]
            clean_logprobs[batch.key] = log_softmax(clean_out, dim=-1)

    src_outs_dict: Dict[int, t.Tensor] = {}
    for batch in task.train_loader:
        patch_batch = batch.corrupt if tree_optimisation else batch.clean
        patch_outs = get_sorted_src_outs(model, patch_batch)
        src_outs_dict[batch.key] = t.stack(list(patch_outs.values()))

    losses, faithfulnesses, regularizes = [], [], []
    set_all_masks(model, val=-init_val if tree_optimisation else init_val)
    with train_mask_mode(model) as patch_masks, mask_fn_mode(model, mask_fn, dropout_p):
        optim = t.optim.Adam(patch_masks, lr=learning_rate)
        for epoch in (epoch_pbar := tqdm(range(epochs))):
            faithfulness_str = f"{faithfulness_target} loss: {faithfulnesses[-1]:.3f}"
            desc = f"Loss: {losses[-1]:.3f}, {faithfulness_str}" if epoch > 0 else ""
            epoch_pbar.set_description_str(f"{SP} Epoch {epoch} " + desc, refresh=False)
            for batch in task.train_loader:
                input_batch = batch.clean if tree_optimisation else batch.corrupt
                patch_outs = src_outs_dict[batch.key].clone().detach()
                with patch_mode(model, t.zeros_like(patch_outs), patch_outs):
                    train_logits = model(input_batch)[out_slice]
                    if faithfulness_target == "kl_div":
                        faithful_term = kl_div(
                            log_softmax(train_logits, dim=-1),
                            clean_logprobs[batch.key],
                            reduction="batchmean",
                            log_target=True,
                        )
                    else:
                        assert faithfulness_target in ["answer", "wrong_answer"]
                        wrong = faithfulness_target == "wrong_answer"
                        faithful_term = batch_avg_answer_val(train_logits, batch, wrong)
                    masks = t.cat([patch_mask.flatten() for patch_mask in patch_masks])
                    if mask_fn == "hard_concrete":
                        masks = sample_hard_concrete(masks, batch_size=1)
                    elif mask_fn == "sigmoid":
                        masks = t.sigmoid(masks)
                    n_mask = n_edges - masks.sum() if tree_optimisation else masks.sum()
                    if circuit_size:
                        n_mask = t.relu(n_mask - circuit_size)
                    regularize = n_mask / (circuit_size if circuit_size else n_edges)
                    for edge in avoid_edges or []:  # Penalize banned edges
                        wgt = (-1 if tree_optimisation else 1) * avoid_lambda / n_avoid
                        penalty = edge.patch_mask(model)[edge.patch_idx]
                        const = regularize_const if mask_fn == "hard_concrete" else 0.0
                        if mask_fn is not None:
                            penalty = t.sigmoid(penalty - const)
                        regularize += wgt * penalty
                    loss = faithful_term + regularize * regularize_lambda
                    losses.append(loss.item())
                    faithfulnesses.append(faithful_term.item())
                    regularizes.append(regularize.item() * regularize_lambda)
                    model.zero_grad()
                    loss.backward()
                    optim.step()
        xtreme_val = max if tree_optimisation else min
        xtreme_torch_val = t.max if tree_optimisation else t.min
        m = abs(xtreme_val([xtreme_torch_val(mask).item() for mask in patch_masks]))

    if show_train_graph:
        title = "Subnetwork Probing Loss History"
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=losses, name="Loss"))
        fig.add_trace(go.Scatter(y=faithfulnesses, name=faithfulness_target.title()))
        fig.add_trace(go.Scatter(y=regularizes, name="Regularization"))
        fig.update_layout(title=title, xaxis_title="Iteration", yaxis_title="Loss")
        fig.show()

    sign = -1 if tree_optimisation else 1
    ps = [(e, m + sign * e.patch_mask(model)[e.patch_idx].item()) for e in model.edges]
    if circuit_size:
        sorted_ps = sorted(ps, key=lambda x: x[1], reverse=True)
        return dict([(e, 1.0) for e, _ in sorted_ps[:circuit_size]])
    else:
        return dict(ps)
